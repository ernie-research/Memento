"""
Accelerated WanM2V inference module.

Optimizations over the original memory2video_learnable_rope_fine.py:
  1. T5 text encoding cache: avoid redundant T5 forwards within a shot
  2. VAE encode dedup in dual-query mode: encode prev_shot video once, reuse for both pools
"""

import os
import logging
import torch

from wan.memory2video_learnable_rope_fine import WanM2V


logger = logging.getLogger(__name__)


class WanM2V_Acce(WanM2V):
    """Drop-in replacement for WanM2V with T5 cache, VAE encode dedup, and torch.compile."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._text_cache = {}
        self._vae_encode_cache = {}

    # ------------------------------------------------------------------
    # Optimization 3: torch.compile for DiT models
    # ------------------------------------------------------------------

    def compile_dit(self, mode="max-autotune-no-cudagraphs", dynamic=None):
        """Compile low_noise_model and high_noise_model with torch.compile.

        Call after model init and FSDP wrapping. First denoising step per
        unique input shape triggers compilation; subsequent steps reuse cache.
        """
        if hasattr(self, 'low_noise_model') and self.low_noise_model is not None:
            logger.info(f"[Acce] Compiling low_noise_model (mode={mode})")
            self.low_noise_model = torch.compile(
                self.low_noise_model, mode=mode, dynamic=dynamic)

        if hasattr(self, 'high_noise_model') and self.high_noise_model is not None:
            logger.info(f"[Acce] Compiling high_noise_model (mode={mode})")
            self.high_noise_model = torch.compile(
                self.high_noise_model, mode=mode, dynamic=dynamic)

    # ------------------------------------------------------------------
    # Optimization 1: T5 text encoding cache
    # ------------------------------------------------------------------

    def _encode_text(self, prompts, device, offload_model=False):
        key = tuple(prompts)
        if key in self._text_cache:
            cached = self._text_cache[key]
            return [t.clone() for t in cached]
        result = super()._encode_text(prompts, device, offload_model)
        self._text_cache[key] = [t.clone() for t in result]
        return result

    def clear_text_cache(self):
        self._text_cache.clear()

    # ------------------------------------------------------------------
    # Optimization 2: VAE encode dedup for dual-query mode
    # ------------------------------------------------------------------

    def update_memory_pool_from_file(self, memory_pool, video_path,
                                     target_h=None, target_w=None,
                                     num_sample_frames=-1):
        """Override with VAE encode caching.

        When the same video_path + target_h + target_w + num_sample_frames
        is called multiple times (dual-query mode encodes same video for
        low and high pools), the expensive VAE encode is done only once.
        """
        if not os.path.exists(video_path):
            logging.warning(f"update_memory_pool_from_file: file not found: {video_path}")
            return memory_pool, 0, []

        cache_key = (video_path, target_h, target_w, num_sample_frames)

        if cache_key in self._vae_encode_cache:
            new_tokens, total, idxs = self._vae_encode_cache[cache_key]
            logging.info(f"[Acce] VAE encode cache HIT for {os.path.basename(video_path)}, "
                         f"skipping redundant encode ({new_tokens.shape[1]} frames)")
        else:
            new_tokens, total, idxs = self._vae_encode_new_tokens(
                video_path, target_h, target_w, num_sample_frames)
            self._vae_encode_cache[cache_key] = (new_tokens, total, idxs)

        if memory_pool is None:
            return new_tokens.clone(), total, idxs

        old_pool = memory_pool.cpu()
        lat_H, lat_W = new_tokens.shape[2], new_tokens.shape[3]
        if old_pool.shape[2] != lat_H or old_pool.shape[3] != lat_W:
            raise ValueError(
                f"update_memory_pool_from_file: old pool spatial size "
                f"({old_pool.shape[2]}x{old_pool.shape[3]}) != "
                f"new tokens ({lat_H}x{lat_W}). "
            )
        return torch.cat([old_pool, new_tokens.clone()], dim=1), total, idxs

    def _vae_encode_new_tokens(self, video_path, target_h, target_w, num_sample_frames):
        """Extract frames from video and VAE-encode them. Returns (new_tokens, total, idxs)."""
        if video_path.endswith('.pt'):
            video_tensor = torch.load(video_path, map_location='cpu')
            total = video_tensor.shape[1]

            if num_sample_frames <= 0 or total <= num_sample_frames:
                idxs = list(range(total))
            else:
                idxs = [round(i * (total - 1) / (num_sample_frames - 1))
                        for i in range(num_sample_frames)]

            imgs = []
            for idx in idxs:
                img = video_tensor[:, idx, :, :]
                if target_h is not None and target_w is not None:
                    img = torch.nn.functional.interpolate(
                        img[None], size=(target_h, target_w), mode='bicubic'
                    ).squeeze(0)
                imgs.append(img)
            img_tensor = torch.stack(imgs, dim=0)
        else:
            import av
            import torchvision.transforms.functional as TF

            container = av.open(video_path)
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            all_frames = []
            for frame in container.decode(stream):
                all_frames.append(frame.to_image().convert("RGB"))
            container.close()

            total = len(all_frames)
            if total == 0:
                logging.warning(f"_vae_encode_new_tokens: no frames decoded from {video_path}")
                return torch.zeros(16, 0, 1, 1), 0, []

            if num_sample_frames <= 0 or total <= num_sample_frames:
                idxs = list(range(total))
            else:
                idxs = [round(i * (total - 1) / (num_sample_frames - 1))
                        for i in range(num_sample_frames)]

            imgs = []
            for idx in idxs:
                img = TF.to_tensor(all_frames[idx]).sub_(0.5).div_(0.5)
                if target_h is not None and target_w is not None:
                    img = torch.nn.functional.interpolate(
                        img[None].cpu(), size=(target_h, target_w), mode='bicubic'
                    ).squeeze(0)
                imgs.append(img)
            img_tensor = torch.stack(imgs, dim=0)

        chunk_size = 8
        new_tokens_list = []
        for i in range(0, len(img_tensor), chunk_size):
            chunk = img_tensor[i:i + chunk_size].to(self.device)
            tokens = self.vae.encode(chunk.unsqueeze(2)).float().squeeze(2).cpu()
            new_tokens_list.append(tokens)
            del chunk
        torch.cuda.empty_cache()

        new_tokens = torch.cat(new_tokens_list, dim=0)
        new_tokens = new_tokens.permute(1, 0, 2, 3)
        return new_tokens, total, idxs

    # ------------------------------------------------------------------
    # Override generate to clear caches at boundaries
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        self._text_cache.clear()
        self._vae_encode_cache.clear()
        try:
            return super().generate(*args, **kwargs)
        finally:
            self._text_cache.clear()
            self._vae_encode_cache.clear()
