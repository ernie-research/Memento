import os
import sys
import argparse
import random
import logging
import json
import gc
import math
import shutil
import tempfile
import subprocess
import time
import torch.nn as nn
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torchvision.utils as vutils
import torch
import torchvision
from torchvision.transforms import functional as TF
import torch.nn.functional as F_torch
import numpy as np
import traceback as tb
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.utils.checkpoint as checkpoint

import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from diffsynth.core.data.unified_dataset_bos import LoadVideoHttp, ImageCropAndResize
from wan.modules.model_memory import WanModel_Memory
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae2_1 import Wan2_1_VAE
from wan.configs import WAN_CONFIGS
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

try:
    from peft import LoraConfig, get_peft_model
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

import bitsandbytes as bnb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MAX_MEMORY_FRAMES = 10


# ---------------------------------------------------------------------------
# StepTimer
# ---------------------------------------------------------------------------

class StepTimer:
    def __init__(self, use_cuda=True, name="timer"):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.name = name
        self.reset()

    def reset(self):
        self.start_times = {}
        self.durations = {}
        self.current_depth = 0
        self.cuda_events = {} if self.use_cuda else None
        self._closed = False

    def start(self, name):
        key = f"{'  ' * self.current_depth}{name}"
        if self._closed:
            return
        if self.use_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.cuda_events[key] = event
        else:
            self.start_times[key] = time.perf_counter()
        self.current_depth += 1

    def end(self, name):
        self.current_depth = max(0, self.current_depth - 1)
        key = f"{'  ' * self.current_depth}{name}"
        if self._closed:
            return None
        duration = 0.0
        if self.use_cuda and key in self.cuda_events:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            torch.cuda.synchronize()
            duration = self.cuda_events[key].elapsed_time(end_event) / 1000.0
            del self.cuda_events[key]
        elif key in self.start_times:
            duration = time.perf_counter() - self.start_times[key]
            del self.start_times[key]
        if duration > 0:
            self.durations.setdefault(key, []).append(duration)
        return duration

    def print_summary(self, window=10, top_n=20):
        stats = {}
        for key, times in self.durations.items():
            t = times[-window:]
            if not t:
                continue
            stats[key.strip()] = {'mean': sum(t)/len(t), 'last': t[-1], 'max': max(t), 'count': len(t)}
        if not stats:
            return
        sorted_stats = sorted(stats.items(), key=lambda x: -x[1]['mean'])[:top_n]
        logger.info(f"\n{'='*80}\nStep Timing ({self.name}, last {window}):")
        for name, s in sorted_stats:
            logger.info(f"  {name:<45} last={s['last']:.3f}s avg={s['mean']:.3f}s max={s['max']:.3f}s")
        logger.info('='*80)

    def close(self):
        self._closed = True
        self.start_times.clear()
        if self.cuda_events:
            self.cuda_events.clear()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StorySequenceDataset(Dataset):
    def __init__(self, data_path, resolution="832*480", frame_num=31, high_fps=16, ffmpeg_bin=None):
        self.height, self.width = map(int, resolution.split('*'))
        self.frame_num = frame_num
        self.high_fps = high_fps
        self.frame_processor = ImageCropAndResize(
            self.height, self.width, self.height * self.width, 16, 16)
        ffmpeg_bin = ffmpeg_bin or shutil.which('ffmpeg') or 'ffmpeg'
        ffprobe_bin = (
            os.path.join(os.path.dirname(ffmpeg_bin), 'ffprobe')
            if ffmpeg_bin != 'ffmpeg' else shutil.which('ffprobe') or 'ffprobe')
        self.video_loader = LoadVideoHttp(
            num_frames=self.frame_num, frame_processor=self.frame_processor,
            ffmpeg_bin=ffmpeg_bin, ffprobe_bin=ffprobe_bin)
        self.video_loader.high_fps = self.high_fps
        self.ffmpeg_bin = ffmpeg_bin
        self.ffprobe_bin = ffprobe_bin
        self.sequences = self._load_sequences(data_path)

    def _get_bos_client(self):
        if not hasattr(self, '_bos_client'):
            try:
                import yaml
                from baidubce.bce_client_configuration import BceClientConfiguration
                from baidubce.auth.bce_credentials import BceCredentials
                from baidubce.services.bos.bos_client import BosClient
            except ImportError:
                raise ImportError("baidubce not installed. Use local video_path or install: pip install bce-python-sdk pyyaml")
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bos_config.yaml")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"BOS config not found: {config_path}")
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            g = cfg['global_config']
            bce_config = BceClientConfiguration(
                credentials=BceCredentials(g['ak'], g['sk']),
                endpoint=g['endpoint'])
            self._bos_client = BosClient(bce_config)
        return self._bos_client

    def convert_bos_to_http(self, bos_url):
        if not bos_url or not isinstance(bos_url, str):
            return None
        if bos_url.startswith("http://") or bos_url.startswith("https://"):
            return bos_url
        bos_client = self._get_bos_client()
        try:
            path = bos_url.replace("bos://", "")
            parts = path.split("/")
            bucket, object_key = parts[0], "/".join(parts[1:])
            return bos_client.generate_pre_signed_url(bucket, object_key, expiration_in_seconds=-1).decode("utf-8")
        except Exception as e:
            logger.warning(f"BOS URL conversion failed: {bos_url}: {e}")
            return bos_url

    def _get_video_url(self, clip):
        """Get loadable video path/URL from clip dict. Supports local path or BOS."""
        if 'video_path' in clip and clip['video_path']:
            return clip['video_path']
        if 'bos_url' in clip and clip['bos_url']:
            return self.convert_bos_to_http(clip['bos_url'])
        return None

    def _load_sequences(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_sequences = json.load(f)
        sequences = []
        for seq in raw_sequences:
            clips = sorted(seq.get('sequence_clips', []), key=lambda x: x.get('sequence_order', 0))
            valid_clips = [c for c in clips if c.get('bos_url') or c.get('video_path')]
            if len(valid_clips) >= 2:
                sequences.append({
                    'sequence_id': seq.get('sequence_id', 'unknown'),
                    'clips': valid_clips,
                    'reconstruct_targets': seq.get('reconstruct_targets', []),
                })
        logger.info(f"Loaded {len(sequences)} valid sequences from {data_path}")
        return sequences

    def _get_video_info(self, url):
        try:
            cmd = [self.ffprobe_bin, '-v', 'quiet', '-select_streams', 'v:0',
                   '-show_entries', 'stream=nb_frames,r_frame_rate', '-of', 'json', url]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            data = json.loads(res.stdout)
            stream = data.get('streams', [{}])[0]
            nb_frames = int(stream.get('nb_frames', 250))
            r = stream.get('r_frame_rate', '24/1')
            num, den = r.split('/')
            fps = float(num) / float(den)
            return nb_frames, fps
        except Exception:
            return 250, 24.0

    def _sample_frames_from_video(self, url, num_frames, fps_sample=4):
        total, video_fps = self._get_video_info(url)
        step = max(1, round(video_fps / fps_sample))
        candidate_indices = list(range(0, total, step))
        if len(candidate_indices) > num_frames:
            candidate_indices = sorted(random.sample(candidate_indices, num_frames))
        elif len(candidate_indices) == 0:
            candidate_indices = [0]

        frames = []
        with tempfile.TemporaryDirectory() as tmp:
            expr = '+'.join([f'eq(n\\,{i})' for i in candidate_indices])
            cmd = [self.ffmpeg_bin, '-loglevel', 'error', '-i', url,
                   '-vf', f"select='{expr}'", '-vsync', 'vfr', f'{tmp}/%d.png', '-y']
            try:
                subprocess.run(cmd, timeout=60, capture_output=True)
            except Exception as e:
                logger.warning(f"ffmpeg frame sampling failed: {e}")
                return [], [], total
            for i in range(1, len(candidate_indices) + 1):
                p = f"{tmp}/{i}.png"
                if os.path.exists(p):
                    img = self.frame_processor(Image.open(p).convert('RGB'))
                    frames.append(TF.to_tensor(img) * 2.0 - 1.0)

        indices_16fps = [round(idx * 16 / video_fps) for idx in candidate_indices]
        total_16fps = round(total * 16 / video_fps)
        return frames, indices_16fps, total_16fps

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        max_retry = 5
        for attempt in range(max_retry):
            try:
                return self._getitem_impl(idx)
            except Exception as e:
                logger.warning(f"[Dataset] idx={idx} attempt {attempt+1}: {type(e).__name__}: {e}")
                idx = (idx + 1) % len(self.sequences)
        raise RuntimeError(f"DataLoader: failed {max_retry} consecutive samples")

    def _getitem_impl(self, idx):
        seq = self.sequences[idx]
        clips_data = []
        for clip in seq['clips']:
            url = self._get_video_url(clip)
            if url is None:
                continue
            video_frames = self.video_loader(url)
            if not video_frames:
                video_frames = [Image.new('RGB', (self.width, self.height))] * self.frame_num
            video_tensor = torch.stack(
                [TF.to_tensor(img) * 2.0 - 1.0 for img in video_frames[:self.frame_num]], dim=1)

            memory_frames, memory_indices, total_frames = self._sample_frames_from_video(
                url, num_frames=MAX_MEMORY_FRAMES, fps_sample=4)
            if len(memory_frames) == 0:
                memory_frames = [torch.zeros(3, self.height, self.width)]
                memory_indices = [0]
                total_frames = self.frame_num

            clips_data.append({
                'video': video_tensor,
                'memory_frames': memory_frames,
                'memory_indices': memory_indices,
                'total_frames': total_frames,
                'caption': clip.get('caption', ""),
            })

        identity_frames_data = []
        for target in seq.get('reconstruct_targets', []):
            ci = target['clip_index']
            fi = target['frame_in_clip_16fps']
            if ci < len(clips_data) and fi < clips_data[ci]['video'].shape[1]:
                identity_frames_data.append({
                    'frame': clips_data[ci]['video'][:, fi],
                    'clip_idx': ci,
                    'frame_in_clip_16fps': fi,
                    'caption': target.get('caption', clips_data[ci].get('caption', '')),
                })

        return {
            'sequence_id': seq['sequence_id'],
            'clips': clips_data,
            'identity_frames': identity_frames_data,
            'sequence_caption': seq.get("subjects", ""),
        }


def sequence_collate_fn(batch):
    return batch[0]



class M2VQueryTrainingModule(torch.nn.Module):
 
    def __init__(
        self,
        checkpoint_dir,
        target_dtype=torch.bfloat16,
        lora_rank=128,
        device_id=0,
        boundary=0.9,
        grad_ckpt=False,
        use_gradient_checkpointing_offload=False,
        train_low_noise_only=True,
        train_high_noise_only=False,
        use_sp=False,
        shift=5,
        uncond_p=0.1,
        num_keyframes=10,
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        text_dim=4096,
        identity_loss_weight=1.0,  # 主体帧重建 loss 系数
        enable_multi_identity_train=False,  # 多主体帧独立 cross-attention
        identity_frame_prob=0.5,  # 开启 identity frame 机制的概率
        skip_prior_shots_prob=0.1,  # 跳过前序 shot 的概率
        compile_t5=False,  # torch.compile 加速 T5 编码
        t5_offload=True,   # 每次 T5 编码后搬回 CPU 省显存；False 则常驻 GPU
        enable_mem_recon_loss=False,  # 是否对 memory 部分加重建 loss
        mem_recon_loss_weight=1.0,    # memory 重建 loss 系数
        split_identity_attn=False,    # 是否启用 identity+memory / memory+generation 分离 attention
        split_learnable_query=False,
        global_query_num=0,
        selected_local_num = 6
    ):
        super().__init__()
        self.config = WAN_CONFIGS['m2v-A14B']
        self.target_dtype = target_dtype
        self.boundary_timestep = int(boundary * 1000)
        self.train_low_noise_only = train_low_noise_only
        self.train_high_noise_only = train_high_noise_only
        self.device_id = device_id
        self.grad_ckpt = grad_ckpt
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.use_sp = use_sp
        self.uncond_p = uncond_p
        self.num_keyframes = num_keyframes
        self.identity_loss_weight = identity_loss_weight
        self.enable_multi_identity_train = enable_multi_identity_train
        self.identity_frame_prob = identity_frame_prob
        self.skip_prior_shots_prob = skip_prior_shots_prob
        self.compile_t5 = compile_t5
        self.t5_offload = t5_offload
        self.enable_mem_recon_loss = enable_mem_recon_loss
        self.mem_recon_loss_weight = mem_recon_loss_weight
        self.split_identity_attn = split_identity_attn
        self.split_learnable_query = split_learnable_query
        self.global_query_num = global_query_num
        self.selected_local_num = selected_local_num
        
        # timestep scheduler
        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        sample_scheduler.set_timesteps(1000, device=device_id, shift=shift)
        self.sample_scheduler = sample_scheduler
        all_timesteps = self.sample_scheduler.timesteps.clone()
        self.register_buffer("low_noise_ts", all_timesteps[all_timesteps < self.boundary_timestep])
        self.register_buffer("high_noise_ts", all_timesteps[all_timesteps >= self.boundary_timestep])
 
        device = torch.device(f"cuda:{device_id}")
 
        # ---- VAE (frozen) ----
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, self.config.vae_checkpoint),
            device=device,
            dtype=torch.bfloat16)
 
        # ---- T5 (frozen, cpu) ----
        self.text_encoder = T5EncoderModel(
            text_len=self.config.text_len,
            dtype=self.config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, self.config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, self.config.t5_tokenizer),
            shard_fn=None,
        )
        if self.compile_t5:
            logger.info("Compiling T5 encoder with torch.compile (mode=reduce-overhead)...")
            self.text_encoder.model = torch.compile(
                self.text_encoder.model, mode="reduce-overhead")
 
        logger.info(f"Creating WanModel_Memory (num_keyframes={num_keyframes})...")
 
        self.low_noise_model = None
        self.high_noise_model = None
 
        if not train_high_noise_only:
            self.low_noise_model = self._create_model(
                checkpoint_dir, self.config.low_noise_checkpoint,
                lora_rank, target_dtype, device)
            logger.info("low_noise_model ready")
 
        if not train_low_noise_only:
            self.high_noise_model = self._create_model(
                checkpoint_dir, self.config.high_noise_checkpoint,
                lora_rank, target_dtype, device)
            logger.info("high_noise_model ready")
 
        if grad_ckpt or use_gradient_checkpointing_offload:
            self._enable_gradient_checkpointing(self.low_noise_model)
            self._enable_gradient_checkpointing(self.high_noise_model)
 
        if use_sp and dist.is_initialized():
            self.sp_size = dist.get_world_size()
        else:
            self.sp_size = 1
 
        self.vae_stride = self.config.vae_stride
        self.patch_size = self.config.patch_size
        self.keyframe_query = self.low_noise_model.keyframe_query if self.low_noise_model is not None else self.high_noise_model.keyframe_query
 
 
    def _create_model(self, checkpoint_dir, subfolder, lora_rank, target_dtype, device):
        """
        优化后的快速加载逻辑：直接在 GPU 上以 target_dtype 加载
        """
        from wan.modules.model import WanModel as WanModel_base
        import gc
        
        logger.info(f"Loading base model to {device} in {target_dtype}...")
        
        base_model = WanModel_base.from_pretrained(
            checkpoint_dir, 
            subfolder=subfolder,
            torch_dtype=target_dtype 
        )
        base_model.to(device) # 确保权重已经在 GPU 上
        base_state_dict = base_model.state_dict()
 
        # 自动推断 in_dim
        pe_key = 'patch_embedding.weight'
        pretrained_in_dim = base_state_dict[pe_key].shape[1] 
        logger.info(f"Detected pretrained in_dim={pretrained_in_dim}")
 
        logger.info("Initializing WanModel_Memory directly on device...")
        
        # 记录原来的默认 dtype
        old_dtype = torch.get_default_dtype()
        # 全局设置为 target_dtype (如 bfloat16)
        torch.set_default_dtype(target_dtype)
        
        try:
            # 仅使用 torch.device 作为上下文管理器
            with torch.device(device):
                model = WanModel_Memory(
                    model_type=self.config.model_type if hasattr(self.config, 'model_type') else 'i2v',
                    patch_size=tuple(self.config.patch_size),
                    text_len=self.config.text_len,
                    in_dim=pretrained_in_dim,
                    dim=self.config.dim,
                    ffn_dim=self.config.ffn_dim,
                    freq_dim=self.config.freq_dim,
                    text_dim=self.config.text_dim if hasattr(self.config, 'text_dim') else 4096,
                    out_dim=self.config.out_dim if hasattr(self.config, 'out_dim') else 16,
                    num_heads=self.config.num_heads,
                    num_layers=self.config.num_layers,
                    window_size=tuple(self.config.window_size) if hasattr(self.config, 'window_size') else (-1, -1),
                    qk_norm=self.config.qk_norm if hasattr(self.config, 'qk_norm') else True,
                    cross_attn_norm=self.config.cross_attn_norm if hasattr(self.config, 'cross_attn_norm') else True,
                    num_keyframes=self.num_keyframes,
                    keyframe_temperature=0.1,
                    split_identity_attn=self.split_identity_attn,
                    split_learnable_query=self.split_learnable_query,
                    global_query_num=self.global_query_num,
                    selected_local_num=self.selected_local_num
                )
        finally:
            # 无论初始化是否成功，都恢复默认 dtype
            torch.set_default_dtype(old_dtype)
        logger.info("Loading state dict with zero-copy assignment...")
        missing, unexpected = model.load_state_dict(base_state_dict, strict=False, assign=True)
        logger.info(f"Loaded base weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        
        del base_state_dict
        del base_model
        gc.collect()
        torch.cuda.empty_cache()
 
        if lora_rank > 0 and HAS_PEFT:
            logger.info("Injecting PEFT LoRA adapters...")
            lora_cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank,
                lora_dropout=0.0,
                bias='none',
                #################
                # target_modules=r"^(?!.*keyframe_query).*blocks\.\d+\.(self_attn|cross_attn)\.(q|k|v|o)|^(?!.*keyframe_query).*blocks\.\d+\.ffn\.[02]$",
                #################
                # target_modules=r"^(?!.*keyframe_query).*blocks\.\d+\.(self_attn|cross_attn)\.(q|k|v|o)|^(?!.*keyframe_query).*blocks\.\d$",
                target_modules=r"^(?!.*keyframe_query).*blocks\.\d+\.(self_attn|cross_attn)\.(q|k|v|o)|^(?!.*keyframe_query).*blocks\.\d+\.ffn\.[02]$",
                use_rslora=True,
            )
            model = get_peft_model(model, lora_cfg)
            model.to(target_dtype)  # LoRA weights 默认 fp32 初始化，强制转回 bf16
 
            for name, param in model.named_parameters():
                if 'keyframe_query' in name:
                    param.requires_grad = True
 
        return model
    
    def _get_underlying_model(self, model):
        if model is None:
            return None
        if hasattr(model, 'base_model'):
            base = model.base_model
            if hasattr(base, 'model'):
                return base.model
        return model
 
    def _enable_gradient_checkpointing(self, model):
        if model is None:
            return
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        underlying = self._get_underlying_model(model)
        if underlying is not None:
            if hasattr(underlying, 'gradient_checkpointing'):
                underlying.gradient_checkpointing = True
            if hasattr(underlying, 'use_gradient_checkpointing_offload'):
                underlying.use_gradient_checkpointing_offload = self.use_gradient_checkpointing_offload
            if hasattr(underlying, 'keyframe_query'):
                underlying.keyframe_query.gradient_checkpointing = True
                underlying.keyframe_query.use_gradient_checkpointing_offload = self.use_gradient_checkpointing_offload
 
    def _t5_ensure_gpu(self, device):
        """将 T5 模型搬到 GPU（如果还在 CPU 上）。"""
        if next(self.text_encoder.model.parameters()).device.type == 'cpu':
            self.text_encoder.model.to(device)
 
    def _t5_offload_cpu(self):
        """将 T5 模型搬回 CPU 释放显存（仅 t5_offload=True 且未 compile 时生效）。"""
        if self.t5_offload and not self.compile_t5:
            self.text_encoder.model.to(torch.device('cpu'))
            torch.cuda.empty_cache()
 
    @torch.no_grad()
    def _encode_text(self, prompts, device):
        # 调用前需确保 T5 已在 GPU 上（由外层 _t5_ensure_gpu 管理）
        try:
            context_list = self.text_encoder(prompts, device)
        except torch.cuda.OutOfMemoryError:
            logger.warning("T5 GPU encoding OOM, falling back to CPU")
            self.text_encoder.model.to(torch.device('cpu'))
            context_list = self.text_encoder(prompts, torch.device('cpu'))
 
        return [c.to(device, dtype=self.target_dtype).contiguous()
                for c in context_list] if isinstance(context_list, list) else context_list.to(device, dtype=self.target_dtype)
    @torch.no_grad()
    def encode_frames_to_memory(self, frames_list, device):
        """
        将像素帧列表编码为 memory_pool latent token。
 
        Args:
            frames_list: List[Tensor (C, H, W)]，像素值 [-1, 1]
            device: target device
 
        Returns:
            memory_pool: (lat_C, M, lat_H, lat_W) 或 None（空列表时）
        """
        if not frames_list:
            return None
        # (M, C, H, W) → (M, C, 1, H, W) → vae → (M, lat_C, 1, lat_H, lat_W) → squeeze → permute
        latents = []
        for frame in frames_list:
            # print(frame.shape, 1111)
            frame = frame.unsqueeze(0)
            latents.append(self.vae.encode(frame.to(device).unsqueeze(2)).float().squeeze(2))
        latents = torch.stack(latents, dim=0) 
        # for frame in frames_list:
        #     frame = frame.unsqueeze(0)
        #     latent = self.vae.encode(frame.to(device).unsqueeze(2))  # VAE 编码
        #     latent = latent.to(dtype=self.target_dtype).squeeze(2)  # 转换为目标 dtype (bfloat16)
        #     latents.append(latent)
        # latents = torch.stack(latents, dim=0)  # (M, lat_C, lat_H, lat_W)
        return latents.squeeze(1).permute(1, 0, 2, 3)  # (lat_C, M, lat_H, lat_W)
 
    @torch.no_grad()
    def update_memory_pool_from_video(self, memory_pool, video_tensor, device,
                                       fps_sample=4, video_fps=24):
        """
        用生成/真实 video 按 fps 采样更新 memory_pool (latent token)。
 
        Args:
            memory_pool: (lat_C, M, lat_H, lat_W) 或 None
            video_tensor: (C, F, H, W) 像素 [-1, 1]
        Returns:
            (lat_C, M_new, lat_H, lat_W)，最多 MAX_MEMORY_FRAMES 帧
        """
        C, F, H, W = video_tensor.shape
        step = max(1, video_fps // fps_sample)
        indices = list(range(0, F, step))
        sampled = video_tensor[:, indices].permute(1, 0, 2, 3).to(device)  # (N, C, H, W)
        new_tokens = self.vae.encode(sampled.unsqueeze(2)).to(dtype=self.target_dtype).squeeze(2)  # (N, lat_C, lat_H, lat_W)
        new_tokens = new_tokens.permute(1, 0, 2, 3)  # (lat_C, N, lat_H, lat_W)
 
        if memory_pool is None:
            pool = new_tokens
        else:
            pool = torch.cat([memory_pool.to(device), new_tokens], dim=1)
 
        M = pool.shape[1]
        if M > MAX_MEMORY_FRAMES:
            idx = torch.linspace(0, M - 1, MAX_MEMORY_FRAMES).long().to(pool.device)
            pool = pool[:, idx]
 
        return pool
 
 
    def forward_single_clip(self, model, video_tensor, memory_pool, prompt, device, first_frame=None, time_indices=None, timer=None, identity_frame_latents=None, identity_frames_caption=None):
        """
        对单个 clip 做一步 flow matching 训练。
        memory_pool 是干净的 latent token，不会被加噪。
 
        Returns:
            loss: scalar tensor
        """
        import traceback as tb
        # ---- VAE encode & text ----
        video_tensor = video_tensor.unsqueeze(0)
        B, C, T, H, W = video_tensor.shape
        M = memory_pool.shape[2]
        M_identity = 0
        if identity_frame_latents is not None:
            M_identity = identity_frame_latents.shape[2]
        C_lat, _, H_lat, W_lat = memory_pool.shape[1:]
        try:
            with torch.no_grad():
                if self.training and random.random() < self.uncond_p:
                    prompt = ""
 
                # T5 文本编码计时
                if timer: timer.start("t5_encode")
                context = self._encode_text([prompt], device)
                if timer: timer.end("t5_encode")
 
                # 所有 T5 编码已完成，offload 到 CPU 腾出显存给 VAE 和 DiT
                self._t5_offload_cpu()
 
                # VAE 视频编码计时
                if timer: timer.start("vae_encode_video")
                x_latent = self.vae.encode(video_tensor).to(dtype=self.target_dtype)
                _, _, T_lat, _, _ = x_latent.shape
                if timer: timer.end("vae_encode_video")
        except torch.cuda.OutOfMemoryError:
            logger.error(f"[OOM] forward_single_clip: VAE/text encode OOM\n{tb.format_exc()}")
            raise
 
        # ---- flow matching ----
        if timer: timer.start("flow_matching_prep")
        latents = x_latent
 
        if self.train_low_noise_only or model is self.low_noise_model:
            idx = torch.randint(0, len(self.low_noise_ts), (1,), device=device)
            timestep = self.low_noise_ts[idx]
        else:
            idx = torch.randint(0, len(self.high_noise_ts), (1,), device=device)
            timestep = self.high_noise_ts[idx]
 
        timestep = timestep.to(dtype=self.target_dtype, device=device)
 
        # ---- 组装 model 输入 ----
        msk = torch.zeros(B, 4, M + T_lat, H_lat, W_lat, device=device, dtype=self.target_dtype)
        msk[:, :, :M] = 1.0
 
        if identity_frame_latents is not None:
            msk_identity_frames = torch.ones(B, 4, M_identity, H_lat, W_lat, device=device, dtype=self.target_dtype)
            msk_identity_frames[:, :, :] = 0.0
            msk = torch.cat([msk_identity_frames, msk], dim=2)
 
        # VAE 编码条件视频
        if timer: timer.start("vae_encode_condition")
        with torch.no_grad():
            if first_frame is not None:
                msk[:, :, M_identity + M] = 1.0  # Target 第一帧有条件（identity 帧之后偏移）
                zeros_rest = torch.zeros(B, C, T - 1, H, W, device=device)
                u_pixel = torch.cat([first_frame, zeros_rest], dim=2)  # [B, C, T, H, W]
                del zeros_rest
            else:
               u_pixel = torch.zeros(B, C, T, H, W, device=device)
        u = self.vae.encode(u_pixel).to(dtype=self.target_dtype)
        del u_pixel
        if timer: timer.end("vae_encode_condition")
 
        latent_y = torch.cat([memory_pool.to(self.target_dtype), u], dim=2)
        if identity_frame_latents is not None:
            # 主体帧重建是 t2v 任务：用零像素帧过 VAE 作为 conditioning，不泄露图像信息
            # 零帧 VAE 输出恒定，只 encode 一次后 repeat
            zero_pixel = torch.zeros(1, C, 1, H, W, device=device)
            zero_lat = self.vae.encode(zero_pixel).squeeze(2)  # (1, C_lat, H_lat, W_lat)
            del zero_pixel
            identity_cond = zero_lat.unsqueeze(2).expand(-1, -1, M_identity, -1, -1).to(dtype=self.target_dtype)
            del zero_lat
            latent_y = torch.cat([identity_cond, latent_y], dim=2)
            del identity_cond
        y = torch.cat([msk, latent_y], dim=1)
 
        full_latents = torch.cat([memory_pool, latents], dim=2)  # [B, C_lat, M+T_lat, H_lat, W_lat]
        if identity_frame_latents is not None:
            full_latents = torch.cat([identity_frame_latents.to(full_latents.dtype), full_latents], dim=2)
 
        # 生成噪声
        if timer: timer.start("noise_generation")
        full_noise = torch.randn_like(full_latents)
        x = self.sample_scheduler.add_noise(full_latents, full_noise, timestep).to(dtype=self.target_dtype, device=device)
        v_target = full_noise - full_latents
        if timer: timer.end("noise_generation")
        if timer: timer.end("flow_matching_prep")
 
 
        del full_latents, full_noise, x_latent, msk, u, latent_y
        torch.cuda.empty_cache() # 保存完后清理一下显存碎片
 
        lat_t = M_identity + M + T_lat
        seq_len = lat_t * H_lat * W_lat // (self.config.patch_size[1] * self.config.patch_size[2])
 
        # ---- model forward ----
        if timer: timer.start("model_forward")
        try:
            with torch.amp.autocast('cuda', dtype=self.target_dtype):
                pred_list = model(
                    x=[x[i] for i in range(B)],
                    t=timestep.to(self.target_dtype),
                    context=context,
                    seq_len=seq_len,
                    y=[y[i] for i in range(B)],
                    memory_size=M,
                    time_indices=time_indices,
                    timer=timer,  # 传入 timer 进行详细计时
                    identity_frames_caption=identity_frames_caption,
                    identity_frame_num=[M_identity],
                    enable_multi_identity_train=self.enable_multi_identity_train,
                )
            if timer: timer.end("model_forward")
        except torch.cuda.OutOfMemoryError:
            logger.error(f"[OOM] forward_single_clip: model forward OOM, seq_len={seq_len}\n{tb.format_exc()}")
            raise
        finally:
            del x, y, context  # forward 完成后立即释放输入张量
 
        # ---- loss compute ----
        if timer: timer.start("loss_compute")
        pred = pred_list[0]
        del pred_list
 
        # 计算生成 loss（target 部分）
        generation_loss = F_torch.mse_loss(pred[:, M_identity + M:].float(), v_target[0, :, M_identity + M:].float())
 
        # 计算主体帧重建 loss
        identity_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        if M_identity > 0:
            identity_loss = F_torch.mse_loss(pred[:, :M_identity].float(), v_target[0, :, :M_identity].float())
 
        # 计算 memory 重建 loss
        mem_recon_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        if self.enable_mem_recon_loss and M > 0:
            mem_recon_loss = F_torch.mse_loss(pred[:, M_identity:M_identity + M].float(), v_target[0, :, M_identity:M_identity + M].float())
 
        # 总 loss
        loss = generation_loss + identity_loss * self.identity_loss_weight + mem_recon_loss * self.mem_recon_loss_weight
 
        del pred, v_target  # v_target 是常量（no_grad），pred 的计算图由 loss 持有，均可释放
        if timer: timer.end("loss_compute")
 
        return loss, identity_loss, generation_loss, mem_recon_loss
 
    def forward(self, seq_data, timer=None):
        """
        从序列中随机选一个非首 clip 训练。
        第一个 clip 到被选 clip 之前的所有 clip 用于构建 memory_pool（不训练）。
 
        Args:
            seq_data: dict with 'clips' (list of clip dicts)
            timer: StepTimer instance for performance profiling
 
        Returns:
            loss: scalar tensor (可直接 backward)
        """
 
        if self.grad_ckpt or self.use_gradient_checkpointing_offload:
            for m in [self.low_noise_model, self.high_noise_model]:
                if m is None:
                    continue
                m.train()
                underlying = self._get_underlying_model(m)
                if underlying:
                    if hasattr(underlying, 'gradient_checkpointing'):
                        underlying.gradient_checkpointing = True
                    if hasattr(underlying, 'use_gradient_checkpointing_offload'):
                        underlying.use_gradient_checkpointing_offload = self.use_gradient_checkpointing_offload
 
        clips = seq_data['clips']
        device = next(self.parameters()).device
 
        if self.train_low_noise_only:
            model = self.low_noise_model
        elif self.train_high_noise_only:
            model = self.high_noise_model
        else:
            use_high_tensor = torch.tensor([1 if random.random() > 0.9 else 0], device=device)
            if dist.is_initialized():
                dist.broadcast(use_high_tensor, src=0) # 主节点广播决策
            use_high = use_high_tensor.item() == 1
            model = self.high_noise_model if (use_high and self.high_noise_model) else self.low_noise_model
 
        # T5 搬上 GPU，后续所有 _encode_text 调用复用，最终在 forward_single_clip 里一次性 offload
        self._t5_ensure_gpu(device)
 
        if len(clips) < 2:
            assert False, f"len(clips)={len(clips)}, should have been filtered before DDP forward"
 
        train_clip_idx = random.randint(1, len(clips) - 1)
 
        # 10% 概率跳过所有前序 shot，只保留上一个 shot，提升模型对单 shot 上下文的鲁棒性。
        skip_prior_shots = (train_clip_idx > 1) and (random.random() < self.skip_prior_shots_prob)
 
        # identity frame 机制：按概率开启，skip_prior_shots 时强制关闭
        use_identity = (random.random() < self.identity_frame_prob) and (not skip_prior_shots)
 
        logger.info(f"[Step] clip={train_clip_idx}/{len(clips)-1} | skip_prior_shots={skip_prior_shots} (p={self.skip_prior_shots_prob}) | use_identity={use_identity} (p={self.identity_frame_prob})")
 
        # 直接根据 clip_idx 过滤：只保留 clip_idx < train_clip_idx 的 identity frames
        identity_frames_before = []
        if use_identity:
            identity_frames_before = [
                item for item in seq_data.get('identity_frames', [])
                if item['clip_idx'] < train_clip_idx
            ]
 
        if identity_frames_before:
            if self.enable_multi_identity_train:
                # 多主体帧模式：使用全部
                selected_identity = identity_frames_before
            else:
                # 单主体帧模式：随机抽 1 个
                selected_identity = [random.choice(identity_frames_before)]
 
            # Debug log: 打印选出的 identity frame 信息
            for i, item in enumerate(selected_identity):
                logger.info(f"[Identity Select] #{i}: clip={item['clip_idx']}, "
                            f"frame_in_clip={item.get('frame_in_clip', '?')}, "
                            f"caption={item['caption'][:100]}...")
 
            identity_frames = [item['frame'] for item in selected_identity]
            raw_captions = [item['caption'] for item in selected_identity]
            if self.training and self.uncond_p > 0.0:
                raw_captions = [c if random.random() > self.uncond_p else "" for c in raw_captions]
            if timer: timer.start("identity_vae_encode")
            identity_frame_latents = self.encode_frames_to_memory(identity_frames, device)
            # encode_frames_to_memory 返回 4D (C, M_id, H, W)，统一加 batch 维变 5D (1, C, M_id, H, W)
            identity_frame_latents = identity_frame_latents.unsqueeze(0)
            if timer: timer.end("identity_vae_encode")
            # T5 编码 identity captions -> List[Tensor], 每个 shape [L, 4096]
            if timer: timer.start("identity_t5_encode")
            identity_frames_caption = self._encode_text(raw_captions, device)
            if timer: timer.end("identity_t5_encode")
            # 保存可视化用的 identity 信息
            identity_vis_info = {
                'frames': [f.detach().cpu() for f in identity_frames],
                'captions': raw_captions,
                'clip_indices': [item['clip_idx'] for item in selected_identity],
                'frame_in_clips': [item.get('frame_in_clip', '?') for item in selected_identity],
            }
        else:
            identity_frame_latents = None
            identity_frames_caption = None
            identity_vis_info = None
 
        cur_train_clip = clips[train_clip_idx]
        
        # 记录全局的物理时间偏移量
        current_time_offset = 0
        pre_time_indices = []
        pre_frames = []
        for i in range(train_clip_idx - 1):
            clip_i = clips[i]
            if not skip_prior_shots:
                mem_frames = clip_i['memory_frames']
                mem_indices = clip_i['memory_indices']
                for frame, local_idx in zip(mem_frames, mem_indices):
                    pre_frames.append(frame)
                    # 真实时间步 = 之前所有视频的总跨度 + 当前帧在当前视频里的相对时间步
                    pre_time_indices.append(current_time_offset + local_idx)
            # 无论是否采样，时间偏移量必须始终累加
            current_time_offset += clip_i['total_frames']
        # 历史 bank：前序 shot 采样后压到 MAX_MEMORY_FRAMES 帧
        if len(pre_frames) > MAX_MEMORY_FRAMES:
            sampled_indices = random.sample(range(len(pre_frames)), MAX_MEMORY_FRAMES)
            sampled_indices.sort()
            pre_frames = [pre_frames[idx] for idx in sampled_indices]
            pre_time_indices = [pre_time_indices[idx] for idx in sampled_indices]
 
        last_clip = clips[train_clip_idx - 1]
        last_caption = last_clip.get('caption', '')
        # last shot 均匀采样 MAX_MEMORY_FRAMES 帧
        last_video = last_clip['video']  # (C, F, H, W)
        F_last = last_video.shape[1]
        if F_last >= MAX_MEMORY_FRAMES:
            uniform_idx = torch.linspace(0, F_last - 1, MAX_MEMORY_FRAMES).long().tolist()
        else:
            uniform_idx = list(range(F_last))
        last_mem_frames = [last_video[:, i].clone() for i in uniform_idx]  # list of (C, H, W)，clone 断开对 last_video 的引用
        last_time_indices = [current_time_offset + i for i in uniform_idx]
        last_num_frames = len(last_mem_frames)
        del last_video  # 完整视频 tensor 不再需要，立即释放
 
        current_time_offset += last_clip['total_frames']  # 累加 Last Shot 的真实总长度
 
        if MAX_MEMORY_FRAMES == 1:
            # 只维持一个 memory：仅取 last shot 的第一帧，跳过历史 bank
            candidate_frames = last_mem_frames[:1]
            del pre_frames, last_mem_frames
            candidate_time_indices = last_time_indices[:1]
            time_indices = candidate_time_indices
        else:
            # 拼接：bank + last shot 候选池，送入 keyframe_query 选帧
            candidate_frames = pre_frames + last_mem_frames
            del pre_frames, last_mem_frames
            candidate_time_indices = pre_time_indices + last_time_indices
            time_indices = candidate_time_indices
 
        target_start_idx = current_time_offset
        target_lat_start_kfq = target_start_idx // 4  # 与 DiT 的 target_lat_start 对齐
        centered_time_indices = []
        for t in time_indices:
            rel_t = t // 4 - target_lat_start_kfq  # latent frame offset，与 DiT 保持同一时间尺度
            rel_t = max(-512, min(511, rel_t))
            centered_time_indices.append(512 + rel_t)
 
        # ---- [Timer] Memory Pool 构建 ----
        if timer: timer.start("memory_pool_build")
        try:
            memory_pool = self.encode_frames_to_memory(candidate_frames, device)  # [16, M, H, W]
            raw_memory_pool = memory_pool.clone()
        except torch.cuda.OutOfMemoryError:
            logger.error(f"[OOM] memory build from {len(candidate_frames)} frames\n{tb.format_exc()}")
            raise
        if timer: timer.end("memory_pool_build")
 
        if MAX_MEMORY_FRAMES == 1:
            # 只有 1 帧 memory，跳过 keyframe_query，直接使用
            selected_memories = memory_pool.unsqueeze(0)  # (1, C_lat, 1, H_lat, W_lat)
            selected_time_indices = time_indices  # 就是那 1 个时间索引
            selected_frame_indices = [0]
            del raw_memory_pool
        else:
            # ---- [Timer] Keyframe Selection ----
            if timer: timer.start("keyframe_selection")
 
            context_lens = None
            context_lens_global = None

            # local query 使用完整 caption；global query 仅使用 global caption 部分
            # 格式: "global caption: [人物描述]. shot caption: [动作描述]."
            _cap = cur_train_clip['caption']
            if 'shot caption:' in _cap:
                _global_part = _cap.split('shot caption:')[0]
                if _global_part.lower().startswith('global caption:'):
                    _global_part = _global_part[len('global caption:'):].strip()
                global_caption = _global_part.rstrip('. ').strip()
            else:
                global_caption = _cap

            # local query: 完整 caption
            local_prompts = [_cap]
            if self.training and self.uncond_p > 0.0:
                local_prompts = [p if random.random() > self.uncond_p else "" for p in local_prompts]
            context = self._encode_text(local_prompts, device)
            context_input = torch.stack([
                torch.cat(
                    [u, u.new_zeros(model.text_len - u.size(0), u.size(1))], dim=0)
                for u in context
            ]).contiguous()
            context = model.text_embedding(context_input).contiguous()
            del context_input

            context_global = None
            if self.split_learnable_query:
                assert self.global_query_num > 0, f"must have global_query_num > 0, got: {self.global_query_num}"
                # global query: 仅 global caption 部分
                global_prompts = [global_caption]
                if self.training and self.uncond_p > 0.0:
                    global_prompts = [p if random.random() > self.uncond_p else "" for p in global_prompts]
                context_global_raw = self._encode_text(global_prompts, device)
                context_global_input = torch.stack([
                    torch.cat(
                        [u, u.new_zeros(model.text_len - u.size(0), u.size(1))], dim=0)
                    for u in context_global_raw
                ]).contiguous()
                context_global = model.text_embedding(context_global_input).contiguous()
                del context_global_input


            dummy_pool = raw_memory_pool.clone().unsqueeze(0).requires_grad_(True)
 
            def _run_keyframe_query(pool, ctxt, ctxt_global=None):
                return model.keyframe_query(
                    memory_latents=pool,
                    freqs=model.freqs.to(device),
                    context=ctxt,
                    context_lens=context_lens,
                    time_indices=[centered_time_indices],
                    context_global=ctxt_global,
                    context_lens_global=context_lens_global,
                )
            if self.grad_ckpt or self.use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    attn_weights = checkpoint.checkpoint(
                        _run_keyframe_query,
                        dummy_pool,
                        context,
                        context_global,
                        use_reentrant=False
                    )
            else:
                attn_weights = _run_keyframe_query(dummy_pool, context, context_global)
            del context
 
            aw = attn_weights[0]  # [K, F]
            # aw_global = attn_weights[0][-self.global_query_num:]
            # aw = attn_weights[0][:-self.global_query_num]
            # # # 如果 query 输出的 K 帧数超过 MAX_MEMORY_FRAMES，只保留置信度最高的 top-MAX_MEMORY_FRAMES 个 query
            # # if aw.shape[0] > MAX_MEMORY_FRAMES:
            # #     confidence = aw.max(dim=-1).values  # [K]
            # #     topk_indices = confidence.topk(MAX_MEMORY_FRAMES).indices  # 保留置信度最高的 query
            # #     topk_indices, _ = topk_indices.sort()  # 保持时间顺序
            # #     aw = aw[topk_indices]  # [MAX_MEMORY_FRAMES, F]
            # aw = torch.cat((aw, aw_global), dim=0)
            selected_frame_indices = aw.argmax(dim=-1).tolist()
            selected_time_indices = [time_indices[idx] for idx in selected_frame_indices]
            selected_latents = torch.einsum('cfhw, kf -> ckhw', raw_memory_pool.to(device), aw.to(raw_memory_pool.dtype))

            
            del raw_memory_pool, attn_weights, aw, dummy_pool
            selected_memories = selected_latents.unsqueeze(0)
            del selected_latents
 
            if timer: timer.end("keyframe_selection")
        
        ### 组装待可视化的帧和数据结构
        vis_data = None
        global_step = seq_data.get('global_step', -1)
        if global_step != -1 and global_step % 30 == 0:
            # candidate_frames = bank(≤6) + last_shot(6)
            bank_frames = candidate_frames[:-last_num_frames] if last_num_frames < len(candidate_frames) else []
            last_shot_frames = candidate_frames[-last_num_frames:]
            sel_keyframes = [candidate_frames[idx] for idx in sorted(selected_frame_indices)]
 
            vis_data = {
                'mem_bank': [f.detach().cpu() for f in bank_frames],
                'last_shot': [f.detach().cpu() for f in last_shot_frames],
                'sel_keyframes': [f.detach().cpu() for f in sel_keyframes],
                'indices': selected_frame_indices,
                'time_steps': selected_time_indices,
                'target_prompt': cur_train_clip.get('caption', ''),
                'last_prompt': last_caption,
                'identity': identity_vis_info,
            }
 
        if memory_pool is None:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # print("finish step 3")
 
        # ---- 训练选中的 clip ----
        train_clip = clips[train_clip_idx]
        video_tensor = train_clip['video'].to(device)
        
        target_start_idx = current_time_offset 
        target_num_frames = video_tensor.shape[1]
        
        T_lat = (target_num_frames - 1) // 4 + 1
        
        # 以 target 起始帧为中心（512），memory 在 [0,511]，target 在 [512, 512+T_lat-1]
        # 与 keyframe_query 的 centered_time_indices 保持一致的 RoPE 编码约定
        target_lat_start = target_start_idx // 4
        mem_lat_abs = [t // 4 for t in selected_time_indices]
 
        mem_latent_indices = []
        for t in mem_lat_abs:
            rel_t = t - target_lat_start  # 负数，表示过去
            rel_t = max(-512, min(511, rel_t))
            mem_latent_indices.append(512 + rel_t)
 
        target_latent_indices = [min(512 + i, 1023) for i in range(T_lat)]
 
        # 为 identity_frame 添加 time_indices = 0
        if identity_frame_latents is not None:
            M_identity = identity_frame_latents.shape[2]
            identity_time_indices = [0] * M_identity
            full_dit_time_indices = identity_time_indices + mem_latent_indices + target_latent_indices
        else:
            full_dit_time_indices = mem_latent_indices + target_latent_indices
 
        del candidate_frames  # vis_data 已构建完毕，像素帧列表可释放
        # print(f"[TrainDiT] mem_lat_abs (latent): {mem_lat_abs}")  # 注释掉 print 减少同步开销
        # print(f"[TrainDiT] mem_latent_indices [0-511]: {mem_latent_indices}")
        # print(f"[TrainDiT] target_latent_indices [512+]: {target_latent_indices}")
        # print(f"[TrainDiT] full_dit_time_indices length={len(full_dit_time_indices)}: {full_dit_time_indices}")
        # 10% 概率使用当前 clip 的第一帧作为 first_frame 条件
        use_first_frame = random.random() < 0.5
        first_frame = None
        if use_first_frame:
            prev_clip = clips[train_clip_idx]
            prev_video = prev_clip['video']  # (C, F_prev, H, W)
            first_frame = prev_video[:, :1].unsqueeze(0).to(device)  # [1, C, 1, H, W]
 
        # ---- [Timer] forward_single_clip ----
        if timer: timer.start("forward_single_clip")
        try:
            loss, identity_loss, generation_loss, mem_recon_loss = self.forward_single_clip(
                model, video_tensor, selected_memories, train_clip['caption'], device,
                first_frame=first_frame,
                time_indices=[full_dit_time_indices],
                timer=timer,
                identity_frame_latents=identity_frame_latents,
                identity_frames_caption=identity_frames_caption,
            )
            if timer: timer.end("forward_single_clip")
 
        except torch.cuda.OutOfMemoryError:
            logger.error(f"[OOM] clip[{train_clip_idx}] forward, video={list(video_tensor.shape)}, mem={list(memory_pool.shape)}\n{tb.format_exc()}")
            raise
        finally:
            del video_tensor, memory_pool, selected_memories
            torch.cuda.empty_cache()
 
        # 存为 side-channel 属性，避免 DDP 遍历非 loss tensor 的 autograd graph
        self._last_identity_loss = identity_loss.detach()
        self._last_generation_loss = generation_loss.detach()
        self._last_mem_recon_loss = mem_recon_loss.detach()
        self._last_vis_data = vis_data
 
        return loss
 
    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
 
    def _collect_trainable_state_dict(self):
        """收集所有可训练参数：LoRA 参数 + keyframe_query 参数。"""
        state = {}
        for k, v in self.named_parameters():
            if v.requires_grad:
                state[k] = v.detach().cpu().clone()
        return state
 
    def save_trainable(self, output_dir, global_step):
        os.makedirs(output_dir, exist_ok=True)
        if self.use_sp and dist.is_initialized():
            dist.barrier()
        state = self._collect_trainable_state_dict()
        if len(state) == 0:
            return None
        path = os.path.join(output_dir, f"trainable_step{global_step}.pth")
        torch.save(state, path)
        logger.info(f"Saved {len(state)} trainable params to {path}")
        return path
 
    def save_checkpoint(self, output_dir, epoch, global_step, optimizer, accelerator=None):
        os.makedirs(output_dir, exist_ok=True)
        if self.use_sp and dist.is_initialized():
            dist.barrier()
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'trainable_state_dict': self._collect_trainable_state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': {
                'train_low_noise_only': self.train_low_noise_only,
                'train_high_noise_only': self.train_high_noise_only,
                'use_sp': self.use_sp,
                'boundary_timestep': self.boundary_timestep,
                'num_keyframes': self.num_keyframes,
            }
        }
        path = os.path.join(output_dir, f"checkpoint_step{global_step}.pth")
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
        return path
 
    def load_checkpoint(self, checkpoint_path, optimizer=None, target='both'):
        if not os.path.exists(checkpoint_path):
            return 0, 0
 
        epoch = 0
        global_step = 0
        saved_state = {}
        checkpoint = {}
        is_safetensors = checkpoint_path.endswith('.safetensors')
 
        # 1. 区分 safetensors 和传统 pth 格式
        if is_safetensors:
            try:
                from safetensors.torch import load_file
            except ImportError:
                raise ImportError("safetensors library is not installed. Please run: pip install safetensors")
 
            logger.info(f"Loading safetensors from {checkpoint_path}")
            checkpoint = load_file(checkpoint_path, device='cpu')
            saved_state = checkpoint
        else:
            logger.info(f"Loading standard checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            saved_state = checkpoint.get('trainable_state_dict',
                                         checkpoint.get('lora_state_dict', checkpoint))
            epoch = checkpoint.get('epoch', 0)
            global_step = checkpoint.get('global_step', 0)
 
        if saved_state:
            model_params = dict(self.named_parameters())
            loaded, skipped, shape_mismatch, reinitialized = 0, 0, 0, 0
            loaded_attn, loaded_ffn, loaded_kfq = 0, 0, 0
            fmt = 'safetensors' if is_safetensors else 'pth'
            logger.info(f"[load_checkpoint] format={fmt}, target={target}, "
                        f"checkpoint_keys={len(saved_state)}, model_params={len(model_params)}")
 
            # ---- DEBUG: 打印 checkpoint 原始 key 样例 ----
            ckpt_keys_list = list(saved_state.keys())
            logger.info(f"[DEBUG-ckpt] Checkpoint raw keys (first 10):")
            for ck in ckpt_keys_list[:10]:
                v = saved_state[ck]
                logger.info(f"[DEBUG-ckpt]   {ck}: shape={list(v.shape)}, dtype={v.dtype}")
            logger.info(f"[DEBUG-ckpt] Checkpoint raw keys (last 5):")
            for ck in ckpt_keys_list[-5:]:
                v = saved_state[ck]
                logger.info(f"[DEBUG-ckpt]   {ck}: shape={list(v.shape)}, dtype={v.dtype}")
            # 统计 checkpoint 中各类 key
            ckpt_lora_a = [k for k in ckpt_keys_list if 'lora_A' in k]
            ckpt_lora_b = [k for k in ckpt_keys_list if 'lora_B' in k]
            ckpt_kfq = [k for k in ckpt_keys_list if 'keyframe_query' in k]
            ckpt_attn = [k for k in ckpt_keys_list if 'self_attn' in k or 'cross_attn' in k]
            ckpt_ffn = [k for k in ckpt_keys_list if '.ffn.' in k]
            logger.info(f"[DEBUG-ckpt] Key breakdown: lora_A={len(ckpt_lora_a)}, lora_B={len(ckpt_lora_b)}, "
                        f"attn={len(ckpt_attn)}, ffn={len(ckpt_ffn)}, keyframe_query={len(ckpt_kfq)}")
 
            # DEBUG: 打印 model_params key 样例
            mp_keys_list = list(model_params.keys())
            logger.info(f"[DEBUG-model] Model param keys (first 10):")
            for mk in mp_keys_list[:10]:
                p = model_params[mk]
                logger.info(f"[DEBUG-model]   {mk}: shape={list(p.shape)}, dtype={p.dtype}, requires_grad={p.requires_grad}")
            mp_lora = [k for k in mp_keys_list if 'lora_' in k]
            mp_kfq = [k for k in mp_keys_list if 'keyframe_query' in k]
            logger.info(f"[DEBUG-model] Model param breakdown: total={len(mp_keys_list)}, lora={len(mp_lora)}, keyframe_query={len(mp_kfq)}")
            # ---- END DEBUG ----
 
            # 2. Key 转换逻辑
            mapped_state = {}
 
            if is_safetensors:
                # safetensors 格式的 key 特征：
                #   - 前缀为 "base_model.model." 而非 "low_noise_model.base_model.model."
                #   - LoRA key 使用 "lora_A.weight" 而非 "lora_A.default.weight"
                #   - 不包含 keyframe_query 参数（需要随机初始化）
                logger.info(f"[load_checkpoint] Safetensors detected: applying key transformations "
                            f"(.default. insertion + model prefix mapping)")
                for k, v in saved_state.items():
                    parsed_k = k
 
                    # 2a. 插入 ".default." 适配器层级
                    #     safetensors: lora_A.weight -> lora_A.default.weight
                    if "lora_A.weight" in parsed_k and "lora_A.default.weight" not in parsed_k:
                        parsed_k = parsed_k.replace("lora_A.weight", "lora_A.default.weight")
                    elif "lora_B.weight" in parsed_k and "lora_B.default.weight" not in parsed_k:
                        parsed_k = parsed_k.replace("lora_B.weight", "lora_B.default.weight")
 
                    # 2b. 添加模型前缀
                    #     safetensors: base_model.model.* -> {low,high}_noise_model.base_model.model.*
                    if parsed_k.startswith("base_model.model."):
                        if self.low_noise_model is not None and target in ('both', 'low'):
                            mapped_state[f"low_noise_model.{parsed_k}"] = v
                        if self.high_noise_model is not None and target in ('both', 'high'):
                            mapped_state[f"high_noise_model.{parsed_k}"] = v
                    else:
                        # 不以 base_model.model. 开头的 key，保持原样
                        mapped_state[parsed_k] = v
 
                # 2c. safetensors 不包含 keyframe_query 参数，对模型中的 keyframe_query 随机初始化
                kfq_init_count = 0
                for k, param in model_params.items():
                    if "keyframe_query" in k and k not in mapped_state and param.requires_grad:
                        with torch.no_grad():
                            if 'queries' in k:
                                torch.nn.init.normal_(param, std=param.shape[-1]**-0.5)
                            elif "lora_B" in k:
                                torch.nn.init.zeros_(param)
                            elif "lora_A" in k:
                                torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                            elif param.dim() >= 2:
                                torch.nn.init.normal_(param, std=0.02)
                            else:
                                torch.nn.init.zeros_(param)
                        kfq_init_count += 1
                logger.info(f"[load_checkpoint] Safetensors: randomly initialized {kfq_init_count} "
                            f"keyframe_query params (not present in safetensors)")
            else:
                # pth 格式：key 已包含 .default. 和完整前缀，但仍需处理旧格式兼容
                for k, v in saved_state.items():
                    parsed_k = k
 
                    # 处理可能缺少的 ".default." (兼容旧 pth)
                    if "lora_A.weight" in parsed_k and "lora_A.default.weight" not in parsed_k:
                        parsed_k = parsed_k.replace("lora_A.weight", "lora_A.default.weight")
                    elif "lora_B.weight" in parsed_k and "lora_B.default.weight" not in parsed_k:
                        parsed_k = parsed_k.replace("lora_B.weight", "lora_B.default.weight")
 
                    if parsed_k.startswith("base_model.model."):
                        if self.low_noise_model is not None and target in ('both', 'low'):
                            mapped_state[f"low_noise_model.{parsed_k}"] = v
                        if self.high_noise_model is not None and target in ('both', 'high'):
                            mapped_state[f"high_noise_model.{parsed_k}"] = v
                    elif "keyframe_query" in parsed_k and parsed_k not in model_params:
                        low_k = f"low_noise_model.base_model.model.{parsed_k}"
                        high_k = f"high_noise_model.base_model.model.{parsed_k}"
                        if self.low_noise_model is not None and target in ('both', 'low'):
                            mapped_state[low_k] = v
                        if self.high_noise_model is not None and target in ('both', 'high'):
                            mapped_state[high_k] = v
                    else:
                        mapped_state[parsed_k] = v
 
            # ---- DEBUG: 打印 key 映射结果 ----
            mapped_keys_list = list(mapped_state.keys())
            logger.info(f"[DEBUG-mapping] Mapped state keys: {len(mapped_keys_list)}")
            logger.info(f"[DEBUG-mapping] Mapped keys (first 10):")
            for mk in mapped_keys_list[:10]:
                logger.info(f"[DEBUG-mapping]   {mk}: shape={list(mapped_state[mk].shape)}")
            # 检查映射后的 key 有多少匹配 model_params
            matched = [k for k in mapped_keys_list if k in model_params]
            unmatched_ckpt = [k for k in mapped_keys_list if k not in model_params]
            missing_in_ckpt = [k for k in model_params if k not in mapped_state]
            logger.info(f"[DEBUG-mapping] Matched to model: {len(matched)}, "
                        f"Unmatched (in ckpt but not in model): {len(unmatched_ckpt)}, "
                        f"Missing (in model but not in ckpt): {len(missing_in_ckpt)}")
            if unmatched_ckpt:
                logger.info(f"[DEBUG-mapping] Unmatched checkpoint keys (first 10):")
                for uk in unmatched_ckpt[:10]:
                    logger.info(f"[DEBUG-mapping]   {uk}")
            if missing_in_ckpt:
                logger.info(f"[DEBUG-mapping] Missing in checkpoint (first 10):")
                for mik in missing_in_ckpt[:10]:
                    logger.info(f"[DEBUG-mapping]   {mik}")
            # ---- END DEBUG mapping ----
 
            for k, param in model_params.items():
                if k in mapped_state:
                    v = mapped_state[k]
                    if param.shape == v.shape:
                        param.data.copy_(v.to(param.device))
                        loaded += 1
                        if "self_attn" in k or "cross_attn" in k:
                            loaded_attn += 1
                        elif ".ffn." in k:
                            loaded_ffn += 1
                        elif "keyframe_query" in k:
                            loaded_kfq += 1
                    else:
                        logger.warning(f"Shape mismatch for {k}: model expects {param.shape}, checkpoint has {v.shape}")
                        shape_mismatch += 1
                else:
                    # safetensors 的 keyframe_query 已在上面预初始化，这里跳过
                    if is_safetensors and "keyframe_query" in k:
                        reinitialized += 1
                        continue
 
                    # pth 格式缺失的参数做初始化：
                    #   - FFN LoRA（新增）：lora_B 归零保证热启行为等价于原模型，lora_A 随机初始化
                    #   - keyframe_query（旧 checkpoint 无此模块）：正态初始化
                    is_ffn_lora = (".ffn." in k) and ("lora_" in k)
                    is_keyframe = "keyframe_query" in k
                    if is_ffn_lora or is_keyframe:
                        with torch.no_grad():
                            if "lora_B" in k:
                                torch.nn.init.zeros_(param)
                            elif "lora_A" in k:
                                torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                            elif 'queries' in k:
                                torch.nn.init.normal_(param, std=param.shape[-1]**-0.5)
                            elif param.dim() >= 2:
                                torch.nn.init.normal_(param, std=0.02)
                            else:
                                torch.nn.init.zeros_(param)
                        reinitialized += 1
                    else:
                        skipped += 1
 
            logger.info(
                f"[load_checkpoint] Loaded {loaded} params "
                f"(attn={loaded_attn}, ffn={loaded_ffn}, keyframe_query={loaded_kfq}). "
                f"Re-initialized {reinitialized} missing params. "
                f"Skipped {skipped} unmatched. "
                f"Shape mismatch {shape_mismatch}."
            )
 
        if optimizer is not None and not is_safetensors:
            if 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logger.info("Successfully loaded optimizer state.")
                except Exception as e:
                    logger.warning(f"Failed to load optimizer state: {e}")
 
        return epoch, global_step
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main():
    parser = argparse.ArgumentParser(description="Memento M2V Training")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--resolution", type=str, default="320*192")
    parser.add_argument("--frame_num", type=int, default=21)
    parser.add_argument("--num_keyframes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Must be 1: each sample is a full sequence")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--boundary", type=float, default=0.9)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--use_gradient_checkpointing_offload", action="store_true",
                        help="Offload gradient checkpointing saved tensors to CPU (saves more VRAM)")
    parser.add_argument("--train_both_models", action="store_true")
    parser.add_argument("--train_high_noise_only", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--resume_from_low", type=str, default=None,
                        help="Hot-start low_noise_model only (e.g. backbone_low_noise.safetensors)")
    parser.add_argument("--resume_from_high", type=str, default=None,
                        help="Hot-start high_noise_model only (e.g. backbone_high_noise.safetensors)")
    parser.add_argument("--save_every_n_steps", type=int, default=100)
    parser.add_argument("--exp_name", type=str, default="m2v_query_train")
    parser.add_argument("--use_sp", action="store_true")
    parser.add_argument("--uncond_p", type=float, default=0.1)
    parser.add_argument("--enable_multi_identity_train", action="store_true",
                        help="Use all identity frames with per-frame cross-attention context (default: random single)")
    parser.add_argument("--identity_frame_prob", type=float, default=0.4,
                        help="Probability of enabling identity frame mechanism per step (default: 0.4)")
    parser.add_argument("--skip_prior_shots_prob", type=float, default=0.1,
                        help="Probability of skipping all prior shots except the last one (default: 0.1)")
    parser.add_argument("--compile_t5", action="store_true",
                        help="Use torch.compile to accelerate T5 encoder (fixed 512-length input)")
    parser.add_argument("--no_t5_offload", action="store_true",
                        help="Keep T5 on GPU (default: offload to CPU after each encode to save VRAM)")
    parser.add_argument("--enable_mem_recon_loss", action="store_true",
                        help="Enable reconstruction loss on memory frames (default: off)")
    parser.add_argument("--mem_recon_loss_weight", type=float, default=1.0,
                        help="Weight for memory reconstruction loss (default: 1.0)")
    parser.add_argument("--identity_loss_weight", type=float, default=1.0,
                        help="Weight for identity frame reconstruction loss (default: 1.0)")
    parser.add_argument("--split_identity_attn", action="store_true",
                        help="Enable split identity+memory / memory+generation self-attention")
    # 加入两个新参数 split_learnable_query / global_query_num
    parser.add_argument("--split_learnable_query", action="store_true",
                        help="Split learnable queries into per-clip and global groups")
    parser.add_argument("--global_query_num", type=int, default=0,
                        help="Number of global queries for split_learnable_query mode (default: 0, disabled)")
    parser.add_argument("--max_memory_frames", type=int, default=10,
                        help="Max number of memory frames per clip (default: 10)")
    parser.add_argument("--selected_local_num", type=int, default=6,
                        help="Max number of memory frames per clip (default: 10)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker 数量（多机时可按需调低）")
    parser.add_argument("--dry_run", action="store_true",
                        help="跑一个 batch 前向传播后退出，用于排查启动错误")
    parser.add_argument("--cosine_annealing_steps", type=int, default=0,
                        help="Cosine annealing 退火步数：从 resume 步开始，经过此步数后 LR 降到 0；0 表示禁用")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="最大训练步数（到达后停止）；0 表示由 cosine_annealing_steps 控制")
    parser.add_argument("--lr_resume_step", type=int, default=-1,
                        help="自定义 LR 从哪一步开始 resume（-1 表示跟随 checkpoint 的 global_step）")
    parser.add_argument("--ffmpeg_bin", type=str, default=None,
                        help="ffmpeg 可执行文件路径（默认自动探测；conda 环境可传 /path/to/conda/bin/ffmpeg）")
    args = parser.parse_args()
 
    global MAX_MEMORY_FRAMES
    MAX_MEMORY_FRAMES = args.max_memory_frames
 
    assert args.batch_size == 1, "batch_size must be 1 (each sample is a full sequence)"
 
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )
 
    local_rank = accelerator.local_process_index
    world_size = accelerator.num_processes
 
    # ------------------------------------------------------------------
    # Dataset & DataLoader
    # ------------------------------------------------------------------
    dataset = StorySequenceDataset(
        args.data_path, resolution=args.resolution, frame_num=args.frame_num,
        ffmpeg_bin=args.ffmpeg_bin)
    if accelerator.is_main_process:
        logger.info(f"Dataset: {len(dataset)} sequences")
 
    dataloader = DataLoader(
        dataset, batch_size=1,
        shuffle=True,
        collate_fn=sequence_collate_fn, num_workers=args.num_workers,
        prefetch_factor=4,           # 增加 prefetch 减少等待时间
        persistent_workers=True,      # 保持 worker 进程，避免每次 epoch 重新创建
        pin_memory=True)              # 使用 pinned memory 加速 GPU 传输
 
    use_grad_ckpt = args.gradient_checkpointing and not args.no_gradient_checkpointing
    train_low_noise_only = not args.train_both_models and not args.train_high_noise_only
    train_high_noise_only = args.train_high_noise_only and not args.train_both_models
 
    model = M2VQueryTrainingModule(
        args.checkpoint_dir,
        lora_rank=args.lora_rank,
        device_id=local_rank,
        boundary=args.boundary,
        grad_ckpt=use_grad_ckpt,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        train_low_noise_only=train_low_noise_only,
        train_high_noise_only=train_high_noise_only,
        use_sp=args.use_sp,
        num_keyframes=args.num_keyframes,
        uncond_p=args.uncond_p,
        enable_multi_identity_train=args.enable_multi_identity_train,
        identity_loss_weight=args.identity_loss_weight,
        identity_frame_prob=args.identity_frame_prob,
        skip_prior_shots_prob=args.skip_prior_shots_prob,
        compile_t5=args.compile_t5,
        t5_offload=not args.no_t5_offload,
        enable_mem_recon_loss=args.enable_mem_recon_loss,
        mem_recon_loss_weight=args.mem_recon_loss_weight,
        split_identity_attn=args.split_identity_attn,
        split_learnable_query=args.split_learnable_query,
        global_query_num=args.global_query_num,
        selected_local_num = args.selected_local_num
    )
 
    gc.collect()
    torch.cuda.empty_cache()
 
    train_params = [p for p in model.parameters() if p.requires_grad]
 
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    if accelerator.is_main_process:
        logger.info(f"trainable params: {total_trainable:,} ({total_trainable / 1e6:.2f}M), "
                    f"frozen params: {total_frozen:,} ({total_frozen / 1e6:.2f}M), "
                    f"all params: {total_trainable + total_frozen:,} ({(total_trainable + total_frozen) / 1e6:.2f}M)")
 
    optimizer = bnb.optim.AdamW8bit(train_params, lr=args.lr)
 
    # ---- Accelerate prepare: model, optimizer, dataloader ----
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
 
    if accelerator.is_main_process:
        opt_param_count = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
        opt_param_num = sum(len(group['params']) for group in optimizer.param_groups)
        logger.info(f"[OPTIMIZER] param_groups={len(optimizer.param_groups)}, "
                    f"total tensors={opt_param_num}, "
                    f"total elements={opt_param_count:,} ({opt_param_count / 1e6:.2f}M)")
        logger.info(f"[Accelerate] distributed_type={accelerator.distributed_type}, "
                    f"mixed_precision={accelerator.mixed_precision}, "
                    f"num_processes={accelerator.num_processes}")
 
    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch = 0
    global_step = 0
    if args.resume_from:
        unwrapped = accelerator.unwrap_model(model)
        start_epoch, global_step = unwrapped.load_checkpoint(args.resume_from, optimizer, target='both')
        start_epoch += 1
    else:
        unwrapped = accelerator.unwrap_model(model)
        loaded_any = False
        if args.resume_from_low:
            _, global_step = unwrapped.load_checkpoint(args.resume_from_low, optimizer=None, target='low')
            loaded_any = True
        if args.resume_from_high:
            _, global_step = unwrapped.load_checkpoint(args.resume_from_high, optimizer=None, target='high')
            loaded_any = True
        if loaded_any:
            start_epoch = 0
            # global_step 保留 checkpoint 里的值，退火从该步继续
 
    # ------------------------------------------------------------------
    # TensorBoard (main process only)
    # ------------------------------------------------------------------
    writer = None
    if accelerator.is_main_process:
        log_dir = os.path.join(args.output_dir, "logs", args.exp_name)
        writer = SummaryWriter(log_dir, purge_step=global_step)

    # ------------------------------------------------------------------
    # LR Scheduler（cosine annealing，从当前 resume 步起退火）
    # ------------------------------------------------------------------
    scheduler = None
    if args.cosine_annealing_steps > 0:
        lr_step = args.lr_resume_step if args.lr_resume_step >= 0 else global_step
        for pg in optimizer.param_groups:
            pg['initial_lr'] = args.lr
        last_ep = lr_step if lr_step > 0 else -1
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.cosine_annealing_steps,
            eta_min=0.0,
            last_epoch=last_ep,
        )
        if lr_step > 0:
            correct_lr = args.lr * (1 + math.cos(math.pi * lr_step / args.cosine_annealing_steps)) / 2
            for pg in optimizer.param_groups:
                pg['lr'] = correct_lr
        if accelerator.is_main_process:
            if args.resume_from:
                resume_desc = f"resume_from={args.resume_from}, global_step={global_step}"
            elif args.resume_from_low or args.resume_from_high:
                sources = []
                if args.resume_from_low:
                    sources.append(f"low={args.resume_from_low}")
                if args.resume_from_high:
                    sources.append(f"high={args.resume_from_high}")
                resume_desc = f"hot-start ({', '.join(sources)}), global_step={global_step} (safetensors 不含 step 元数据时为 0)"
            else:
                resume_desc = f"from scratch, global_step={global_step}"
            logger.info(
                f"CosineAnnealingLR enabled: T_max={args.cosine_annealing_steps} steps, "
                f"initial_lr={args.lr:.2e}, eta_min=0 | {resume_desc}"
            )
 
    # ------------------------------------------------------------------
    # Training loop with timing
    # ------------------------------------------------------------------
    # 初始化计时器
    timer = StepTimer(use_cuda=True, name="train_step")
    dataloader_time_start = time.perf_counter()

    epoch = start_epoch
    training_done = False
    while not training_done:
        # 多机 DistributedSampler 需要 set_epoch 确保每个 epoch 数据顺序不同
        if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
        if args.use_sp:
            torch.manual_seed(epoch + 42)
            random.seed(epoch + 42)
            np.random.seed(epoch + 42)

        model.train()
        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        total_steps = args.max_steps if args.max_steps > 0 else (args.cosine_annealing_steps if args.cosine_annealing_steps > 0 else -1)
        data_iter = tqdm(dataloader,
                         desc=f"step {global_step}/{total_steps}",
                         dynamic_ncols=True) if accelerator.is_main_process else dataloader

        for step, seq_data in enumerate(data_iter):
            # ---- [Timer] Dataloader 时间 ----
            dataloader_time = time.perf_counter() - dataloader_time_start

            seq_data['global_step'] = global_step
            try:
                # find_unused_parameters=False 要求所有 rank 要么都做这个 step 要么都跳过，
                # 否则 DDP bucket 永远不 ready 导致死锁。
                # 在进 DDP forward 之前同步 skip 决策。
                skip_step = torch.tensor(
                    1 if len(seq_data.get('clips', [])) < 2 else 0,
                    device=accelerator.device, dtype=torch.long)
                if accelerator.num_processes > 1:
                    dist.all_reduce(skip_step, op=dist.ReduceOp.MAX)
                if skip_step.item():
                    logger.warning(f"[rank {accelerator.process_index}] skipping step, n_clips < 2")
                    optimizer.zero_grad()
                    dataloader_time_start = time.perf_counter()
                    continue

                # ---- [Timer] Step 总时间 ----
                timer.start("step_total")
 
                # accumulate() handles gradient_accumulation_steps internally
                with accelerator.accumulate(model):
                    # ---- [Timer] Forward ----
                    timer.start("forward_total")
                    loss = model(seq_data, timer=timer)
                    # 从 side-channel 属性读取监控用的 sub-losses 和 vis_data（已 detach，不影响 DDP）
                    underlying = model.module if hasattr(model, 'module') else model
                    identity_loss = getattr(underlying, '_last_identity_loss', None)
                    generation_loss = getattr(underlying, '_last_generation_loss', None)
                    mem_recon_loss = getattr(underlying, '_last_mem_recon_loss', None)
                    vis_data = getattr(underlying, '_last_vis_data', None)
                    timer.end("forward_total")
 
                    # ---- [Timer] Backward ----
                    timer.start("backward")
                    accelerator.backward(loss)
                    timer.end("backward")
 
                    if accelerator.sync_gradients:
                        timer.start("clip_grad")
                        accelerator.clip_grad_norm_(train_params, max_norm=1.0)
                        timer.end("clip_grad")
 
                    # ---- [Timer] Optimizer Step ----
                    timer.start("optimizer_step")
                    optimizer.step()
                    if scheduler is not None and accelerator.sync_gradients:
                        scheduler.step()
                    optimizer.zero_grad()
                    timer.end("optimizer_step")
 
                timer.end("step_total")
 
                # 跨 rank 收集 loss 均值，避免只看 rank 0 的局部值
                loss_tensor = torch.tensor([
                    loss.item(),
                    identity_loss.item() if identity_loss is not None else 0.0,
                    generation_loss.item() if generation_loss is not None else 0.0,
                    mem_recon_loss.item() if mem_recon_loss is not None else 0.0,
                ], device=accelerator.device)
                if accelerator.num_processes > 1:
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                loss_val = loss_tensor[0].item()
                identity_loss_val = loss_tensor[1].item()
                generation_loss_val = loss_tensor[2].item()
                mem_recon_loss_val = loss_tensor[3].item()
                epoch_loss_sum += loss_val
                epoch_loss_count += 1
 
                # ---- 打印计时统计 ----
                if accelerator.is_main_process:
                    current_lr = optimizer.param_groups[0]['lr']
                    # 更新 tqdm 显示
                    if hasattr(data_iter, 'set_postfix'):
                        data_iter.set_postfix(
                            loss=f"{loss_val:.4f}",
                            id=f"{identity_loss_val:.4f}",
                            gen=f"{generation_loss_val:.4f}",
                            mem=f"{mem_recon_loss_val:.4f}",
                            lr=f"{current_lr:.2e}",
                            seq=seq_data.get('sequence_id', '?')[:12],
                            step=f"{global_step}",
                            dt=f"{dataloader_time:.1f}s"
                        )
                    if scheduler is not None:
                        logger.info(f"[Step {global_step}] lr={current_lr:.6e}")
 
                    # 记录 dataloader 时间
                    timer.start("dataloader")
                    timer.end("dataloader")
                    # 用实际测量的时间覆盖自动计时
                    if "dataloader" in timer.durations:
                        timer.durations["dataloader"][-1] = dataloader_time
 
                    # 每 2 步打印详细计时
                    if global_step > 0 and global_step % 2 == 0:
                        timer.print_summary(window=10, top_n=20)
                        # 打印 GPU 显存使用情况
 
                    if writer is not None:
                        writer.add_scalar('train/loss_total', loss_val, global_step)
                        writer.add_scalar('train/loss_identity', identity_loss_val, global_step)
                        writer.add_scalar('train/loss_generation', generation_loss_val, global_step)
                        writer.add_scalar('train/loss_mem_recon', mem_recon_loss_val, global_step)
                        writer.add_scalar('train/lr', current_lr, global_step)
                        if vis_data is not None:
                            def denorm(frame_list):
                                if not frame_list: return None
                                t = torch.stack(frame_list, dim=0).to(torch.float32)
                                t = torch.clamp((t + 1.0) / 2.0, 0, 1)
                                return t
 
                            if vis_data.get('mem_bank'):
                                grid_mem = vutils.make_grid(denorm(vis_data['mem_bank']), nrow=8)
                                writer.add_image('Memory_Selection/1_Memory_Bank', grid_mem, global_step)
 
                            if vis_data.get('last_shot'):
                                grid_last = vutils.make_grid(denorm(vis_data['last_shot']), nrow=8)
                                writer.add_image('Memory_Selection/2_Last_Shot', grid_last, global_step)
 
                            if vis_data.get('sel_keyframes'):
                                grid_sel = vutils.make_grid(denorm(vis_data['sel_keyframes']), nrow=8)
                                writer.add_image('Memory_Selection/3_Selected_Keyframes', grid_sel, global_step)
 
                            idx_str = (
                                f"**Indices**: {vis_data.get('indices')}\n\n"
                                f"**Timesteps**: {vis_data.get('time_steps')}\n\n"
                                f"**Target Video Prompt**: {vis_data.get('target_prompt', '')}\n\n"
                                f"**Last Shot Prompt**: {vis_data.get('last_prompt', '')}\n"
                            )
                            writer.add_text('Memory_Selection/Info', idx_str, global_step)
 
                            # ---- Identity Frames 可视化 ----
                            id_info = vis_data.get('identity')
                            if id_info and id_info.get('frames'):
                                grid_id = vutils.make_grid(denorm(id_info['frames']), nrow=8)
                                writer.add_image('Identity_Frames/1_Frames', grid_id, global_step)
 
                                id_lines = []
                                for i, (cidx, fidx, cap) in enumerate(zip(
                                        id_info['clip_indices'],
                                        id_info['frame_in_clips'],
                                        id_info['captions'])):
                                    cap_short = cap[:120] + '...' if len(cap) > 120 else cap
                                    id_lines.append(f"- Frame {i}: clip={cidx}, frame_in_clip={fidx}, caption: {cap_short}")
                                id_str = (
                                    f"**Num Identity Frames**: {len(id_info['frames'])}\n\n"
                                    + "\n\n".join(id_lines)
                                )
                                writer.add_text('Identity_Frames/Info', id_str, global_step)
                global_step += 1
                # 到达停止条件，结束训练
                stop_step = args.max_steps if args.max_steps > 0 else args.cosine_annealing_steps
                if stop_step > 0 and global_step >= stop_step:
                    training_done = True

            except torch.cuda.OutOfMemoryError:
                import traceback
                logger.error(f"[OOM] step={step}, global_step={global_step}, "
                             f"seq={seq_data.get('sequence_id', '?')}, clips={len(seq_data.get('clips', []))}")
                logger.error(f"[OOM] traceback:\n{traceback.format_exc()}")
                logger.error(f"[OOM] CUDA memory summary:\n{torch.cuda.memory_summary(local_rank)}")
                gc.collect()
                torch.cuda.empty_cache()
                continue
            except Exception:
                import traceback
                logger.error(f"[ERROR] step={step}, global_step={global_step}\n{traceback.format_exc()}")
                raise
 
            if accelerator.is_main_process and global_step % args.save_every_n_steps == 0 and global_step > 0:
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_trainable(args.output_dir, global_step)
                unwrapped.save_checkpoint(args.output_dir, epoch, global_step, optimizer, accelerator)
 
            # ---- [Timer] GC & 显存清理 ----
            timer.start("gc_cleanup")
            if 'clips' in seq_data:
                for c in seq_data['clips']:
                    c.pop('video', None)
                    c.pop('memory_frames', None)
            del seq_data
            gc.collect()
            torch.cuda.empty_cache()
            timer.end("gc_cleanup")
 
            # 准备下一个 dataloader 计时
            dataloader_time_start = time.perf_counter()

            if training_done:
                break

        # epoch end：记录平均 loss，按需保存
        if accelerator.is_main_process and epoch_loss_count > 0:
            epoch_avg = epoch_loss_sum / epoch_loss_count
            logger.info(f"Epoch {epoch} avg loss: {epoch_avg:.4f} (global_step={global_step})")
            if writer is not None:
                writer.add_scalar('train/loss_epoch', epoch_avg, epoch)

        epoch += 1

    # 训练结束，保存最终 checkpoint
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_trainable(args.output_dir, global_step)
        unwrapped.save_checkpoint(args.output_dir, epoch, global_step, optimizer, accelerator)
        logger.info(f"Training finished at global_step={global_step}, checkpoint saved.")

    accelerator.wait_for_everyone()
    if writer:
        writer.flush()
        writer.close()
 
 
if __name__ == "__main__":
    main()