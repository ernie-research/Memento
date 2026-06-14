import math
import re
import types
import copy
import gc
import glob
import random
import av
from PIL import Image
from contextlib import contextmanager
from functools import partial
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict
from tqdm import tqdm
import os
import sys
from wan.distributed.fsdp import shard_model
from wan.distributed.sequence_parallel_memory import sp_attn_forward, sp_dit_forward
from wan.distributed.util import get_world_size
from wan.modules.model_memory import WanModel_Memory
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae2_1 import Wan2_1_VAE
from wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from decord import VideoReader, cpu
import torchvision.transforms.functional as TF
from safetensors.torch import load_file

import logging
logger = logging.getLogger()


def _load_base_state_dict(checkpoint_dir, subfolder):
    """
    直接从 safetensors 分片文件读取 state_dict，避免先实例化 WanModel_base 再 del 的双倍内存开销。
    """
    import json
    subfolder_path = os.path.join(checkpoint_dir, subfolder)
    index_path = os.path.join(subfolder_path, "diffusion_pytorch_model.safetensors.index.json")

    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]

        pe_shard_file = weight_map["patch_embedding.weight"]
        pe_shard_path = os.path.join(subfolder_path, pe_shard_file)
        from safetensors import safe_open
        with safe_open(pe_shard_path, framework="pt", device="cpu") as f_st:
            pretrained_in_dim = f_st.get_tensor("patch_embedding.weight").shape[1]

        shard_files = sorted(set(weight_map.values()))
        shard_paths = [os.path.join(subfolder_path, f) for f in shard_files]

        from concurrent.futures import ThreadPoolExecutor
        def _load_shard(path):
            return load_file(path, device="cpu")

        state_dict = {}
        with ThreadPoolExecutor(max_workers=min(len(shard_paths), 4)) as pool:
            for partial_dict in pool.map(_load_shard, shard_paths):
                state_dict.update(partial_dict)

        logging.info(f"Loaded base state_dict from {len(shard_files)} shards in {subfolder_path}")
    else:
        single_path = os.path.join(subfolder_path, "diffusion_pytorch_model.safetensors")
        if not os.path.exists(single_path):
            raise FileNotFoundError(f"Cannot find safetensors weights in {subfolder_path}")
        state_dict = load_file(single_path, device="cpu")
        pretrained_in_dim = state_dict["patch_embedding.weight"].shape[1]
        logging.info(f"Loaded base state_dict from {single_path}")

    return state_dict, pretrained_in_dim


def _create_model_on_meta(config, pretrained_in_dim, num_keyframes=10, split_identity_attn=False,
                          split_learnable_query=False, global_query_num=0, selected_local_num=6):
    orig_init_weights = WanModel_Memory.init_weights
    WanModel_Memory.init_weights = lambda self: None

    try:
        with torch.device('meta'):
            model = WanModel_Memory(
                model_type=config.model_type if hasattr(config, 'model_type') else 'i2v',
                patch_size=tuple(config.patch_size),
                text_len=config.text_len,
                in_dim=pretrained_in_dim,
                dim=config.dim,
                ffn_dim=config.ffn_dim,
                freq_dim=config.freq_dim,
                text_dim=config.text_dim if hasattr(config, 'text_dim') else 4096,
                out_dim=config.out_dim if hasattr(config, 'out_dim') else 16,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                window_size=tuple(config.window_size) if hasattr(config, 'window_size') else (-1, -1),
                qk_norm=config.qk_norm if hasattr(config, 'qk_norm') else True,
                cross_attn_norm=config.cross_attn_norm if hasattr(config, 'cross_attn_norm') else True,
                num_keyframes=num_keyframes,
                keyframe_temperature=0.1,
                split_identity_attn=split_identity_attn,
                split_learnable_query=split_learnable_query,
                global_query_num=global_query_num,
                selected_local_num = selected_local_num
            )
    finally:
        WanModel_Memory.init_weights = orig_init_weights

    return model


def _materialize_meta_params(model):
    for name, param in model.named_parameters():
        if param.device.type == 'meta':
            real = torch.empty(param.shape, dtype=param.dtype, device='cpu')
            nn.init.normal_(real, std=0.02)
            setattr_nested(model, name, nn.Parameter(real, requires_grad=param.requires_grad))
    for name, buf in model.named_buffers():
        if buf.device.type == 'meta':
            real = torch.empty(buf.shape, dtype=buf.dtype, device='cpu')
            real.zero_()
            setattr_nested(model, name, real)


def setattr_nested(model, name, value):
    parts = name.split('.')
    obj = model
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


class WanM2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
        finetune_checkpoint_dir=None,
        load_low_noise_only=False,
        load_high_noise_only=False,
        text_encoder=None,
        vae=None,
        split_identity_attn=False,
        split_learnable_query=False,
        global_query_num=0,
        selected_local_num = 6,
        use_both_query=False,
    ):
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu
        self.split_identity_attn = split_identity_attn
        self.split_learnable_query = split_learnable_query
        self.global_query_num = global_query_num
        self.selected_local_num = selected_local_num
        self.use_both_query = use_both_query
        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.param_dtype = config.param_dtype
        print(self.param_dtype, "!!!!!!")

        if t5_fsdp or dit_fsdp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        if text_encoder is None:
            shard_fn = partial(shard_model, device_id=device_id)
            self.text_encoder = T5EncoderModel(
                text_len=config.text_len,
                dtype=config.t5_dtype,
                device=torch.device('cpu'),
                checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
                tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
                shard_fn=shard_fn if t5_fsdp else None,
            )
        else:
            self.text_encoder = text_encoder

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        if vae is None:
            self.vae = Wan2_1_VAE(
                vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
                device=self.device)
        else:
            self.vae = vae

        logging.info(f"Creating WanM2V from {checkpoint_dir}")

        # 从 finetune ckpt 中读取 num_keyframes，优先级：ckpt.config > model config.json > 默认值10
        def _read_num_keyframes_from_ckpt(ckpt_path, default=10):
            if ckpt_path is None or not os.path.exists(ckpt_path):
                return default
            try:
                if ckpt_path.endswith('.safetensors'):
                    sd = load_file(ckpt_path, device="cpu")
                    for k, v in sd.items():
                        if 'keyframe_query.queries' in k and 'global' not in k:
                            nkf = v.shape[0] + self.global_query_num
                            logging.info(f"[num_keyframes] inferred {nkf} from safetensors {ckpt_path} (queries shape {v.shape[0]} + global {self.global_query_num})")
                            return nkf
                    return default
                ck = torch.load(ckpt_path, map_location="cpu", mmap=True)
                nkf = ck.get('config', {}).get('num_keyframes', None)
                if nkf is not None:
                    logging.info(f"[num_keyframes] read {nkf} from ckpt {ckpt_path}")
                    return nkf
            except Exception as e:
                logging.warning(f"[num_keyframes] failed to read from {ckpt_path}: {e}")
            return default

        _lora_weight_low  = getattr(config.low_noise_lora,  'weight', None)
        _lora_weight_high = getattr(config.high_noise_lora, 'weight', None)
        _ckpt_num_keyframes_low  = _read_num_keyframes_from_ckpt(_lora_weight_low,  config.get('num_keyframes', 10))
        _ckpt_num_keyframes_high = _read_num_keyframes_from_ckpt(_lora_weight_high, config.get('num_keyframes', 10))

        if load_high_noise_only:
            logging.info("Skipping low_noise_model loading (load_high_noise_only=True)")
            self.low_noise_model = None
        else:
            base_state_dict, pretrained_in_dim = _load_base_state_dict(
                checkpoint_dir, config.low_noise_checkpoint)

            self.low_noise_model = _create_model_on_meta(
                config, pretrained_in_dim, num_keyframes=_ckpt_num_keyframes_low,
                split_identity_attn=self.split_identity_attn,
                split_learnable_query=self.split_learnable_query,
                global_query_num=self.global_query_num,
                selected_local_num=self.selected_local_num)

            missing, unexpected = self.low_noise_model.load_state_dict(base_state_dict, strict=False, assign=True)
            _materialize_meta_params(self.low_noise_model)
            logging.info(f"low_noise base weights loaded. missing={len(missing)} unexpected={len(unexpected)}")
            kfq_missing = [k for k in missing if 'keyframe_query' not in k]
            if kfq_missing:
                logging.warning(f"low_noise: non-keyframe_query missing keys (first 5): {kfq_missing[:5]}")
            del base_state_dict
            gc.collect()

            if finetune_checkpoint_dir is not None:
                finetune_low_noise_model_path = os.path.join(finetune_checkpoint_dir, "backbone_low_noise.pth")
                if os.path.exists(finetune_low_noise_model_path):
                    logger.info(f"Loading finetune model from {finetune_low_noise_model_path}")
                    state_dict = torch.load(finetune_low_noise_model_path, map_location="cpu", mmap=True)
                    self.low_noise_model.load_state_dict(state_dict, strict=False)
            
            # assert False, f"{config}"
            # if "low_noise_lora" in config and config.low_noise_lora.enabled:
            self.low_noise_model = self._load_lora(
                self.low_noise_model, config.low_noise_lora)
            self.low_noise_model = self._configure_model(
                model=self.low_noise_model,
                use_sp=use_sp,
                dit_fsdp=dit_fsdp,
                shard_fn=shard_fn,
                convert_model_dtype=convert_model_dtype)

        if load_low_noise_only:
            logging.info("Skipping high_noise_model loading (load_low_noise_only=True)")
            self.high_noise_model = None
        else:
            base_state_dict, pretrained_in_dim = _load_base_state_dict(
                checkpoint_dir, config.high_noise_checkpoint)

            self.high_noise_model = _create_model_on_meta(
                config, pretrained_in_dim, num_keyframes=_ckpt_num_keyframes_high,
                split_identity_attn=self.split_identity_attn,
                split_learnable_query=self.split_learnable_query,
                global_query_num=self.global_query_num,
                selected_local_num = self.selected_local_num)

            missing, unexpected = self.high_noise_model.load_state_dict(base_state_dict, strict=False, assign=True)
            _materialize_meta_params(self.high_noise_model)
            logging.info(f"high_noise base weights loaded. missing={len(missing)} unexpected={len(unexpected)}")
            kfq_missing = [k for k in missing if 'keyframe_query' not in k]
            if kfq_missing:
                logging.warning(f"high_noise: non-keyframe_query missing keys (first 5): {kfq_missing[:5]}")
            del base_state_dict
            gc.collect()

            if finetune_checkpoint_dir is not None:
                finetune_high_noise_model_path = os.path.join(finetune_checkpoint_dir, "backbone_high_noise.pth")
                if os.path.exists(finetune_high_noise_model_path):
                    logger.info(f"Loading finetune model from {finetune_high_noise_model_path}")
                    state_dict = torch.load(finetune_high_noise_model_path, map_location="cpu", mmap=True)
                    self.high_noise_model.load_state_dict(state_dict, strict=False)

            # if "high_noise_lora" in config and config.high_noise_lora.enabled:
            self.high_noise_model = self._load_lora(
                self.high_noise_model, config.high_noise_lora)
            self.high_noise_model = self._configure_model(
                model=self.high_noise_model,
                use_sp=use_sp,
                dit_fsdp=dit_fsdp,
                shard_fn=shard_fn,
                convert_model_dtype=convert_model_dtype)

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    @torch.no_grad()
    def filter_memory_by_query(self, memory_pool, prompt, max_memory_size, fix=0, offload_model=True,
                                time_indices=None, max_memory_frames=None,
                                model_override=None, current_absolute_time=None):
        M_total = memory_pool.shape[1]
        if M_total <= max_memory_size:
            return memory_pool, time_indices if time_indices is not None else [], list(range(M_total))

        model = model_override if model_override is not None else (
            self.low_noise_model if self.low_noise_model is not None else self.high_noise_model)
        base_model = model.base_model.model if hasattr(model, 'base_model') else model
        kfq = base_model.keyframe_query

        latents = memory_pool.unsqueeze(0).to(self.device)

        context = self._encode_text([prompt], self.device, offload_model=offload_model)
        print("!!!!!!!!!!!shot caption:", prompt)
        context_tensor = context[0].unsqueeze(0)
        context_lens = torch.tensor([context[0].shape[0]], device=self.device)

        center = 512
        lat_stride = self.vae_stride[0]  # 4
        if time_indices is not None and len(time_indices) == M_total:
            if current_absolute_time is not None:
                # 与训练公式对齐：训练用 abs_pixel//4 - current_pixel//4
                # time_indices 为相对偏移（abs - current），通过 current 重建绝对值后再按训练公式计算
                # 避免 (a-b)//k 与 a//k - b//k 因 Python floor 除法产生的 ±1 偏差
                target_lat = current_absolute_time // lat_stride
                t_idx = [[center + max(-512, min(511,
                          (current_absolute_time + t) // lat_stride - target_lat))
                          for t in time_indices]]
            else:
                # fallback：无法获取绝对时间时退回原始公式
                t_idx = [[center + max(-512, min(511, t // lat_stride)) for t in time_indices]]
        else:
            FRAMES_PER_SHOT = 10
            PIXEL_FRAMES_PER_SHOT = 81
            # fallback：用 latent 帧步长（而非像素步长）与训练对齐
            lat_frames_per_shot = (PIXEL_FRAMES_PER_SHOT - 1) // lat_stride + 1  # 21
            step = max(1, lat_frames_per_shot // FRAMES_PER_SHOT)  # 2
            t_idx = [[center - (M_total - i) * step for i in range(M_total)]]
        print(f"[FilterQuery] M_total={M_total}, max_memory_size={max_memory_size}, t_idx={t_idx}")

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from contextlib import nullcontext

        if isinstance(model, FSDP):
            model.to(self.device)
            ctx = FSDP.summon_full_params(model, offload_to_cpu=True, writeback=False)
        else:
            ctx = nullcontext()

        with ctx:
            import copy
            kfq_gpu = copy.deepcopy(kfq).to(self.device)
            text_proj_gpu = copy.deepcopy(base_model.text_embedding).to(self.device)
            freqs_gpu = base_model.freqs.to(self.device)

        latents_infer = latents.to(self.device)
        # 与训练对齐：零填充 context 到 text_len 再过 text_embedding 投影
        text_len = base_model.text_len
        context_tensor_padded = torch.cat([
            context_tensor,
            context_tensor.new_zeros(1, text_len - context_tensor.size(1), context_tensor.size(2))
        ], dim=1).to(self.device)
        context_lens_infer = None

        kfq_gpu.eval()
        with torch.amp.autocast('cuda', dtype=self.param_dtype):
            context_projected = text_proj_gpu(context_tensor_padded)

            # split_learnable_query 模式下 global query 需要 context_global；
            # filter 阶段只有一个 prompt，global/local 复用同一段 context。
            kfq_kwargs = dict(
                freqs=freqs_gpu,
                context=context_projected,
                context_lens=context_lens_infer,
                time_indices=t_idx,
            )
            if self.split_learnable_query and self.global_query_num > 0:
                if "global caption:" in prompt and "shot caption:" in prompt:
                    global_caption = prompt.split("shot caption:")[0].replace("global caption:", "").strip().rstrip(";").strip()
                else:
                    global_caption = prompt
                print("!!!!!!!!!!!global caption:", global_caption)
                context_global = self._encode_text([global_caption], self.device, offload_model=offload_model)
                context_global_tensor = context_global[0].unsqueeze(0)
                context_global_padded = torch.cat([
                    context_global_tensor,
                    context_global_tensor.new_zeros(1, text_len - context_global_tensor.size(1), context_global_tensor.size(2))
                ], dim=1).to(self.device)
                context_global_projected = text_proj_gpu(context_global_padded)
                kfq_kwargs["context_global"] = context_global_projected
                kfq_kwargs["context_lens_global"] = None

            attn_weights_list = kfq_gpu(latents_infer, **kfq_kwargs)
            print("!!!!!!!!!!!attn_weights_list:", attn_weights_list)

        del kfq_gpu, text_proj_gpu, freqs_gpu
        torch.cuda.empty_cache()
        
        attn_weights = attn_weights_list[0]  # [K, F]  (eval 模式下为 one-hot argmax)

        # ---- split_learnable_query 模式下 local/global query 分离选帧 ----
        # attn_weights 行布局（与 _forward_single_item 中 torch.cat([queries_i, queries_global]) 一致）：
        #   行 0 ~ (K - global_query_num - 1)：local  queries
        #   行 (K - global_query_num) ~ K-1  ：global queries
        #
        # local  部分：从 (K - global_query_num) 个 local  query 里取 top max_memory_frames 行，
        #              各自 argmax 选帧，最终去重取 max_memory_size 帧
        # global 部分：用全部 global_query_num 个 global query，各自 argmax 选帧（不受 max_memory_frames 限制）
        #              选出的帧追加到 local 结果后面（供可视化区分；不计入 max_memory_size 配额）

        split_lq = self.split_learnable_query and self.global_query_num > 0
        if split_lq:
            n_global = self.global_query_num
            F_patches = attn_weights.shape[1]
            n_local_queries = min(self.selected_local_num, F_patches)   # 与内部 k_to_select 对齐
            aw_local  = attn_weights[:n_local_queries]
            aw_global = attn_weights[-n_global:]
        else:
            aw_local  = attn_weights                             # [K, F]
            aw_global = None

        # ---- local query 选帧：selected_local_num 帧，其中 fix 帧固定 ----
        aw = aw_local

        # 每个 local query 独立选一帧
        per_query_frames = aw.argmax(dim=-1).tolist()
        selected_set = sorted(set(per_query_frames))

        # fix: 前 fix 帧固定占用 local 配额
        actual_fix = min(fix, M_total)
        local_budget = self.selected_local_num
        if actual_fix > 0:
            fixed_indices = list(range(actual_fix))
            remaining = sorted([idx for idx in selected_set if idx >= actual_fix])
            query_budget = max(0, local_budget - actual_fix)
            selected_set = fixed_indices + remaining[:query_budget]
            logging.info(f"[FilterQuery] fix={actual_fix}: locked {fixed_indices}, "
                         f"query-selected {remaining[:query_budget]}")
        else:
            selected_set = selected_set[:local_budget]

        # Force-fill: 如果去重后帧数不足 local_budget，按 attention score 排名补充
        if len(selected_set) < local_budget:
            frame_scores = aw.max(dim=0).values  # [F] 每帧的最大 attention score
            already_selected = set(selected_set)
            candidate_indices = [i for i in range(M_total) if i not in already_selected]
            candidate_scores = [(i, frame_scores[i].item()) for i in candidate_indices]
            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            fill_count = local_budget - len(selected_set)
            fill_indices = [idx for idx, _ in candidate_scores[:fill_count]]
            selected_set = sorted(selected_set + fill_indices)
            logging.info(f"[FilterQuery] Force-fill: added {fill_indices} to reach local_budget={local_budget}")

        # ---- global query 选帧：排除 local 已选帧，保序去重，与训练侧 global_scores_masked 逻辑一致 ----
        global_frames_to_append = []
        if aw_global is not None:
            already_local = set(selected_set)
            per_global_frames = aw_global.argmax(dim=-1).tolist()
            global_deduped = list(dict.fromkeys(f for f in per_global_frames if f not in already_local))
            global_frames_to_append = global_deduped[:self.global_query_num]
            logging.info(f"[FilterQuery] Global query: per_global_frames={per_global_frames}, "
                         f"global_frames_to_append={global_frames_to_append}")

        # local 帧在前（升序），global 帧固定追加在后
        selected_indices = sorted(selected_set) + global_frames_to_append
        logging.info(f"[FilterQuery] local={sorted(selected_set)}, global={global_frames_to_append}, "
                     f"total={len(selected_indices)}")

        # 跨 rank 同步选帧结果，防止浮点非确定性导致各 rank 不一致
        if dist.is_initialized():
            n_selected = torch.tensor([len(selected_indices)], dtype=torch.long, device=self.device)
            dist.broadcast(n_selected, src=0)
            if self.rank == 0:
                selected_indices_tensor = torch.tensor(selected_indices, dtype=torch.long, device=self.device)
            else:
                selected_indices_tensor = torch.zeros(n_selected.item(), dtype=torch.long, device=self.device)
            dist.broadcast(selected_indices_tensor, src=0)
            selected_indices = selected_indices_tensor.tolist()

        logging.info(f"KeyframeQuery 智能筛选: 从 {M_total} 帧中选中了 {selected_indices}, "
                     f"per_query_frames={per_query_frames}")

        filtered_pool = memory_pool[:, selected_indices, :, :]
        filtered_time_indices = [time_indices[i] for i in selected_indices] if time_indices is not None else []
        return filtered_pool.cpu(), filtered_time_indices, selected_indices

    def _load_lora(self, model, lora_config):
        lora_cfg = copy.deepcopy(lora_config)
        lora_cfg.pop("enabled")
        lora_weight = lora_cfg.pop("weight", None)

        lora_cfg = LoraConfig(**lora_cfg)
        model = get_peft_model(model, lora_cfg)

        if lora_weight is not None:
            logger.info(f"Loading trainable weights (LoRA + Query) from {lora_weight}")
            ext = os.path.splitext(lora_weight)[1]
            if ext == ".safetensors":
                state_dict = load_file(lora_weight, device="cpu")
            else:
                checkpoint = torch.load(lora_weight, map_location="cpu", mmap=True)
                state_dict = checkpoint.get('trainable_state_dict',
                             checkpoint.get('lora_state_dict', checkpoint))

            # 注释@longbin debug用
            stripped = {}
            for k, v in state_dict.items():
                # 剥离训练时的 "low_noise_model." / "high_noise_model." 前缀
                for prefix in ("low_noise_model.", "high_noise_model."):
                    if k.startswith(prefix):
                        k = k[len(prefix):]
                        break
                stripped[k] = v
            state_dict = stripped

            # 适配 Wan2.2-MI2V-A14B 等 checkpoint：
            # checkpoint key 格式为 lora_A.weight / lora_B.weight（无 .default），
            # 而 peft get_peft_model 生成的 key 为 lora_A.default.weight / lora_B.default.weight，
            # 需要补上 .default 层才能匹配。
            adapted = {}
            for k, v in state_dict.items():
                new_k = re.sub(r'\.lora_([AB])\.weight$', r'.lora_\1.default.weight', k)
                adapted[new_k] = v
            state_dict = adapted

            # 过滤掉形状不匹配的 key（如 keyframe_query.queries 因 global_query_num 变化导致维度不同），
            # 让模型保留当前初始化值，避免 RuntimeError（strict=False 不处理形状不匹配）
            filtered_state_dict = {}
            model_state = model.state_dict()
            for k, v in state_dict.items():
                if k in model_state and model_state[k].shape != v.shape:
                    logger.warning(f"_load_lora: skipping '{k}' due to shape mismatch "
                                   f"(ckpt {list(v.shape)} vs model {list(model_state[k].shape)})")
                    continue
                filtered_state_dict[k] = v
            state_dict = filtered_state_dict

            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            # 只统计 LoRA / keyframe_query 相关的 miss（忽略 backbone frozen 参数的 miss 是正常的）
            def _is_trainable_key(k):
                return 'lora_A' in k or 'lora_B' in k or 'keyframe_query' in k

            trainable_missing    = [k for k in missing_keys    if _is_trainable_key(k)]
            trainable_unexpected = [k for k in unexpected_keys if _is_trainable_key(k)]
            lora_missing    = [k for k in missing_keys    if 'lora_A' in k or 'lora_B' in k]
            kfq_missing     = [k for k in missing_keys    if 'keyframe_query' in k]
            kfq_unexpected  = [k for k in unexpected_keys if 'keyframe_query' in k]

            logger.info(f"LoRA + KeyframeQuery injection summary for {lora_weight}:")
            logger.info(f"  - Total keys in loaded state_dict: {len(state_dict)}")
            logger.info(f"  - LoRA missing: {len(lora_missing)}  |  KeyframeQuery missing: {len(kfq_missing)}")
            logger.info(f"  - Trainable unexpected keys: {len(trainable_unexpected)}  (total unexpected: {len(unexpected_keys)})")
            if trainable_unexpected:
                logger.warning(f"  - First 5 trainable unexpected: {trainable_unexpected[:5]}")
            if trainable_missing:
                logger.warning(f"  - First 5 trainable missing: {trainable_missing[:5]}")

            lora_params_found = 0
            kfq_params_found  = 0
            for name, param in model.named_parameters():
                if 'lora_A' in name or 'lora_B' in name:
                    lora_params_found += 1
                if 'keyframe_query' in name:
                    kfq_params_found += 1
            logger.info(f"  - LoRA tensors in model: {lora_params_found}  |  KeyframeQuery tensors: {kfq_params_found}")
        return model

    @torch.no_grad()
    def encode_memory_from_files(self, memory_files, target_h=None, target_w=None):
        imgs = []
        for filename in memory_files:
            if filename.endswith(".mp4"):
                container = av.open(filename)
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                frames = []
                for frame in container.decode(stream):
                    img = frame.to_image().convert("RGB")
                    frames.append(img)
                total = len(frames)
                if total == 0:
                    raise RuntimeError(f"No video frames decoded from {filename}")
                idxs = [0, total - 1]
                mids = random.sample(list(range(1, total - 1)), min(2, total - 2))
                idxs.extend(mids)
                idxs = sorted(set(idxs))
                for i in idxs:
                    img = TF.to_tensor(frames[i]).sub_(0.5).div_(0.5)
                    imgs.append(img)
                container.close()
            else:
                img = Image.open(filename).convert("RGB")
                img = TF.to_tensor(img).sub_(0.5).div_(0.5)
                imgs.append(img)

        if target_h is not None and target_w is not None:
            imgs = [torch.nn.functional.interpolate(
                img[None], size=(target_h, target_w), mode='bicubic').squeeze(0) for img in imgs]

        imgs_t = torch.stack(imgs, dim=0).to(self.device)
        memory_pool = self.vae.encode(imgs_t.unsqueeze(2)).float().squeeze(2)
        memory_pool = memory_pool.permute(1, 0, 2, 3)
        return memory_pool

    # MAX_MEMORY_FRAMES = 10

    # @torch.no_grad()
    # def update_memory_pool_from_file(self, memory_pool, video_path, target_h=None, target_w=None, num_sample_frames=-1):
    #     if not os.path.exists(video_path):
    #         logging.warning(f"update_memory_pool_from_file: file not found: {video_path}")
    #         return memory_pool, 0, []

    #     container = av.open(video_path)
    #     stream = container.streams.video[0]
    #     stream.thread_type = "AUTO"
    #     all_frames = []
    #     for frame in container.decode(stream):
    #         all_frames.append(frame.to_image().convert("RGB"))
    #     container.close()

    #     total = len(all_frames)
    #     if total == 0:
    #         logging.warning(f"update_memory_pool_from_file: no frames decoded from {video_path}")
    #         return memory_pool, 0, []

    #     if num_sample_frames <= 0 or total <= num_sample_frames:
    #         idxs = list(range(total))
    #     else:
    #         idxs = [round(i * (total - 1) / (num_sample_frames - 1)) for i in range(num_sample_frames)]

    #     imgs = []
    #     for idx in idxs:
    #         img = TF.to_tensor(all_frames[idx]).sub_(0.5).div_(0.5)
    #         if target_h is not None and target_w is not None:
    #             img = torch.nn.functional.interpolate(img[None].cpu(), size=(target_h, target_w), mode='bicubic').squeeze(0)
    #         imgs.append(img)

    #     img_tensor = torch.stack(imgs, dim=0)
        
    #     chunk_size = 8
    #     new_tokens_list = []
        
    #     for i in range(0, len(img_tensor), chunk_size):
    #         chunk = img_tensor[i:i+chunk_size].to(self.device)
    #         tokens = self.vae.encode(chunk.unsqueeze(2)).float().squeeze(2).cpu()
    #         new_tokens_list.append(tokens)
    #         del chunk
    #     torch.cuda.empty_cache()

    #     new_tokens = torch.cat(new_tokens_list, dim=0)
    #     new_tokens = new_tokens.permute(1, 0, 2, 3)

    #     if memory_pool is None:
    #         return new_tokens, total, idxs

    #     old_pool = memory_pool.cpu()
    #     lat_H, lat_W = new_tokens.shape[2], new_tokens.shape[3]
    #     if old_pool.shape[2] != lat_H or old_pool.shape[3] != lat_W:
    #         # 尺寸不一致说明分辨率配置有误，报错而非静默插值（插值会污染 latent）
    #         raise ValueError(
    #             f"update_memory_pool_from_file: old pool spatial size "
    #             f"({old_pool.shape[2]}x{old_pool.shape[3]}) != "
    #             f"new tokens ({lat_H}x{lat_W}). "
    #             f"Check target_h/target_w consistency across shots."
    #         )
    #     return torch.cat([old_pool, new_tokens], dim=1), total, idxs
    @torch.no_grad()
    def update_memory_pool_from_file(self, memory_pool, video_path, target_h=None, target_w=None, num_sample_frames=-1):
        if not os.path.exists(video_path):
            logging.warning(f"update_memory_pool_from_file: file not found: {video_path}")
            return memory_pool, 0, []

        if video_path.endswith('.pt'):
            # === 无损 Tensor 读取分支 ===
            # 读取出的 Tensor 形状为 [3, F, H, W]，数值范围已经是 [-1, 1] FP32
            video_tensor = torch.load(video_path, map_location='cpu')
            total = video_tensor.shape[1]
            
            if num_sample_frames <= 0 or total <= num_sample_frames:
                idxs = list(range(total))
            else:
                idxs = [round(i * (total - 1) / (num_sample_frames - 1)) for i in range(num_sample_frames)]
                
            imgs = []
            for idx in idxs:
                img = video_tensor[:, idx, :, :]
                if target_h is not None and target_w is not None:
                    # 使用插值对齐尺寸
                    img = torch.nn.functional.interpolate(img[None], size=(target_h, target_w), mode='bicubic').squeeze(0)
                imgs.append(img)
            img_tensor = torch.stack(imgs, dim=0)

        else:
            # === 原本的 MP4 读取分支 ===
            container = av.open(video_path)
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            all_frames = []
            for frame in container.decode(stream):
                all_frames.append(frame.to_image().convert("RGB"))
            container.close()

            total = len(all_frames)
            if total == 0:
                logging.warning(f"update_memory_pool_from_file: no frames decoded from {video_path}")
                return memory_pool, 0, []

            if num_sample_frames <= 0 or total <= num_sample_frames:
                idxs = list(range(total))
            else:
                idxs = [round(i * (total - 1) / (num_sample_frames - 1)) for i in range(num_sample_frames)]

            imgs = []
            for idx in idxs:
                img = TF.to_tensor(all_frames[idx]).sub_(0.5).div_(0.5)
                if target_h is not None and target_w is not None:
                    img = torch.nn.functional.interpolate(img[None].cpu(), size=(target_h, target_w), mode='bicubic').squeeze(0)
                imgs.append(img)
            img_tensor = torch.stack(imgs, dim=0)

        # === 以下是统一的 VAE Encode 流程，无需修改 ===
        chunk_size = 8
        new_tokens_list = []
        
        for i in range(0, len(img_tensor), chunk_size):
            chunk = img_tensor[i:i+chunk_size].to(self.device)
            tokens = self.vae.encode(chunk.unsqueeze(2)).float().squeeze(2).cpu()
            new_tokens_list.append(tokens)
            del chunk
        torch.cuda.empty_cache()

        new_tokens = torch.cat(new_tokens_list, dim=0)
        new_tokens = new_tokens.permute(1, 0, 2, 3)

        if memory_pool is None:
            return new_tokens, total, idxs

        old_pool = memory_pool.cpu()
        lat_H, lat_W = new_tokens.shape[2], new_tokens.shape[3]
        if old_pool.shape[2] != lat_H or old_pool.shape[3] != lat_W:
            raise ValueError(
                f"update_memory_pool_from_file: old pool spatial size "
                f"({old_pool.shape[2]}x{old_pool.shape[3]}) != "
                f"new tokens ({lat_H}x{lat_W}). "
            )
        return torch.cat([old_pool, new_tokens], dim=1), total, idxs
    
    @torch.no_grad()
    def update_memory_pool(self, memory_pool, prev_shot_video):
        if prev_shot_video is None:
            return memory_pool

        C, F, H, W = prev_shot_video.shape
        all_frames = prev_shot_video.permute(1, 0, 2, 3).to(self.device)
        new_tokens = self.vae.encode(all_frames.unsqueeze(2)).float().squeeze(2)
        new_tokens = new_tokens.permute(1, 0, 2, 3)

        if memory_pool is None:
            pool = new_tokens
        else:
            pool = torch.cat([memory_pool.to(self.device), new_tokens], dim=1)

        return pool

    def _encode_text(self, prompts, device, offload_model=False):
        if not self.t5_cpu:
            self.text_encoder.model.to(device)
            context = self.text_encoder(prompts, device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder(prompts, torch.device('cpu'))
            context = [t.to(device) for t in context]
        return context

    def _compute_seq_len(self, lat_f, lat_h, lat_w, num_keyframes):
        seq_len = (lat_f + num_keyframes) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        return int(math.ceil(seq_len / self.sp_size)) * self.sp_size

    @staticmethod
    def _is_fsdp(model):
        return model is not None and isinstance(model, FSDP)

    def _prepare_model_for_timestep(self, t, boundary, offload_model):
        if t.item() >= boundary:
            model = self.high_noise_model
            if offload_model and self.low_noise_model is not None and not self._is_fsdp(self.low_noise_model):
                self.low_noise_model.cpu()
            if model is not None and not self._is_fsdp(model):
                model.to(self.device)
        else:
            model = self.low_noise_model
            if offload_model and self.high_noise_model is not None and not self._is_fsdp(self.high_noise_model):
                self.high_noise_model.cpu()
            if model is not None and not self._is_fsdp(model):
                model.to(self.device)

        if model is None:
            model = self.low_noise_model if self.high_noise_model is None else self.high_noise_model
            if model is not None and not self._is_fsdp(model):
                model.to(self.device)

        return model

    @torch.no_grad()
    def generate(
        self,
        input_prompt,
        memory_dir=None,
        prev_shot_video_path=None,
        first_frame_file=None,
        motion_frames_file=None,
        max_area=720 * 1280,
        frame_num=81,
        shift=5.0,
        sample_solver='unipc',
        sampling_steps=40,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        offload_model=True,
        max_memory_size=8,
        fix=0,
        num_sample_frames=10,
        storymem_mode=False,
        reconstruct_caption=None,
        idt_back_mode=False,
        single_pool_query='low',  # 'low' | 'high'：单池模式下用哪个模型的 query 选帧
    ):
        assert first_frame_file is None or motion_frames_file is None
        guide_scale = (guide_scale, guide_scale) if isinstance(guide_scale, float) else guide_scale

        F = frame_num
        FORCE_TRAIN_RESOLUTION = True

        if FORCE_TRAIN_RESOLUTION:
            h = 480
            w = 832
            lat_h = h // self.vae_stride[1]
            lat_w = w // self.vae_stride[2]
            logging.info(f"[FORCED] Using training resolution: {h}x{w} (latent: {lat_h}x{lat_w})")

            if first_frame_file is not None:
                first_frame = Image.open(first_frame_file).convert("RGB")
                first_frame = first_frame.resize((w, h), Image.BICUBIC)
                first_frame = TF.to_tensor(first_frame).sub_(0.5).div_(0.5).to(self.device)
                logging.info(f"[FORCED] Resized first_frame to {w}x{h}")
            elif motion_frames_file is not None:
                from decord import VideoReader, cpu as dcpu
                vr = VideoReader(motion_frames_file, ctx=dcpu())
                frames = []
                for i in range(min(5, len(vr))):
                    frame = Image.fromarray(vr[i].asnumpy()).convert("RGB")
                    frame = frame.resize((w, h), Image.BICUBIC)
                    frames.append(TF.to_tensor(frame).sub_(0.5).div_(0.5).to(self.device))
                motion_frames = torch.stack(frames, dim=0)
                logging.info(f"[FORCED] Resized motion_frames to {w}x{h}")
        else:
            if first_frame_file is not None:
                first_frame = Image.open(first_frame_file).convert("RGB")
                first_frame = TF.to_tensor(first_frame).sub_(0.5).div_(0.5).to(self.device)
                h_img, w_img = first_frame.shape[1:]
                aspect_ratio = h_img / w_img
            elif motion_frames_file is not None:
                from decord import VideoReader, cpu as dcpu
                vr = VideoReader(motion_frames_file, ctx=dcpu())
                frames = [TF.to_tensor(Image.fromarray(vr[i].asnumpy()).convert("RGB")).sub_(0.5).div_(0.5).to(self.device)
                          for i in range(min(5, len(vr)))]
                motion_frames = torch.stack(frames, dim=0)
                h_img, w_img = motion_frames.shape[-2:]
                aspect_ratio = h_img / w_img
            else:
                aspect_ratio = 9 / 16

            lat_h = round(np.sqrt(max_area * aspect_ratio)) // self.vae_stride[1] // self.patch_size[1] * self.patch_size[1]
            lat_w = round(np.sqrt(max_area / aspect_ratio)) // self.vae_stride[2] // self.patch_size[2] * self.patch_size[2]
            h = lat_h * self.vae_stride[1]
            w = lat_w * self.vae_stride[2]

        lat_f = (F - 1) // self.vae_stride[0] + 1

        memory_pool = None
        mem_time_indices_absolute = []
        current_absolute_time = 0

        # --- dual-query 状态变量，后续构造 y 时使用 ---
        memory_pool_low = None
        memory_pool_high = None
        mem_time_indices_absolute_low = None
        mem_time_indices_absolute_high = None
        use_dual_query = (self.use_both_query
                          and self.low_noise_model is not None
                          and self.high_noise_model is not None)

        # dual-query 独立 pool：各模型维护自己的 pool，下一 shot 从各自 pool 继续累积
        _loaded_pool_low = None
        _loaded_pool_high = None
        _loaded_indices_low = None
        _loaded_indices_high = None

        pool_path = os.path.join(memory_dir, "memory_pool.pt") if memory_dir is not None else None

        if pool_path is not None and os.path.exists(pool_path):
            ckpt = torch.load(pool_path, map_location="cpu")
            if isinstance(ckpt, dict):
                if ckpt.get("use_both_query", False) and use_dual_query:
                    _loaded_pool_low = ckpt["pool_raw_low"]
                    _loaded_pool_high = ckpt["pool_raw_high"]
                    _loaded_indices_low = ckpt.get("absolute_indices_raw_low", [])
                    _loaded_indices_high = ckpt.get("absolute_indices_raw_high", [])
                    current_absolute_time = ckpt.get("current_time", 0)
                    memory_pool = _loaded_pool_low
                    mem_time_indices_absolute = _loaded_indices_low
                    logging.info(f"[use_both_query] Loaded independent raw pools: "
                                 f"low={list(_loaded_pool_low.shape)}, high={list(_loaded_pool_high.shape)}")
                else:
                    memory_pool = ckpt["pool"]
                    mem_time_indices_absolute = ckpt.get("absolute_indices", [])
                    current_absolute_time = ckpt.get("current_time", 0)
                    if not mem_time_indices_absolute and "time_indices" in ckpt:
                        old_rel = ckpt["time_indices"]
                        mem_time_indices_absolute = [current_absolute_time + r for r in old_rel]
                        logging.info(f"[Compat] Restored absolute_indices from legacy time_indices field: {mem_time_indices_absolute}")
            else:
                memory_pool = ckpt
                mem_time_indices_absolute = []
            logging.info(f"Loaded memory pool from {pool_path}, shape={list(memory_pool.shape)}, current_time={current_absolute_time}")

        if prev_shot_video_path is not None:
            pool_sample_size = num_sample_frames

            # 各模型独立 raw pool = 各自上一 shot 筛选帧 + 本 shot 新采样帧
            if use_dual_query and _loaded_pool_low is not None and _loaded_pool_high is not None:
                # LOW pool
                memory_pool_low_acc, prev_total_frames, new_idxs = self.update_memory_pool_from_file(
                    _loaded_pool_low, prev_shot_video_path,
                    target_h=h, target_w=w,
                    num_sample_frames=pool_sample_size,
                )
                new_absolute_indices = [current_absolute_time + idx for idx in new_idxs]
                mem_indices_low_acc = _loaded_indices_low + new_absolute_indices
                current_absolute_time += prev_total_frames

                # HIGH pool
                memory_pool_high_acc, _, _ = self.update_memory_pool_from_file(
                    _loaded_pool_high, prev_shot_video_path,
                    target_h=h, target_w=w,
                    num_sample_frames=pool_sample_size,
                )
                mem_indices_high_acc = _loaded_indices_high + new_absolute_indices

                memory_pool = memory_pool_low_acc
                mem_time_indices_absolute = mem_indices_low_acc

                logging.info(f"[use_both_query] Independent raw pools: "
                             f"low={memory_pool_low_acc.shape[1]} frames, "
                             f"high={memory_pool_high_acc.shape[1]} frames, "
                             f"current_time={current_absolute_time}")
            else:
                # 共享 pool（首次或非 dual-query）
                memory_pool, prev_total_frames, new_idxs = self.update_memory_pool_from_file(
                    memory_pool, prev_shot_video_path,
                    target_h=h, target_w=w,
                    num_sample_frames=pool_sample_size,
                )
                new_absolute_indices = [current_absolute_time + idx for idx in new_idxs]
                mem_time_indices_absolute = mem_time_indices_absolute + new_absolute_indices
                current_absolute_time += prev_total_frames

                # dual-query 但首次（尚无独立 pool）：复制共享 pool 给两个模型
                if use_dual_query:
                    memory_pool_low_acc = memory_pool.clone()
                    memory_pool_high_acc = memory_pool.clone()
                    mem_indices_low_acc = list(mem_time_indices_absolute)
                    mem_indices_high_acc = list(mem_time_indices_absolute)

            logging.info(f"Updated absolute indices: {mem_time_indices_absolute}, current_time={current_absolute_time}")

            # effective_max = local 帧上限（不含 global），用于触发 filter 和限制 local 选帧数量
            effective_max = min(num_sample_frames, max_memory_size)

            # memory 合并后超过 effective_max 才做 query 裁剪
            if use_dual_query:
                # 分别对各自 pool 做 filter
                need_filter_low = memory_pool_low_acc is not None and memory_pool_low_acc.shape[1] > effective_max
                need_filter_high = memory_pool_high_acc is not None and memory_pool_high_acc.shape[1] > effective_max

                if need_filter_low or need_filter_high:
                    rel_indices_low = [t - current_absolute_time for t in mem_indices_low_acc]
                    rel_indices_high = [t - current_absolute_time for t in mem_indices_high_acc]

                    if need_filter_low:
                        memory_pool_low, sel_rel_low, sel_pos_low = self.filter_memory_by_query(
                            memory_pool_low_acc,
                            prompt=input_prompt,
                            max_memory_size=effective_max,
                            fix=fix,
                            offload_model=offload_model,
                            time_indices=rel_indices_low,
                            max_memory_frames=num_sample_frames,
                            model_override=self.low_noise_model,
                            current_absolute_time=current_absolute_time,
                        )
                        mem_time_indices_absolute_low = [current_absolute_time + rel for rel in sel_rel_low]
                    else:
                        memory_pool_low = memory_pool_low_acc
                        mem_time_indices_absolute_low = mem_indices_low_acc
                        sel_pos_low = list(range(memory_pool_low_acc.shape[1]))

                    if need_filter_high:
                        memory_pool_high, sel_rel_high, sel_pos_high = self.filter_memory_by_query(
                            memory_pool_high_acc,
                            prompt=input_prompt,
                            max_memory_size=effective_max,
                            fix=fix,
                            offload_model=offload_model,
                            time_indices=rel_indices_high,
                            max_memory_frames=num_sample_frames,
                            model_override=self.high_noise_model,
                            current_absolute_time=current_absolute_time,
                        )
                        mem_time_indices_absolute_high = [current_absolute_time + rel for rel in sel_rel_high]
                    else:
                        memory_pool_high = memory_pool_high_acc
                        mem_time_indices_absolute_high = mem_indices_high_acc
                        sel_pos_high = list(range(memory_pool_high_acc.shape[1]))

                    # split_learnable_query 时 global 去重可能导致两池帧数不等，截到相同长度
                    if memory_pool_low.shape[1] != memory_pool_high.shape[1]:
                        M_common = min(memory_pool_low.shape[1], memory_pool_high.shape[1])
                        logging.warning(
                            f"[use_both_query] M mismatch (low={memory_pool_low.shape[1]}, "
                            f"high={memory_pool_high.shape[1]}), truncating to {M_common}")
                        memory_pool_low  = memory_pool_low[:, :M_common]
                        memory_pool_high = memory_pool_high[:, :M_common]
                        mem_time_indices_absolute_low  = mem_time_indices_absolute_low[:M_common]
                        mem_time_indices_absolute_high = mem_time_indices_absolute_high[:M_common]
                        sel_pos_low  = sel_pos_low[:M_common]
                        sel_pos_high = sel_pos_high[:M_common]

                    # 对外暴露 low 的结果
                    memory_pool = memory_pool_low
                    mem_time_indices_absolute = mem_time_indices_absolute_low

                    # Debug: 对比两组 query 的选帧差异
                    common_frames = set(sel_pos_low) & set(sel_pos_high)
                    low_only = sorted(set(sel_pos_low) - set(sel_pos_high))
                    high_only = sorted(set(sel_pos_high) - set(sel_pos_low))
                    logging.info(f"[use_both_query] === Dual Query Frame Selection Debug ===")
                    logging.info(f"[use_both_query] LOW  pool size before filter: {memory_pool_low_acc.shape[1]}, "
                                 f"HIGH pool size before filter: {memory_pool_high_acc.shape[1]}")
                    logging.info(f"[use_both_query] Selected per model: {memory_pool_low.shape[1]}")
                    logging.info(f"[use_both_query] LOW  noise selected positions: {sel_pos_low}")
                    logging.info(f"[use_both_query] HIGH noise selected positions: {sel_pos_high}")
                    logging.info(f"[use_both_query] Common frames: {sorted(common_frames)} ({len(common_frames)}/{len(sel_pos_low)})")
                    logging.info(f"[use_both_query] LOW-only frames: {low_only}")
                    logging.info(f"[use_both_query] HIGH-only frames: {high_only}")
                    logging.info(f"[use_both_query] LOW  absolute time indices: {mem_time_indices_absolute_low}")
                    logging.info(f"[use_both_query] HIGH absolute time indices: {mem_time_indices_absolute_high}")

                    logging.info(f"Memory pool trimmed via KeyframeQuery to {memory_pool.shape[1]} frames")
                else:
                    # 两个 pool 都未超限，不需要 filter
                    memory_pool_low = memory_pool_low_acc
                    memory_pool_high = memory_pool_high_acc
                    mem_time_indices_absolute_low = mem_indices_low_acc
                    mem_time_indices_absolute_high = mem_indices_high_acc
                    memory_pool = memory_pool_low
                    mem_time_indices_absolute = mem_time_indices_absolute_low
                    logging.info(f"[use_both_query] Both pools ({memory_pool_low.shape[1]} frames) "
                                 f"<= effective_max ({effective_max}), no query needed")

            elif memory_pool is not None and memory_pool.shape[1] > effective_max:
                mem_time_indices_relative = [t - current_absolute_time for t in mem_time_indices_absolute]
                _single_query_model = (
                    self.high_noise_model if single_pool_query == 'high' and self.high_noise_model is not None
                    else None  # None → filter_memory_by_query 内部默认用 low_noise_model
                )
                logging.info(f"[single_pool_query={single_pool_query}] using "
                             f"{'high_noise_model' if _single_query_model is not None else 'low_noise_model'} query")
                memory_pool, selected_relative_indices, _ = self.filter_memory_by_query(
                    memory_pool,
                    prompt=input_prompt,
                    max_memory_size=effective_max,
                    fix=fix,
                    offload_model=offload_model,
                    time_indices=mem_time_indices_relative,
                    max_memory_frames=num_sample_frames,
                    current_absolute_time=current_absolute_time,
                    model_override=_single_query_model,
                )
                mem_time_indices_absolute = [current_absolute_time + rel for rel in selected_relative_indices]
                logging.info(f"Memory pool trimmed via KeyframeQuery to {memory_pool.shape[1]} frames")
                logging.info(f"Selected absolute indices: {mem_time_indices_absolute}")
            else:
                logging.info(f"Memory pool ({memory_pool.shape[1] if memory_pool is not None else 0} frames) <= effective_max ({effective_max}), no query needed")

            # 保存 memory pool
            if pool_path is not None and memory_pool is not None:
                if self.rank == 0:
                    os.makedirs(memory_dir, exist_ok=True)
                    tmp_path = pool_path + ".tmp"
                    if use_dual_query and memory_pool_low is not None and memory_pool_high is not None:
                        torch.save({
                            "pool_raw_low":  memory_pool_low.cpu(),
                            "pool_raw_high": memory_pool_high.cpu(),
                            "absolute_indices_raw_low":  mem_time_indices_absolute_low,
                            "absolute_indices_raw_high": mem_time_indices_absolute_high,
                            "current_time": current_absolute_time,
                            "use_both_query": True,
                        }, tmp_path)
                        logging.info(f"Saved memory pool to {pool_path}, "
                                     f"low={memory_pool_low.shape[1]}, "
                                     f"high={memory_pool_high.shape[1]}")
                    else:
                        torch.save({
                            "pool": memory_pool.cpu(),
                            "absolute_indices": mem_time_indices_absolute,
                            "current_time": current_absolute_time
                        }, tmp_path)
                        logging.info(f"Saved memory pool to {pool_path}, M={memory_pool.shape[1]}")
                    os.replace(tmp_path, pool_path)
                if dist.is_initialized():
                    dist.barrier()   # 所有 rank 等 rank=0 写完再继续读

        if memory_pool is None:
            logging.warning("memory_pool is None, using zero placeholder (1 frame). "
                            "Provide memory_dir with pre-existing keyframes for better results.")
            memory_pool = torch.zeros(16, 1, lat_h, lat_w, dtype=torch.float32)

        memory_pool = memory_pool.to(self.device)
        M = memory_pool.shape[1]
        print("M:", M)

        # 按绝对时间升序排列 memory 帧，保证 RoPE indices 单调递增
        if len(mem_time_indices_absolute) == M and M > 1:
            sorted_order = sorted(range(M), key=lambda i: mem_time_indices_absolute[i])
            if sorted_order != list(range(M)):
                logging.info(f"[Memory] Reordering memory frames by absolute time: {sorted_order}")
                memory_pool = memory_pool[:, sorted_order, :, :]
                mem_time_indices_absolute = [mem_time_indices_absolute[i] for i in sorted_order]

        # dual-query: 同样处理 high noise pool
        if use_dual_query and memory_pool_high is not None:
            memory_pool_high = memory_pool_high.to(self.device)
            if len(mem_time_indices_absolute_high) == M and M > 1:
                sorted_order_high = sorted(range(M), key=lambda i: mem_time_indices_absolute_high[i])
                if sorted_order_high != list(range(M)):
                    logging.info(f"[Memory] Reordering HIGH memory frames by absolute time: {sorted_order_high}")
                    memory_pool_high = memory_pool_high[:, sorted_order_high, :, :]
                    mem_time_indices_absolute_high = [mem_time_indices_absolute_high[i] for i in sorted_order_high]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # 与训练对齐：始终使用 input_prompt 作为 context（训练中用的是 clip caption）
        context      = self._encode_text([input_prompt], self.device, offload_model=offload_model)
        context_null = self._encode_text([n_prompt],     self.device, offload_model=offload_model)

        # 计算 identity frame 数量
        use_subject_recon = reconstruct_caption is not None and len(reconstruct_caption) > 0
        M_identity = 1 if use_subject_recon else 0

        # T5 编码 identity frame caption（与训练 _encode_text(raw_captions) 对齐）
        identity_frames_caption = None
        identity_frames_caption_null = None
        if M_identity > 0:
            identity_frames_caption = self._encode_text(
                [reconstruct_caption] * M_identity, self.device, offload_model=offload_model)
            # CFG: 用空字符串作为 identity 的 unconditional caption，
            # 保留 cross-attention split 结构，只替换内容，使 CFG 能放大重建信号
            identity_frames_caption_null = self._encode_text(
                [""] * M_identity, self.device, offload_model=offload_model)
            logging.info(f"[SubjectRecon] Encoded identity_frames_caption: {len(identity_frames_caption)} items, "
                         f"shape={identity_frames_caption[0].shape}")

        target_start_time = current_absolute_time
        target_lat_start = target_start_time // self.vae_stride[0]

        # 绝对 latent 时间：memory 帧
        if storymem_mode:
            # storymem 模式：使用固定间隔 5 (latent 帧) 代替真实物理时间间隔
            STORYMEM_FIXED_STEP = 20
            mem_lat_abs = [target_lat_start - (M - i) * STORYMEM_FIXED_STEP for i in range(M)]
            logging.info(f"[StorymemMode] Using fixed latent step={STORYMEM_FIXED_STEP} for memory RoPE indices: {mem_lat_abs}")
        elif len(mem_time_indices_absolute) == M:
            mem_lat_abs = [t // self.vae_stride[0] for t in mem_time_indices_absolute]
            # 保证 memory 帧 latent index 严格小于 target_lat_start，
            # 防止最后一帧 memory 与第一帧 target 共享同一个 RoPE 时间位置（都是 512）
            # 典型场景：81帧视频，最后采样帧 pixel=80, 80//4=20, target_lat_start=81//4=20 → 碰撞
            clamped = False
            for j in range(len(mem_lat_abs)):
                if mem_lat_abs[j] >= target_lat_start:
                    mem_lat_abs[j] = target_lat_start - 1
                    clamped = True
            if clamped:
                logging.info(
                    f"[RoPE] Clamped memory latent indices to < target_lat_start ({target_lat_start}): "
                    f"{mem_lat_abs}"
                )
        else:
            FRAMES_PER_SHOT = 10
            lat_frames_per_shot = (81 - 1) // self.vae_stride[0] + 1
            mem_step = max(1, lat_frames_per_shot // FRAMES_PER_SHOT)
            mem_lat_abs = [target_lat_start - (M - i) * mem_step for i in range(M)]

        center = 512

        # Memory RoPE indices: 相对于 target_lat_start 的偏移，clamp 到 [-512, 511]
        mem_time_indices = [
            center + max(-512, min(511, t - target_lat_start))
            for t in mem_lat_abs
        ]

        # Target RoPE indices: 从 center 开始，与旧训练代码（lora 训练约定）一致
        tgt_time_indices = [center + min(511, i) for i in range(lat_f)]

        full_time_indices = [mem_time_indices + tgt_time_indices]

        # 主题重建：在 time_indices 最前面插入 identity frame 的 RoPE 索引（与训练一致，用 0）
        if M_identity > 0:
            # identity_time_indices = [0] * M_identity
            if idt_back_mode:
                # idt_back_mode: identity 紧跟 memory 末尾，使用与 memory 相同的固定间隔 5
                last_mem_idx = mem_time_indices[-1] if mem_time_indices else center
                identity_time_indices = [
                    max(center - 512, min(center + 511, last_mem_idx + (i + 1) * 5))
                    for i in range(M_identity)
                ]
                full_time_indices = [mem_time_indices + identity_time_indices + tgt_time_indices]
            else:
                identity_time_indices = [0] * M_identity
                full_time_indices = [identity_time_indices + full_time_indices[0]]
            logging.info(f"[SubjectRecon] identity frame time indices: {identity_time_indices}")

        # dual-query: 为 high noise 模型计算独立的 full_time_indices_high
        full_time_indices_high = None
        if use_dual_query and mem_time_indices_absolute_high is not None:
            if storymem_mode:
                mem_lat_abs_high = [target_lat_start - (M - i) * STORYMEM_FIXED_STEP for i in range(M)]
            elif len(mem_time_indices_absolute_high) == M:
                mem_lat_abs_high = [t // self.vae_stride[0] for t in mem_time_indices_absolute_high]
                for j in range(len(mem_lat_abs_high)):
                    if mem_lat_abs_high[j] >= target_lat_start:
                        mem_lat_abs_high[j] = target_lat_start - 1
            else:
                FRAMES_PER_SHOT = 10
                lat_frames_per_shot = (81 - 1) // self.vae_stride[0] + 1
                mem_step = max(1, lat_frames_per_shot // FRAMES_PER_SHOT)
                mem_lat_abs_high = [target_lat_start - (M - i) * mem_step for i in range(M)]

            mem_time_indices_high = [
                center + max(-512, min(511, t - target_lat_start))
                for t in mem_lat_abs_high
            ]
            full_time_indices_high = [mem_time_indices_high + tgt_time_indices]
            if M_identity > 0:
                if idt_back_mode:
                    last_mem_idx_high = mem_time_indices_high[-1] if mem_time_indices_high else center
                    idt_time_high = [
                        max(center - 512, min(center + 511, last_mem_idx_high + (i + 1) * 5))
                        for i in range(M_identity)
                    ]
                    full_time_indices_high = [mem_time_indices_high + idt_time_high + tgt_time_indices]
                else:
                    full_time_indices_high = [identity_time_indices + full_time_indices_high[0]]
            logging.info(f"[use_both_query] full_time_indices_high computed, len={len(full_time_indices_high[0])}")

        print(f"\n{'='*80}")
        print(f"[INFERENCE RoPE] Shot Generation (center={center}, target from 512+)")
        print(f"{'='*80}")
        print(f"Memory frames: M={M}")
        print(f"Target latent frames: lat_f={lat_f}")
        print(f"Absolute pixel indices (memory): {mem_time_indices_absolute}")
        print(f"mem_lat_abs (latent): {mem_lat_abs}")
        print(f"target_lat_start={target_lat_start}")
        print(f"Memory RoPE indices (center+offset): {mem_time_indices}")
        print(f"  → Memory offsets from center: {[t - center for t in mem_time_indices]}")
        print(f"  → Memory inter-frame gaps: {[mem_time_indices[i+1]-mem_time_indices[i] for i in range(len(mem_time_indices)-1)]}")
        print(f"Target RoPE indices: {tgt_time_indices[:5]}...{tgt_time_indices[-3:]}")
        print(f"Full RoPE sequence length: {len(full_time_indices[0])}")
        print(f"{'='*80}\n")

        F_pix = (lat_f - 1) * self.vae_stride[0] + 1  # 目标 pixel 帧数
        lat_t = M_identity + M + lat_f

        msk = torch.zeros(1, 4, M + lat_f, lat_h, lat_w, device=self.device, dtype=torch.float32)

        msk[:, :, :M] = 1.0

        if first_frame_file is not None:
            msk[:, :, M] = 1.0
        elif motion_frames_file is not None:
            n_mot_lat = min((5 - 1) // self.vae_stride[0] + 1, lat_f)
            msk[:, :, M : M + n_mot_lat] = 1.0

        # 主题重建：在 mask 最前面插入 identity frame 的 mask（值为 3.0，与训练一致）
        if M_identity > 0:
            # msk_identity = torch.ones(1, 4, M_identity, lat_h, lat_w, device=self.device, dtype=torch.float32) * 3.0
            msk_identity = torch.ones(1, 4, M_identity, lat_h, lat_w, device=self.device, dtype=torch.float32) * 0.0
            if idt_back_mode:
                msk = torch.cat([msk, msk_identity], dim=2)
            else:
                msk = torch.cat([msk_identity, msk], dim=2)
            del msk_identity
            logging.info(f"[SubjectRecon] Prepended {M_identity} identity mask slots (msk=0.0), total temporal: {msk.shape[2]}")

        msk = msk.squeeze(0) # [4, M_identity+M+lat_f, lat_h, lat_w]
        # ============================================================

        # C_lat = memory_pool.shape[0]
        # u = torch.zeros(C_lat, lat_f, lat_h, lat_w, device=self.device, dtype=torch.float32)

        # with torch.no_grad():
        #     if first_frame_file is not None:
        #         ref_pixel = torch.nn.functional.interpolate(
        #             first_frame[None].cpu(), size=(h, w), mode='bicubic'
        #         ).to(self.device)
        #         first_frame_lat = self.vae.encode(ref_pixel.unsqueeze(2)).float().squeeze(0)  # [C_lat, 1, lat_h, lat_w]
        #         u[:, 0:1] = first_frame_lat
        #         del ref_pixel, first_frame_lat
        #     elif motion_frames_file is not None:
        #         mot_pix = torch.nn.functional.interpolate(
        #             motion_frames.to(self.device), size=(h, w), mode='bicubic'
        #         ).transpose(0, 1).unsqueeze(0)  # [1, 3, 5, h, w]
        #         mot_lat = self.vae.encode(mot_pix).float().squeeze(0)  # [C_lat, n_mot_lat, lat_h, lat_w]
        #         u[:, :mot_lat.shape[1]] = mot_lat
        #         del mot_pix, mot_lat, motion_frames
        if first_frame_file is not None:
            ref_pixel = torch.nn.functional.interpolate(
                first_frame[None].cpu(), size=(h, w), mode='bicubic'
            ).to(self.device)
            zeros_rest = torch.zeros(3, F_pix - 1, h, w, device=self.device)
            u_pixel = torch.cat([
                ref_pixel.transpose(0, 1),
                zeros_rest
            ], dim=1).unsqueeze(0)
            del ref_pixel, zeros_rest
        elif motion_frames_file is not None:
            mot_pix = torch.nn.functional.interpolate(
                motion_frames.to(self.device), size=(h, w), mode='bicubic'
            ).transpose(0, 1)
            zeros_rest = torch.zeros(3, F_pix - 5, h, w, device=self.device)
            u_pixel = torch.cat([mot_pix, zeros_rest], dim=1).unsqueeze(0)
            del mot_pix, zeros_rest, motion_frames
        else:
            u_pixel = torch.zeros(1, 3, F_pix, h, w, device=self.device)

        u = self.vae.encode(u_pixel).float().squeeze(0)
        del u_pixel
        # 主题重建：构造 identity frame 的零 latent conditioning（与训练一致，t2v 任务不泄露图像）
        identity_cond = None
        if M_identity > 0:
            zero_pixel = torch.zeros(1, 3, 1, h, w, device=self.device)
            zero_lat = self.vae.encode(zero_pixel).float().squeeze(0).squeeze(1)  # [16, lat_h, lat_w]
            del zero_pixel
            identity_cond = zero_lat.unsqueeze(1).expand(-1, M_identity, -1, -1)  # [16, M_identity, lat_h, lat_w]
            del zero_lat
            if idt_back_mode:
                # idt_back_mode=True:  [mem | identity | tgt]
                latent_y = torch.cat([memory_pool.float(), identity_cond, u], dim=1)
            else:
                # idt_back_mode=False: [identity | mem | tgt]
                latent_y = torch.cat([identity_cond, memory_pool.float(), u], dim=1)
            print(f"[SubjectRecon] identity conditioning inserted, latent_y: {list(latent_y.shape)}")
        else:
            latent_y = torch.cat([memory_pool.float(), u], dim=1)  # [16, M+lat_f, lat_h, lat_w]

        y = torch.cat([msk, latent_y], dim=0)  # [20, M_identity+M+lat_f, lat_h, lat_w]
        del latent_y

        # dual-query: 构造 y_high（msk 相同，只有 memory latent 内容不同）
        y_high = None
        if use_dual_query and memory_pool_high is not None:
            if M_identity > 0:
                if idt_back_mode:
                    latent_y_high = torch.cat([memory_pool_high.float(), identity_cond, u], dim=1)
                else:
                    latent_y_high = torch.cat([identity_cond, memory_pool_high.float(), u], dim=1)
            else:
                latent_y_high = torch.cat([memory_pool_high.float(), u], dim=1)
            y_high = torch.cat([msk, latent_y_high], dim=0)
            del latent_y_high
            logging.info(f"[use_both_query] y_high constructed, shape={list(y_high.shape)}")

        del u, msk
        if identity_cond is not None:
            del identity_cond

        # =====================================================================
        # 可视化模型输入 y（SP 安全版本）
        # decode 在所有 rank 上执行（避免集合通信死锁），只有 rank=0 写文件
        # =====================================================================
        try:
            import torchvision
            y_latent = y[4:].detach().float()   # [16, M_identity+M+lat_f, lat_h, lat_w]
            y_mask   = y[:4].detach().float()   # [4,  M_identity+M+lat_f, lat_h, lat_w]

            with torch.no_grad():
                # latent_y 布局取决于 idt_back_mode：
                #   idt_back_mode=False: [identity | mem | tgt]
                #   idt_back_mode=True:  [mem | identity | tgt]
                vis_idt_offset = M if idt_back_mode else 0
                vis_mem_offset = 0 if idt_back_mode else M_identity

                # 1a-0. Identity 帧：逐帧单独 decode
                id_frames_list = []
                for ii in range(M_identity):
                    frm_lat = y_latent[:, vis_idt_offset + ii:vis_idt_offset + ii + 1, :, :]  # [16, 1, lat_h, lat_w]
                    frm_dec = self.vae.decode(
                        frm_lat.unsqueeze(0).to(self.device)
                    ).float().clamp(-1, 1).squeeze(0)                # [3, F_pix_1, h, w]
                    id_frames_list.append(frm_dec[:, 0].cpu())       # [3, h, w]
                    del frm_dec

                id_frames = (torch.stack(id_frames_list, dim=0) + 1) / 2 \
                    if M_identity > 0 else torch.empty(0, 3, h, w)   # [M_identity, 3, h, w]

                # 1a. Memory 帧：逐帧单独 decode（避免 3D VAE 时序污染）
                mem_frames_list = []
                for mi in range(vis_mem_offset, vis_mem_offset + M):
                    frm_lat = y_latent[:, mi:mi+1, :, :]          # [16, 1, lat_h, lat_w]
                    frm_dec = self.vae.decode(
                        frm_lat.unsqueeze(0).to(self.device)
                    ).float().clamp(-1, 1).squeeze(0)             # [3, F_pix_1, h, w]
                    mem_frames_list.append(frm_dec[:, 0].cpu())   # [3, h, w]
                    del frm_dec

                mem_frames = (torch.stack(mem_frames_list, dim=0) + 1) / 2 \
                    if M > 0 else torch.empty(0, 3, h, w)         # [M, 3, h, w]

                # 1b. Target 帧：整体 decode（跳过 identity + memory，两种模式偏移相同）
                tgt_lat = y_latent[:, M + M_identity:, :, :]      # [16, lat_f, lat_h, lat_w]
                tgt_dec = self.vae.decode(
                    tgt_lat.unsqueeze(0).to(self.device)
                ).float().clamp(-1, 1).squeeze(0)                 # [3, F_pix, h, w]
                tgt_frames = tgt_dec[:, ::self.vae_stride[0]]     # [3, lat_f, h, w]
                tgt_frames = ((tgt_frames + 1) / 2
                              ).permute(1, 0, 2, 3).cpu()         # [lat_f, 3, h, w]
                del tgt_dec

            # full_frames 按实际 y 中的帧顺序排列，与 mask 热图对应
            if idt_back_mode:
                # [mem | identity | tgt]
                full_frames = torch.cat([mem_frames, id_frames, tgt_frames], dim=0)
                mem_start, idt_start, tgt_start = 0, M, M + M_identity
            else:
                # [identity | mem | tgt]
                full_frames = torch.cat([id_frames, mem_frames, tgt_frames], dim=0)
                idt_start, mem_start, tgt_start = 0, M_identity, M_identity + M

            # 蓝框标注 identity 起始分界线
            if M_identity > 0 and idt_start < full_frames.shape[0]:
                b = 2
                full_frames[idt_start, :, :b,  :] = torch.tensor([0., 0., 1.]).view(3, 1, 1)
                full_frames[idt_start, :, -b:, :] = torch.tensor([0., 0., 1.]).view(3, 1, 1)
                full_frames[idt_start, :, :,  :b] = torch.tensor([0., 0., 1.]).view(3, 1, 1)
                full_frames[idt_start, :, :, -b:] = torch.tensor([0., 0., 1.]).view(3, 1, 1)

            # 红框标注 target 起始分界线
            if tgt_start < full_frames.shape[0]:
                b = 2
                full_frames[tgt_start, :, :b,  :] = torch.tensor([1., 0., 0.]).view(3, 1, 1)
                full_frames[tgt_start, :, -b:, :] = torch.tensor([1., 0., 0.]).view(3, 1, 1)
                full_frames[tgt_start, :, :,  :b] = torch.tensor([1., 0., 0.]).view(3, 1, 1)
                full_frames[tgt_start, :, :, -b:] = torch.tensor([1., 0., 0.]).view(3, 1, 1)

            if self.rank == 0:
                vis_dir = os.path.join(memory_dir, "vis_input") \
                    if memory_dir is not None else "./vis_input"
                os.makedirs(vis_dir, exist_ok=True)
                if not hasattr(self, '_vis_shot_idx'):
                    existing = glob.glob(os.path.join(vis_dir, "shot*_y_full.png"))
                    if existing:
                        indices = [int(m.group(1)) for f in existing
                                   if (m := re.search(r'shot(\d+)_y_full\.png',
                                                      os.path.basename(f)))]
                        self._vis_shot_idx = max(indices) + 1 if indices else 0
                    else:
                        self._vis_shot_idx = 0
                shot_vis_idx = self._vis_shot_idx
                self._vis_shot_idx = shot_vis_idx + 1

                # 保存拼合图（memory + target 横排）
                torchvision.utils.save_image(
                    full_frames,
                    os.path.join(vis_dir, f"shot{shot_vis_idx:02d}_y_full.png"),
                    nrow=M_identity + M + lat_f, normalize=False, padding=2, pad_value=0.5,
                )
                logging.info(f"[VIS] shot{shot_vis_idx:02d}_y_full.png saved  "
                             f"(M_identity={M_identity}, M={M}, lat_f={lat_f})")

                # mask 灰度热图（4 channel 均值 → RGB 重复）
                mask_mean = y_mask.mean(dim=0).unsqueeze(1).repeat(1, 3, 1, 1).clamp(0, 1).cpu()
                torchvision.utils.save_image(
                    mask_mean,
                    os.path.join(vis_dir, f"shot{shot_vis_idx:02d}_y_mask.png"),
                    nrow=M_identity + M + lat_f, normalize=False, padding=2, pad_value=0.5,
                )
                logging.info(f"[VIS] shot{shot_vis_idx:02d}_y_mask.png saved")

                # 统计 txt
                y_stat_path = os.path.join(vis_dir, f"shot{shot_vis_idx:02d}_y_stats.txt")
                with open(y_stat_path, "w") as f_stat:
                    f_stat.write(f"y shape: {list(y.shape)}\n")
                    f_stat.write(f"M_identity={M_identity}, M={M}, lat_f={lat_f}, lat_h={lat_h}, lat_w={lat_w}\n")
                    f_stat.write(f"use_subject_recon={use_subject_recon}\n")
                    f_stat.write(f"y_mask  (y[:4]):    mean={y_mask.mean():.4f} std={y_mask.std():.4f} "
                                 f"min={y_mask.min():.4f} max={y_mask.max():.4f}\n")
                    f_stat.write(f"y_latent (y[4:]):   mean={y_latent.mean():.4f} std={y_latent.std():.4f} "
                                 f"min={y_latent.min():.4f} max={y_latent.max():.4f}\n")
                    y_mem = y_latent[:, mem_start:mem_start + M]
                    y_tgt = y_latent[:, tgt_start:]
                    f_stat.write(f"y_mem   (y[4:, {mem_start}:{mem_start+M}]): mean={y_mem.mean():.4f} std={y_mem.std():.4f}\n")
                    f_stat.write(f"y_tgt   (y[4:, {tgt_start}:]): mean={y_tgt.mean():.4f} std={y_tgt.std():.4f}\n")
                    if M_identity > 0:
                        y_id = y_latent[:, idt_start:idt_start + M_identity]
                        f_stat.write(f"y_identity (y[4:, {idt_start}:{idt_start+M_identity}]): mean={y_id.mean():.4f} std={y_id.std():.4f}\n")
                    ch_mean = y_latent.float().mean(dim=(1, 2, 3))
                    ch_std  = y_latent.float().std(dim=(1, 2, 3))
                    f_stat.write(f"channel-wise mean: {[f'{v:.3f}' for v in ch_mean.tolist()]}\n")
                    f_stat.write(f"channel-wise std:  {[f'{v:.3f}' for v in ch_std.tolist()]}\n")
                    f_stat.write(f"input_prompt: {input_prompt}\n")
                    if use_subject_recon:
                        f_stat.write(f"reconstruct_caption: {reconstruct_caption}\n")
                logging.info(f"[VIS] shot{shot_vis_idx:02d}_y_stats.txt saved")

            del full_frames, id_frames, mem_frames, tgt_frames, y_latent, y_mask

        except Exception as _vis_e:
            logging.warning(f"[VIS] Visualization failed: {_vis_e}", exc_info=True)
        # =====================================================================
        # 可视化 END
        # =====================================================================

        max_seq_len = lat_t * lat_h * lat_w // (self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        noise = torch.randn(
            16, lat_t, lat_h, lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)
        if dist.is_initialized():
            dist.broadcast(noise, src=0)
        latent = noise

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low  = getattr(self.low_noise_model,  'no_sync', noop_no_sync)
        no_sync_high = getattr(self.high_noise_model, 'no_sync', noop_no_sync)

        if offload_model:
            torch.cuda.empty_cache()

        debug_dir = os.path.join(memory_dir, "debug") if memory_dir is not None else "./debug"
        os.environ["ROPE_DEBUG_DIR"] = debug_dir
        with (
            torch.amp.autocast('cuda', dtype=self.param_dtype),
            no_sync_low(),
            no_sync_high(),
        ):
            boundary = self.boundary * self.num_train_timesteps

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler, device=self.device, sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            arg_c = {
                'context': [context[0]],
                'seq_len': max_seq_len,
                'memory_size': M,
                'y': [y.to(self.param_dtype)],
                'time_indices': full_time_indices,
                'identity_frames_caption': identity_frames_caption,
                'identity_frame_num': [M_identity],
            }
            arg_null = {
                'context': [context_null[0]],
                'seq_len': max_seq_len,
                'memory_size': M,
                'y': [y.to(self.param_dtype)],
                'time_indices': full_time_indices,
                'identity_frames_caption': identity_frames_caption_null,
                # 'identity_frames_caption': identity_frames_caption,
                'identity_frame_num': [M_identity],
            }

            # dual-query: 为 high noise 模型构建独立的 arg 字典
            if use_dual_query and y_high is not None:
                arg_c_high = {
                    'context': [context[0]],
                    'seq_len': max_seq_len,
                    'memory_size': M,
                    'y': [y_high.to(self.param_dtype)],
                    'time_indices': full_time_indices_high,
                    'identity_frames_caption': identity_frames_caption,
                    'identity_frame_num': [M_identity],
                }
                arg_null_high = {
                    'context': [context_null[0]],
                    'seq_len': max_seq_len,
                    'memory_size': M,
                    'y': [y_high.to(self.param_dtype)],
                    'time_indices': full_time_indices_high,
                    'identity_frames_caption': identity_frames_caption_null,
                    'identity_frame_num': [M_identity],
                }
            else:
                arg_c_high = arg_c
                arg_null_high = arg_null

            # Debug: verify identity reconstruction setup
            if M_identity > 0:
                _model_check = self.low_noise_model if self.low_noise_model is not None else self.high_noise_model
                _is_sp = hasattr(_model_check, '__self__') or (hasattr(_model_check, 'forward') and getattr(_model_check.forward, '__func__', None) is not type(_model_check).forward)
                logging.info(
                    f"[SubjectRecon DEBUG] M_identity={M_identity}, "
                    f"identity_frames_caption={'None' if identity_frames_caption is None else f'List[{len(identity_frames_caption)}], shape={identity_frames_caption[0].shape}, device={identity_frames_caption[0].device}'}, "
                    f"identity_frame_num={[M_identity]}, "
                    f"sp_size={self.sp_size}, "
                    f"model_forward={type(_model_check).__name__}.{getattr(_model_check.forward, '__name__', '?')}"
                )

            for _step_i, t in enumerate(tqdm(timesteps)):
                # 与训练对齐：训练时 x[memory] = (1-σ_t)*clean_memory + σ_t*noise，
                # 推理若不处理则 x[memory] 由 ODE solver 积分得到，分布与训练不一致。
                # 每步强制将 latent 的 memory 位置重置为正确的加噪值。
                sigma_t = t.item() / self.num_train_timesteps
                _mem_offset = M_identity if not idt_back_mode else 0
                _clean_mem = (
                    memory_pool_high
                    if (use_dual_query and memory_pool_high is not None and t.item() >= boundary)
                    else memory_pool
                )
                latent[:, _mem_offset : _mem_offset + M] = (
                    (1.0 - sigma_t) * _clean_mem
                    + sigma_t * noise[:, _mem_offset : _mem_offset + M]
                )

                latent_model_input = [latent.to(self.device)]
                timestep = torch.stack([t]).to(self.device)
                model = self._prepare_model_for_timestep(t, boundary, offload_model)
                sample_guide_scale = guide_scale[1] if t.item() >= boundary else guide_scale[0]

                # dual-query: 根据 timestep 选择对应模型的 arg
                if t.item() >= boundary:
                    cur_arg_c, cur_arg_null = arg_c_high, arg_null_high
                else:
                    cur_arg_c, cur_arg_null = arg_c, arg_null

                # Debug: dual-query 模型切换日志
                if use_dual_query and y_high is not None:
                    if _step_i == 0:
                        _which = "HIGH_noise" if t.item() >= boundary else "LOW_noise"
                        logging.info(f"[use_both_query] Step 0: t={t.item():.1f}, boundary={boundary:.1f}, "
                                     f"using {_which} arg (y differs={cur_arg_c['y'][0].data_ptr() != arg_c['y'][0].data_ptr()})")
                    elif _step_i > 0:
                        _prev_t = timesteps[_step_i - 1].item()
                        if _prev_t >= boundary and t.item() < boundary:
                            logging.info(f"[use_both_query] Step {_step_i}: boundary crossed! "
                                         f"t={t.item():.1f} < boundary={boundary:.1f}, "
                                         f"switching from HIGH_noise to LOW_noise arg")

                noise_pred_cond = model(
                    latent_model_input, t=timestep, **cur_arg_c)[0]
                if offload_model:
                    torch.cuda.empty_cache()

                noise_pred_uncond = model(
                    latent_model_input, t=timestep, **cur_arg_null)[0]
                if offload_model:
                    torch.cuda.empty_cache()

                # Debug: 首末两步打印 cond/uncond 在 identity 和 target 区域的差异
                if M_identity > 0 and _step_i in (0, len(timesteps) - 1):
                    _diff = noise_pred_cond - noise_pred_uncond
                    _dbg_idt_offset = M if idt_back_mode else 0
                    _dbg_tgt_offset = M + M_identity
                    _id_diff = _diff[:, _dbg_idt_offset:_dbg_idt_offset + M_identity]
                    _tgt_diff = _diff[:, _dbg_tgt_offset:]
                    logging.info(
                        "[SubjectRecon STEP %d] t=%.4f, guide=%.1f | "
                        "cond-uncond identity: mean=%.6f std=%.6f | "
                        "cond-uncond target: mean=%.6f std=%.6f | "
                        "cond identity: mean=%.4f std=%.4f | uncond identity: mean=%.4f std=%.4f",
                        _step_i, t.item(), sample_guide_scale,
                        _id_diff.mean().item(), _id_diff.std().item(),
                        _tgt_diff.mean().item(), _tgt_diff.std().item(),
                        noise_pred_cond[:, _dbg_idt_offset:_dbg_idt_offset + M_identity].mean().item(),
                        noise_pred_cond[:, _dbg_idt_offset:_dbg_idt_offset + M_identity].std().item(),
                        noise_pred_uncond[:, _dbg_idt_offset:_dbg_idt_offset + M_identity].mean().item(),
                        noise_pred_uncond[:, _dbg_idt_offset:_dbg_idt_offset + M_identity].std().item(),
                    )

                noise_pred = noise_pred_uncond + sample_guide_scale * (
                    noise_pred_cond - noise_pred_uncond)
                del noise_pred_cond, noise_pred_uncond
                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0), t, latent.unsqueeze(0),
                    return_dict=False, generator=seed_g,
                )[0]
                latent = temp_x0.squeeze(0)
                del noise_pred, temp_x0, latent_model_input, timestep

            x0 = latent
            del latent, noise

            # Debug: 对比 identity / memory / target 各部分的 latent 统计量
            if M_identity > 0:
                id_lat = x0[:, :M_identity]
                mem_lat = x0[:, M_identity:M_identity + M] if M > 0 else None
                tgt_lat = x0[:, M_identity + M:]
                logging.info(
                    "[SubjectRecon LATENT] identity: mean=%.4f std=%.4f min=%.4f max=%.4f | "
                    "memory: %s | "
                    "target: mean=%.4f std=%.4f min=%.4f max=%.4f",
                    id_lat.mean().item(), id_lat.std().item(), id_lat.min().item(), id_lat.max().item(),
                    ("mean=%.4f std=%.4f" % (mem_lat.mean().item(), mem_lat.std().item())) if mem_lat is not None else "N/A",
                    tgt_lat.mean().item(), tgt_lat.std().item(), tgt_lat.min().item(), tgt_lat.max().item(),
                )

            # Diffusion 已结束，提前释放不再需要的大 tensor，为 VAE decode 腾出显存
            del y, context, context_null, identity_frames_caption, sample_scheduler, arg_c, arg_null
            if y_high is not None:
                del y_high
            del arg_c_high, arg_null_high
            used_memory_cpu = memory_pool.cpu()
            del memory_pool
            used_memory_high_cpu = memory_pool_high.cpu() if memory_pool_high is not None else None
            if memory_pool_high is not None:
                del memory_pool_high

            if offload_model:
                if self.low_noise_model is not None and not self._is_fsdp(self.low_noise_model):
                    self.low_noise_model.cpu()
                if self.high_noise_model is not None and not self._is_fsdp(self.high_noise_model):
                    self.high_noise_model.cpu()
                gc.collect()
                torch.cuda.empty_cache()

            video = None
            video_full = None
            recon_frames = None
            if self.rank == 0:
                # x0 布局取决于 idt_back_mode：
                #   idt_back_mode=False: [identity | mem | tgt]
                #   idt_back_mode=True:  [mem | identity | tgt]
                idt_offset = M if idt_back_mode else 0
                mem_offset = 0 if idt_back_mode else M_identity
                tgt_offset = M + M_identity

                # 主题重建：提取并 decode identity frame 部分
                if M_identity > 0:
                    recon_frames = []
                    for ri in range(M_identity):
                        frm = self.vae.decode(
                            x0[:, idt_offset + ri:idt_offset + ri + 1].unsqueeze(0)
                        ).float().clamp_(-1, 1).squeeze(0)
                        recon_frames.append(frm[:, 0])  # [3, h, w]
                    recon_frames = torch.stack(recon_frames, dim=0)  # [M_identity, 3, h, w]
                    torch.cuda.empty_cache()

                # 只 decode target 部分（跳过 identity + memory）
                video = self.vae.decode(x0[:, tgt_offset:].unsqueeze(0)).float().clamp_(-1, 1).squeeze(0)
                torch.cuda.empty_cache()

                tgt_video = video
                if M > 0:
                    mem_frames = []
                    for mi in range(M):
                        frm = self.vae.decode(
                            used_memory_cpu[:, mi:mi+1].unsqueeze(0).to(self.device)
                        ).float().clamp_(-1, 1).squeeze(0)
                        mem_frames.append(frm[:, 0:1])
                    mem_video = torch.cat(mem_frames, dim=1)
                    video_full = torch.cat([mem_video, tgt_video], dim=1)
                else:
                    video_full = tgt_video

                # 将 identity 重建帧拼入 video_full，idt_back_mode 决定位置
                if recon_frames is not None:
                    # recon_frames: [M_identity, 3, h, w] → [3, M_identity, h, w]
                    id_video = recon_frames.permute(1, 0, 2, 3)
                    if idt_back_mode:
                        # [mem | identity | tgt]
                        video_full = torch.cat([mem_video, id_video, tgt_video], dim=1) if M > 0 else torch.cat([id_video, tgt_video], dim=1)
                    else:
                        # [identity | mem | tgt]
                        video_full = torch.cat([id_video, video_full], dim=1)

                # 可视化重建结果（与 vis_input 同目录，保证 shot 编号对齐）
                if recon_frames is not None:
                    try:
                        import torchvision
                        recon_vis_dir = os.path.join(
                            memory_dir if memory_dir is not None else "./",
                            "recon_vis"
                        )
                        os.makedirs(recon_vis_dir, exist_ok=True)
                        if not hasattr(self, '_recon_shot_idx'):
                            self._recon_shot_idx = getattr(self, '_vis_shot_idx', 0)
                        shot_idx = self._recon_shot_idx
                        self._recon_shot_idx += 1

                        # 保存每帧重建结果
                        for ri in range(recon_frames.shape[0]):
                            frame_img = (recon_frames[ri].clamp(-1, 1) + 1) / 2  # [3, h, w], [0,1]
                            torchvision.utils.save_image(
                                frame_img,
                                os.path.join(recon_vis_dir, f"shot{shot_idx:02d}_recon_{ri:02d}.png"),
                            )
                        # 保存 grid 拼合图
                        recon_grid = (recon_frames.clamp(-1, 1) + 1) / 2
                        torchvision.utils.save_image(
                            recon_grid,
                            os.path.join(recon_vis_dir, f"shot{shot_idx:02d}_recon_grid.png"),
                            nrow=M_identity, padding=2, pad_value=0.5,
                        )
                        logging.info(f"[SubjectRecon VIS] Saved {M_identity} reconstructed frames to {recon_vis_dir}/shot{shot_idx:02d}_recon_*.png")
                    except Exception as e:
                        logging.warning(f"[SubjectRecon VIS] Visualization failed: {e}")

                # recon_frames 移到 CPU 以便返回
                if recon_frames is not None:
                    recon_frames = recon_frames.cpu()
            del x0

        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.barrier()

        return video, used_memory_cpu, used_memory_high_cpu, video_full, recon_frames

    def forward(
        self,
        model,
        target_video: torch.Tensor,
        memory_pool: torch.Tensor,
        prompts: list,
        reference_image: torch.Tensor = None,
    ):
        B = target_video.shape[0]
        device = target_video.device

        with torch.no_grad():
            context = self._encode_text(prompts, device)

            x_latent = self.vae.encode(target_video).float()
            _, lat_C, lat_F, lat_H, lat_W = x_latent.shape

            y_latent = None
            if reference_image is not None:
                ref_enc = self.vae.encode(reference_image.unsqueeze(2)).float()
                y_latent = torch.cat([
                    ref_enc,
                    torch.zeros(B, lat_C, lat_F - 1, lat_H, lat_W, device=device)
                ], dim=2)

        x_1 = x_latent
        x_0 = torch.randn_like(x_1)

        t_continuous = torch.rand((B,), device=device)
        t_model = (t_continuous * self.num_train_timesteps).to(device)

        t_expand = t_continuous.view(B, 1, 1, 1, 1)
        noisy_x = t_expand * x_1 + (1.0 - t_expand) * x_0
        target_velocity = x_0 - x_1

        x_list = [noisy_x[i] for i in range(B)]

        memory_pool_list = [memory_pool[i] for i in range(B)]

        if y_latent is not None:
            y_mask = torch.zeros(B, 4, lat_F, lat_H, lat_W, device=device, dtype=y_latent.dtype)
            y_full = torch.cat([y_mask, y_latent], dim=1)
        else:
            y_full = torch.zeros(B, 20, lat_F, lat_H, lat_W, device=device, dtype=noisy_x.dtype)
        y_list = [y_full[i] for i in range(B)]

        num_keyframes = model.num_keyframes
        seq_len = self._compute_seq_len(lat_F, lat_H, lat_W, num_keyframes)

        with torch.amp.autocast('cuda', dtype=self.param_dtype):
            pred_list, selected_memories, attn_weights = model(
                x=x_list,
                t=t_model,
                context=context,
                seq_len=seq_len,
                y=y_list,
                memory_pool=memory_pool_list,
            )

        loss = sum(
            torch.nn.functional.mse_loss(pred.float(), target_velocity[i].float())
            for i, pred in enumerate(pred_list)
        ) / B

        return loss, selected_memories, attn_weights