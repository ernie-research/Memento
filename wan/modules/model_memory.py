# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

# from .attention import flash_attention
from wan.modules.attention import flash_attention
import torch.utils.checkpoint as checkpoint
from diffsynth.core.gradient.gradient_checkpoint import gradient_checkpoint_forward
__all__ = ['WanModel_Memory']

ROPE_MAX_LEN = 1024
MEMORY_ROPE_SHIFT = 5

def sinusoidal_embedding_1d(position, dim, theta=10000):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(theta, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast('cuda', enabled=False)
def rope_params(position, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        position,
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs, memory_size=0, time_indices=None):
    if isinstance(time_indices, int):
        memory_size = time_indices
        time_indices = None
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    
    max_rope_idx = freqs[0].shape[0] - 1

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))

        if time_indices is not None:
            t_list = time_indices[i]

            if len(t_list) > f:
                t_list = t_list[:f]
            elif len(t_list) < f:
                last_val = t_list[-1] if len(t_list) > 0 else ROPE_MAX_LEN // 2
                t_list = t_list + [last_val] * (f - len(t_list))

            idx_t = torch.tensor(t_list, dtype=torch.long, device=freqs[0].device)
            idx_t = torch.clamp(idx_t, 0, max_rope_idx)
            freqs_t = freqs[0][idx_t].view(f, 1, 1, -1)

        elif memory_size > 0:
            pos_t = torch.cat([
                torch.arange(-memory_size * MEMORY_ROPE_SHIFT, 0, MEMORY_ROPE_SHIFT, device=freqs[0].device),
                torch.arange(0, f - memory_size, device=freqs[0].device)
            ])
            idx_t = (pos_t + ROPE_MAX_LEN // 2).long()
            idx_t = torch.clamp(idx_t, 0, max_rope_idx) # 同步加上保护
            freqs_t = freqs[0][idx_t].view(f, 1, 1, -1)
        else:
            freqs_t = freqs[0][:f].view(f, 1, 1, -1)

        freqs_i = torch.cat([
            freqs_t.expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()

class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def reset_parameters(self):
        if hasattr(self, 'weight'):
            nn.init.ones_(self.weight)
    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, memory_size, time_indices=None, identity_seq_lens=None, split_identity_attn=False):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [ROPE_MAX_LEN, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if identity_seq_lens is not None and split_identity_attn:
            assert b == 1, "Split attention 目前仅支持 batch_size=1"
            assert memory_size > 0

            # 先对完整 q, k 做 RoPE，再切片
            q = rope_apply(q, grid_sizes, freqs, memory_size, time_indices)
            k = rope_apply(k, grid_sizes, freqs, memory_size, time_indices)

            id_len = identity_seq_lens[0].item()
            total_valid = seq_lens[0].item()
            F_total, H, W = grid_sizes[0].tolist()
            hw = int(H * W)
            mem_len = memory_size * hw
            # print(memory_size, H, W, mem_len, "2222")
            gen_len = total_valid - id_len - mem_len
            print(total_valid, id_len, mem_len, gen_len, "3333")

            # ---- 切片（三段在序列维度上连续）----
            #  [0, id_len)                           → identity
            #  [id_len, id_len + mem_len)            → memory
            #  [id_len + mem_len, total_valid)       → generation

            # SA1 范围: [0, id_len + mem_len)  — identity + memory，本身就连续
            sa1_end = id_len + mem_len
            q1, k1, v1 = q[:, :sa1_end], k[:, :sa1_end], v[:, :sa1_end]
            sa1_lens = torch.tensor([sa1_end], device=x.device, dtype=seq_lens.dtype)

            # SA2 范围: [id_len, total_valid)  — memory + generation，也连续
            q2, k2, v2 = q[:, id_len:total_valid], k[:, id_len:total_valid], v[:, id_len:total_valid]
            sa2_lens = torch.tensor([mem_len + gen_len], device=x.device, dtype=seq_lens.dtype)

            # ---- 两次 flash attention 并行（计算图独立）----
            out1 = flash_attention(
                q=q1, k=k1, v=v1,
                k_lens=sa1_lens,
                window_size=self.window_size,
            )  # [1, sa1_end, n, d]

            out2 = flash_attention(
                q=q2, k=k2, v=v2,
                k_lens=sa2_lens,
                window_size=self.window_size,
            )  # [1, mem_len + gen_len, n, d]
    
            # ---- 拆分各段输出 ----
            id_out    = out1[:, :id_len]           # identity  from SA1
            mem_out_1 = out1[:, id_len:sa1_end]    # memory    from SA1
            mem_out_2 = out2[:, :mem_len]          # memory    from SA2
            gen_out   = out2[:, mem_len:]          # generation from SA2
    
            # ---- memory 输出：固定 0.5 gate 融合 ----
            # mem_out = 0.5 * mem_out_1 + 0.5 * mem_out_2
            mem_out = 0.5 * mem_out_1 + 0.5 * mem_out_2
    
            # ---- 重新拼装完整序列（含 padding 位置补零）----
            result = q.new_zeros(b, s, n, d)
            result[:, :id_len]                     = id_out
            result[:, id_len:id_len + mem_len]     = mem_out
            result[:, id_len + mem_len:total_valid] = gen_out
            x = result
        else:
            x = flash_attention(
                q=rope_apply(q, grid_sizes, freqs, memory_size, time_indices),
                k=rope_apply(k, grid_sizes, freqs, memory_size, time_indices),
                v=v,
                k_lens=seq_lens,
                window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

class WanCrossAttention(WanSelfAttention):

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, visual_tokens=False):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)
        self.visual_tokens = visual_tokens

    def forward(self, x, context, context_lens, grid_sizes=None, freqs=None, time_indices=None, identity_context=None, identity_seq_lens=None, enable_multi_identity_train=False, identity_frame_num=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            identity_context: 单帧模式下为 Tensor [N, text_len, dim]; 多帧模式下为 List[Tensor], 每个 [1, text_len, dim]
            identity_seq_lens(Tensor): 每个 batch item 的 identity token 总数
            enable_multi_identity_train(bool): 是否启用多主体帧独立 cross-attention
            identity_frame_num(list): 每个 batch item 的 identity 帧数
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # Get output projection weight dtype for consistency
        out_dtype = self.o.weight.dtype

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d).contiguous()
        k = self.norm_k(self.k(context)).view(b, -1, n, d).contiguous()
        if self.visual_tokens:
            k = rope_apply(k, grid_sizes, freqs, time_indices=time_indices)
        v = self.v(context).view(b, -1, n, d).contiguous()

        if identity_context is None:
            # compute attention
            x = flash_attention(q, k, v, k_lens=context_lens)
        elif enable_multi_identity_train and isinstance(identity_context, list):
            # 多帧模式：每帧 spatial tokens 独立与各自 caption context 做 cross-attention
            assert b == 1
            total_identity_tokens = identity_seq_lens[0].item()
            num_frames = identity_frame_num[0]
            tokens_per_frame = total_identity_tokens // num_frames

            # Q 按 visual token 顺序拆分：identity 部分 vs memory+target 部分
            q_identity_all = q[:, :total_identity_tokens]
            q_ori = q[:, total_identity_tokens:]

            # K 和 V 来自 text context，不做拆分；memory+target 的 Q attend to 完整 clip caption
            x_ori = flash_attention(q_ori, k, v, k_lens=context_lens)

            # 每帧 identity tokens 独立 attend to 该帧的 caption context
            x_identity_parts = []
            for frame_i in range(num_frames):
                start = frame_i * tokens_per_frame
                end = start + tokens_per_frame
                q_frame = q_identity_all[:, start:end]

                # 该帧的 caption context → K, V projection
                ctx_i = identity_context[frame_i]  # [1, text_len, dim]
                k_frame = self.norm_k(self.k(ctx_i)).view(b, -1, n, d).contiguous()
                v_frame = self.v(ctx_i).view(b, -1, n, d).contiguous()

                x_frame = flash_attention(q_frame, k_frame, v_frame, k_lens=None)
                x_identity_parts.append(x_frame)

            x_identity = torch.cat(x_identity_parts, dim=1)
            x = torch.cat((x_identity, x_ori), dim=1)
        else:
            # 单帧模式：identity Q attend to identity_context 的 K,V
            assert b == 1
            k_identity = self.norm_k(self.k(identity_context)).view(b, -1, n, d).contiguous()
            v_identity = self.v(identity_context).view(b, -1, n, d).contiguous()
            q_identity = q[:, :identity_seq_lens[0]]
            q_ori = q[:, identity_seq_lens[0]:]

            x_ori = flash_attention(q_ori, k, v, k_lens=context_lens)
            x_identity = flash_attention(q_identity, k_identity, v_identity, k_lens=None)
            x = torch.cat((x_identity, x_ori), dim=1)
        # output - ensure dtype matches Linear layer weights
        x = x.flatten(2).contiguous().to(out_dtype)
        x = self.o(x)
        return x

class WanCrossAttention_old(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        # gradient checkpointing offload flag，由外层 WanModel_Memory 统一设置
        self.use_gradient_checkpointing_offload = False
    def reset_parameters(self):
        if hasattr(self, 'modulation'):
            nn.init.normal_(self.modulation, std=self.dim**-0.5)
    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        memory_size,
        time_indices,
        identity_context=None,
        identity_seq_lens=0,
        enable_multi_identity_train=False,
        identity_frame_num=None,
        split_identity_attn=False,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [ROPE_MAX_LEN, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens, grid_sizes, freqs, memory_size, time_indices=time_indices, identity_seq_lens=identity_seq_lens, split_identity_attn=split_identity_attn)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[2].squeeze(2)

        # self-attention wrong
        # e[0]/e[1] 是 float32 的 per-token modulation，直接 cast 到 x.dtype(bf16) 再做 modulation，
        # 避免 norm1(x).float() 产生 [seq_len, dim] 的 float32 大张量触发 OOM
        # e0_bf = e[0].squeeze(2).to(x.dtype)
        # e1_bf = e[1].squeeze(2).to(x.dtype)
        # x_normed = self.norm1(x) * (1 + e1_bf) + e0_bf
        # del e0_bf, e1_bf
        # y = self.self_attn(
        #     x_normed,
        #     seq_lens, grid_sizes, freqs, memory_size, time_indices=time_indices, identity_seq_lens=identity_seq_lens)
        # del x_normed
        # x = x + y * e[2].squeeze(2).to(x.dtype)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, identity_context, identity_seq_lens):
            x = x + self.cross_attn(
                self.norm3(x), context, context_lens,
                identity_context=identity_context,
                identity_seq_lens=identity_seq_lens,
                enable_multi_identity_train=enable_multi_identity_train,
                identity_frame_num=identity_frame_num,
            )
            ### right
            y = self.ffn(
                self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            with torch.amp.autocast('cuda', dtype=torch.float32):
                x = x + y * e[5].squeeze(2)
            
            # wrong implementation
            # e3_bf = e[3].squeeze(2).to(x.dtype)
            # e4_bf = e[4].squeeze(2).to(x.dtype)
            # y = self.ffn(self.norm2(x) * (1 + e4_bf) + e3_bf)
            # del e3_bf, e4_bf
            # with torch.amp.autocast('cuda', dtype=torch.float32):
            #     x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, identity_context, identity_seq_lens)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)
    def reset_parameters(self):
        if hasattr(self, 'modulation'):
            nn.init.normal_(self.modulation, std=self.dim**-0.5)
    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x

class KeyframeQuery(nn.Module):
    def __init__(self, dim, num_keyframes, patch_size=(1, 2, 2), temperature=1.0, num_heads=16, gradient_checkpointing=True, use_gradient_checkpointing_offload=False, split_learnable_query=False, global_query_num=2, selected_local_num=6):
        super().__init__()
        self.num_keyframes = num_keyframes
        self.temperature = temperature
        self.patch_size = patch_size
        self.gradient_checkpointing = gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        
        # 专用的 16 通道 Patch Embedding
        # 彻底解决 16通道 Memory Latent 送给 36通道主模型 Patching 导致的特征错乱问题
        self.memory_patcher = nn.Conv3d(
            16, dim, kernel_size=patch_size, stride=patch_size)

        # 避开 LoRA 关键词
        self.query_to_text = WanCrossAttention(dim, num_heads=num_heads, qk_norm=True)
        self.query_to_visual = WanCrossAttention(dim, num_heads=num_heads, qk_norm=True, visual_tokens=True)
        self.split_learnable_query = split_learnable_query
        self.selected_local_num = selected_local_num
        self.global_query_num = global_query_num
        if self.split_learnable_query and global_query_num == 0:
            raise ValueError(
                "split_learnable_query=True requires global_query_num > 0, "
                "please set global_query_num when using split_learnable_query."
            )
        if global_query_num:
            self.queries_global = nn.Parameter(torch.randn(global_query_num, dim) / dim**0.5)
            self.queries = nn.Parameter(torch.randn(num_keyframes - global_query_num, dim) / dim**0.5)
        else:
            self.queries = nn.Parameter(torch.randn(num_keyframes, dim) / dim**0.5)
            self.queries_global = None
        print("num_keyframes: ", num_keyframes)


    def reset_parameters(self):
        if hasattr(self, 'queries') and self.queries is not None:
            dim = self.queries.shape[1]
            nn.init.normal_(self.queries, std=dim**-0.5)
        if hasattr(self, 'queries_global') and self.queries_global is not None:
            dim = self.queries_global.shape[1]
            nn.init.normal_(self.queries_global, std=dim**-0.5)

    def _forward_single_item(self, memory_latent_i, queries_param, ctx_i, ctx_len_i, grid_sizes_i, freqs, t_idx, F_patches, H_patches, W_patches, ctx_global=None, ctx_len_global=None):

        x_i = self.memory_patcher(memory_latent_i.unsqueeze(0)) # [1, dim, F_p, H_p, W_p]
        tokens = x_i.flatten(2).transpose(1, 2) # [1, F_patches * H_patches * W_patches, dim]

        queries_i = queries_param.unsqueeze(0).to(tokens.dtype) # [1, K, dim]
        if self.split_learnable_query and self.queries_global is not None:
            queries_global = self.queries_global.unsqueeze(0).to(tokens.dtype)
            queries_global = queries_global + self.query_to_text(queries_global, ctx_global, ctx_len_global)
            queries_i = queries_i + self.query_to_text(queries_i, ctx_i, ctx_len_i)
            print("queries_global:", queries_global.shape, queries_global)
            print("queries_i:", queries_i.shape, queries_i)
            queries_i = torch.cat([queries_i, queries_global], dim=1)
        else:
            queries_i = queries_i + self.query_to_text(queries_i, ctx_i, ctx_len_i)
        
        queries_enhanced = queries_i + self.query_to_visual(
            queries_i, tokens, context_lens=None,
            grid_sizes=grid_sizes_i, freqs=freqs, time_indices=t_idx
        )

        b, n, d = 1, self.query_to_visual.num_heads, self.query_to_visual.head_dim
        q = self.query_to_visual.q(queries_enhanced).view(b, -1, n, d)  
        k = self.query_to_visual.k(tokens).view(b, -1, n, d)  

        k = rope_apply(k, grid_sizes_i, freqs, time_indices=t_idx)  

        q = q.transpose(1, 2)  
        k = k.transpose(1, 2)  

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        scores = scores.mean(dim=1).squeeze(0)
        assert scores.shape[0] == self.num_keyframes, (
            f"Query count mismatch: got {scores.shape[0]}, expected {self.num_keyframes}. "
            f"Check num_keyframes={self.num_keyframes} and global_query_num={self.global_query_num}."
        )
        scores = scores.view(self.num_keyframes, F_patches, H_patches * W_patches).mean(dim=-1)
        print("scores:", scores.shape, scores)
        if self.global_query_num:
            global_frame_scores_global = scores[-self.global_query_num:,:].mean(dim=0)
            global_frame_scores_local = scores[ : -self.global_query_num,:].mean(dim=0)
            global_frame_scores = scores.mean(dim=0)
            # global_frame_scores = scores.mean(dim=0) # [F_patches]
            print("global_frame_scores_global:", global_frame_scores_global.shape, global_frame_scores_global)

            print("global_frame_scores_local:", global_frame_scores_local.shape, global_frame_scores_local)

            print("global_frame_scores_global", global_frame_scores_global.shape, global_frame_scores_global)


            k_to_select = min(self.selected_local_num, F_patches)
            topk_values, topk_indices = torch.topk(global_frame_scores_local, k_to_select)

            print("topk_indices", topk_indices.shape, topk_indices)

            # global 选帧时排除 local 已选帧，避免重合
            global_scores_masked = global_frame_scores_global.clone()
            global_scores_masked[topk_indices] = float('-inf')
            k_to_select_global = min(self.global_query_num, max(1, F_patches - k_to_select))
            topk_values_global, topk_indices_global = torch.topk(global_scores_masked, k_to_select_global)
            print("topk_values_global", topk_values_global.shape, topk_indices_global)
            topk_indices = torch.cat([topk_indices, topk_indices_global])
        else:
            global_frame_scores = scores.mean(dim=0) # [F_patches]
            k_to_select = min(self.selected_local_num, F_patches)
            topk_values, topk_indices = torch.topk(global_frame_scores, k_to_select)
        # attn_weights 的 shape: [num_keyframes, F_patches]
        # 每一行对应一个 query，表示该 query 选择哪个 frame
        attn_weights = torch.zeros(len(topk_indices), F_patches, device=scores.device, dtype=scores.dtype)
        if self.training:
            if self.split_learnable_query:
                prob_local  = torch.softmax(global_frame_scores_local  / self.temperature, dim=-1)
                # prob_global 也在 mask 后的分布上计算，梯度不流向已被 local 选中的帧
                masked_global_scores_for_prob = global_frame_scores_global.clone()
                masked_global_scores_for_prob[topk_indices[:k_to_select]] = float('-inf')
                prob_global = torch.softmax(masked_global_scores_for_prob / self.temperature, dim=-1)
                for i, idx in enumerate(topk_indices):
                    if i < k_to_select:
                        attn_weights[i, idx] = 1.0 - prob_local[idx].detach()  + prob_local[idx]
                    else:
                        attn_weights[i, idx] = 1.0 - prob_global[idx].detach() + prob_global[idx]
            else:
                prob = torch.softmax(global_frame_scores / self.temperature, dim=-1)
                for i, idx in enumerate(topk_indices):
                    attn_weights[i, idx] = 1.0 - prob[idx].detach() + prob[idx]
        else:
            for i, idx in enumerate(topk_indices):
                attn_weights[i, idx] = 1.0

        return attn_weights

    def forward(self, memory_latents, freqs, context, context_lens=None, time_indices=None, context_global=None, context_lens_global=None):
        """
        memory_latents: [B, 16, F, H, W] 原始 VAE Latents
        context_global: [B, L, C] global caption embeddings (used when split_learnable_query=True)
        context_lens_global: [B] lengths of global caption (used when split_learnable_query=True)
        """
        device = self.queries.device
        
        B, C_in, F, H, W = memory_latents.shape

        F_p = F // self.patch_size[0]
        H_p = H // self.patch_size[1]
        W_p = W // self.patch_size[2]

        grid_sizes = torch.tensor([[F_p, H_p, W_p]], device=device).repeat(B, 1)
        attn_weights_list = []

        for i in range(B):
            # 取出当前样本 (体积很小)
            memory_latent_i = memory_latents[i]

            ctx_i = context[i:i+1]
            ctx_len_i = context_lens[i:i+1] if context_lens is not None else None
            t_idx = time_indices[i:i+1] if time_indices is not None else None
            grid_sizes_i = grid_sizes[i:i+1]

            # split_learnable_query 模式下，取出 global caption context
            ctx_global_i = context_global[i:i+1] if context_global is not None else None
            ctx_len_global_i = context_lens_global[i:i+1] if context_lens_global is not None else None

            if self.training and (self.gradient_checkpointing or self.use_gradient_checkpointing_offload):
                memory_latent_i = memory_latent_i.contiguous()
                ctx_i = ctx_i.contiguous()

                attn_weights = gradient_checkpoint_forward(
                    self._forward_single_item,
                    self.gradient_checkpointing,
                    self.use_gradient_checkpointing_offload,
                    memory_latent_i, self.queries, ctx_i, ctx_len_i, grid_sizes_i, freqs, t_idx, F_p, H_p, W_p, ctx_global_i, ctx_len_global_i,
                )
            else:
                attn_weights = self._forward_single_item(
                    memory_latent_i, self.queries, ctx_i, ctx_len_i, grid_sizes_i, freqs, t_idx, F_p, H_p, W_p, ctx_global_i, ctx_len_global_i,
                )

            attn_weights_list.append(attn_weights)

        return attn_weights_list
        
class WanModel_Memory(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True
    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 num_keyframes=1,
                 keyframe_temperature=0.1,
                 split_identity_attn=False,
                 split_learnable_query = False,
                 global_query_num = 0,
                 selected_local_num = 6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.gradient_checkpointing = False
        self.use_gradient_checkpointing_offload = False
        self.split_identity_attn = split_identity_attn
        self.split_learnable_query = split_learnable_query
        self.global_query_num = global_query_num
        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps) for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        with torch.device("cpu"):
            self.freqs = torch.cat([
                rope_params(torch.arange(-ROPE_MAX_LEN // 2, ROPE_MAX_LEN // 2), d - 4 * (d // 6)),
                rope_params(torch.arange(ROPE_MAX_LEN), 2 * (d // 6)),
                rope_params(torch.arange(ROPE_MAX_LEN), 2 * (d // 6))
            ], dim=1)
        self.selected_local_num = selected_local_num
        # initialize weights
        self.init_weights()
        self.keyframe_query = KeyframeQuery(
            dim=dim,
            num_keyframes=num_keyframes,
            patch_size=patch_size,               
            temperature=keyframe_temperature,
            num_heads=num_heads,
            split_learnable_query=self.split_learnable_query,
            global_query_num=self.global_query_num,
            selected_local_num = self.selected_local_num
        )

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        memory_size,
        time_indices=None,
        y=None,
        timer=None,
        identity_frames_caption=None,
        identity_frame_num=[0],
        enable_multi_identity_train=False,
        split_identity_attn=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            memory_size (`int`):
                Number of the memory frames for rotary embeddings
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            timer: Optional StepTimer instance for performance profiling
            identity_frames_caption (List[Tensor], *optional*):
                T5-encoded identity captions, each with shape [L, text_dim]
            identity_frame_num (list):
                Number of identity frames per batch item
            enable_multi_identity_train (bool):
                Whether to use per-frame cross-attention for multi-identity

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        identity_seq_lens = torch.tensor([identity_frame * u.shape[3] * u.shape[4] for identity_frame, u in zip(identity_frame_num, x)], dtype=torch.long)
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(t, self.freq_dim).unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # identity context
        identity_context = None
        if identity_frames_caption is not None:
            if enable_multi_identity_train:
                # 多帧模式：每帧单独编码，保持为 List[Tensor]
                identity_context = []
                for cap in identity_frames_caption:
                    padded = torch.cat([cap, cap.new_zeros(self.text_len - cap.size(0), cap.size(1))])
                    embedded = self.text_embedding(padded.unsqueeze(0))  # [1, text_len, dim]
                    identity_context.append(embedded)
            else:
                # 单帧模式：stack 后一起编码
                identity_context = self.text_embedding(
                    torch.stack([
                        torch.cat(
                            [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                        for u in identity_frames_caption
                    ]))

        # resolve split_identity_attn: forward 参数优先，否则用 self 属性
        _split_identity_attn = split_identity_attn if split_identity_attn is not None else self.split_identity_attn

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            memory_size=memory_size,
            time_indices=time_indices,
            identity_context=identity_context,
            identity_seq_lens=identity_seq_lens,
            enable_multi_identity_train=enable_multi_identity_train,
            identity_frame_num=identity_frame_num,
            split_identity_attn=_split_identity_attn
        )

        # Block 层计时
        if timer is not None:
            timer.start("model_blocks")

        for block in self.blocks:
            block.use_gradient_checkpointing_offload = self.use_gradient_checkpointing_offload
            x = gradient_checkpoint_forward(
                block,
                self.training and self.gradient_checkpointing,
                self.training and self.use_gradient_checkpointing_offload,
                x, **kwargs,
            )

        if timer is not None:
            timer.end("model_blocks")

        # head
        if timer is not None:
            timer.start("model_head")
        x = self.head(x, e)
        if timer is not None:
            timer.end("model_head")

        # unpatchify
        if timer is not None:
            timer.start("model_unpatchify")
        x = self.unpatchify(x, grid_sizes)
        if timer is not None:
            timer.end("model_unpatchify")
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)