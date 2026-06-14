# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

from ..modules.model import sinusoidal_embedding_1d
from ..modules.attention import flash_attention
from .ulysses import distributed_attention
from .util import all_to_all, gather_forward, get_rank, get_world_size

ROPE_MAX_LEN = 1024
MEMORY_ROPE_SHIFT = 5


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs, memory_size=0, time_indices=None):
    """
    x:          [B, L, N, C].   (L = seq_len / sp_size after SP chunking)
    grid_sizes: [B, 3].         (full grid: F, H, W)
    freqs:      [M, C // 2].    (memory model: temporal centered at 512)
    memory_size: int             number of memory temporal frames
    time_indices: list[list[int]] custom temporal RoPE indices per sample
    """
    if isinstance(time_indices, int):
        memory_size = time_indices
        time_indices = None

    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    max_rope_idx = freqs[0].shape[0] - 1

    sp_size = get_world_size()
    sp_rank = get_rank()

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
            s, n, -1, 2))

        # --- temporal freqs: same logic as non-SP rope_apply in model_memory.py ---
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

        elif memory_size is not None and memory_size > 0:
            pos_t = torch.cat([
                torch.arange(-memory_size * MEMORY_ROPE_SHIFT, 0, MEMORY_ROPE_SHIFT, device=freqs[0].device),
                torch.arange(0, f - memory_size, device=freqs[0].device)
            ])
            idx_t = (pos_t + ROPE_MAX_LEN // 2).long()
            idx_t = torch.clamp(idx_t, 0, max_rope_idx)
            freqs_t = freqs[0][idx_t].view(f, 1, 1, -1)
        else:
            freqs_t = freqs[0][:f].view(f, 1, 1, -1)

        freqs_i = torch.cat([
            freqs_t.expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # --- SP: pad to full distributed length, then slice this rank's portion ---
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        s_per_rank = s
        freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) *
                                                       s_per_rank), :, :]
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


def sp_dit_forward(self, x, t, context, seq_len, y=None, memory_size=None, time_indices=None, **extra_kwargs):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    if self.model_type == 'i2v':
        assert y is not None

    # 提取 identity 参数
    identity_frames_caption = extra_kwargs.get('identity_frames_caption', None)
    identity_frame_num = extra_kwargs.get('identity_frame_num', [0])
    enable_multi_identity_train = extra_kwargs.get('enable_multi_identity_train', False)
    split_identity_attn = extra_kwargs.get('split_identity_attn', None)
    _split_identity_attn = split_identity_attn if split_identity_attn is not None else getattr(self, 'split_identity_attn', False)

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

    # 计算 identity_seq_lens (patch embedding 之后、flatten 之前)
    identity_seq_lens = torch.tensor(
        [id_frame * u.shape[3] * u.shape[4] for id_frame, u in zip(identity_frame_num, x)],
        dtype=torch.long)

    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    # time embeddings
    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    with torch.amp.autocast('cuda', dtype=torch.float32):
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim,
                                    t).unflatten(0, (bt, seq_len)).float())
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    # identity context
    identity_context = None
    if identity_frames_caption is not None:
        if enable_multi_identity_train:
            identity_context = []
            for cap in identity_frames_caption:
                padded = torch.cat([cap, cap.new_zeros(self.text_len - cap.size(0), cap.size(1))])
                embedded = self.text_embedding(padded.unsqueeze(0))
                identity_context.append(embedded)
        else:
            identity_context = self.text_embedding(
                torch.stack([
                    torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in identity_frames_caption
                ]))

    # Context Parallel
    x = torch.chunk(x, get_world_size(), dim=1)[get_rank()]
    e = torch.chunk(e, get_world_size(), dim=1)[get_rank()]
    e0 = torch.chunk(e0, get_world_size(), dim=1)[get_rank()]

    # 计算 local identity_seq_lens (cross-attn 在 chunked x 上需要 local 值)
    chunk_size = x.shape[1]
    sp_rank = get_rank()
    chunk_start = sp_rank * chunk_size
    local_identity_seq_lens = torch.tensor(
        [max(0, min(int(id_len.item()) - chunk_start, chunk_size)) for id_len in identity_seq_lens],
        dtype=torch.long)

    # 在每个 block 的 self_attn 上存储 global identity_seq_lens (sp_attn_forward 使用)
    for block in self.blocks:
        block.self_attn._sp_global_identity_seq_lens = identity_seq_lens

    # 当本 rank 没有 identity tokens 时，不传 identity_context，避免 cross-attn 对空 tensor 做 attention
    local_identity_context = identity_context if local_identity_seq_lens.sum().item() > 0 else None

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
        identity_context=local_identity_context,
        identity_seq_lens=local_identity_seq_lens,
        enable_multi_identity_train=enable_multi_identity_train,
        identity_frame_num=identity_frame_num,
        split_identity_attn=_split_identity_attn,
    )
    for block in self.blocks:
        x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # Context Parallel
    x = gather_forward(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]


def sp_attn_forward(self, x, seq_lens=None, grid_sizes=None, freqs=None,
                    memory_size=0, time_indices=None,
                    identity_seq_lens=None, split_identity_attn=False, **kwargs):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    q = rope_apply(q, grid_sizes, freqs, memory_size, time_indices)
    k = rope_apply(k, grid_sizes, freqs, memory_size, time_indices)

    def half(t):
        return t if t.dtype in half_dtypes else t.to(torch.bfloat16)

    # 读取 sp_dit_forward 中存储的全局 identity_seq_lens
    global_identity_seq_lens = getattr(self, '_sp_global_identity_seq_lens', None)

    if (split_identity_attn and global_identity_seq_lens is not None
            and global_identity_seq_lens[0].item() > 0 and memory_size > 0):
        q_full = all_to_all(q, scatter_dim=2, gather_dim=1)
        k_full = all_to_all(k, scatter_dim=2, gather_dim=1)
        v_full = all_to_all(v, scatter_dim=2, gather_dim=1)

        assert b == 1
        id_len = global_identity_seq_lens[0].item()
        total_valid = seq_lens[0].item()
        F_total, H, W = grid_sizes[0].tolist()
        hw = int(H * W)
        mem_len = memory_size * hw
        gen_len = total_valid - id_len - mem_len

        # SA1: identity + memory
        sa1_end = id_len + mem_len
        q1, k1, v1 = q_full[:, :sa1_end], k_full[:, :sa1_end], v_full[:, :sa1_end]
        sa1_lens = torch.tensor([sa1_end], device=x.device, dtype=seq_lens.dtype)

        # SA2: memory + generation
        q2, k2, v2 = q_full[:, id_len:total_valid], k_full[:, id_len:total_valid], v_full[:, id_len:total_valid]
        sa2_lens = torch.tensor([mem_len + gen_len], device=x.device, dtype=seq_lens.dtype)

        out1 = flash_attention(q=q1, k=k1, v=v1, k_lens=sa1_lens, window_size=self.window_size)
        out2 = flash_attention(q=q2, k=k2, v=v2, k_lens=sa2_lens, window_size=self.window_size)

        # 拆分各段输出
        id_out = out1[:, :id_len]
        mem_out_1 = out1[:, id_len:sa1_end]
        mem_out_2 = out2[:, :mem_len]
        gen_out = out2[:, mem_len:]

        mem_out = 0.5 * mem_out_1 + 0.5 * mem_out_2

        # 重新拼装完整序列
        L_full = q_full.shape[1]
        n_local = q_full.shape[2]
        result = out1.new_zeros(b, L_full, n_local, d)
        result[:, :id_len] = id_out
        result[:, id_len:id_len + mem_len] = mem_out
        result[:, id_len + mem_len:total_valid] = gen_out

        x = all_to_all(result, scatter_dim=1, gather_dim=2)
    else:
        q_full = all_to_all(q, scatter_dim=2, gather_dim=1)
        k_full = all_to_all(k, scatter_dim=2, gather_dim=1)
        v_full = all_to_all(v, scatter_dim=2, gather_dim=1)
        x = flash_attention(q_full, k_full, v_full, k_lens=seq_lens, window_size=self.window_size)
        x = all_to_all(x, scatter_dim=1, gather_dim=2)

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x
