import torch
import torch.nn.functional as F
from torch import nn
import triton
import triton.language as tl

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except ImportError:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def _repeat_kv_for_gqa(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    if x.size(1) == num_heads:
        return x
    repeat = num_heads // x.size(1)
    return x.repeat_interleave(repeat, dim=1)


def _gather_cache(cache: torch.Tensor, block_table: torch.Tensor, seqlen: int) -> torch.Tensor:
    block_size = cache.size(1)
    block_table = block_table[: (seqlen + block_size - 1) // block_size]
    x = cache.index_select(0, block_table).reshape(-1, cache.size(2), cache.size(3))
    return x[:seqlen]


def _attention_one(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, num_heads: int, causal: bool) -> torch.Tensor:
    q = q.transpose(0, 1).unsqueeze(0)
    k = _repeat_kv_for_gqa(k.transpose(0, 1).unsqueeze(0), num_heads)
    v = _repeat_kv_for_gqa(v.transpose(0, 1).unsqueeze(0), num_heads)
    o = F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=causal)
    return o.squeeze(0).transpose(0, 1)


def _attention_one_with_prefix(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, num_heads: int) -> torch.Tensor:
    q_len, k_len = q.size(0), k.size(0)
    q = q.transpose(0, 1).unsqueeze(0)
    k = _repeat_kv_for_gqa(k.transpose(0, 1).unsqueeze(0), num_heads)
    v = _repeat_kv_for_gqa(v.transpose(0, 1).unsqueeze(0), num_heads)
    q_pos = torch.arange(k_len - q_len, k_len, device=q.device).view(q_len, 1)
    k_pos = torch.arange(k_len, device=q.device).view(1, k_len)
    mask = k_pos <= q_pos
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    return o.squeeze(0).transpose(0, 1)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            if flash_attn_varlen_func is not None:
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables)
            else:
                outputs = []
                cu_q = context.cu_seqlens_q.tolist()
                cu_k = context.cu_seqlens_k.tolist()
                for i in range(len(cu_q) - 1):
                    q_i = q[cu_q[i]:cu_q[i + 1]]
                    if context.block_tables is None:
                        k_i = k[cu_k[i]:cu_k[i + 1]]
                        v_i = v[cu_k[i]:cu_k[i + 1]]
                        outputs.append(_attention_one(q_i, k_i, v_i, self.scale, self.num_heads, True))
                    else:
                        k_i = _gather_cache(k_cache, context.block_tables[i], cu_k[i + 1] - cu_k[i])
                        v_i = _gather_cache(v_cache, context.block_tables[i], cu_k[i + 1] - cu_k[i])
                        outputs.append(_attention_one_with_prefix(q_i, k_i, v_i, self.scale, self.num_heads))
                o = torch.cat(outputs, dim=0)
        else:    # decode
            if flash_attn_with_kvcache is not None:
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables,
                                            softmax_scale=self.scale, causal=True)
            else:
                outputs = []
                for i in range(q.size(0)):
                    seqlen = int(context.context_lens[i])
                    k_i = _gather_cache(k_cache, context.block_tables[i], seqlen)
                    v_i = _gather_cache(v_cache, context.block_tables[i], seqlen)
                    outputs.append(_attention_one(q[i:i + 1], k_i, v_i, self.scale, self.num_heads, False))
                o = torch.cat(outputs, dim=0).unsqueeze(1)
        return o
