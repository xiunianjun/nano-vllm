import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    enable_prefix_cache: bool = True
    # 打开后启用 V1 GPU+CPU 两级 prefix cache；权重仍常驻 GPU，只 offload prefix KV blocks。
    enable_cpu_kv_offload: bool = False
    # V2: request 结束后把 GPU prefix blocks 留在 inactive LRU cache 中，直到显存需要时再淘汰。
    enable_gpu_lru_retention: bool = True
    # V3: 不再 prefill 后全量写回 CPU，只维护一段可安全淘汰的 CPU-backed inactive window。
    enable_lazy_cpu_kv_writeback: bool = False
    lazy_writeback_watermark_ratio: float = 0.5
    # <= 0 表示不限制 CPU prefix cache；> 0 时按 LRU 淘汰 CPU backing，后续 miss 只能 recompute。
    cpu_prefix_cache_gb_limit: float = 0.0

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.lazy_writeback_watermark_ratio >= 0
        assert self.cpu_prefix_cache_gb_limit >= 0
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
