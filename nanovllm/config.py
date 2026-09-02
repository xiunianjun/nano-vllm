import os
from dataclasses import dataclass, field
from enum import Enum
from transformers import AutoConfig


class KVCachePolicy(Enum):
    GPU_RECOMPUTE = (False, False, False, "gpu_prefix_cache_recompute_baseline")
    GPU_LRU = (False, True, False, "gpu_prefix_cache_recompute_baseline")
    CPU_EAGER = (True, False, False, "cpu_prefix_cache_v1")
    CPU_EAGER_GPU_LRU = (True, True, False, "cpu_prefix_cache_v2_lru")
    CPU_LAZY_GPU_LRU = (True, True, True, "cpu_prefix_cache_v3_lazy_writeback")

    def __init__(self, cpu_offload: bool, gpu_lru: bool, lazy_writeback: bool, benchmark_mode: str):
        self.cpu_offload = cpu_offload
        self.gpu_lru = gpu_lru
        self.lazy_writeback = lazy_writeback
        self.benchmark_mode = benchmark_mode

    @classmethod
    def from_flags(cls, cpu_offload: bool, gpu_lru: bool, lazy_writeback: bool):
        if not cpu_offload:
            return cls.GPU_LRU if gpu_lru else cls.GPU_RECOMPUTE
        if not gpu_lru:
            return cls.CPU_EAGER
        return cls.CPU_LAZY_GPU_LRU if lazy_writeback else cls.CPU_EAGER_GPU_LRU


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
    # V3 bounded CPU cache: prefer evicting redundant GPU-resident CPU copies.
    # Disable only for the pure-CPU-LRU ablation.
    enable_gpu_aware_cpu_eviction: bool = True
    lazy_writeback_watermark_ratio: float = 0.5
    # > 0 时直接指定安全 victim window；0 保持按 max_num_batched_tokens 推导的旧行为。
    lazy_writeback_target_blocks: int = 0
    # <= 0 表示不限制 CPU prefix cache；> 0 时按 LRU 淘汰 CPU backing，后续 miss 只能 recompute。
    cpu_prefix_cache_gb_limit: float = 0.0
    # > 0 时预分配固定容量 pinned CPU KV block pool，避免 writeback 热路径临时 pin_memory 分配。
    cpu_prefix_pool_gb: float = 0.0
    kv_cache_policy: KVCachePolicy = field(init=False)

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.lazy_writeback_watermark_ratio >= 0
        assert self.lazy_writeback_target_blocks >= 0
        assert self.cpu_prefix_cache_gb_limit >= 0
        assert self.cpu_prefix_pool_gb >= 0
        self.kv_cache_policy = KVCachePolicy.from_flags(
            self.enable_cpu_kv_offload,
            self.enable_gpu_lru_retention,
            self.enable_lazy_cpu_kv_writeback,
        )
        if self.enable_cpu_kv_offload and self.cpu_prefix_cache_gb_limit > 0:
            # A configured CPU cache limit is also a physical pinned-memory cap.
            # Use a fixed pool so the writeback path cannot allocate beyond it.
            if self.cpu_prefix_pool_gb == 0:
                self.cpu_prefix_pool_gb = self.cpu_prefix_cache_gb_limit
            assert self.cpu_prefix_pool_gb <= self.cpu_prefix_cache_gb_limit
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
