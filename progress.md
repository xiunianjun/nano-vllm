# nano-vLLM GPU+CPU Prefix Cache 进展

## 研究方向

目标：把 nano-vLLM 现有 GPU prefix cache 扩展为 GPU + CPU 两级 prefix cache。

请求查找 prefix KV 时：

```text
GPU hit              -> 直接复用 GPU KV
GPU miss + CPU hit   -> H2D restore 到 GPU，跳过 prefix prefill
GPU miss + CPU miss  -> 正常 prefill / recompute
```

约束：模型权重始终完整驻留 GPU，只人为限制 GPU KV cache capacity；第一阶段不考虑 SSD、不引入 ShareGPT、不把 preemption-driven swapping 作为主实验。

实现上需要贴合 nano-vLLM 当前 block 抽象：`BlockManager` 管理的是 logical block，一个 logical block id 对应所有 layer 的 K/V slice。

```text
kv_cache shape = [K/V][layer][block_id][token_offset][kv_head][head_dim]
Qwen3-8B: 1 logical block = 256 tokens = 36 MiB 全层 KV
```

## 环境

```text
GPU: NVIDIA H200 NVL
Model: /data/datasets/models-hf/Qwen3-8B
venv: .venv-fa28
Python: 3.10.19
Torch: 2.8.0+cu128
FlashAttention: 2.8.3.post1
```

说明：`.venv-fa28` 中已有匹配的 `flash-attn` wheel，后续 benchmark 优先使用该环境，避免本地编译。

## 代码入口

```text
bench_long_doc_qa.py
scripts/run_prefix_cache_cases.sh
```

`bench_long_doc_qa.py` 负责生成 synthetic long-doc / branching workload，并输出 JSON metrics。`scripts/run_prefix_cache_cases.sh` 负责批量跑 baseline / V1 / V2 / V3，并为每个 case 生成 `summary.json`。

## V1 实现

V1 已实现 CPU prefix cache backing store，重点是 correctness 和可观测性。

核心路径：

- prompt prefill 完成后，完整 prefix blocks 立刻 async D2H 写回 CPU。
- decode 新生成 tokens 暂不主动纳入 prefix cache；后续请求重新 prefill 成完整 block 后再进入 cache。
- 写回前用 block hash + token ids 去重：CPU 已有或正在 pending writeback 时跳过。
- waiting request 调度时查最长连续 prefix：GPU hit 直接复用，CPU hit 分配 GPU block 并同步 H2D restore，miss 后剩余 token 正常 prefill。
- pending writeback 的 GPU block 会被保护，D2H 完成后再释放；V1 选择优先保证正确性。

已验证 smoke test：

| 场景 | 结果 |
|---|---|
| 重复 prompt 写回 | 第二次请求不会重复 D2H，同一 prefix block 写回去重生效 |
| `D0 -> D1 -> D0` 小 GPU cache | 第三次 `D0` GPU miss + CPU hit，可 H2D restore 并跳过 prefix prefill |

## V2 实现：GPU LRU Retention

V2 已实现。它不做调度感知预取，目标是把 V1 中“request 结束后 GPU KV 是否还能复用”的随缘行为，改成明确的 inactive GPU prefix cache policy。

核心语义：

```text
active block   = 正在被 running / waiting request 引用，不能淘汰
inactive block = request 已结束或释放引用，KV 仍在 GPU，hash 仍有效，可作为 GPU prefix cache 命中
free block     = 没有有效 cache 内容，或已经被 LRU eviction 选中可覆盖
```

V2 路径：

- request prefill 完成后仍按 V1 主动 async D2H 写回 CPU backing。
- request finish 后，完整 prefix blocks 不立刻进入普通 free list，而是进入 inactive GPU LRU。
- 释放同一个 request 的 blocks 时按逆序入队：tail prefix 先进入 LRU/free queue，在相同 recency 下更早被淘汰；靠近 prefix 根部的 block 更通用，因此尽量留久一点。这个策略对齐 vLLM 的 prefix-aware LRU 细节。
- 新 request prefix lookup 时，如果命中 inactive block，则从 LRU 中移出并重新变成 active GPU hit。
- 需要新 GPU block 时，优先使用真正空闲 block；不够时从 inactive LRU 端驱逐 victim；仍不够时才进入现有 preemption 路径。
- eviction 优先选择已经 CPU-backed 的 inactive block；没有 CPU backing 的 inactive block 只有在空间仍不足时才被淘汰。
- pending writeback 已改成 block 粒度：已完成 D2H 的 block 可以先进入 CPU-backed LRU/可淘汰集合，未完成 D2H 的 block 继续 protected。

V2 重点指标：

```text
gpu_prefix_miss_request_count
cpu_sync_swapin_request_count
cpu_sync_swapin_block_count
cpu_sync_swapin_token_count
cpu_prefix_restore_latency_sum
gpu_lru_hit_block_count
gpu_lru_hit_token_count
gpu_lru_eviction_count
gpu_lru_cached_block_count
gpu_lru_cached_block_peak
```

预期收益：相比 V1，V2 不是 overlap H2D，而是减少同步 swapin 次数、H2D bytes 和 H2D latency，同时提升 GPU prefix reuse。

## V3 设计：内存感知 Lazy Writeback

V3 在 V2 LRU 上只做一个优化：memory-aware selective writeback。

核心变化：不再 prefill 后全量写回 CPU，而是只给最可能被回收的 inactive GPU blocks 做 lazy D2H。动机是 CPU cache 也有限，V1/V2 的全量 backing 会让大量“GPU 上已经保留、短期不需要 CPU 副本”的 KV 占住 CPU 空间，反而挤掉更有价值的 prefix，甚至迫使它们被丢弃或下沉到更慢层级。V3 希望把 CPU 空间优先留给即将失去 GPU 副本、且未来仍可能复用的 KV，同时减少不必要的 PCIe D2H 流量。

Lazy writeback 的触发时机参考 vLLM simple CPU offload 的 safety-window 思路，但当前 nano-vLLM 实现更保守：只在一次 allocation 之后发现 free blocks 低于目标水位时，才扫描 inactive LRU 前沿并补一批异步 D2H。这样避免每个 decode step 都做 lazy window 维护。

```text
scheduler step
  -> allocation 真的发生
  -> free blocks < target_free
  -> 扫描 inactive LRU victim window
  -> 挑选 GPU-only victim blocks 异步写回 CPU
```

核心阈值也参考 vLLM，不写死固定 block 数，而是按单轮 scheduler step 可能新增的最大 KV blocks 估算 safety window：

```text
target_blocks = ceil(max_num_batched_tokens / block_size)
target_free   = target_blocks * (1 + lazy_writeback_watermark_ratio)
```

vLLM 当前 lazy CPU offload 使用 `lazy_writeback_watermark_ratio = 1.0`，即大约保留 `2 * ceil(max_num_batched_tokens / block_size)` 个 already-backed inactive/free blocks。我们的场景里 `max_num_batched_tokens` 可能为了长 prompt 设得偏宽，直接用 `1.0` 会比较激进，容易提前复制过多 KV 到 CPU。因此 V3 不再只看单个默认值，而是把 watermark 当成实验变量。

当 already-backed window 不足时，从 inactive LRU 的 eviction end 选择 GPU-only blocks，批量异步 D2H。这样 GPU 需要腾空间时，可以优先淘汰 already-backed blocks，避免在关键路径上同步 D2H；同时又不会像 V1/V2 那样把所有 prefix 都常驻 CPU。

实现上新增 pinned CPU block pool：设置 `cpu_prefix_pool_gb > 0` 后，初始化阶段预分配固定数量的 pinned CPU KV blocks，writeback 热路径只从 pool 取 buffer，不再临时 `torch.empty(..., pin_memory=True)`。CPU LRU 淘汰时释放的是有效 backing，并把 pooled buffer 还回 free list。

### V3 性能 Bug 记录

早期 V3 lazy writeback 虽然使用异步 D2H，但在 scheduler 热路径里逐 block 临时分配 pinned CPU tensor，导致 `schedule_time` 从 V2 的约 `0.54s` 放大到数秒级。profile 后确认主要开销来自 `torch.empty(..., pin_memory=True)`，而不是 CUDA copy 本身；改为初始化阶段预分配 `cpu_prefix_pool_gb` 后，V3 的调度开销基本回到 V2 同级。后续实验需要同时报告有效 CPU cache 占用和预分配 pool 容量，避免把 reserved memory 误读成实际 cached KV。

### 热路径优化（2026-09-02）

- decode admission 用可驱逐 block 计数替代每 token 扫描 inactive LRU。
- decode 阶段复用 pinned CPU/CUDA buffers；CUDA Graph 直接填充 graph backing buffers。
- writeback pending record 改为具名结构和直接索引，allocator pending 状态收敛到 `Block`，移除重复 active/pending sets；哈希命中后的 token 比对继续保留碰撞保护。
- Qwen3-0.6B 短测中吞吐从 `10.0350` 提升到 `10.1480 req/s`（`+1.13%`），query elapsed 从 `0.7972s` 降至 `0.7883s`（`-1.11%`）；输出/trace 哈希与 cache 行为一致。该短测使用 `--enforce-eager`，未覆盖 CUDA Graph 部分收益。
- D2H writeback 改为每批共享 CUDA events（event 数从 `1+2N` 降为固定 `3`）；同参数确认轮中 event/submit/lazy-maintain 局部耗时分别下降约 `14%/19%/15%`，但端到端吞吐比上一轮低 `0.44%`，短测尚不能证明整体收益。
- `Sequence` 缓存连续完整 block 的 chained hash，prefix lookup 与 `hash_blocks` 复用同一结果，避免 miss/retry 路径重复构造 NumPy 数组并计算 xxhash；token_ids 比对仍作为 hash collision 保护。
- 三个兼容 CLI 开关在配置入口统一映射为 `KVCachePolicy`，Scheduler、BlockManager、LLMEngine 和 benchmark mode 不再各自重复拼接 V1/V2/V3 条件；无效组合按既有语义归一化，旧脚本无需修改。
- 第 7/8 项用相同 Qwen3-0.6B 短测确认两轮：平均吞吐 `9.9529 req/s`、query elapsed `0.8038s`，相对第 5 项确认轮分别为 `-1.49%/+1.51%`；schedule time 平均反而小幅下降 `0.29%`，波动来自 model runner time 增加 `1.54%`。因此本轮结论是调度侧重构正确但端到端无可证明提速；两轮输出/trace 哈希及 GPU hit、CPU restore、eviction、writeback 数均完全一致。

V3 block 状态：

```text
active GPU                -> running/waiting request 正在引用
inactive GPU only         -> 可 GPU hit，但还没有 CPU backing
inactive GPU + CPU backed -> 可 GPU hit，也可无痛 evict GPU
pending writeback         -> D2H 未完成，暂时不能覆盖
CPU only                  -> GPU miss 后可 demand restore
dropped                   -> CPU/GPU 都没有，后续只能 recompute
```

需要新增配置：

```text
lazy_writeback_watermark_ratio = 0.5
cpu_prefix_cache_gb_limit
cpu_prefix_pool_gb
```

后续扫描 `lazy_writeback_watermark_ratio = 0 / 0.25 / 0.5 / 1.0`。`0` 只覆盖一轮最大 allocation demand，`1.0` 对齐 vLLM 的 100% watermark；中间值用于观察 CPU 占用和同步 swapin 的折中。暂不把 `2.0` 作为主扫描点，除非需要专门验证更保守窗口对 sync swapin 的上限收益。

关键指标：

```text
inactive_cpu_backed_block_count
inactive_gpu_only_block_count
victim_window_safe_or_pending_block_count
lazy_writeback_scheduled/completed_block_count
evict_already_backed_block_count
evict_gpu_only_sync_writeback_block_count
evict_gpu_only_drop_block_count
cpu_cache_evicted_block_count
cpu_prefix_cache_live_gb / cpu_prefix_cache_live_gb_peak
cpu_prefix_pool_reserved_gb
cpu_prefix_pool_free_block_count / cpu_prefix_pool_used_block_count
cpu_prefix_pool_on_demand_alloc_count
```

内存指标语义：`live_gb` 是当前真正有效、可命中的 CPU prefix backing；`reserved_gb` 是预分配 pinned pool 的物理容量。论文图表优先看 `live_gb/live_gb_peak`，工程排障同时看 `reserved_gb` 和 `on_demand_alloc_count`。

V3 的 benchmark 不只看速度，还要看 memory-latency tradeoff：在相同 cache pressure 下，扫描 `lazy_writeback_watermark_ratio` 和 `cpu_prefix_cache_gb_limit`，找到 CPU memory 明显低于 V2、但 sync swapin / TTFT / request latency 接近 V2 的参数区间。`cpu_prefix_cache_gb_limit = 0` 表示不限制 CPU cache，用作上界对照。

专用脚本：

```text
scripts/run_v3_memory_sweep.sh
```

默认脚本现在固定 `watermark = 0.5`，扫描 `CPU limit = 20 / 12 / 11 / 10.5 / 10 / 5 / 3 / 2 / 1 GB`；其它 watermark 只在需要复查 GPU safety window 时手动覆盖。本轮不再做 document length sweep，优先固定 `document_length = 8192`；同时把 `target_working_set_gb` 和 `gpu_kv_cache_gb` 放到 `20.0 / 8.0`，保持 working set/GPU 约 `2.5x`、单 prompt/GPU 约 `14%`。这样 cache pressure 主要来自更多 documents，而不是少量超长 request。`cpu_prefix_pool_gb` 需要覆盖实验可能达到的 live backing 峰值并按 block 粒度留余量；例如本轮 actual working set 是 20.25GB，设置 20GB 会出现少量 on-demand allocation，21GB 才能完全走 pool。

## V4 设计：调度感知 Prefetch / OPT Eviction

V4 再引入 scheduler-aware 优化，不和 V3 的内存优化混在一个版本里。

计划方向：

- scheduler inspect waiting/running 队列，提前恢复即将使用的 CPU-resident prefix，尽量把 H2D restore 从关键路径移走。
- eviction 从朴素 LRU 升级为近似 OPT：在当前可见调度窗口内，优先驱逐最晚才会再次访问、或不会再访问的 inactive block。
- `sync_swapin` 指标只统计 GPU miss、CPU hit、且预取没赶上导致仍在关键路径同步 H2D 的情况；预取成功不计入 sync swapin。
- 新增 prefetch accuracy 指标，例如 prefetch true positive、false positive、false negative，用于观察预取取多了还是取少了。

## Workload Cases

当前参考 LMCache Long-Document QA 的 workload 语义，但文档内容使用 synthetic token IDs。warmup 阶段访问所有 reusable prefixes，不计入最终指标；measured 阶段访问相同 prefix + 不同 query suffix。

| case | 目的 | 访问特征 |
|---|---|---|
| `case0_functional` | 单文档功能性校验 | 同一 document 做两次 QA，第二次应命中 GPU prefix cache |
| `cascade_tile` | 级联污染 / cache thrashing sanity | warmup `D0,D1,...`，query 再从 `D0` 开始；working set 大于 GPU cache 时容易连续 miss |
| `hot_cold_sharing` | 冷热 document prefix sharing | hot documents 高频访问，cold documents 低频访问；同时出现 GPU hit 和 GPU miss |
| `branching_prefix_sharing` | 分叉 request 部分共享 | 多个 branch 共享同一个 root prefix，同时各自有不同 branch suffix |

## V2 Benchmark 参数原则

V2 LRU 要测的是 GPU inactive prefix cache 是否能减少 CPU swapin，而不是单个超长 request 把 GPU KV 撑爆。因此后续主实验按下面原则调参：

大白话：用更多不同 document 扩大总 working set，同时降低单个 request 的 KV 占比。不要主要依赖少量超长 document 制造 cache pressure。

需要同时控制两个比例：

```text
single_prompt_KV / GPU_KV
working_set_KV / GPU_KV
```

含义：

- `single_prompt_KV / GPU_KV`：一个 request 的 prompt/query/decode 最多占多少 GPU KV。它决定 continuous batching 是否还有空间、inactive LRU 是否有空间留住历史 prefix。目标先放在 `0.25-0.4`。
- `working_set_KV / GPU_KV`：所有可复用 document prefix 的总 KV 量相对 GPU KV cache 的大小。它决定是否有长期 cache pressure。目标先放在 `2-4`。

调参顺序：

1. 固定 `document_length >= 4096`，但避免单个 prompt 占据 GPU KV 的大部分。
2. 根据目标 `single_prompt_KV / GPU_KV` 选择 `gpu_kv_cache_gb`。
3. 根据目标 `working_set_KV / GPU_KV` 自动计算 `num_documents`，优先增加 document 数量，而不是继续拉长 document。
4. 使用 Poisson arrival、`max_num_seqs=8`、合适的 `max_num_batched_tokens` 跑 serving 场景，确认确实在 continuous batching 下比较 recompute baseline 与 V2 LRU。

粗略公式：

```text
single_prompt_KV ~= (document_length + query_length + output_len) * kv_bytes_per_token
working_set_KV ~= num_documents * reusable_prefix_length * kv_bytes_per_token
```

当前参数检查：Qwen3-8B bf16 的 KV 约为 `144 MiB / 1K tokens`。旧版 `gpu_kv_cache_gb=1.1` 实际只有约 `1.09 GB` KV blocks，因此单请求占比偏高：

| document length | single prompt KV | single_prompt_KV / GPU_KV | 粗略可同时容纳完整 prompt 数 |
|---:|---:|---:|---:|
| 4096 | 0.578 GB | 0.53 | 1 |
| 6144 | 0.859 GB | 0.79 | 1 |
| 7680 | 1.070 GB | 0.98 | 1 |

这个配置可以触发 recompute/restore，但不适合作为 V2 LRU 主实验：GPU 几乎没有空间同时容纳 active requests 和 inactive prefix，LRU 会退化成频繁 swapin/swapout。

后续 benchmark 需要把 `num_documents` 作为由 working set ratio 自动推导的参数，而不是写死。`bench_long_doc_qa.py` 已在 JSON 中输出 `working_set_to_gpu_kv_ratio`、`single_prompt_to_gpu_kv_ratio` 和 `single_prompt_fit_count_est`，用于检查每轮参数是否落在目标范围。

建议先保留两个主档位：

| profile | 目的 | 建议参数 |
|---|---|---|
| `moderate_lru` | 干净验证 V2 LRU 是否减少同步 swapin | `document_length=4096`, `gpu_kv_cache_gb≈2.0`, `target_working_set_gb≈6.0`, `max_num_seqs=8`, `max_num_batched_tokens=4*(doc+query)`, `request_rate=0.5-1.0 req/s` |
| `serving_concurrency` | 更接近 serving 的 continuous batching + cache pressure | `document_length=4096/6144`, `gpu_kv_cache_gb≈4.0`, `target_working_set_gb≈12.0`, `max_num_seqs=8`, `max_num_batched_tokens=4*(doc+query)`, `request_rate` 做 sweep |

`moderate_lru` 的关键是单请求占比约 0.3，GPU 能同时放下多个 active prompt，并留下 inactive LRU 空间；`serving_concurrency` 则用更大的 GPU KV budget 承接并发，同时通过更多 documents 维持 `working_set_KV / GPU_KV ~= 3`。旧版 `gpu_kv_cache_gb=1.1, target_working_set_gb=1.5` 只保留为 capacity-pressure sanity，不作为 V2 主实验配置。

## 实验结果

### 当前可复用结果：8K Serving 对比

结果目录：

```text
exp/prefix_cache_serving_20260901_181816/
```

这轮结果是当前主口径：Poisson arrival、continuous batching、8K document prefix，同时包含 recompute baseline / V1 / V2 / V3。后续图表和结论优先使用这一轮。

基础配置：

```text
model = /data/datasets/models-hf/Qwen3-8B
runs = 2
document_length = 8192
query_length = 96
output_len = 16
target_working_set_gb = 20.0
gpu_kv_cache_gb = 8.0
max_num_seqs = 8
max_num_batched_tokens = 33152
arrival_mode = poisson
request_rate = 2 req/s
lazy_writeback_watermark_ratio = 0.5
cpu_prefix_cache_gb_limit = 0
```

参数检查：单 prompt 约占 GPU KV 的 `14.3%`，估算可同时容纳 `6` 个完整 prompt；working set/GPU 约 `2.54x`。V3 lazy target 为 `195 blocks`，小于实际 `227 blocks`，不再出现旧版 target window 大于总 GPU blocks 的问题。

图表：

![Measured prefill total time](exp/prefix_cache_serving_20260901_181816/figures/prefill_total.svg)

![Median TTFT](exp/prefix_cache_serving_20260901_181816/figures/ttft_median.svg)

![Synchronous CPU restore blocks](exp/prefix_cache_serving_20260901_181816/figures/sync_swapin_blocks.svg)

![Peak CPU prefix KV memory](exp/prefix_cache_serving_20260901_181816/figures/cpu_memory_peak.svg)

核心结果为 2 次运行均值。TTFT/request/queueing 是 measured requests 的 per-request 分布统计；`prefill total` 是 measured 阶段所有 prefill step 的总 wall time。

| case | mode | docs / requests | recompute tok | CPU restore tok | sync swapin blocks | GPU LRU hit tok | CPU peak | prefill total | median TTFT | min TTFT | median req latency | queue avg / max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case0 | baseline | 1 / 1 | 0 | 0 | 0 | 8,192 | 0.00 GB | 0.066s | 0.103s | 0.103s | 0.584s | 0.000 / 0.000s |
| case0 | V1 | 1 / 1 | 0 | 8,192 | 32 | 0 | 1.12 GB | 0.073s | 0.100s | 0.100s | 0.476s | 0.000 / 0.000s |
| case0 | V2 | 1 / 1 | 0 | 0 | 0 | 8,192 | 1.12 GB | 0.050s | 0.079s | 0.079s | 0.467s | 0.000 / 0.000s |
| case0 | V3 | 1 / 1 | 0 | 0 | 0 | 8,192 | 0.00 GB | 0.049s | 0.076s | 0.076s | 0.457s | 0.000 / 0.000s |
| cascade | baseline | 18 / 18 | 147,456 | 0 | 0 | 0 | 0.00 GB | 4.507s | 0.420s | 0.277s | 1.180s | 0.066 / 0.236s |
| cascade | V1 | 18 / 18 | 0 | 147,456 | 576 | 0 | 20.25 GB | 1.106s | 0.111s | 0.085s | 0.521s | 0.013 / 0.051s |
| cascade | V2 | 18 / 18 | 0 | 147,456 | 576 | 0 | 20.25 GB | 1.136s | 0.104s | 0.085s | 0.548s | 0.013 / 0.063s |
| cascade | V3 | 18 / 18 | 49,664 | 97,792 | 382 | 0 | 14.59 GB | 7.157s | 2.205s | 0.221s | 2.609s | 1.302 / 3.749s |
| hot/cold | baseline | 18 / 72 | 148,608 | 0 | 0 | 355,200 | 0.00 GB | 6.307s | 0.081s | 0.061s | 0.613s | 0.021 / 0.257s |
| hot/cold | V1 | 18 / 72 | 0 | 516,096 | 2,016 | 0 | 20.25 GB | 3.958s | 0.095s | 0.063s | 0.526s | 0.011 / 0.057s |
| hot/cold | V2 | 18 / 72 | 0 | 147,968 | 578 | 359,936 | 20.25 GB | 2.923s | 0.080s | 0.062s | 0.510s | 0.009 / 0.031s |
| hot/cold | V3 | 18 / 72 | 49,920 | 115,584 | 452 | 264,576 | 19.12 GB | 8.081s | 0.129s | 0.062s | 0.830s | 0.279 / 2.800s |
| branching | baseline | 35 / 140 | 90,240 | 0 | 0 | 614,272 | 0.00 GB | 7.550s | 0.081s | 0.061s | 0.537s | 0.013 / 0.160s |
| branching | V1 | 35 / 140 | 0 | 681,984 | 2,664 | 0 | 20.25 GB | 7.203s | 0.093s | 0.062s | 0.537s | 0.011 / 0.062s |
| branching | V2 | 35 / 140 | 0 | 90,240 | 352 | 599,936 | 20.25 GB | 5.471s | 0.079s | 0.060s | 0.507s | 0.011 / 0.047s |
| branching | V3 | 35 / 140 | 46,336 | 43,904 | 172 | 546,688 | 16.88 GB | 9.820s | 0.090s | 0.061s | 0.579s | 0.036 / 0.777s |

对比结论：

- `case0_functional` 通过：同一 document 第二次访问能复用 prefix。V3 没有写 CPU 是合理的，因为单文档一直留在 GPU，不需要 backing。
- `cascade_tile` 是 LRU 无效边界 case。V1/V2 都需要完整同步 restore；V3 可以少存约 `5.66 GB` CPU KV，但因为部分 prefix 没有 backing，会回到 recompute，TTFT 和排队尾延迟明显变差。
- `hot_cold_sharing` 是主目标 case。V2 相比 V1 把同步 swapin blocks 从 `2016` 降到 `578`，约 `71%`；prefill total 从 `3.96s` 降到 `2.92s`。V3 只省约 `1.12 GB` CPU，但引入 `49,920` recompute tokens，当前 watermark=0.5 不是好参数点。
- `branching_prefix_sharing` 也符合预期。V2 相比 V1 把同步 swapin blocks 从 `2664` 降到 `352`，约 `87%`；prefill total 从 `7.20s` 降到 `5.47s`。V3 省 `3.38 GB` CPU，但同样产生 recompute，需要后续 sweep 找折中。
- 当前 V2 的性能提升数字是可信的：queueing latency 整体较低，说明这轮不是旧版一次性批量提交造成的大排队；收益主要来自减少关键路径 CPU restore 和增加 GPU LRU hit。
- 当前 V3 的功能逻辑可用，但默认参数不是最终结论。下一轮 V3 应扫描 watermark 和 CPU limit，用 memory-latency tradeoff 选点，而不是只看 `0.5 / unlimited CPU`。

### 旧结果处理

以下结果不再用于当前论文/简历项目主结论：

```text
exp/prefix_cache_serving_20260901_154339/
exp/prefix_cache_serving_20260901_181056/
exp/v3_memory_sweep_20260901_160936/
exp/cpu_kv_memory_doclen_sweep_20260901_095444/
exp/cpu_kv_memory_doclen_sweep_20260901_095330/
exp/cpu_kv_memory_doclen_sweep_20260901_095156/
exp/doclen_sweep_*/
exp/prefix_cache_v1_vs_recompute_*/
exp/prefix_cache_serving_poisson_*/
exp/v2_lru_*/
exp/smoke_*/
```

原因：这些结果分别使用了旧统计口径、旧 V1/V2/V3 语义，或旧参数。尤其 `prefix_cache_serving_20260901_181056` 的 V3 配置中 lazy target window 大于实际 GPU KV blocks，不能用于评估 selective writeback。

当前可复用结果只有：

```text
exp/prefix_cache_serving_20260901_181816/
```

### V3 Watermark / CPU Cap 初步扫描

结果目录：

```text
exp/v3_hotcold_gpu_wm_sweep_20260901_214009/
```

本轮只跑 `hot_cold_sharing`，固定 `document_length=8192`、`gpu_kv_cache_gb=8.0`、`target_working_set_gb=20.0`、Poisson arrival、`max_num_seqs=8`。目的不是重跑完整主实验，而是给 V3 找 memory-latency tradeoff 的参数范围。

第一步先让 CPU cap 失效，只看 GPU lazy writeback watermark。`watermark=1.0` 没跑，因为当前 `max_num_batched_tokens=33152`、`block_size=256` 时 target window 会超过实际 GPU block 数；因此这轮扫 `0 / 0.25 / 0.5 / 0.7`。

| mode | GPU watermark | CPU peak | sync swapin blocks | doc recompute est | prefill total | schedule time | median TTFT | median req latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V2 | - | 20.25 GB | 577 | 0 | 3.126s | 0.536s | 0.088s | 0.552s |
| V3 | 0 | 20.25 GB | 513 | 0 | 3.715s | 0.575s | 0.088s | 0.544s |
| V3 | 0.25 | 20.25 GB | 545 | 0 | 3.205s | 0.577s | 0.079s | 0.510s |
| V3 | 0.5 | 20.25 GB | 577 | 0 | 2.968s | 0.577s | 0.080s | 0.502s |
| V3 | 0.7 | 20.25 GB | 578 | 0 | 3.467s | 0.590s | 0.089s | 0.548s |

结论：在 CPU 不限时，`watermark=0.25/0.5` 都没有触发 recompute，sync swapin 与 V2 接近；`0.5` 是这轮最稳的点。调度开销已经回到 V2 同级，说明 pinned CPU pool 和 allocation 后触发基本解决了早期 V3 的 scheduler 热路径开销问题。

第二步固定 `watermark=0.5`，改用 stream warmup 扫描 CPU cache cap。stream warmup 只把请求流前 30% 作为稳态预热，不再强制所有 document 都先访问一次；这比 all-doc warmup 更接近 serving 场景，也避免把无限 CPU backing 的 V2 设成过强上界。

结果目录：

```text
exp/v3_hotcold_cpu_cap_stream_20260901_225019/
```

参照 recompute baseline 使用当前可复用 hot/cold 结果：`prefill total = 6.307s`、`median TTFT = 0.081s`、`median request latency = 0.613s`。

![V3 stream CPU cap speedup vs recompute baseline](exp/v3_hotcold_cpu_cap_stream_20260901_225019/figures/stream_cpu_cap_speedup_vs_recompute.svg)

![V3 stream CPU cap live memory](exp/v3_hotcold_cpu_cap_stream_20260901_225019/figures/stream_cpu_cap_memory.svg)

![V3 stream CPU cap recompute/swapin](exp/v3_hotcold_cpu_cap_stream_20260901_225019/figures/stream_cpu_cap_recompute_swapin.svg)

| CPU cap | CPU peak | CPU evict blocks | CPU hit blocks | sync swapin blocks | doc recompute est | prefill total | prefill speedup vs recompute | median TTFT | median req latency |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 GB | 1.02 GB | 5,885 | 0 | 0 | 120,064 | 6.091s | 1.04x | 0.188s | 0.703s |
| 3 GB | 3.02 GB | 4,158 | 0 | 0 | 110,848 | 5.226s | 1.21x | 0.102s | 0.630s |
| 5 GB | 5.03 GB | 2,533 | 0 | 0 | 106,496 | 4.792s | 1.32x | 0.096s | 0.601s |
| 8 GB | 8.02 GB | 913 | 0 | 0 | 106,496 | 4.580s | 1.38x | 0.085s | 0.572s |
| 10 GB | 10.02 GB | 540 | 2 | 2 | 105,984 | 4.635s | 1.36x | 0.084s | 0.578s |
| 12.5 GB | 12.52 GB | 381 | 0 | 0 | 106,496 | 4.623s | 1.36x | 0.090s | 0.584s |
| 14 GB | 14.03 GB | 324 | 14 | 14 | 102,912 | 4.519s | 1.40x | 0.086s | 0.584s |
| 15 GB | 15.01 GB | 248 | 62 | 62 | 90,624 | 4.280s | 1.47x | 0.087s | 0.555s |
| 16 GB | 16.03 GB | 153 | 128 | 128 | 73,728 | 3.790s | 1.66x | 0.082s | 0.535s |
| 17 GB | 17.02 GB | 125 | 128 | 128 | 73,728 | 3.838s | 1.64x | 0.086s | 0.545s |
| 18 GB | 18.04 GB | 32 | 128 | 128 | 73,728 | 4.008s | 1.57x | 0.105s | 0.702s |
| 19 GB | 19.02 GB | 4 | 128 | 128 | 73,728 | 3.763s | 1.68x | 0.084s | 0.533s |
| 20 GB | 19.12 GB | 0 | 128 | 128 | 73,728 | 3.806s | 1.66x | 0.085s | 0.541s |

结论：stream warmup 下，低 CPU cap 不再像 all-doc warmup 那样立刻崩掉，因为热点请求大量由 GPU LRU 命中覆盖；但 `1GB` 已经丢掉 request latency 收益。`5-12.5GB` 仍有 prefill 加速，不过几乎没有 CPU hit，主要收益来自 GPU LRU 而不是 CPU restore。`16GB` 左右开始达到和 `19/20GB` 接近的 CPU restore / recompute 水平，可作为下一轮细扫的中心点；`18GB` 的端到端 latency 有单次抖动，暂不单独下结论。

当前 water marker 记录：GPU lazy writeback watermark 固定为 `0.5`；CPU watermark 这一轮等价为硬 cap 扫描，点位是 `1 / 3 / 5 / 8 / 10 / 12.5 / 14 / 15 / 16 / 17 / 18 / 19 / 20 GB`。后续如果要找最终推荐参数，可以围绕 `14-17GB` 做多次重复。

## 下一步

1. 后续 benchmark 不再默认使用 all-doc warmup 做主结论；改用 serving-style stream：生成一条长请求流，前一段只负责形成 cache steady state，后一段计入指标。
2. `bench_long_doc_qa.py` 默认已切到 `--warmup-mode stream --stream-warmup-ratio 0.3`；脚本也会显式写入这两个参数。`all_docs` 只保留给 sanity / 旧结果复现，不再作为主实验口径。
3. V3 当前建议参数：`lazy_writeback_watermark_ratio = 0.5`；stream 口径下 CPU cap 可围绕 `14-17GB` 继续细扫，`16GB` 是目前比较稳的候选点。
4. 后续主实验改成 cache pressure sweep：固定 `document_length = 8192`，通过改变 `num_documents` 扫描 `working_set_KV / GPU_KV`，例如 `0.5x / 1x / 2x / 3x / 4x`。
5. `cascade_tile` 保留为 sanity / stress；性能主张主要来自 `hot_cold_sharing` 和 `branching_prefix_sharing`。
6. V4 再做 scheduler-aware prefetch / OPT eviction，并补充 prefetch true/false positive、false negative 等准确性指标。
