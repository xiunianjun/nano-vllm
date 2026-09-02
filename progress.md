# nano-vLLM GPU+CPU Prefix Cache 进展

> 更新时间：2026-09-02
>
> 当前有效实验：`exp/prefix_cache_hotset12_20260902_110808/`
>
> 当前主结论只覆盖 `hot_cold_sharing`；旧实验结果已全部清除。

## 1. 项目目标

本项目把 nano-vLLM 的 GPU prefix cache 扩展为 GPU+CPU 两级 KV cache，目标场景是：热点 prefix 的总 KV working set 已超过 GPU KV cache，GPU 无法保留全部热点，但 CPU 仍有容量保存可复用 KV。

请求查找 prefix KV 时有三条路径：

```text
GPU hit              -> 直接复用 GPU KV
GPU miss + CPU hit   -> H2D restore 到 GPU，跳过 prefix prefill
GPU miss + CPU miss  -> 正常 prefill / recompute
```

模型权重始终完整驻留 GPU；实验只限制 GPU KV cache 容量。本阶段不考虑 SSD，也不把 preemption swapping 当作主要机制。

实现沿用 nano-vLLM 的 logical block 抽象：一个 block id 对应所有 layer 的 K/V slice。当前 Qwen3-8B、block size 256 tokens 时，一个完整 KV block 约为 36 MiB。

## 2. 版本语义

| 模式 | CPU offload | GPU inactive LRU | CPU writeback | 目的 |
|---|---:|---:|---|---|
| GPU-only baseline | 否 | 是 | 无 | GPU miss 后重新计算 prefix |
| V1 | 是 | 否 | eager，全量 | 验证 CPU backing/restore 的基础收益 |
| V2 | 是 | 是 | eager，全量 | 用 GPU LRU 减少 V1 的同步 H2D restore |
| V3 | 是 | 是 | lazy，按水位选择 | 在 V2 基础上探索 CPU memory/latency tradeoff |

重要口径：当前脚本中的 `baseline` 实际参数是 `--no-enable-cpu-kv-offload --enable-gpu-lru-retention`，所以它是 **GPU-only LRU baseline**，不是“关闭所有 cache 的纯 recompute baseline”。原始 JSON 的 `mode=gpu_prefix_cache_recompute_baseline` 只是遗留名称，分析时不得据此误判。

### V1：CPU eager backing

- prompt prefill 完成后，完整 prefix blocks 异步 D2H 写回 CPU。
- CPU backing 以 block hash 为 key；token ids 继续用于 hash collision 校验。
- GPU miss、CPU hit 时分配 GPU block并同步 H2D，剩余未命中 token 才 prefill。
- pending D2H 的 GPU block受到保护，复制完成前不能被覆盖。
- V1 不保留 inactive GPU prefix，因此重复访问主要走 CPU restore。

### V2：CPU eager backing + GPU LRU retention

- V1 的 CPU eager writeback 保持不变。
- 请求结束后，完整 prefix block 进入 inactive GPU LRU，不立即丢弃。
- 新请求命中 inactive block 时重新激活，避免一次 CPU H2D。
- 需要空间时先使用真正 free block，再驱逐 inactive LRU block；优先驱逐已有 CPU backing 的 block。
- 同一请求按逆序释放，使更通用、靠近 prefix 根部的 block在相同 recency 下保留更久。

V2 相对 V1 的核心评价指标是 GPU LRU hits、同步 swapin blocks、H2D bytes、restore time 和 prefill time，而不是只看最终吞吐。

### V3：memory-aware lazy writeback

V3 不再把所有完成的 prefix 立刻写入 CPU，只在 allocation 后 GPU 可安全回收空间低于目标水位时，从 inactive LRU 的 eviction end 选择 GPU-only blocks 做异步 D2H：

```text
target_blocks = ceil(max_num_batched_tokens / block_size)
target_free   = target_blocks * (1 + lazy_writeback_watermark_ratio)
```

也可以用 `lazy_writeback_target_blocks` 直接指定绝对 block 数；大于 0 时覆盖上述按最大 batch 推导的值，便于按实际 workload 校准安全窗口。

当前状态包括 active GPU、inactive GPU-only、inactive GPU+CPU-backed、pending writeback、CPU-only 和 dropped。V3 使用预分配 pinned CPU block pool，避免在 scheduler 热路径逐 block 分配 pinned tensor；配置 CPU cap 后，pool 同时是物理内存硬上限，容量不足时按 CPU LRU 复用或拒绝 writeback。

需要区分：

- `cpu_prefix_cache_live_gb`：当前真正有效、可命中的 CPU backing。
- `cpu_prefix_pool_reserved_gb`：预分配的物理 pinned-memory 容量。
- `cpu_prefix_cache_gb_limit=0`：CPU cache 不设逻辑上限；不是“CPU 容量为 0”。
- 当前没有独立的“CPU watermark”；CPU 侧配置是 hard cap/pool capacity。

GPU sweep 期间 CPU cap 保持为 `0`（unlimited），当前选出的 GPU watermark 为绝对值 `40 blocks`。它是当前 workload 下未发生 GPU-only eviction 的最小已测点；若需要为不同 seed 或 batch 波动留保守余量，可使用 `60 blocks`。

## 3. 已完成的工程优化

- decode admission 使用可驱逐 block 计数，避免逐 token 扫描 inactive LRU。
- decode 路径复用 pinned CPU/CUDA buffer；CUDA Graph 路径直接填 graph backing buffer。
- pending writeback 使用具名记录和直接索引，allocator pending 状态收敛到 `Block`，移除重复状态集合。
- 一批 D2H writeback 共享 CUDA events，event 数从随 block 数增长降为固定数量。
- `Sequence` 缓存连续完整 block 的 chained hashes，lookup 与 `hash_blocks` 复用，保留 token-id 碰撞保护。
- V1/V2/V3 的兼容 CLI 开关在配置入口统一映射为 `KVCachePolicy`，减少 Scheduler、BlockManager、LLMEngine 和 benchmark 中重复的布尔组合判断。
- V3 的 bounded CPU cache 改为 GPU-residency-aware eviction：优先淘汰 GPU 仍有副本且不在近期 victim window 内的 CPU backing，并保护即将从 GPU 驱逐的 block；V1/V2 的纯 CPU LRU 语义保持不变。
- benchmark 增加分阶段计时、CPU/GPU cache 计数、working-set 比例、trace/output 校验和、配对比较及 Student-t 95% CI。

这些优化已经过小模型短测和行为一致性检查；当前 8B 主实验是在全部优化之后运行。

## 4. 当前主实验设计

### 4.1 为什么改成 12 个热点文档

旧 workload 只有 2 个热点文档，每个 prefix KV 约 1.125 GiB，热点 working set 仅约 2.25 GiB，小于实际 7.98 GiB GPU KV cache。GPU-only baseline 能长期装下全部热点，因此测不到 offloading 面向的容量压力，V1/V2 的 CPU 搬运只会显得多余。

当前改为 12 个热点文档：

```text
hot working set   = 12 * 1.125 GiB = 13.5 GiB
GPU KV capacity   = 7.9805 GiB
hot/GPU ratio     = 1.69x
total working set = 20.25 GiB = 2.54x GPU KV
```

GPU 估算只能同时容纳 6 个完整 prompt，明显少于 12 个热点 prefix。这样 GPU-only LRU 会发生真实热点 thrashing，而 CPU offload 有机会避免反复 recompute。

### 4.2 有效配置

| 类别 | 配置 |
|---|---|
| GPU | NVIDIA H200 NVL，`CUDA_VISIBLE_DEVICES=1` |
| 模型 | `/data/datasets/models-hf/Qwen3-8B` |
| 环境 | `.venv-fa28`，Python 3.10.19，Torch 2.8.0+cu128，FlashAttention 2.8.3.post1 |
| 执行模式 | `--enforce-eager`，temperature 0 |
| document/query/output | 8192 / 96 / 16 tokens |
| block size | 256 tokens |
| target/actual working set | 20.0 / 20.25 GiB，18 documents |
| requested/actual GPU KV | 8.0 / 7.98046875 GiB，227 blocks |
| single prompt KV | 1.14038 GiB；约占 GPU KV 14.29%；估算可容纳 6 个 |
| 并发限制 | `max_num_seqs=8` |
| prefill batch | `max_num_batched_tokens=33152 = 4 * (8192+96)` |
| arrival | Poisson，target 2 req/s |
| warmup | stream，前 30% 请求不计入测量 |
| hot/cold | 12 hot docs，hot probability 0.8，repeat count 20 |
| 请求数 | 共 360；warmup 108；measured 252 |
| 重复 | 3 个 seed，每个 mode 使用同 seed 配对 |
| 主实验 V3（历史测量点） | watermark ratio 0.5；CPU cap 0；CPU pool 20.25 GiB |

每个 measured trace 都实际覆盖 18 个文档，realized working set 为 20.25 GiB。warmup 在 seed 1 覆盖 17 个文档，seed 2/3 覆盖 18 个。三轮 mode 顺序轮换：

```text
run 1: baseline -> V1 -> V2 -> V3
run 2: V1 -> V2 -> V3 -> baseline
run 3: V2 -> V3 -> baseline -> V1
```

这能降低固定执行顺序与温度/系统漂移的混淆，但不能完全消除时间漂移。

### 4.3 运行命令

```bash
GPU=1 \
RUNS=3 \
RUN_CASE0=0 \
RUN_CASCADE=0 \
RUN_HOT_COLD=1 \
RUN_BRANCHING=0 \
HOT_DOCUMENTS=12 \
HOT_REQUEST_RATIO=0.8 \
HOT_REPEAT_COUNT=20 \
EXP_DIR=exp/prefix_cache_hotset12_20260902_110808 \
scripts/run_prefix_cache_cases.sh
```

`scripts/run_prefix_cache_cases.sh` 当前通用默认值里 `HOT_DOCUMENTS=2`、`HOT_REPEAT_COUNT=4`，因此复现本轮时必须保留以上显式覆盖；不能只依赖默认参数。

## 5. Benchmark 严谨性与统计口径

- warmup 和 measured 是同一请求流的连续两段，只有 measured 阶段进入最终统计。
- 同一 run id 的 baseline/V1/V2/V3 使用相同 arrival seed 和 shuffle seed，逐 run 做 paired comparison。
- 三轮执行顺序旋转，避免所有模式永远占据相同运行位置。
- temperature 为 0；校验各模式输出 hash 一致。
- 校验配对 trace 的 prompt/arrival fingerprint 一致。
- 校验 CPU live cache 不超过配置的物理 budget。
- 模式均独立启动进程，避免 cache 状态跨模式泄漏。
- summary 同时保存三轮 mean、stdev、min/max 和基于 Student-t 的 95% CI；加速比优先使用同 seed 的 paired ratio 均值。

本轮三项自动校验全部通过：

```text
paired_trace_ok        = true
greedy_output_match_ok = true
cpu_physical_budget_ok = true
```

指标解释：

- request latency、TTFT、queueing 是每轮 252 个 measured requests 的分布统计，再对三轮统计量求均值。
- prefill/decode time 是 measured window 内相应 engine steps 的累计时间。
- `document_recomputed_tokens_est` 是按未复用文档 prefix 估算的重算量，用于跨模式解释机制。
- achieved throughput 在本轮主要由 2 req/s 的 offered load 限制，不是系统饱和吞吐；性能结论应以 latency、TTFT、prefill 和 cache traffic 为主。

## 6. 当前有效结果

结果目录：`exp/prefix_cache_hotset12_20260902_110808/hot_cold_sharing/`

以下均为 3 次运行的均值：

| mode | median req | p99 req | median TTFT | p99 TTFT | prefill total | decode total | achieved req/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPU-only baseline | 0.857s | 2.215s | 0.253s | 1.359s | 41.062s | 47.719s | 2.067 |
| V1 | 0.542s | 0.923s | 0.063s | 0.124s | 14.200s | 63.056s | 2.070 |
| V2 | 0.537s | 0.915s | 0.061s | 0.122s | 12.663s | 65.719s | 2.070 |
| V3 | 0.522s | 0.946s | 0.060s | 0.117s | 12.410s | 64.070s | 2.068 |

机制指标：

| mode | recompute tokens | sync swapin blocks | GPU LRU hit blocks | GPU evictions | H2D | CPU peak | restore time | schedule time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPU-only baseline | 1,234,603 | 0 | 2,441 | 4,824 | 0 | 0 | 0 | 0.133s |
| V1 | 2,731 | 7,541 | 0 | 0 | 265.13 GiB | 20.25 GiB | 5.709s | 5.997s |
| V2 | 2,731 | 4,740 | 2,780 | 4,751 | 166.65 GiB | 20.25 GiB | 3.588s | 3.893s |
| V3 | 2,731 | 4,739 | 2,792 | 4,750 | 166.60 GiB | 20.25 GiB | 3.588s | 4.000s |

主要 paired speedup：

| 对比 | median req | p99 req | median TTFT | p99 TTFT | prefill total |
|---|---:|---:|---:|---:|---:|
| V1 / baseline | 1.580x | 2.429x | 4.009x | 10.878x | 2.892x |
| V2 / baseline | 1.598x | 2.552x | 4.134x | 11.396x | 3.248x |
| V3 / baseline | 1.641x | 2.394x | 4.203x | 11.416x | 3.311x |
| V2 / V1 | 1.011x | 1.071x | 1.031x | 1.024x | 1.123x |
| V3 / V2 | 1.028x | 1.054x | 1.018x | 1.066x | 1.023x |

## 7. 结果分析

### 7.1 Offloading 在目标压力下有效

新 workload 修正了旧实验的根本问题。热点 working set 为 GPU KV 的 1.69 倍后，GPU-only baseline 即使有 LRU，仍平均重算约 123.5 万 document tokens；V1/V2/V3 仅约 2,731。对应 V1 已取得 1.58x median request、2.43x p99 request 和 2.89x prefill 加速，说明 CPU offloading 在“GPU 装不下热点 prefix”的目标场景中确实有效。

### 7.2 V2 的主要收益是减少 V1 的 H2D

V2 相比 V1：

- sync swapin blocks：7,541 -> 4,740，减少 37.1%。
- H2D：265.13 -> 166.65 GiB，减少 37.1%。
- restore time：5.709 -> 3.588s，减少 37.2%。
- prefill paired speedup：1.123x。

端到端 median request 只提升约 1.1%，不是 V2 没生效，而是本轮 arrival rate 没有把吞吐打满，且 decode 累计时间在 offload 模式中高于 baseline。V2 的机制收益应由 traffic/restore/prefill 指标直接说明。

### 7.3 本轮不能证明 V3 的内存收益

本轮 V3 的 CPU cap 为 unlimited，pool 又覆盖完整 20.25 GiB working set，因此最终 CPU peak 与 V2 相同。V3 与 V2 的 restore、H2D 和 GPU LRU 行为也几乎一致；本轮只能说明 lazy path 在该配置下达到 V2 级性能，不能声称节省 CPU 内存。

要评价 V3，必须保持当前 12-hot-doc workload 不变，再扫描 CPU hard cap，画 CPU peak/eviction/recompute/TTFT 的 tradeoff。旧 workload 下得到的 V3 cap/watermark 扫描已经删除，不应继续引用。

### 7.4 GPU watermark follow-up sweep

保持 hot/cold workload、CPU cap unlimited 和 seed 1 不变，用绝对 block 数复用相同 trace（SHA256 均为 `ffeec039...`）扫描 GPU 安全窗口：

| GPU watermark | GPU-only evictions | sync swapin blocks | document recompute tokens | prefill time |
|---:|---:|---:|---:|---:|
| 130 | 0 | 4,798 | 8,192 | 14.15s |
| 60 | 0 | 4,797 | 8,192 | 12.92s |
| 40 | 0 | 4,796 | 8,192 | 12.12s |
| 30 | 9 | 4,428 | 102,912 | 14.55s |
| 10 | 2,537 | 281 | 1,167,360 | 42.67s |

临界区间位于 `30--40 blocks`：30 blocks 已因少量 GPU-only eviction 放大为整段文档 recompute，40 blocks 则与 60/130 的 cache traffic 和 recompute 行为一致。因此当前实验固定使用 `40 blocks`，而不是原先根据 `max_num_batched_tokens` 保守推导出的 130 blocks。

这里的 40-block 水位主要用于吸收当前 request-level KV allocation 的瞬时 burst，并不代表 D2H 带宽本身需要这么大的长期 backlog。后续若改为 block-wise 或 layer-wise 的渐进分配/回收，分配突发会更平滑，所需安全窗口预计还能进一步缩小。

该 follow-up 是单 seed 快速定位；40 blocks 若作为跨 workload 或生产默认值，仍需补不同 seed、arrival rate 和 batch 形态验证。当前 workload 下更保守的选择是 60 blocks。

### 7.5 GPU-aware CPU LRU：duplicate 容量浪费与初步修复

固定 40-block GPU watermark、16 GiB CPU hard cap 和 seed 1 后，旧实现末态有 455 个 CPU block、224 个 GPU prefix block，其中 199 个同时存在于 CPU/GPU。CPU cache 的 43.7% 被 duplicate 占据，CPU+GPU 合并后只有 480 个 unique blocks，无法覆盖 576-block working set。

原因不是单一的“读回后没有删除 CPU copy”，而是 V3 的正常状态迁移与局部 LRU 决策共同造成：lazy D2H 会有意保留 GPU victim 的 CPU backing；CPU restore 后 backing 仍保留并刷新 CPU recency；同一 prefix 再次在 GPU 命中后可能成为 GPU MRU，但 CPU 仍有副本。旧 CPU LRU 不知道 GPU residency，容量满时可能先淘汰 CPU-only 的唯一副本，却长期保留 CPU+GPU duplicate。

本轮把 V3 CPU eviction 改成分层策略：

1. 先按 CPU LRU 淘汰“GPU 仍有副本、且不在 GPU LRU 前 40 个近期 victim 中”的 duplicate；
2. 若没有这类 duplicate，再按 CPU LRU 淘汰非保护项；
3. GPU victim window 内的 CPU backing 最后才允许淘汰，避免下一次 GPU eviction 立即失去可恢复副本。

同一输入 trace 和同一输出（trace/output SHA256 均一致）的 16 GiB 单-seed 对照如下：

| 指标 | 纯 CPU LRU | GPU-aware CPU LRU | 变化 |
|---|---:|---:|---:|
| CPU/GPU duplicate blocks | 199 | 103 | -48.2% |
| duplicate / CPU blocks | 43.7% | 22.6% | -21.1 pp |
| CPU+GPU unique blocks | 480 | 576 | +20.0%，覆盖完整 working set |
| document recompute tokens | 315,648 | 90,880 | -71.2% |
| prefill tokens | 339,840 | 115,072 | -66.1% |
| sync swapin blocks | 3,598 | 4,467 | +24.2% |
| H2D traffic | 126.49 GiB | 157.04 GiB | +24.2% |
| D2H traffic | 43.17 GiB | 152.30 GiB | +252.8% |
| prefill total | 19.35s | 14.71s | -24.0% |
| median request | 537.7ms | 529.9ms | -1.5% |
| p90 request | 826.1ms | 697.5ms | -15.6% |
| p90 TTFT | 259.5ms | 105.4ms | -59.4% |

这里性能提高并不是 GPU LRU 本身“更准”，而是 CPU 淘汰从局部 recency 变成了考虑两级 residency 的全局价值判断。新策略增加了 869 个 H2D swapin blocks，同时多得到 9 个 GPU cache hits；二者恰好替代 878 个 prefill blocks。当前测量中单 block H2D 约 0.76ms，而避免的 prefill CUDA 计算约 6.4ms/block，因此用更多廉价传输换掉昂贵 recompute，prefill 和 tail latency 反而改善。

代价是 migration churn 明显上升：CPU eviction 从 1,228 增至 4,332，D2H 增至 152.30 GiB，GPU-only eviction 从 4 增至 221。末态 duplicate 中 active/protected/unprotected 分别为 0/40/63，说明 40-block 硬保护窗口外仍可能存在很快再次成为 GPU victim 的 backing。这个结果证明了容量方向有效，但当前策略还不是最终最优；后续应增加 hysteresis 或扩大软保护区，在 recompute 与迁移流量之间找平衡，然后再继续 CPU cap sweep。

上述结论来自单 seed 快速验证，结果目录为 `exp/v3_cpu_blocks40_coarse_20260902_145434/`（旧策略）与 `exp/v3_cpu_gpuaware_16gb_20260902_1535/`（新策略）。

### 7.6 CPU memory boundary 与去重收益

为隔离 GPU-aware CPU eviction 的收益，benchmark 新增 `--no-enable-gpu-aware-cpu-eviction` 消融开关；关闭后 V3 恢复为原来的纯 CPU LRU（Naive V3），其他配置保持不变。固定 hot/cold workload、40-block GPU watermark 和 seed 1，得到：

| Naive V3 CPU cap | CPU blocks | GPU blocks | duplicate | unique | document recompute tokens | prefill | TTFT p90 | request p90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 GiB | 568 | 224 | 216 | 576 | 66,560 | 13.49s | 84.58ms | 649.08ms |
| **19 GiB** | 540 | 224 | 224 | 540 | **105,472** | **14.37s** | **91.38ms** | **678.21ms** |
| 18 GiB | 512 | 224 | 224 | 512 | 156,160 | 15.71s | 135.19ms | 754.29ms |

CPU 内存拐点不由 E2E median 单独决定。主判据是 `document_recomputed_tokens_est` 是否阶跃上升，再由 prefill time 和 TTFT p90 确认；request latency 只作辅助，因为它还混入 decode、排队和到达过程噪声。从 19 GiB 降到 18 GiB 后，recompute 增加 48.1%、TTFT p90 增加 47.9%、prefill 增加 9.3%，因此 Naive V3 的单-seed 实用边界取 **19 GiB**。若要求末态 CPU+GPU 覆盖全部 576 个 unique blocks，则其严格容量边界约为 **20 GiB**。

最新 GPU-aware V3 的已测实用边界取 **15 GiB**：其 document recompute 为 100,864，与 Naive V3 19 GiB 的 105,472 相当；14 GiB 已升至 130,560。按这一相同的实用性能边界口径，CPU 内存收益为：

1. 原 Naive V3 相比 V2：`20.25 -> 19 GiB`，节省 **1.25 GiB（6.2%）**；
2. GPU-aware 去重相比 Naive V3：`19 -> 15 GiB`，再节省 **4 GiB（21.1%）**；
3. 最新 V3 相比 V2：`20.25 -> 15 GiB`，累计节省 **5.25 GiB（25.9%）**。

以上百分比以比较对象的 CPU 内存为分母，使用配置值便于表达；实际 block 对应的 peak 分别约为 20.25、18.98 和 14.98 GiB，结论不变。该边界来自相同 trace 的单 seed 快速定位，适合说明当前 workload 下的容量差异，不应解释为跨 seed 的统计置信区间。

### 7.7 仍需解释的性能现象

- offload 模式 decode total 为 63--66s，高于 baseline 的 47.7s。可能涉及 restore/transfer 对 model step 的干扰、不同 batch/step 形态或 eager-mode 开销，需要单独 profile。
- achieved throughput 四种模式都约 2.07 req/s，是 arrival-limited 结果，不能外推最大吞吐能力。
- 本轮强制 eager，不能据此评价 CUDA Graph 路径优化。
- 只有 3 个独立 seed。median 和 prefill 趋势较清楚，但 p99 与 V2/V3 之间的小差异 CI 很宽，不宜下显著性结论。
- warmup/measurement 边界仍有 inflight requests：baseline 三轮为 `[3,2,0]`，V1/V2 为 `[2,0,0]`，V3 为 `[3,0,0]`。输入 trace 配对一致，但不同 engine 速度会改变边界 inflight 状态；这是 serving-style continuous warmup 的剩余噪声源。

## 8. 当前结论与下一步

当前可以支持的结论：

1. 当热点 prefix working set 明确超过 GPU KV capacity 时，CPU prefix offloading 相比 GPU-only LRU 显著减少 recompute，并改善 TTFT、request latency 和 prefill time。
2. V2 GPU LRU retention 明确减少 V1 的同步 H2D/restore，prefill 收益可测；在 2 req/s 的低压 Poisson workload 下，端到端 median 收益被 decode 与 arrival limit 稀释。
3. V3 GPU 安全窗口在当前 workload 下可从保守推导值 130 blocks 缩至 40 blocks；30 blocks 已出现 GPU-only eviction，因此不能继续缩小 request-level 安全窗口。
4. 16 GiB 下 GPU-aware CPU LRU 将 duplicate 从 199 降至 103、unique coverage 从 480 提至完整的 576 blocks，并用更多 H2D 换取更少 recompute；但迁移 churn 与 GPU-only eviction 同时上升，尚需进一步约束。
5. 按 recompute、prefill 和 TTFT 共同确定的实用性能边界，Naive V3 为 19 GiB，GPU-aware V3 为 15 GiB：去重策略再节省 4 GiB（21.1%），最新 V3 相比 V2 的 20.25 GiB 共节省 5.25 GiB（25.9%）。

建议后续按优先级进行：

1. 为 GPU-aware CPU eviction 增加 hysteresis/软保护区，降低刚淘汰 CPU backing 又发生 GPU eviction 的迁移 churn，并用 16 GiB 同 trace 复核 recompute、GPU-only eviction 和 D2H/H2D。
2. 用更多 seed 在 14--16 GiB 附近复核 GPU-aware V3 的边界，并在 18--20 GiB 附近复核 Naive V3；当前单-seed 粗扫已将两者的实用边界分别定位为 15 和 19 GiB。
3. 在不同 seed、arrival rate 和 batch 形态下复核 40 blocks；若需要无需复核即可使用的保守点，采用 60 blocks。
4. 增加 request-rate sweep，直到接近饱和，分别报告 offered load、achieved throughput、queueing 和 tail latency。
5. profile offload 模式较高的 decode time，拆分 model runner、transfer 和 scheduler 干扰。
6. 如需“纯 recompute”对照，新增同时关闭 CPU offload 和 GPU LRU 的独立模式，并给现有 baseline 重命名，避免口径混淆。
7. 最终结果至少增加到 5 个 seeds，并保留 raw per-run 数据、paired ratios 和 Student-t CI。

## 9. 结果保留策略

`exp/` 下旧实验、旧图表、旧 summary 和无效中断 campaign 已于 2026-09-02 全部清除。当前只保留：

```text
exp/prefix_cache_hotset12_20260902_110808/
exp/prefix_cache_hotset12_20260902_110808.log
exp/v3_gpu_wm_coarse_hotset12_20260902_123022/
exp/v3_gpu_blocks_fast_20260902_142113/
exp/v3_gpu_blocks30_fast_20260902_143636/
exp/v3_gpu_blocks40_fast_20260902_144158/
exp/v3_cpu_blocks40_coarse_20260902_145434/
exp/v3_cpu_gpuaware_16gb_20260902_1535/
exp/v3_cpu_gpuaware_8gb_20260902_153724/
exp/v3_cpu_gpuaware_14gb_20260902_154300/
exp/v3_cpu_gpuaware_15gb_20260902_154712/
exp/v3_cpu_naive_boundary_20260902_160124/
```

其中 `prefix_cache_hotset12_20260902_110808/hot_cold_sharing/summary.json` 是主实验分析入口；GPU block sweep 的 130/60/40/30/10 原始结果分别保存在上述 GPU sweep 目录；CPU 目录保留 GPU-aware 8/14/15/16 GiB sweep 与 Naive V3 18/19/20 GiB 边界数据。后续实验必须使用新的时间戳目录，不覆盖已有结果。
