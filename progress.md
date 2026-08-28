# nano-vLLM GPU+CPU Prefix Cache 进展

## 当前研究方向

当前主线：

```text
将 nano-vLLM 现有 GPU prefix cache 扩展为 GPU + CPU 两级 prefix cache。
```

研究场景：

- 模型权重始终完整驻留 GPU。
- 只人为限制 GPU KV Cache capacity。
- 请求之间存在可复用 prefix。
- GPU prefix cache miss 时，未来目标是优先从 CPU prefix cache restore KV，而不是重新 prefill。

重要判断：

```text
nano-vLLM 当前不是 layer-wise page abstraction。
```

它的 `BlockManager` 管理的是 logical block/page。一个 logical block id 同时对应所有 layer 的 K/V slice：

```text
kv_cache shape = [K/V][layer][block_id][token_offset][kv_head][head_dim]
```

因此不能直接照搬 LMCache 的 layer-wise KV page 管理策略。LMCache Long-Document QA 仍然可作为 workload 语义参考，但 CPU cache / eviction / restore 的实现必须贴合 nano-vLLM 当前的 logical block 设计。

对 Qwen3-8B：

```text
1 logical block = 256 tokens 的全层 KV
1 logical block = 36 MiB
1 layer 内的 block slice = 1 MiB
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

说明：`.venv-fa28` 中已有匹配的 `flash-attn` wheel，后续 benchmark 优先使用该环境，避免本地编译 `flash-attn` 把服务器资源打满。

## Benchmark 入口

主 benchmark 脚本：

```bash
bench_long_doc_qa.py
```

统一 case runner：

```bash
scripts/run_prefix_cache_cases.sh
```

运行方式：

```bash
GPU=1 VENV=.venv-fa28 OUT_DIR=results/prefix_cache_cases/latest_smoke \
  scripts/run_prefix_cache_cases.sh
```

结果会保存到：

```text
results/prefix_cache_cases/<run_name>/
```

## Workload 语义

参考 LMCache Long-Document QA benchmark 的 workload 语义，但不照搬其 KV cache 实现。

核心语义：

- warmup 阶段先访问所有 reusable prefixes，不计入最终指标。
- measured query 阶段再次访问相同 prefix，但使用不同 query suffix。
- same document / same branch 需要保证 prefix token 完全一致。
- query suffix 不同，用于模拟同一文档或同一分支上的不同问题。

当前不使用 ShareGPT，不引入真实长文档数据集。文档和 branch 都用 synthetic token IDs 生成。

## 新实现路线

### V1: CPU Prefix Cache Offload

目标：先实现正确的 GPU + CPU 两级 prefix cache，不做复杂调度预取。

#### 主动写回

当一个 request 完成后，将它已经计算出的 reusable prefix KV 写回 CPU memory。

```text
request finished
-> GPU KV still remains available
-> async D2H writeback to CPU pinned memory
-> CPU copy becomes reusable after copy finishes
```

这里 GPU KV 不需要因为写回 CPU 而立即释放。一个 prefix block 可以同时存在于 GPU 和 CPU：

```text
GPU_RESIDENT + CPU_RESIDENT
```

如果 GPU 显存还够，可以先调度下一个请求，再在 copy stream 上异步写回刚完成 request 的 KV，从而和后续计算 overlap。

异步 D2H 的完成通知机制：

```text
copy_stream:
  cpu_tensor.copy_(gpu_tensor, non_blocking=True)
  event.record(copy_stream)

later:
  if event.query():
      state = CPU_RESIDENT
```

也就是说，CUDA event 是完成通知。写回完成前，对应 GPU block 不能被别人覆盖，否则 CPU copy 会读到不完整或错误数据。状态上可以记为：

```text
WRITEBACK_PENDING / EVICTING
```

完成后才允许该 GPU block 进入可回收状态。

#### 读入 request

调度新 request 时查 prefix：

```text
GPU hit
-> 直接复用 GPU KV

GPU miss + CPU hit
-> 同步 H2D restore 到 GPU
-> 跳过对应 prefix prefill

GPU miss + CPU miss
-> 正常 prefill/recompute
```

第一版 `swapin` 可以先同步实现，保证正确性。若 GPU 空间不足，则触发 eviction。

#### 抢占 / 驱逐

eviction 优先级：

```text
1. 优先驱逐已经 CPU_RESIDENT 的 GPU blocks
   -> 只释放/覆盖 GPU copy，不需要 D2H

2. 若没有可直接驱逐的 clean blocks
   -> 按 LRU 选择 victim
   -> 同步 D2H 写回 CPU
   -> 再释放/覆盖 GPU block
```

第一版先追求状态正确和 benchmark 可解释，不急着做最优 overlap。

### V2: Scheduler-Aware Prefetch

在 V1 正确后，再利用调度信息优化 eviction 和 prefetch。

#### OPT-like eviction

驱逐时 inspect 当前 benchmark/scheduler 队列，选择未来窗口内最晚再次被访问的 prefix block：

```text
evict block whose next use is farthest in the future
```

这接近 OPT 策略，适合作为研究型 upper-bound / oracle policy。

#### 主动预取

调度当前 request 后，如果 GPU KV 还有空间，可以提前把后续请求可能需要的 CPU-resident KV copy 回 GPU：

```text
current request running
GPU has spare blocks
next likely request has CPU KV but no GPU KV
-> async H2D prefetch
```

预取同样需要状态保护：

```text
PREFETCHING
GPU_RESIDENT
CPU_RESIDENT
```

`PREFETCHING` 完成前不能被当作完整 GPU hit。完成通知仍使用 CUDA event。

## 已实现 Workload Case

### Case 1: cascade_tile

目标：观察 GPU prefix cache 的级联污染 / cache thrashing。

访问模式：

```text
warmup:  D0, D1, D2, ...
query:   D0, D1, D2, ...
```

当 working set 略大于 GPU KV cache 时，第二轮从 `D0` 开始 miss。`D0` 重新 prefill 时分配的新 blocks 会覆盖后续 document 的部分 prefix blocks，导致后续 document 在被访问前也失效，形成连续 miss。

这个 case 可以作为“调度顺序和 cache eviction 不配合”的 worst-case。

### Case 2: hot_cold_sharing

目标：观察正常冷热 document prefix sharing。

访问模式：

```text
hot documents: 频繁访问
cold documents: 偶尔访问
```

请求形态：

```text
D0 + Q1
D1 + Q1
D0 + Q2
D5 + Q1
D1 + Q2
...
```

这里会同时出现：

- GPU prefix hit；
- GPU miss 后重新 prefill；
- hot prefix 比 cold prefix 更容易留在 GPU cache。

### Case 3: branching_prefix_sharing

目标：观察不同 request 之间的部分 prefix sharing。

请求形态：

```text
RootPrefix + BranchA + Query1
RootPrefix + BranchA + Query2
RootPrefix + BranchB + Query1
RootPrefix + BranchB + Query2
```

复用层次：

- 不同 branch 之间共享 `RootPrefix`。
- 同一 branch 下不同 query 共享 `RootPrefix + BranchPrefix`。
- query suffix 不共享。

这个 case 更接近多请求分叉场景，也更适合后续验证 CPU prefix cache 是否能保存被 GPU 淘汰的中间 prefix。

## 当前 Baseline 配置

三组 case 使用同一基础配置：

```text
model = /data/datasets/models-hf/Qwen3-8B
GPU = CUDA_VISIBLE_DEVICES=1
document_length = 1024
query_length = 64
output_len = 8
target_working_set_gb = 1.0
gpu_kv_cache_gb = 1.1
max_num_seqs = 1
enforce_eager = true
```

实际模型 KV 参数：

```text
kv_bytes_per_token = 147456
gpu_kv_cache_gb_actual = 1.08984375
num_kvcache_blocks = 31
block_size = 256
```

## Baseline 结果

本轮结果目录：

```text
results/prefix_cache_cases/latest_smoke/
```

| case | workload | query requests | prefix reused tokens | estimated recomputed prefix tokens | query elapsed |
|---|---|---:|---:|---:|---:|
| cascade_tile | long_doc_qa | 8 | 0 | 8192 | 1.83 s |
| hot_cold_sharing | long_doc_qa | 32 | 25344 | 7424 | 7.15 s |
| branching_prefix_sharing | branching_prefix | 56 | 52736 | 4608 | 12.40 s |

观察：

- `cascade_tile` 出现全 miss，说明该 case 成功暴露了级联污染。
- `hot_cold_sharing` 同时出现 reuse 和 recompute，适合作为主线 document-level sharing baseline。
- `branching_prefix_sharing` reuse 更多，说明部分 prefix sharing 已经能被 nano-vLLM 的 GPU prefix cache 捕获。

## 指标说明

当前 prefix-cache 主线主要看：

- `prefix_cache_lookup_count`
- `prefix_cache_reused_token_count`
- `prefill_token_count`
- `document_recomputed_tokens_est`
- `query_latency_sec`
- `query_elapsed_sec`

其中 `document_recomputed_tokens_est` 是当前 GPU-only baseline 下估算的 prefix 重算量。后续加入 CPU prefix cache 后，目标是把这部分 GPU miss 从重新 prefill 转换为 CPU KV restore。

## 下一步

1. 固化当前 GPU-only prefix cache baseline，作为 V1/V2 对照组。
2. 实现 logical block 粒度的 CPU prefix cache backing store。
3. request 完成时主动写回 CPU，并维护 `GPU_RESIDENT`、`WRITEBACK_PENDING`、`CPU_RESIDENT` 等状态。
4. GPU miss + CPU hit 时同步 restore 到 GPU，并跳过对应 prefix prefill。
5. eviction 优先驱逐已经 CPU-resident 的 clean GPU blocks；否则 LRU 同步写回。
6. V1 正确后，再实现 scheduler-aware OPT eviction 和 async prefetch。
7. 用 `cascade_tile`、`hot_cold_sharing`、`branching_prefix_sharing` 对比 GPU-only 与 GPU+CPU。
