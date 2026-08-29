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

## 当前 V1 实现

### 目标

V1 已经从旧的 request-level preemption swapping，调整为当前主线需要的 **CPU prefix cache backing store**：

```text
GPU prefix hit
-> 直接复用 GPU KV

GPU miss + CPU prefix hit
-> 同步 H2D restore 到 GPU KV block
-> 跳过对应 prefix prefill

GPU miss + CPU miss
-> 正常 prefill / recompute
```

当前 V1 不做复杂 scheduler-aware prefetch，也不做 OPT eviction。重点是先保证 GPU+CPU 两级 prefix cache 的状态和 benchmark 语义正确。

### Prefix 写回时机

```text
request prompt prefill 完成
-> 已形成完整 block hash 的 prefix blocks 立刻启动 async D2H writeback
-> request 继续进入 decode
```

这样可以覆盖一种重要情况：request prefill 完后进入 decode，如果 decode 阶段又因为显存压力被抢占，它的 prompt prefix 已经有机会在 CPU 上留下 backing copy。

decode 新生成的 tokens 当前 **不主动纳入 prefix cache**。这点和 vLLM 的策略更接近：

```text
prompt prefill prefix -> 可以进入 prefix cache
new decode tokens -> 先不进入 prefix cache
后续请求重新 prefill 形成完整 block 后 -> 再纳入 prefix cache
```

### 写回去重

V1 写回前会做去重，避免重复 D2H：

```text
CPU 已经有相同 prefix block
-> skip writeback

相同 prefix block 正在 pending writeback
-> skip writeback

CPU 没有且 pending 也没有
-> 发起 async D2H writeback
```

当前去重依据是 block hash + token_ids 校验：

- `BlockManager.cpu_hash_to_token_ids` 保存 CPU prefix cache 的元数据。
- `Scheduler._pending_writeback_hashes()` 保存正在写回的 prefix hash。
- `ModelRunner.cpu_prefix_cache` 保存真正的 CPU KV tensor。

### Pending / Protected Block

异步写回通过独立 CUDA copy stream 发起：

```text
copy_stream:
  cpu_block.copy_(gpu_block, non_blocking=True)
  done_event.record()
```

Python 侧没有使用回调，而是由 scheduler 轮询 CUDA event：

```text
done_event.query()       # 非阻塞检查
done_event.synchronize() # 必要时阻塞等待
```

写回未完成前，对应 GPU block 处于 protected 状态，不能回到 free list。否则新的 request 可能覆盖这个 block，导致 CPU copy 读到错误 KV。

当前状态由几处结构共同表达：

- GPU resident: `BlockManager.hash_to_block_id`
- CPU resident: `BlockManager.cpu_hash_to_token_ids` + `ModelRunner.cpu_prefix_cache`
- writeback pending: `Scheduler.pending_prefix_writebacks`
- per-request writeback started: `Sequence.prefix_writeback_started`

### Request 读入

调度 waiting request 时，`BlockManager.get_allocate_plan()` 查找从 prompt 开头开始的最长连续 prefix：

```text
1. GPU hit: 直接引用已有 GPU block
2. CPU hit: 分配新的 GPU block，加入 restore_entries
3. miss: 停止 prefix lookup，剩余 token 正常 prefill
```

如果存在 `restore_entries`，scheduler 会调用 `ModelRunner.restore_prefix_blocks()` 做同步 H2D restore。restore 完成后，才继续 prefill 剩余 query suffix。

### 抢占语义

当前 preemption 不是主实验路径，只保留作 sanity。开启 CPU prefix offload 时，抢占会处理 pending writeback 的保护：

```text
request 被抢占
-> 如果某些 blocks 正在 D2H writeback，标记 release_on_complete
-> 释放其他 blocks
-> pending blocks 等 D2H 完成后再释放
```

因此不会强制释放正在写回的 block。若写回较慢并导致显存暂时不足，V1 选择等待 pending writeback 完成，优先保证正确性。

`recompute_pending_tokens` 在 CPU prefix offload 场景下按 V1 invariant 简化：完整 prefix blocks 已经在 CPU 上或正在 pending writeback，不计入 recompute；只有 decode 阶段最后一个未满 block 计入。

之前考虑过逐 block 检查 CPU backing / pending 状态。那种写法能覆盖 chunked prefill 未完成、hash 状态异常、后续 eviction/prefetch 引入不一致等复杂情况，但当前 V1 先保持更清晰的语义：prefill 完立即写回完整 prefix block，decode 新 tokens 暂不纳入 prefix cache。

### 已验证 Smoke Test

#### 重复 prompt 去重

同一个 prompt 连续请求两次，第二次不再重复写回同一个 prefix block：

```text
cpu_prefix_writeback_count = 1
cpu_prefix_d2h_bytes = 29360128
```

说明 CPU 已有 backing copy 时，写回前去重生效。

#### CPU restore

使用小 GPU cache 跑：

```text
D0 -> D1 -> D0
```

第三次 `D0` 发生 GPU miss，但 CPU prefix cache 命中并 restore：

```text
cpu_prefix_writeback_count = 2
cpu_prefix_cache_hit_count = 1
cpu_prefix_restore_count = 1
cpu_prefix_restored_token_count = 256
```

说明 GPU miss + CPU hit -> H2D restore -> skip prefix prefill 的路径已经可用。

## 后续 V2 方向

### Scheduler-Aware Prefetch

V2 再做调度协同：调度当前 request 后，如果 GPU KV 还有空间，可以提前把后续请求可能需要的 CPU-resident KV copy 回 GPU。

```text
current request running
GPU has spare blocks
next likely request has CPU KV but no GPU KV
-> async H2D prefetch
```

### OPT-like Eviction

驱逐时 inspect 当前 benchmark/scheduler 队列，选择未来窗口内最晚再次访问的 prefix block：

```text
evict block whose next use is farthest in the future
```

这可以作为研究型 oracle policy，用于评估调度和 cache 协同的收益上界。

### Pending Writeback Backpressure

V1 为了正确性会保护 pending blocks。V2 可以进一步控制这件事：

- 限制 pending writeback 的 block 数或总 bytes。
- 优先写回 hot prefix。
- 当显存紧张时优先等待最早完成的 pending copy。
- 暂不建议强制释放正在 D2H 的 block，除非同时设计安全的 abort/ignore 机制。

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


## V1 CPU Prefix Cache Smoke 结果

本轮结果目录：

```text
results/prefix_cache_cases/v1_smoke/
```

| case | mode | prefix reused tokens | CPU restored tokens | estimated recomputed prefix tokens | prefill tokens | query elapsed |
|---|---|---:|---:|---:|---:|---:|
| cascade_tile | GPU-only | 0 | 0 | 8192 | 8704 | 1.79 s |
| cascade_tile | CPU V1 | 0 | 8192 | 0 | 512 | 1.90 s |
| hot_cold_sharing | GPU-only | 25344 | 0 | 7424 | 9472 | 7.07 s |
| hot_cold_sharing | CPU V1 | 25344 | 7424 | 0 | 2048 | 7.21 s |
| branching_prefix_sharing | GPU-only | 52736 | 0 | 4608 | 8192 | 12.59 s |
| branching_prefix_sharing | CPU V1 | 52736 | 4608 | 0 | 3584 | 12.88 s |

观察：

- V1 的 CPU backing store 已能在 GPU miss 后 restore prefix KV，并跳过对应 prefix prefill。
- 三个 case 中，GPU-only 的 prefix recompute 都被 CPU V1 消除。
- 当前 V1 已将 request-finished writeback 改为 `copy_stream + CUDA event` 异步 D2H；完成前 GPU blocks 会被延迟释放。
- CPU hit 后的 H2D restore 仍是同步等待，因为当前 request 继续 prefill 前必须保证 KV 已在 GPU。
- 下一步性能优化重点是 clean block eviction、减少重复 writeback、scheduler-aware prefetch。

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
3. request 完成时通过 `copy_stream + CUDA event` 异步写回 CPU，并在完成前延迟释放对应 GPU blocks。
4. GPU miss + CPU hit 时同步 restore 到 GPU，并跳过对应 prefix prefill。
5. eviction 优先驱逐已经 CPU-resident 的 clean GPU blocks；否则 LRU 同步写回。
6. 后续实现 scheduler-aware OPT eviction 和 async prefetch。
7. 用 `cascade_tile`、`hot_cold_sharing`、`branching_prefix_sharing` 对比 GPU-only 与 GPU+CPU。
