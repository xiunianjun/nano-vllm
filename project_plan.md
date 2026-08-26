# nano-vLLM KV Cache Offloading Research Plan

## 1. Research Question

研究场景限定为：

> **模型权重本身可以完整驻留 GPU，但随着 serving concurrency 或 context length 增长，KV Cache working set 超过剩余 HBM 容量。此时，应该如何利用 CPU memory 扩展 KV Cache capacity，并通过 scheduler 与 memory manager 的协同降低 offloading 带来的性能损失？**

本项目只研究 **KV Cache memory management**，不研究 weight offloading，也暂时不考虑 SSD。

核心对象为：

```text
GPU HBM
├── Model weights          # 始终驻留 GPU
├── Runtime workspace
├── Activations
└── KV Cache               # 本项目唯一主动管理/限制的对象
        ↕
CPU pinned memory
```

---

## 2. Experimental Setup

### Model

使用：

```text
Qwen3-8B
```

主要实验在：

```text
1 × NVIDIA H200
```

上进行。【建议用第二张，这个环境目前是两张】

暂时不使用双卡 tensor parallel，以避免引入 TP、跨卡通信、KV shard 等额外变量。

### Simulating KV Memory Pressure

人为限制 nano-vLLM 可使用的 **GPU KV Cache capacity**。

具体通过显式指定：

```text
num_kvcache_blocks
```

实现。

例如设置不同的 KV Cache budget：

```text
8 GB
16 GB
32 GB
64 GB
```

对应不同数量的 KV blocks。

因此实验语义是：

> **模型完整驻留 H200，只人为限制 serving engine 分配给 KV Cache 的 HBM quota。**

目标是构造：

```text
KV working set > GPU KV capacity
```

从而稳定地产生 KV memory pressure。

---

# Part I. KV Offloading Without Prefix Sharing

## 3. Goal

第一阶段先隔离最基础的问题：

> 当 GPU KV Cache 不足时，把 KV Cache 丢弃并在之后 recompute，和把 KV Cache 暂存到 CPU 再恢复相比，性能差异如何？

为了排除 prefix reuse 的影响：

* 关闭 prefix sharing；或
* 直接使用独立随机输入，使不同 request 之间几乎不存在 exact prefix reuse。

Benchmark 初期直接基于 nano-vLLM 自带的：

```text
bench.py
```

进行修改。

---

## 4. Version Evolution

### V1 — Recompute Baseline

保持 nano-vLLM 原始策略。

当 GPU KV Cache 空间不足时：

```text
running request
      ↓
KV capacity insufficient
      ↓
preempt victim request
      ↓
discard its KV Cache
      ↓
request returns to waiting queue
      ↓
scheduled again later
      ↓
recompute / prefill lost KV
```

该版本作为 baseline。

需要测量：

* throughput；
* request latency；
* preemption count；
* recomputed token count；
* GPU KV occupancy。

---

### V2 — Synchronous CPU Offloading

加入 CPU KV Cache tier。

当需要驱逐一个 request 的 KV 时：

```text
GPU KV
   ↓ synchronous D2H
CPU pinned memory
```

request 恢复运行时：

```text
CPU KV
   ↓ synchronous H2D
GPU KV
   ↓
resume execution
```

第一版使用简单的：

```text
LRU victim selection
```

作为 eviction policy。

V2 的目的不是追求最终性能，而是回答最基础的问题：

> **CPU swap 的数据迁移成本是否低于直接丢弃 KV 后的 recomputation 成本？**

需要额外记录：

* D2H bytes；
* H2D bytes；
* swap-out count；
* swap-in count；
* swap-out latency；
* swap-in latency；
* CPU KV occupancy。

---

### V3 — Asynchronous Eviction

将 V2 的 synchronous swap-out 改为 asynchronous swap-out。

例如：

```text
request A execution
████████████████

                 request A becomes unlikely to run soon
                         ↓

compute stream:
████████████████████████████

copy stream:
             █████ D2H(A KV)
```

目标是在 GPU 仍执行其他 request 时，把未来可能需要驱逐的 KV 提前搬到 CPU。

需要注意：

> 发起异步 D2H 后，对应 GPU KV block 不能立即被重新分配。只有 D2H copy 完成后，该 GPU block 才真正可以释放。

因此需要显式维护 block/request residency state，例如：

```text
GPU_RESIDENT
EVICTING
CPU_RESIDENT
PREFETCHING
```

V3：

```text
swap-out: asynchronous
swap-in : synchronous
```

目的是单独验证：

> **D2H eviction 与 GPU computation overlap 能带来多少收益？**

---

### V4 — Scheduler-Memory Coordination

在 V3 基础上进一步加入：

```text
asynchronous eviction
+
asynchronous prefetch
+
scheduler-memory coordination
```

核心思想：

> Scheduler 已经知道 waiting queue 中哪些 request 可能即将被调度，因此 memory manager 不应该等 request 真正被 schedule 后才开始 H2D。

例如：

```text
Waiting Queue:
A
B
C
D

A KV currently in CPU
```

scheduler 预计 A 很快会运行时：

```text
CPU KV(A)
     ↓ async H2D
GPU
```

并尽可能使：

```text
prefetch completion
        ↓
request becomes runnable
```

而不是：

```text
request scheduled
        ↓
blocking H2D
        ↓
execution
```

因此 V4 的核心是：

> **利用 scheduler 的未来 request information 提前完成 KV movement。**

需要额外记录：

* prefetch count；
* prefetch hit rate；
* blocking swap-in count；
* wasted prefetch count；
* waiting time caused by KV migration。

---

# Part II. Prefix Sharing + CPU Offloading

## 5. Motivation

第二阶段保留 nano-vLLM 的 prefix sharing / prefix caching。

研究问题变为：

> Prefix caching 已经消除了重复 KV，但在去重后的 KV working set 仍然超过 GPU HBM 时，CPU offloading 应该如何与 prefix reuse 协同？

这里不能再简单地把：

```text
request == KV object
```

因为多个 request 可以共享同一批 prefix KV blocks。

例如：

```text
Request A:
[B1][B2][B3][A4]

Request B:
[B1][B2][B3][B4]

Request C:
[B1][B2][C3][C4]
```

其中：

```text
B1 / B2
```

可能被多个 request 同时复用。

因此：

* scheduler 的调度对象仍然是 **request**；
* cache residency / eviction 的管理对象应该是 **KV block**。

---

## 6. Version Evolution with Prefix Sharing

### V1 — Prefix Caching Baseline

使用 nano-vLLM 原始 prefix caching：

```text
prefix caching enabled
CPU offloading disabled
```

作为 baseline。

---

### V2 — Prefix Caching + Naive CPU Offloading

打开 Part I 中实现的 CPU offloading。

此版本只实现基本的 block-level offloading，不显式利用未来 prefix reuse 信息。

例如采用：

```text
block-level LRU
```

决定 GPU block victim。

目标是回答：

> 单纯把 CPU tier 加到 prefix caching 下面，能够获得多少收益？

---

### V3 — Prefix-Reuse-Aware Scheduling and Cache Management

进一步让 scheduler 和 cache manager 协同利用 prefix reuse 信息。

这里需要区分两个粒度。

#### Scheduler：request level

Scheduler 决定：

```text
哪个 request 接下来运行？
```

可以考虑：

* 该 request 有多少 prefix blocks 已经 GPU-resident；
* 有多少 prefix blocks CPU-resident；
* 恢复该 request 需要多少 H2D traffic；
* waiting queue 中其他 request 是否共享这些 blocks。

#### Cache Manager：block level

Cache manager 决定：

```text
哪些 KV blocks 应该 retain / evict / prefetch？
```

不能简单采用 request-level LRU。

例如：

```text
Block X:
currently shared by / reusable by many requests

Block Y:
only useful to one request
```

即使 X 的访问时间更老，也可能比 Y 更值得留在 GPU。

第一版无需设计复杂 cost model。

可以先采用简单的：

```text
waiting queue 中近期 request 会使用的 block
                ↓
           mark as protected

其他 blocks
                ↓
              LRU
```

也就是说：

> **scheduler 提供 future reuse hints，BlockManager 根据这些 hints 做 block-level eviction 和 prefetch。**

后续再考虑加入：

* reuse count；
* reference count；
* reuse distance；
* migration cost；

形成更复杂的 eviction score。

---

# Part III. Workloads

## 7. Workload A — Random Long Requests

用于 Part I。

直接修改 nano-vLLM `bench.py`，生成不同 request 独立的随机 tokens：

```text
Request A: random tokens
Request B: different random tokens
Request C: different random tokens
...
```

使：

```text
prefix hit rate ≈ 0
```

建议控制：

```text
Model: Qwen3-8B

Prompt length:
4K / 8K / 16K

Output length:
固定，例如 256 / 512

Number of requests:
32 / 64 / 128

GPU KV budget:
8 / 16 / 32 / 64 GB
```

第一步先找到：

> **KV capacity 从哪个位置开始导致明显 preemption / throughput degradation。**

即 memory-pressure knee point。

---

## 8. Workload B — ShareGPT Multi-Turn

用于 Part II。

从 ShareGPT 中选择较长的 conversation。

将一个 conversation：

```text
User1
Assistant1
User2
Assistant2
User3
Assistant3
...
```

展开成多个 request：

```text
A1
A2
A3
...
```

其中：

```text
A1 = conversation A 的较早阶段

A2 = A1
   + 后续 assistant response
   + 新 user message

A3 = A2
   + 后续 assistant response
   + 新 user message
```

因此天然形成：

```text
A1 prefix of A2
A2 prefix of A3
```

从而产生真实的 prefix reuse。

---

## 9. Interleaving Conversations

不能按：

```text
A1 A2 A3 A4
B1 B2 B3 B4
```

顺序执行。

这种 workload temporal locality 太强，GPU prefix cache 本身就很容易命中。

应该将多个 conversations interleave，例如：

```text
A1 B1 C1 D1
A2 B2 E1 C2
A3 F1 B3 ...
```

这样：

* 同一 conversation 内存在 prefix sharing；
* 不同 conversation 之间基本不存在 prefix sharing；
* 同一个 prefix 再次被使用前存在一定 reuse distance。

---

## 10. Locality Control

人为构造三档 reuse locality：

```text
High locality
Medium locality
Low locality
```

本质上通过控制：

```text
distance(A1, A2)
distance(A2, A3)
```

实现。

例如：

```text
High locality:
A1 B1 A2 C1 A3 ...

Medium locality:
A1 B1 C1 D1 A2 ...

Low locality:
A1 B1 C1 D1 E1 F1 G1 ... A2
```

这样可以观察：

> reuse distance 如何影响 GPU retention、CPU offloading 和 scheduler-aware prefetch 的收益。

---

# Part IV. Metrics

## 11. Required Metrics

不要只记录最终 throughput。

至少记录：

### Serving Performance

```text
total execution time
throughput
request latency
TTFT（如果方便）
```

### KV Memory

```text
GPU KV occupancy
CPU KV occupancy
peak occupancy
```

### Baseline Recomputation

```text
preemption count
recomputed token count
```

### CPU Offloading

```text
swap-out count
swap-in count

D2H bytes
H2D bytes

D2H time
H2D time

blocking migration time
```

### Async Coordination

```text
prefetch count
prefetch hit rate
blocking swap-in count
wasted prefetch count
```

### Prefix Sharing

```text
prefix cache hit rate
number of shared blocks
GPU-resident reusable blocks
CPU-resident reusable blocks
```

---

# Part V. Scope and Implementation Principles

## 12. Current Scope

当前版本只研究：

```text
GPU HBM
   ↕
CPU pinned memory
```

明确不做：

```text
SSD offloading
weight offloading
tensor parallelism
multi-GPU KV transfer
RDMA
KV quantization
custom attention kernels
```

除非 CPU-only 方案已经完整实现和验证，否则不要扩展 scope。

---
