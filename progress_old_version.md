# nano-vLLM KV Offloading 阶段进展

## 当前阶段

V1 recompute baseline 已经实现，并完成了 smoke test 和一轮长序列复测。

该 baseline 保持 nano-vLLM 在 GPU KV 压力下的原始行为：

```text
KV capacity insufficient
-> preempt a running request
-> discard its GPU KV cache
-> move it back to waiting
-> prefill again later
```

也就是：GPU KV Cache 不够时，抢占 running request，丢弃它的 KV，之后重新调度时重新 prefill/recompute。

## 环境

- GPU: NVIDIA H200 NVL
- baseline 主测试模型: `/data/datasets/models-hf/Qwen3-8B`
- 小模型 smoke test: `/data/datasets/models-hf/Qwen3-0.6B`
- ShareGPT 数据: `data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json`
- 当前可用 FlashAttention 环境: `.venv-fa28`

FlashAttention 环境版本：

```text
venv = .venv-fa28
python = 3.10.19
torch = 2.8.0+cu128
flash-attn = 2.8.3.post1
```

说明：之前为了避免服务器被 `flash-attn` 源码编译打满，曾临时使用 PyTorch SDPA fallback。现在 `.venv-fa28` 中已经安装了匹配的 `flash-attn` wheel，后续性能测试优先使用该环境。

## 已实现指标

V1 benchmark 当前记录：

- `throughput_tok_per_sec`
- `request_latency_avg`
- `request_latency_min`
- `request_latency_max`
- `preemption_count`
- `recomputed_token_count`
- `total_output_tokens`
- `elapsed_sec`
- `num_kvcache_blocks`

当前不记录：

- GPU KV occupancy 相关指标

## Benchmark 入口

baseline benchmark 脚本：

```bash
bench_baseline.py
```

脚本支持通过下面参数显式控制 GPU KV capacity：

```bash
--num-kvcache-blocks
```

或者：

```bash
--kv-cache-gb
```

脚本会在正式计时前运行同形状 warmup，用于降低冷启动和首次 shape/kernel 初始化带来的噪声。

## 最新短序列 Baseline 测试

两组测试使用同一个 workload：

```text
model = /data/datasets/models-hf/Qwen3-8B
num_seqs = 2
input_len = 250 tokens/request
output_len = 16 tokens/request
total_output_tokens = 32
max_model_len = 320
GPU = CUDA_VISIBLE_DEVICES=1
```

### 不发生 Recompute

```text
num_kvcache_blocks = 4
preemption_count = 0
recomputed_token_count = 0
throughput_tok_per_sec = 37.11
request_latency_avg = 0.86 s
elapsed_sec = 0.86 s
```

### 发生 Recompute

```text
num_kvcache_blocks = 2
preemption_count = 1
recomputed_token_count = 256
throughput_tok_per_sec = 25.89
request_latency_avg = 1.02 s
elapsed_sec = 1.24 s
```

## 复现命令

不发生 recompute：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench_baseline.py \
  --model /data/datasets/models-hf/Qwen3-8B \
  --num-seqs 2 \
  --min-input-len 250 \
  --max-input-len 250 \
  --min-output-len 16 \
  --max-output-len 16 \
  --max-model-len 320 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 512 \
  --num-kvcache-blocks 4 \
  --warmup-iters 1 \
  --enforce-eager \
  --no-use-tqdm
```

发生 recompute：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench_baseline.py \
  --model /data/datasets/models-hf/Qwen3-8B \
  --num-seqs 2 \
  --min-input-len 250 \
  --max-input-len 250 \
  --min-output-len 16 \
  --max-output-len 16 \
  --max-model-len 320 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 512 \
  --num-kvcache-blocks 2 \
  --warmup-iters 1 \
  --enforce-eager \
  --no-use-tqdm
```

## 长序列 Baseline Sweep

这组测试使用 FlashAttention 环境：

```text
venv = .venv-fa28
torch = 2.8.0+cu128
flash-attn = 2.8.3.post1
prefix cache = disabled
```

第一次长序列 sweep 暴露了一个重要问题：nano-vLLM 的 prefix cache 会通过 hash 复用已经释放的 KV block，所以被 preempt 的 request 重新调度时，可能不会真正 recompute 完整 prefix。为了符合 V1 `without prefix sharing` 的实验设定，`bench_baseline.py` 当前默认关闭 prefix cache。

所有行使用同一个 workload：

```text
model = /data/datasets/models-hf/Qwen3-8B
num_seqs = 2
output_len = 32 tokens/request
GPU = CUDA_VISIBLE_DEVICES=1
warmup_iters = 1
```

| input_len | 场景 | KV blocks | preemptions | recomputed tokens | TPS | avg latency | avg latency overhead | tail latency overhead |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1020 | fit | 10 | 0 | 0 | 52.63 | 1.22 s | 0.0% | 0.0% |
| 1020 | recompute | 8 | 1 | 1024 | 39.82 | 1.25 s | +2.5% | +32.2% |
| 2044 | fit | 18 | 0 | 0 | 62.60 | 1.02 s | 0.0% | 0.0% |
| 2044 | recompute | 16 | 1 | 2048 | 36.82 | 1.35 s | +32.4% | +70.0% |
| 3068 | fit | 26 | 0 | 0 | 60.98 | 1.05 s | 0.0% | 0.0% |
| 3068 | recompute | 24 | 1 | 3072 | 36.43 | 1.37 s | +30.8% | +67.4% |

观察：关闭 prefix cache 后，`recomputed_token_count` 会随序列长度增长。由于当前每组只有 2 个 request，且只有其中 1 个被 preempt，`avg latency` 会被未抢占 request 稀释；`tail latency overhead` 更能体现 recompute 带来的尾延迟惩罚。

## V2 同步 CPU KV Offloading

V2 synchronous CPU offloading 已完成第一版最小实现。

当前策略：

```text
preempt victim request
-> 同步 D2H copy: GPU KV blocks -> CPU pinned memory
-> 释放 GPU KV blocks
-> request 回到 waiting queue
-> 重新调度时分配 GPU KV blocks
-> 同步 H2D copy: CPU pinned memory -> GPU KV blocks
-> 只 prefill 尚未写入 KV 的最后 token，继续 decode
```

当前实现仍使用 scheduler 原有的 victim 选择方式：decode 阶段 KV 不足时，优先 preempt running queue 尾部 request。这个行为接近当前 FCFS 队列下的 LRU-ish victim，后续可以单独替换成更明确的 LRU policy。

新增配置：

```text
enable_cpu_kv_offload: bool = False
```

benchmark 使用：

```bash
--enable-cpu-kv-offload
```

新增 V2 指标：

- `d2h_bytes`
- `h2d_bytes`
- `swap_out_count`
- `swap_in_count`
- `swap_out_latency_avg`
- `swap_in_latency_avg`
- `swap_out_latency_max`
- `swap_in_latency_max`
- `cpu_kv_bytes`
- `cpu_kv_bytes_peak`

### V1 vs V2 小对比

测试配置：

```text
model = /data/datasets/models-hf/Qwen3-8B
num_seqs = 2
input_len = 1020 tokens/request
output_len = 32 tokens/request
num_kvcache_blocks = 8
prefix cache = disabled
venv = .venv-fa28
GPU = CUDA_VISIBLE_DEVICES=1
```

| 模式 | preemptions | recomputed tokens | D2H | H2D | swap-out | swap-in | TPS | avg latency | tail latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 recompute | 1 | 1024 | 0 MB | 0 MB | 0 ms | 0 ms | 38.72 | 1.28 s | 1.65 s |
| V2 sync offload | 1 | 0 | 144 MB | 144 MB | 3.38 ms | 3.04 ms | 39.98 | 1.24 s | 1.60 s |

观察：在这组小 case 中，V2 成功把 `recomputed_token_count` 从 1024 降到 0；同步搬运 144 MB KV 的 D2H/H2D 总耗时约 6.4 ms。这个结果只说明 V2 数据路径已经跑通，正式结论还需要更大请求数、更多长度和重复实验。

## 更多 input length 的 V1/V2 对比

这组测试继续使用 FlashAttention 环境 `.venv-fa28`，并关闭 prefix cache。每个点只跑 1 次，因此主要用于观察趋势，后续正式结果需要重复实验取均值和方差。

固定配置：

```text
model = /data/datasets/models-hf/Qwen3-8B
num_seqs = 2
output_len = 32 tokens/request
warmup_iters = 1
GPU = CUDA_VISIBLE_DEVICES=1
```

每个 input length 都设置 KV capacity 让两个 prompt 刚好放得下，但 decode 跨 block 时触发 1 次 preemption。

| input_len | V1 recompute tokens | V2 swap size | swap-out | swap-in | V1 TPS | V2 TPS | TPS delta | V1 tail latency | V2 tail latency | tail delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 508 | 512 | 72 MB | 2.57 ms | 1.55 ms | 39.04 | 40.62 | +4.0% | 1.64 s | 1.58 s | -3.9% |
| 1020 | 1024 | 144 MB | 4.93 ms | 3.05 ms | 32.70 | 38.57 | +17.9% | 1.96 s | 1.66 s | -15.2% |
| 1532 | 1536 | 216 MB | 8.03 ms | 4.63 ms | 38.13 | 37.46 | -1.8% | 1.68 s | 1.71 s | +1.8% |
| 2044 | 2048 | 288 MB | 10.08 ms | 6.03 ms | 37.51 | 37.11 | -1.0% | 1.71 s | 1.72 s | +1.1% |
| 2556 | 2560 | 360 MB | 12.46 ms | 7.51 ms | 36.89 | 37.20 | +0.8% | 1.73 s | 1.72 s | -0.8% |
| 3068 | 3072 | 432 MB | 14.97 ms | 9.01 ms | 34.96 | 35.87 | +2.6% | 1.83 s | 1.78 s | -2.6% |
| 3580 | 3584 | 504 MB | 18.31 ms | 10.49 ms | 33.73 | 34.15 | +1.2% | 1.90 s | 1.87 s | -1.2% |

观察：V2 的 D2H/H2D 搬运量和 swap latency 基本随 input length 线性增长。当前单次结果下，V2 在多数长度上和 V1 接近或略优，但 1532/2044 两个点略差，说明小规模单次 benchmark 仍有噪声。更可靠的结论需要增加重复次数、更多并发和更强 KV pressure。

## 下一步

扩大 V1 recompute baseline 与 V2 synchronous CPU offloading 的对比实验：

```text
V1 recompute cost vs V2 synchronous swap cost
```

优先补充：

- 更多 `input_len`
- 更多 `num_seqs` / concurrency
- 多次重复实验，报告均值和方差
- 对比 `request_latency_max` 或 P95/P99 tail latency
