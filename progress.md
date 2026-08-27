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

## 下一步

开始 V2 synchronous CPU KV offloading：

```text
GPU KV -> CPU pinned memory on eviction
CPU pinned memory -> GPU KV on resume
```

第一组核心对比应该是：

```text
V1 recompute cost vs V2 synchronous swap cost
```
