import argparse
import json
import time
from random import randint, seed

import torch
from transformers import AutoConfig

from nanovllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Recompute baseline benchmark.")
    parser.add_argument("--model", default="/data/datasets/models-hf/Qwen3-0.6B")
    parser.add_argument("--num-seqs", type=int, default=32)
    parser.add_argument("--min-input-len", type=int, default=100)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--min-output-len", type=int, default=1)
    parser.add_argument("--max-output-len", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--kvcache-block-size", type=int, default=256)
    parser.add_argument("--num-kvcache-blocks", type=int, default=-1)
    parser.add_argument("--kv-cache-gb", type=float, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--enable-prefix-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-tqdm", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def kv_cache_gb_to_blocks(model: str, kv_cache_gb: float, block_size: int, tensor_parallel_size: int = 1) -> int:
    hf_config = AutoConfig.from_pretrained(model)
    dtype = getattr(hf_config, "dtype", None) or getattr(hf_config, "torch_dtype", torch.bfloat16)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype.removeprefix("torch."))
    num_kv_heads = hf_config.num_key_value_heads // tensor_parallel_size
    head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
    block_bytes = 2 * hf_config.num_hidden_layers * block_size * num_kv_heads * head_dim * dtype.itemsize
    return int(kv_cache_gb * (1 << 30) // block_bytes)


def make_requests(args, seed_offset: int = 0):
    seed(args.seed + seed_offset)
    prompts = [[randint(0, 10000) for _ in range(randint(args.min_input_len, args.max_input_len))] for _ in range(args.num_seqs)]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(args.min_output_len, args.max_output_len))
        for _ in range(args.num_seqs)
    ]
    return prompts, sampling_params


def main():
    args = parse_args()
    seed(args.seed)
    num_kvcache_blocks = args.num_kvcache_blocks
    if args.kv_cache_gb is not None:
        num_kvcache_blocks = kv_cache_gb_to_blocks(args.model, args.kv_cache_gb, args.kvcache_block_size)

    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kvcache_block_size=args.kvcache_block_size,
        num_kvcache_blocks=num_kvcache_blocks,
        enable_prefix_cache=args.enable_prefix_cache,
    )

    llm.generate([[0]], SamplingParams(max_tokens=1), use_tqdm=False)
    for i in range(args.warmup_iters):
        warmup_prompts, warmup_sampling_params = make_requests(args, seed_offset=1000 + i)
        llm.generate(warmup_prompts, warmup_sampling_params, use_tqdm=False)
    llm.reset_metrics()

    prompts, sampling_params = make_requests(args)
    start = time.time()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=args.use_tqdm)
    elapsed = time.time() - start

    total_output_tokens = sum(len(output["token_ids"]) for output in outputs)
    metrics = llm.get_metrics()
    metrics.update({
        "num_seqs": args.num_seqs,
        "total_output_tokens": total_output_tokens,
        "elapsed_sec": elapsed,
        "throughput_tok_per_sec": total_output_tokens / elapsed,
        "num_kvcache_blocks": num_kvcache_blocks,
    })
    print(json.dumps(metrics, indent=2, sort_keys=True))
    llm.exit()


if __name__ == "__main__":
    main()
