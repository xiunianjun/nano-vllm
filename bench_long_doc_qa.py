import argparse
import json
import math
import random
import time

import torch
from transformers import AutoConfig, AutoTokenizer

from nanovllm import LLM, SamplingParams


GB = 1 << 30


def parse_args():
    parser = argparse.ArgumentParser(description="LMCache-style long document QA baseline for nano-vLLM GPU prefix cache.")
    parser.add_argument("--model", default="/data/datasets/models-hf/Qwen3-8B")
    parser.add_argument("--document-length", type=int, default=2048)
    parser.add_argument("--query-length", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=8)
    parser.add_argument("--num-documents", type=int, default=None)
    parser.add_argument("--target-working-set-gb", type=float, default=2.0)
    parser.add_argument("--gpu-kv-cache-gb", type=float, default=1.0)
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--repeat-mode", choices=("tile", "random", "interleave"), default="tile")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--kvcache-block-size", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-tqdm", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def model_dtype(hf_config):
    dtype = getattr(hf_config, "dtype", None) or getattr(hf_config, "torch_dtype", torch.bfloat16)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype.removeprefix("torch."))
    return dtype


def kv_bytes_per_token(hf_config, tensor_parallel_size=1):
    dtype = model_dtype(hf_config)
    num_kv_heads = hf_config.num_key_value_heads // tensor_parallel_size
    head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
    return 2 * hf_config.num_hidden_layers * num_kv_heads * head_dim * dtype.itemsize


def kv_cache_gb_to_blocks(hf_config, kv_cache_gb, block_size, tensor_parallel_size=1):
    block_bytes = kv_bytes_per_token(hf_config, tensor_parallel_size) * block_size
    return max(1, int(kv_cache_gb * GB // block_bytes))


def num_documents_for_working_set(hf_config, target_gb, document_length):
    doc_bytes = kv_bytes_per_token(hf_config) * document_length
    return max(1, math.ceil(target_gb * GB / doc_bytes))


def repeat_doc_ids(doc_ids, repeat_count, mode, seed):
    if mode == "tile":
        return doc_ids * repeat_count
    if mode == "interleave":
        out = []
        for doc_id in doc_ids:
            out.extend([doc_id] * repeat_count)
        return out
    out = doc_ids * repeat_count
    rng = random.Random(seed)
    rng.shuffle(out)
    return out


def choose_token_ids(tokenizer):
    vocab_size = len(tokenizer)
    hi_id = tokenizer.encode(" hi", add_special_tokens=False)[0]
    warm_query_id = tokenizer.encode(" warm", add_special_tokens=False)[0]
    measured_query_id = tokenizer.encode(" query", add_special_tokens=False)[0]
    doc_base_id = min(1000, vocab_size - 4096)
    if doc_base_id < 0:
        doc_base_id = 1
    return vocab_size, hi_id, warm_query_id, measured_query_id, doc_base_id


def make_prompt(doc_id, args, ids, warmup):
    vocab_size, hi_id, warm_query_id, measured_query_id, doc_base_id = ids
    doc_marker = (doc_base_id + doc_id) % vocab_size
    document = [doc_marker] + [hi_id] * (args.document_length - 1)
    query_marker = warm_query_id if warmup else measured_query_id
    query = [query_marker] + [hi_id] * (args.query_length - 1)
    return document + query


def run_round(llm, prompts, sampling_params, use_tqdm):
    per_request_latencies = []
    round_start = time.perf_counter()
    for prompt in prompts:
        start = time.perf_counter()
        llm.generate([prompt], sampling_params, use_tqdm=use_tqdm)
        per_request_latencies.append(time.perf_counter() - start)
    elapsed = time.perf_counter() - round_start
    return elapsed, per_request_latencies


def summarize_latencies(latencies):
    if not latencies:
        return {"avg": 0.0, "min": 0.0, "max": 0.0}
    return {
        "avg": sum(latencies) / len(latencies),
        "min": min(latencies),
        "max": max(latencies),
    }


def main():
    args = parse_args()
    hf_config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    ids = choose_token_ids(tokenizer)

    num_documents = args.num_documents
    if num_documents is None:
        num_documents = num_documents_for_working_set(hf_config, args.target_working_set_gb, args.document_length)

    num_kvcache_blocks = kv_cache_gb_to_blocks(hf_config, args.gpu_kv_cache_gb, args.kvcache_block_size)
    max_model_len = args.max_model_len or args.document_length + args.query_length + args.output_len + 8
    max_num_batched_tokens = args.max_num_batched_tokens or max_model_len
    prompt_length = args.document_length + args.query_length
    blocks_per_prompt = math.ceil(prompt_length / args.kvcache_block_size)
    if num_kvcache_blocks < blocks_per_prompt:
        raise ValueError(
            f"GPU KV cache is too small for one prompt: {num_kvcache_blocks} blocks < {blocks_per_prompt} blocks. "
            f"Increase --gpu-kv-cache-gb or reduce --document-length/--query-length."
        )

    doc_ids = list(range(num_documents))
    warmup_prompts = [make_prompt(doc_id, args, ids, warmup=True) for doc_id in doc_ids]
    measured_doc_ids = repeat_doc_ids(doc_ids, args.repeat_count, args.repeat_mode, args.shuffle_seed)
    measured_prompts = [make_prompt(doc_id, args, ids, warmup=False) for doc_id in measured_doc_ids]
    sampling_params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=args.output_len)

    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        max_model_len=max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kvcache_block_size=args.kvcache_block_size,
        num_kvcache_blocks=num_kvcache_blocks,
        enable_prefix_cache=True,
        enable_cpu_kv_offload=False,
    )

    llm.generate([[ids[1]]], SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=1), use_tqdm=False)
    warmup_elapsed, warmup_latencies = run_round(llm, warmup_prompts, sampling_params, args.use_tqdm)
    llm.reset_metrics()
    query_elapsed, query_latencies = run_round(llm, measured_prompts, sampling_params, args.use_tqdm)
    metrics = llm.get_metrics()
    llm.exit()

    bytes_per_token = kv_bytes_per_token(hf_config)
    working_set_gb = num_documents * args.document_length * bytes_per_token / GB
    gpu_cache_gb_actual = num_kvcache_blocks * args.kvcache_block_size * bytes_per_token / GB
    total_document_tokens = len(measured_prompts) * args.document_length
    cached_doc_tokens_upper_bound = min(metrics["prefix_cache_reused_token_count"], total_document_tokens)
    document_recomputed_tokens_est = total_document_tokens - cached_doc_tokens_upper_bound

    result = {
        "model": args.model,
        "mode": "gpu_prefix_cache_recompute_baseline",
        "repeat_mode": args.repeat_mode,
        "repeat_count": args.repeat_count,
        "num_documents": num_documents,
        "document_length": args.document_length,
        "query_length": args.query_length,
        "output_len": args.output_len,
        "kv_bytes_per_token": bytes_per_token,
        "target_working_set_gb": args.target_working_set_gb,
        "working_set_gb_actual": working_set_gb,
        "gpu_kv_cache_gb_requested": args.gpu_kv_cache_gb,
        "gpu_kv_cache_gb_actual": gpu_cache_gb_actual,
        "num_kvcache_blocks": num_kvcache_blocks,
        "blocks_per_prompt": blocks_per_prompt,
        "warmup_requests": len(warmup_prompts),
        "warmup_elapsed_sec": warmup_elapsed,
        "warmup_latency_sec": summarize_latencies(warmup_latencies),
        "query_requests": len(measured_prompts),
        "query_elapsed_sec": query_elapsed,
        "query_latency_sec": summarize_latencies(query_latencies),
        "total_document_tokens": total_document_tokens,
        "document_recomputed_tokens_est": document_recomputed_tokens_est,
    }
    result.update(metrics)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
