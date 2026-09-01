import argparse
import json
import math
import random
import statistics
import time

import torch
from transformers import AutoConfig, AutoTokenizer

from nanovllm import LLM, SamplingParams


GB = 1 << 30


def parse_args():
    parser = argparse.ArgumentParser(description="LMCache-style long document QA baseline for nano-vLLM GPU prefix cache.")
    parser.add_argument("--model", default="/data/datasets/models-hf/Qwen3-8B")
    parser.add_argument("--workload", choices=("long_doc_qa", "branching_prefix"), default="long_doc_qa")
    parser.add_argument("--document-length", type=int, default=2048)
    parser.add_argument("--query-length", type=int, default=64)
    parser.add_argument("--root-length", type=int, default=512)
    parser.add_argument("--branch-length", type=int, default=512)
    parser.add_argument("--branch-count", type=int, default=None)
    parser.add_argument("--output-len", type=int, default=8)
    parser.add_argument("--num-documents", type=int, default=None)
    parser.add_argument("--target-working-set-gb", type=float, default=2.0)
    parser.add_argument("--gpu-kv-cache-gb", type=float, default=1.0)
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--repeat-mode", choices=("tile", "random", "interleave", "hot_cold"), default="tile")
    parser.add_argument("--arrival-mode", choices=("batch", "poisson"), default="batch")
    parser.add_argument("--request-rate", type=float, default=None, help="Poisson arrival rate in requests/sec.")
    parser.add_argument("--arrival-seed", type=int, default=0)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--hot-documents", type=int, default=2)
    parser.add_argument("--hot-request-ratio", type=float, default=0.7)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--kvcache-block-size", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enable-cpu-kv-offload", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-gpu-lru-retention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-lazy-cpu-kv-writeback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lazy-writeback-watermark-ratio", type=float, default=0.5)
    parser.add_argument("--cpu-prefix-cache-gb-limit", type=float, default=0.0)
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


def branch_count_for_working_set(hf_config, target_gb, root_length, branch_length):
    bytes_per_token = kv_bytes_per_token(hf_config)
    target_tokens = target_gb * GB / bytes_per_token
    return max(1, math.ceil((target_tokens - root_length) / branch_length))


def repeat_doc_ids(doc_ids, repeat_count, mode, seed, hot_documents=2, hot_request_ratio=0.7):
    if mode == "tile":
        return doc_ids * repeat_count
    if mode == "interleave":
        out = []
        for doc_id in doc_ids:
            out.extend([doc_id] * repeat_count)
        return out
    if mode == "random":
        out = doc_ids * repeat_count
        rng = random.Random(seed)
        rng.shuffle(out)
        return out

    assert mode == "hot_cold"
    rng = random.Random(seed)
    hot_count = max(1, min(hot_documents, len(doc_ids)))
    hot_doc_ids = doc_ids[:hot_count]
    cold_doc_ids = doc_ids[hot_count:] or hot_doc_ids
    total_requests = len(doc_ids) * repeat_count
    out = []
    for _ in range(total_requests):
        if rng.random() < hot_request_ratio:
            out.append(rng.choice(hot_doc_ids))
        else:
            out.append(rng.choice(cold_doc_ids))
    return out


def choose_token_ids(tokenizer):
    vocab_size = len(tokenizer)
    hi_id = tokenizer.encode(" hi", add_special_tokens=False)[0]
    root_id = tokenizer.encode(" root", add_special_tokens=False)[0]
    warm_query_id = tokenizer.encode(" warm", add_special_tokens=False)[0]
    measured_query_id = tokenizer.encode(" query", add_special_tokens=False)[0]
    doc_base_id = min(1000, vocab_size - 4096)
    if doc_base_id < 0:
        doc_base_id = 1
    return vocab_size, hi_id, root_id, warm_query_id, measured_query_id, doc_base_id


def make_prompt(doc_id, args, ids, warmup, query_variant=0):
    vocab_size, hi_id, root_id, warm_query_id, measured_query_id, doc_base_id = ids
    query_marker = warm_query_id if warmup else measured_query_id
    variant_marker = (doc_base_id + 2048 + query_variant) % vocab_size
    if args.workload == "branching_prefix":
        branch_marker = (doc_base_id + doc_id) % vocab_size
        root = [root_id] * args.root_length
        branch = [branch_marker] + [hi_id] * (args.branch_length - 1)
        query = [query_marker, variant_marker] + [hi_id] * max(0, args.query_length - 2)
        return root + branch + query

    doc_marker = (doc_base_id + doc_id) % vocab_size
    document = [doc_marker] + [hi_id] * (args.document_length - 1)
    query = [query_marker, variant_marker] + [hi_id] * max(0, args.query_length - 2)
    return document + query


def record_step_metrics(llm, num_tokens, step_time):
    if num_tokens > 0:
        llm.step_metrics["prefill_step_count"] += 1
        llm.step_metrics["prefill_step_time_sec"] += step_time
        llm.step_metrics["prefill_token_count_timed"] += num_tokens
    else:
        llm.step_metrics["decode_step_count"] += 1
        llm.step_metrics["decode_step_time_sec"] += step_time
        llm.step_metrics["decode_token_count_timed"] += -num_tokens


def run_round(llm, prompts, sampling_params, use_tqdm):
    before = len(llm.request_latencies)
    round_start = time.perf_counter()
    llm.generate(prompts, [sampling_params] * len(prompts), use_tqdm=use_tqdm)
    elapsed = time.perf_counter() - round_start
    return elapsed, llm.request_latencies[before:], None


def poisson_arrival_offsets(num_requests, request_rate, seed):
    if request_rate is None or request_rate <= 0:
        raise ValueError("--request-rate must be positive when --arrival-mode=poisson")
    rng = random.Random(seed)
    offsets = []
    t = 0.0
    for i in range(num_requests):
        if i > 0:
            t += rng.expovariate(request_rate)
        offsets.append(t)
    return offsets


def run_poisson_round(llm, prompts, sampling_params, request_rate, seed):
    before = len(llm.request_latencies)
    arrival_offsets = poisson_arrival_offsets(len(prompts), request_rate, seed)
    round_start = time.perf_counter()
    next_request = 0
    finished = 0

    while finished < len(prompts):
        now = time.perf_counter()
        while next_request < len(prompts) and round_start + arrival_offsets[next_request] <= now:
            seq = llm.add_request(prompts[next_request], sampling_params)
            # Latency uses the planned client arrival time. If the engine is inside a long CUDA step,
            # requests that arrive during that step are charged queueing from their true arrival time.
            llm.request_start_times[seq.seq_id] = round_start + arrival_offsets[next_request]
            next_request += 1

        if llm.is_finished():
            if next_request < len(prompts):
                sleep_until = round_start + arrival_offsets[next_request]
                time.sleep(max(0.0, sleep_until - time.perf_counter()))
                continue
            break

        step_start = time.perf_counter()
        output, num_tokens = llm.step()
        record_step_metrics(llm, num_tokens, time.perf_counter() - step_start)
        for seq_id, _token_ids in output:
            finished += 1
            llm.request_latencies.append(time.perf_counter() - llm.request_start_times.pop(seq_id, step_start))

    elapsed = time.perf_counter() - round_start
    metadata = {
        "arrival_offsets_sec": arrival_offsets,
        "planned_arrival_span_sec": arrival_offsets[-1] if arrival_offsets else 0.0,
        "request_rate_target": request_rate,
        "request_rate_actual": len(prompts) / elapsed if elapsed else 0.0,
    }
    return elapsed, llm.request_latencies[before:], metadata


def summarize_latencies(latencies):
    if not latencies:
        return {"count": 0, "avg": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(latencies),
        "avg": sum(latencies) / len(latencies),
        "median": statistics.median(latencies),
        "min": min(latencies),
        "max": max(latencies),
    }


def main():
    args = parse_args()
    hf_config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    ids = choose_token_ids(tokenizer)

    if args.workload == "branching_prefix":
        num_documents = args.branch_count
        if num_documents is None:
            num_documents = branch_count_for_working_set(
                hf_config, args.target_working_set_gb, args.root_length, args.branch_length
            )
        reusable_prefix_length = args.root_length + args.branch_length
    else:
        num_documents = args.num_documents
        if num_documents is None:
            num_documents = num_documents_for_working_set(hf_config, args.target_working_set_gb, args.document_length)
        reusable_prefix_length = args.document_length

    num_kvcache_blocks = kv_cache_gb_to_blocks(hf_config, args.gpu_kv_cache_gb, args.kvcache_block_size)
    max_model_len = args.max_model_len or reusable_prefix_length + args.query_length + args.output_len + 8
    max_num_batched_tokens = args.max_num_batched_tokens or max_model_len
    prompt_length = reusable_prefix_length + args.query_length
    blocks_per_prompt = math.ceil(prompt_length / args.kvcache_block_size)
    if num_kvcache_blocks < blocks_per_prompt:
        raise ValueError(
            f"GPU KV cache is too small for one prompt: {num_kvcache_blocks} blocks < {blocks_per_prompt} blocks. "
            f"Increase --gpu-kv-cache-gb or reduce --document-length/--query-length."
        )

    doc_ids = list(range(num_documents))
    warmup_prompts = [make_prompt(doc_id, args, ids, warmup=True, query_variant=0) for doc_id in doc_ids]
    measured_doc_ids = repeat_doc_ids(
        doc_ids,
        args.repeat_count,
        args.repeat_mode,
        args.shuffle_seed,
        args.hot_documents,
        args.hot_request_ratio,
    )
    measured_prompts = [
        make_prompt(doc_id, args, ids, warmup=False, query_variant=i)
        for i, doc_id in enumerate(measured_doc_ids)
    ]
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
        enable_cpu_kv_offload=args.enable_cpu_kv_offload,
        enable_gpu_lru_retention=args.enable_gpu_lru_retention,
        enable_lazy_cpu_kv_writeback=args.enable_lazy_cpu_kv_writeback,
        lazy_writeback_watermark_ratio=args.lazy_writeback_watermark_ratio,
        cpu_prefix_cache_gb_limit=args.cpu_prefix_cache_gb_limit,
    )

    llm.generate([[ids[1]]], SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=1), use_tqdm=False)
    warmup_elapsed, warmup_latencies, _ = run_round(llm, warmup_prompts, sampling_params, args.use_tqdm)
    llm.reset_metrics()
    if args.arrival_mode == "poisson":
        query_elapsed, query_latencies, arrival_metadata = run_poisson_round(
            llm, measured_prompts, sampling_params, args.request_rate, args.arrival_seed
        )
    else:
        query_elapsed, query_latencies, arrival_metadata = run_round(llm, measured_prompts, sampling_params, args.use_tqdm)
    metrics = llm.get_metrics()
    llm.exit()

    bytes_per_token = kv_bytes_per_token(hf_config)
    if args.workload == "branching_prefix":
        working_set_tokens = args.root_length + num_documents * args.branch_length
    else:
        working_set_tokens = num_documents * args.document_length
    working_set_gb = working_set_tokens * bytes_per_token / GB
    gpu_cache_gb_actual = num_kvcache_blocks * args.kvcache_block_size * bytes_per_token / GB
    single_prompt_tokens_est = reusable_prefix_length + args.query_length + args.output_len
    single_prompt_kv_gb_est = single_prompt_tokens_est * bytes_per_token / GB
    total_document_tokens = len(measured_prompts) * reusable_prefix_length
    restored_tokens = metrics.get("cpu_prefix_cache_restored_token_count", 0)
    reused_or_restored = metrics["prefix_cache_reused_token_count"] + restored_tokens
    cached_doc_tokens_upper_bound = min(reused_or_restored, total_document_tokens)
    document_recomputed_tokens_est = total_document_tokens - cached_doc_tokens_upper_bound

    result = {
        "model": args.model,
        "mode": (
            "cpu_prefix_cache_v3_lazy_writeback"
            if args.enable_cpu_kv_offload and args.enable_gpu_lru_retention and args.enable_lazy_cpu_kv_writeback
            else "cpu_prefix_cache_v2_lru"
            if args.enable_cpu_kv_offload and args.enable_gpu_lru_retention
            else "cpu_prefix_cache_v1"
            if args.enable_cpu_kv_offload
            else "gpu_prefix_cache_recompute_baseline"
        ),
        "enable_cpu_kv_offload": args.enable_cpu_kv_offload,
        "enable_gpu_lru_retention": args.enable_gpu_lru_retention,
        "enable_lazy_cpu_kv_writeback": args.enable_lazy_cpu_kv_writeback,
        "lazy_writeback_watermark_ratio": args.lazy_writeback_watermark_ratio,
        "cpu_prefix_cache_gb_limit": args.cpu_prefix_cache_gb_limit,
        "workload": args.workload,
        "repeat_mode": args.repeat_mode,
        "repeat_count": args.repeat_count,
        "arrival_mode": args.arrival_mode,
        "request_rate_target": args.request_rate,
        "arrival_seed": args.arrival_seed,
        "hot_documents": args.hot_documents,
        "hot_request_ratio": args.hot_request_ratio,
        "measured_doc_ids": measured_doc_ids,
        "num_documents": num_documents,
        "document_length": args.document_length,
        "root_length": args.root_length,
        "branch_length": args.branch_length,
        "reusable_prefix_length": reusable_prefix_length,
        "query_length": args.query_length,
        "output_len": args.output_len,
        "kv_bytes_per_token": bytes_per_token,
        "target_working_set_gb": args.target_working_set_gb,
        "working_set_tokens": working_set_tokens,
        "working_set_gb_actual": working_set_gb,
        "working_set_to_gpu_kv_ratio": working_set_gb / gpu_cache_gb_actual if gpu_cache_gb_actual else 0,
        "single_prompt_tokens_est": single_prompt_tokens_est,
        "single_prompt_kv_gb_est": single_prompt_kv_gb_est,
        "single_prompt_to_gpu_kv_ratio": single_prompt_kv_gb_est / gpu_cache_gb_actual if gpu_cache_gb_actual else 0,
        "single_prompt_fit_count_est": int(gpu_cache_gb_actual // single_prompt_kv_gb_est) if single_prompt_kv_gb_est else 0,
        "gpu_kv_cache_gb_requested": args.gpu_kv_cache_gb,
        "gpu_kv_cache_gb_actual": gpu_cache_gb_actual,
        "num_kvcache_blocks": num_kvcache_blocks,
        "blocks_per_prompt": blocks_per_prompt,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "warmup_requests": len(warmup_prompts),
        "warmup_elapsed_sec": warmup_elapsed,
        "warmup_latency_sec": summarize_latencies(warmup_latencies),
        "query_requests": len(measured_prompts),
        "query_elapsed_sec": query_elapsed,
        "query_latency_sec": summarize_latencies(query_latencies),
        "arrival_metadata": arrival_metadata or {},
        "planned_arrival_span_sec": (arrival_metadata or {}).get("planned_arrival_span_sec", 0.0),
        "request_rate_actual": (arrival_metadata or {}).get("request_rate_actual", len(measured_prompts) / query_elapsed if query_elapsed else 0.0),
        "total_document_tokens": total_document_tokens,
        "document_recomputed_tokens_est": document_recomputed_tokens_est,
    }
    result.update(metrics)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
