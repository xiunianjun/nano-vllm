import argparse
import hashlib
import json
import math
import random
import statistics
import time

import torch
from transformers import AutoConfig, AutoTokenizer

from nanovllm import LLM, SamplingParams
from nanovllm.config import KVCachePolicy


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
    parser.add_argument("--warmup-mode", choices=("all_docs", "stream", "none"), default="stream")
    parser.add_argument("--stream-warmup-ratio", type=float, default=0.3)
    parser.add_argument("--request-rate", type=float, default=None, help="Poisson arrival rate in requests/sec.")
    parser.add_argument("--arrival-seed", type=int, default=0)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--hot-documents", type=int, default=2)
    parser.add_argument("--hot-request-ratio", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 selects deterministic greedy decoding.")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--kvcache-block-size", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enable-cpu-kv-offload", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-gpu-lru-retention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-lazy-cpu-kv-writeback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--enable-gpu-aware-cpu-eviction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer evicting redundant GPU-resident CPU entries; disable for the naive V3 ablation.",
    )
    parser.add_argument("--lazy-writeback-watermark-ratio", type=float, default=0.5)
    parser.add_argument(
        "--lazy-writeback-target-blocks",
        type=int,
        default=0,
        help="Absolute lazy-writeback victim window. 0 derives it from the batched-token limit.",
    )
    parser.add_argument("--cpu-prefix-cache-gb-limit", type=float, default=0.0)
    parser.add_argument("--cpu-prefix-pool-gb", type=float, default=0.0, help="Pinned CPU KV pool size. 0 means auto for CPU offload.")
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
    outputs = llm.generate(prompts, [sampling_params] * len(prompts), use_tqdm=use_tqdm)
    elapsed = time.perf_counter() - round_start
    return elapsed, llm.request_latencies[before:], outputs


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
    request_indices = {}
    outputs_by_index = {}

    while finished < len(prompts):
        now = time.perf_counter()
        while next_request < len(prompts) and round_start + arrival_offsets[next_request] <= now:
            seq = llm.add_request(prompts[next_request], sampling_params)
            # Charge queueing from the planned client arrival, including arrivals
            # that occur while the engine is inside a long CUDA step.
            llm.request_start_times[seq.seq_id] = round_start + arrival_offsets[next_request]
            request_indices[seq.seq_id] = next_request
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
        completed_at = time.perf_counter()
        for seq_id, token_ids in output:
            finished += 1
            request_index = request_indices.pop(seq_id)
            outputs_by_index[request_index] = token_ids
            request_start = llm.request_start_times.pop(seq_id, step_start)
            llm.request_latencies.append(completed_at - request_start)

    elapsed = time.perf_counter() - round_start
    arrival_span = arrival_offsets[-1] - arrival_offsets[0] if len(arrival_offsets) > 1 else 0.0
    metadata = {
        "arrival_offsets_sec": arrival_offsets,
        "planned_arrival_span_sec": arrival_span,
        "request_rate_target": request_rate,
        "offered_rate_realized": (len(prompts) - 1) / arrival_span if arrival_span else 0.0,
        "achieved_throughput": len(prompts) / elapsed if elapsed else 0.0,
        "continuous_warmup": False,
    }
    outputs = [
        {"token_ids": outputs_by_index[i], "text": llm.tokenizer.decode(outputs_by_index[i])}
        for i in range(len(prompts))
    ]
    return elapsed, llm.request_latencies[before:], outputs, metadata


def run_continuous_poisson_round(llm, warmup_prompts, measured_prompts, sampling_params, request_rate, seed):
    """Run warmup and measured requests on one Poisson timeline."""
    prompts = warmup_prompts + measured_prompts
    warmup_count = len(warmup_prompts)
    offsets = poisson_arrival_offsets(len(prompts), request_rate, seed)
    round_start = time.perf_counter()
    measurement_start = None
    warmup_inflight_at_boundary = 0
    next_request = finished = warmup_finished = 0
    local_start_times = {}
    request_indices = {}
    measured_seq_ids = set()
    warmup_latencies = []
    measured_outputs = {}

    if warmup_count == 0:
        llm.reset_metrics()
        measurement_start = round_start

    while finished < len(prompts):
        now = time.perf_counter()
        while next_request < len(prompts) and round_start + offsets[next_request] <= now:
            if next_request == warmup_count and measurement_start is None:
                warmup_inflight_at_boundary = warmup_count - warmup_finished
                llm.reset_metrics()
                measurement_start = time.perf_counter()
            seq = llm.add_request(prompts[next_request], sampling_params)
            planned_start = round_start + offsets[next_request]
            local_start_times[seq.seq_id] = planned_start
            request_indices[seq.seq_id] = next_request
            llm.request_start_times[seq.seq_id] = planned_start
            if next_request >= warmup_count:
                measured_seq_ids.add(seq.seq_id)
            next_request += 1

        if llm.is_finished():
            if next_request < len(prompts):
                sleep_until = round_start + offsets[next_request]
                time.sleep(max(0.0, sleep_until - time.perf_counter()))
                continue
            break

        step_start = time.perf_counter()
        output, num_tokens = llm.step()
        record_step_metrics(llm, num_tokens, time.perf_counter() - step_start)
        completed_at = time.perf_counter()
        for seq_id, token_ids in output:
            finished += 1
            request_index = request_indices.pop(seq_id)
            latency = completed_at - local_start_times.pop(seq_id)
            if seq_id in measured_seq_ids:
                measured_outputs[request_index - warmup_count] = token_ids
                llm.request_latencies.append(latency)
            else:
                warmup_finished += 1
                warmup_latencies.append(latency)
            llm.request_start_times.pop(seq_id, None)

    end = time.perf_counter()
    measurement_start = measurement_start or end
    measured_offsets = offsets[warmup_count:]
    arrival_span = measured_offsets[-1] - measured_offsets[0] if len(measured_offsets) > 1 else 0.0
    measurement_elapsed = end - measurement_start
    metadata = {
        "arrival_offsets_sec": offsets,
        "measured_arrival_offsets_sec": measured_offsets,
        "planned_arrival_span_sec": arrival_span,
        "request_rate_target": request_rate,
        "offered_rate_realized": (len(measured_offsets) - 1) / arrival_span if arrival_span else 0.0,
        "achieved_throughput": len(measured_prompts) / measurement_elapsed if measurement_elapsed else 0.0,
        "warmup_inflight_at_measurement_start": warmup_inflight_at_boundary,
        "continuous_warmup": True,
    }
    outputs = [
        {"token_ids": measured_outputs[i], "text": llm.tokenizer.decode(measured_outputs[i])}
        for i in range(len(measured_prompts))
    ]
    return (
        measurement_elapsed,
        list(llm.request_latencies),
        outputs,
        measurement_start - round_start,
        warmup_latencies,
        metadata,
    )


def percentile(values, quantile):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_latencies(latencies):
    if not latencies:
        return {"count": 0, "avg": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(latencies),
        "avg": sum(latencies) / len(latencies),
        "median": statistics.median(latencies),
        "p90": percentile(latencies, 0.90),
        "p99": percentile(latencies, 0.99),
        "min": min(latencies),
        "max": max(latencies),
    }


def workload_profile(doc_ids, args, bytes_per_token):
    counts = {str(doc_id): doc_ids.count(doc_id) for doc_id in sorted(set(doc_ids))}
    last_positions = {}
    reuse_distances = []
    for position, doc_id in enumerate(doc_ids):
        if doc_id in last_positions:
            reuse_distances.append(len(set(doc_ids[last_positions[doc_id] + 1:position])))
        last_positions[doc_id] = position
    unique_count = len(counts)
    if args.workload == "branching_prefix":
        realized_tokens = (args.root_length if unique_count else 0) + unique_count * args.branch_length
    else:
        realized_tokens = unique_count * args.document_length
    return {
        "request_count": len(doc_ids),
        "unique_document_count": unique_count,
        "access_counts": counts,
        "reuse_distance": summarize_latencies(reuse_distances),
        "realized_working_set_tokens": realized_tokens,
        "realized_working_set_gb": realized_tokens * bytes_per_token / GB,
    }


def main():
    args = parse_args()
    kv_cache_policy = KVCachePolicy.from_flags(
        args.enable_cpu_kv_offload,
        args.enable_gpu_lru_retention,
        args.enable_lazy_cpu_kv_writeback,
    )
    if args.warmup_mode == "stream" and not 0 <= args.stream_warmup_ratio < 1:
        raise ValueError("--stream-warmup-ratio must be in [0, 1)")
    if args.arrival_mode == "poisson" and (args.request_rate is None or args.request_rate <= 0):
        raise ValueError("--request-rate must be positive when --arrival-mode=poisson")
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

    bytes_per_token = kv_bytes_per_token(hf_config)
    if args.workload == "branching_prefix":
        working_set_tokens = args.root_length + num_documents * args.branch_length
    else:
        working_set_tokens = num_documents * args.document_length
    working_set_gb = working_set_tokens * bytes_per_token / GB

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

    cpu_prefix_pool_gb = args.cpu_prefix_pool_gb
    auto_cpu_prefix_pool = args.enable_cpu_kv_offload and cpu_prefix_pool_gb <= 0
    if auto_cpu_prefix_pool:
        # Offload 的公平默认值：所有 V1/V2/V3 都使用 pinned pool，避免把
        # torch.empty(..., pin_memory=True) 的临时分配开销混进策略对比。
        # CPU cap experiments count the pinned pool itself against the budget.
        # Allocation rounds down to whole blocks, so this cannot reserve over cap.
        cpu_prefix_pool_gb = args.cpu_prefix_cache_gb_limit or working_set_gb
    if args.cpu_prefix_cache_gb_limit > 0 and cpu_prefix_pool_gb > args.cpu_prefix_cache_gb_limit:
        raise ValueError(
            "--cpu-prefix-pool-gb cannot exceed --cpu-prefix-cache-gb-limit; "
            "the latter is a physical host-memory budget"
        )

    doc_ids = list(range(num_documents))
    request_doc_ids = repeat_doc_ids(
        doc_ids,
        args.repeat_count,
        args.repeat_mode,
        args.shuffle_seed,
        args.hot_documents,
        args.hot_request_ratio,
    )
    if args.warmup_mode == "stream":
        # Serving-style steady state: 前一段请求只负责形成 cache 状态，后一段才计入指标。
        # 这样不会强制每个 document 都被 warmup 一次，CPU cap 淘汰冷 prefix 才是合理策略。
        warmup_count = int(len(request_doc_ids) * args.stream_warmup_ratio)
        warmup_doc_ids = request_doc_ids[:warmup_count]
        measured_doc_ids = request_doc_ids[warmup_count:]
    elif args.warmup_mode == "none":
        warmup_doc_ids = []
        measured_doc_ids = request_doc_ids
    else:
        warmup_doc_ids = doc_ids
        measured_doc_ids = request_doc_ids

    warmup_prompts = [
        make_prompt(doc_id, args, ids, warmup=True, query_variant=i)
        for i, doc_id in enumerate(warmup_doc_ids)
    ]
    measured_prompts = [
        make_prompt(doc_id, args, ids, warmup=False, query_variant=i)
        for i, doc_id in enumerate(measured_doc_ids)
    ]
    if not measured_prompts:
        raise ValueError("measurement set is empty; use --stream-warmup-ratio below 1")
    sampling_params = SamplingParams(temperature=args.temperature, ignore_eos=True, max_tokens=args.output_len)

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
        enable_gpu_aware_cpu_eviction=args.enable_gpu_aware_cpu_eviction,
        lazy_writeback_watermark_ratio=args.lazy_writeback_watermark_ratio,
        lazy_writeback_target_blocks=args.lazy_writeback_target_blocks,
        cpu_prefix_cache_gb_limit=args.cpu_prefix_cache_gb_limit,
        cpu_prefix_pool_gb=cpu_prefix_pool_gb,
    )

    llm.generate([[ids[1]]], SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=1), use_tqdm=False)
    if args.arrival_mode == "poisson" and args.warmup_mode == "stream":
        (
            query_elapsed,
            query_latencies,
            query_outputs,
            warmup_elapsed,
            warmup_latencies,
            arrival_metadata,
        ) = run_continuous_poisson_round(
            llm,
            warmup_prompts,
            measured_prompts,
            sampling_params,
            args.request_rate,
            args.arrival_seed,
        )
    else:
        if warmup_prompts:
            warmup_elapsed, warmup_latencies, _ = run_round(
                llm, warmup_prompts, sampling_params, args.use_tqdm
            )
        else:
            warmup_elapsed, warmup_latencies = 0.0, []
        llm.reset_metrics()
        if args.arrival_mode == "poisson":
            query_elapsed, query_latencies, query_outputs, arrival_metadata = run_poisson_round(
                llm, measured_prompts, sampling_params, args.request_rate, args.arrival_seed
            )
        else:
            query_elapsed, query_latencies, query_outputs = run_round(
                llm, measured_prompts, sampling_params, args.use_tqdm
            )
            arrival_metadata = {
                "arrival_offsets_sec": [],
                "planned_arrival_span_sec": 0.0,
                "request_rate_target": None,
                "offered_rate_realized": 0.0,
                "achieved_throughput": len(measured_prompts) / query_elapsed if query_elapsed else 0.0,
                "continuous_warmup": False,
            }
    metrics = llm.get_metrics()
    llm.exit()

    gpu_cache_gb_actual = num_kvcache_blocks * args.kvcache_block_size * bytes_per_token / GB
    single_prompt_tokens_est = reusable_prefix_length + args.query_length + args.output_len
    single_prompt_kv_gb_est = single_prompt_tokens_est * bytes_per_token / GB
    total_document_tokens = len(measured_prompts) * reusable_prefix_length
    restored_tokens = metrics.get("cpu_prefix_cache_restored_token_count", 0)
    reused_or_restored = metrics["prefix_cache_reused_token_count"] + restored_tokens
    cached_doc_tokens_upper_bound = min(reused_or_restored, total_document_tokens)
    document_recomputed_tokens_est = total_document_tokens - cached_doc_tokens_upper_bound
    warmup_profile = workload_profile(warmup_doc_ids, args, bytes_per_token)
    measured_profile = workload_profile(measured_doc_ids, args, bytes_per_token)
    measured_profile["working_set_to_gpu_kv_ratio"] = (
        measured_profile["realized_working_set_gb"] / gpu_cache_gb_actual if gpu_cache_gb_actual else 0.0
    )
    output_token_ids = [output["token_ids"] for output in query_outputs]
    output_sha256 = hashlib.sha256(
        json.dumps(output_token_ids, separators=(",", ":")).encode()
    ).hexdigest()
    trace_descriptor = {
        "workload": args.workload,
        "warmup_doc_ids": warmup_doc_ids,
        "measured_doc_ids": measured_doc_ids,
        "arrival_offsets_sec": arrival_metadata.get("arrival_offsets_sec", []),
        "temperature": args.temperature,
        "output_len": args.output_len,
    }
    trace_sha256 = hashlib.sha256(
        json.dumps(trace_descriptor, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    result = {
        "model": args.model,
        "mode": kv_cache_policy.benchmark_mode,
        "enable_cpu_kv_offload": args.enable_cpu_kv_offload,
        "enable_gpu_lru_retention": args.enable_gpu_lru_retention,
        "enable_lazy_cpu_kv_writeback": args.enable_lazy_cpu_kv_writeback,
        "enable_gpu_aware_cpu_eviction": args.enable_gpu_aware_cpu_eviction,
        "lazy_writeback_watermark_ratio": args.lazy_writeback_watermark_ratio,
        "lazy_writeback_target_blocks_requested": args.lazy_writeback_target_blocks,
        "cpu_prefix_cache_gb_limit": args.cpu_prefix_cache_gb_limit,
        "cpu_prefix_pool_gb": cpu_prefix_pool_gb,
        "cpu_prefix_pool_gb_requested": args.cpu_prefix_pool_gb,
        "auto_cpu_prefix_pool": auto_cpu_prefix_pool,
        "workload": args.workload,
        "repeat_mode": args.repeat_mode,
        "repeat_count": args.repeat_count,
        "warmup_mode": args.warmup_mode,
        "stream_warmup_ratio": args.stream_warmup_ratio,
        "arrival_mode": args.arrival_mode,
        "request_rate_target": args.request_rate,
        "arrival_seed": args.arrival_seed,
        "hot_documents": args.hot_documents,
        "hot_request_ratio": args.hot_request_ratio,
        "warmup_doc_ids": warmup_doc_ids,
        "measured_doc_ids": measured_doc_ids,
        "warmup_workload_profile": warmup_profile,
        "measured_workload_profile": measured_profile,
        "num_documents": num_documents,
        "document_length": args.document_length,
        "root_length": args.root_length,
        "branch_length": args.branch_length,
        "reusable_prefix_length": reusable_prefix_length,
        "query_length": args.query_length,
        "output_len": args.output_len,
        "temperature": args.temperature,
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
        "warmup_inflight_at_measurement_start": (arrival_metadata or {}).get(
            "warmup_inflight_at_measurement_start", 0
        ),
        "offered_rate_realized": arrival_metadata.get("offered_rate_realized", 0.0),
        "achieved_throughput": arrival_metadata.get(
            "achieved_throughput", len(measured_prompts) / query_elapsed if query_elapsed else 0.0
        ),
        # Backward-compatible alias; prefer achieved_throughput in new analyses.
        "request_rate_actual": arrival_metadata.get(
            "achieved_throughput", len(measured_prompts) / query_elapsed if query_elapsed else 0.0
        ),
        "trace_sha256": trace_sha256,
        "output_sha256": output_sha256,
        "output_token_ids": output_token_ids,
        "total_document_tokens": total_document_tokens,
        "document_recomputed_tokens_est": document_recomputed_tokens_est,
    }
    result.update(metrics)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
