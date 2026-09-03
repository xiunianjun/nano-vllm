#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV="${VENV:-.venv-fa28}"
PYTHON="${PYTHON:-$VENV/bin/python}"
MODEL="${MODEL:-/data/datasets/models-hf/Qwen3-8B}"
GPU="${GPU:-1}"
EXP_DIR="${EXP_DIR:-exp/prefix_cache_serving_$(date +%Y%m%d_%H%M%S)}"
RUNS="${RUNS:-3}"

DOC_LEN="${DOC_LEN:-8192}"
QUERY_LEN="${QUERY_LEN:-96}"
OUT_LEN="${OUT_LEN:-16}"
TARGET_WS_GB="${TARGET_WS_GB:-20.0}"
GPU_KV_GB="${GPU_KV_GB:-8.0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
PREFILL_BATCH_MULT="${PREFILL_BATCH_MULT:-4}"
REQUEST_RATE="${REQUEST_RATE:-2.0}"
WARMUP_MODE="${WARMUP_MODE:-stream}"
STREAM_WARMUP_RATIO="${STREAM_WARMUP_RATIO:-0.3}"
RUN_CASE0="${RUN_CASE0:-0}"
RUN_CASCADE="${RUN_CASCADE:-0}"
RUN_HOT_COLD="${RUN_HOT_COLD:-1}"
RUN_HOT_COLD_BURST="${RUN_HOT_COLD_BURST:-1}"
RUN_BRANCHING="${RUN_BRANCHING:-0}"
ARRIVAL_BURST_SIZE="${ARRIVAL_BURST_SIZE:-1}"
HOT_DOCUMENTS="${HOT_DOCUMENTS:-2}"
HOT_REQUEST_RATIO="${HOT_REQUEST_RATIO:-0.7}"
HOT_REPEAT_COUNT="${HOT_REPEAT_COUNT:-4}"
MODES="${MODES:-baseline v1 v2 v3 v4}"
LAZY_WRITEBACK_TARGET_BLOCKS="${LAZY_WRITEBACK_TARGET_BLOCKS:-40}"
# V1/V2 的 eager backing 保留完整 working set；V3/V4 使用已选出的内存边界。
EAGER_CPU_PREFIX_CACHE_GB_LIMIT="${EAGER_CPU_PREFIX_CACHE_GB_LIMIT:-0}"
LAZY_CPU_PREFIX_CACHE_GB_LIMIT="${LAZY_CPU_PREFIX_CACHE_GB_LIMIT:-15}"
ROOT_LEN="${ROOT_LEN:-$((DOC_LEN / 2))}"
BRANCH_LEN="${BRANCH_LEN:-$((DOC_LEN - ROOT_LEN))}"
PROMPT_LEN="$((DOC_LEN + QUERY_LEN))"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$((PREFILL_BATCH_MULT * PROMPT_LEN))}"
CASE0_MAX_NUM_BATCHED_TOKENS="$PROMPT_LEN"

mkdir -p "$EXP_DIR"

COMMON_BASE=(
  --model "$MODEL"
  --document-length "$DOC_LEN"
  --query-length "$QUERY_LEN"
  --output-len "$OUT_LEN"
  --target-working-set-gb "$TARGET_WS_GB"
  --gpu-kv-cache-gb "$GPU_KV_GB"
  --warmup-mode "$WARMUP_MODE"
  --stream-warmup-ratio "$STREAM_WARMUP_RATIO"
  --temperature 0
  --enforce-eager
  --no-use-tqdm
)

run_once() {
  local case_dir="$1"
  local mode="$2"
  local run_id="$3"
  local max_num_seqs="$4"
  local max_num_batched_tokens="$5"
  local arrival_mode="$6"
  local request_rate="$7"
  shift 7
  local output="$case_dir/${mode}_run${run_id}.json"
  local offload_args=(--no-enable-cpu-kv-offload --enable-gpu-lru-retention)
  if [[ "$mode" == "v1" ]]; then
    offload_args=(
      --enable-cpu-kv-offload
      --no-enable-gpu-lru-retention
      --cpu-prefix-cache-gb-limit "$EAGER_CPU_PREFIX_CACHE_GB_LIMIT"
    )
  elif [[ "$mode" == "v2" ]]; then
    offload_args=(
      --enable-cpu-kv-offload
      --enable-gpu-lru-retention
      --cpu-prefix-cache-gb-limit "$EAGER_CPU_PREFIX_CACHE_GB_LIMIT"
    )
  elif [[ "$mode" == "v3" ]]; then
    offload_args=(
      --enable-cpu-kv-offload
      --enable-gpu-lru-retention
      --enable-lazy-cpu-kv-writeback
      --lazy-writeback-target-blocks "$LAZY_WRITEBACK_TARGET_BLOCKS"
      --cpu-prefix-cache-gb-limit "$LAZY_CPU_PREFIX_CACHE_GB_LIMIT"
    )
  elif [[ "$mode" == "v4" ]]; then
    offload_args=(
      --enable-cpu-kv-offload
      --enable-gpu-lru-retention
      --enable-lazy-cpu-kv-writeback
      --enable-scheduler-aware-prefetch
      --lazy-writeback-target-blocks "$LAZY_WRITEBACK_TARGET_BLOCKS"
      --cpu-prefix-cache-gb-limit "$LAZY_CPU_PREFIX_CACHE_GB_LIMIT"
    )
  fi
  # The two seeds are paired across modes, while each independent run gets a new trace.
  local arrival_args=(--arrival-mode "$arrival_mode" --arrival-seed "$run_id" --shuffle-seed "$run_id")
  if [[ "$arrival_mode" == "poisson" ]]; then
    arrival_args+=(--request-rate "$request_rate")
  fi
  echo "[$(date '+%F %T')] running $(basename "$case_dir") $mode run=$run_id arrival=$arrival_mode max_seqs=$max_num_seqs max_batched_tokens=$max_num_batched_tokens"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" bench_long_doc_qa.py \
    "${COMMON_BASE[@]}" \
    --max-num-seqs "$max_num_seqs" \
    --max-num-batched-tokens "$max_num_batched_tokens" \
    "${arrival_args[@]}" \
    "${offload_args[@]}" "$@" | tee "$output"
}

summarize_case() {
  local case_dir="$1"
  "$PYTHON" - "$case_dir" > "$case_dir/summary.json" <<'PY_SUMMARY'
import json
import math
import re
import statistics
import sys
from pathlib import Path

case_dir = Path(sys.argv[1])
keys = [
    "query_elapsed_sec",
    "offered_rate_realized",
    "achieved_throughput",
    "planned_arrival_span_sec",
    "warmup_inflight_at_measurement_start",
    "request_latency_count",
    "request_latency_avg",
    "request_latency_median",
    "request_latency_p90",
    "request_latency_p99",
    "request_latency_min",
    "request_latency_max",
    "ttft_latency_count",
    "ttft_latency_avg",
    "ttft_latency_median",
    "ttft_latency_p90",
    "ttft_latency_p99",
    "ttft_latency_min",
    "ttft_latency_max",
    "queueing_latency_count",
    "queueing_latency_avg",
    "queueing_latency_median",
    "queueing_latency_p90",
    "queueing_latency_p99",
    "queueing_latency_min",
    "queueing_latency_max",
    "prefill_step_time_sec",
    "prefill_step_time_total_sec",
    "prefill_step_time_avg_sec",
    "prefill_timed_tok_per_sec",
    "prefill_token_count",
    "decode_step_count",
    "decode_step_time_sec",
    "decode_step_time_total_sec",
    "decode_step_time_avg_sec",
    "decode_timed_tokens",
    "prefix_cache_reused_token_count",
    "gpu_prefix_miss_request_count",
    "cpu_prefix_cache_restored_token_count",
    "cpu_sync_swapin_request_count",
    "cpu_sync_swapin_block_count",
    "cpu_sync_swapin_token_count",
    "gpu_lru_hit_block_count",
    "gpu_lru_hit_token_count",
    "gpu_lru_eviction_count",
    "gpu_lru_evicted_cpu_backed_block_count",
    "gpu_lru_evicted_gpu_only_block_count",
    "gpu_lru_cached_block_count",
    "gpu_lru_cached_block_peak",
    "inactive_cpu_backed_block_count",
    "inactive_gpu_only_block_count",
    "inactive_pending_writeback_block_count",
    "inactive_safe_or_pending_block_count",
    "safe_allocatable_block_count",
    "lazy_writeback_target_block_count",
    "lazy_writeback_scheduled_block_count",
    "lazy_writeback_completed_block_count",
    "scheduler_visible_request_count_max",
    "scheduler_lookahead_touch_block_count",
    "scheduler_prefetch_planned_request_count",
    "scheduler_prefetch_planned_block_count",
    "scheduler_prefetch_completed_block_count",
    "scheduler_prefetch_useful_block_count",
    "scheduler_prefetch_wasted_block_count",
    "scheduler_prefetch_targeted_eviction_count",
    "scheduler_prefetch_targeted_writeback_count",
    "scheduler_prefetch_wait_count",
    "scheduler_prefetch_wait_wall_sec",
    "scheduler_prefetch_capacity_rejected_request_count",
    "scheduler_prefetch_no_cpu_target_count",
    "scheduler_prefetch_h2d_bytes",
    "scheduler_prefetch_d2h_bytes",
    "scheduler_prefetch_latency_sum",
    "document_recomputed_tokens_est",
    "cpu_prefix_d2h_bytes",
    "cpu_prefix_h2d_bytes",
    "cpu_prefix_kv_bytes",
    "cpu_prefix_kv_bytes_peak",
    "cpu_prefix_kv_gb",
    "cpu_prefix_kv_gb_peak",
    "cpu_prefix_cache_block_count",
    "cpu_prefix_cache_eviction_count",
    "cpu_prefix_cache_evicted_bytes",
    "cpu_prefix_cache_evicted_metadata_count",
    "cpu_prefix_pool_gb",
    "cpu_prefix_pool_gb_requested",
    "cpu_prefix_pool_reserved_gb",
    "cpu_prefix_pool_reserved_block_count",
    "cpu_prefix_pool_free_block_count",
    "cpu_prefix_pool_used_block_count",
    "cpu_prefix_pool_reuse_count",
    "cpu_prefix_pool_on_demand_alloc_count",
    "cpu_prefix_pool_exhausted_count",
    "cpu_prefix_pool_writeback_rejected_count",
    "cpu_prefix_restore_latency_sum",
    "cpu_prefix_writeback_latency_sum",
    "schedule_time_sec",
    "model_runner_call_time_sec",
    "postprocess_time_sec",
    "prefill_prepare_wall_sec",
    "prefill_sample_prepare_wall_sec",
    "prefill_model_cuda_sec",
    "prefill_sampler_wall_sec",
    "prefill_model_runner_wall_sec",
    "prefill_model_runner_count",
    "max_num_seqs",
    "max_num_batched_tokens",
    "working_set_to_gpu_kv_ratio",
    "single_prompt_kv_gb_est",
    "single_prompt_to_gpu_kv_ratio",
    "single_prompt_fit_count_est",
]
# Two-sided 95% Student-t critical values; normal approximation beyond 30 df.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
       7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
       13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
       19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
       25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def stats(values):
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    critical = T95.get(len(values) - 1, 1.96) if len(values) > 1 else 0.0
    half_width = critical * stdev / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "stdev": stdev,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "min": min(values),
        "max": max(values),
    }


mode_rows = {}
for mode in ("baseline", "v1", "v2", "v3", "v4"):
    rows = {}
    for path in sorted(case_dir.glob(f"{mode}_run*.json")):
        match = re.search(r"_run(\d+)\.json$", path.name)
        if match:
            rows[int(match.group(1))] = json.loads(path.read_text())
    if rows:
        mode_rows[mode] = rows

summary = {"case": case_dir.name, "modes": {}}
for mode, rows_by_run in mode_rows.items():
    rows = list(rows_by_run.values())
    mode_summary = {"runs": len(rows), "run_ids": sorted(rows_by_run)}
    for key in keys:
        values = [row.get(key, 0) for row in rows]
        mode_summary[key] = stats(values)
    for key in (
        "arrival_mode", "request_rate_target", "num_documents", "reusable_prefix_length",
        "working_set_gb_actual", "gpu_kv_cache_gb_actual", "query_requests",
        "measured_workload_profile", "temperature",
    ):
        mode_summary[key] = rows[0].get(key)
    summary["modes"][mode] = mode_summary


def mean(mode, key):
    return summary["modes"][mode][key]["mean"]


def add_speedups(dst, lhs, rhs, label):
    if lhs not in mode_rows or rhs not in mode_rows:
        return
    common_runs = sorted(set(mode_rows[lhs]) & set(mode_rows[rhs]))
    for out_key, metric in (
        ("request_latency_median_speedup", "request_latency_median"),
        ("request_latency_p99_speedup", "request_latency_p99"),
        ("ttft_median_speedup", "ttft_latency_median"),
        ("ttft_p99_speedup", "ttft_latency_p99"),
        ("queueing_avg_speedup", "queueing_latency_avg"),
        ("queueing_p99_speedup", "queueing_latency_p99"),
        ("prefill_time_speedup", "prefill_step_time_sec"),
        ("decode_time_speedup", "decode_step_time_sec"),
    ):
        denominator = mean(lhs, metric)
        dst[f"{out_key}_{label}"] = mean(rhs, metric) / denominator if denominator else 0.0
        paired = []
        for run_id in common_runs:
            lhs_value = mode_rows[lhs][run_id].get(metric, 0)
            rhs_value = mode_rows[rhs][run_id].get(metric, 0)
            if lhs_value:
                paired.append(rhs_value / lhs_value)
        dst[f"{out_key}_{label}_paired"] = stats(paired) if paired else None


summary["comparison"] = {}
add_speedups(summary["comparison"], "v1", "baseline", "v1_over_baseline")
add_speedups(summary["comparison"], "v2", "baseline", "v2_over_baseline")
add_speedups(summary["comparison"], "v2", "v1", "v2_over_v1")
add_speedups(summary["comparison"], "v3", "v2", "v3_over_v2")
add_speedups(summary["comparison"], "v3", "v1", "v3_over_v1")
add_speedups(summary["comparison"], "v4", "v3", "v4_over_v3")

trace_mismatches = []
output_mismatches = []
all_runs = sorted(set().union(*(set(rows) for rows in mode_rows.values()))) if mode_rows else []
for run_id in all_runs:
    per_run = {mode: rows[run_id] for mode, rows in mode_rows.items() if run_id in rows}
    trace_hashes = {mode: row.get("trace_sha256") for mode, row in per_run.items()}
    output_hashes = {mode: row.get("output_sha256") for mode, row in per_run.items()}
    if len(set(trace_hashes.values())) > 1:
        trace_mismatches.append({"run_id": run_id, "hashes": trace_hashes})
    if len(set(output_hashes.values())) > 1:
        output_mismatches.append({"run_id": run_id, "hashes": output_hashes})

budget_violations = []
for mode, rows in mode_rows.items():
    for run_id, row in rows.items():
        limit = row.get("cpu_prefix_cache_gb_limit", 0) or 0
        if not row.get("enable_cpu_kv_offload") or limit <= 0:
            continue
        reserved = row.get("cpu_prefix_pool_reserved_gb", 0)
        on_demand = row.get("cpu_prefix_pool_on_demand_alloc_count", 0)
        if reserved > limit + 1e-9 or on_demand != 0:
            budget_violations.append({
                "mode": mode,
                "run_id": run_id,
                "limit_gb": limit,
                "reserved_gb": reserved,
                "on_demand_alloc_count": on_demand,
            })

summary["validation"] = {
    "paired_trace_ok": not trace_mismatches,
    "trace_mismatches": trace_mismatches,
    "greedy_output_match_ok": not output_mismatches,
    "output_mismatches": output_mismatches,
    "cpu_physical_budget_ok": not budget_violations,
    "cpu_physical_budget_violations": budget_violations,
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY_SUMMARY
  echo "[$(date '+%F %T')] wrote $case_dir/summary.json"
}

run_case() {
  local name="$1"
  local max_num_seqs="$2"
  local max_num_batched_tokens="$3"
  local arrival_mode="$4"
  local request_rate="$5"
  shift 5
  local case_dir="$EXP_DIR/$name"
  mkdir -p "$case_dir"
  read -r -a mode_list <<< "$MODES"
  local mode_count="${#mode_list[@]}"
  for run_id in $(seq 1 "$RUNS"); do
    # Rotate the first mode each run to counterbalance thermal/order effects.
    local start_index=$(( (run_id - 1) % mode_count ))
    for offset in $(seq 0 $((mode_count - 1))); do
      local mode="${mode_list[$(( (start_index + offset) % mode_count ))]}"
      run_once "$case_dir" "$mode" "$run_id" "$max_num_seqs" "$max_num_batched_tokens" "$arrival_mode" "$request_rate" "$@"
    done
  done
  summarize_case "$case_dir"
}

if [[ "$RUN_CASE0" == "1" ]]; then
  run_case case0_functional 1 "$CASE0_MAX_NUM_BATCHED_TOKENS" batch "$REQUEST_RATE" \
    --warmup-mode all_docs \
    --workload long_doc_qa \
    --num-documents 1 \
    --repeat-mode tile \
    --repeat-count 1
fi

if [[ "$RUN_CASCADE" == "1" ]]; then
  run_case cascade_tile "$MAX_NUM_SEQS" "$MAX_NUM_BATCHED_TOKENS" poisson "$REQUEST_RATE" \
    --warmup-mode all_docs \
    --workload long_doc_qa \
    --repeat-mode tile \
    --repeat-count 1
fi

if [[ "$RUN_HOT_COLD" == "1" ]]; then
  run_case hot_cold_sharing "$MAX_NUM_SEQS" "$MAX_NUM_BATCHED_TOKENS" poisson "$REQUEST_RATE" \
    --workload long_doc_qa \
    --repeat-mode hot_cold \
    --repeat-count "$HOT_REPEAT_COUNT" \
    --hot-documents "$HOT_DOCUMENTS" \
    --hot-request-ratio "$HOT_REQUEST_RATIO"
fi

if [[ "$RUN_HOT_COLD_BURST" == "1" ]]; then
  run_case hot_cold_burst "$MAX_NUM_SEQS" "$MAX_NUM_BATCHED_TOKENS" poisson "$REQUEST_RATE" \
    --arrival-burst-size "$ARRIVAL_BURST_SIZE" \
    --workload long_doc_qa \
    --repeat-mode hot_cold \
    --repeat-count "$HOT_REPEAT_COUNT" \
    --hot-documents "$HOT_DOCUMENTS" \
    --hot-request-ratio "$HOT_REQUEST_RATIO"
fi

if [[ "$RUN_BRANCHING" == "1" ]]; then
  run_case branching_prefix_sharing "$MAX_NUM_SEQS" "$MAX_NUM_BATCHED_TOKENS" poisson "$REQUEST_RATE" \
    --workload branching_prefix \
    --root-length "$ROOT_LEN" \
    --branch-length "$BRANCH_LEN" \
    --repeat-mode hot_cold \
    --repeat-count 4 \
    --hot-documents 2 \
    --hot-request-ratio 0.7
fi

"$PYTHON" - "$EXP_DIR" > "$EXP_DIR/summary.json" <<'PY_ROOT_SUMMARY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
out = {}
for path in sorted(root.glob("*/summary.json")):
    out[path.parent.name] = json.loads(path.read_text())
print(json.dumps(out, indent=2, sort_keys=True))
PY_ROOT_SUMMARY

echo "All results are in $EXP_DIR"
