#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV="${VENV:-.venv-fa28}"
PYTHON="${PYTHON:-$VENV/bin/python}"
MODEL="${MODEL:-/data/datasets/models-hf/Qwen3-8B}"
GPU="${GPU:-1}"
EXP_DIR="${EXP_DIR:-exp/v3_memory_sweep_$(date +%Y%m%d_%H%M%S)}"
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
WATERMARKS="${WATERMARKS:-0.5}"
CPU_LIMITS_GB="${CPU_LIMITS_GB:-20 12 11 10.5 10 5 3 2 1}"
RUN_BRANCHING="${RUN_BRANCHING:-0}"
ROOT_LEN="${ROOT_LEN:-$((DOC_LEN / 2))}"
BRANCH_LEN="${BRANCH_LEN:-$((DOC_LEN - ROOT_LEN))}"

PROMPT_LEN="$((DOC_LEN + QUERY_LEN))"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$((PREFILL_BATCH_MULT * PROMPT_LEN))}"

mkdir -p "$EXP_DIR"

COMMON_ARGS=(
  --model "$MODEL"
  --document-length "$DOC_LEN"
  --query-length "$QUERY_LEN"
  --output-len "$OUT_LEN"
  --target-working-set-gb "$TARGET_WS_GB"
  --gpu-kv-cache-gb "$GPU_KV_GB"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --arrival-mode poisson
  --request-rate "$REQUEST_RATE"
  --warmup-mode "$WARMUP_MODE"
  --stream-warmup-ratio "$STREAM_WARMUP_RATIO"
  --temperature 0
  --enforce-eager
  --no-use-tqdm
)

slug() {
  local value="$1"
  echo "${value//./p}"
}

run_json() {
  local output="$1"
  shift
  echo "[$(date '+%F %T')] running $output"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" bench_long_doc_qa.py "$@" | tee "$output"
}

run_case() {
  local case_name="$1"
  shift
  local case_dir="$EXP_DIR/$case_name"
  mkdir -p "$case_dir"

  local configs=("v2")
  for watermark in $WATERMARKS; do
    for cpu_limit in $CPU_LIMITS_GB; do
      configs+=("v3:$watermark:$cpu_limit")
    done
  done
  local config_count="${#configs[@]}"

  for run_id in $(seq 1 "$RUNS"); do
    # Pair trace seeds across configurations and rotate execution order by run.
    local start_index=$(( (run_id - 1) % config_count ))
    for offset in $(seq 0 $((config_count - 1))); do
      local config="${configs[$(( (start_index + offset) % config_count ))]}"
      if [[ "$config" == "v2" ]]; then
        run_json "$case_dir/v2_reference_run${run_id}.json" \
          "${COMMON_ARGS[@]}" \
          --arrival-seed "$run_id" \
          --shuffle-seed "$run_id" \
          --enable-cpu-kv-offload \
          --enable-gpu-lru-retention \
          "$@"
      else
        local kind watermark cpu_limit
        IFS=: read -r kind watermark cpu_limit <<< "$config"
        local combo_dir="$case_dir/wm_$(slug "$watermark")__cpu_$(slug "$cpu_limit")gb"
        mkdir -p "$combo_dir"
        run_json "$combo_dir/v3_run${run_id}.json" \
          "${COMMON_ARGS[@]}" \
          --arrival-seed "$run_id" \
          --shuffle-seed "$run_id" \
          --enable-cpu-kv-offload \
          --enable-gpu-lru-retention \
          --enable-lazy-cpu-kv-writeback \
          --lazy-writeback-watermark-ratio "$watermark" \
          --cpu-prefix-cache-gb-limit "$cpu_limit" \
          "$@"
      fi
    done
  done
}

run_case cascade_tile \
  --workload long_doc_qa \
  --repeat-mode tile \
  --repeat-count 1

run_case hot_cold_sharing \
  --workload long_doc_qa \
  --repeat-mode hot_cold \
  --repeat-count 4 \
  --hot-documents 2 \
  --hot-request-ratio 0.7

if [[ "$RUN_BRANCHING" == "1" ]]; then
  run_case branching_prefix_sharing \
    --workload branching_prefix \
    --root-length "$ROOT_LEN" \
    --branch-length "$BRANCH_LEN" \
    --repeat-mode hot_cold \
    --repeat-count 4 \
    --hot-documents 2 \
    --hot-request-ratio 0.7
fi

"$PYTHON" - "$EXP_DIR" > "$EXP_DIR/summary.json" <<'PY_SUMMARY'
import json
import math
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
metrics = [
    "query_elapsed_sec",
    "offered_rate_realized",
    "achieved_throughput",
    "request_latency_median",
    "request_latency_p90",
    "request_latency_p99",
    "ttft_latency_median",
    "ttft_latency_p90",
    "ttft_latency_p99",
    "ttft_latency_min",
    "queueing_latency_avg",
    "queueing_latency_p90",
    "queueing_latency_p99",
    "queueing_latency_max",
    "prefill_step_time_sec",
    "prefill_step_time_avg_sec",
    "decode_step_time_sec",
    "prefix_cache_reused_token_count",
    "cpu_prefix_cache_restored_token_count",
    "cpu_sync_swapin_request_count",
    "cpu_sync_swapin_block_count",
    "gpu_lru_hit_block_count",
    "gpu_lru_eviction_count",
    "inactive_cpu_backed_block_count",
    "inactive_gpu_only_block_count",
    "inactive_safe_or_pending_block_count",
    "lazy_writeback_target_block_count",
    "lazy_writeback_scheduled_block_count",
    "lazy_writeback_completed_block_count",
    "cpu_prefix_kv_gb",
    "cpu_prefix_kv_gb_peak",
    "cpu_prefix_writeback_submit_wall_sec",
    "cpu_prefix_writeback_cpu_alloc_wall_sec",
    "cpu_prefix_pool_exhausted_count",
    "cpu_prefix_pool_writeback_rejected_count",
    "cpu_prefix_pool_on_demand_alloc_count",
    "cpu_prefix_pool_reuse_count",
    "cpu_prefix_pool_used_block_count",
    "cpu_prefix_pool_free_block_count",
    "cpu_prefix_pool_reserved_gb",
    "cpu_prefix_cache_live_gb_peak",
    "cpu_prefix_cache_live_gb",
    "cpu_prefix_cache_block_count",
    "cpu_prefix_cache_eviction_count",
    "cpu_prefix_cache_evicted_bytes",
    "cpu_prefix_d2h_bytes",
    "cpu_prefix_h2d_bytes",
    "cpu_prefix_restore_latency_sum",
    "cpu_prefix_writeback_latency_sum",
    "working_set_to_gpu_kv_ratio",
    "single_prompt_to_gpu_kv_ratio",
]

def load_rows(paths):
    return [json.loads(path.read_text()) for path in sorted(paths)]

def metric_stats(values):
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    # Normal CI is explicitly approximate; raw values and stdev are retained.
    half_width = 1.96 * stdev / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "stdev": stdev,
        "ci95_low_approx": mean - half_width,
        "ci95_high_approx": mean + half_width,
        "min": min(values),
        "max": max(values),
    }


def aggregate(rows):
    out = {"runs": len(rows)}
    if not rows:
        return out
    for key in metrics:
        values = [row.get(key, 0) for row in rows]
        out[key] = metric_stats(values)
    for key in (
        "num_documents",
        "query_requests",
        "document_length",
        "target_working_set_gb",
        "gpu_kv_cache_gb_actual",
        "max_num_seqs",
        "max_num_batched_tokens",
        "request_rate_target",
    ):
        out[key] = rows[0].get(key)
    return out

def mean(row, key):
    return row[key]["mean"]

summary = {}
for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
    v2_rows = load_rows(case_dir.glob("v2_reference_run*.json"))
    case = {"v2_reference": aggregate(v2_rows), "v3": {}}
    v2 = case["v2_reference"]
    for combo_dir in sorted(path for path in case_dir.iterdir() if path.is_dir()):
        rows = load_rows(combo_dir.glob("v3_run*.json"))
        combo = aggregate(rows)
        if rows and v2["runs"]:
            combo["vs_v2"] = {
                "cpu_peak_ratio": mean(combo, "cpu_prefix_kv_gb_peak") / mean(v2, "cpu_prefix_kv_gb_peak") if mean(v2, "cpu_prefix_kv_gb_peak") else 0,
                "ttft_median_ratio": mean(combo, "ttft_latency_median") / mean(v2, "ttft_latency_median") if mean(v2, "ttft_latency_median") else 0,
                "request_latency_median_ratio": mean(combo, "request_latency_median") / mean(v2, "request_latency_median") if mean(v2, "request_latency_median") else 0,
                "prefill_time_ratio": mean(combo, "prefill_step_time_sec") / mean(v2, "prefill_step_time_sec") if mean(v2, "prefill_step_time_sec") else 0,
                "swapin_block_delta": mean(combo, "cpu_sync_swapin_block_count") - mean(v2, "cpu_sync_swapin_block_count"),
            }
        trace_mismatches = []
        output_mismatches = []
        for run_index, (v2_row, v3_row) in enumerate(zip(v2_rows, rows), start=1):
            if v2_row.get("trace_sha256") != v3_row.get("trace_sha256"):
                trace_mismatches.append(run_index)
            if v2_row.get("output_sha256") != v3_row.get("output_sha256"):
                output_mismatches.append(run_index)
        budget_violations = [
            run_index
            for run_index, row in enumerate(rows, start=1)
            if row.get("cpu_prefix_pool_reserved_gb", 0) > row.get("cpu_prefix_cache_gb_limit", 0) + 1e-9
            or row.get("cpu_prefix_pool_on_demand_alloc_count", 0) != 0
        ]
        combo["validation"] = {
            "paired_trace_ok": not trace_mismatches and len(v2_rows) == len(rows),
            "trace_mismatch_runs": trace_mismatches,
            "greedy_output_match_ok": not output_mismatches and len(v2_rows) == len(rows),
            "output_mismatch_runs": output_mismatches,
            "cpu_physical_budget_ok": not budget_violations,
            "cpu_physical_budget_violation_runs": budget_violations,
        }
        case["v3"][combo_dir.name] = combo
    summary[case_dir.name] = case

print(json.dumps(summary, indent=2, sort_keys=True))
PY_SUMMARY

echo "All V3 memory sweep results are in $EXP_DIR"
