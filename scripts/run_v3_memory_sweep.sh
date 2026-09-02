#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV="${VENV:-.venv-fa28}"
PYTHON="${PYTHON:-$VENV/bin/python}"
MODEL="${MODEL:-/data/datasets/models-hf/Qwen3-8B}"
GPU="${GPU:-1}"
SWEEP_STAGE="${SWEEP_STAGE:-gpu}"
EXP_DIR="${EXP_DIR:-exp/v3_${SWEEP_STAGE}_sweep_$(date +%Y%m%d_%H%M%S)}"
RUNS="${RUNS:-3}"
RUN_V2_REFERENCE="${RUN_V2_REFERENCE:-1}"
GPU_AWARE_CPU_EVICTION="${GPU_AWARE_CPU_EVICTION:-1}"

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
GPU_WATERMARKS="${GPU_WATERMARKS:-0 0.25 0.5 0.7}"
GPU_TARGET_BLOCKS="${GPU_TARGET_BLOCKS:-}"
CPU_LIMITS_GB="${CPU_LIMITS_GB:-0 16 12 8 4}"
FIXED_GPU_WATERMARK="${FIXED_GPU_WATERMARK:-}"
FIXED_GPU_TARGET_BLOCKS="${FIXED_GPU_TARGET_BLOCKS:-}"
HOT_DOCUMENTS="${HOT_DOCUMENTS:-12}"
HOT_REQUEST_RATIO="${HOT_REQUEST_RATIO:-0.8}"
HOT_REPEAT_COUNT="${HOT_REPEAT_COUNT:-20}"

case "$SWEEP_STAGE" in
  gpu)
    # Keep CPU capacity ineffective while locating the GPU writeback-window knee.
    if [[ -n "$GPU_TARGET_BLOCKS" ]]; then
      WATERMARKS=""
    else
      WATERMARKS="$GPU_WATERMARKS"
    fi
    ACTIVE_CPU_LIMITS="0"
    ;;
  cpu)
    if [[ -n "$FIXED_GPU_TARGET_BLOCKS" ]]; then
      GPU_TARGET_BLOCKS="$FIXED_GPU_TARGET_BLOCKS"
      WATERMARKS=""
    elif [[ -n "$FIXED_GPU_WATERMARK" ]]; then
      GPU_TARGET_BLOCKS=""
      WATERMARKS="$FIXED_GPU_WATERMARK"
    else
      echo "FIXED_GPU_TARGET_BLOCKS or FIXED_GPU_WATERMARK is required for SWEEP_STAGE=cpu" >&2
      exit 2
    fi
    ACTIVE_CPU_LIMITS="$CPU_LIMITS_GB"
    ;;
  *)
    echo "SWEEP_STAGE must be gpu or cpu, got: $SWEEP_STAGE" >&2
    exit 2
    ;;
esac

PROMPT_LEN="$((DOC_LEN + QUERY_LEN))"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$((PREFILL_BATCH_MULT * PROMPT_LEN))}"

mkdir -p "$EXP_DIR"

echo "stage=$SWEEP_STAGE runs=$RUNS run_v2_reference=$RUN_V2_REFERENCE gpu_aware_cpu_eviction=$GPU_AWARE_CPU_EVICTION gpu_watermarks='$WATERMARKS' gpu_target_blocks='$GPU_TARGET_BLOCKS' cpu_limits_gb='$ACTIVE_CPU_LIMITS'"
echo "hot_documents=$HOT_DOCUMENTS hot_ratio=$HOT_REQUEST_RATIO repeat_count=$HOT_REPEAT_COUNT"

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
if [[ "$GPU_AWARE_CPU_EVICTION" == "0" ]]; then
  COMMON_ARGS+=(--no-enable-gpu-aware-cpu-eviction)
fi

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

  local configs=()
  if [[ "$RUN_V2_REFERENCE" == "1" ]]; then
    configs+=("v2")
  fi
  for target_blocks in $GPU_TARGET_BLOCKS; do
    for cpu_limit in $ACTIVE_CPU_LIMITS; do
      configs+=("v3blocks:$target_blocks:$cpu_limit")
    done
  done
  for watermark in $WATERMARKS; do
    for cpu_limit in $ACTIVE_CPU_LIMITS; do
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
        local kind window_value cpu_limit
        IFS=: read -r kind window_value cpu_limit <<< "$config"
        local combo_name window_args
        if [[ "$kind" == "v3blocks" ]]; then
          combo_name="blocks_$(slug "$window_value")__cpu_$(slug "$cpu_limit")gb"
          window_args=(--lazy-writeback-target-blocks "$window_value")
        else
          combo_name="wm_$(slug "$window_value")__cpu_$(slug "$cpu_limit")gb"
          window_args=(--lazy-writeback-watermark-ratio "$window_value")
        fi
        local combo_dir="$case_dir/$combo_name"
        mkdir -p "$combo_dir"
        run_json "$combo_dir/v3_run${run_id}.json" \
          "${COMMON_ARGS[@]}" \
          --arrival-seed "$run_id" \
          --shuffle-seed "$run_id" \
          --enable-cpu-kv-offload \
          --enable-gpu-lru-retention \
          --enable-lazy-cpu-kv-writeback \
          "${window_args[@]}" \
          --cpu-prefix-cache-gb-limit "$cpu_limit" \
          "$@"
      fi
    done
  done
}

run_case hot_cold_sharing \
  --workload long_doc_qa \
  --repeat-mode hot_cold \
  --repeat-count "$HOT_REPEAT_COUNT" \
  --hot-documents "$HOT_DOCUMENTS" \
  --hot-request-ratio "$HOT_REQUEST_RATIO"

"$PYTHON" - "$EXP_DIR" > "$EXP_DIR/summary.json" <<'PY_SUMMARY'
import json
import math
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
metrics = [
    "query_elapsed_sec",
    "warmup_inflight_at_measurement_start",
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
    "recomputed_token_count",
    "document_recomputed_tokens_est",
    "cpu_prefix_cache_restored_token_count",
    "cpu_sync_swapin_request_count",
    "cpu_sync_swapin_block_count",
    "gpu_lru_hit_block_count",
    "gpu_lru_eviction_count",
    "gpu_lru_evicted_cpu_backed_block_count",
    "gpu_lru_evicted_gpu_only_block_count",
    "inactive_cpu_backed_block_count",
    "inactive_gpu_only_block_count",
    "inactive_pending_writeback_block_count",
    "inactive_safe_or_pending_block_count",
    "safe_allocatable_block_count",
    "lazy_writeback_target_block_count",
    "lazy_writeback_scheduled_block_count",
    "lazy_writeback_completed_block_count",
    "lazy_writeback_after_alloc_trigger_count",
    "lazy_writeback_after_alloc_skip_count",
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
    "gpu_prefix_cached_block_count",
    "cpu_gpu_duplicate_block_count",
    "cpu_gpu_active_duplicate_block_count",
    "cpu_gpu_protected_duplicate_block_count",
    "cpu_gpu_unprotected_duplicate_block_count",
    "cpu_cache_gpu_duplicate_ratio",
    "gpu_cache_cpu_backed_ratio",
    "cpu_gpu_duplicate_union_ratio",
    "cpu_prefix_cache_eviction_count",
    "cpu_prefix_cache_preferred_duplicate_eviction_count",
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

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
       7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
       13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
       19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
       25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def metric_stats(values):
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
                "swapin_block_ratio": mean(combo, "cpu_sync_swapin_block_count") / mean(v2, "cpu_sync_swapin_block_count") if mean(v2, "cpu_sync_swapin_block_count") else 0,
                "document_recompute_delta": mean(combo, "document_recomputed_tokens_est") - mean(v2, "document_recomputed_tokens_est"),
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
            if row.get("cpu_prefix_cache_gb_limit", 0) > 0
            and (
                row.get("cpu_prefix_pool_reserved_gb", 0)
                > row.get("cpu_prefix_cache_gb_limit", 0) + 1e-9
                or row.get("cpu_prefix_pool_on_demand_alloc_count", 0) != 0
            )
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

"$PYTHON" - "$EXP_DIR/summary.json" > "$EXP_DIR/sweep_table.tsv" <<'PY_TABLE'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
columns = (
    "config", "runs", "target_blocks", "sync_swapin_blocks", "gpu_only_evictions",
    "writeback_scheduled", "writeback_completed", "document_recompute_tokens",
    "cpu_peak_gb", "duplicate_blocks", "cpu_dup_ratio", "gpu_dup_ratio", "union_dup_ratio",
    "h2d_gib", "prefill_sec", "median_ttft_sec", "median_request_sec",
)
print("\t".join(columns))
case = summary["hot_cold_sharing"]


def mean(row, key):
    return row.get(key, {}).get("mean", 0)


rows = []
if case["v2_reference"].get("runs", 0):
    rows.append(("v2_reference", case["v2_reference"]))
rows.extend(sorted(case["v3"].items()))
for name, row in rows:
    values = (
        name,
        row.get("runs", 0),
        mean(row, "lazy_writeback_target_block_count"),
        mean(row, "cpu_sync_swapin_block_count"),
        mean(row, "gpu_lru_evicted_gpu_only_block_count"),
        mean(row, "lazy_writeback_scheduled_block_count"),
        mean(row, "lazy_writeback_completed_block_count"),
        mean(row, "document_recomputed_tokens_est"),
        mean(row, "cpu_prefix_kv_gb_peak"),
        mean(row, "cpu_gpu_duplicate_block_count"),
        mean(row, "cpu_cache_gpu_duplicate_ratio"),
        mean(row, "gpu_cache_cpu_backed_ratio"),
        mean(row, "cpu_gpu_duplicate_union_ratio"),
        mean(row, "cpu_prefix_h2d_bytes") / (1024 ** 3),
        mean(row, "prefill_step_time_sec"),
        mean(row, "ttft_latency_median"),
        mean(row, "request_latency_median"),
    )
    print("\t".join(str(value) for value in values))
PY_TABLE

echo "All V3 memory sweep results are in $EXP_DIR"
