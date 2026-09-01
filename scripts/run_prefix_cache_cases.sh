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

DOC_LEN="${DOC_LEN:-4096}"
QUERY_LEN="${QUERY_LEN:-96}"
OUT_LEN="${OUT_LEN:-16}"
TARGET_WS_GB="${TARGET_WS_GB:-1.0}"
GPU_KV_GB="${GPU_KV_GB:-1.1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
PREFILL_BATCH_MULT="${PREFILL_BATCH_MULT:-4}"
REQUEST_RATE="${REQUEST_RATE:-2.0}"
RUN_BRANCHING="${RUN_BRANCHING:-0}"
MODES="${MODES:-baseline v1 v2}"
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
    offload_args=(--enable-cpu-kv-offload --no-enable-gpu-lru-retention)
  elif [[ "$mode" == "v2" ]]; then
    offload_args=(--enable-cpu-kv-offload --enable-gpu-lru-retention)
  fi
  local arrival_args=(--arrival-mode "$arrival_mode" --arrival-seed "$run_id")
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
import statistics
import sys
from pathlib import Path

case_dir = Path(sys.argv[1])
keys = [
    "query_elapsed_sec",
    "request_rate_actual",
    "planned_arrival_span_sec",
    "request_latency_count",
    "request_latency_avg",
    "request_latency_median",
    "request_latency_min",
    "request_latency_max",
    "ttft_latency_count",
    "ttft_latency_avg",
    "ttft_latency_median",
    "ttft_latency_min",
    "ttft_latency_max",
    "queueing_latency_count",
    "queueing_latency_avg",
    "queueing_latency_median",
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
    "gpu_lru_cached_block_count",
    "gpu_lru_cached_block_peak",
    "document_recomputed_tokens_est",
    "cpu_prefix_d2h_bytes",
    "cpu_prefix_h2d_bytes",
    "cpu_prefix_kv_bytes",
    "cpu_prefix_kv_bytes_peak",
    "cpu_prefix_kv_gb",
    "cpu_prefix_kv_gb_peak",
    "cpu_prefix_cache_block_count",
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
summary = {"case": case_dir.name, "modes": {}}
for mode in ("baseline", "v1", "v2"):
    rows = []
    for path in sorted(case_dir.glob(f"{mode}_run*.json")):
        rows.append(json.loads(path.read_text()))
    if not rows:
        continue
    mode_summary = {"runs": len(rows)}
    for key in keys:
        values = [row.get(key, 0) for row in rows]
        mode_summary[key] = {
            "mean": statistics.fmean(values),
            "min": min(values),
            "max": max(values),
        }
    for key in (
        "arrival_mode", "request_rate_target", "num_documents", "reusable_prefix_length",
        "working_set_gb_actual", "gpu_kv_cache_gb_actual", "query_requests"
    ):
        mode_summary[key] = rows[0].get(key)
    summary["modes"][mode] = mode_summary

def mean(mode, key):
    return summary["modes"][mode][key]["mean"]

def add_speedups(dst, lhs, rhs, label):
    if lhs not in summary["modes"] or rhs not in summary["modes"]:
        return
    for out_key, metric in (
        ("query_elapsed_speedup", "query_elapsed_sec"),
        ("request_latency_median_speedup", "request_latency_median"),
        ("ttft_median_speedup", "ttft_latency_median"),
        ("queueing_avg_speedup", "queueing_latency_avg"),
        ("queueing_max_speedup", "queueing_latency_max"),
        ("prefill_time_speedup", "prefill_step_time_sec"),
        ("decode_time_speedup", "decode_step_time_sec"),
    ):
        denom = mean(lhs, metric)
        dst[f"{out_key}_{label}"] = mean(rhs, metric) / denom if denom else 0

summary["comparison"] = {}
add_speedups(summary["comparison"], "v1", "baseline", "v1_over_baseline")
add_speedups(summary["comparison"], "v2", "baseline", "v2_over_baseline")
add_speedups(summary["comparison"], "v2", "v1", "v2_over_v1")
if "baseline" in summary["modes"]:
    summary["comparison"]["baseline_recomputed_tokens_mean"] = mean("baseline", "document_recomputed_tokens_est")
if "v1" in summary["modes"]:
    summary["comparison"]["v1_restored_tokens_mean"] = mean("v1", "cpu_prefix_cache_restored_token_count")
if "v2" in summary["modes"]:
    summary["comparison"]["v2_restored_tokens_mean"] = mean("v2", "cpu_prefix_cache_restored_token_count")
    summary["comparison"]["v2_gpu_lru_hit_tokens_mean"] = mean("v2", "gpu_lru_hit_token_count")
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
  for run_id in $(seq 1 "$RUNS"); do
    for mode in $MODES; do
      run_once "$case_dir" "$mode" "$run_id" "$max_num_seqs" "$max_num_batched_tokens" "$arrival_mode" "$request_rate" "$@"
    done
  done
  summarize_case "$case_dir"
}

run_case case0_functional 1 "$CASE0_MAX_NUM_BATCHED_TOKENS" batch "$REQUEST_RATE" \
  --workload long_doc_qa \
  --num-documents 1 \
  --repeat-mode tile \
  --repeat-count 1

run_case cascade_tile "$MAX_NUM_SEQS" "$MAX_NUM_BATCHED_TOKENS" poisson "$REQUEST_RATE" \
  --workload long_doc_qa \
  --repeat-mode tile \
  --repeat-count 1

run_case hot_cold_sharing "$MAX_NUM_SEQS" "$MAX_NUM_BATCHED_TOKENS" poisson "$REQUEST_RATE" \
  --workload long_doc_qa \
  --repeat-mode hot_cold \
  --repeat-count 4 \
  --hot-documents 2 \
  --hot-request-ratio 0.7 \
  --shuffle-seed 0

if [[ "$RUN_BRANCHING" == "1" ]]; then
  run_case branching_prefix_sharing "$MAX_NUM_SEQS" "$MAX_NUM_BATCHED_TOKENS" poisson "$REQUEST_RATE" \
    --workload branching_prefix \
    --root-length "$ROOT_LEN" \
    --branch-length "$BRANCH_LEN" \
    --repeat-mode hot_cold \
    --repeat-count 4 \
    --hot-documents 2 \
    --hot-request-ratio 0.7 \
    --shuffle-seed 1
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
