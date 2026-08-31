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
  local offload_flag="--no-enable-cpu-kv-offload"
  if [[ "$mode" == "v1" ]]; then
    offload_flag="--enable-cpu-kv-offload"
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
    "$offload_flag" "$@" | tee "$output"
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
    "cpu_prefix_cache_restored_token_count",
    "document_recomputed_tokens_est",
    "cpu_prefix_d2h_bytes",
    "cpu_prefix_h2d_bytes",
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
]
summary = {"case": case_dir.name, "modes": {}}
for mode in ("baseline", "v1"):
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

base = summary["modes"].get("baseline")
v1 = summary["modes"].get("v1")
if base and v1:
    def mean(mode, key):
        return summary["modes"][mode][key]["mean"]
    summary["comparison"] = {
        "query_elapsed_speedup_v1_over_baseline": mean("baseline", "query_elapsed_sec") / mean("v1", "query_elapsed_sec") if mean("v1", "query_elapsed_sec") else 0,
        "request_latency_median_speedup_v1_over_baseline": mean("baseline", "request_latency_median") / mean("v1", "request_latency_median") if mean("v1", "request_latency_median") else 0,
        "ttft_median_speedup_v1_over_baseline": mean("baseline", "ttft_latency_median") / mean("v1", "ttft_latency_median") if mean("v1", "ttft_latency_median") else 0,
        "queueing_avg_speedup_v1_over_baseline": mean("baseline", "queueing_latency_avg") / mean("v1", "queueing_latency_avg") if mean("v1", "queueing_latency_avg") else 0,
        "queueing_max_speedup_v1_over_baseline": mean("baseline", "queueing_latency_max") / mean("v1", "queueing_latency_max") if mean("v1", "queueing_latency_max") else 0,
        "prefill_time_speedup_v1_over_baseline": mean("baseline", "prefill_step_time_sec") / mean("v1", "prefill_step_time_sec") if mean("v1", "prefill_step_time_sec") else 0,
        "decode_time_speedup_v1_over_baseline": mean("baseline", "decode_step_time_sec") / mean("v1", "decode_step_time_sec") if mean("v1", "decode_step_time_sec") else 0,
        "baseline_recomputed_tokens_mean": mean("baseline", "document_recomputed_tokens_est"),
        "v1_restored_tokens_mean": mean("v1", "cpu_prefix_cache_restored_token_count"),
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
  for run_id in $(seq 1 "$RUNS"); do
    run_once "$case_dir" baseline "$run_id" "$max_num_seqs" "$max_num_batched_tokens" "$arrival_mode" "$request_rate" "$@"
    run_once "$case_dir" v1 "$run_id" "$max_num_seqs" "$max_num_batched_tokens" "$arrival_mode" "$request_rate" "$@"
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
