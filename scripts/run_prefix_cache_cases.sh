#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV="${VENV:-.venv-fa28}"
PYTHON="${PYTHON:-$VENV/bin/python}"
MODEL="${MODEL:-/data/datasets/models-hf/Qwen3-8B}"
GPU="${GPU:-1}"
OUT_DIR="${OUT_DIR:-results/prefix_cache_cases/$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUT_DIR"

COMMON=(
  --model "$MODEL"
  --document-length 1024
  --query-length 64
  --output-len 8
  --target-working-set-gb 1.0
  --gpu-kv-cache-gb 1.1
  --max-num-seqs 1
  --enforce-eager
  --no-use-tqdm
)

run_case() {
  local name="$1"
  shift
  local output="$OUT_DIR/${name}.json"
  echo "[$(date '+%F %T')] running $name"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" bench_long_doc_qa.py "${COMMON[@]}" "$@" | tee "$output"
  echo "[$(date '+%F %T')] wrote $output"
}

run_pair() {
  local name="$1"
  shift
  run_case "${name}_gpu_only" --no-enable-cpu-kv-offload "$@"
  run_case "${name}_cpu_v1" --enable-cpu-kv-offload "$@"
}

# Case 1: sequential scan over a working set that is slightly larger than GPU KV.
# This intentionally exposes cascading cache pollution / thrashing.
run_pair cascade_tile \
  --workload long_doc_qa \
  --repeat-mode tile \
  --repeat-count 1

# Case 2: normal document-level prefix sharing with hot and cold documents.
# Hot documents are revisited often; cold documents create pressure and misses.
run_pair hot_cold_sharing \
  --workload long_doc_qa \
  --repeat-mode hot_cold \
  --repeat-count 4 \
  --hot-documents 2 \
  --hot-request-ratio 0.7 \
  --shuffle-seed 0

# Case 3: partial prefix sharing across branched requests.
# Different branches share a common root prefix; requests on the same branch share root+branch.
run_pair branching_prefix_sharing \
  --workload branching_prefix \
  --root-length 512 \
  --branch-length 512 \
  --repeat-mode hot_cold \
  --repeat-count 4 \
  --hot-documents 2 \
  --hot-request-ratio 0.7 \
  --shuffle-seed 1

echo "All results are in $OUT_DIR"
