#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV="${VENV:-.venv-fa28}"
PYTHON="${PYTHON:-$VENV/bin/python}"
EXP_ROOT="${EXP_ROOT:-exp/doclen_sweep_serving_poisson_$(date +%Y%m%d_%H%M%S)}"
DOC_LENS="${DOC_LENS:-4096 6144 7680}"

mkdir -p "$EXP_ROOT"

for doc_len in $DOC_LENS; do
  doc_dir="$EXP_ROOT/doc_${doc_len}"
  echo "[$(date '+%F %T')] starting doc_len=$doc_len -> $doc_dir"
  EXP_DIR="$doc_dir" DOC_LEN="$doc_len" VENV="$VENV" PYTHON="$PYTHON" \
    scripts/run_prefix_cache_cases.sh
done

"$PYTHON" - "$EXP_ROOT" > "$EXP_ROOT/summary.json" <<'PY_ROOT_SUMMARY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = {}
for doc_dir in sorted(root.glob("doc_*"), key=lambda p: int(p.name.split("_", 1)[1])):
    summary_path = doc_dir / "summary.json"
    if summary_path.exists():
        out[doc_dir.name] = json.loads(summary_path.read_text())
print(json.dumps(out, indent=2, sort_keys=True))
PY_ROOT_SUMMARY

"$PYTHON" scripts/plot_doclen_sweep.py "$EXP_ROOT"

echo "All doc length sweep results are in $EXP_ROOT"
