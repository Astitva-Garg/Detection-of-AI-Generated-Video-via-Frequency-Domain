#!/usr/bin/env bash
# Run every post-generation stage in order (each stage is incremental / resumable).
# Usage: bash run_pipeline.sh            (uses the venv python at $PY, default C:/ie643_env)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-/c/ie643_env/Scripts/python.exe}"
export PYTHONUNBUFFERED=1

step() { echo; echo "=== $(date '+%H:%M:%S') $*"; }
step "content filter";          "$PY" -m src.data.content_filter | tail -3
step "normalise + manifest";    "$PY" -m src.data.build_dataset --workers 3 | grep -A10 "split  " || true
step "frequency features";      "$PY" -m src.features.extract --workers 3 | grep -v "/" || true
step "pixel baseline features"; "$PY" -m src.features.pixel_baseline --workers 3
step "train";                   "$PY" -m src.models.train 2>&1 | grep "^{" || true
step "evaluate";                "$PY" -m src.eval.evaluate 2>&1 | grep -v "Warning\|eval_set\|warn" | tail -45
step "grouped CV";              "$PY" -m src.eval.cv 2>&1 | grep -v "Warning\|eval_set\|warn" | tail -15
step "figures";                 "$PY" -m src.viz.figures 2>&1 | grep -c saved
step "app reference + examples"; rm -f examples/*.mp4; "$PY" -m src.viz.reference
step "done"
