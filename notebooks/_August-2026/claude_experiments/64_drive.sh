#!/bin/bash
# Doc 64 driver. gpu1 ONLY, strictly sequential.
# Usage: ./64_drive.sh <logfile> <spec> [<spec> ...]
#   spec = problem|arm|seed|tag|outdir
set -u
export CUDA_VISIBLE_DEVICES=1
export NCE_MODEL_CACHE=/home/cohenn1/NCE/.model_cache
export NCE_THREADS=6
PY=/home/cohenn1/NCE/venv/bin/python
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="$1"; shift

for spec in "$@"; do
  IFS='|' read -r prob arm seed tag odir <<< "$spec"
  echo "=== $(date -Is) $prob $arm s$seed ===" >> "$LOG"
  t0=$(date +%s)
  "$PY" "$HERE/64_run.py" --problem "$prob" --arm "$arm" --seed "$seed" \
      --merge sub --D 10 --ib 10 --ecl 1025 --num-epochs 500 \
      --mem-frac 0.1 --sample-frac 0.2 --bw-ecl 1024 \
      --tag "$tag" --outdir "$odir" >> "$LOG" 2>&1
  rc=$?
  echo "=== done rc=$rc in $(( $(date +%s) - t0 ))s ===" >> "$LOG"
done
echo "ALL DONE $(date -Is)" >> "$LOG"
