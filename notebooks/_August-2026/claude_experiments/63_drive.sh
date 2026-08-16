#!/bin/bash
# Doc 63 driver. gpu1 ONLY (gpu0/gpu3 carry a live 12-day paper rerun whose
# timings are a published result; gpu2 is retired). Strictly sequential.
# Usage: ./63_drive.sh <logfile> <spec> [<spec> ...]
#   spec = problem|arm|seed|num_epochs|tag|outdir
set -u
export CUDA_VISIBLE_DEVICES=1
export NCE_MODEL_CACHE=/home/cohenn1/NCE/.model_cache
export NCE_THREADS=6
PY=/home/cohenn1/NCE/venv/bin/python
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="$1"; shift

for spec in "$@"; do
  IFS='|' read -r prob arm seed epochs tag odir <<< "$spec"
  echo "=== $(date -Is) $prob $arm s$seed e$epochs ===" >> "$LOG"
  t0=$(date +%s)
  "$PY" "$HERE/63_run.py" --problem "$prob" --arm "$arm" --seed "$seed" \
      --num-epochs "$epochs" --merge sub --D 10 --ib 10 --ecl 1025 \
      --local-error 1 --tag "$tag" --outdir "$odir" >> "$LOG" 2>&1
  rc=$?
  echo "=== done rc=$rc in $(( $(date +%s) - t0 ))s ===" >> "$LOG"
done
echo "ALL DONE $(date -Is)" >> "$LOG"
