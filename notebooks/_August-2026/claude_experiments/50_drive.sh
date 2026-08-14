#!/bin/bash
# Doc 50 sequential driver. gpu3 ONLY. One process at a time.
# usage: 44_drive.sh <logfile> <spec> [<spec> ...]
#   spec = problem|merge|D|seed|arm|samplecap|memcap|samplefrac|memfrac|tag|outdir
set -u
export CUDA_VISIBLE_DEVICES=3
export NCE_THREADS=14
PY=/home/cohenn1/NCE/venv/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
LOG="$1"; shift
echo "=== driver start $(date -Is) ===" >> "$LOG"
for spec in "$@"; do
  IFS='|' read -r prob merge D seed arm scap mcap sfrac mfrac tag odir <<< "$spec"
  t0=$(date +%s)
  echo "--- $(date -Is) START $spec" >> "$LOG"
  $PY "$HERE/50_run.py" --problem "$prob" --merge "$merge" --D "$D" \
      --seed "$seed" --arm "$arm" --sample-cap "$scap" --mem-cap "$mcap" \
      --sample-frac "$sfrac" --mem-frac "$mfrac" \
      --tag "$tag" --outdir "$odir" >> "$LOG" 2>&1
  rc=$?
  t1=$(date +%s)
  echo "--- $(date -Is) END $spec rc=$rc seconds=$((t1-t0))" >> "$LOG"
done
echo "=== driver done $(date -Is) ===" >> "$LOG"
