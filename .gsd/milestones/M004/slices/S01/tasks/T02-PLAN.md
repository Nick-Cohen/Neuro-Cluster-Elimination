---
estimated_steps: 5
estimated_files: 4
---

# T02: Run full selection pipeline on 4 GPUs and verify results

**Slice:** S01 — Hard Bucket Selection & Precomputation
**Milestone:** M004

## Description

Execute the selection pipeline on the 4× TITAN RTX machine. This is the operational proof that retires the three key risks: (1) Trainer↔FastGM coupling works with error_tracking, (2) multi-GPU subprocess spawning works, (3) enough hard buckets exist at threshold 0.1. Phase 1 runs 24 problems across 4 GPUs (~2–6 hours). Phase 2 precomputes exact messages for hard buckets (~minutes). Verification confirms data integrity.

## Steps

1. Pre-flight checks:
   - `nvidia-smi` — confirm 4 GPUs available and memory is free
   - `ps aux | grep python | grep -v grep` — kill stale workers from prior runs
   - Verify scripts are executable: `python -c "import ast; ast.parse(open('scripts/select_hard_buckets.py').read())"`
   - Ensure `data/hard_buckets/` directory will be created by the script (or create it)

2. Launch Phase 1 + Phase 2 in background:
   - `python scripts/select_hard_buckets.py --threshold 0.1 --gpus 0,1,2,3`
   - Use `bg_shell` start with no timeout (estimated 2–6 hours)
   - Monitor periodically via `bg_shell` digest/highlights for progress and errors

3. On completion — verify data:
   - Run `python scripts/verify_hard_buckets.py`
   - Check `data/hard_buckets/bucket_list.json` — how many hard buckets found?
   - If < 3 hard buckets at threshold 0.1: re-run with `--threshold 0.05 --skip-phase1` (reuses existing Phase 1 results, just changes the filtering threshold and re-runs Phase 2)
   - Inspect a sample `.pt` file: load it, print keys, tensor shapes, verify exact_fw and exact_bw are finite

4. Validate operational correctness:
   - Spot-check: for one hard bucket, verify that `selection_error` in manifest matches the last entry's `abs_log_Z_err` in `selection_results.json`
   - Verify no stale GPU processes remain after completion: `nvidia-smi`

5. Report results and ping Discord:
   - Summarize: total NN buckets found, hard buckets found, threshold used, wall-clock time
   - List the hard buckets with their problem keys, bucket labels, and selection errors
   - Ping Discord with the summary

## Must-Haves

- [ ] All 24 problems complete Phase 1 without crashes (some may have 0 hard buckets — that's fine)
- [ ] At least 1 hard bucket identified (if 0 at threshold 0.1, retry with lower threshold)
- [ ] `python scripts/verify_hard_buckets.py` exits 0
- [ ] `bucket_list.json` is valid and consistent with `.pt` files on disk
- [ ] No stale GPU processes after completion
- [ ] Discord pinged with results

## Verification

- `python scripts/verify_hard_buckets.py` exits 0
- `ls data/hard_buckets/*.pt | wc -l` ≥ 1
- `python -c "import json; d=json.load(open('data/hard_buckets/bucket_list.json')); print(f'{len(d[\"buckets\"])} hard buckets'); assert len(d['buckets']) >= 1"`
- `nvidia-smi` shows no leftover python processes

## Observability Impact

- Signals added/changed: None — this task runs existing scripts, no new code
- How a future agent inspects this: `data/hard_buckets/selection_results.json` for full Phase 1 data; `bucket_list.json` for curated list; `verify_hard_buckets.py` for integrity check
- Failure state exposed: Worker stderr output in coordinator's console; `selection_results.json` records per-problem errors for debugging

## Inputs

- `scripts/select_hard_buckets_worker.py` — from T01
- `scripts/select_hard_buckets.py` — from T01
- `scripts/verify_hard_buckets.py` — from T01
- 4× NVIDIA TITAN RTX GPUs available

## Expected Output

- `data/hard_buckets/*.pt` — per-bucket cached data files (count depends on hard bucket availability)
- `data/hard_buckets/bucket_list.json` — manifest of hard buckets
- `data/hard_buckets/selection_results.json` — full Phase 1 results for all 24 problems
- Discord ping with results summary
