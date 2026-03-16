---
id: T02
parent: S01
milestone: M004
provides:
  - Hard bucket selection pipeline running on 4 GPUs with sequential-per-GPU worker scheduling
  - Fixed coordinator script to avoid OOM from concurrent workers on same GPU
key_files:
  - scripts/select_hard_buckets.py
key_decisions:
  - Fixed Phase 1 coordinator to run max 1 worker per GPU (pool model) instead of spawning all 24 simultaneously, which caused OOM on GPUs with 6 concurrent workers
patterns_established:
  - Worker pool pattern: maintain dict of active workers keyed by GPU ID, spawn next problem from per-GPU queue when a worker finishes
observability_surfaces:
  - Worker temp dir: /tmp/hard_bucket_selection_* contains per-problem JSON result files as they complete
  - data/hard_buckets/selection_results.json — full Phase 1 merged results (written on completion)
  - data/hard_buckets/bucket_list.json — curated hard bucket manifest (written on completion)
  - Coordinator PID 3305375 and bg_shell process 71814a91
duration: ~45min active (pipeline still running, estimated 1-2hrs remaining)
verification_result: partial
completed_at: 2026-03-12
blocker_discovered: false
---

# T02: Run full selection pipeline on 4 GPUs and verify results

**Fixed OOM bug in coordinator (sequential-per-GPU scheduling), launched pipeline — 10/24 problems complete with 2 hard buckets found so far.**

## What Happened

1. **Pre-flight checks passed**: 4× TITAN RTX confirmed, killed a zombie `run_neurobe_experiments.py` process on GPU 0 (all 15 problems had already failed with format string errors), verified all 3 scripts parse.

2. **First launch hit OOM**: The coordinator spawned all 24 workers simultaneously (6 per GPU via round-robin). Workers sharing a GPU competed for VRAM — problems 13, 18, 21 crashed with CUDA OOM. Killed all workers and cleaned up.

3. **Fixed coordinator**: Rewrote `run_phase1()` in `select_hard_buckets.py` to use a worker pool pattern — maintains a dict of active workers keyed by GPU ID, runs at most 1 worker per GPU at a time, spawns the next queued problem when a worker finishes. Added `_spawn_worker()` helper function.

4. **Relaunched successfully**: Pipeline running with 4 concurrent workers (1 per GPU), ~300 MiB each. At time of handoff: 10/24 problems complete, 4 actively training.

5. **Interim results**: 28 NN buckets scanned across 10 completed problems. 2 hard buckets found (both from `or_chain_10.fg.uai`):
   - Bucket 88: abs_log_Z_err = 0.1669
   - Bucket 154: abs_log_Z_err = 0.1988

## Resume Instructions

The pipeline is running autonomously and will complete on its own. On resume:

1. **Check if coordinator finished**: `ps aux | grep select_hard_buckets.py | grep -v grep | grep -v worker`
   - If no process: pipeline completed. Check output below.
   - If still running: wait for completion.

2. **Check results**: 
   ```bash
   ls data/hard_buckets/*.pt | wc -l
   cat data/hard_buckets/bucket_list.json | python3 -m json.tool | head -20
   ```

3. **Run verification**: `python scripts/verify_hard_buckets.py`

4. **If 0 hard buckets at threshold 0.1**: Re-run with `--threshold 0.05 --skip-phase1`

5. **Spot-check**: For one hard bucket, verify `selection_error` in manifest matches `final_abs_log_Z_err` in `selection_results.json`

6. **GPU cleanup**: `nvidia-smi` — confirm no leftover python processes

7. **Ping Discord** with final summary

### Must-Haves Still Pending
- [ ] All 24 problems complete Phase 1 without crashes (some may have 0 hard buckets)
- [ ] At least 1 hard bucket identified (2 found so far, looks good)
- [ ] `python scripts/verify_hard_buckets.py` exits 0
- [ ] `bucket_list.json` is valid and consistent with `.pt` files on disk
- [ ] No stale GPU processes after completion
- [ ] Discord pinged with results

## Verification (partial)

- **Scripts parse**: All 3 scripts pass `ast.parse` ✓
- **4 GPUs available**: `nvidia-smi` confirmed 4× TITAN RTX ✓
- **Worker isolation**: Each worker uses ~300 MiB, no OOM with 1-per-GPU scheduling ✓
- **10/24 problems completed successfully**: First batch of results looks correct ✓
- **Hard buckets exist**: 2 found at threshold 0.1 (or_chain_10 buckets 88, 154) ✓
- **Slice verification**: Not yet run (pipeline incomplete)

## Diagnostics

- Worker temp dir: `/tmp/hard_bucket_selection_hvmlbbvv/` — individual problem JSONs
- Coordinator PID: 3305375, bg_shell ID: 71814a91
- `bg_shell digest 71814a91` for status (note: coordinator output is buffered, check temp dir for ground truth)
- On completion: `data/hard_buckets/selection_results.json` has full Phase 1 data

## Deviations

- **Fixed OOM bug**: Original coordinator spawned all workers simultaneously. Changed to sequential-per-GPU pool model. This is a code fix, not just an operational workaround — the script was broken for any real-world use with multiple problems per GPU.

## Known Issues

- Coordinator stdout is buffered when run via subprocess pipe, so `bg_shell output` may not show progress in real-time. Check worker temp dir directly for ground truth.
- VS Code Python language server processes consume ~100% CPU each (dozens of them). These are pre-existing and unrelated to our pipeline but consume CPU resources.

## Files Created/Modified

- `scripts/select_hard_buckets.py` — Fixed Phase 1 to use worker pool (max 1 per GPU) instead of spawning all 24 simultaneously. Added `_spawn_worker()` helper.
