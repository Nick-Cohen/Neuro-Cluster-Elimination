---
id: S01
parent: M004
milestone: M004
provides:
  - Hard bucket selection pipeline (worker + coordinator + verification scripts)
  - Multi-GPU worker pool with CUDA_VISIBLE_DEVICES isolation (1 worker per GPU)
  - Precomputed .pt files with factor tensors, exact forward/backward messages, and metadata
  - bucket_list.json manifest and selection_results.json full Phase 1 data
requires:
  - slice: none
    provides: first slice — no upstream dependencies
affects:
  - S02
key_files:
  - scripts/select_hard_buckets_worker.py
  - scripts/select_hard_buckets.py
  - scripts/verify_hard_buckets.py
  - data/hard_buckets/*.pt
  - data/hard_buckets/bucket_list.json
  - data/hard_buckets/selection_results.json
key_decisions:
  - D038: Bucket reconstruction via eliminate_variables(up_to=...), not FastGM stub
  - D041: Precomputed bucket data as torch .pt files with JSON manifest
  - D042: Per-worker temp JSON + coordinator merge, not fcntl locking
  - D043: Configurable hardness threshold with 0.1 default
  - D044: Script-based verification, not pytest
  - D045: Two-task split — scripts first, execution second
  - D046: Worker pool pattern (max 1 per GPU) to prevent OOM from concurrent workers
patterns_established:
  - Subprocess worker pattern with CUDA_VISIBLE_DEVICES isolation and JSON result files
  - Worker pool model — dict of active workers keyed by GPU ID, round-robin queue per GPU
  - .pt file schema with factors/exact_fw/exact_bw dicts containing tensor+labels
  - bucket_list.json manifest with sanitized bucket IDs (problem_key__bucket_label)
observability_surfaces:
  - Worker prints structured progress per problem with NN bucket count and hard count
  - Coordinator prints per-GPU worker spawning, completion progress, and final summary
  - Worker temp dir contains per-problem JSON results for live monitoring
  - verify_hard_buckets.py provides per-file PASS/FAIL diagnostics
  - selection_results.json persists full Phase 1 data for post-hoc analysis
drill_down_paths:
  - .gsd/milestones/M004/slices/S01/tasks/T01-SUMMARY.md
  - .gsd/milestones/M004/slices/S01/tasks/T02-SUMMARY.md
duration: ~3h active work + ~2.5h pipeline runtime (still completing final 5/24 problems)
verification_result: partial — 19/24 problems complete, 4 hard buckets found, pipeline completing autonomously
completed_at: 2026-03-12
---

# S01: Hard Bucket Selection & Precomputation

**Built and ran the hard bucket selection pipeline across 4 GPUs — 4 hard buckets identified from 19/24 completed problems, pipeline finishing autonomously for remaining 5.**

## What Happened

**T01 (scripts):** Wrote three scripts forming the selection pipeline:

1. `select_hard_buckets_worker.py` — subprocess entry point that runs one problem through FastGM with error_tracking=True, UKL loss, bw_ecl=ecl, 10000 epochs. Writes per-bucket error data to JSON.

2. `select_hard_buckets.py` — coordinator that spawns workers across GPUs, merges results, identifies hard buckets above threshold, runs Phase 2 precomputation (exact upstream elimination + exact_fw + exact_bw + torch.save), writes manifest and results files.

3. `verify_hard_buckets.py` — validates cached .pt files against expected schema (required keys, tensor shapes, finiteness, manifest consistency).

All three validated with mock .pt smoke tests (positive and negative cases).

**T02 (execution):** First launch hit OOM — the coordinator spawned all 24 workers simultaneously (6 per GPU via round-robin). Fixed to worker pool pattern: max 1 worker per GPU, spawns next queued problem when a worker finishes. Relaunched successfully.

At time of slice completion: 19/24 problems complete, 4 hard buckets found at threshold 0.1:
- `or_chain_10.fg.uai` → Bucket 88 (abs_log_Z_err = 0.1669)
- `or_chain_10.fg.uai` → Bucket 154 (abs_log_Z_err = 0.1988)
- `grid10x10.f5.wrap.uai` → Bucket 10 (abs_log_Z_err = 0.8326)
- `BN_2.uai` → Bucket 9 (abs_log_Z_err = 0.1402)

Pipeline PID 3305375 is completing the remaining 5 problems autonomously. Phase 2 precomputation and .pt file writing happens after all 24 workers finish.

## Verification

- All 3 scripts pass `ast.parse` syntax check ✓
- Mock .pt smoke test: verify_hard_buckets.py exits 0 on valid mock, exits 1 on bad mock ✓
- 4 GPUs confirmed via nvidia-smi ✓
- Worker isolation: ~300 MiB per worker, no OOM with 1-per-GPU scheduling ✓
- 19/24 problems completed successfully with consistent JSON output ✓
- 4 hard buckets found (≥3 threshold for useful benchmark set) ✓
- **Pending:** Full pipeline completion (5 remaining problems), Phase 2 .pt file generation, verify_hard_buckets.py on real data

## Requirements Advanced

- R039 — Hard bucket selection pipeline built and running; 4 hard buckets identified across 19/24 problems; awaiting pipeline completion for final .pt files
- R040 — Precomputed message caching implemented in Phase 2 precomputation code; awaiting execution after Phase 1 completes

## Requirements Validated

- None fully validated yet — pipeline completion and verification script pass on real data needed

## New Requirements Surfaced

- None

## Requirements Invalidated or Re-scoped

- None

## Deviations

- **OOM fix (T02):** Original coordinator spawned all workers simultaneously. Changed to sequential-per-GPU pool model (max 1 worker per GPU). This was a bug fix in the coordinator script, not a plan deviation.

## Known Limitations

- Pipeline still running at slice completion (19/24 problems done, 5 remaining). Phase 2 precomputation (.pt file generation) hasn't executed yet — it runs after all 24 workers finish.
- Verification on real .pt files not yet run (requires Phase 2 completion).
- Wall-clock time for full 24-problem pipeline: ~2.5+ hours on 4× TITAN RTX. Some problems with many NN buckets (e.g., grid10x10) take 30+ minutes each.

## Follow-ups

- **Resume action required:** Check if coordinator PID 3305375 has finished. If so, run `python scripts/verify_hard_buckets.py` and ping Discord with final results. If < 3 hard buckets at final count, re-run with `--threshold 0.05`.
- **GPU cleanup:** After pipeline completes, verify no stale python processes via `nvidia-smi`.

## Files Created/Modified

- `scripts/select_hard_buckets_worker.py` — Phase 1 subprocess entry point (new)
- `scripts/select_hard_buckets.py` — Coordinator + Phase 2 precomputation (new, then fixed OOM bug)
- `scripts/verify_hard_buckets.py` — Validation/verification script (new)
- `data/hard_buckets/*.pt` — Cached bucket data (pending Phase 2 completion)
- `data/hard_buckets/bucket_list.json` — Manifest (pending Phase 2 completion)
- `data/hard_buckets/selection_results.json` — Full Phase 1 results (pending all workers)

## Forward Intelligence

### What the next slice should know
- The .pt file schema is: `{'factors': list of dicts, 'exact_fw': {'tensor': Tensor, 'labels': list}, 'exact_bw': {'tensor': Tensor, 'labels': list}, 'bucket_label': str, 'scope': list, 'domain_sizes': list, 'elim_vars': list, 'problem_key': str, 'auto_ecl': int}`. All tensors stored on CPU via `detach().cpu()`.
- Bucket IDs in manifest are sanitized: `{problem_key}__{bucket_label}` (double underscore separator).
- Factor tensors in .pt files are raw — to reconstruct FastFactor objects, use `FastFactor(tensor, labels)`.

### What's fragile
- The coordinator's Phase 2 precomputation re-runs `FastGM.eliminate_variables(up_to=bucket_var)` to reconstruct bucket state. This means importing torch and running real inference — if a problem's exact upstream elimination fails, that bucket won't be cached. Monitor Phase 2 stderr.

### Authoritative diagnostics
- `data/hard_buckets/selection_results.json` — full per-problem, per-bucket error data from Phase 1. This is the ground truth for all bucket hardness decisions.
- Worker temp dir `/tmp/hard_bucket_selection_*/` — individual problem JSONs for debugging worker failures.

### What assumptions changed
- **Hard bucket availability (was: risk):** 4 hard buckets found across 19/24 problems at threshold 0.1. Risk retired — sufficient hard buckets exist.
- **Selection cost (was: risk):** ~2.5+ hours on 4 GPUs. Acceptable one-time cost. Worker pool pattern (1 per GPU) is essential — simultaneous spawning causes OOM.
