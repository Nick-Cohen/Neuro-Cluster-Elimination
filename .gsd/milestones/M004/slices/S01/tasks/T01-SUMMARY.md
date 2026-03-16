---
id: T01
parent: S01
milestone: M004
provides:
  - Phase 1 worker script (subprocess entry point for single-problem error-tracked training)
  - Phase 1 coordinator script (multi-GPU spawning, result merging, Phase 2 precomputation)
  - Verification script (validates cached .pt files and manifest consistency)
key_files:
  - scripts/select_hard_buckets_worker.py
  - scripts/select_hard_buckets.py
  - scripts/verify_hard_buckets.py
key_decisions:
  - Worker uses sys.executable for subprocess python to ensure venv consistency
  - Phase 2 imports torch/nce only after Phase 1 workers complete (coordinator stays torch-free during fork)
  - .pt files store tensors on CPU (detach().cpu()) so they load on any device via map_location
patterns_established:
  - Subprocess worker pattern with CUDA_VISIBLE_DEVICES isolation and JSON result files
  - .pt file schema with factors/exact_fw/exact_bw dicts containing tensor+labels
  - bucket_list.json manifest with sanitized bucket IDs (problem_key__bucket_label)
observability_surfaces:
  - Worker prints structured progress line per problem with NN bucket count and hard count
  - Coordinator prints per-GPU worker spawning, completion progress, and final summary
  - Phase 2 prints per-bucket precomputation progress with tensor shapes
  - Worker writes error JSON with traceback on failure; coordinator prints stderr from failed workers
  - verify_hard_buckets.py provides per-file PASS/FAIL diagnostics
duration: 30m
verification_result: passed
completed_at: 2026-03-12
blocker_discovered: false
---

# T01: Write Phase 1 worker and Phase 2 precomputation scripts

**Created three scripts forming the hard bucket selection pipeline: worker, coordinator+precomputation, and verification.**

## What Happened

Wrote three scripts that implement the two-phase hard bucket selection pipeline:

1. `select_hard_buckets_worker.py` (~100 lines): Subprocess entry point that trains one problem from small_problems with `error_tracking=True`, UKL loss, `bw_ecl=ecl`, 10000 epochs. Writes per-bucket error data (including abs_log_Z_err at each checkpoint) to a JSON output file. Handles errors with traceback capture.

2. `select_hard_buckets.py` (~260 lines): Coordinator that spawns workers across GPUs with round-robin assignment, waits for completion with progress reporting, merges results into `selection_results.json`, identifies hard buckets above threshold, runs Phase 2 precomputation (exact upstream elimination + exact_fw + exact_bw), and writes `bucket_list.json` manifest. Phase 2 imports torch only after all workers complete.

3. `verify_hard_buckets.py` (~150 lines): Validates cached .pt files against the expected schema — checks required keys, tensor/labels sub-keys on exact_fw/exact_bw, factor list structure, tensor finiteness (no NaN/Inf), shape consistency (domain_sizes product matches fw tensor numel), and manifest-to-disk cross-check.

## Verification

- All three scripts pass syntax check via `python -c "import ast; ast.parse(...)"`
- Mock .pt file smoke test: created a valid mock with matching manifest, `verify_hard_buckets.py --dir /tmp/mock_hard_buckets` exits 0 with "2 checks passed, 0 failed"
- Negative test: created a bad mock with missing keys and NaN tensors, verify correctly exits 1 and reports specific issues (missing keys, NaN detection, missing manifest)
- Slice-level checks 2 and 3 (actual .pt files and manifest existence in data/hard_buckets/) correctly show "not yet" — these require T02 to run the pipeline

## Diagnostics

- Run `python scripts/verify_hard_buckets.py --dir <path>` to validate any set of cached bucket data
- Read `selection_results.json` for full Phase 1 training data (all buckets, all problems)
- Read `bucket_list.json` for the curated hard bucket list
- Worker error JSONs in temp dir contain full tracebacks on failure
- Coordinator prints stderr from failed workers (last 10 lines)

## Deviations

None.

## Known Issues

None.

## Files Created/Modified

- `scripts/select_hard_buckets_worker.py` — Phase 1 subprocess entry point, trains one problem with error tracking
- `scripts/select_hard_buckets.py` — Phase 1 coordinator + Phase 2 precomputation + manifest generation
- `scripts/verify_hard_buckets.py` — Validates .pt file schema, tensor shapes, finiteness, and manifest consistency
