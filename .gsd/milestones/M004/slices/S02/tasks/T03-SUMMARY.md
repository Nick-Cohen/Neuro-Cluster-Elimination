---
id: T03
parent: S02
milestone: M004
provides:
  - scripts/verify_benchmark_training.py — end-to-end verification script for the single-bucket training harness
key_files:
  - scripts/verify_benchmark_training.py
key_decisions:
  - Config derived from small_problems defaults — minimal configs missing fields Trainer.__init__ requires (lower_dim, debug, traced_losses, etc.); starting from the full default config and overriding verification-specific fields is more robust than building from scratch
patterns_established:
  - Synthetic .pt fallback — script self-contains by generating a .pt from smokers_20 when S01 data isn't available, matching the exact Phase 2 schema (factors, exact_fw, exact_bw, metadata)
  - Structured PASS/FAIL checks — each validation is a named check with diagnostic on failure, enabling programmatic and human inspection
observability_surfaces:
  - "[Verify]" prefixed stdout messages for lifecycle tracking
  - Exit code 0/1 as machine-readable pass/fail signal
  - Per-check PASS/FAIL lines with diagnostic context on failure
  - Summary block with bucket_id, epochs, final_loss, final_error, wall_time
duration: 30m
verification_result: passed
completed_at: 2026-03-12
blocker_discovered: false
---

# T03: End-to-end verification script

**Created `scripts/verify_benchmark_training.py` — self-contained end-to-end verification of the full train_single_bucket() pipeline with synthetic .pt fallback, structured validation checks, and clear PASS/FAIL output.**

## What Happened

Wrote the verification script with four components:

1. **`_find_pt_file()`** — checks `data/hard_buckets/bucket_list.json` manifest then scans for any .pt file. Returns None if nothing found.

2. **`_generate_synthetic_pt()`** — creates a .pt matching S01's Phase 2 schema from smokers_20 (problem 0). Runs exact elimination to the first NN-eligible bucket (label 400), computes exact_fw via `bucket.compute_message_exact()` and exact_bw via `get_backward_message()`, saves with all required schema fields.

3. **Main flow** — finds or generates .pt, builds nn_config from small_problems default config (with verification overrides: UKL loss, [3,3] hidden, 100k epochs capped by time limit), calls `train_single_bucket()`, runs validation.

4. **`_run_checks()`** — 8 structured checks: epochs_completed > 0, error_tracking_data non-empty, wall_time > 0, losses non-empty, output dir exists, loss.png non-zero, local_error.png non-zero, metrics.json parseable with all 9 required keys.

## Verification

**Script exit code:** `python scripts/verify_benchmark_training.py --time-limit 30` exits 0 on CUDA.

**All 8 checks pass:**
- epochs_completed=1337
- error_tracking_data has 10 entries
- wall_time=30.02s
- losses has 1337 entries
- output dir exists
- loss.png exists (38042 bytes)
- local_error.png exists (21575 bytes)
- metrics.json valid with all 9 required keys

**CPU path also verified:** `--device cpu --time-limit 15` exits 0 with all checks passing.

**Slice-level verification (all pass):**
- ✅ `python scripts/verify_benchmark_training.py` exits 0
- ✅ Output folder contains loss.png, local_error.png, metrics.json
- ✅ metrics.json parseable with expected keys present
- ✅ loss.png and local_error.png are valid PNG files (non-zero size)
- ✅ Error tracking data has entries at expected checkpoint epochs (10 entries)
- ✅ epochs_completed > 0 (1337 epochs in 30s)
- ✅ Synthetic .pt generated and used (no real .pt files from S01 available)

## Diagnostics

- Run: `python scripts/verify_benchmark_training.py --time-limit 30` — exit code tells the story
- Stdout has `[Verify]` prefixed lifecycle messages and per-check PASS/FAIL lines
- On failure: prints which specific check failed and actual vs expected values
- Output files in `/tmp/benchmark_training_verify/{bucket_id}/` for manual inspection

## Deviations

- Config construction uses `copy.deepcopy(small_problems.configs['default'][0])` as base instead of building a minimal dict from scratch. The task plan specified explicit fields, but Trainer.__init__ requires many fields (`lower_dim`, `debug`, `traced_losses`, `skip_early_stopping`, etc.) that `prepare_config(strict=False)` doesn't auto-populate. Starting from the full default config is the correct approach.

## Known Issues

None.

## Files Created/Modified

- `scripts/verify_benchmark_training.py` — end-to-end verification script (~270 lines) with synthetic .pt fallback, structured validation checks, and PASS/FAIL output
