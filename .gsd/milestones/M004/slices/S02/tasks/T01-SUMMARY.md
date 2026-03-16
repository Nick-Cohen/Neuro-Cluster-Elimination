---
id: T01
parent: S02
milestone: M004
provides:
  - nce/benchmark/ module with train_single_bucket() core function
  - Custom epoch loop with wall-clock time limit and checkpoint error tracking
  - Preloaded exact message error tracking (FactorNN.to_exact() pattern)
key_files:
  - nce/benchmark/__init__.py
  - nce/benchmark/training.py
key_decisions:
  - Custom epoch loop over Trainer.train() — avoids 600-line monolith's concerns (early stopping, validation, display, traced losses)
  - Trainer used for __init__ chain only (SampleGenerator, DataLoader, DataPreprocessor, loss_fn)
  - exact_contribution precomputed once before loop, not per checkpoint
  - Config goes through prepare_config(strict=False) for backward compat
patterns_established:
  - _load_bucket_data() reconstructs FastFactor objects from .pt schema on target device
  - _find_problem() maps problem_key (modelfile) to small_problems index
  - _reconstruct_bucket() uses eliminate_variables(up_to=target_var, exact=True) pattern from S01
  - Checkpoint error formula: (FactorNN.to_exact() * exact_bw).sum_all_entries() - exact_contribution
observability_surfaces:
  - "[BenchmarkTraining]" prefixed prints at load, reconstruct, train start, checkpoint, time limit, complete
  - Return dict contains full training metadata (epochs_completed, losses, error_tracking_data, wall_time, config_used)
duration: 30min
verification_result: passed
completed_at: 2026-03-12
blocker_discovered: false
---

# T01: Build train_single_bucket() core with custom training loop

**Built `nce/benchmark/training.py` with the full single-bucket training lifecycle: .pt loading, FastGM reconstruction, custom epoch loop with time-limit and checkpoint error tracking.**

## What Happened

Created the `nce/benchmark/` module with three helper functions and the main `train_single_bucket()` entry point:

- `_load_bucket_data(pt_path, device)` — loads .pt file, reconstructs FastFactor objects (exact_fw, exact_bw, bucket factors) on target device, returns structured dict with metadata
- `_find_problem(problem_key)` — maps modelfile string to small_problems index with clear error on miss
- `_reconstruct_bucket(problem_idx, bucket_label, nn_config, device)` — creates FastGM, runs `eliminate_variables(up_to=target_var, exact=True)`, returns live (fastgm, bucket) pair

The main function follows the designed lifecycle: load .pt → find problem → merge config (ecl from .pt, enforce error_tracking=False, sampling_scheme='all') → prepare_config() → reconstruct → create Net + Trainer (setup only) → load_all() → batch creation → precompute exact_contribution → epoch 0 checkpoint → custom loop with train_epoch + checkpoint error + time check → return result dict.

The epoch loop mirrors train.py's pattern: calls `trainer.train_epoch(batches)`, records loss, steps scheduler, computes error at checkpoint epochs via `FactorNN(net, dp).to_exact()` × `exact_bw` − `exact_contribution`. Time limit checked at epoch boundary (overshoots by at most one epoch duration).

No plotting or metrics.json output in this task — that's T02's scope.

## Verification

- `python -c "from nce.benchmark.training import train_single_bucket; print('import ok')"` → passes ✓
- `python -c "from nce.benchmark import train_single_bucket; print('public API ok')"` → passes ✓
- Signature verification: `inspect.signature()` confirms exact parameter names match spec ✓
- Code review: epoch loop pattern matches train.py:497-517 for loss aggregation and error tracking ✓
- All 10 must-haves from task plan checked and satisfied ✓

**Slice-level verification (partial, expected for T01):**
- ❌ `verify_benchmark_training.py` — not yet written (T03)
- ❌ metrics.json / loss.png / local_error.png — not yet generated (T02)
- ✅ Import succeeds, function signature correct
- ❌ End-to-end execution — requires .pt files (either real from S01 Phase 2 or synthetic from T03)

## Diagnostics

- Read stdout for `[BenchmarkTraining]` prefixed messages showing lifecycle progress
- Return value dict contains `epochs_completed`, `losses`, `error_tracking_data`, `wall_time`, `config_used` for programmatic inspection
- Exceptions include problem_key and bucket_label context in error messages
- If time limit hit mid-training, partial results returned with `epochs_completed` reflecting actual count

## Deviations

None.

## Known Issues

- No .pt files exist yet (S01 Phase 2 hasn't completed). End-to-end testing deferred to T03 which will generate a synthetic .pt if needed.
- The `output_dir` parameter is accepted but not used in T01 — T02 will add plot/metrics output.

## Files Created/Modified

- `nce/benchmark/__init__.py` — Module init with public export of train_single_bucket (new)
- `nce/benchmark/training.py` — Core training function with helpers (~280 lines, new)
