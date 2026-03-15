---
id: T02
parent: S02
milestone: M003
provides:
  - Experiment runner script for all 15 neurobe_binary problems
  - Fixed NEUROBE_DEFAULTS missing keys (approximation_method, debug, traced_losses, optimizer)
  - Fixed neurobe_weighted_mse closure passing spurious bw_hat argument
  - Fixed train.py normalizing_constant print crash for minmax_01 mode
key_files:
  - scripts/run_neurobe_experiments.py
  - notebooks/March-2025/neurobe_comparison_results.csv (pending — experiment in progress)
key_decisions:
  - Added approximation_method, debug, traced_losses, optimizer to NEUROBE_DEFAULTS rather than per-config overrides — keeps neurobe_mode self-contained
patterns_established:
  - neurobe_mode configs should be self-sufficient through NEUROBE_DEFAULTS without requiring callers to add boilerplate keys
observability_surfaces:
  - Per-problem progress prints during run (problem name, NNs trained, log_Z, time)
  - CSV output with Status and Error columns for post-hoc failure inspection
  - Script exit code 0 = all 15 success, 1 = any failure
duration: ~45min (implementation + debugging; experiment runtime estimated 1-3hrs additional)
verification_result: partial — smoke test passed (BN_3: 1 NN, log_Z=-12.753501, 12s), full run in progress
completed_at: 2026-03-15
blocker_discovered: false
---

# T02: Run neurobe_mode experiments on all 15 problems

**Fixed 3 bugs in the neurobe_mode training pipeline and launched full 15-problem experiment on CUDA (running in background).**

## What Happened

Step 1 (GPU check): Confirmed 3x TITAN RTX available, 1MiB used, no stale experiment processes.

Step 2 (experiment script): Wrote `scripts/run_neurobe_experiments.py` — iterates neurobe_binary benchmark set, runs `FastGM(model, config, device='cuda')` + `eliminate_variables(all=True)` per problem, captures `log_partition_function`, `num_trained`, wall time. Per-problem try/except, writes CSV, prints summary.

Step 3 (bug fixes): Three bugs discovered and fixed during smoke testing:

1. **Missing `approximation_method` in NEUROBE_DEFAULTS** — `process_bucket()` reads `self.config.get('approximation_method')` which returned `None` because the schema default isn't injected by `prepare_config`. Fix: added `'approximation_method': 'nn'` to NEUROBE_DEFAULTS.

2. **Missing `debug`, `traced_losses`, `optimizer` in NEUROBE_DEFAULTS** — `Trainer.__init__` accesses these with `self.config['key']` (bracket access, no default). Fix: added all three to NEUROBE_DEFAULTS (`False`, `[]`, `'adam'`).

3. **`neurobe_weighted_mse` closure bug** — The wrapper lambda passed `bw_hat` as 3rd positional arg to `neurobe_weighted_mse(outputs, targets, ln_min, ln_max, sum_ln)`, causing "multiple values for argument 'ln_min'". Fix: removed `bw_hat` from the positional args since the function doesn't use backward messages.

4. **`normalizing_constant` print crash in minmax_01 mode** — Line 210 of `train.py` tried `f"{self.data_preprocessor.normalizing_constant:.4f}"` but minmax_01 mode leaves `normalizing_constant=None`. Fix: conditional print that shows minmax_01 stats instead.

Step 4 (launch): Smoke-tested BN_3 (1 NN) — succeeded in 12s, log_Z=-12.753501, early stopped at epoch 10. Launched full 15-problem run via bg_shell. Process is actively training on GPU 0 (543MiB, 5% utilization, 195% CPU). BN_1's first NN early-stopped at epoch 66. Run is in progress.

## Verification

**Passed:**
- `python -m pytest tests/ -v` → 134 passed, 0 failed (after all fixes)
- `python scripts/verify_nn_counts.py` → 15/15 MATCH
- Smoke test: BN_3 completed successfully (1 NN, log_Z=-12.753501, 12s)
- GPU available and free before launch (nvidia-smi: 1MiB on all 3 GPUs)

**Pending (experiment still running):**
- CSV exists at `notebooks/March-2025/neurobe_comparison_results.csv` with 15 data rows
- All 15 Status=success
- NCE_NNs match expected counts

**Slice-level verification:**
- ✅ `python -c "from nce.benchmark_problems import neurobe_binary; print(len(neurobe_binary.problems))"` → 15
- ✅ `python scripts/verify_nn_counts.py` → all 15 match
- ✅ `python -m pytest tests/ -v` → 134 passed
- ⏳ `ls notebooks/March-2025/neurobe_comparison_results.csv` → pending (experiment running)
- ⏳ Combined comparison table — T03

## Diagnostics

- Check experiment status: `ps aux | grep run_neurobe | grep -v grep` (should show python process)
- Check GPU usage: `nvidia-smi` (GPU 0 should show memory in use)
- When complete, CSV at `notebooks/March-2025/neurobe_comparison_results.csv`
- Process PID: 3291719 (launched from bg_shell id af3efaf4)

## Resume Notes

**If experiment completes successfully:** Verify CSV has 15 rows all Status=success, check NCE_NNs match expected. Mark T02 fully done.

**If experiment fails partway:** Check CSV for partial results, inspect Error column. Rerun only failed problems or full run. The script is idempotent (overwrites CSV).

**To check if experiment is done:** `ls -la notebooks/March-2025/neurobe_comparison_results.csv` — if file exists AND process is no longer running, it's done.

## Deviations

- Three bug fixes to core NCE code (config_schema.py, train.py) were necessary to make neurobe_mode work end-to-end. These are not deviations from the plan — they're the expected integration work when running the pipeline for the first time with these configs.

## Known Issues

- Experiment is in progress — CSV will be written only when all 15 problems complete (or fail)
- tqdm notebook progress bars don't emit to stdout in non-notebook context, so per-epoch progress isn't visible in bg_shell output (only early stopping messages and per-problem summaries appear)

## Files Created/Modified

- `scripts/run_neurobe_experiments.py` — **new** — experiment runner for 15 neurobe_binary problems
- `nce/config_schema.py` — **modified** — added `approximation_method`, `debug`, `traced_losses`, `optimizer` to NEUROBE_DEFAULTS
- `nce/neural_networks/train.py` — **modified** — fixed neurobe_weighted_mse closure bw_hat bug; fixed normalizing_constant print for minmax_01 mode
- `notebooks/March-2025/` — **created** — output directory for results
