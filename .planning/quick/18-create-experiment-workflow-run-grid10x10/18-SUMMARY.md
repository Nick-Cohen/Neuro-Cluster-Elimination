---
phase: quick-18
plan: 1
subsystem: experiment-workflow
tags: [experiment, grid10x10, ukl, benchmark, workflow]
dependency_graph:
  requires: [catalog_utils, FastGM, bucket.py, graphical_model.py]
  provides: [grid10x10_ukl_runner, grid10x10_ukl_analysis, per_bucket_training_log]
  affects: [nce/inference/bucket.py, nce/inference/graphical_model.py]
tech_stack:
  added: []
  patterns: [nohup-background-execution, per_bucket_training_log-pattern, model-cache-offline-access]
key_files:
  created:
    - notebooks/3-2026/claude_experiments/grid10x10_ukl/run_grid10x10_ukl.py
    - notebooks/3-2026/claude_experiments/grid10x10_ukl/analyze_results.py
    - notebooks/3-2026/claude_experiments/grid10x10_ukl/results/grid10x10_f10_ukl_results.csv
    - notebooks/3-2026/claude_experiments/grid10x10_ukl/results/experiment_results.pkl
    - notebooks/3-2026/claude_experiments/grid10x10_ukl/results/logz_comparison.png
    - notebooks/3-2026/claude_experiments/grid10x10_ukl/results/bucket_epochs_histogram.png
    - .model_cache/grids/grid10x10.f10.uai
    - .model_cache/grids/grid10x10.f10.uai.ord
  modified:
    - nce/inference/bucket.py
    - nce/inference/graphical_model.py
decisions:
  - fastgm.buckets is empty after elimination (buckets deleted post-compute_message_nn); per-bucket data must be stored during inference on fastgm.per_bucket_training_log
  - grid10x10.f10.uai not in model cache; copied from /home/cohenn1/UAI/Submissions/IBIA-PR-V2/test-results/1200/
  - .ord file generated via pyGMs eliminationOrder('minfill') since no pre-computed order existed offline
  - CUDA_VISIBLE_DEVICES=1 chosen (GPU 1 had 14.8GB free) as GPU 0 was at 99% utilization
  - tqdm shows 0/500 in background nohup (non-TTY) - normal behavior, training still runs correctly
metrics:
  duration: 35 min (including 3 failed attempts due to network timeout and per-bucket bug)
  completed: 2026-03-09
  tasks_completed: 2
  files_created: 8
  files_modified: 2
---

# Phase quick-18 Plan 1: Grid10x10 UKL Experiment Workflow Summary

**One-liner:** Self-contained grid10x10.f10 UKL experiment runner with CSV/pickle outputs, analysis script, and reproducible workflow pattern.

## What Was Built

1. **run_grid10x10_ukl.py**: Complete experiment runner script with:
   - Full 42-field config (UKL loss, ecl=1024, bw_ecl=0, num_epochs=500, iB=10, CUDA)
   - Pre-flight CUDA check (exits with error if no GPU)
   - Loads model from catalog (`get_catalog()['grids/grid10x10.f10']`)
   - Runs FastGM inference with timing
   - Collects per-bucket training data from `fastgm.per_bucket_training_log`
   - Outputs CSV (12 columns) and pickle to `results/`
   - Discord ping on completion

2. **analyze_results.py**: Analysis script that:
   - Loads pickle file (works from pickle alone -- no NCE imports needed at analysis time)
   - Prints summary table
   - Generates bar chart: log_Z_star vs log_Z_hat (`logz_comparison.png`)
   - Generates per-bucket epochs histogram (`bucket_epochs_histogram.png`)
   - Prints per-bucket hidden sizes table
   - Regenerates CSV from pickle (round-trip proof)

## Experiment Results

| Metric | Value |
|--------|-------|
| Model | grid10x10.f10.uai |
| Width | 12 |
| Num vars | 100 |
| log_Z_star | 303.085957 (log10) |
| log_Z_hat | 316.571228 (log10) |
| err | +13.485271 (log10) |
| abs_err | 13.485271 (log10) |
| num_trained | 6 NN buckets |
| time_seconds | 85.1s |
| architecture | nbe,1 |
| loss_fn | unnormalized_kl |

**NN Buckets (6 total):** Variables 4, 15, 24, 42, 45, 51 -- all trained 500 epochs, hidden sizes 11-13.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocker] grid10x10.f10.uai not in model cache (network timeout)**
- **Found during:** Task 2 first run
- **Issue:** Catalog tried to download `grid10x10.f10.uai` from sli.ics.uci.edu which timed out
- **Fix:** Copied existing file from `/home/cohenn1/UAI/Submissions/IBIA-PR-V2/test-results/1200/grid10x10.f10.uai` to `.model_cache/grids/`
- **Also needed:** Generated `.ord` elimination order file via `pyGMs.eliminationOrder('minfill')` since no pre-computed order existed
- **Commit:** 8cc2401

**2. [Rule 1 - Bug] Per-bucket data collection used wrong approach (fastgm.buckets is empty after inference)**
- **Found during:** Task 2 first run, second attempt
- **Issue:** Runner iterated `fastgm.buckets.items()` but buckets are deleted from the dict during elimination (`del self.buckets[var]` in graphical_model.py line 307). Dict is empty after `get_log_partition_function()` returns.
- **Fix:**
  - Added `per_bucket_training_log = []` attribute to `FastGM.__init__`
  - Modified `bucket.py compute_message_nn()` to append `{label, epochs_trained, hidden_sizes}` to `self.gm.per_bucket_training_log` (alongside existing `self.gm.num_trained += 1`)
  - Updated runner to use `fastgm.per_bucket_training_log` instead of `fastgm.buckets`
- **Files modified:** `nce/inference/bucket.py`, `nce/inference/graphical_model.py`, `run_grid10x10_ukl.py`
- **Commit:** 8cc2401

**3. [Rule 3 - Blocker] FastGM returned None from uai_to_GM (no .vo order file)**
- **Found during:** Task 2 initial debugging
- **Root cause:** `FastGM._load_from_uai()` calls `uai_to_GM(uai_file, order_file=uai+'.vo', elim_order=None)`. With no `.vo` file and no pre-computed order from catalog (catalog couldn't download `.ord`), `uai_to_GM` returns `None`. Calling `None.condition(evid)` raised AttributeError.
- **Fix:** Resolved by generating the `.ord` file (Fix #1 above) -- catalog then reads it successfully and passes `elim_order` to `uai_to_GM`.
- **Commit:** 8cc2401

## Lessons for Future Agents

1. **fastgm.buckets is empty after inference.** Do NOT iterate `fastgm.buckets` after `get_log_partition_function()`. Use `fastgm.per_bucket_training_log` (now populated by bucket.py).

2. **Model cache is offline.** If a model is not in `.model_cache/grids/`, copying from `/home/cohenn1/UAI/` is the solution. Network access to sli.ics.uci.edu times out. You'll also need a `.ord` file -- generate with `pyGMs.eliminationOrder('minfill')`.

3. **tqdm shows 0% in nohup background.** This is normal. Training still runs correctly.

4. **GPU selection.** GPUs 1 and 2 typically have more free memory than GPU 0 (which hosts many vscode-server processes). Check with `nvidia-smi` before selecting.

## Output Files

- `notebooks/3-2026/claude_experiments/grid10x10_ukl/run_grid10x10_ukl.py` -- Runner script
- `notebooks/3-2026/claude_experiments/grid10x10_ukl/analyze_results.py` -- Analysis script
- `notebooks/3-2026/claude_experiments/grid10x10_ukl/results/grid10x10_f10_ukl_results.csv` -- 12-column CSV
- `notebooks/3-2026/claude_experiments/grid10x10_ukl/results/experiment_results.pkl` -- Full state pickle
- `notebooks/3-2026/claude_experiments/grid10x10_ukl/results/logz_comparison.png` -- Bar chart
- `notebooks/3-2026/claude_experiments/grid10x10_ukl/results/bucket_epochs_histogram.png` -- Epochs histogram

## Commits

- `42ca576`: feat(quick-18): create grid10x10.f10 UKL experiment runner and analysis scripts
- `8cc2401`: feat(quick-18): execute grid10x10.f10 UKL experiment and fix per-bucket data collection

## Self-Check: PASSED

- `notebooks/3-2026/claude_experiments/grid10x10_ukl/run_grid10x10_ukl.py` -- FOUND
- `notebooks/3-2026/claude_experiments/grid10x10_ukl/analyze_results.py` -- FOUND
- `notebooks/3-2026/claude_experiments/grid10x10_ukl/results/grid10x10_f10_ukl_results.csv` -- FOUND
- `notebooks/3-2026/claude_experiments/grid10x10_ukl/results/experiment_results.pkl` -- FOUND
- `notebooks/3-2026/claude_experiments/grid10x10_ukl/results/logz_comparison.png` -- FOUND
- `notebooks/3-2026/claude_experiments/grid10x10_ukl/results/bucket_epochs_histogram.png` -- FOUND
- Commit `42ca576` -- FOUND
- Commit `8cc2401` -- FOUND
