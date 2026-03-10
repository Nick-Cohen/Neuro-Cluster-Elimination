---
phase: quick-20
plan: 1
subsystem: visualization
tags: [benchmark, wmse, ukl, visualization, csv, plots]
dependency_graph:
  requires: [benchmark WMSE vs UKL result JSONs in results/{config}/*.json]
  provides: [per-problem PNG bar charts, summary.csv]
  affects: [benchmark analysis workflow]
tech_stack:
  added: []
  patterns: [matplotlib symlog scale, pandas pivot, per-problem grouped bar charts]
key_files:
  created:
    - notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/visualize_updated.py
    - notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/results/updated_graphs_and_table/summary.csv
    - notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/results/updated_graphs_and_table/ (24 PNG files)
  modified: []
decisions:
  - Use symlog scale with linthresh=1e-3 for log-scale plots to handle near-zero values gracefully
  - Ground truth dashed line at y=0 on absolute error plots
  - Summary CSV covers all 96 completed experiments (not just the 12 clean problems)
metrics:
  duration: 2 min
  completed: 2026-03-10
  tasks_completed: 1
  files_created: 27
---

# Quick Task 20: Update Benchmark WMSE/UKL Graphs with Per-Problem Error Plots Summary

**One-liner:** Per-problem absolute error bar charts (linear + symlog scale) and 96-row summary CSV for the 24-problem WMSE vs UKL benchmark, using Red/Orange/Gold/Green/Blue color scheme.

## Objective Achieved

Created `visualize_updated.py` that reads existing benchmark WMSE vs UKL result JSONs and produces:
1. 24 PNG bar charts (12 clean problems x 2 scale types: linear + symlog)
2. A 96-row summary CSV with all required columns
3. Printed config analysis confirming track_errors=False for all experiments

## Tasks Completed

| Task | Name | Commit | Files Created/Modified |
|------|------|--------|------------------------|
| 1 | Create visualize_updated.py with per-problem absolute error plots and summary CSV | 47754d9 | visualize_updated.py, 24 PNGs, summary.csv |

## Key Details

### Data Summary
- **Total completed experiments loaded:** 96 (out of 120 total; 24 failed on 3 deer_rescaled models)
- **Clean problems (all 5 configs):** 12 problems
- **Problems with 4/5 configs:** 9 additional problems

### Plots Generated
- 12 problems x 2 scale types = 24 PNG files
- Each plot: 5 bars (one per config) with exact color scheme + black dashed line at y=0
- Linear scale: raw absolute error values
- Log scale: symlog with linthresh=1e-3 for visibility of near-zero errors

### Summary CSV Columns
`problem_name, width, num_vars, log_Z_ground_truth, log_Z_hat, err, abs_err, num_trained, time, num_samples, architecture, loss_fn, config_name`

### Config Analysis Answer
- `track_errors=False` in all experiments (default from small_problems config, not overridden by benchmark runner)

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check

- [x] `notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/visualize_updated.py` - FOUND (220 lines, above 150 minimum)
- [x] `results/updated_graphs_and_table/*.png` - 24 PNG files FOUND
- [x] `results/updated_graphs_and_table/summary.csv` - FOUND (96 rows, 13 columns)
- [x] Commit 47754d9 - FOUND

## Self-Check: PASSED
