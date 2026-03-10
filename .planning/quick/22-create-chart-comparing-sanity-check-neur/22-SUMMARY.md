---
phase: quick-22
plan: 1
subsystem: visualization
tags: [comparison, paper-results, wmb, neurobe, bar-chart, matplotlib]
dependency_graph:
  requires: [NCE-Data/NeuroBE_paper_results.csv, notebooks/March-2026/claude_experiments/nbe_eval_results/nbe_full_epochs.txt]
  provides: [notebooks/3-2026/claude_experiments/compare_paper_vs_sanity_check.py, notebooks/3-2026/claude_experiments/compare_paper_vs_sanity_check.png]
  affects: []
tech_stack:
  added: []
  patterns: [grouped-bar-chart, log-scale-y-axis, dual-subplot-figure, raw-values-table]
key_files:
  created:
    - notebooks/3-2026/claude_experiments/compare_paper_vs_sanity_check.py
    - notebooks/3-2026/claude_experiments/compare_paper_vs_sanity_check.png
  modified: []
decisions:
  - "Used log-scale y-axis to handle 5-order-of-magnitude error range (0.0007 to 215)"
  - "5 bar groups per problem: Paper WMB, Paper NeuroBE avg, Paper NeuroBE min, Our WMB (ecl=2^(iB-1)), Our WMB (ecl=2^22)"
  - "grid40x40.f10 exact WMB bar absent in chart (Phase 3 network timeout); marked as 'err' in table"
  - "Right subplot uses matplotlib table for raw values display (no extra library dependency)"
  - "Rotated value labels 90 degrees to prevent overlap on dense grouped bars"
metrics:
  duration: 2 min
  completed_date: 2026-03-10
  tasks_completed: 1
  files_created: 2
---

# Phase quick-22 Plan 1: Paper vs Sanity Check Comparison Chart Summary

**One-liner:** Grouped bar chart (log scale) comparing paper WMB/NeuroBE errors vs our WMB errors for 4 overlapping problems using refZ from paper CSV as ground truth.

## What Was Built

A Python script that produces a side-by-side comparison figure with:

- **Left subplot:** Grouped bar chart (5 bars per problem, log-scale y-axis) showing absolute error in log Z for:
  1. Paper WMB error
  2. Paper NeuroBE average error
  3. Paper NeuroBE minimum error
  4. Our WMB error (ecl=2^(iB-1), from nbe_full_epochs.txt - pure WMB, 0 NN buckets)
  5. Our exact WMB error (ecl=2^22, from Phase 3 results)

- **Right subplot:** Raw values table with columns: Problem, refZ, PaperWMB, PaperNBE avg, PaperNBE min, OurWMB (ecl=2^(iB-1)), OurWMB (ecl=2^22)

## Task Commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create comparison chart script and generate PNG | 5453c6f | notebooks/3-2026/claude_experiments/compare_paper_vs_sanity_check.py |

## Key Results

Error values computed as |our_logZ - refZ| and |phase3_logZ - refZ|:

| Problem | refZ | Paper WMB | Paper NeuroBE avg | Paper NeuroBE min | Our WMB (ecl=2^(iB-1)) | Our WMB (ecl=2^22) |
|---------|------|-----------|-------------------|-------------------|------------------------|-------------------|
| pedigree13 | -31.18 | 6.4696 | 1.1100 | 0.7600 | 0.5841 | 7.7566 |
| grid40x40.f10 | 5490 | 215.4500 | 24.0000 | 16.5100 | 140.6514 | N/A (errored) |
| grid20x20.f10 | 1311.98 | 80.8600 | 1.3700 | 0.0800 | 114.6690 | 20.5342 |
| rbm_20 | 58.53 | 0.0007 | 0.2300 | 0.0020 | 0.5428 | 0.0006 |

**Notable observations:**
- pedigree13: our WMB (ecl=2^(iB-1)) at 0.58 is actually BETTER than paper WMB (6.47) - our i-bound setting may differ
- grid20x20.f10 and grid40x40.f10: our WMB errors are worse than paper WMB (high errors suggest the ecl=2^(iB-1) threshold may not be ideal for these)
- rbm_20: our exact WMB (ecl=2^22) at 0.0006 matches paper WMB (0.0007) very closely

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check

- [x] Script exists: /home/cohenn1/NCE/notebooks/3-2026/claude_experiments/compare_paper_vs_sanity_check.py (251 lines, >= 60 required)
- [x] PNG exists: /home/cohenn1/NCE/notebooks/3-2026/claude_experiments/compare_paper_vs_sanity_check.png (145,957 bytes, non-empty)
- [x] Reads from NCE-Data/NeuroBE_paper_results.csv via pd.read_csv
- [x] Hardcodes nbe_full_epochs results dict
- [x] All 4 matching problems represented in chart
- [x] Errors computed using refZ from paper CSV as ground truth
- [x] pedigree13 spot-check: |(-31.764139) - (-31.18)| = 0.584139 confirmed
- [x] Commit exists: 5453c6f

## Self-Check: PASSED
