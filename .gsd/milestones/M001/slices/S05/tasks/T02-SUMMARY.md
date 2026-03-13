---
id: T02
parent: S05
milestone: M001
provides:
  - compare_experiments() function in nce/visualization/comparison.py
key_files:
  - nce/visualization/comparison.py
  - nce/visualization/__init__.py
key_decisions:
  - Use numpy for bar chart positioning; grouped bars with offset calculation
  - Shared bucket labels defined as present in ≥2 experiments (not necessarily all)
patterns_established:
  - compare_experiments returns Figure with summary bar chart as first visible axis, overlays after
  - _get_final_loss helper extracts last (epoch, loss) tuple value from entry
observability_surfaces:
  - ValueError on empty sources list with descriptive message
  - ValueError on labels/sources length mismatch
  - fig.axes inspectable — first visible axis is summary panel, remaining are per-bucket overlays
  - Summary axis title contains "no shared buckets" when experiments have disjoint labels
  - UserWarning when shared labels exceed max_subplots
duration: 15m
verification_result: passed
completed_at: 2026-03-12
blocker_discovered: false
---

# T02: Create compare_experiments function

**Built `compare_experiments()` for cross-experiment training comparison with summary bar chart and per-bucket overlay panels.**

## What Happened

Created `nce/visualization/comparison.py` with `compare_experiments(sources, labels, save_path, max_subplots)`. The function normalizes each source via `_extract_training_log()` from T01, builds a bucket-label index across experiments, identifies shared labels (present in ≥2 experiments), then renders: (1) a grouped bar chart of final loss per bucket per experiment, and (2) per-shared-bucket overlay subplots with learning curves from each experiment. Uses `squeeze=False` pattern from T01 for consistent axis indexing. Auto-generates "Experiment N" labels when none provided. Updated `__init__.py` to re-export `compare_experiments`.

## Verification

- `python -c "from nce.visualization import compare_experiments; print('ok')"` — passes
- Smoke test: two synthetic state dicts with 3 overlapping + 1 unique bucket label each → Figure has 4 visible axes (1 summary + 3 overlays) — passes
- Edge case: two experiments with zero overlapping labels → 1 visible axis (summary only), title mentions "no shared buckets" — passes
- Edge case: empty sources list → `ValueError` raised — passes
- Edge case: single experiment → summary-only figure — passes
- Edge case: `save_path` kwarg → file written to disk with nonzero size — passes
- Slice import check: `from nce.visualization import plot_learning_curves, compare_experiments` — passes
- Slice verification script (`scripts/verify_s05_visualization.py`) does not exist yet — expected, created in T03

## Diagnostics

- `fig.axes` on returned Figure: first visible axis is summary bar chart, remaining are per-bucket overlays
- Summary axis title: "Final Loss per Bucket — Summary" (or "no shared buckets" variant)
- Overlay axes titled "Bucket {label}" with legend identifying experiments
- `ValueError` raised with descriptive message on empty sources or label-count mismatch
- `UserWarning` emitted when shared bucket count exceeds `max_subplots`

## Deviations

None.

## Known Issues

None.

## Files Created/Modified

- `nce/visualization/comparison.py` — new file: `compare_experiments()` function with `_get_final_loss` and `_draw_summary_bars` helpers
- `nce/visualization/__init__.py` — added re-export of `compare_experiments`
