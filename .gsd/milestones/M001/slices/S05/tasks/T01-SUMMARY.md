---
id: T01
parent: S05
milestone: M001
provides:
  - nce/visualization/ module with plot_learning_curves and _extract_training_log
key_files:
  - nce/visualization/__init__.py
  - nce/visualization/learning_curves.py
key_decisions:
  - Use matplotlib Agg backend by default for headless safety
  - squeeze=False on subplots for consistent 2D axes array indexing
patterns_established:
  - _extract_training_log normalizes FastGM/dict/path to list of entry dicts — reusable by T02
  - Visualization functions return Figure, never call plt.show()
observability_surfaces:
  - TypeError with source type name on unrecognized input
  - UserWarning when max_subplots truncates entries
  - fig.axes / ax.lines / ax.get_title() inspectable by agents
duration: 15m
verification_result: passed
completed_at: 2026-03-12
blocker_discovered: false
---

# T01: Create visualization module with plot_learning_curves

**Built `nce/visualization/` module with `plot_learning_curves()` and `_extract_training_log()` helper accepting FastGM, dict, or file path sources.**

## What Happened

Created the `nce/visualization/` package with two files. `_extract_training_log(source)` normalizes three input types (live FastGM via attribute, state dict via key lookup, file path via `load_state()`) and raises `TypeError` with the source type name on anything else. `plot_learning_curves()` filters to entries with non-empty losses, optionally filters by `bucket_labels`, caps at `max_subplots` with a warning, computes an automatic grid layout (up to 4 columns), plots train loss on every subplot and val loss where present, annotates hidden_sizes in subplot titles, and handles the zero-entry edge case with an informational text figure. `save_path` kwarg writes to disk. No `plt.show()` calls. No imports from `nce.inference.bucket` or `nce.inference.factor_nn`.

## Verification

All checks passed:
- `python -c "from nce.visualization import plot_learning_curves; print('ok')"` → ok
- `python -c "from nce.visualization.learning_curves import _extract_training_log; print('ok')"` → ok
- Smoke test with synthetic state dict: 3 entries → 3 subplots with correct line counts
- Empty losses entry correctly skipped (2 entries, 1 empty → 1 subplot)
- Zero plottable entries → informational figure, no crash
- max_subplots=5 with 25 entries → 5 visible subplots + warning emitted
- save_path wrote 44KB PNG to disk
- bucket_labels=[5, 3] filter → 2 subplots
- TypeError on int input with type name in message
- No `plt.show()` calls in module (only in docstring text)
- No forbidden imports from nce.inference.bucket or nce.inference.factor_nn

Slice-level checks (partial — T01 is intermediate):
- `from nce.visualization import plot_learning_curves` → PASS
- `from nce.visualization import compare_experiments` → expected ImportError (T02 not yet implemented)
- `scripts/verify_s05_visualization.py` → not yet created (T03)

## Diagnostics

- `_extract_training_log()` raises `TypeError` naming the unrecognized type for easy debugging
- `UserWarning` emitted when entries exceed `max_subplots`
- Returned `Figure` inspectable: `fig.axes` for subplot count, `ax.get_lines()` for plotted series, `ax.get_title()` for bucket label + hidden_sizes

## Deviations

None.

## Known Issues

None.

## Files Created/Modified

- `nce/visualization/__init__.py` — package init with re-export of plot_learning_curves
- `nce/visualization/learning_curves.py` — _extract_training_log helper + plot_learning_curves function
