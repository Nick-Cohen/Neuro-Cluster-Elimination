---
estimated_steps: 5
estimated_files: 3
---

# T01: Create visualization module with plot_learning_curves

**Slice:** S05 — Visualization Module
**Milestone:** M001

## Description

Create `nce/visualization/` module with the core `plot_learning_curves()` function and a shared `_extract_training_log()` helper. The function accepts a live FastGM, a state dict (from `load_state()`), or a file path (str/Path), and produces a matplotlib Figure with one subplot per NN-trained bucket showing loss over epochs. Train loss always plotted; validation loss shown when present. Handles edge cases (empty losses, zero NN entries, large bucket counts via `max_subplots`). Returns the Figure — no `plt.show()` calls.

## Steps

1. Create `nce/visualization/__init__.py` with re-exports of `plot_learning_curves` (and placeholder for `compare_experiments` to be added in T02)
2. Create `nce/visualization/learning_curves.py`:
   - Implement `_extract_training_log(source)` helper:
     - If source has `per_bucket_training_log` attribute → use it (FastGM)
     - If source is a dict with `per_bucket_training_log` key → use it (state dict)
     - If source is str/Path → call `load_state(source)` and extract
     - Otherwise → raise `TypeError` naming the unrecognized type
   - Implement `plot_learning_curves(source, save_path=None, max_subplots=20, bucket_labels=None)`:
     - Extract training log via helper
     - Filter to entries with non-empty `losses` (skip entries where training didn't happen)
     - If `bucket_labels` provided, filter to matching labels
     - Cap at `max_subplots` entries (take first N, warn if truncated)
     - Compute grid layout: `ncols = min(4, n_entries)`, `nrows = ceil(n_entries / ncols)`
     - Create subplots, plot each entry: epochs on x-axis (from tuple[0]), loss on y-axis (from tuple[1])
     - Plot val_losses if non-empty
     - Subplot title: `f"Bucket {entry['label']}"` with hidden_sizes annotation
     - Handle edge case: zero plottable entries → return empty Figure with text annotation
     - If `save_path` provided, call `fig.savefig(save_path, bbox_inches='tight')`
     - Return Figure
3. Verify import works: `python -c "from nce.visualization import plot_learning_curves; print('ok')"`
4. Quick smoke test with a synthetic training log dict to confirm subplot creation
5. Document memory note in docstring: caller should `plt.close(fig)` when done

## Must-Haves

- [ ] `_extract_training_log` accepts FastGM, dict, and str/Path — raises TypeError on unknown
- [ ] `plot_learning_curves` returns matplotlib Figure with one subplot per NN entry
- [ ] Train loss plotted for every entry; val loss plotted when `val_losses` is non-empty
- [ ] Empty losses list → entry skipped (no empty subplot)
- [ ] Zero plottable entries → Figure with informational text, not a crash
- [ ] `max_subplots` caps subplot count; excess entries noted in figure title
- [ ] `save_path` kwarg writes figure to disk
- [ ] No `plt.show()` calls
- [ ] No imports from `nce.inference.bucket` or `nce.inference.factor_nn`

## Verification

- `python -c "from nce.visualization import plot_learning_curves; print('ok')"` — passes
- `python -c "from nce.visualization.learning_curves import _extract_training_log; print('ok')"` — passes
- Quick script: create a synthetic state dict with 3 entries, call `plot_learning_curves`, verify `len(fig.axes) == 3`

## Observability Impact

- Signals added/changed: `_extract_training_log` raises `TypeError` with source type name on bad input
- How a future agent inspects this: `fig.axes` list shows subplot count; `ax.lines` shows plotted series; `ax.get_title()` shows bucket labels
- Failure state exposed: TypeError message names the unrecognized type; empty-data case produces informational figure instead of crash

## Inputs

- S04 data contract: `per_bucket_training_log` entries contain `label` (int), `losses` (list of (epoch, loss_value) tuples), `val_losses` (same format, may be empty), `epochs_trained` (int), `hidden_sizes` (list or string)
- `nce/state/state.py` — `load_state()` for path-based input normalization
- D006 — visualization in separate `nce/visualization/` module

## Expected Output

- `nce/visualization/__init__.py` — re-exports `plot_learning_curves`
- `nce/visualization/learning_curves.py` — `_extract_training_log()` helper + `plot_learning_curves()` function
