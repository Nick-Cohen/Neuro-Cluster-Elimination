---
estimated_steps: 5
estimated_files: 3
---

# T02: Create compare_experiments function

**Slice:** S05 — Visualization Module
**Milestone:** M001

## Description

Implement `compare_experiments()` in `nce/visualization/comparison.py` for cross-experiment training comparison (R015). Accepts a list of sources (FastGMs, state dicts, or paths) with optional labels. Produces a Figure with two panel types: (1) a summary bar chart of final loss per bucket for each experiment, and (2) per-bucket overlaid learning curves for bucket labels that appear in multiple experiments. Uses `_extract_training_log()` from T01 for input normalization.

## Steps

1. Create `nce/visualization/comparison.py`:
   - Import `_extract_training_log` from `learning_curves.py`
   - Implement `compare_experiments(sources, labels=None, save_path=None, max_subplots=20)`:
     - Normalize each source via `_extract_training_log()`
     - If `labels` not provided, auto-generate as "Experiment 1", "Experiment 2", etc.
     - Build a bucket-label index: for each experiment, map bucket label → entry
     - Find shared bucket labels (labels present in ≥2 experiments)
     - **Summary panel** (first subplot): grouped bar chart, x-axis = bucket labels, bars = final loss per experiment, grouped by experiment label. Uses last entry in each bucket's `losses` list for the final value.
     - **Per-bucket overlays** (remaining subplots): one subplot per shared bucket label (capped at `max_subplots`), overlaid learning curves from each experiment with distinct colors/labels. Legend identifies experiments.
     - Handle edge cases: no shared buckets (summary only, note in title), single experiment (degenerate — just show summary), no losses in any entry
     - If `save_path` provided, save figure
     - Return Figure
2. Update `nce/visualization/__init__.py` to re-export `compare_experiments`
3. Verify import: `python -c "from nce.visualization import compare_experiments; print('ok')"`
4. Quick smoke test with two synthetic state dicts sharing bucket labels
5. Test edge case: two experiments with no overlapping bucket labels (should still produce summary panel)

## Must-Haves

- [ ] `compare_experiments` accepts list of FastGMs, state dicts, or file paths
- [ ] Auto-generates labels when not provided
- [ ] Summary panel shows final loss per bucket grouped by experiment
- [ ] Per-bucket overlays for shared bucket labels with legend
- [ ] Handles no shared buckets gracefully (summary only)
- [ ] `max_subplots` caps overlay subplots
- [ ] `save_path` writes to disk
- [ ] No `plt.show()` calls
- [ ] No imports from inference layer

## Verification

- `python -c "from nce.visualization import compare_experiments; print('ok')"` — passes
- Quick script: two synthetic state dicts with 3 overlapping bucket labels → Figure has 1 summary + 3 overlay subplots

## Observability Impact

- Signals added/changed: `compare_experiments` raises `ValueError` if sources list is empty
- How a future agent inspects this: `fig.axes` for subplot count; summary axis title identifies the panel; per-bucket axes titled by bucket label
- Failure state exposed: Empty sources → ValueError; no training data in any experiment → informational figure

## Inputs

- `nce/visualization/learning_curves.py` — `_extract_training_log()` helper (from T01)
- S04 data contract: training log entry schema (label, losses, val_losses, epochs_trained, hidden_sizes)

## Expected Output

- `nce/visualization/comparison.py` — `compare_experiments()` function
- `nce/visualization/__init__.py` — updated to re-export `compare_experiments`
