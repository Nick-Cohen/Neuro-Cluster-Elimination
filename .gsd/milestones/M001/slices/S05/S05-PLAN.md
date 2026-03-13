# S05: Visualization Module

**Goal:** Provide two standalone plotting functions that work on live FastGM objects, loaded state dicts, or file paths — producing per-NN learning curve subplots and cross-experiment comparison plots.
**Demo:** Call `plot_learning_curves(fastgm)` on a post-inference FastGM and get a multi-subplot figure with one subplot per NN bucket showing loss over epochs. Call `compare_experiments([state1, state2], labels=["run1", "run2"])` and get overlay plots comparing training across experiments.

## Must-Haves

- `nce/visualization/` module exists with `__init__.py`, `learning_curves.py`, `comparison.py`
- `plot_learning_curves()` accepts FastGM, state dict, or file path; returns matplotlib Figure
- One subplot per NN-trained bucket showing train loss (and val loss if present) over epochs
- Handles edge cases: empty losses, zero NN buckets, large bucket counts (max_subplots cap)
- `compare_experiments()` accepts list of sources with optional labels; returns Figure
- Summary panel comparing final loss across buckets for each experiment
- Per-bucket overlaid learning curves when bucket labels match across experiments
- Optional `save_path` kwarg on both functions for file output
- No `plt.show()` calls in module code — return Figure, caller decides
- No imports from `nce.inference.bucket` or `nce.inference.factor_nn` — works purely from dict schema

## Proof Level

- This slice proves: contract (functions accept documented inputs, produce correct outputs)
- Real runtime required: yes (verification runs actual inference to generate real data)
- Human/UAT required: yes (visual inspection of generated plots for readability — but functional correctness is script-verified)

## Verification

- `python scripts/verify_s05_visualization.py` — verification script that:
  1. Runs inference on rbm_20 to get a live FastGM with loss curves
  2. Saves state via `save_state()`, loads it back
  3. Calls `plot_learning_curves()` with live FastGM, loaded state dict, and file path — verifies each returns a Figure with correct subplot count
  4. Calls `compare_experiments()` with two sources — verifies returns a Figure
  5. Tests `save_path` kwarg produces a file on disk
  6. Tests edge cases: empty training log, entries with empty losses
  7. All checks PASS/FAIL with summary, exit 0/1
- `python -c "from nce.visualization import plot_learning_curves, compare_experiments; print('ok')"` — module importable

## Observability / Diagnostics

- Runtime signals: Functions raise `ValueError` with descriptive messages on bad input (unrecognized source type, empty data)
- Inspection surfaces: Returned `Figure` objects can be inspected via `fig.axes` (subplot count), `ax.lines` (plotted series), `ax.get_title()` (bucket labels)
- Failure visibility: `_extract_training_log()` raises `TypeError` naming the unrecognized source type; plotting functions propagate matplotlib errors with context
- Redaction constraints: none

## Integration Closure

- Upstream surfaces consumed: `nce.state.load_state()` for loading state dicts from file paths; `FastGM.per_bucket_training_log` attribute for live objects; state dict schema from S04 (keys: `per_bucket_training_log` containing entries with `label`, `losses`, `val_losses`, `epochs_trained`, `hidden_sizes`)
- New wiring introduced in this slice: `nce/visualization/` module with public API re-exported from `__init__.py`
- What remains before the milestone is truly usable end-to-end: S06 (logging), S07 (regression verification)

## Tasks

- [x] **T01: Create visualization module with plot_learning_curves** `est:45m`
  - Why: Core function for R013 and R014 — per-NN learning curve subplots from any source
  - Files: `nce/visualization/__init__.py`, `nce/visualization/learning_curves.py`
  - Do: Create `_extract_training_log(source)` helper that normalizes FastGM/dict/path inputs. Implement `plot_learning_curves(source, save_path=None, max_subplots=20, bucket_labels=None)` — computes grid layout, creates one subplot per NN entry, plots train loss and optional val loss, handles empty losses and zero-entry edge cases. No `plt.show()`. Return Figure.
  - Verify: `python -c "from nce.visualization import plot_learning_curves; print('ok')"`
  - Done when: Function importable, accepts state dict with known entries, returns Figure with correct subplot count

- [x] **T02: Create compare_experiments function** `est:45m`
  - Why: Cross-experiment comparison for R015 — side-by-side training comparison across configs
  - Files: `nce/visualization/comparison.py`, `nce/visualization/__init__.py`
  - Do: Implement `compare_experiments(sources, labels=None, save_path=None, max_subplots=20)` — normalize all sources via `_extract_training_log`, build bucket-label index across experiments, produce (1) summary panel of final loss per bucket per experiment, (2) per-bucket overlaid learning curves for matching bucket labels. Handle mismatched bucket sets gracefully. Return Figure.
  - Verify: `python -c "from nce.visualization import compare_experiments; print('ok')"`
  - Done when: Function importable, accepts list of two state dicts, returns Figure with summary + per-bucket overlays

- [ ] **T03: End-to-end verification script** `est:30m`
  - Why: Proves R013, R014, R015 work on real inference data — the objective stopping condition for S05
  - Files: `scripts/verify_s05_visualization.py`
  - Do: Follow S04's verification pattern (check/summarize_and_exit). Run inference on rbm_20 (3 epochs, ecl=2^19, same config as S04 verification). Exercise both functions with FastGM, state dict, and file path inputs. Verify subplot counts, save_path output, edge cases (empty log, empty losses). Clean up temp files.
  - Verify: `python scripts/verify_s05_visualization.py` — all checks pass, exit 0
  - Done when: All verification checks pass, script exits 0

## Files Likely Touched

- `nce/visualization/__init__.py`
- `nce/visualization/learning_curves.py`
- `nce/visualization/comparison.py`
- `scripts/verify_s05_visualization.py`
