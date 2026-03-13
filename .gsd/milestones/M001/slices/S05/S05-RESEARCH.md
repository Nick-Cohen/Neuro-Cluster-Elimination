# S05: Visualization Module — Research

**Date:** 2026-03-12

## Summary

S05 creates a new `nce/visualization/` module (D006) with two main functions: `plot_learning_curves(fastgm_or_state)` for per-NN loss subplots, and `compare_experiments(states)` for side-by-side cross-experiment comparison. The data contract from S04 is clean and well-defined — each `per_bucket_training_log` entry contains `losses` and `val_losses` as lists of `(epoch, loss_value)` tuples, plus `label`, `epochs_trained`, and `hidden_sizes`. The functions must accept both live FastGM objects (via `per_bucket_training_log` attribute) and loaded state dicts (from `load_state()`). matplotlib 3.10.1 is available with `agg` backend (headless), so all functions should return figures and optionally save to file.

The existing `nce/utils/plots.py` has inline loss-curve plotting embedded in `plot_fastfactor_comparison()` and `plot_validation_comparison()`, but those are tightly coupled to live factor/tensor data during inference — not reusable for post-hoc inspection. The new module is a clean separation (D006).

## Recommendation

Build `nce/visualization/` with three files:
- `__init__.py` — re-exports public API
- `learning_curves.py` — `plot_learning_curves()` function
- `comparison.py` — `compare_experiments()` function

Design both functions to accept a flexible input: either a live FastGM, a loaded state dict, or a file path (str/Path). A small helper `_extract_training_log(source)` normalizes these inputs to a list of training log entry dicts.

`plot_learning_curves()` creates a grid of subplots, one per NN bucket, showing train loss and (if present) validation loss over epochs. Return the matplotlib Figure for programmatic use. Optional `save_path` kwarg writes to disk.

`compare_experiments()` takes a list of sources (FastGMs, state dicts, or paths) with optional labels, and produces overlay plots: (1) a summary panel comparing final loss across all buckets for each experiment, and (2) per-bucket overlaid learning curves when bucket labels match across experiments. This supports the R015 use case of comparing configs/loss functions on the same problem.

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| Figure layout for variable # of subplots | `matplotlib.pyplot.subplots()` with computed rows/cols | Standard, handles arbitrary grid sizes |
| Loss curve data extraction | `nce.state.load_state()` | Already handles deserialization; don't duplicate |
| Backend-agnostic saving | `fig.savefig()` | Works with any matplotlib backend, supports png/pdf/svg |

## Existing Code and Patterns

- `nce/state/state.py` — `save_state()` produces dicts with `per_bucket_training_log` key; `load_state()` returns them. This is the primary data source for visualization from saved state.
- `nce/utils/plots.py` — Has inline loss-curve plotting logic (extract epoch/loss from `(epoch, value)` tuples, plot with `plt.plot()`). Pattern to follow for consistency, but not to extend — D006 says separate module.
- `nce/inference/bucket.py:328-339` — The capture site: losses are `t.losses` (list of `(epoch, loss_value)` tuples), val_losses are `t.val_losses` (same format, may be empty).
- `nce/inference/graphical_model.py:57` — Schema comment documenting the per_bucket_training_log structure.
- `scripts/verify_s04_state_preservation.py` — Verification pattern: accumulate named PASS/FAIL checks, print summary. Follow this for S05 verification.

## Constraints

- **matplotlib 3.10.1 with `agg` backend** — no interactive display. Functions must return `Figure` objects and support `save_path` for file output. Don't call `plt.show()` from the module itself (caller's choice).
- **Loss data format is `(epoch, loss_value)` tuples** — epochs may not start at 0, may not be contiguous (if using validation intervals). Code must handle this gracefully.
- **`val_losses` may be empty** — many configs don't produce validation losses. The plotting code must handle entries with empty `val_losses` lists without error.
- **Variable number of NN buckets** — from 0 (all exact) to 100+. Subplot grid must scale: small problems get one row, large problems get multi-page or scrollable figures.
- **Bucket labels are ints** — `entry['label']` is the elimination variable label (int). Use it for subplot titles.
- **`hidden_sizes` can be a list or the string `'nbe,N'`** — the nbe convention uses strings like `'nbe,3'`. Display code must handle both.
- **State dicts from `load_state()` do not contain the model name** — config dict has problem metadata only if the user added it. Labels for compare_experiments should be provided by the caller.
- **No dependency on running inference** — visualization must work purely from saved state or a post-inference FastGM. Never import bucket.py or factor_nn.py.

## Common Pitfalls

- **Calling `plt.show()` in library code** — breaks non-interactive use. Return the Figure; let the caller decide whether to show or save.
- **Fixed subplot grid size** — hardcoding e.g. 4×4 breaks for problems with 2 buckets or 50 buckets. Compute grid dimensions from the number of entries.
- **Not closing figures** — matplotlib leaks memory on unclosed figures. Document that callers should `plt.close(fig)` after use, or provide a context manager pattern.
- **Assuming losses are non-empty** — a bucket might have `epochs_trained: 0` and `losses: []` (e.g., if training was skipped). Guard against plotting empty data.
- **Tight coupling to FastGM internals** — the module should work from the dict schema, not reach into FastGM private attributes. Accept `per_bucket_training_log` as input, not the full object.

## Open Risks

- **Large number of buckets may produce unreadable figures** — 100+ subplots on one figure is impractical. May need pagination or filtering by bucket label. Mitigate with a `max_subplots` parameter (default ~20) and optional bucket selection.
- **Cross-experiment bucket matching** — `compare_experiments` needs to align buckets across experiments. If the same model was run with different elimination orders, bucket labels won't match. Document this limitation; match by label only.
- **No existing test infrastructure for visual output** — verification must check that files are created and non-empty, and that the right number of subplots exist. Can't pixel-diff outputs without reference images.

## Skills Discovered

| Technology | Skill | Status |
|------------|-------|--------|
| matplotlib | `davila7/claude-code-templates@matplotlib` (310 installs) | available — low value for this scope |
| matplotlib | `mindrally/skills@matplotlib-best-practices` (104 installs) | available — low value for this scope |

Neither skill is needed — the plotting is standard matplotlib subplots with well-understood patterns.

## Sources

- S04 task summaries (T01, T02, T03) — data contract, loss format, state dict structure
- `nce/state/state.py` — save/load API and state dict schema
- `nce/utils/plots.py` — existing loss curve plotting pattern (inline, coupled to live factors)
- `nce/inference/bucket.py:328-339` — per_bucket_training_log capture site
- D006 decision — visualization in separate `nce/visualization/` module
