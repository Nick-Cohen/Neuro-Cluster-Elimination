---
id: T02
parent: S02
milestone: M004
provides:
  - nce/benchmark/plots.py with plot_loss_curve() and plot_local_error_curve()
  - _write_metrics() helper for structured JSON output
  - Output stage wired into train_single_bucket() — plots + metrics written to {output_dir}/{bucket_id}/
key_files:
  - nce/benchmark/plots.py
  - nce/benchmark/training.py
  - nce/benchmark/__init__.py
key_decisions:
  - Best-effort output — plots and metrics each wrapped in try/except so a plot failure doesn't prevent metrics from being written
  - Semilogy scale for both plots — loss magnitudes span orders of magnitude, linear scale hides convergence behavior
  - _json_default handler converts torch tensors and sets for JSON serialization rather than requiring callers to sanitize
patterns_established:
  - Plot functions are standalone (accept data + path, return path) — usable outside train_single_bucket()
  - matplotlib.use("Agg") + savefig(bbox_inches="tight") + plt.close(fig) pattern consistent with nce/visualization/
observability_surfaces:
  - "[BenchmarkTraining] Saving outputs to {path}/" print at output stage entry
  - "[BenchmarkTraining] WARNING: ..." prints if plot or metrics writing fails
  - metrics.json — machine-readable record with all training data, config hash, bucket metadata, timestamp
  - Return dict now includes output_dir, loss_plot_path, error_plot_path, metrics_path keys
duration: 15min
verification_result: passed
completed_at: 2026-03-12
blocker_discovered: false
---

# T02: Add plot generation and metrics output

**Created `nce/benchmark/plots.py` with loss and local-error plotting functions, added `_write_metrics()` to training.py, and wired the full output stage into `train_single_bucket()` — producing `{output_dir}/{bucket_id}/loss.png`, `local_error.png`, and `metrics.json` after every training run.**

## What Happened

Created `plots.py` with two functions following the existing matplotlib pattern from `nce/visualization/learning_curves.py`. Both use semilogy scale, Agg backend, and explicit figure cleanup.

Added `_write_metrics()` helper to `training.py` with a `_json_default` handler for torch tensors and sets. Metrics include all required keys: epochs_completed, final_loss, final_local_error, error_tracking, losses, wall_time, config_hash (MD5 of sorted config), bucket_metadata, and ISO timestamp.

Wired a new output stage (section 13) into `train_single_bucket()` after the result dict is built. Creates `{output_dir}/{bucket_id}/` directory, generates both plots, writes metrics.json, and adds all output paths to the return dict. Each output step is best-effort — a plot failure doesn't prevent metrics from being written.

Updated `__init__.py` to export `plot_loss_curve` and `plot_local_error_curve`.

## Verification

- `python -c "from nce.benchmark.plots import plot_loss_curve, plot_local_error_curve; print('ok')"` — passed
- Smoke test with synthetic data: both PNGs generated with > 0 bytes — passed
- `_write_metrics()` unit test: all 9 required keys present, config_hash is 32-char MD5 hex, values match input — passed
- `from nce.benchmark import train_single_bucket, plot_loss_curve, plot_local_error_curve` — passed

### Slice-level verification (partial — T02 is intermediate):
- ✅ Import checks pass
- ✅ Plot functions produce valid PNG files (non-zero size)
- ✅ metrics.json parseable with expected keys present
- ⏳ `scripts/verify_benchmark_training.py` — T03's deliverable

## Diagnostics

- Check output folder for `loss.png`, `local_error.png`, `metrics.json` after any training run
- Parse `metrics.json` for structured data: `json.load()` gives all training metadata
- If plots fail, `[BenchmarkTraining] WARNING:` messages include the exception and output_path
- Return dict keys `loss_plot_path`, `error_plot_path`, `metrics_path` are only present if the corresponding output succeeded

## Deviations

None.

## Known Issues

None.

## Files Created/Modified

- `nce/benchmark/plots.py` — new; plot_loss_curve() and plot_local_error_curve() functions
- `nce/benchmark/training.py` — added imports (hashlib, json, os, datetime), _json_default(), _write_metrics(), and output stage (section 13) in train_single_bucket()
- `nce/benchmark/__init__.py` — updated to export plot functions
