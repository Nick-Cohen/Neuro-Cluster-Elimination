---
estimated_steps: 4
estimated_files: 3
---

# T02: Add plot generation and metrics output

**Slice:** S02 — Single-Bucket Training Harness with Plots
**Milestone:** M004

## Description

Create `nce/benchmark/plots.py` with two plotting functions (loss curve and local error curve) and wire the output stage into `train_single_bucket()`. After training completes (or time limit is hit), the function creates an output folder `{output_dir}/{bucket_id}/`, generates loss.png and local_error.png, writes metrics.json, and includes these paths in the return dict.

Follows the matplotlib pattern established in `nce/visualization/learning_curves.py`: `matplotlib.use("Agg")` before pyplot import, `fig.savefig(path, bbox_inches="tight")`, explicit `plt.close(fig)` for memory cleanup.

## Steps

1. Create `nce/benchmark/plots.py` with:
   - `plot_loss_curve(losses, output_path, title=None)` — takes list of `(epoch, loss_value)` tuples, plots loss over epochs on a semilogy scale, saves PNG to `output_path`. Title defaults to "Training Loss". Labels axes. Returns the path written.
   - `plot_local_error_curve(error_tracking_data, output_path, title=None)` — takes list of `(epoch, loss, log_z_err, abs_log_z_err)` tuples, plots `abs_log_z_err` over epochs on a semilogy scale, saves PNG. Title defaults to "Local Error (|log Z err|)". Returns the path written.
   - Both functions: `matplotlib.use("Agg")`, create fig+ax, save with `bbox_inches="tight"`, `plt.close(fig)`.

2. Add `_write_metrics(result, nn_config, bucket_data, output_path)` helper to `training.py`:
   - Builds metrics dict: `{epochs_completed, final_loss, final_local_error, error_tracking: [...], losses: [...], wall_time, config_hash, bucket_metadata: {problem_key, bucket_label, auto_ecl, selection_error}, timestamp}`
   - `config_hash` = `hashlib.md5(json.dumps(sorted config items, default=str).encode()).hexdigest()`
   - Writes to `output_path` as JSON with indent=2
   - Handles non-serializable types (convert torch tensors to floats)

3. Wire output stage into `train_single_bucket()`:
   - After training loop completes, create `{output_dir}/{bucket_id}/` directory
   - Call `plot_loss_curve()` and `plot_local_error_curve()`
   - Call `_write_metrics()`
   - Add output paths to return dict: `output_dir`, `loss_plot_path`, `error_plot_path`, `metrics_path`
   - Handle partial completion: if training failed mid-loop, still write whatever data was collected

4. Update `nce/benchmark/__init__.py` to also export plot functions for direct use.

## Must-Haves

- [ ] `plot_loss_curve()` produces a valid PNG file from loss data
- [ ] `plot_local_error_curve()` produces a valid PNG file from error tracking data
- [ ] Both plots use semilogy scale (log y-axis for loss/error magnitudes)
- [ ] Both plots use `matplotlib.use("Agg")` — no interactive backend
- [ ] `metrics.json` contains all required keys: epochs_completed, final_loss, final_local_error, error_tracking, losses, wall_time, config_hash, bucket_metadata, timestamp
- [ ] Output folder structure: `{output_dir}/{bucket_id}/loss.png`, `local_error.png`, `metrics.json`
- [ ] `train_single_bucket()` produces output folder as part of its return flow

## Verification

- `python -c "from nce.benchmark.plots import plot_loss_curve, plot_local_error_curve; print('ok')"` succeeds
- Quick smoke test: create synthetic loss data, call plot functions, verify PNG files exist and are > 0 bytes
- `python -c "
from nce.benchmark.plots import plot_loss_curve, plot_local_error_curve
import tempfile, os
d = tempfile.mkdtemp()
plot_loss_curve([(0, 1.0), (10, 0.5), (50, 0.1)], os.path.join(d, 'loss.png'))
plot_local_error_curve([(0, 1.0, 0.5, 0.5), (10, 0.5, 0.2, 0.2)], os.path.join(d, 'err.png'))
assert os.path.getsize(os.path.join(d, 'loss.png')) > 0
assert os.path.getsize(os.path.join(d, 'err.png')) > 0
print('plots ok')
"`

## Observability Impact

- Signals added/changed: `[BenchmarkTraining]` print for "Saving outputs to {output_dir}/{bucket_id}/"
- How a future agent inspects this: Check output folder for loss.png, local_error.png, metrics.json; parse metrics.json for structured data
- Failure state exposed: If plotting fails, exception includes output_path; metrics.json written even if plots fail (best-effort output)

## Inputs

- `nce/benchmark/training.py` from T01 — `train_single_bucket()` function to wire output into
- `nce/visualization/learning_curves.py` — matplotlib pattern to follow
- S02-RESEARCH.md — output structure specification

## Expected Output

- `nce/benchmark/plots.py` — ~80 lines with two plotting functions
- `nce/benchmark/training.py` — modified to include metrics writing and plot generation in output stage
- `nce/benchmark/__init__.py` — updated exports
