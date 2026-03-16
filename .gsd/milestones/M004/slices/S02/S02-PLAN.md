# S02: Single-Bucket Training Harness with Plots

**Goal:** A `train_single_bucket()` function in `nce/benchmark/training.py` that loads a precomputed bucket from `.pt` cache, reconstructs the live FastGM/bucket, trains the NN with a wall-clock time limit, tracks local error at checkpoint epochs using preloaded exact messages, and produces per-bucket output folders with loss and local-error PNG plots plus a metrics.json file.

**Demo:** `python scripts/verify_benchmark_training.py` loads a `.pt` file (real or synthetic), trains for 30 seconds, and produces `{output_dir}/{bucket_id}/loss.png`, `local_error.png`, and `metrics.json` with correct contents.

## Must-Haves

- `train_single_bucket(bucket_pt_path, nn_config, time_limit_seconds, output_dir, device)` function that returns a result dict with `{epochs_completed, final_loss, final_local_error, error_tracking_data, wall_time}`
- Custom epoch loop (not Trainer.train()) with wall-clock time limit checked at epoch boundaries
- Uses Trainer infrastructure for setup only (SampleGenerator, DataLoader, DataPreprocessor, loss_fn) — not the training loop
- Preloaded exact_fw/exact_bw from .pt files for error tracking (no recomputation)
- Local error computed at checkpoint epochs via `FactorNN.to_exact()` pattern from train.py:507-517
- Per-bucket output folder with `loss.png`, `local_error.png`, `metrics.json`
- Loss plot shows training loss over epochs
- Local error plot shows `abs_log_z_err` over epochs
- metrics.json includes epoch count, final loss, final local error, full error tracking data, wall time, config hash, bucket metadata
- Verification script that exercises the full pipeline end-to-end

## Proof Level

- This slice proves: integration
- Real runtime required: yes — real FastGM reconstruction, real Net training, real FactorNN.to_exact()
- Human/UAT required: yes — visual inspection of PNG plots for correctness (reasonable loss curves, decreasing error)

## Verification

- `python scripts/verify_benchmark_training.py` exits 0, producing output folder with loss.png, local_error.png, metrics.json
- metrics.json parseable with expected keys present
- loss.png and local_error.png are valid PNG files (non-zero size)
- Error tracking data has entries at expected checkpoint epochs
- `epochs_completed > 0` (training actually ran)
- If real .pt files exist: runs on a real hard bucket; otherwise: generates a synthetic .pt from a small problem and runs on that

## Observability / Diagnostics

- Runtime signals: `[BenchmarkTraining]` prefixed print statements at key lifecycle points (load, reconstruct, train start, checkpoint, time limit hit, complete)
- Inspection surfaces: `metrics.json` per bucket — machine-readable record of everything that happened during training
- Failure visibility: Exceptions during reconstruction or training are caught with context (problem_key, bucket_label, phase) and re-raised. metrics.json written even on partial completion (with `error` field if applicable).
- Redaction constraints: none — no secrets in this pipeline

## Integration Closure

- Upstream surfaces consumed: S01's `data/hard_buckets/*.pt` files and `bucket_list.json` manifest; `nce/benchmark_problems/small_problems` for model lookup; `nce/neural_networks/train.py` for Trainer setup infrastructure; `nce/inference/factor_nn.py` for `FactorNN.to_exact()`
- New wiring introduced in this slice: `nce/benchmark/training.py` module with `train_single_bucket()` public API; `nce/benchmark/plots.py` for plot generation
- What remains before the milestone is truly usable end-to-end: S03 — multi-GPU CLI entry point (`scripts/bucket_benchmark.py`), JSONL history tracking, comparison charts

## Tasks

- [ ] **T01: Build train_single_bucket() core with custom training loop** `est:2h`
  - Why: The algorithmic core of the slice — loads .pt data, reconstructs live FastGM/bucket, creates Net/Trainer infrastructure, runs custom epoch loop with time-limit and error tracking. Everything else builds on this.
  - Files: `nce/benchmark/__init__.py`, `nce/benchmark/training.py`
  - Do: Create `nce/benchmark/` module. Implement `train_single_bucket(bucket_pt_path, nn_config, time_limit_seconds, output_dir, device)` with: (1) load .pt and extract schema fields, (2) look up problem by problem_key in small_problems, (3) reconstruct FastGM via eliminate_variables(up_to=bucket_var, exact=True), (4) create Net and Trainer for setup only (SampleGenerator, DataLoader, DataPreprocessor, loss_fn), (5) load training data via dataloader.load_all(), (6) precompute exact_contribution from preloaded exact_fw/exact_bw, (7) custom epoch loop with: train_epoch → record loss → checkpoint error via FactorNN.to_exact() → check time limit. Config must go through prepare_config(). Set error_tracking=False in config to prevent Trainer from computing exact messages inline. Enforce sampling_scheme='all'. Return result dict. No plotting in this task.
  - Verify: `python -c "from nce.benchmark.training import train_single_bucket; print('import ok')"` succeeds
  - Done when: Module imports cleanly, function signature matches spec, all coupling points (Trainer init, SampleGenerator, DataLoader, FactorNN.to_exact) are wired correctly

- [ ] **T02: Add plot generation and metrics output** `est:1h`
  - Why: Completes the output layer — loss.png, local_error.png, and metrics.json per bucket. Makes training results inspectable without code.
  - Files: `nce/benchmark/plots.py`, `nce/benchmark/training.py`
  - Do: Create `nce/benchmark/plots.py` with `plot_loss_curve(losses, output_path)` and `plot_local_error_curve(error_tracking_data, output_path)`. Follow matplotlib.use("Agg") pattern from visualization/learning_curves.py. Wire into `train_single_bucket()` as the final stage before return. Add metrics.json writing with config_hash (hashlib.md5 of sorted config items), bucket metadata, timing, full error tracking data. Create output_dir/{bucket_id}/ folder structure. Handle partial completion (write what we have if time limit hit).
  - Verify: Import `from nce.benchmark.plots import plot_loss_curve, plot_local_error_curve` succeeds; functions accept expected arguments
  - Done when: Plot functions produce valid PNG files from test data; metrics.json written with all expected keys

- [ ] **T03: End-to-end verification script** `est:1.5h`
  - Why: Proves the full pipeline works with real data. The objective stopping condition for the slice.
  - Files: `scripts/verify_benchmark_training.py`
  - Do: Write script that: (1) checks for real .pt files in data/hard_buckets/; if found, uses the first one; if not, generates a synthetic .pt from smokers_20 (problem 0) by running a quick exact elimination and saving in S01's schema, (2) calls train_single_bucket() with a 30-second time limit and a simple config (UKL, hidden_sizes=[3,3], lr=0.01), (3) validates output: checks loss.png, local_error.png, metrics.json exist and are non-empty, parses metrics.json and asserts expected keys, checks epochs_completed > 0 and error_tracking_data is non-empty. Run the script on GPU.
  - Verify: `python scripts/verify_benchmark_training.py` exits 0
  - Done when: Script passes end-to-end, producing valid output folder with plots and metrics

## Files Likely Touched

- `nce/benchmark/__init__.py` (new)
- `nce/benchmark/training.py` (new)
- `nce/benchmark/plots.py` (new)
- `scripts/verify_benchmark_training.py` (new)
