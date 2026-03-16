---
estimated_steps: 5
estimated_files: 2
---

# T01: Build train_single_bucket() core with custom training loop

**Slice:** S02 — Single-Bucket Training Harness with Plots
**Milestone:** M004

## Description

Create the `nce/benchmark/` module with `training.py` containing the `train_single_bucket()` function. This is the algorithmic core of the benchmark harness: it loads a precomputed `.pt` file, reconstructs the live FastGM and bucket via `eliminate_variables(up_to=...)`, creates training infrastructure (Net, Trainer for setup only), and runs a custom epoch loop with wall-clock time-limit checking and local error computation at checkpoint epochs.

The custom loop replaces Trainer.train() — Trainer is only used for its __init__ chain (which creates SampleGenerator, DataPreprocessor, DataLoader) and `_get_loss_fn()`. The epoch loop mirrors train.py's pattern but is focused: iterate epochs, call `trainer.train_epoch(batches)`, record loss, compute error at checkpoints via `FactorNN.to_exact()`, check wall-clock time. No early stopping, no validation, no display_intermediate, no traced losses.

## Steps

1. Create `nce/benchmark/__init__.py` with public API export of `train_single_bucket`.

2. Create `nce/benchmark/training.py` with helper functions:
   - `_load_bucket_data(pt_path, device)` — loads .pt file, reconstructs FastFactor objects for exact_fw and exact_bw (moving tensors to device), returns structured dict
   - `_find_problem(problem_key)` — looks up the model and config index in `small_problems` by matching `model.modelfile == problem_key`
   - `_reconstruct_bucket(problem_idx, bucket_label, nn_config, device)` — creates FastGM, calls `eliminate_variables(up_to=target_var, exact=True)`, returns `(fastgm, bucket)`

3. Implement the main `train_single_bucket(bucket_pt_path, nn_config, time_limit_seconds, output_dir, device)` function:
   - Load .pt data via `_load_bucket_data()`
   - Find problem via `_find_problem()`
   - Merge nn_config with per-bucket metadata (ecl from .pt, iB from .pt or config)
   - Pass config through `prepare_config()` with `error_tracking=False`, `sampling_scheme='all'`
   - Reconstruct FastGM and bucket via `_reconstruct_bucket()`
   - Create `Net(bucket, hidden_sizes=config['hidden_sizes'])`
   - Create `Trainer(net, bucket)` for setup (gets SampleGenerator, DataLoader, DataPreprocessor, loss_fn)
   - Load all training data via `trainer.dataloader.load_all()`
   - Create batches from all_data (same pattern as train.py:364-375)
   - Precompute `exact_contribution = (exact_fw * exact_bw).sum_all_entries()` — once, before loop
   - Get checkpoint epochs via `get_error_tracking_epochs(config['num_epochs'])`
   - Custom epoch loop with time check, loss recording, checkpoint error tracking
   - Return result dict: `{epochs_completed, final_loss, final_local_error, error_tracking_data, losses, wall_time, bucket_id, config_used}`

4. Implement the checkpoint error computation (following train.py:507-517 pattern):
   - `approx_factor = FactorNN(net, trainer.data_preprocessor)`
   - `approx_exact = approx_factor.to_exact()`
   - `approx_contribution = (approx_exact * exact_bw).sum_all_entries()`
   - `log_z_err = approx_contribution - exact_contribution`
   - Record `(epoch, loss, log_z_err, abs(log_z_err))`

5. Add `[BenchmarkTraining]` prefixed print statements at lifecycle points: load complete, reconstruction complete, training start (with epoch count and time limit), each checkpoint, time limit hit, training complete.

## Must-Haves

- [ ] `nce/benchmark/__init__.py` exists with `from .training import train_single_bucket`
- [ ] `train_single_bucket()` has the correct signature: `(bucket_pt_path, nn_config, time_limit_seconds, output_dir, device)`
- [ ] .pt loading handles S01 schema: factors, exact_fw, exact_bw (each with tensor + labels), bucket_label, scope, domain_sizes, elim_vars, problem_key, auto_ecl
- [ ] Config goes through `prepare_config()` before use
- [ ] `error_tracking` forced to `False` in config (benchmark handles it externally)
- [ ] `sampling_scheme` forced to `'all'`
- [ ] Custom epoch loop checks `time.time() - start > time_limit_seconds` at epoch boundaries
- [ ] Checkpoint error uses FactorNN.to_exact() pattern, not recomputing exact messages
- [ ] `exact_contribution` computed once before loop, not per checkpoint
- [ ] Returns result dict with all specified keys

## Verification

- `python -c "from nce.benchmark.training import train_single_bucket; print('import ok')"` succeeds
- `python -c "from nce.benchmark import train_single_bucket; print('public API ok')"` succeeds
- Code review: verify the epoch loop pattern matches train.py:497-517 for loss aggregation and error tracking

## Observability Impact

- Signals added/changed: `[BenchmarkTraining]` print statements at load, reconstruct, train start, checkpoint, time limit, complete
- How a future agent inspects this: Read stdout for structured progress; return value dict contains all training metadata
- Failure state exposed: Exceptions include problem_key and bucket_label context; partial results returned if time limit hit mid-training

## Inputs

- S02-RESEARCH.md — approach design, coupling chain, pitfalls
- S01-SUMMARY.md — .pt file schema, bucket_list.json format
- `scripts/select_hard_buckets.py:199-310` — Phase 2 precomputation pattern for reconstruction
- `nce/neural_networks/train.py:339-395, 497-517` — error tracking setup and checkpoint computation
- `nce/neural_networks/train.py:942-968` — train_epoch method
- `nce/neural_networks/train.py:968-1001` — _make_dataloader pattern

## Expected Output

- `nce/benchmark/__init__.py` — module init with public export
- `nce/benchmark/training.py` — ~250 lines containing `train_single_bucket()` and helper functions
