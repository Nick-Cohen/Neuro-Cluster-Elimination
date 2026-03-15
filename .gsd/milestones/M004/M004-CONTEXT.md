# M004: Single-Bucket Learning Benchmark — Context

**Gathered:** 2026-03-15
**Status:** Queued — pending auto-mode execution

## Project Description

NCE is a Python package for neural network-based approximate inference on probabilistic graphical models. This milestone builds a reusable benchmarking harness that evaluates NN learning quality on a curated set of "hard" single buckets, tracks performance history across runs, and enables systematic comparison of configs, loss functions, and hyperparameters over time.

## Why This Milestone

There is currently no systematic way to evaluate whether a config change improves NN learning quality on hard buckets. The existing `error_tracking` infrastructure in `train.py` computes local errors during a full inference run, but it's coupled to the full elimination pipeline — you can't isolate a single bucket, preload its messages, and benchmark different configs against it. Researchers need a fast feedback loop: change a config, run the benchmark, see if hard-bucket local errors improved vs prior best.

## User-Visible Outcome

### When this milestone is complete, the user can:

- Run `python scripts/bucket_benchmark.py config.yaml fast --gpus 0,1,2,3` and get 1-minute-per-bucket training across ~10 hard buckets in parallel on 4 GPUs
- Run the same command with `slow` for 1-hour-per-bucket training
- See per-bucket output folders with loss-over-epochs and local-error-over-epochs PNG plots
- See a comparison chart of this run's local errors vs the historical best for each bucket (among runs of equal or shorter duration)
- Find run metadata (config, timing, epochs completed, local errors) appended to a JSONL history file for long-term tracking
- Re-run the one-time bucket selection script to refresh the hard-bucket list if problems or ecl values change

### Entry point / environment

- Entry point: CLI script `scripts/bucket_benchmark.py <config.yaml> <fast|slow> [--gpus 0,1,2,3]`
- Environment: local dev with CUDA GPUs (4× NVIDIA TITAN RTX)
- Live dependencies involved: none (PyTorch, matplotlib — all local)

## Completion Class

- Contract complete means: benchmark script runs, produces output folder with plots and JSONL history entry, parallel GPU execution works, time limits are respected
- Integration complete means: uses real `small_problems` models, real `get_backward_message`, real `Trainer` with error tracking, real `FactorNN.to_exact()` for local error computation
- Operational complete means: none (no services)

## Final Integrated Acceptance

To call this milestone complete, we must prove:

- Running `python scripts/bucket_benchmark.py config.yaml fast --gpus 0,1,2,3` completes within ~(num_buckets / 4) × 1 minute, produces per-bucket output folders with loss and local error plots, and appends a JSONL history entry
- The comparison chart correctly shows this run's results vs the historical best from the JSONL file (filtered to runs of equal or shorter duration)
- The bucket selection precomputation script identifies hard buckets with local error > 0.1 across the 24 small_problems and saves the curated list + precomputed messages to disk

## Risks and Unknowns

- **Bucket selection cost** — Running all 24 problems × 10000 epochs to find hard buckets is expensive (potentially hours of GPU time). Must be a one-time precomputation step with cached results.
- **Precomputed backward message storage** — Backward messages at various ecl levels (2^2 through 2^25) could be large tensors. Need to verify disk space is manageable for 24 problems × multiple bw_ecl levels.
- **Time-limited training accuracy** — Finishing at epoch boundaries means actual training time may overshoot the target (fast=1min, slow=1h) by one epoch. For slow mode with large messages, a single epoch could take significant time.
- **Exact backward solvability** — The spec requires that selected test buckets have exactly solvable backward messages. At least one problem may fail this check. Need a validation step that pings Discord if a selected bucket doesn't have exact bw.
- **Error tracking coupling** — The existing `error_tracking` code in `train.py` computes exact bw messages inline during training. The benchmark needs to preload these instead for efficiency. May need to refactor or extend the error tracking path.

## Existing Codebase / Prior Art

- `nce/neural_networks/train.py` lines 339–395 — Existing `error_tracking` setup that computes exact forward and backward messages and tracks `(epoch, loss, log_Z_err, abs_log_Z_err)` at checkpoint epochs. This is the core local error computation pattern the benchmark will reuse.
- `nce/neural_networks/train.py:16` — `get_error_tracking_epochs()` generates checkpoint schedule: [0, 1, 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000, then every 5000].
- `nce/utils/backward_message.py` — `get_backward_message()` computes backward messages with configurable `iB` and `backward_ecl`. Used by error tracking.
- `nce/inference/graphical_model.py:438–461` — Track errors computes `nn_contribution - exact_contribution` using `(nn_factor * exact_bw).sum_all_entries()`.
- `nce/benchmark_problems/small_problems.py` — 24 problems with `auto_ecl` values from `problem_ecl_values.csv`. These are the candidate problems for bucket selection.
- `notebooks/_1-2026/problem_ecl_values.csv` — Pre-computed auto_ecl values per problem (the "largest ecl without leading to zero NN training" values).
- `nce/inference/factor_nn.py` — `FactorNN.to_exact()` converts NN approximation to exact factor for local error computation.
- `nce/inference/bucket.py` — `FastBucket.compute_message_exact()` computes the exact forward message for a bucket.

> See `.gsd/DECISIONS.md` for all architectural and pattern decisions — it is an append-only register; read it during planning, append to it during execution.

## Relevant Requirements

This milestone introduces new requirements:

- R039 — Hard bucket selection: one-time precomputation identifying buckets with local error > 0.1 across 24 small_problems (up to 10 buckets)
- R040 — Precomputed message caching: exact forward messages, exact backward messages, and approximate backward messages at specified bw_ecl levels saved to disk per selected bucket
- R041 — Time-limited single-bucket training: train with epoch-boundary timeout (fast=1min, slow=1h), save NN weights, compute local error at checkpoints
- R042 — Multi-GPU parallel execution: one bucket per GPU, cycling through bucket list as GPUs become available
- R043 — Per-bucket output: loss-over-epochs and local-error-over-epochs PNG plots in timestamped output folder
- R044 — Historical comparison: JSONL history file with per-run metadata; comparison chart showing current vs best-ever local errors for runs of equal or shorter duration
- R045 — CLI entry point: `python scripts/bucket_benchmark.py <config.yaml> <fast|slow> [--gpus 0,1,2,3]`

## Scope

### In Scope

- One-time bucket selection script across 24 small_problems (UKL + bw + auto_ecl + 10000 epochs, full-batch training)
- Curated hard-bucket list (up to 10 buckets with local error > 0.1) saved to disk
- Precomputed and cached forward messages, backward messages (exact + approximate at specified bw_ecl levels)
- Validation that selected buckets have exactly solvable backward messages (Discord ping if not)
- Time-limited training with epoch-boundary stops
- Checkpoint-based local error and loss tracking at spaced intervals (epoch 10, 20, 50, 100, 500, 1000, 2000, ...) or time-based (every 1min for slow epochs)
- Per-bucket output folder with matplotlib PNG plots (loss over epochs, local error over epochs)
- JSONL history file recording config, timing, epochs completed, local errors per bucket per run
- Comparison chart: this run's local errors vs historical best (filtered by duration ≤ current run's duration)
- Multi-GPU parallel execution (1 bucket per GPU, configurable GPU list)
- YAML config input processed through `prepare_config()`
- Fast mode (1 min/bucket) and slow mode (1 hour/bucket)

### Out of Scope / Non-Goals

- Modifying the core inference algorithm or training loop logic
- Interactive or real-time visualization
- Multi-bucket training (training multiple buckets on the same GPU simultaneously)
- Automatic config search or hyperparameter optimization
- CI/CD integration

## Technical Constraints

- Python 3.11, PyTorch 2.0.1+cu117
- 4× NVIDIA TITAN RTX (24GB each) available
- Must use existing `small_problems` benchmark set and `auto_ecl` values
- Must use existing `error_tracking` computation pattern (exact_fw * exact_bw vs approx_fw * exact_bw)
- YAML config must be compatible with `prepare_config()` from `nce/config_schema.py`
- Bucket selection uses UKL loss function with backward information enabled

## Integration Points

- `nce/benchmark_problems/small_problems.py` — Source of 24 candidate problems with auto_ecl values
- `nce/neural_networks/train.py` — Trainer class for single-bucket training; error_tracking infrastructure for local error computation
- `nce/utils/backward_message.py` — `get_backward_message()` for computing backward messages at various ecl levels
- `nce/inference/graphical_model.py` — FastGM for running full inference during bucket selection phase
- `nce/config_schema.py` — `prepare_config()` for validating YAML config input
- `nce/inference/factor_nn.py` — `FactorNN.to_exact()` for converting trained NN to exact factor for local error computation

## Open Questions

- **Which problem has non-exact backward messages?** — User mentioned knowing of one problem where exact backward solvability may not hold. Need to identify it during bucket selection and handle gracefully.
- **Checkpoint schedule for slow epochs** — If a single epoch takes >1 minute (possible for large messages in slow mode), should we fall back to time-based checkpoints (every 1min) instead of epoch-based? Current thinking: yes, use time-based fallback.
- **bw_ecl for training vs evaluation** — The config specifies one bw_ecl level for training. The precomputed messages at multiple ecl levels (2^2 through 2^25) are available for future use or evaluation. Current thinking: train with one bw_ecl from config; cache multiple levels for potential future comparison runs.
- **Exact backward message at bw_ecl=2^30** — The existing error_tracking code uses `backward_ecl=2**30` to get "exact" backward messages. Should the benchmark use the same value, or should exact backward be computed without mini-bucket approximation? Current thinking: use `backward_ecl=2**30` (same as existing code) which is effectively exact for these problem sizes.
