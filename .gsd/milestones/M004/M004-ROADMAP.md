# M004: Single-Bucket Learning Benchmark

**Vision:** A reusable benchmark harness that evaluates NN learning quality on curated "hard" buckets, with time-controlled training, historical tracking, and multi-GPU parallel execution — all invoked via a single CLI command.

## Success Criteria

- Running `python scripts/select_hard_buckets.py` identifies hard buckets (local error > 0.1 after 10000 epochs) across 24 small_problems and saves precomputed messages + metadata to disk as `.pt` files
- Running `python scripts/bucket_benchmark.py config.yaml fast --gpus 0,1,2,3` completes within ~(num_buckets / 4) × 1 minute, producing per-bucket output folders with loss and local error PNG plots
- Each benchmark run appends a structured entry to a JSONL history file with config hash, timing, epochs completed, and per-bucket local errors
- A comparison chart in the output folder shows this run's local errors vs the historical best (filtered to runs of equal or shorter duration)
- Multi-GPU execution distributes one bucket per GPU, cycling through the bucket list as GPUs become free

## Key Risks / Unknowns

- **Trainer ↔ FastGM coupling** — Trainer/SampleGenerator deeply reference `bucket.gm`. Isolated single-bucket training requires either a minimal FastGM stub or running `eliminate_variables(up_to=...)` to reconstruct bucket state. If reconstruction is too fragile, the entire benchmark approach is in question.
- **Bucket selection cost** — Running 24 problems × 10000 epochs with error tracking is hours of GPU time. If we can't parallelize and cache this, the selection step becomes a blocker.
- **Hard bucket availability** — If fewer than 3 problems produce buckets with local error > 0.1, the benchmark set is too small to be useful. The threshold may need tuning.
- **Backward message exactness** — `get_backward_message()` with `backward_ecl=2**30` should be effectively exact for these problem sizes, but at least one problem may not have exactly solvable backward messages.

## Proof Strategy

- **Trainer ↔ FastGM coupling** → retire in S01 by building the selection script that runs full inference, extracts bucket state, and trains a precomputed bucket through the real Trainer path. If Trainer can't operate on a reconstructed bucket, we discover it here.
- **Bucket selection cost** → retire in S01 by running the selection across 4 GPUs with cached results. The cost is paid once; subsequent slices use cached `.pt` files.
- **Hard bucket availability** → retire in S01 by examining selection results. If < 3 hard buckets found, we lower the threshold (configurable parameter, default 0.1).
- **Backward message exactness** → retire in S01 by validating exact backward solvability per bucket during selection; Discord ping if a bucket fails.

## Verification Classes

- Contract verification: pytest for any new library-level utilities; script exit codes and output file checks for CLI tools
- Integration verification: real `small_problems` models through real `FastGM.eliminate_variables()`, real `Trainer.train()`, real `get_backward_message()` — no mocks
- Operational verification: multi-GPU subprocess isolation via `CUDA_VISIBLE_DEVICES`, JSONL file integrity under concurrent writes
- UAT / human verification: visual inspection of per-bucket PNG plots and comparison chart

## Milestone Definition of Done

This milestone is complete only when all are true:

- Hard bucket selection script has been run and produced cached `.pt` files with valid factor tensors, exact forward/backward messages, and bucket metadata
- Single-bucket training with preloaded data produces correct loss curves and local error tracking that match the existing `error_tracking` pattern's output format
- `python scripts/bucket_benchmark.py config.yaml fast --gpus 0,1,2,3` runs end-to-end, produces per-bucket output folders with PNG plots, and appends a JSONL history entry
- Comparison chart correctly shows current run vs historical best (filtered by duration ≤ current)
- All success criteria re-checked against live behavior on the 4× TITAN RTX machine

## Requirement Coverage

- Covers: R039, R040, R041, R042, R043, R044, R045
- Partially covers: none
- Leaves for later: none
- Orphan risks: none — all 7 active M004 requirements are mapped

## Slices

- [x] **S01: Hard Bucket Selection & Precomputation** `risk:high` `depends:[]`
  > After this: `python scripts/select_hard_buckets.py` runs all 24 small_problems across 4 GPUs, identifies hard buckets (local error > 0.1), and saves per-bucket `.pt` files containing factor tensors, exact forward/backward messages, and metadata to `data/hard_buckets/`. The cached bucket list and precomputed messages are on disk, ready for benchmark training.
- [x] **S02: Single-Bucket Training Harness with Plots** `risk:medium` `depends:[S01]`
  > After this: A training harness loads a precomputed bucket from `.pt` cache, trains its NN with a time limit (epoch-boundary timeout), tracks loss and local error at checkpoint epochs, and produces per-bucket output folders with loss-over-epochs and local-error-over-epochs PNG plots. Runnable as `python scripts/bucket_benchmark.py config.yaml fast` on a single GPU.
- [ ] **S03: Multi-GPU CLI, History Tracking & Comparison** `risk:low` `depends:[S01, S02]`
  > After this: `python scripts/bucket_benchmark.py config.yaml fast --gpus 0,1,2,3` distributes training across 4 GPUs (1 bucket per GPU), appends run metadata to a JSONL history file, and produces a comparison chart of this run's local errors vs the historical best for runs of equal or shorter duration. Full user-visible milestone outcome delivered.

## Boundary Map

### S01 → S02

Produces:
- `data/hard_buckets/` directory with per-bucket `.pt` files, each containing: `{'factors': List[FastFactor], 'exact_fw': FastFactor, 'exact_bw': FastFactor, 'bucket_label': str, 'scope': list, 'domain_sizes': list, 'elim_vars': list, 'problem_key': str, 'auto_ecl': int, 'approx_bw': dict[int, FastFactor]}` (approx_bw keyed by bw_ecl level)
- `data/hard_buckets/bucket_list.json` manifest with bucket IDs, problem keys, local errors from selection run, and metadata
- A proven pattern for reconstructing bucket state from cached `.pt` files through `eliminate_variables(up_to=...)`

Consumes:
- nothing (first slice)

### S02 → S03

Produces:
- `nce/benchmark/` module with `train_single_bucket(bucket_data, config, time_limit, output_dir)` function that returns `{epochs_completed, final_loss, local_errors_at_checkpoints, wall_time}`
- Per-bucket output folder structure: `{output_dir}/{bucket_id}/loss.png`, `local_error.png`, `metrics.json`
- Checkpoint epoch schedule function (reusing/extending `get_error_tracking_epochs()`)

Consumes:
- S01's `data/hard_buckets/*.pt` files and `bucket_list.json`

### S03 (final assembly)

Produces:
- `scripts/bucket_benchmark.py` CLI entry point with `<config.yaml> <fast|slow> [--gpus 0,1,2,3]` interface
- Multi-GPU subprocess orchestration (1 bucket per GPU, cycling)
- `data/hard_buckets/history.jsonl` with per-run records: `{timestamp, config_hash, config, mode, time_limit_per_bucket, buckets: [{bucket_id, epochs_completed, wall_time, local_errors_at_checkpoints, final_local_error}]}`
- Comparison chart PNG: current run's final local errors vs historical best per bucket (filtered by duration ≤ current run's duration)

Consumes:
- S01's cached bucket data
- S02's `train_single_bucket()` function and output folder structure
