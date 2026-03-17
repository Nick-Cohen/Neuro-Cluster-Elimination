# S03: Multi-GPU CLI, History Tracking & Comparison — Research

**Date:** 2026-03-16

## Summary

S03 is integration work — wiring S01's precomputed bucket data and S02's single-bucket training harness into a multi-GPU CLI with JSONL history and comparison charts. The building blocks exist and are battle-tested. Two prerequisite issues must be resolved first: (1) the `nce/benchmark/` module from S02 exists on `main` but was deleted on the S03 branch during a divergent commit, and (2) S01's Phase 2 never executed — `data/hard_buckets/` is empty with no `.pt` files or manifest.

The multi-GPU pattern is proven in `scripts/select_hard_buckets.py` (S01 coordinator) — worker pool with max 1 per GPU (D046), subprocess isolation via `CUDA_VISIBLE_DEVICES`, poll-based cycling. The benchmark CLI reuses this verbatim, replacing the S01 worker (full inference run) with a benchmark worker (calls `train_single_bucket()` for one bucket). The JSONL history file uses per-worker temp files merged by the coordinator (D042). The comparison chart is a grouped bar chart of per-bucket final local errors: current run vs historical best (filtered by duration ≤ current).

Phase 1 results exist in `/tmp/hard_bucket_selection_hvmlbbvv/` — 21/24 problems succeeded (3 OOM failures on the largest problems), 4 hard buckets identified at threshold 0.1. The data can be assembled into `selection_results.json` and Phase 2 can be run via `select_hard_buckets.py --skip-phase1`. This is a one-time ~5-minute GPU task per hard bucket (exact upstream elimination + backward message computation). After that, the 4 `.pt` files and `bucket_list.json` manifest are on disk permanently.

## Requirements Targeted

| Req | Description | What S03 must deliver |
|-----|-------------|----------------------|
| R042 | Multi-GPU parallel execution | Worker pool distributing 1 bucket per GPU, cycling as GPUs free up |
| R044 | Historical comparison tracking | JSONL history file with per-run metadata; comparison chart vs historical best (filtered by duration ≤ current) |
| R045 | CLI benchmark entry point | `python scripts/bucket_benchmark.py config.yaml fast [--gpus 0,1,2,3]` |
| R043 | Per-bucket benchmark output (integration) | Multi-GPU flow produces same per-bucket output folders (loss.png, local_error.png, metrics.json) as S02's single-GPU path |

## Recommendation

Four tasks, ordered by dependency:

**T01: Restore S02 code + generate S01 .pt files.** Cherry-pick `nce/benchmark/{__init__.py, training.py, plots.py}` and `scripts/verify_benchmark_training.py` from `main`. Then assemble Phase 1 results from `/tmp/hard_bucket_selection_hvmlbbvv/` into `data/hard_buckets/selection_results.json` and run `python scripts/select_hard_buckets.py --skip-phase1` to execute Phase 2. Verify with `python scripts/verify_hard_buckets.py`. This is a prerequisite for everything else.

**T02: Benchmark worker script + multi-GPU coordinator.** Build `scripts/bucket_benchmark_worker.py` (subprocess entry point — loads config, calls `train_single_bucket()` for one bucket, writes result JSON to temp file). Build `scripts/bucket_benchmark.py` CLI with `<config.yaml> <fast|slow> [--gpus 0,1,2,3]` interface. Coordinator: loads manifest, creates timestamped output dir, spawns workers (1 per GPU, cycling), collects results.

**T03: JSONL history tracking + comparison chart.** Add `plot_comparison_chart()` to `nce/benchmark/plots.py`. After all workers complete, coordinator: (1) merges per-worker temp results, (2) appends structured JSONL entry to `data/hard_buckets/history.jsonl`, (3) generates comparison chart PNG (current vs historical best, filtered by duration ≤ current). Handle first-run case (no history to compare against).

**T04: End-to-end verification on 4 GPUs.** Run `python scripts/bucket_benchmark.py config.yaml fast --gpus 0,1,2,3` with all 4 hard buckets. Verify: per-bucket output folders exist with loss.png + local_error.png + metrics.json, JSONL history entry was appended, comparison chart was generated (or skipped gracefully on first run). Run a second time to verify comparison chart shows current vs previous.

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| Single-bucket training | `nce.benchmark.training.train_single_bucket()` (S02, on main) | Full lifecycle: .pt load → reconstruct → train → checkpoint errors → plots → metrics |
| Multi-GPU subprocess isolation | `scripts/select_hard_buckets.py` worker pool (S01) | Proven on this machine — poll-based, 1-per-GPU, CUDA_VISIBLE_DEVICES, queue cycling |
| Loss/error plots | `nce.benchmark.plots.{plot_loss_curve, plot_local_error_curve}` (S02) | Correct PNGs with Agg backend, semilogy scale |
| Config validation | `nce.config_schema.prepare_config()` | Handles flat/nested detection, alias resolution, defaults |
| Checkpoint schedule | `nce.neural_networks.train.get_error_tracking_epochs()` | Tested schedule: 0, 1, 5, 10, 25, 50, 100, ... |
| Phase 2 precomputation | `scripts/select_hard_buckets.py:run_phase2()` | Exact upstream elimination + exact_fw + exact_bw + torch.save to .pt |

## Existing Code and Patterns

- `scripts/select_hard_buckets.py:run_phase1()` lines 57–120 — **Worker pool pattern.** `active = {}` keyed by GPU ID, poll with `proc.poll()`, spawn next from `gpu_queues[g]` when worker finishes, sleep 5s between polls. S03's coordinator copies this pattern exactly, substituting benchmark workers for selection workers.

- `nce/benchmark/training.py` (on main) — **`train_single_bucket()` API.** Takes `(bucket_pt_path, nn_config, time_limit_seconds, output_dir, device)`, returns `{epochs_completed, final_loss, final_local_error, error_tracking_data, losses, wall_time, bucket_id, config_used}`. The benchmark worker subprocess wraps exactly this call.

- `nce/benchmark/training.py:_write_metrics()` — **Config hash computation.** `hashlib.md5(sorted config items)`. The JSONL history entry should use this same hash for config reproducibility tracking.

- `scripts/select_hard_buckets.py:run_phase2()` lines 175–270 — **Phase 2 precomputation.** Runs `eliminate_variables(up_to=target_var, exact=True)`, computes exact_fw and exact_bw, saves to .pt. This is the code that will generate the missing .pt files.

- `notebooks/_1-2026/experiment_runner.py:spawn_worker()` — **Subprocess spawn pattern.** `env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)`, `subprocess.Popen` with stdout/stderr pipes and text mode. The benchmark worker subprocess follows this pattern.

- `scripts/verify_benchmark_training.py:_generate_synthetic_pt()` (on main) — **Synthetic .pt fallback.** Creates a valid .pt from smokers_20 when real S01 data isn't available. Useful for testing the CLI without waiting for real .pt generation.

## Constraints

- **S02 code not on S03 branch.** `nce/benchmark/` files exist only on `main` (squash-merged from S02 branch). Must cherry-pick or `git checkout main -- nce/benchmark/ scripts/verify_benchmark_training.py` before any S03 implementation.

- **S01 .pt files missing.** `data/hard_buckets/` is empty. Phase 1 results exist in `/tmp/hard_bucket_selection_hvmlbbvv/` (21/24 problems, 4 hard buckets at threshold 0.1). Phase 2 must run to generate .pt files. The `--skip-phase1` flag exists for this exact scenario — assemble selection_results.json from tmp data, then run Phase 2.

- **Tmp directory is ephemeral.** `/tmp/hard_bucket_selection_hvmlbbvv/` could be cleaned on reboot. First task must persist this data immediately.

- **Worker subprocess must not import torch at module level.** `CUDA_VISIBLE_DEVICES` must be set in the subprocess environment before torch is imported. The worker script must be a subprocess entry point, not an inline import.

- **Max 1 worker per GPU (D046).** Concurrent workers on the same GPU cause OOM (observed in S01 T02). Worker pool model is mandatory.

- **Per-worker temp files + coordinator merge (D042).** Each GPU worker writes to a separate temp JSON file. Coordinator merges after all workers finish and appends one JSONL entry to history.jsonl. No concurrent writes to the history file.

- **Time limits: fast=60s, slow=3600s per bucket.** These are wall-clock limits checked at epoch boundaries in `train_single_bucket()`. Overshoot by one epoch is acceptable (documented behavior).

- **Config YAML format must match prepare_config() input.** The CLI loads YAML with `yaml.safe_load()` and passes through `prepare_config()`. Expected format: flat dict or nested sections. NOT the experiment_runner format (which has `architectures:` lists).

- **4× NVIDIA TITAN RTX (24GB each).** Available GPUs 0-3. Default `--gpus 0,1,2,3`.

## Common Pitfalls

- **Forgetting to restore S02 code first.** Any import of `nce.benchmark` will fail with `ModuleNotFoundError` until the files are cherry-picked from main.

- **Tmp directory vanishing.** The Phase 1 results in `/tmp/` are not persisted. Must be copied to `data/hard_buckets/selection_results.json` in the very first task before anything else.

- **JSONL append vs overwrite.** Use `open('history.jsonl', 'a')` to append. Each run adds one JSON line. First run creates the file.

- **Comparison chart with no history.** First benchmark run has no previous data. Chart should either skip gracefully or show only the current run's bars without "historical best."

- **Worker config missing required Trainer keys.** If the user's YAML config doesn't include all keys Trainer.__init__ requires (lower_dim, debug, traced_losses, optimizer, etc.), the worker will fail. The benchmark must set sensible defaults for missing keys (same as S02's verify script, which derived config from small_problems defaults). See D037.

- **`_json_default` not handling numpy types.** The JSONL history merge may encounter numpy float/int types from aggregation. Add handlers for numpy scalar types.

- **Forgetting to free GPU memory between workers.** If a worker subprocess crashes, GPU memory may not be freed. The coordinator should check `nvidia-smi` output or rely on subprocess termination to reclaim memory. Subprocess isolation (each worker is a separate process) naturally handles this.

## Open Risks

- **Phase 2 runtime.** Exact upstream elimination + backward message computation for 4 hard buckets takes ~2-15 minutes total on GPU. This is a one-time cost but must happen before the CLI is usable. If any bucket's backward elimination exceeds GPU memory, that bucket won't be cached.

- **3 failed OOM problems.** Problems 18, 19, 21 failed Phase 1 with OOM. These are the largest problems (deer_rescaled K10.F2, deer_rescaled K15.F1.5, and a BN variant). They might have hard buckets, but we can't know without solving the OOM issue. For now, 4 hard buckets from 21 problems is sufficient.

- **JSONL file size over time.** Each history entry includes per-bucket training data (loss curves, error tracking tuples). For 4 buckets × 10000 epochs, one entry is ~500KB-1MB. After hundreds of runs, history.jsonl could grow large. Not a concern for the foreseeable future but worth noting.

- **Config hash collisions.** MD5 of sorted config items is not cryptographic. Two similar configs could collide. Acceptable for a research tool.

## Skills Discovered

| Technology | Skill | Status |
|------------|-------|--------|
| PyTorch | `mindrally/skills@pytorch` (107 installs) | available, not needed — patterns well-established |
| matplotlib | `ovachiever/droid-tings@matplotlib` (28 installs) | available, not needed — simple grouped bar chart |

No skills needed — this is integration work using established codebase patterns.

## Sources

- S01 coordinator: `scripts/select_hard_buckets.py` — worker pool pattern, Phase 2 precomputation
- S02 training harness: `main:nce/benchmark/training.py` — `train_single_bucket()` API, `_write_metrics()`, `_load_bucket_data()`
- S02 plots: `main:nce/benchmark/plots.py` — `plot_loss_curve()`, `plot_local_error_curve()`
- S02 verification: `main:scripts/verify_benchmark_training.py` — synthetic .pt fallback, structured validation
- Experiment runner: `notebooks/_1-2026/experiment_runner.py` — subprocess GPU isolation, wave scheduling
- Phase 1 temp data: `/tmp/hard_bucket_selection_hvmlbbvv/` — 24 files (21 succeeded, 3 OOM), 4 hard buckets at threshold 0.1
- S01 summary: `.gsd/milestones/M004/slices/S01/S01-SUMMARY.md` — .pt schema, bucket ID convention, worker pool pattern
- Decision register: D042 (per-worker temp + merge), D046 (max 1 per GPU), D047 (custom epoch loop)
