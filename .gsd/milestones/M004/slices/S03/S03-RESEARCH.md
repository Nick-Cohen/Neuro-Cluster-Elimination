# S03: Multi-GPU CLI, History Tracking & Comparison — Research

**Date:** 2026-03-12

## Summary

S03 is a straightforward integration slice — the hard algorithmic work was done in S01 (bucket selection) and S02 (single-bucket training harness). S03 wires them together with a CLI entry point, multi-GPU subprocess orchestration, JSONL history tracking, and a comparison chart. All the building blocks exist and are battle-tested.

The main complication is that **S02's code (`nce/benchmark/` module + `scripts/verify_benchmark_training.py`) exists on the `gsd/M004/S02` branch but is not on the current S03 branch**. This must be resolved as the first task — bring the code forward via `git checkout gsd/M004/S02 -- nce/benchmark/ scripts/verify_benchmark_training.py`. Similarly, **S01's Phase 2 never completed** — the `data/hard_buckets/` directory is empty (no `.pt` files, no `bucket_list.json`). The Phase 1 results exist in `/tmp/hard_bucket_selection_hvmlbbvv/` (22/24 problems) and identified 4 hard buckets. S03's CLI should handle the missing-data case gracefully (via the synthetic .pt fallback pattern from S02's verification script, or by running Phase 2 as a prerequisite).

The multi-GPU pattern is well-established in both the S01 coordinator (`scripts/select_hard_buckets.py`) and `notebooks/_1-2026/experiment_runner.py`. S03 reuses the worker-pool model (D046: max 1 worker per GPU, spawn next when one finishes). The JSONL history uses per-worker temp files merged by the coordinator (D042). The comparison chart is a simple matplotlib grouped bar chart.

## Recommendation

Three tasks, ordered by dependency:

**T01: Bring S02 code forward + complete S01 data.** Cherry-pick S02's `nce/benchmark/` module and verification script onto the S03 branch. Then run `select_hard_buckets.py --skip-phase1` to execute Phase 2 (precompute .pt files from existing Phase 1 results), or copy Phase 1 results to `data/hard_buckets/selection_results.json` and run Phase 2. Verify with `verify_hard_buckets.py`.

**T02: CLI entry point + multi-GPU orchestration + JSONL history.** Build `scripts/bucket_benchmark.py` with the full CLI interface. The script: (1) parses `<config.yaml> <fast|slow> [--gpus 0,1,2,3]`, (2) loads `bucket_list.json` manifest, (3) creates a timestamped output directory, (4) spawns benchmark worker subprocesses (one per GPU, cycling through buckets as GPUs free up), (5) each worker calls `train_single_bucket()` and writes a temp JSON result file, (6) coordinator merges worker results into a JSONL history entry and appends to `data/hard_buckets/history.jsonl`.

**T03: Comparison chart + end-to-end verification.** Add `plot_comparison_chart()` to `nce/benchmark/plots.py`. Reads `history.jsonl`, filters to runs with duration ≤ current run's per-bucket time limit, finds historical best per bucket, plots grouped bars (current vs best). Wire into the coordinator's post-merge step. Run full end-to-end on 4 GPUs to verify.

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| Single-bucket training | `nce/benchmark/training.train_single_bucket()` (S02) | Full lifecycle: .pt loading, reconstruction, custom epoch loop, error tracking, plots, metrics |
| Multi-GPU subprocess isolation | `scripts/select_hard_buckets.py` worker pool pattern (S01) | Battle-tested on this exact machine; handles CUDA_VISIBLE_DEVICES, worker pool (1 per GPU), stdout/stderr capture |
| Per-run output: loss/error plots | `nce/benchmark/plots.plot_loss_curve()`, `plot_local_error_curve()` (S02) | Already produces correct PNGs with Agg backend |
| Config validation + YAML parsing | `nce/config_schema.prepare_config()` + `yaml.safe_load()` | Handles flat/nested detection, alias resolution, neurobe_mode expansion |
| Checkpoint epoch schedule | `nce/neural_networks/train.get_error_tracking_epochs()` | Tested schedule used by S02's training loop |
| Precomputed bucket data loading | `nce/benchmark/training._load_bucket_data()` (S02) | Handles tensor device mapping, FastFactor reconstruction |

## Existing Code and Patterns

- `scripts/select_hard_buckets.py` — **Worker pool orchestration pattern.** The `run_phase1()` function implements exactly the multi-GPU pattern S03 needs: `active = {}` keyed by GPU ID, poll with `proc.poll()`, spawn next from queue when worker finishes, merge results after all complete. Copy and adapt — the S01 coordinator handles the same fundamental problem (distribute N items across M GPUs, one-at-a-time per GPU).

- `nce/benchmark/training.py` (on `gsd/M004/S02` branch) — **`train_single_bucket()` is the complete single-bucket training pipeline.** Returns a result dict with `{epochs_completed, final_loss, final_local_error, error_tracking_data, losses, wall_time, bucket_id, config_used}`. The benchmark CLI worker subprocess just needs to call this function and write the result to a temp JSON file.

- `nce/benchmark/plots.py` (on `gsd/M004/S02` branch) — **Loss and error plotting.** Two standalone functions. The comparison chart is a new addition — a grouped bar chart showing per-bucket final errors for current run vs historical best.

- `scripts/verify_benchmark_training.py` (on `gsd/M004/S02` branch) — **Synthetic .pt fallback pattern.** `_generate_synthetic_pt()` creates a valid .pt file from smokers_20 without needing S01's pipeline. This pattern should be reused in the CLI's `--dry-run` mode or when .pt files are missing.

- `notebooks/_1-2026/experiment_runner.py:spawn_worker()` — **Clean subprocess spawn pattern.** `env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)`, subprocess.Popen with stdout/stderr pipes, text mode. The benchmark worker script should follow this exact pattern.

- `data/hard_buckets/bucket_list.json` (schema from S01) — **Manifest format.** `{threshold, selection_date, num_problems, total_nn_buckets, buckets: [{id, problem_key, bucket_label, selection_error, auto_ecl, file}]}`. The CLI reads this to know which .pt files to train.

## Constraints

- **S02 code not on S03 branch.** Files `nce/benchmark/__init__.py`, `nce/benchmark/training.py`, `nce/benchmark/plots.py`, and `scripts/verify_benchmark_training.py` exist only on `gsd/M004/S02`. Must be checked out to the current branch before any S03 work begins.

- **S01 Phase 2 data missing.** `data/hard_buckets/` is empty — no `.pt` files, no `bucket_list.json`. Phase 1 results exist in `/tmp/hard_bucket_selection_hvmlbbvv/` (22/24 problems). Phase 2 must be run (or the CLI must handle missing data gracefully).

- **Worker subprocess must import torch AFTER CUDA_VISIBLE_DEVICES is set.** This is already handled by the subprocess pattern (child process inherits env), but the worker script must not import torch at module level if CUDA_VISIBLE_DEVICES is set dynamically.

- **JSONL history per-worker + merge (D042).** Each GPU worker writes results to a separate temp file. The coordinator merges after all workers finish and appends a single structured entry to `history.jsonl`. No concurrent writes to the history file.

- **Max 1 worker per GPU (D046).** The worker pool pattern from S01 must be followed — concurrent workers on the same GPU cause OOM.

- **`train_single_bucket()` takes a flat nn_config dict** that goes through `prepare_config()`. The CLI must load YAML, merge any per-benchmark overrides, and pass a valid flat dict. The config must include all Trainer-required keys (see D037).

- **Time limits: fast=60s, slow=3600s per bucket.** These are wall-clock limits checked at epoch boundaries. Overshoot by at most one epoch is acceptable (documented behavior from S02).

- **Config hash for history tracking.** `_write_metrics()` already computes `hashlib.md5(sorted config items)`. The JSONL history entry should include this same hash for reproducibility tracking.

## Common Pitfalls

- **Forgetting to bring S02 code forward.** The `nce/benchmark/` module only exists on the S02 branch. Attempting to import from it on S03 will fail with `ModuleNotFoundError`. Must be resolved before any implementation work.

- **Assuming .pt files exist.** S01's Phase 2 never ran. The CLI must either (a) require `bucket_list.json` and error clearly if missing, or (b) fall back to synthetic generation. Option (a) is simpler and more honest — the user should run `select_hard_buckets.py` first.

- **Worker script importing torch at module level.** If the benchmark worker script is structured as a module that imports `nce.benchmark.training` at the top, torch will be imported before `CUDA_VISIBLE_DEVICES` is set. The worker must be a script that sets the env var or reads it from the already-set subprocess env before importing.

- **JSONL append vs overwrite.** Use `open('history.jsonl', 'a')` to append, not `'w'`. Each run appends one JSON object (one line).

- **Comparison chart with no history.** The first run has nothing to compare against. The chart should either show "No previous runs" or skip the comparison bar entirely.

- **Config YAML format mismatch.** The CLI's config.yaml should be a flat or nested dict (matching `prepare_config()` input). It should NOT be the experiment_runner YAML format (which has `architectures:` lists and `bw_ecl:` lists). Document the expected format.

- **`_json_default` not handling all types.** The existing `_json_default` in training.py handles tensors and sets. JSONL entries may include numpy types or pathlib.Path objects — add handlers.

## Open Risks

- **S01 data completeness.** 2/24 problems never completed Phase 1 (problems 16 and 20). Those problems may or may not have hard buckets. Given 4 hard buckets from 22 problems, this is acceptable — but worth noting.

- **Phase 2 runtime.** Running Phase 2 (precomputing .pt files for 4 hard buckets) takes 5-15 minutes as it runs exact upstream elimination + exact backward message computation. This is a one-time cost but must happen before the CLI is usable.

- **Comparison chart clarity.** With only 4 hard buckets, the grouped bar chart may look sparse. Consider also plotting the error tracking curves side-by-side for richer comparison.

- **Config hash collision.** MD5 of sorted config items is fast but not cryptographic. Two nearly-identical configs could theoretically collide. Acceptable for a research tool — worst case is comparing against the wrong "previous best."

## Skills Discovered

| Technology | Skill | Status |
|------------|-------|--------|
| PyTorch | `mindrally/skills@pytorch` (107 installs) | available, not needed — patterns well established |
| matplotlib | `ovachiever/droid-tings@matplotlib` (28 installs) | available, not needed — simple bar chart |

No skills needed for this slice — it's integration work using established patterns.

## Sources

- S01 coordinator: `scripts/select_hard_buckets.py` — multi-GPU worker pool pattern, Phase 2 precomputation
- S02 training harness: `gsd/M004/S02:nce/benchmark/training.py` — `train_single_bucket()` API
- S02 plots: `gsd/M004/S02:nce/benchmark/plots.py` — loss/error curve functions
- S02 verification: `gsd/M004/S02:scripts/verify_benchmark_training.py` — synthetic .pt fallback pattern
- Experiment runner: `notebooks/_1-2026/experiment_runner.py` — subprocess GPU isolation pattern
- Phase 1 temp data: `/tmp/hard_bucket_selection_hvmlbbvv/problem_*.json` — 22/24 problems complete, 4 hard buckets found at threshold 0.1
- Decision register: D042 (per-worker JSONL + merge), D046 (max 1 worker per GPU), D047 (custom epoch loop)
