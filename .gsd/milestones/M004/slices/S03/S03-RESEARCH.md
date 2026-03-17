# S03: Multi-GPU CLI, History Tracking & Comparison — Research

**Date:** 2026-03-17

## Summary

S03 integrates S01's precomputed bucket data with S02's single-bucket training harness into the full user-facing benchmark CLI. The building blocks are battle-tested and the patterns are proven, but **two prerequisite gaps** must be resolved before any S03 code can work:

1. **S02 code deleted on the S03 branch.** `nce/benchmark/{__init__.py, training.py, plots.py}` and `scripts/verify_benchmark_training.py` exist on `main` (confirmed: `git show main:nce/benchmark/__init__.py` succeeds) but were deleted when the S03 branch diverged. Must restore via `git checkout main -- nce/benchmark/ scripts/verify_benchmark_training.py`.

2. **S01 Phase 2 never executed.** `data/hard_buckets/` is empty — no `.pt` files, no `bucket_list.json`, no `selection_results.json`. However, Phase 1 results are intact in `/tmp/hard_bucket_selection_hvmlbbvv/` (24 files, all 24 problems reported — 21 succeeded, 3 OOM on problems 18/19/21). The 4 hard buckets (threshold 0.1) are: `grid10x10.f5.wrap.uai` bucket 10 (0.8326), `or_chain_10.fg.uai` buckets 88 (0.1669) and 154 (0.1988), `BN_2.uai` bucket 9 (0.1402). Must assemble `selection_results.json` from tmp data and run `python scripts/select_hard_buckets.py --skip-phase1` to execute Phase 2 and generate `.pt` files.

The multi-GPU CLI reuses S01's worker pool pattern verbatim (D046: max 1 worker per GPU, poll-based cycling, `CUDA_VISIBLE_DEVICES` isolation). The benchmark worker subprocess is simple: load YAML config, call `train_single_bucket()`, write result JSON to temp file. The coordinator manages the pool, merges results, writes JSONL history (D042: per-worker temp files + coordinator merge), and generates the comparison chart. New code needed: ~200 lines for the worker script, ~250 lines for the coordinator CLI, ~80 lines for `plot_comparison_chart()` in `nce/benchmark/plots.py`, and ~50 lines for JSONL history I/O.

## Requirements Targeted

| Req | Description | What S03 must deliver |
|-----|-------------|----------------------|
| R042 | Multi-GPU parallel execution | Worker pool distributing 1 bucket per GPU, cycling as GPUs free up. Default: `--gpus 0,1,2,3`. |
| R043 | Per-bucket benchmark output (integration) | Multi-GPU flow must produce same per-bucket output folders (loss.png, local_error.png, metrics.json) as S02's single-GPU path. Already handled by `train_single_bucket()`. |
| R044 | Historical comparison tracking | JSONL history file (`data/hard_buckets/history.jsonl`) with per-run metadata (config_hash, timing, mode, per-bucket epochs/errors). Comparison chart: current vs historical best filtered by `time_limit_per_bucket ≤ current`. |
| R045 | CLI benchmark entry point | `python scripts/bucket_benchmark.py config.yaml fast [--gpus 0,1,2,3]` runs the full pipeline end-to-end. YAML parsed via `yaml.safe_load()`, passed through `prepare_config()`. |

Additionally, this slice must complete the **S01 Phase 2 gap** (generating `.pt` files from existing Phase 1 results) to satisfy R039 and R040 preconditions.

## Recommendation

Four tasks, dependency-ordered:

**T01: Restore prerequisites + generate .pt files.**
- `git checkout main -- nce/benchmark/ scripts/verify_benchmark_training.py` to restore S02 code.
- Assemble `data/hard_buckets/selection_results.json` from `/tmp/hard_bucket_selection_hvmlbbvv/` (script-assisted: read 24 JSONs, build merged structure matching coordinator's output format).
- Run `python scripts/select_hard_buckets.py --skip-phase1` to execute Phase 2 (exact upstream elimination + exact_fw + exact_bw + torch.save for each of the 4 hard buckets). Expected runtime: ~2-15 minutes on 1 GPU.
- Validate with `python scripts/verify_hard_buckets.py`.
- Smoke-test: `python scripts/verify_benchmark_training.py --time-limit 15 --device cuda` using a real .pt file.

**T02: Benchmark worker + multi-GPU coordinator CLI.**
- Build `scripts/bucket_benchmark_worker.py` — subprocess entry point. Receives `--pt-path`, `--config` (JSON string or YAML path), `--time-limit`, `--output-dir`. Imports torch after launch, calls `train_single_bucket()`, writes result dict as JSON to stdout or temp file.
- Build `scripts/bucket_benchmark.py` — CLI entry point with `<config.yaml> <fast|slow> [--gpus 0,1,2,3]` interface. Loads YAML via `yaml.safe_load()` → `prepare_config()`. Loads `bucket_list.json` manifest. Creates timestamped output directory under `data/hard_buckets/runs/YYYY-MM-DD_HHMM/`. Manages worker pool (copy of S01's pool pattern). Collects per-worker result JSONs.
- Time limits: fast=60s, slow=3600s.

**T03: JSONL history + comparison chart.**
- Add `plot_comparison_chart()` to `nce/benchmark/plots.py`. Grouped bar chart: x-axis = bucket IDs, y-axis = final local error (log scale), two bars per bucket (current run in blue, historical best in orange). Handle first-run case (no history → only current bars, no comparison).
- After all workers complete, coordinator: (1) merges per-worker temp results, (2) builds run record with config_hash, mode, time_limit_per_bucket, per-bucket results, (3) appends one JSON line to `data/hard_buckets/history.jsonl`, (4) reads history file, filters to runs with `time_limit_per_bucket ≤ current`, finds per-bucket best, (5) generates comparison chart PNG in the output directory.
- JSONL schema per line: `{"timestamp": str, "config_hash": str, "config": dict, "mode": str, "time_limit_per_bucket": int, "gpus": list, "buckets": [{"bucket_id": str, "epochs_completed": int, "wall_time": float, "final_loss": float, "final_local_error": float, "error_tracking_data": list}]}`.

**T04: End-to-end verification on 4 GPUs.**
- Run `python scripts/bucket_benchmark.py config.yaml fast --gpus 0,1,2,3` with a test YAML config.
- Verify: per-bucket output folders with loss.png + local_error.png + metrics.json, JSONL history entry appended, comparison chart generated (or gracefully showing current-only on first run).
- Run a second time to verify comparison chart shows current vs previous.
- Check multi-GPU distribution: 4 buckets on 4 GPUs should complete in ~1 minute wall time (each bucket gets its own GPU).

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| Single-bucket training | `nce.benchmark.training.train_single_bucket()` (on main) | Full lifecycle: .pt load → reconstruct → train → checkpoint errors → plots → metrics. 446 lines, tested via verify script. |
| Multi-GPU subprocess isolation | `scripts/select_hard_buckets.py` run_phase1() worker pool | Proven on this machine — poll-based, 1-per-GPU, `CUDA_VISIBLE_DEVICES`, queue cycling. Handles failures gracefully. |
| Loss/error plots | `nce.benchmark.plots.{plot_loss_curve, plot_local_error_curve}` (on main) | Correct PNGs with Agg backend, semilogy scale, explicit plt.close(). |
| Config validation + YAML | `nce.config_schema.prepare_config()` + `yaml.safe_load()` | Handles flat/nested detection, alias resolution, defaults. PyYAML 5.3.1 installed. |
| Checkpoint schedule | `nce.neural_networks.train.get_error_tracking_epochs()` | Tested: 0, 1, 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000, then every 5000. |
| Config hash | `nce.benchmark.training._write_metrics()` pattern | `hashlib.md5(json.dumps(sorted(items)))` — already used in metrics.json. Reuse for JSONL. |
| Phase 2 precomputation | `scripts/select_hard_buckets.py:run_phase2()` | Exact upstream elimination + exact_fw + exact_bw + torch.save to .pt. Ready to use via `--skip-phase1`. |
| Synthetic .pt for testing | `scripts/verify_benchmark_training.py:_generate_synthetic_pt()` | Creates valid .pt from smokers_20 when real data missing. Useful for worker smoke tests. |

## Existing Code and Patterns

### S01 Worker Pool Pattern
`scripts/select_hard_buckets.py:run_phase1()` lines 57–120 — The exact pattern for S03's multi-GPU coordinator. Key elements:
- `active = {}` dict keyed by GPU ID → (problem_index, gpu_id, proc, output_path)
- `gpu_queues = {g: deque() for g in gpus}` — round-robin assignment, sequential within each GPU
- Poll loop: `proc.poll()` for each active worker, `time.sleep(5)` between polls
- On worker finish: read stdout/stderr, launch next from `gpu_queues[g]`, remove GPU from active if queue empty
- `_spawn_worker()` helper: `env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)`, `subprocess.Popen` with text pipes

### S02 Training API
`main:nce/benchmark/training.py` — `train_single_bucket(bucket_pt_path, nn_config, time_limit_seconds, output_dir, device)` returns:
```python
{
    'epochs_completed': int,
    'final_loss': float,
    'final_local_error': float,
    'error_tracking_data': [(epoch, loss, log_z_err, abs_log_z_err), ...],
    'losses': [(epoch, loss_val), ...],
    'wall_time': float,
    'bucket_id': str,
    'config_used': dict,
    'output_dir': str,  # per-bucket subfolder path
    'loss_plot_path': str,  # optional
    'error_plot_path': str,  # optional
    'metrics_path': str,  # optional
}
```
The function handles its own per-bucket output folder creation (`{output_dir}/{bucket_id}/`) and saves plots + metrics.json. The CLI coordinator just needs to call this via subprocess and collect results.

### Config Hash Pattern
`main:nce/benchmark/training.py:_write_metrics()` lines ~105-115:
```python
config_str = json.dumps(sorted(nn_config.items(), key=lambda x: str(x[0])), default=str)
config_hash = hashlib.md5(config_str.encode()).hexdigest()
```
Reuse this same hash computation in the JSONL history entry for consistency.

### S02 Per-Bucket Output Structure
Each bucket produces: `{output_dir}/{bucket_id}/loss.png`, `local_error.png`, `metrics.json`. The `bucket_id` format: `{safe_problem_key}__bucket_{label}` (e.g., `or_chain_10_fg_uai__bucket_88`). This is already handled by `train_single_bucket()`.

### Phase 1 Data Schema (from /tmp)
Each `problem_N.json`:
```python
{
    "problem_index": int,
    "problem_key": str,       # e.g., "smokers_20.uai"
    "model_file": str,        # same as problem_key for small_problems
    "auto_ecl": int,
    "buckets": [
        {
            "label": int,
            "error_data": [(epoch, loss, log_Z_err, abs_log_Z_err), ...],
            "final_abs_log_Z_err": float,
            "num_epochs": int,
        }, ...
    ]
}
```
Error entries (problems 18, 19, 21): `{"problem_index": int, "error": str, "traceback": str}` with no `buckets` key.

## Constraints

- **S02 files must be restored first.** Any import of `nce.benchmark` on the S03 branch will fail with `ModuleNotFoundError`. Must run `git checkout main -- nce/benchmark/ scripts/verify_benchmark_training.py` before any implementation.

- **S01 .pt files must be generated first.** `data/hard_buckets/` is empty. Phase 1 temp data in `/tmp/` is ephemeral and could be cleaned on reboot. Must persist to `selection_results.json` and run Phase 2 ASAP.

- **Worker subprocess must not import torch at module level (confirmed pattern from S01).** `CUDA_VISIBLE_DEVICES` must be set in the subprocess env before torch is imported. Worker script must be a standalone entry point.

- **Max 1 worker per GPU (D046).** Concurrent workers on same GPU cause OOM. Worker pool model is mandatory.

- **Per-worker temp files + coordinator merge (D042).** Each worker writes to separate temp JSON. Coordinator merges after all finish. No concurrent writes to history.jsonl.

- **Config YAML must be compatible with `prepare_config()` input.** The CLI should accept a standard flat or nested YAML config dict (NOT the `experiment_runner.py` format with `architectures:` lists).

- **4× NVIDIA TITAN RTX (24GB each).** GPUs 0-3 available. GPU 0 has ~6GB used by Jupyter kernel but 18GB free. All 4 GPUs usable for benchmark.

- **Time limits: fast=60s, slow=3600s per bucket.** Checked at epoch boundaries in `train_single_bucket()`. Overshoot by at most one epoch is acceptable.

- **PyYAML 5.3.1 installed.** Use `yaml.safe_load()`. No need for additional YAML deps.

- **`train_single_bucket()` requires specific Trainer keys.** Config must include `approximation_method`, `debug`, `traced_losses`, `optimizer`, `sampling_scheme='all'`. The function sets `sampling_scheme` and `error_tracking=False` internally, but other Trainer-required keys come from the user's config or defaults. If the user's YAML omits them, must set sensible defaults (pattern: merge with `small_problems.configs['default'][problem_idx]` or hard-code a BENCHMARK_DEFAULTS dict).

## Common Pitfalls

- **Forgetting to restore S02 code first.** Any `from nce.benchmark import ...` will fail on the current S03 branch. This is the very first step of T01.

- **Tmp directory vanishing.** `/tmp/hard_bucket_selection_hvmlbbvv/` could be cleaned by the system. Must copy/persist in T01 before anything else.

- **JSONL append vs overwrite.** Use `open('history.jsonl', 'a')` for appending. Each run adds one JSON line. First run creates the file. `json.loads(line)` to read each line.

- **Comparison chart with no history.** First benchmark run has no previous data to compare against. Chart should show only current run's bars with a note "No historical data for comparison" or skip the chart gracefully and print a message.

- **Worker config missing required Trainer keys.** If the user's YAML config doesn't include `lower_dim`, `debug`, `traced_losses`, `optimizer`, etc., Trainer.__init__ will KeyError. Must provide defaults. See D037 — NEUROBE_DEFAULTS includes these, but benchmark needs its own defaults dict or should merge with small_problems default configs.

- **numpy types in JSON serialization.** The JSONL merge may encounter numpy scalar types. The `_json_default` handler in training.py handles `torch.Tensor` and `set` but not numpy. Add `numpy.integer` → `int`, `numpy.floating` → `float` handlers.

- **`log_z_err` can be a torch scalar.** The error_tracking_data tuples from `train_single_bucket()` may contain torch tensors (from `approx_contribution - exact_contribution`). JSONL serialization needs `_json_default` or explicit `.item()` conversion. The training code already does `abs(log_z_err)` which returns Python float for scalar tensors, but `log_z_err` itself may still be a tensor.

- **Stale GPU processes.** If a worker subprocess crashes, GPU memory is reclaimed when the process exits (subprocess isolation handles this). But if the coordinator is killed before workers finish, orphaned workers remain. Add cleanup in coordinator's exception handler.

## Open Risks

- **Phase 2 runtime.** Exact upstream elimination + backward message computation for 4 hard buckets. `grid10x10.f5.wrap.uai` is the largest problem and its backward message at `bw_ecl=2^30` could be expensive. Expected: 2-15 minutes total on 1 GPU. If a bucket's backward elimination exceeds 24GB GPU memory, that bucket won't be cached — monitor Phase 2 output.

- **3 OOM problems excluded.** Problems 18, 19, 21 (deer_rescaled variants) failed Phase 1 with OOM. They might have hard buckets, but we can't know without solving OOM. 4 hard buckets from 21 problems is sufficient for now.

- **`train_single_bucket` config sensitivity.** The function merges user config with bucket metadata and calls `prepare_config()`. If the user passes a config designed for a different purpose (e.g., neurobe_mode), unexpected interactions could occur. The function already sets `error_tracking=False` and `sampling_scheme='all'` but doesn't override other potentially conflicting settings.

- **JSONL file growth.** Each history entry includes per-bucket loss/error curves. For 4 buckets × 10000 max epochs, each loss entry is ~20 bytes. With all checkpoints, one run adds ~100-500KB. After hundreds of runs: ~50MB. Acceptable for research use.

## Skills Discovered

| Technology | Skill | Status |
|------------|-------|--------|
| PyTorch | `mindrally/skills@pytorch` (107 installs) | available — not needed, patterns established |
| matplotlib | `ovachiever/droid-tings@matplotlib` (28 installs) | available — not needed, simple bar chart |

No skills needed. This is integration/orchestration work using proven codebase patterns.

## Sources

- S01 coordinator worker pool: `scripts/select_hard_buckets.py` lines 57-120 (run_phase1) and lines 175-270 (run_phase2)
- S02 training harness: `main:nce/benchmark/training.py` — 446 lines, `train_single_bucket()` API
- S02 plots: `main:nce/benchmark/plots.py` — 88 lines, `plot_loss_curve()`, `plot_local_error_curve()`
- S02 verification: `main:scripts/verify_benchmark_training.py` — 391 lines, synthetic .pt fallback
- Experiment runner subprocess pattern: `notebooks/_1-2026/experiment_runner.py` — `spawn_worker()`, wave scheduling
- Phase 1 temp data: `/tmp/hard_bucket_selection_hvmlbbvv/` — 24 JSON files (21 success, 3 OOM), 4 hard buckets at threshold 0.1
- S01 summary: `.gsd/milestones/M004/slices/S01/S01-SUMMARY.md` — .pt schema, bucket ID convention, D046 worker pool
- S02 placeholder summary: `.gsd/milestones/M004/slices/S02/S02-SUMMARY.md` — doctor placeholder, inspect task summaries for real data
- Decision register: D042 (per-worker temp + merge), D046 (max 1 per GPU), D047 (custom epoch loop), D048 (synthetic .pt fallback)
- Config schema: `nce/config_schema.py` — `prepare_config(config_dict, strict=False)`
- GPU status: 4× TITAN RTX, GPU 0 has ~6GB used (Jupyter), GPUs 1-3 clean
