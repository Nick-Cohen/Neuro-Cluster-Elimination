# S03: Multi-GPU CLI, History Tracking & Comparison — Research

**Date:** 2026-03-17 (updated 21:37 PDT — fresh audit of all prerequisites, code state, GPU state)

## Summary

S03 is the final integration slice for M004 — wiring S01's precomputed bucket data and S02's `train_single_bucket()` harness into a multi-GPU CLI with JSONL history tracking and comparison charts. This is low-risk integration work: all building blocks exist and are battle-tested. The multi-GPU subprocess pattern is proven in `scripts/select_hard_buckets.py` (S01 coordinator — handles worker pool, poll-based scheduling, CUDA isolation). The single-bucket training API is a verified 446-line module with full lifecycle coverage (load → reconstruct → train → checkpoint → plot → metrics).

**Two prerequisite gaps must be resolved before new S03 code can run:**

1. **S02 code missing on S03 branch.** `nce/benchmark/{__init__.py, training.py, plots.py}` and `scripts/verify_benchmark_training.py` exist on both `main` and `gsd/M004/S02` branches but are deleted on the current `gsd/M004/S03` branch (confirmed: `git diff main gsd/M004/S03 -- nce/benchmark/` shows 548 lines deleted). Must restore via `git checkout main -- nce/benchmark/ scripts/verify_benchmark_training.py`.

2. **S01 Phase 2 never executed.** `data/hard_buckets/` directory is empty — no `.pt` files, no `bucket_list.json`, no `selection_results.json`. Phase 1 results are intact in `/tmp/hard_bucket_selection_hvmlbbvv/` (24 JSON files — 21 success, 3 OOM for problems 18/19/21). 4 hard buckets identified at threshold 0.1:
   - `grid10x10.f5.wrap.uai` bucket 10 — abs_log_Z_err = 0.8326
   - `or_chain_10.fg.uai` bucket 154 — abs_log_Z_err = 0.1988
   - `or_chain_10.fg.uai` bucket 88 — abs_log_Z_err = 0.1669
   - `BN_2.uai` bucket 9 — abs_log_Z_err = 0.1402

   Must persist Phase 1 data from `/tmp/` to permanent storage and run Phase 2 to generate `.pt` files. **Phase 1 tmp data is ephemeral — persist immediately.**

The new S03 code is ~580 lines across four files:
- `scripts/bucket_benchmark_worker.py` (~120 lines) — subprocess entry point
- `scripts/bucket_benchmark.py` (~250 lines) — CLI coordinator with worker pool
- `nce/benchmark/plots.py` additions (~80 lines) — comparison chart function
- History I/O and integration wiring (~130 lines)

All 4 GPUs verified clean (1 MiB used each, 24GB total each). No stale Python GPU processes.

## Requirements Targeted

| Req | Description | What S03 must deliver |
|-----|-------------|----------------------|
| R042 | Multi-GPU parallel execution | Worker pool distributing 1 bucket per GPU, cycling as GPUs free up. `--gpus 0,1,2,3` interface. Reuse S01's poll-based pool pattern (D046: max 1 per GPU). |
| R043 | Per-bucket benchmark output (integration) | Multi-GPU flow produces same per-bucket output folders as S02's single-GPU path. Already handled by `train_single_bucket()` — S03 orchestrates subprocess calls that delegate to it. |
| R044 | Historical comparison tracking | JSONL file at `data/hard_buckets/history.jsonl`. Per-run record: `{timestamp, config_hash, config, mode, time_limit_per_bucket, buckets: [{bucket_id, epochs_completed, wall_time, local_errors_at_checkpoints, final_local_error}]}`. Comparison chart: grouped bar chart showing current run vs historical best per bucket, filtered by `time_limit_per_bucket ≤ current run's time limit`. |
| R045 | CLI benchmark entry point | `python scripts/bucket_benchmark.py config.yaml fast [--gpus 0,1,2,3]`. YAML via `yaml.safe_load()`, config through `prepare_config()`. Modes: fast=60s/bucket, slow=3600s/bucket. |

**Prerequisite completion:** This slice must also complete the S01 Phase 2 gap (generating `.pt` files from Phase 1 results) to satisfy R039 and R040 preconditions.

## Recommendation

Four tasks, dependency-ordered:

**T01: Restore prerequisites + generate .pt files (~30min active + ~10min GPU).**
Three steps: (a) `git checkout main -- nce/benchmark/ scripts/verify_benchmark_training.py` to restore S02 code. (b) Assemble `data/hard_buckets/selection_results.json` from the 24 JSON files in `/tmp/hard_bucket_selection_hvmlbbvv/` — Python script to merge into the coordinator's expected format (`{threshold, selection_date, num_problems, gpus, results: [...]}`). (c) Run `python scripts/select_hard_buckets.py --skip-phase1` on GPU to execute Phase 2. Validate with `python scripts/verify_hard_buckets.py`. Smoke-test single-bucket training with `python scripts/verify_benchmark_training.py --time-limit 15 --device cuda`.

**T02: Benchmark worker + multi-GPU coordinator CLI (~45min).**
Build `scripts/bucket_benchmark_worker.py` — subprocess entry point receiving `--pt-path`, `--config-json`, `--time-limit`, `--output-dir`, `--result-path`. Imports torch only after launch. Calls `train_single_bucket()`, serializes result to `--result-path` as JSON. Build `scripts/bucket_benchmark.py` — CLI coordinator with `<config.yaml> <fast|slow> [--gpus 0,1,2,3]` interface. Loads YAML → merges over `small_problems.configs['default'][problem_idx]` per bucket → spawns workers via pool pattern. Time limits: fast=60s, slow=3600s. Output dir: `data/hard_buckets/runs/YYYY-MM-DD_HHMM/`.

**T03: JSONL history tracking + comparison chart (~30min).**
Add `plot_comparison_chart()` to `nce/benchmark/plots.py` — grouped bar chart, x=bucket IDs, y=final local error (log scale), two series (current=blue, historical best=orange). Handle first-run (no history). After all workers complete, coordinator: merges per-worker temp results, builds JSONL record, appends to `data/hard_buckets/history.jsonl`, reads history for per-bucket best (filtered by `time_limit_per_bucket ≤ current`), generates comparison PNG.

**T04: End-to-end verification on 4 GPUs (~10min).**
Create `configs/benchmark_default.yaml`. Run `python scripts/bucket_benchmark.py configs/benchmark_default.yaml fast --gpus 0,1,2,3`. Verify: per-bucket output folders, JSONL history entry, comparison chart. Run again to verify comparison chart shows current vs previous. Multi-GPU: 4 buckets × 4 GPUs should complete in ~1 minute wall time.

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| Single-bucket training | `nce.benchmark.training.train_single_bucket()` on `main` (446 lines) | Full lifecycle: .pt load → reconstruct → train → checkpoint errors → plots → metrics. Verified in S02 (8/8 checks pass, 1337 epochs in 30s). |
| Multi-GPU subprocess isolation | `scripts/select_hard_buckets.py:run_phase1()` lines 57–120 | Proven on this machine. Poll-based, 1-per-GPU (D046), CUDA_VISIBLE_DEVICES, queue cycling. Handled OOM gracefully. |
| Loss/error plots | `nce.benchmark.plots.{plot_loss_curve, plot_local_error_curve}` on `main` | Correct PNGs with Agg backend, semilogy scale, explicit `plt.close(fig)`. |
| Config validation + YAML | `nce.config_schema.prepare_config()` + `yaml.safe_load()` | Handles flat/nested detection, alias resolution, defaults. PyYAML 5.3.1 installed. |
| Checkpoint epoch schedule | `nce.neural_networks.train.get_error_tracking_epochs()` | Tested: 0, 1, 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000, then every 5000. |
| Config hash computation | `nce.benchmark.training._write_metrics()` pattern | `hashlib.md5(json.dumps(sorted(items)))` — reuse for JSONL consistency. |
| Phase 2 precomputation | `scripts/select_hard_buckets.py:run_phase2()` lines 175–270 | Exact upstream elimination + exact_fw + exact_bw + torch.save. Invoke via `--skip-phase1`. |
| Synthetic .pt for testing | `scripts/verify_benchmark_training.py:_generate_synthetic_pt()` | Creates valid .pt from smokers_20 when real data is missing. |

## Existing Code and Patterns

### S01 Worker Pool Pattern (reuse verbatim for T02)
`scripts/select_hard_buckets.py:run_phase1()` — The exact multi-GPU coordination pattern:
- `active = {}` dict keyed by GPU ID → `(problem_index, gpu_id, proc, output_path)`
- `gpu_queues = {g: deque() for g in gpus}` — round-robin assignment, sequential within each GPU
- Poll loop: `proc.poll()` for each active worker, `time.sleep(5)` between polls
- On worker finish: read stdout/stderr, launch next queued item, remove GPU from active if queue empty
- `_spawn_worker()`: sets `env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)`, uses `subprocess.Popen` with text pipes

For S03, the queue items change from problem indices to bucket .pt paths, and workers call `train_single_bucket()` instead of running full inference. The pool mechanics are identical.

### S02 Training API (consumed by T02 worker)
`main:nce/benchmark/training.py:train_single_bucket(bucket_pt_path, nn_config, time_limit_seconds, output_dir, device)` returns:
```python
{
    'epochs_completed': int,
    'final_loss': float,
    'final_local_error': float,
    'error_tracking_data': [(epoch, loss, log_z_err, abs_log_z_err), ...],
    'losses': [(epoch, loss_val), ...],
    'wall_time': float,
    'bucket_id': str,              # e.g. "or_chain_10_fg_uai__bucket_88"
    'config_used': dict,
    'output_dir': str,             # per-bucket subfolder path
    'loss_plot_path': str,         # optional (absent if plot failed)
    'error_plot_path': str,        # optional
    'metrics_path': str,           # optional
}
```
The function handles per-bucket output folder creation and plot/metrics generation internally. The subprocess worker just needs to serialize this result dict to JSON.

Key internals:
- `_load_bucket_data()` reconstructs FastFactors from .pt schema on target device
- `_find_problem()` maps `problem_key` (modelfile string) to `small_problems` index
- `_reconstruct_bucket()` uses `eliminate_variables(up_to=target_var, exact=True)` — takes ~3s/bucket
- Config merging: sets `ecl` from .pt metadata, `error_tracking=False`, `sampling_scheme='all'`, then passes through `prepare_config(strict=False)`
- Custom epoch loop calls `trainer.train_epoch(batches)` — NOT `Trainer.train()` (D047)

### Config Base Pattern (critical for avoiding KeyError)
`scripts/verify_benchmark_training.py` line ~320: config built from `copy.deepcopy(small_problems.configs['default'][0])` with user overrides merged on top. This ensures all 39 Trainer-required keys are present. The benchmark CLI should do the same: load user YAML, then merge over a base from `small_problems.configs['default']` indexed by the bucket's problem.

The 39 keys in `small_problems.configs['default'][0]`: `approximation_method`, `backward_iB`, `batch_size`, `bw_ecl`, `debug`, `device`, `display_intermediate`, `dope_factors`, `ecl`, `fdb`, `gather_message_stats`, `hidden_sizes`, `iB`, `inverse_time_decay_constant`, `loss_fn`, `lower_dim`, `lr`, `lr_decay`, `min_lr`, `momentum`, `nbe_early_stopping`, `num_epochs`, `num_epochs2`, `num_samples`, `optimizer`, `patience`, `plot_messages`, `populate_bw_factors`, `sampling_scheme`, `seed`, `set_size`, `skip_early_stopping`, `stratify_samples`, `traced_losses`, `track_errors`, `use_bw_approx`, `use_linspace_bias`, `use_memorizer`, `val_set`.

### Phase 1 Data Schema
Each `problem_N.json` in `/tmp/hard_bucket_selection_hvmlbbvv/`:
```python
# Success case:
{
    "problem_index": int,
    "problem_key": str,       # e.g., "BN_5.uai"
    "model_file": str,        # same as problem_key
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
# Error case (problems 18, 19, 21 — OOM):
{"problem_index": int, "error": str, "traceback": str}
```

### selection_results.json Schema (coordinator writes, --skip-phase1 reads)
```python
{
    "threshold": float,
    "selection_date": str,      # ISO format
    "num_problems": 24,
    "gpus": [int, ...],
    "results": [...]            # list of 24 entries (problem_N.json contents)
}
```

### bucket_list.json Manifest Schema (Phase 2 output, benchmark reads)
```python
{
    "threshold": float,
    "selection_date": str,
    "num_problems": 24,
    "total_nn_buckets": int,
    "buckets": [
        {
            "id": str,           # "or_chain_10_fg_uai__88" (sanitized)
            "problem_key": str,  # "or_chain_10.fg.uai"
            "bucket_label": int,
            "selection_error": float,
            "auto_ecl": int,
            "file": str,         # "or_chain_10_fg_uai__bucket_88.pt"
        }, ...
    ]
}
```

### JSONL History Record Schema (S03 defines, from M004 roadmap)
```python
{
    "timestamp": str,               # ISO 8601 UTC
    "config_hash": str,             # MD5 of sorted config items
    "config": dict,                 # the prepared config (for reproducing)
    "mode": str,                    # "fast" or "slow"
    "time_limit_per_bucket": int,   # seconds (60 or 3600)
    "output_dir": str,              # path to run's output directory
    "buckets": [
        {
            "bucket_id": str,
            "epochs_completed": int,
            "wall_time": float,
            "local_errors_at_checkpoints": [(epoch, abs_log_z_err), ...],
            "final_local_error": float,
        }, ...
    ]
}
```

### .pt File Schema (S01 Phase 2 output)
```python
{
    'factors': [{'tensor': Tensor (CPU), 'labels': [int, ...]}, ...],
    'exact_fw': {'tensor': Tensor (CPU), 'labels': [int, ...]},
    'exact_bw': {'tensor': Tensor (CPU), 'labels': [int, ...]},
    'bucket_label': int,
    'scope': [int, ...],
    'domain_sizes': [int, ...],
    'elim_vars': [{'label': int, 'states': int}, ...],
    'problem_key': str,         # e.g. "or_chain_10.fg.uai"
    'model_file': str,
    'auto_ecl': int,
    'selection_error': float,
    'selection_epochs': int,
}
```

## Constraints

- **S02 files must be restored before any implementation.** `from nce.benchmark import ...` fails with `ModuleNotFoundError` on current branch. Step 1 of T01.

- **Phase 1 tmp data is ephemeral.** `/tmp/hard_bucket_selection_hvmlbbvv/` could be cleaned on reboot. Must copy to permanent storage in T01 before anything else.

- **Worker subprocess must not import torch at module level.** `CUDA_VISIBLE_DEVICES` must be set in subprocess env before torch import. Worker script must use lazy imports — same pattern as `select_hard_buckets_worker.py`.

- **Max 1 worker per GPU (D046).** Concurrent workers on same GPU cause CUDA OOM. Worker pool model mandatory.

- **Per-worker temp files + coordinator merge (D042).** Each GPU worker writes its result to a separate temp JSON file. Coordinator merges after all workers finish. No concurrent JSONL writes.

- **Config YAML must be compatible with `prepare_config()`.** The CLI accepts a standard flat or nested NCE YAML config dict. NOT the `experiment_runner.py` format with `architectures:` lists.

- **`train_single_bucket()` requires all 39 Trainer-init keys.** Config must include `approximation_method`, `debug`, `traced_losses`, `optimizer`, `lower_dim`, etc. Strategy: load user YAML, merge over `copy.deepcopy(small_problems.configs['default'][problem_idx])`. Let `train_single_bucket()` apply its own overrides (ecl from .pt, sampling_scheme='all', error_tracking=False).

- **Problem index varies per bucket.** Each .pt file has a different `problem_key`, which maps to a different index in `small_problems.problems`. The base config must be looked up per-bucket from `small_problems.configs['default']` using the right index. The worker can resolve this: read `problem_key` from .pt, look up index, get base config.

- **4× NVIDIA TITAN RTX, 24GB each.** All clean (0 GPU processes, 1 MiB used each). Verified via nvidia-smi.

- **Time limits: fast=60s, slow=3600s.** Epoch-boundary checking in `train_single_bucket()`. Overshoot by at most one epoch duration.

- **PyYAML 5.3.1 installed.** Use `yaml.safe_load()`.

- **Python 3.11 in venv at `venv/bin/python`.** Workers must use `sys.executable` for subprocess spawning to ensure correct environment.

## Common Pitfalls

- **Forgetting to restore S02 code before implementing.** Any `from nce.benchmark import ...` fails on current branch. This is literally step 1.

- **Tmp directory vanishing before persistence.** `/tmp/hard_bucket_selection_hvmlbbvv/` could be auto-cleaned. Must copy/persist in T01 before anything else.

- **Worker config missing required Trainer keys.** If user's YAML omits `lower_dim`, `debug`, `traced_losses`, etc., `Trainer.__init__` will KeyError. Must merge user config over `small_problems.configs['default'][problem_idx]`.

- **JSONL append vs overwrite.** Must use `open('history.jsonl', 'a')` for appending. First run creates the file. Read with `[json.loads(line) for line in open(...)]`.

- **Comparison chart with no history.** First benchmark run has no previous data. Chart should show only current run's bars (single series), or skip the "best" series with a note.

- **numpy/torch types in JSON serialization.** Result dict from `train_single_bucket()` may contain torch scalars. Need `_json_default` handler: `torch.Tensor → .item()`, `numpy.integer → int`, `numpy.floating → float`, `set → list`. The existing handler in `training.py` covers torch and set; must add numpy handling.

- **`error_tracking_data` contains torch tensors.** The `(epoch, loss, log_z_err, abs_log_z_err)` tuples may have torch scalar elements. Must convert to Python floats before JSON serialization in the worker result.

- **Stale GPU processes from crashed workers.** Subprocess isolation reclaims GPU memory on exit. But if coordinator is killed, workers may be orphaned. Add `atexit` or signal handler to kill active workers.

- **Config hash consistency between coordinator and worker.** Both the JSONL history entry and the per-bucket `metrics.json` need matching config hashes. Compute the hash on the prepared config in the coordinator and pass to workers, or accept that `train_single_bucket()` independently computes its own hash from its own view of the config.

- **Base config index lookup per bucket.** Different buckets come from different problems. The base config must be `small_problems.configs['default'][problem_idx]` where `problem_idx` varies per bucket. Can't use a single base config for all buckets.

## Open Risks

- **Phase 2 runtime for grid10x10.** The largest hard bucket is `grid10x10.f5.wrap.uai` bucket 10. Exact upstream elimination + `get_backward_message(backward_ecl=2**30)` for this problem could use significant memory/time. Expected: fits in 24GB (auto_ecl ~2^15 = 32768 entries). Monitor Phase 2 stdout.

- **3 OOM problems excluded.** Problems 18, 19, 21 failed Phase 1 with CUDA OOM. They might contain hard buckets. But 4 hard buckets from 21 problems meets the ≥3 threshold. Not a blocker.

- **First-run comparison chart edge case.** No history exists for the first benchmark run. `plot_comparison_chart()` must handle this gracefully — show only current bars or skip chart with a message.

- **Benchmark YAML config design.** Users need a working sample config to actually invoke the CLI. Should create `configs/benchmark_default.yaml` with sensible defaults (UKL loss, [3,3] hidden, lr=0.01, num_epochs=100000, sampling_scheme=all).

- **Config hash stability across config merge.** The coordinator merges user YAML over `small_problems.configs['default']` per bucket. Different buckets may have different base configs (different `ecl`, `iB` values). The JSONL history's `config_hash` should hash only the user-specified config portion (the part that's consistent across buckets), not the per-bucket merged config, to enable meaningful "same config" comparison across runs.

## Skills Discovered

| Technology | Skill | Status |
|------------|-------|--------|
| PyTorch | `mindrally/skills@pytorch` (107 installs) | available — not needed, patterns well-established |
| matplotlib | `ovachiever/droid-tings@matplotlib` (28 installs) | available — not needed, simple additions to existing module |

No skills needed. This is pure integration/orchestration work using established codebase patterns.

## Sources

- S01 coordinator (worker pool): `scripts/select_hard_buckets.py` — 460 lines, `run_phase1()` lines 57–120, `run_phase2()` lines 175–270
- S01 worker subprocess: `scripts/select_hard_buckets_worker.py` — 110 lines, lazy torch import pattern
- S02 training harness: `main:nce/benchmark/training.py` — 446 lines, full `train_single_bucket()` API
- S02 plots: `main:nce/benchmark/plots.py` — 88 lines, `plot_loss_curve()`, `plot_local_error_curve()`
- S02 verification: `main:scripts/verify_benchmark_training.py` — 391 lines, synthetic .pt fallback
- S02 task summaries: `gsd/M004/S02:.gsd/milestones/M004/slices/S02/tasks/{T01,T02,T03}-SUMMARY.md`
- Phase 1 temp data: `/tmp/hard_bucket_selection_hvmlbbvv/` — 24 JSON files (21 success, 3 OOM), 4 hard buckets at threshold 0.1
- Experiment runner: `notebooks/_1-2026/experiment_runner.py` — subprocess spawning with CUDA_VISIBLE_DEVICES, wave-based scheduling
- Config schema: `nce/config_schema.py:prepare_config()` — line 411
- small_problems defaults: `small_problems.configs['default']` — 39-key flat config, 24 entries (one per problem)
- Decision register: D042 (per-worker temp + merge), D046 (max 1 per GPU), D047 (custom epoch loop)
- JSONL schema: M004-ROADMAP boundary map for S03
- GPU state: `nvidia-smi` — 4× TITAN RTX, 1 MiB used each, 24GB total each, 0 GPU processes
- Git state: current branch `gsd/M004/S03`, S02 code on `main` (identical to `gsd/M004/S02`), 548 lines deleted on current branch
