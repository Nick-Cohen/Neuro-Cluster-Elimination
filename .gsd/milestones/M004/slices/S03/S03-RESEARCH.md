# S03: Multi-GPU CLI, History Tracking & Comparison — Research

**Date:** 2026-03-17

## Summary

S03 is the final integration slice — wiring S01's precomputed bucket data and S02's `train_single_bucket()` harness into a multi-GPU CLI with JSONL history tracking and visual comparison charts. All building blocks exist and are battle-tested on this machine. The multi-GPU subprocess pattern is proven in `scripts/select_hard_buckets.py` (S01 coordinator); the single-bucket training API is a 446-line module on `main` with verified end-to-end behavior (1337 epochs in 30s, 8/8 checks pass).

**Two prerequisite gaps** must be resolved before any new S03 code can work:

1. **S02 code deleted on S03 branch.** `nce/benchmark/{__init__.py, training.py, plots.py}` and `scripts/verify_benchmark_training.py` exist on `main` but were deleted when the S03 branch diverged (confirmed: `git diff --stat main gsd/M004/S03 -- nce/benchmark/` shows 939 lines deleted). Must restore via `git checkout main -- nce/benchmark/ scripts/verify_benchmark_training.py`.

2. **S01 Phase 2 never executed.** `data/hard_buckets/` is empty — no `.pt` files, no `bucket_list.json`, no `selection_results.json`. Phase 1 results are intact in `/tmp/hard_bucket_selection_hvmlbbvv/` (24 JSON files, all 24 problems reported — 21 succeeded, 3 OOM). 4 hard buckets identified at threshold 0.1: `grid10x10.f5.wrap.uai` bucket 10 (0.8326), `or_chain_10.fg.uai` buckets 88 (0.1669) and 154 (0.1988), `BN_2.uai` bucket 9 (0.1402). Must assemble `selection_results.json` from tmp data and run `python scripts/select_hard_buckets.py --skip-phase1` to execute Phase 2 and generate `.pt` files. **Phase 1 tmp data in `/tmp/` is ephemeral — must persist immediately.**

New S03 code is ~580 lines across four files: ~120 lines for `scripts/bucket_benchmark_worker.py` (subprocess entry point), ~250 lines for `scripts/bucket_benchmark.py` (CLI coordinator), ~80 lines added to `nce/benchmark/plots.py` (comparison chart), and ~130 lines for history I/O + integration wiring. All GPUs are clean (4× TITAN RTX, 0 processes, 24GB each free).

## Requirements Targeted

| Req | Description | What S03 must deliver |
|-----|-------------|----------------------|
| R042 | Multi-GPU parallel execution | Worker pool distributing 1 bucket per GPU, cycling as GPUs free up. Default: `--gpus 0,1,2,3`. Pattern: S01's `run_phase1()` worker pool (D046: max 1 per GPU). |
| R043 | Per-bucket benchmark output (integration) | Multi-GPU flow must produce same per-bucket output folders (`loss.png`, `local_error.png`, `metrics.json`) as S02's single-GPU path. Already handled by `train_single_bucket()` — S03 just orchestrates subprocess calls. |
| R044 | Historical comparison tracking | JSONL history file (`data/hard_buckets/history.jsonl`) with per-run metadata (config_hash, timing, mode, per-bucket epochs/errors). Comparison chart: current run's final local errors vs historical best per bucket, filtered by `time_limit_per_bucket ≤ current run's time limit`. |
| R045 | CLI benchmark entry point | `python scripts/bucket_benchmark.py config.yaml fast [--gpus 0,1,2,3]` runs the full pipeline end-to-end. YAML parsed via `yaml.safe_load()`, config passed through `prepare_config()`. |

Additionally, this slice must complete the **S01 Phase 2 gap** (generating `.pt` files from existing Phase 1 results) to satisfy R039 and R040 preconditions.

## Recommendation

Four tasks, dependency-ordered:

**T01: Restore prerequisites + generate .pt files (~30min active, ~10min GPU).** Three steps: (a) `git checkout main -- nce/benchmark/ scripts/verify_benchmark_training.py` to restore S02 code. (b) Assemble `data/hard_buckets/selection_results.json` from the 24 JSON files in `/tmp/hard_bucket_selection_hvmlbbvv/` by scripting the merge into the coordinator's expected format. (c) Run `python scripts/select_hard_buckets.py --skip-phase1` on GPU to execute Phase 2 (exact upstream elimination + exact_fw + exact_bw + torch.save for the 4 hard buckets). Validate with `python scripts/verify_hard_buckets.py`. Smoke-test single-bucket training with `python scripts/verify_benchmark_training.py --time-limit 15 --device cuda`.

**T02: Benchmark worker + multi-GPU coordinator CLI (~45min).** Build `scripts/bucket_benchmark_worker.py` — subprocess entry point receiving `--pt-path`, `--config-json`, `--time-limit`, `--output-dir`, `--result-path`. Imports torch only after launch (CUDA_VISIBLE_DEVICES already set by coordinator). Calls `train_single_bucket()`, writes serializable result JSON to `--result-path`. Build `scripts/bucket_benchmark.py` — CLI with `<config.yaml> <fast|slow> [--gpus 0,1,2,3]` interface. Loads YAML via `yaml.safe_load()` → `prepare_config()`. Loads `data/hard_buckets/bucket_list.json` manifest. Creates timestamped output dir `data/hard_buckets/runs/YYYY-MM-DD_HHMM/`. Manages worker pool (copy of S01's pool pattern). Time limits: fast=60s, slow=3600s.

**T03: JSONL history tracking + comparison chart (~30min).** Add `plot_comparison_chart()` to `nce/benchmark/plots.py` — grouped bar chart with x-axis = bucket IDs, y-axis = final local error (log scale), two bars per bucket (current run blue, historical best orange). Handle first-run (no history → only current bars). After all workers complete, coordinator: merges temp results, builds JSONL record with config_hash + timing + per-bucket results, appends to `data/hard_buckets/history.jsonl`, reads history to find per-bucket best (filtered by `time_limit_per_bucket ≤ current`), generates comparison PNG.

**T04: End-to-end verification on 4 GPUs (~10min).** Run `python scripts/bucket_benchmark.py config.yaml fast --gpus 0,1,2,3` with a test YAML config. Verify: (a) per-bucket output folders with loss.png + local_error.png + metrics.json, (b) JSONL history entry appended, (c) comparison chart generated (current-only on first run). Run again to verify comparison chart shows current vs previous. Check multi-GPU: 4 buckets × 4 GPUs should complete in ~1 minute wall time.

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| Single-bucket training | `nce.benchmark.training.train_single_bucket()` (on `main`, 446 lines) | Full lifecycle: .pt load → reconstruct → train → checkpoint errors → plots → metrics. Verified via S02/T03 (8/8 checks pass). |
| Multi-GPU subprocess isolation | `scripts/select_hard_buckets.py:run_phase1()` lines 57–120 | Proven on this machine — poll-based, 1-per-GPU (D046), `CUDA_VISIBLE_DEVICES`, queue cycling. Handled OOM and failures gracefully. |
| Loss/error plots | `nce.benchmark.plots.{plot_loss_curve, plot_local_error_curve}` (on `main`) | Correct PNGs with Agg backend, semilogy scale, explicit `plt.close(fig)`. |
| Config validation + YAML | `nce.config_schema.prepare_config()` + `yaml.safe_load()` | Handles flat/nested detection, alias resolution, defaults. PyYAML 5.3.1 installed. |
| Checkpoint schedule | `nce.neural_networks.train.get_error_tracking_epochs()` | Tested: 0, 1, 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000, then every 5000. |
| Config hash | `nce.benchmark.training._write_metrics()` pattern | `hashlib.md5(json.dumps(sorted(items)))` — reuse for JSONL consistency. |
| Phase 2 precomputation | `scripts/select_hard_buckets.py:run_phase2()` | Exact upstream elimination + exact_fw + exact_bw + torch.save. Use via `--skip-phase1`. |
| Synthetic .pt for testing | `scripts/verify_benchmark_training.py:_generate_synthetic_pt()` | Creates valid .pt from smokers_20 when real data missing. |

## Existing Code and Patterns

### S01 Worker Pool Pattern (reuse verbatim for T02)
`scripts/select_hard_buckets.py:run_phase1()` — The exact pattern for multi-GPU coordination:
- `active = {}` dict keyed by GPU ID → `(problem_index, gpu_id, proc, output_path)`
- `gpu_queues = {g: deque() for g in gpus}` — round-robin assignment, sequential within each GPU
- Poll loop: `proc.poll()` for each active worker, `time.sleep(5)` between polls
- On worker finish: read stdout/stderr, launch next queued item, remove GPU from active if queue empty
- `_spawn_worker()`: sets `env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)`, uses `subprocess.Popen` with text pipes

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
    'bucket_id': str,  # e.g. "or_chain_10_fg_uai__bucket_88"
    'config_used': dict,
    'output_dir': str,   # per-bucket subfolder
    'loss_plot_path': str,  # optional
    'error_plot_path': str, # optional
    'metrics_path': str,    # optional
}
```
The function handles its own per-bucket output folder creation and plot/metrics generation. The subprocess worker just needs to serialize this result dict to JSON.

### S02 Verify Script Config Pattern (reuse for benchmark YAML config defaults)
`main:scripts/verify_benchmark_training.py` line 320: config built from `small_problems.configs['default'][0]` with overrides for loss_fn, hidden_sizes, lr, num_epochs, sampling_scheme. This pattern ensures all Trainer-required keys are present. The benchmark CLI should do similar: load user YAML, then merge onto a base from `small_problems.configs['default']` so no Trainer key is missing.

### Phase 1 Data Schema (from `/tmp/hard_bucket_selection_hvmlbbvv/`)
Each `problem_N.json`:
```python
{
    "problem_index": int,
    "problem_key": str,       # e.g., "smokers_20.uai"
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
```
Error entries (problems 18, 19, 21 — OOM): `{"problem_index": int, "error": str, "traceback": str}` with no `buckets` key.

### selection_results.json Schema (coordinator writes, --skip-phase1 reads)
```python
{
    "threshold": float,
    "selection_date": str,
    "num_problems": 24,
    "gpus": [int, ...],
    "results": [problem_N.json contents, ...]  # list of 24 entries
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
            "id": str,           # "or_chain_10_fg_uai__88"
            "problem_key": str,
            "bucket_label": int,
            "selection_error": float,
            "auto_ecl": int,
            "file": str,         # "or_chain_10_fg_uai__bucket_88.pt"
        }, ...
    ]
}
```

## Constraints

- **S02 files must be restored first.** Any `from nce.benchmark import ...` on the S03 branch fails with `ModuleNotFoundError`. Must run `git checkout main -- nce/benchmark/ scripts/verify_benchmark_training.py` before any implementation.

- **Phase 1 tmp data is ephemeral.** `/tmp/hard_bucket_selection_hvmlbbvv/` could be cleaned on reboot. Must persist to `selection_results.json` in T01 immediately.

- **Worker subprocess must not import torch at module level.** `CUDA_VISIBLE_DEVICES` must be set in subprocess env before torch import. Worker script must be standalone entry point with lazy imports (same pattern as S01's `select_hard_buckets_worker.py`).

- **Max 1 worker per GPU (D046).** Concurrent workers on same GPU cause OOM. Worker pool model mandatory.

- **Per-worker temp files + coordinator merge (D042).** Each worker writes to separate temp JSON. Coordinator merges after all workers finish. No concurrent JSONL writes.

- **Config YAML must be compatible with `prepare_config()`.** CLI accepts a standard flat or nested YAML config dict (NOT the experiment_runner format with `architectures:` lists — that's a different config schema).

- **`train_single_bucket()` requires all Trainer-init keys.** Config must include `approximation_method`, `debug`, `traced_losses`, `optimizer`, `lower_dim`, `sampling_scheme`, and all other keys from `small_problems.configs['default']`. Strategy: load user YAML, merge over a copy of `small_problems.configs['default'][problem_idx]`, let `train_single_bucket` apply its own overrides (ecl from .pt, sampling_scheme='all', error_tracking=False). This ensures no KeyError from Trainer.__init__.

- **4× NVIDIA TITAN RTX, 24GB each.** All 4 GPUs clean (verified: 0 processes, ~24GB free each).

- **Time limits: fast=60s, slow=3600s per bucket.** Checked at epoch boundaries in `train_single_bucket()`. Overshoot by at most one epoch is acceptable and documented.

- **PyYAML 5.3.1 installed.** Use `yaml.safe_load()`.

- **Python 3.11 in venv.** All scripts must use venv python or `sys.executable` for subprocess spawning.

## Common Pitfalls

- **Forgetting to restore S02 code before implementing.** Any `from nce.benchmark import ...` fails. This is step 1 of T01.

- **Tmp directory vanishing before persistence.** `/tmp/hard_bucket_selection_hvmlbbvv/` could be auto-cleaned. Must copy to permanent storage in T01 before any other work.

- **Worker config missing required Trainer keys.** If user's YAML omits `lower_dim`, `debug`, `traced_losses`, `optimizer`, etc., `Trainer.__init__` will KeyError. Must provide defaults by merging user config over `small_problems.configs['default']`. See D037 precedent.

- **JSONL append vs overwrite.** Must use `open('history.jsonl', 'a')` for appending. First run creates the file. Read with `[json.loads(line) for line in open(...)]`.

- **Comparison chart with no history.** First benchmark run has no previous data. Chart should show only current run's bars with a note, or coordinator should skip chart generation and print a message.

- **numpy/torch types in JSON serialization.** JSONL merge encounters torch scalars and numpy types. Need `_json_default` handler: `torch.Tensor → .item()`, `numpy.integer → int`, `numpy.floating → float`, `set → list`. The existing handler in `training.py` covers torch and set; must also handle numpy.

- **`log_z_err` may be a torch scalar.** The `error_tracking_data` tuples from `train_single_bucket()` may contain torch tensors. JSONL serialization needs explicit `.item()` conversion or the `_json_default` handler.

- **Stale GPU processes from crashed workers.** Subprocess isolation means GPU memory is reclaimed when the subprocess exits. But if coordinator is killed, workers may be orphaned. Add cleanup in coordinator's exception/signal handler (catch SIGTERM/SIGINT, kill all active workers).

- **Config hash drift.** The JSONL history entry's `config_hash` must match the per-bucket `metrics.json` config_hash for correlation. Both should use the same hash function on the same prepared config dict. The existing `_write_metrics()` hashes `sorted(nn_config.items())` — the coordinator should compute the hash before spawning workers (on the prepared config), and workers inherit it.

## Open Risks

- **Phase 2 runtime for grid10x10.** The largest hard bucket is `grid10x10.f5.wrap.uai` bucket 10. Exact upstream elimination and `get_backward_message(backward_ecl=2**30)` for this problem could use significant memory. Expected: fits in 24GB (auto_ecl for grid10x10.f5.wrap is ~2^15 = 32768 entries, well within memory). But monitor Phase 2 output.

- **3 OOM problems excluded from selection.** Problems 18, 19, 21 (deer_rescaled K20, K15, deer_rescaled K10.F2) failed Phase 1 with OOM. They might contain hard buckets. 4 hard buckets from 21 problems is sufficient (meets ≥3 threshold from M004 roadmap).

- **First-run comparison chart edge case.** The very first benchmark run has no history to compare against. The comparison chart function must handle this gracefully — show only current run's data or skip chart entirely.

- **Benchmark YAML config design.** The user needs a sample benchmark config YAML to actually run the CLI. Should create a default `configs/benchmark_default.yaml` with sensible defaults for the benchmark use case (UKL loss, [3,3] hidden, lr=0.01, num_epochs=100000, sampling_scheme=all).

## Skills Discovered

| Technology | Skill | Status |
|------------|-------|--------|
| PyTorch | `mindrally/skills@pytorch` (107 installs) | available — not needed |
| matplotlib | `ovachiever/droid-tings@matplotlib` (28 installs) | available — not needed |

No skills needed. This is integration/orchestration work using established codebase patterns.

## Sources

- S01 coordinator (worker pool pattern): `scripts/select_hard_buckets.py` — 460 lines, `run_phase1()` lines 57–120, `run_phase2()` lines 175–270
- S01 worker subprocess: `scripts/select_hard_buckets_worker.py` — 110 lines, lazy torch import pattern
- S02 training harness: `main:nce/benchmark/training.py` — 446 lines, `train_single_bucket()` API with 14-key return dict
- S02 plots: `main:nce/benchmark/plots.py` — 88 lines, `plot_loss_curve()`, `plot_local_error_curve()`
- S02 verification: `main:scripts/verify_benchmark_training.py` — 391 lines, synthetic .pt fallback, 8 structured checks
- S02 task summaries: `gsd/M004/S02:.gsd/milestones/M004/slices/S02/tasks/{T01,T02,T03}-SUMMARY.md`
- Phase 1 temp data: `/tmp/hard_bucket_selection_hvmlbbvv/` — 24 JSON files (21 success, 3 OOM), 4 hard buckets at threshold 0.1
- Experiment runner pattern: `notebooks/_1-2026/experiment_runner.py` — subprocess spawning with CUDA_VISIBLE_DEVICES
- Config schema: `nce/config_schema.py:prepare_config(config_dict, strict=False)` — line 411
- small_problems defaults: `small_problems.configs['default'][i]` — 39-key flat config dict with all Trainer-required keys
- Decision register: D042 (per-worker temp + merge), D046 (max 1 per GPU), D047 (custom epoch loop)
- GPU status verified: 4× TITAN RTX, all clean, 24GB free each
