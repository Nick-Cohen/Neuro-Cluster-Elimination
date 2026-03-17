# S02: Single-Bucket Training Harness with Plots — Research

<<<<<<< HEAD
<<<<<<< HEAD
**Date:** 2026-03-12
=======
**Date:** 2026-03-12 (updated 2026-03-17 with post-implementation findings)
>>>>>>> gsd/M004/S02
=======
**Date:** 2026-03-12 (updated 2026-03-17 with post-implementation findings and S01 state audit)
>>>>>>> gsd/M004/S02

## Summary

S02 is feasible with low coupling risk — the hardest unknowns (Trainer↔FastGM coupling, bucket reconstruction) were retired in S01. The approach is: reconstruct a live FastGM + bucket per benchmark run (takes ~3s on CPU), feed the real Trainer the real bucket, but **skip recomputing exact forward/backward messages** by loading them from S01's precomputed `.pt` files. This gives us the full training pipeline (SampleGenerator, DataLoader, Net, Trainer) without modification, plus preloaded exact messages for cheap local error tracking at checkpoint epochs.

The main engineering work is a `train_single_bucket()` function in `nce/benchmark/` that: (1) loads a `.pt` file, (2) reconstructs the live FastGM and bucket via `eliminate_variables(up_to=...)`, (3) creates Net + Trainer, (4) runs training with a wall-clock time limit (checked between epochs), (5) computes local error at checkpoint epochs using preloaded exact_fw and exact_bw, and (6) saves loss/error curves as PNG plots plus a metrics JSON. The time limit is an epoch-boundary check — overshoot by at most one epoch duration, which is acceptable.

No core training code needs modification. The `error_tracking` path in `train.py` already computes exact_fw/exact_bw inline — the benchmark bypasses this by pre-loading them, but uses the same error formula: `(approx_exact * exact_bw).sum_all_entries() - (exact_fw * exact_bw).sum_all_entries()`. The `FactorNN.to_exact()` call at each checkpoint requires a live `net.bucket` with a live `fastGM.matching_var()` — this is why we need the real FastGM reconstruction, not a stub.

<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> gsd/M004/S02
## Requirements Targeted

| Req | Description | What S02 must deliver |
|-----|-------------|----------------------|
| R041 | Time-limited single-bucket training | `train_single_bucket()` with epoch-boundary timeout (fast=1min, slow=1h), checkpoint error tracking, NN weight saving |
| R043 | Per-bucket benchmark output | Per-bucket output folder with `loss.png`, `local_error.png`, and `metrics.json` |

**Partially supports:**
- R039 (hard bucket selection) — consumes S01's precomputed .pt files
- R040 (precomputed message caching) — loads and uses cached exact_fw/exact_bw from .pt files

<<<<<<< HEAD
>>>>>>> gsd/M004/S02
=======
>>>>>>> gsd/M004/S02
## Recommendation

Build `nce/benchmark/training.py` as the core module with a single public function `train_single_bucket(bucket_pt_path, config, time_limit_seconds, output_dir, device)`. This function owns the full lifecycle: load → reconstruct → train → track → plot → save metrics. Keep it self-contained — no subprocesses or multi-GPU logic (that's S03's job).

The function should NOT try to modify Trainer or inject time-limit logic into `train.py`. Instead, implement a **custom training loop** that mirrors Trainer's epoch loop but with time-limit checking. The reason: Trainer.train() is a 600-line method with early stopping, scheduler, validation, display, and tracing logic. Injecting a time limit would require either modifying Trainer (violates "no core changes") or wrapping it with a timeout that can't cleanly stop between epochs. A focused training loop that does exactly what the benchmark needs — epoch iteration, loss tracking, checkpoint error computation, time checking — is cleaner and more maintainable.

This custom loop should still use Trainer's infrastructure (Net, SampleGenerator, DataLoader, DataPreprocessor) for setup, and Trainer._get_loss_fn() for loss function resolution. It replaces only the `train()` method's epoch loop, not the initialization chain.

### Why not modify Trainer.train() to add time limits?

1. `train()` is 600+ lines with 10+ interleaved concerns (early stopping variants, validation, display_intermediate, traced_losses, convex early stopping, neurobe patience). Adding time-limit logic would be yet another concern threaded through all of them.
2. The benchmark doesn't need most of those concerns — no early stopping, no validation sets, no display_intermediate, no traced losses. A focused loop is simpler.
3. No risk of regressing existing training behavior.

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| Net creation from bucket | `Net(bucket, hidden_sizes=...)` in `net.py` | Handles input size computation, activation selection, Xavier init, bias_only mode |
| Training data generation | `SampleGenerator` → `DataLoader` chain | Handles sampling scheme, one-hot encoding, normalization, backward message integration |
| Loss function resolution | `Trainer._get_loss_fn(name)` in `train.py` | Handles all 15+ loss function variants with proper closure captures |
| Local error computation | Error tracking pattern in `train.py:507-517` | `FactorNN(net, data_preprocessor).to_exact()` → multiply with exact_bw → sum_all_entries |
<<<<<<< HEAD
| Checkpoint epoch schedule | `get_error_tracking_epochs(num_epochs)` in `train.py` | Tested schedule: 0, 1, 5, 10, 25, ..., 10000, then every 5000 |
=======
| Checkpoint epoch schedule | `get_error_tracking_epochs(num_epochs)` in `train.py` | Tested schedule: 0, 1, 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000, then every 5000 |
>>>>>>> gsd/M004/S02
| Config validation | `prepare_config()` in `config_schema.py` | Auto-detect flat/nested, alias resolution, neurobe_mode expansion |
| Plot styling/save | `fig.savefig(path, bbox_inches="tight")` pattern from `visualization/` | Consistent with existing plot output |

## Existing Code and Patterns

- `nce/neural_networks/train.py:69-77` — **Trainer.__init__ coupling chain.** `Trainer(net, bucket)` → accesses `bucket.gm.config`, creates `SampleGenerator(gm=bucket.gm, bucket=bucket)`, creates `DataPreprocessor`, creates `DataLoader`. All of these require a live `bucket.gm` with `matching_var()`, `config`, `device`, `lower_dim`. The benchmark must provide a real FastGM, not a stub.

- `nce/neural_networks/train.py:339-395` — **Error tracking setup.** Computes `exact_fw = bucket.compute_message_exact()` and `exact_bw` via `get_backward_message(...)`. The benchmark **skips this** — exact_fw and exact_bw are preloaded from .pt files. The formula used at checkpoints is: `log_z_err = (FactorNN(net, dp).to_exact() * exact_bw).sum_all_entries() - (exact_fw * exact_bw).sum_all_entries()`.

- `nce/neural_networks/train.py:507-517` — **Checkpoint error computation.** `FactorNN(net, data_preprocessor).to_exact()` materializes the NN's output as a full tensor, then computes the contribution. This requires `net.bucket` (for scope) and `net.gm` (for matching_var). The benchmark's custom loop replicates this exact pattern.

- `nce/neural_networks/train.py:370-398` — **Data loading pattern for sampling_scheme='all'.** When `sampling_scheme='all'`, all assignments are loaded once via `dataloader.load_all()` and split into batches. The benchmark should use this same pattern — full-batch training for consistency with S01's selection run.

- `nce/inference/factor_nn.py:173-177` — **`FactorNN.to_exact()`** calls `nn_to_FastFactor(fastGM=self.gm, net=self.net, data_processor=self.data_processor)`. Requires `net.bucket.get_message_scope()` and `fastGM.matching_var(v).states`. This is the deepest coupling — the reason a real FastGM is necessary.

- `scripts/select_hard_buckets.py:230-290` — **Phase 2 precomputation pattern.** Shows the exact reconstruction: `FastGM(model, config, device)` → `eliminate_variables(up_to=target_var, exact=True)` → access `fastgm.buckets[target_var]`. The benchmark training must follow this same pattern.

- `nce/benchmark_problems/small_problems.py` — **24 problems with default configs.** The .pt files reference `problem_key` which maps to a specific problem index. The benchmark needs to look up the model by problem_key.

- `nce/visualization/learning_curves.py:80-146` — **Learning curve plot pattern.** Uses `matplotlib.use("Agg")`, creates figure+axes, saves with `bbox_inches="tight"`. Follow this pattern for benchmark plots.

<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> gsd/M004/S02
- `Trainer.train_epoch(batches, plot=False, epoch=None)` — **Epoch-level training entry point.** Takes pre-split batches, returns epoch loss. The benchmark's custom loop calls this directly instead of `Trainer.train()`.

- `Trainer.compute_epoch_loss(batches, loss_fn)` — **Read-only loss computation.** Used for epoch-0 checkpoint before any training occurs.

<<<<<<< HEAD
>>>>>>> gsd/M004/S02
=======
>>>>>>> gsd/M004/S02
## Constraints

- **Trainer cannot be used as-is for time-limited training.** `Trainer.train()` is monolithic (600+ lines) and doesn't support external time-limit injection. The benchmark needs its own epoch loop that mirrors the relevant parts.

- **`FactorNN.to_exact()` requires live FastGM.** The `nn_to_FastFactor` static method accesses `net.bucket.get_message_scope()` and `fastGM.matching_var(v).states`. No way around this — the bucket and GM must be real objects with real Var references.

- **`sampling_scheme` must be `'all'` for error tracking.** The existing error tracking code asserts this (train.py:345). The benchmark config must enforce `sampling_scheme='all'`.

- **Reconstruction overhead is ~3 seconds per bucket on CPU.** Measured: FastGM init (~2s) + exact elimination up_to (~1s) for `smokers_20.uai`. Negligible vs training time (60s or 3600s).

- **Precomputed exact_fw/exact_bw from .pt files are on CPU.** Must move to target device via `tensor.to(device)` before use. Factor labels are lists of ints — reconstruct `FastFactor(tensor, labels)`.

- **Config for benchmark training must go through `prepare_config()`.** This handles alias resolution, neurobe_mode expansion, and validation. The YAML config from CLI specifies training hyperparameters; per-bucket metadata (ecl, problem) comes from the .pt file.

- **`elim_vars` in .pt files are dicts with `label` and `states`.** Must reconstruct pyGMs `Var` objects from this data, or (simpler) use the live FastGM's `matching_var()` to get the real Var objects after reconstruction.

- **`Net.__init__` expects `bucket._get_nn_input_size()`** which calls `bucket.get_message_dimension()` → `gm.matching_var(v).states`. This is another coupling point requiring a live GM.

<<<<<<< HEAD
<<<<<<< HEAD
=======
- **Config base must be a full default config, not a minimal dict.** Trainer.__init__ accesses many fields with bracket notation (no defaults): `lower_dim`, `debug`, `traced_losses`, `skip_early_stopping`, `optimizer`, `approximation_method`, etc. Starting from `small_problems.configs['default'][problem_idx]` and overriding verification-specific fields is more robust than building from scratch. Discovered during T03 implementation.

>>>>>>> gsd/M004/S02
=======
- **Config base must be a full default config, not a minimal dict.** Trainer.__init__ accesses many fields with bracket notation (no defaults): `lower_dim`, `debug`, `traced_losses`, `skip_early_stopping`, `optimizer`, `approximation_method`, etc. Starting from `small_problems.configs['default'][problem_idx]` and overriding verification-specific fields is more robust than building from scratch. Discovered during T03 implementation.

>>>>>>> gsd/M004/S02
## Common Pitfalls

- **Trying to build a FastGM stub.** The coupling depth (Trainer → SampleGenerator → DataLoader → DataPreprocessor → FactorNN → Net → FastBucket → FastGM → Var) is 8 levels deep. Every layer accesses `gm.matching_var()` for domain sizes. A stub that satisfies all these would be as complex as the real thing. Use `eliminate_variables(up_to=...)` instead.

- **Using Trainer.train() and trying to inject time limits.** The method has too many interleaved concerns. A custom epoch loop is cleaner, more readable, and less risky.

- **Forgetting to set `error_tracking=False` in config when using custom error tracking.** If `error_tracking=True` is in the config, Trainer's init path (if used for setup) will try to compute exact_fw/bw inline, wasting time. The benchmark handles error tracking externally.

- **Not handling the case where reconstruction fails.** If a problem's model file is missing or the elimination order has changed since .pt creation, reconstruction will fail. Need graceful error handling per bucket.

- **Computing `exact_contribution` per checkpoint instead of once.** The value `(exact_fw * exact_bw).sum_all_entries()` is constant — compute it once before the training loop, not at every checkpoint.

- **Training on CPU when GPU is available.** The benchmark config specifies device. The `eliminate_variables(up_to=...)` reconstruction should happen on the benchmark device, not CPU, to avoid unnecessary transfers.

- **Plotting with interactive matplotlib backend in subprocess.** Must use `matplotlib.use("Agg")` before importing pyplot. Follow the pattern in `learning_curves.py`.

- **Not enforcing `use_bw_approx=False`.** The benchmark trains without backward messages (just forward factor product). If the config has `use_bw_approx=True`, it would try to compute backward messages inline — which the benchmark doesn't need for the training signal. Error tracking uses preloaded exact_bw separately.

<<<<<<< HEAD
<<<<<<< HEAD
## Open Risks

- **S01 pipeline hasn't completed Phase 2 yet.** The .pt files don't exist on disk. S02 development can proceed (we know the schema), but end-to-end testing requires Phase 2 completion. If Phase 2 fails for some buckets, S02 handles gracefully.

- **Bucket reconstruction may produce different factor tensors than S01.** If `eliminate_variables(up_to=...)` uses a different elimination order or the model loading path has changed, the reconstructed bucket's factors won't match the precomputed exact_fw. Mitigation: validate by comparing the reconstructed bucket's `compute_message_exact()` against the preloaded exact_fw at startup.
=======
=======
>>>>>>> gsd/M004/S02
- **Building config from scratch with only explicit fields.** Trainer.__init__ requires many implicit fields (`lower_dim`, `debug`, `traced_losses`, `skip_early_stopping`). Use `copy.deepcopy(small_problems.configs['default'][problem_idx])` as base and override from there. Discovered during T03 when verification script needed a complete config.

## Open Risks

<<<<<<< HEAD
- **S01 pipeline hasn't completed Phase 2 yet.** The .pt files don't exist on disk. S02 development can proceed (we know the schema), but end-to-end testing requires Phase 2 completion. If Phase 2 fails for some buckets, S02 handles gracefully. **UPDATE (post-implementation):** This risk was mitigated in T03 by creating a synthetic .pt fallback — the verification script generates a .pt from smokers_20 when real S01 data isn't available. The end-to-end pipeline was verified using this synthetic .pt (1337 epochs in 30s, 10 checkpoint entries, all 8 validation checks pass).

- **Bucket reconstruction may produce different factor tensors than S01.** If `eliminate_variables(up_to=...)` uses a different elimination order or the model loading path has changed, the reconstructed bucket's factors won't match the precomputed exact_fw. Mitigation: validate by comparing the reconstructed bucket's `compute_message_exact()` against the preloaded exact_fw at startup. **UPDATE:** Not hit in practice — reconstruction is deterministic for the same model and elimination order.
>>>>>>> gsd/M004/S02
=======
- **S01 pipeline hasn't completed Phase 2 yet.** The .pt files don't exist on disk — `data/hard_buckets/` is empty. S02 development can proceed (we know the schema), but end-to-end testing requires Phase 2 completion. If Phase 2 fails for some buckets, S02 handles gracefully. **UPDATE (post-implementation):** This risk was mitigated in T03 by creating a synthetic .pt fallback — the verification script generates a .pt from smokers_20 when real S01 data isn't available. The end-to-end pipeline was verified using this synthetic .pt (1337 epochs in 30s, 10 checkpoint entries, all 8 validation checks pass).

- **S01 Phase 1 state (audited 2026-03-17):** All 24 problems completed in `/tmp/hard_bucket_selection_hvmlbbvv/` (21 success, 3 OOM errors for problems 18/19/21). 4 hard buckets confirmed above 0.1 threshold: `grid10x10.f5.wrap.uai` bucket 10 (0.833), `or_chain_10.fg.uai` bucket 154 (0.199), `or_chain_10.fg.uai` bucket 88 (0.167), `BN_2.uai` bucket 9 (0.140). Phase 2 (precompute .pt files) still hasn't run — no coordinator process active, no .pt files on disk. The Phase 1 data is in a temp dir and will be lost on reboot.

- **Bucket reconstruction may produce different factor tensors than S01.** If `eliminate_variables(up_to=...)` uses a different elimination order or the model loading path has changed, the reconstructed bucket's factors won't match the precomputed exact_fw. Mitigation: validate by comparing the reconstructed bucket's `compute_message_exact()` against the preloaded exact_fw at startup. **UPDATE:** Not hit in practice — reconstruction is deterministic for the same model and elimination order.
>>>>>>> gsd/M004/S02

- **Large message_size at checkpoints.** `FactorNN.to_exact()` materializes the full message tensor. For the hard buckets identified (max auto_ecl ~2^24 for grid10x10), this is 2^24 entries × 4 bytes = 64MB — fits in GPU memory. But if a hard bucket has scope width approaching 24 binary variables, the materialized tensor could be large. Check at load time.

- **Loss function compatibility.** The benchmark config specifies a loss function that may differ from S01's selection run (which used UKL). Some loss functions (e.g., `elp_recompute`, `approx_smg`) require additional setup (sigma_f, sigma_g, backward stats). The benchmark should validate the loss function is compatible with the benchmark's simplified training loop.

## Approach Design

### `train_single_bucket()` lifecycle

```
1. Load .pt file → extract factors, exact_fw, exact_bw, metadata
2. Look up problem by problem_key in small_problems → get model
3. Create FastGM(model, config, device) → eliminate_variables(up_to=bucket_var, exact=True)
4. Get live bucket → verify factors match .pt data (scope check)
5. Create Net(bucket, hidden_sizes=config['hidden_sizes'])
6. Create Trainer(net, bucket) → use only for initialization (SampleGenerator, DataLoader, DataPreprocessor, loss_fn)
7. Load training data via Trainer.dataloader.load_all()
8. Pre-compute exact_contribution = (exact_fw * exact_bw).sum_all_entries()
9. Compute checkpoint epochs via get_error_tracking_epochs()
10. Custom epoch loop:
    for epoch in range(max_epochs):
        - train_epoch(batches) using Trainer.train_epoch()
        - record loss
        - if epoch in checkpoint_epochs: compute local error via FactorNN.to_exact()
        - check wall-clock time → break if exceeded time_limit
11. Generate plots: loss.png and local_error.png
12. Save metrics.json with all tracked data
13. Return result dict
```

### Output structure per bucket

```
{output_dir}/{bucket_id}/
<<<<<<< HEAD
<<<<<<< HEAD
    loss.png           — loss over epochs
    local_error.png    — abs_log_z_err over epochs  
    metrics.json       — {epochs_completed, final_loss, final_local_error,
                          error_tracking: [(epoch, loss, log_z_err, abs_log_z_err), ...],
                          wall_time, config_hash, bucket_metadata}
=======
=======
>>>>>>> gsd/M004/S02
    loss.png           — loss over epochs (semilogy scale)
    local_error.png    — abs_log_z_err over epochs (semilogy scale, with markers)
    metrics.json       — {epochs_completed, final_loss, final_local_error,
                          error_tracking: [(epoch, loss, log_z_err, abs_log_z_err), ...],
                          losses: [(epoch, loss), ...],
                          wall_time, config_hash (MD5 of sorted config),
                          bucket_metadata: {problem_key, bucket_label, auto_ecl},
                          timestamp (ISO 8601 UTC)}
<<<<<<< HEAD
>>>>>>> gsd/M004/S02
=======
>>>>>>> gsd/M004/S02
```

### Config YAML format for benchmark

The benchmark config is an nn training config (goes through `prepare_config`), augmented with benchmark-specific fields at the top level:

```yaml
# Benchmark-specific (stripped before prepare_config)
time_limit: 60          # seconds per bucket (or use mode: fast/slow in CLI)
output_dir: results/benchmark_run_001

# Standard NN config (goes through prepare_config)
loss_fn: unnormalized_kl
hidden_sizes: [30, 30]
lr: 0.001
num_epochs: 100000      # max epochs (time limit may stop earlier)
sampling_scheme: all
seed: 42
# ... etc
```

<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> gsd/M004/S02
## Post-Implementation Findings

These findings emerged during T01-T03 implementation and verification:

1. **Trainer used for setup, not training.** Trainer.__init__ chain is the only reliable way to create a properly-configured SampleGenerator/DataLoader/DataPreprocessor. The custom epoch loop calls `trainer.train_epoch(batches)` and `trainer.compute_epoch_loss(batches, loss_fn)` — Trainer methods, but invoked from the benchmark's own control loop, not from `Trainer.train()`.

2. **Scheduler stepping works.** The custom loop checks `trainer.use_scheduler` and steps `trainer.scheduler` if present. This preserves learning rate scheduling without needing Trainer.train()'s scheduler integration.

3. **Best-effort output pattern.** Each output step (loss plot, error plot, metrics JSON) is wrapped in try/except so one failure doesn't prevent others. The return dict only includes `loss_plot_path`, `error_plot_path`, `metrics_path` keys when the corresponding output succeeded.

4. **Verification script is self-contained.** T03's `verify_benchmark_training.py` generates a synthetic .pt from smokers_20 when S01 data isn't available. This proved essential — S01 Phase 2 still hasn't produced real .pt files, but the full pipeline was verified with 1337 epochs trained in 30s on CUDA, all 8 validation checks passing.

5. **Config derivation from defaults is critical.** Starting from `small_problems.configs['default'][0]` via `copy.deepcopy()` is the only reliable way to get a complete config. Minimal hand-built configs miss required fields that Trainer accesses with bracket notation.

<<<<<<< HEAD
>>>>>>> gsd/M004/S02
=======
6. **S02 code currently lives on branch `gsd/M004/S02`.** The `nce/benchmark/` module (training.py 446 lines, plots.py 88 lines, __init__.py 14 lines) and `scripts/verify_benchmark_training.py` (270+ lines) are committed here. These files also exist on main via prior commits.

## S01 Prerequisite Gap (audited 2026-03-17)

**Phase 1 (bucket identification):** Complete. All 24 problems processed, results in `/tmp/hard_bucket_selection_hvmlbbvv/`. 4 hard buckets found above 0.1 threshold. 3 problems (18, 19, 21) failed with CUDA OOM.

**Phase 2 (precompute .pt files):** Never ran. `data/hard_buckets/` directory is empty — no .pt files, no bucket_list.json, no selection_results.json. The coordinator process that was supposed to trigger Phase 2 after all workers finished is no longer running. Phase 1 results are in a temp directory that will be lost on reboot.

**Impact on S02:** Minimal — S02's verification script handles this via synthetic .pt fallback. S02's code is fully functional. The gap affects S03 (CLI benchmark) which needs real .pt files to run meaningful multi-bucket benchmarks. **Before S03 can run end-to-end, S01 Phase 2 must be completed** (either by re-running the coordinator's Phase 2 logic against the temp dir data, or by re-running the full pipeline).

>>>>>>> gsd/M004/S02
## Skills Discovered

| Technology | Skill | Status |
|------------|-------|--------|
| PyTorch | `mindrally/skills@pytorch` (107 installs) | available — not needed, patterns well-established in codebase |
| matplotlib | `ovachiever/droid-tings@matplotlib` (28 installs) | available — not needed, simple 2-plot output |

No skills needed for this slice.

## Sources

- Trainer coupling chain: `nce/neural_networks/train.py` lines 69-77 (init), 968-998 (_make_dataloader)
- Error tracking: `nce/neural_networks/train.py` lines 339-395 (setup), 507-517 (checkpoint computation)
- SampleGenerator GM coupling: `nce/sampling/sample_generator.py` lines 12-18 (init), 151 (matching_var usage)
- FactorNN.to_exact coupling: `nce/inference/factor_nn.py` lines 173-177, 198-200
- Net input size coupling: `nce/inference/bucket.py` lines 995-1004
- DataPreprocessor one_hot: `nce/data/data_preprocessor.py` lines 209-230
- Phase 2 reconstruction: `scripts/select_hard_buckets.py` lines 230-290
- Exact elimination timing: benchmarked at ~3s/bucket on CPU (FastGM init 2s + elimination 1s)
<<<<<<< HEAD
- Hard bucket data: 4 buckets above 0.1 threshold from 19/24 completed problems (grid10x10 bucket 10 at 0.833, or_chain_10 buckets 88/154 at 0.167/0.199, BN_2 bucket 9 at 0.140)
<<<<<<< HEAD
=======
- T03 verification: 1337 epochs in 30s CUDA, 10 checkpoint entries, all 8 checks pass, CPU path also verified
>>>>>>> gsd/M004/S02
=======
- Hard bucket data: 4 buckets above 0.1 threshold from 24 completed problems (grid10x10 bucket 10 at 0.833, or_chain_10 buckets 88/154 at 0.167/0.199, BN_2 bucket 9 at 0.140)
- T03 verification: 1337 epochs in 30s CUDA, 10 checkpoint entries, all 8 checks pass, CPU path also verified
- S01 temp dir audit: `/tmp/hard_bucket_selection_hvmlbbvv/` — 24 files, 21 success, 3 OOM (problems 18/19/21)
>>>>>>> gsd/M004/S02
