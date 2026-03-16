# M004: Single-Bucket Learning Benchmark — Research

**Date:** 2026-03-15

## Summary

The benchmark harness is feasible with the existing codebase — the critical building blocks (error tracking, backward message computation, multi-GPU subprocess spawning, `prepare_config()`) all exist and are battle-tested. The main engineering challenge is **decoupling single-bucket training from the full elimination pipeline**. Today, `Trainer` is tightly coupled to a live `FastBucket` (which references a live `FastGM`), and a bucket's factors only reach their final state after upstream messages have been propagated via `eliminate_variables(up_to=...)`. The benchmark must either (a) run the full forward elimination up to each target bucket as a precomputation step and cache the resulting factor tensors, or (b) refactor Trainer to accept pre-materialized tensors directly. Option (a) is strongly preferred — it requires no changes to core training code and the precomputation cost is paid once.

The second challenge is **bucket selection cost**. Running all 24 problems × 10000 epochs with error tracking to find "hard" buckets is expensive. The existing `error_tracking` infrastructure in `train.py` computes exact forward and backward messages per bucket and tracks `(epoch, loss, log_Z_err, abs_log_Z_err)` at checkpoint epochs — this is exactly the pattern needed. But running 24 full inference runs at 10000 epochs will take hours. This must be a one-time precomputation with cached results, and it should be the first slice to prove out.

The recommended approach: **three slices, risk-ordered**. S01: bucket selection + precomputation (highest risk — if we can't identify hard buckets or cache messages, the rest is moot). S02: single-bucket training harness with time limits, error tracking, and plots (the core benchmark logic). S03: multi-GPU CLI, JSONL history, and historical comparison (integration and polish). The existing `experiment_runner.py` subprocess pattern with `CUDA_VISIBLE_DEVICES` isolation should be reused for S03.

## Recommendation

Build the benchmark as a standalone `scripts/` module, not integrated into the inference pipeline. The bucket selection and benchmark training are *evaluation tools*, not modifications to the core algorithm. Key patterns to reuse:

- **Subprocess GPU isolation** from `experiment_runner.py` — one worker per GPU via `CUDA_VISIBLE_DEVICES`
- **`error_tracking` pattern** from `train.py` lines 339–395 — precompute exact_fw and exact_bw, track `(epoch, loss, log_Z_err, abs_log_Z_err)` at checkpoints
- **`prepare_config()` from `config_schema.py`** — YAML config → flat dict
- **`small_problems` benchmark set** — 24 problems with auto_ecl values, ready to use
- **`get_backward_message()` with `backward_ecl=2**30`** — "effectively exact" backward messages for these problem sizes

The precomputed bucket data should be saved as a directory of `.pt` files (one per bucket) containing: factor tensors, exact forward message, exact backward message, bucket metadata (label, scope, domain sizes, elim_vars). This avoids re-running the elimination pipeline on every benchmark invocation.

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| Multi-GPU process spawning | `experiment_runner.py` subprocess pattern | Battle-tested, clean CUDA isolation via `CUDA_VISIBLE_DEVICES`, wave-based scheduling |
| Config validation + YAML parsing | `prepare_config()` + PyYAML (installed) | Already handles flat/nested detection, neurobe_mode expansion, alias resolution |
| Error tracking at checkpoints | `train.py` error_tracking code (lines 339–395) | Computes exact fw/bw messages, tracks local error at sparse checkpoint epochs |
| Backward message computation | `get_backward_message()` in `backward_message.py` | Handles all edge cases (scalar factors, WMB partitions, factor list mode) |
| Learning curve plots | `nce/visualization/learning_curves.py` | Extracts training log, generates per-bucket subplots |
| Checkpoint epoch schedule | `get_error_tracking_epochs()` | Tested schedule: 0, 1, 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000, then every 5000 |

## Existing Code and Patterns

- `nce/neural_networks/train.py:339–395` — **Error tracking setup pattern.** Computes `exact_fw = bucket.compute_message_exact()` and `exact_bw` via `get_backward_message(..., backward_ecl=2**30)`, then `exact_contribution = (exact_fw * exact_bw).sum_all_entries()`. At checkpoint epochs: `approx_exact = FactorNN(net, data_preprocessor).to_exact()`, `approx_contribution = (approx_exact * exact_bw).sum_all_entries()`, `log_z_err = approx_contribution - exact_contribution`. This is the exact local error computation the benchmark needs.
- `nce/neural_networks/train.py:16–32` — **`get_error_tracking_epochs()`** generates the checkpoint schedule. Benchmark can reuse or customize.
- `nce/inference/bucket.py:67–400` — **`compute_message_nn()`** is the complete single-bucket training flow: creates Net, creates Trainer, calls `t.train()`, captures `per_bucket_training_log`, creates `FactorNN`. The benchmark's single-bucket trainer needs to replicate this flow but with preloaded factors rather than a live bucket.
- `nce/inference/bucket.py:12–35` — **`FastBucket.__init__`** takes `(gm, label, factors, device, elim_vars)`. The `gm` reference is used by `Trainer` → `SampleGenerator` → `DataLoader`. This coupling is the main obstacle to isolated bucket training.
- `nce/utils/backward_message.py` — **`get_backward_message()`** with `backward_ecl=2**30` and `approximation_method='wmb'` gives "effectively exact" backward messages. The `return_factor_list=False` path materializes the full product tensor — needed for local error computation.
- `nce/benchmark_problems/small_problems.py` — **24 problems with `_AUTO_ECL` values.** Ready to iterate. `set_bw_ecl()` helper for bulk config updates.
- `notebooks/_1-2026/experiment_runner.py` — **Multi-GPU orchestration pattern.** `assign_experiments_to_gpus()` (round-robin), `spawn_worker()` (subprocess with `CUDA_VISIBLE_DEVICES`), wave-based `run_experiments()`. Reuse the subprocess pattern but simplify — benchmark workers are simpler (one bucket, not a full experiment combination).
- `nce/inference/graphical_model.py:679–730` — **`get_large_message_buckets()`** identifies NN-eligible buckets by `iB` and `ecl` thresholds. Useful during bucket selection to find which buckets were trained as NNs.
- `nce/inference/graphical_model.py:245–440` — **`eliminate_variables(up_to=...)`** runs elimination up to (but not including) a variable. After this call, the target bucket's factors include all upstream messages. This is how precomputation must prepare each bucket.
- `nce/config_schema.py:411–470` — **`prepare_config()`** auto-detects flat/nested, validates, expands neurobe_mode. The benchmark CLI should load YAML, pass through `prepare_config()`.
- `nce/state/state.py` — **State serialization pattern.** Pickle-based, training metadata extraction. The benchmark's precomputed bucket data should follow a similar pattern but using `torch.save()` for GPU tensor serialization.
- `nce/inference/graphical_model.py:57` — **`error_tracking_data`** stored as `List of (bucket_label, [(epoch, loss, log_Z_err, abs_log_Z_err), ...])`. Benchmark's JSONL history should include this same tuple format.

## Constraints

- **FastBucket ↔ FastGM coupling.** `FastBucket` references `self.gm` for config, device, `matching_var()`, `_training_logger`, and `per_bucket_training_log`. `Trainer` → `SampleGenerator` uses `bucket.gm` to access the graphical model's variable info. Isolated bucket training requires either (a) providing a minimal FastGM stub that satisfies these references, or (b) running `eliminate_variables(up_to=bucket_var)` to get the bucket into its correct state with a real FastGM. For precomputation: option (b) is mandatory. For benchmark re-training: option (a) is viable if we cache enough metadata alongside the factor tensors.
- **`compute_message_exact()` requires all factors to have materialized tensors.** FactorNN objects in upstream messages have tensors that are lazily computed. During precomputation, all upstream elimination must complete so the target bucket has only concrete `FastFactor` objects.
- **`error_tracking` requires `sampling_scheme='all'`.** This is already asserted in `train.py:345`. The benchmark config must enforce this.
- **`to_exact()` materializes the full message tensor.** For large messages (ecl > 2^20), this can consume significant GPU memory. The benchmark's selected buckets should have message sizes that fit in GPU memory for both the NN output and the exact comparison.
- **`get_backward_message()` runs a separate `eliminate_variables(all_but=bucket_scope)` on a downstream GM copy.** This is expensive for the first computation but the result is cached as a precomputed tensor. At `backward_ecl=2**30`, the downstream GM does exact elimination (no WMB partitions), which is only feasible if the backward problem's induced width is manageable.
- **Subprocess GPU isolation requires careful import ordering.** `torch` must be imported AFTER `CUDA_VISIBLE_DEVICES` is set (as documented in `worker.py`). Worker scripts must follow this pattern.
- **PyYAML 5.3.1 installed.** Can use `yaml.safe_load()` for config parsing. No need for additional YAML dependencies.
- **4× NVIDIA TITAN RTX (24GB each).** Sufficient for all 24 small_problems. The largest auto_ecl is ~2^24 (deer_rescaled K10.F2), which at binary domains = 2^24 entries × 4 bytes = 64MB — fits comfortably.

## Common Pitfalls

- **Running bucket selection as 24 sequential full-inference runs.** At 10000 epochs × ~2-7 NN buckets each, this is 10-40 minutes per problem on GPU — potentially 4-16 hours total. Must be parallelized across 4 GPUs from the start, and should be a one-time step with persistent cache.
- **Confusing "local error" computation context.** The error tracking in `train.py` computes local error *during* training (inside the elimination pipeline, where backward messages are computed inline). The benchmark needs to precompute both forward and backward messages *before* training starts, then pass them to the training loop. This is a different execution order than the existing code.
- **Assuming bucket factors are stable before elimination.** A bucket's factor list changes as upstream messages arrive during `eliminate_variables()`. The precomputation must run `eliminate_variables(up_to=bucket_var)` to get the bucket into its final state before caching its factors.
- **Forgetting that `eliminate_variables` destroys upstream buckets.** After `eliminate_variables(up_to=var)`, all buckets before `var` in the elimination order are deleted (`del self.buckets[var]`). The precomputation must capture the target bucket's state before calling any further elimination.
- **Time-limited training overshooting.** Checking wall-clock time only between epochs means the last epoch could push over the time limit. For "fast" mode (1 min), a single epoch on a large message could take seconds to minutes. The overshoot is bounded by one epoch duration — acceptable but should be documented.
- **JSONL history file corruption on concurrent writes.** If multiple GPUs finish simultaneously and try to append to the same JSONL file, writes could interleave. Use file locking (e.g., `fcntl.flock`) or have each worker write a separate file that the coordinator merges.
- **Precomputed tensor device mismatch.** `torch.save` saves tensors on their current device. When loading on a different GPU (via `CUDA_VISIBLE_DEVICES`), must use `torch.load(..., map_location='cuda:0')` or move to the target device.

## Open Risks

- **Backward message exactness for all 24 problems.** The context mentions "at least one problem where exact backward solvability may not hold." `get_backward_message()` with `backward_ecl=2**30` uses WMB with effectively infinite ecl — this should produce exact results for all problems in the small set (max auto_ecl is ~2^24). But if a problem's *backward* induced width exceeds what WMB can handle, the "exact" backward message is actually approximate. Need to verify per-problem during bucket selection.
- **Number of hard buckets.** The requirement says "up to 10 buckets with local error > 0.1." If fewer than 3 problems have hard buckets at this threshold, the benchmark may not be useful. The threshold may need tuning — suggest making it configurable with 0.1 as default.
- **Bucket selection GPU time.** Running all 24 problems × 10000 epochs with `error_tracking=True` means computing exact_fw and exact_bw for every NN bucket during training, plus `to_exact()` at every checkpoint. This roughly doubles training time. Even parallelized on 4 GPUs, expect 2-8 hours for the full selection run.
- **Precomputed data size.** Each cached bucket includes: factor tensors (varies, but typically KB to MB), exact forward message (up to 2^24 × 4B = 64MB), exact backward message (same). For 10 hard buckets, total cache is likely 100MB-1GB — manageable.
- **Trainer coupling depth.** The `Trainer.__init__` creates a `SampleGenerator` that references `bucket.gm` for variable domain info, elimination order, and device. Building a minimal "stub" FastGM that satisfies all these references without running real elimination is tricky — there may be deep attribute accesses that require real pyGMs `Var` objects. The safest approach is to reconstruct a real FastGM from the original model and run `eliminate_variables(up_to=...)` during each benchmark run. But this adds per-run overhead (seconds, not minutes — elimination without NN training is fast for exact buckets).

## Candidate Requirements (Advisory)

These are observations from research that may warrant requirements but should not auto-expand scope:

- **CR-01: Configurable hardness threshold.** The 0.1 local error threshold is hardcoded in R039. Consider making it a parameter of the bucket selection script (default 0.1) so researchers can experiment with different thresholds without modifying code.
- **CR-02: Bucket warm-up reconstruction.** Each benchmark run must reconstruct the target bucket's factor state by running exact elimination up to that bucket. This takes seconds per bucket but should be documented and profiled. Could become a requirement if overhead is significant.
- **CR-03: JSONL file locking.** R044 specifies a single JSONL history file. With multi-GPU parallel writes, file corruption is possible. Should specify the concurrency strategy (per-worker files + coordinator merge, or fcntl locking).
- **CR-04: Cache invalidation.** Precomputed messages in R040 are tied to specific (problem, ecl, bw_ecl) combinations. If a user changes ecl values or the elimination order changes, the cache is stale. Should there be a cache version/hash mechanism, or is manual re-running of the selection script sufficient?
- **CR-05: Dry-run mode.** A `--dry-run` flag that lists selected buckets and estimated runtime without training would be useful for verification.

## Skills Discovered

| Technology | Skill | Status |
|------------|-------|--------|
| PyTorch | `mindrally/skills@pytorch` (107 installs) | available |
| PyTorch research | `tondevrel/scientific-agent-skills@pytorch-research` (6 installs) | available |
| matplotlib | `ovachiever/droid-tings@matplotlib` (28 installs) | available |

None are essential for this milestone — the codebase already has well-established PyTorch and matplotlib patterns. The pytorch skill could be useful if unfamiliar GPU memory management patterns arise during implementation.

## Sources

- Error tracking pattern: `nce/neural_networks/train.py` lines 339–395, 508–517
- Backward message computation: `nce/utils/backward_message.py` (full file)
- Multi-GPU subprocess pattern: `notebooks/_1-2026/experiment_runner.py` (full file)
- Bucket training flow: `nce/inference/bucket.py` lines 67–400
- Small problems benchmark set: `nce/benchmark_problems/small_problems.py` (24 models, auto_ecl values)
- Config schema: `nce/config_schema.py` `prepare_config()` at line 411
- Auto ECL data: `notebooks/_1-2026/problem_ecl_values.csv` (2–7 NNs per problem at auto_ecl)
- GPU environment: `nvidia-smi` — 4× NVIDIA TITAN RTX, 24GB each
