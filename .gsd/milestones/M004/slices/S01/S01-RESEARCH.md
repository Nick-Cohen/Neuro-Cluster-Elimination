# S01: Hard Bucket Selection & Precomputation — Research

**Date:** 2026-03-12

## Summary

This slice's job is to build `scripts/select_hard_buckets.py` — a one-time script that runs all 24 `small_problems`, identifies buckets where local error > threshold after 10000 epochs of UKL+bw training, and saves precomputed messages + factor tensors to `data/hard_buckets/` as `.pt` files.

The critical finding: **the existing `error_tracking` infrastructure in `Trainer.train()` does almost exactly what we need**. It already computes exact_fw and exact_bw per bucket, tracks `(epoch, loss, log_Z_err, abs_log_Z_err)` at checkpoint epochs, and stores results in `FastGM.error_tracking_data`. The selection script can run standard `FastGM.eliminate_variables(all=True)` with `error_tracking=True` in the config, then inspect `fastgm.error_tracking_data` to find hard buckets.

The second critical finding: **precomputation of cached bucket data requires a second pass**. After identifying hard buckets, the script must re-create a fresh FastGM and run `eliminate_variables(up_to=bucket_var)` with `exact=True` to get the target bucket into its final state (with upstream messages propagated as exact factors). Then it caches: the bucket's factors (as raw tensors), exact_fw, exact_bw, and metadata. This two-pass approach avoids modifying any core code — Phase 1 (selection) uses the existing training pipeline as-is, Phase 2 (precomputation) uses exact elimination to capture factor state.

The coupling depth between Trainer→SampleGenerator→bucket.gm is significant but manageable. SampleGenerator needs `gm.matching_var(v).states` (for domain sizes), `gm.config` (for fdb flag), and `gm.device`. Trainer needs `bucket.gm.config`, `bucket.gm._training_logger`, `bucket.gm.traced_losses_data`, and `bucket.gm.logSS`. All of these are satisfied by a real FastGM constructed from the model — no stubs needed for S01.

## Recommendation

**Two-phase approach within a single script:**

**Phase 1 — Selection run:** For each problem, create a FastGM with `error_tracking=True`, `use_bw_approx=True`, `bw_ecl=<training_bw_ecl>`, `loss_fn='unnormalized_kl'`, `sampling_scheme='all'`, `ecl=auto_ecl`, `num_epochs=10000`. Run `eliminate_variables(all=True)`. Inspect `fastgm.error_tracking_data` to identify buckets where final `abs_log_Z_err > threshold`. Distribute 24 problems across 4 GPUs using subprocess spawning with `CUDA_VISIBLE_DEVICES`.

**Phase 2 — Precomputation:** For each hard bucket identified in Phase 1, create a fresh FastGM from the same model. Run `eliminate_variables(up_to=matching_var(bucket_label), exact=True)` to propagate exact upstream messages. Then:
- `exact_fw = bucket.compute_message_exact()`
- `exact_bw, _ = get_backward_message(gm, bucket_label, iB=100, backward_ecl=2**30, return_factor_list=False)`
- Cache factor tensors, labels, exact_fw, exact_bw, and metadata to `data/hard_buckets/{problem_key}_{bucket_label}.pt`

Phase 2 is fast (seconds per bucket — exact elimination without NN training) and can run sequentially on a single GPU.

**Save format:** Each `.pt` file contains a dict with:
```python
{
    'factors': [{'tensor': t, 'labels': l} for each factor],
    'exact_fw': {'tensor': t, 'labels': l},
    'exact_bw': {'tensor': t, 'labels': l},
    'bucket_label': int,
    'scope': list[int],  # message scope
    'domain_sizes': list[int],
    'elim_vars': [{'label': int, 'states': int}],
    'problem_key': str,  # e.g. 'bn/BN_3'
    'model_file': str,  # e.g. 'BN_3.uai'
    'auto_ecl': int,
    'selection_error': float,  # abs_log_Z_err from selection run
    'selection_epochs': int,   # epochs completed in selection run
}
```

**Manifest:** `data/hard_buckets/bucket_list.json` lists all cached buckets with IDs, problem keys, local errors, and paths.

## Requirements Targeted

| Req | Description | How S01 Addresses It |
|-----|-------------|---------------------|
| R039 | Hard bucket selection precomputation | Full delivery — script identifies hard buckets across 24 small_problems |
| R040 | Precomputed message caching | Partial — caches exact_fw and exact_bw. Approximate bw at multiple ecl levels deferred to Phase 2 extension or S02 if needed |

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| Error tracking during training | `Trainer.train()` with `config['error_tracking'] = True` | Already computes exact fw/bw per bucket, tracks local error at checkpoints. No need to reimplement. |
| Checkpoint epoch schedule | `get_error_tracking_epochs(10000)` | Tested schedule: 0,1,5,10,25,50,100,200,500,1000,2000,5000,10000 |
| Backward message computation | `get_backward_message(gm, bucket_label, backward_ecl=2**30, return_factor_list=False)` | Handles all edge cases (scalar factors, WMB, factor lists) |
| Multi-GPU subprocess spawning | `experiment_runner.py` subprocess pattern | `CUDA_VISIBLE_DEVICES` isolation, per-GPU logging |
| Small problems with auto_ecl | `small_problems` benchmark set | 24 models with auto_ecl values, ready to iterate |
| Config validation | `prepare_config()` | Auto-detection, validation, alias resolution |

## Existing Code and Patterns

- **`nce/neural_networks/train.py:342–358`** — Error tracking setup. Reads `config['error_tracking']`, asserts `sampling_scheme='all'`, computes `exact_fw = bucket.compute_message_exact()` and `exact_bw` via `get_backward_message(..., backward_ecl=2**30)`. Tracks `(epoch, loss, log_Z_err, abs_log_Z_err)` at checkpoint epochs. The selection script enables this flag and reads `fastgm.error_tracking_data` after `eliminate_variables(all=True)`.

- **`nce/inference/bucket.py:380–381`** — After training, `compute_message_nn()` copies `t.error_tracking_data` to `self.gm.error_tracking_data.append((self.label, t.error_tracking_data))`. This is how per-bucket error data flows back to FastGM.

- **`nce/inference/graphical_model.py:245–320`** — `eliminate_variables(up_to=var)` runs elimination for all vars before `var` in elim_order. After this call, `self.buckets[var]` has its final factor list (upstream messages propagated). The `exact=True` flag forces `compute_message_exact()` for all upstream buckets regardless of ecl.

- **`nce/utils/backward_message.py:34–60`** — `get_backward_message()` creates a downstream FastGM from backward factors and eliminates all but the bucket scope. With `backward_ecl=2**30`, this is effectively exact for all small_problems (max induced width is ~24, and 2^30 >> 2^24).

- **`nce/benchmark_problems/small_problems.py`** — 24 problems, `_AUTO_ECL` dict, `set_bw_ecl()` helper. Default configs have `use_bw_approx=True` and `bw_ecl=0`. Selection script must call `set_bw_ecl(small_problems, 'default', <bw_ecl>)` or override per-config.

- **`nce/inference/graphical_model.py:57`** — `error_tracking_data` stored as `List of (bucket_label, [(epoch, loss, log_Z_err, abs_log_Z_err), ...])`.

- **`nce/inference/graphical_model.py:679–730`** — `get_large_message_buckets(iB, ecl)` identifies NN-eligible buckets. Useful during Phase 2 to verify which buckets were trained as NNs.

- **`notebooks/_1-2026/experiment_runner.py:252–295`** — `spawn_worker()` pattern: `env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)`, `subprocess.Popen(cmd, env=env, ...)`. Reuse this for distributing problems across 4 GPUs.

## Constraints

- **`error_tracking` requires `sampling_scheme='all'`** — Asserted at `train.py:344`. The default small_problems configs already use `'all'`, so this is satisfied.

- **`error_tracking` computes exact_bw inline during training** — The `get_backward_message()` call at `train.py:347` runs a full exact downstream elimination for each NN bucket at training start. This adds time but is already the existing pattern.

- **`eliminate_variables(up_to=var)` takes a `Var` object, not an int** — Must use `fastgm.matching_var(bucket_label)` to get the Var. The `up_to` parameter is compared via `self.elim_order.index(up_to)`.

- **`eliminate_variables` deletes upstream buckets** — After `eliminate_variables(up_to=var)`, all buckets before `var` are deleted (`del self.buckets[var]`). The target bucket's factors include upstream messages at this point. Must capture the target bucket state immediately.

- **Factor tensors in buckets may include FactorNN objects from upstream** — During Phase 2, using `exact=True` ensures all upstream messages are computed exactly, so the target bucket receives only `FastFactor` objects (not `FactorNN`).

- **`SampleGenerator` accesses `gm.matching_var(v).states` for domain sizes** — This requires a real FastGM with populated `self.vars`. No stub possible.

- **`Trainer` accesses `bucket.gm._training_logger`** — If `_training_logger` is None (no log_file in config), the accesses are guarded by `if self.bucket.gm._training_logger:`. Safe to leave as None.

- **Subprocess GPU isolation requires torch imported AFTER `CUDA_VISIBLE_DEVICES`** — Worker scripts must set env var before any torch import.

- **4× NVIDIA TITAN RTX, 24GB each** — Available per `nvidia-smi`. All 24 small_problems fit comfortably (max auto_ecl ~2^24 = 64MB for binary domains).

- **`bw_ecl` for training** — The selection config needs a non-zero `bw_ecl` to enable backward information during UKL training. The specific value affects training quality. Based on context, `bw_ecl = ecl` (same as forward ecl) is the most representative choice for each problem (matches the auto_ecl per-problem).

## Common Pitfalls

- **Forgetting to set `error_tracking=True` in config** — The key is `'error_tracking'` (not `'track_errors'`). `track_errors` is a different feature that tracks NN errors via `get_backward_message` in `process_bucket`. `error_tracking` is the Trainer-level feature that records checkpoint data.

- **Confusing `bw_ecl` with `backward_ecl` for error tracking** — `bw_ecl` controls the backward message quality used during *training* (the approximate backward info the loss function sees). `backward_ecl=2**30` in the error tracking code controls the *exact* backward message used for *evaluation*. These are independent: training can use approximate bw, while error tracking uses exact bw for ground truth.

- **Running Phase 1 sequentially** — 24 problems × 10000 epochs × 2–7 NN buckets = hours of GPU time. Must parallelize across 4 GPUs from the start. Round-robin assignment: 6 problems per GPU.

- **Caching FactorNN objects instead of raw tensors** — FactorNN objects contain references to Net and DataPreprocessor which are not serializable and fragile across code changes. Cache only raw tensors + labels.

- **Not handling the case where Phase 1 finds < 3 hard buckets** — The threshold (0.1 default) may be too high for some problems at auto_ecl. Script must report findings and allow threshold adjustment via CLI arg (D043).

- **Tensor device mismatch on load** — `torch.save()` preserves device. When loading on a different GPU via `CUDA_VISIBLE_DEVICES`, must use `torch.load(path, map_location='cuda:0')` or `map_location='cpu'`.

## Open Risks

- **How many hard buckets exist at threshold 0.1?** This is the primary risk S01 retires. The CSV shows 2–7 NNs per problem, with 24 problems = ~82 NN buckets total. We need at least 3–5 to have `abs_log_Z_err > 0.1` after 10000 epochs. If too few, we lower the threshold (configurable via `--threshold`).

- **Backward message exactness for all 24 problems** — `backward_ecl=2**30` with WMB should be exact for all small_problems (max auto_ecl ~2^24.2 for deer_rescaled K10.F2). But the *backward* induced width may differ from the forward. If any problem's backward elimination exceeds 2^30 entries per table (extremely unlikely for these sizes), the error tracking's "exact" backward is actually approximate. Phase 1 will reveal this via WMB partition warnings in the output.

- **Selection run wall-clock time** — ~82 NN buckets × 10000 epochs at `sampling_scheme='all'`. Per-epoch cost depends on message size. At auto_ecl, message sizes are bounded by ~2^24 entries. Full-batch training on a 2^24-entry message at 10000 epochs could take 10–30 minutes per bucket. Worst case: ~40 hours total. With 4 GPUs: ~10 hours. More realistically: most buckets have much smaller messages, so 2–6 hours total.

- **Phase 2 precomputation memory** — Computing exact_bw for a bucket with scope size 24 (2^24 = 16M entries) requires materializing that tensor. At 4 bytes per float32, that's 64MB — fits in GPU memory easily.

- **Approximate backward messages at multiple ecl levels (R040)** — R040 asks for approximate bw at `bw_ecl` levels (2^2, 2^3, 2^5, 2^10, 2^15, 2^25). This is straightforward to add in Phase 2 by calling `get_backward_message()` with each ecl level. Each call creates a downstream FastGM and eliminates — this takes seconds per bucket per ecl level. However, this adds 6× per-bucket precomputation. For 10 hard buckets, that's 60 additional backward message computations. If this scope is too large for S01, it can be deferred — exact bw is sufficient for the benchmark training in S02.

## Architecture Sketch

```
scripts/select_hard_buckets.py
├── Phase 1: Selection (parallel across 4 GPUs)
│   ├── worker subprocess per problem (CUDA_VISIBLE_DEVICES=N)
│   │   ├── Load model from small_problems catalog
│   │   ├── Create config with error_tracking=True, UKL, bw, auto_ecl, 10000 epochs
│   │   ├── FastGM(model=model, nn_config=config, device='cuda')
│   │   ├── fastgm.eliminate_variables(all=True)
│   │   └── Write error_tracking_data to temp JSON file
│   └── Coordinator: merge results, identify hard buckets (abs_log_Z_err > threshold)
│
├── Phase 2: Precomputation (sequential, fast)
│   ├── For each hard bucket:
│   │   ├── Fresh FastGM from same model
│   │   ├── eliminate_variables(up_to=matching_var(bucket_label), exact=True)
│   │   ├── exact_fw = bucket.compute_message_exact()
│   │   ├── exact_bw = get_backward_message(..., backward_ecl=2**30)
│   │   ├── [Optional: approx_bw at multiple ecl levels]
│   │   └── torch.save({factors, exact_fw, exact_bw, metadata}, path)
│   └── Write bucket_list.json manifest
│
└── Output: data/hard_buckets/
    ├── bucket_list.json
    ├── bn_BN_3__bucket_5.pt
    ├── segmentation_10_14__bucket_12.pt
    └── ...
```

## Task Decomposition (Advisory)

1. **T01: Selection worker script** — Write `scripts/select_hard_buckets_worker.py` that takes a problem index, config, and output path. Runs one problem through FastGM with error_tracking, writes results JSON. ~100 lines.

2. **T02: Selection coordinator + Phase 2 precomputation** — Write `scripts/select_hard_buckets.py` that spawns workers across GPUs, merges results, identifies hard buckets, runs Phase 2 precomputation, and writes the `bucket_list.json` manifest. ~200 lines.

3. **T03: Run the selection and verify** — Execute the script on 4 GPUs. Verify hard buckets found, precomputed data loads correctly, exact_fw/exact_bw tensors have expected shapes. Discord ping with results.

Alternatively, T01+T02 could be combined into a single task if the worker logic is kept inline (Phase 1 uses `subprocess` with a self-contained worker function).

## Skills Discovered

| Technology | Skill | Status |
|------------|-------|--------|
| PyTorch | `mindrally/skills@pytorch` (107 installs) | available — not needed, existing patterns sufficient |
| matplotlib | `ovachiever/droid-tings@matplotlib` (28 installs) | available — not needed for S01 (no plots) |

No skills are needed for this slice. The codebase has well-established patterns for all required operations.

## Sources

- Error tracking pattern: `nce/neural_networks/train.py` lines 342–358, 508–517
- Error data flow: `nce/inference/bucket.py` lines 380–381
- Backward message: `nce/utils/backward_message.py` full file
- Elimination: `nce/inference/graphical_model.py` lines 245–320
- Small problems: `nce/benchmark_problems/small_problems.py` full file
- NN bucket counts: `notebooks/_1-2026/problem_ecl_values.csv`
- Multi-GPU pattern: `notebooks/_1-2026/experiment_runner.py` lines 252–380
- GPU environment: `nvidia-smi` — 4× NVIDIA TITAN RTX, 24GB each
