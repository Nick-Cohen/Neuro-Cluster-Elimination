# S04: FastGM State Preservation — Research

**Date:** 2026-03-12

## Summary

FastGM already pickles cleanly after elimination — verified on a toy problem with exact elimination. After `eliminate_variables(all=True)`, all buckets are deleted (`del self.buckets[var]` at graphical_model.py:309), leaving only scalar attributes, the config dict, pyGMs `Var` objects, and lists. Both `Var` and `torch.Tensor` pickle fine. The `per_bucket_training_log` survives on FastGM but currently only stores `{label, epochs_trained, hidden_sizes}` — loss curves live in `Trainer.losses` (list of `(epoch, loss_value)` tuples) and get passed to `FactorNN` but are lost when the bucket is deleted.

The core work is: (1) extend the training log append in bucket.py to capture loss curves and optionally NN weights + DataPreprocessor state before bucket deletion, (2) create the `nce/state/` module with `save_state()`/`load_state()` that extracts a clean serializable dict from FastGM, and (3) ensure round-trip pickle fidelity on a real NN-trained problem.

## Recommendation

**Capture-at-source + thin state module.** Extend `per_bucket_training_log` entries at the existing append site (bucket.py:329) to include `losses`, `val_losses`, and optionally `nn_state_dict` + `data_preprocessor_state`. The `nce/state/` module should be a thin layer: `save_state()` extracts a plain dict from FastGM attrs, `load_state()` returns that dict (not a reconstructed FastGM — reconstruction requires the original model/UAI file). This keeps the module simple and avoids the circular-reference pitfalls of pickling live NN objects.

For R011 (undo-normalization), store the `normalizing_constant` scalar from `DataPreprocessor` in the per-bucket log when weights are saved. A standalone `undo_normalization(output_tensor, normalizing_constant)` function in the state module can convert NN outputs back to log10 space without needing the original DataPreprocessor instance.

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| Serialization format | `pickle` (stdlib) | FastGM already pickles; torch state_dicts pickle fine; no need for custom format |
| NN weight serialization | `net.state_dict()` + `pickle` | Standard PyTorch pattern; `torch.save`/`torch.load` is just pickle underneath |
| pyGMs Var pickling | Built-in `__getstate__` on Var | Verified: `pickle.dumps(Var(0,2))` round-trips correctly |

## Existing Code and Patterns

- `nce/inference/bucket.py:328-334` — The **capture site**: `per_bucket_training_log.append({...})` runs right after training, before bucket deletion. Extend this dict to include `t.losses`, `t.val_losses`, and optionally `net.state_dict()` + preprocessor state.
- `nce/inference/bucket.py:340` — `FactorNN(net, t.data_preprocessor, losses=t.losses)` — losses are already passed here; we just need to also push them to the training log.
- `nce/neural_networks/train.py:80-81` — `self.losses = []` and `self.val_losses = []` — list of `(epoch, loss_value)` tuples, populated during training loop.
- `nce/data/data_preprocessor.py:156-172` — `undo_normalization()` method: adds back `normalizing_constant`, divides by `ln10`. Only needs the scalar `normalizing_constant` to function.
- `nce/neural_networks/net.py:17-18` — `self.bucket = bucket; self.gm = self.bucket.gm` — Net holds circular refs to bucket and FastGM. **Never pickle a live Net object**; always extract `state_dict()` first.
- `nce/inference/graphical_model.py:57` — `self.per_bucket_training_log = []` — initialized in `__init__`, survives elimination.
- `nce/inference/graphical_model.py:309` — `del self.buckets[var]` — bucket destruction point. All NN state must be captured before this line.
- `claude_files/test_ukl_vs_scaled_mse.py` — Existing pattern of pickling experiment results dicts. Follow this style for state serialization.

## Constraints

- **Net has circular references** (`Net → bucket → FastGM → buckets → ... → Net`). Cannot pickle live Net objects. Must extract `state_dict()` (a plain OrderedDict of tensors) before serialization.
- **DataPreprocessor references FastBucket** via `one_hot_encode()` method, but only during training. For undo-normalization, only `normalizing_constant` (a scalar tensor) is needed.
- **Buckets are deleted during elimination** — all state capture must happen at the existing append site in `compute_message_nn()` (bucket.py:328-334), before `del self.buckets[var]` at graphical_model.py:309.
- **CUDA tensors in state dicts** — `state_dict()` contains CUDA tensors if training was on GPU. Must `.cpu()` them before pickling for portability (pickle works but unpickling requires same CUDA setup otherwise).
- **`nce/state/` module** — D004 mandates this as a separate module, not embedded in FastGM internals.
- **Default save mode** — D005: training metadata only by default, NN weights optional via flag.
- **Loss curve capture** — D008: extend per_bucket_training_log to include full loss curves before bucket deletion.

## Common Pitfalls

- **Pickling CUDA tensors then loading on CPU-only machine** — Move state_dict tensors to CPU before saving. Use `map_location='cpu'` on load if using `torch.load`.
- **Circular references in Net** — Never attempt `pickle.dumps(net)`. Always use `net.state_dict()` which is a plain dict of tensors with no back-references.
- **Large state dicts bloating save files** — A typical NN with `hidden_sizes=[64, 64]` and ~20 input features has ~5K parameters (~20KB). With 50+ NN buckets, weights add ~1MB. Manageable, but flag should default to off (D005).
- **DataPreprocessor.normalizing_constant is a scalar tensor on device** — Must `.cpu().item()` or `.cpu()` it before saving to avoid CUDA dependency on load.
- **Assuming FastGM can be reconstructed from saved state** — It can't without the original model/UAI file. The state module saves inspection data (loss curves, weights, metadata), not a full reconstructable snapshot. Document this clearly.
- **`val_losses` may be empty** — Validation losses are only populated when validation is configured. Handle gracefully (save empty list).

## Open Risks

- **NN training on real problems is slow** — Even 2-epoch training on `nbe_sanity_check` problem 0 (1077 vars) timed out at 180s. Verification testing needs either a very small problem with NN-eligible buckets, or patience. Consider using `small_problems` with low `ecl` to force a few NN buckets on a tractable problem.
- **`stats` attribute on FastGM** — `self.stats` is passed externally and could contain anything. If it holds unpicklable objects, the whole pickle fails. Need to test or handle gracefully (skip if unpicklable, or document constraint).
- **Future attribute additions** — Any new attribute added to FastGM that holds a live torch module or circular reference will break pickling silently. Consider adding a smoke test that pickles FastGM after NN elimination.

## Skills Discovered

| Technology | Skill | Status |
|------------|-------|--------|
| PyTorch | N/A | No specialized skill needed — standard state_dict/pickle patterns |
| pickle | N/A | stdlib, well-understood |

## Sources

- Verified pickle behavior via live testing in venv (pyGMs Var, torch state_dict, torch Tensor, FastGM post-elimination)
- Code audit of bucket.py, graphical_model.py, train.py, factor_nn.py, data_preprocessor.py, net.py
