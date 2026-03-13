---
estimated_steps: 5
estimated_files: 2
---

# T02: Create nce/state/ module with save_state, load_state, and undo_normalization

**Slice:** S04 — FastGM State Preservation
**Milestone:** M001

## Description

Per D004, state preservation lives in a separate `nce/state/` module. This task creates the module with three functions: `save_state()` extracts a serializable dict from a post-inference FastGM, `load_state()` deserializes it, and `undo_normalization()` converts NN outputs back to log10 space using a saved normalizing constant. The module is a thin layer — it does not reconstruct a live FastGM, just preserves and restores inspection data.

## Steps

1. **Create `nce/state/__init__.py`**: Export `save_state`, `load_state`, `undo_normalization` from `nce.state.state`.

2. **Create `nce/state/state.py` with `save_state(fastgm, path, save_weights=False)`**:
   - Extract a dict with keys: `per_bucket_training_log` (the extended list from T01), `config` (the flat config dict), `logZ` (partition function estimate, if available — check `hasattr(fastgm, 'logZ')`), `elim_order` (list of variable labels), `num_trained` (int), `bucket_complexities` (list). For each key, use `getattr(fastgm, key, None)` with sensible defaults.
   - If `save_weights=False`, strip any `nn_state_dict` and `normalizing_constant` from training log entries before saving (they may be present if the config had `save_nn_weights=True` during inference). This ensures the metadata-only mode produces small files regardless of capture-time config.
   - Pickle the dict to `path` using `pickle.dump` with protocol 4 (widely compatible).
   - Wrap in try/except to catch `TypeError` (unpicklable objects) and re-raise with context message naming the problematic attribute.

3. **Implement `load_state(path) -> dict`**:
   - `pickle.load` from `path`. Return the dict.
   - Handle `FileNotFoundError` with a clear message.
   - Handle `pickle.UnpicklingError` with context about the path.

4. **Implement `undo_normalization(outputs, normalizing_constant) -> tensor`**:
   - Match the logic from `DataPreprocessor.undo_normalization()`: `outputs = outputs + normalizing_constant`, then `outputs = outputs / ln10` where `ln10 = torch.log(torch.tensor(10.0)).to(outputs.device)`.
   - Accept `normalizing_constant` as a Python float (as stored in T01) — convert to tensor on the outputs' device.
   - Return the tensor in log10 space.

5. **Verify module imports and function signatures**: Run import check and confirm all three functions are callable.

## Must-Haves

- [ ] `nce/state/__init__.py` exports `save_state`, `load_state`, `undo_normalization`
- [ ] `save_state` extracts a plain dict from FastGM and pickles it
- [ ] `save_state` with `save_weights=False` strips NN state dicts from training log entries
- [ ] `load_state` returns the dict from a pickle file
- [ ] `undo_normalization` matches DataPreprocessor.undo_normalization logic using a scalar constant
- [ ] Error handling: TypeError on save, FileNotFoundError and UnpicklingError on load

## Verification

- `python -c "from nce.state import save_state, load_state, undo_normalization; print('ok')"` — all three importable
- `python -c "import torch; from nce.state import undo_normalization; t = torch.tensor([1.0, 2.0]); r = undo_normalization(t, 5.0); print(r); assert r.shape == t.shape; print('undo ok')"` — undo_normalization works standalone

## Observability Impact

- Signals added/changed: `save_state` raises `TypeError` with attribute name if unpicklable object encountered
- How a future agent inspects this: `state = load_state(path); state.keys()` — plain dict, fully inspectable
- Failure state exposed: FileNotFoundError and UnpicklingError surface clearly on load; TypeError surfaces on save

## Inputs

- `nce/inference/graphical_model.py` — FastGM attributes to extract (per_bucket_training_log, config, logZ, elim_order, num_trained, bucket_complexities)
- `nce/data/data_preprocessor.py:157-172` — undo_normalization logic to replicate
- T01 output: extended per_bucket_training_log schema with losses, val_losses, optional nn_state_dict and normalizing_constant

## Expected Output

- `nce/state/__init__.py` — module init with exports
- `nce/state/state.py` — save_state, load_state, undo_normalization implementations
