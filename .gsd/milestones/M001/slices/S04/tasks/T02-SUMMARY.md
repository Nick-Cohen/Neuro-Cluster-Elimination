---
id: T02
parent: S04
milestone: M001
provides:
  - nce.state module with save_state, load_state, undo_normalization functions
key_files:
  - nce/state/__init__.py
  - nce/state/state.py
key_decisions:
  - Deep copy training log before stripping weights to avoid mutating the live FastGM object
  - Convert elim_order Var objects to plain int labels for pickle portability
patterns_established:
  - State extraction via getattr with sensible defaults — no tight coupling to FastGM internals
  - Error re-raising with context message naming the problematic attribute/path
observability_surfaces:
  - save_state raises TypeError with attribute context on unpicklable objects
  - load_state raises FileNotFoundError and pickle.UnpicklingError with path context
  - Loaded state is a plain dict — inspectable with state.keys() and standard Python
duration: 15m
verification_result: passed
completed_at: 2026-03-12
blocker_discovered: false
---

# T02: Create nce/state/ module with save_state, load_state, and undo_normalization

**Created `nce/state/` module with three functions for saving, loading, and denormalizing post-inference FastGM state.**

## What Happened

Built `nce/state/state.py` with:
- `save_state(fastgm, path, save_weights=False)` — extracts per_bucket_training_log, config, logZ, elim_order (as int labels), num_trained, bucket_complexities into a dict; deep-copies training log and strips nn_state_dict/normalizing_constant when save_weights=False; pickles with protocol 4.
- `load_state(path)` — unpickles and returns the dict with clear error handling.
- `undo_normalization(outputs, normalizing_constant)` — standalone version of DataPreprocessor.undo_normalization: adds constant back, divides by ln(10). Accepts float constant, handles device placement.

`nce/state/__init__.py` re-exports all three functions.

## Verification

- `python -c "from nce.state import save_state, load_state, undo_normalization; print('ok')"` — **passed**
- `python -c "import torch; from nce.state import undo_normalization; t = torch.tensor([1.0, 2.0]); r = undo_normalization(t, 5.0); assert r.shape == t.shape; print('undo ok')"` — **passed**, output `tensor([2.6058, 3.0401])` matches manual calculation
- Full round-trip with mock FastGM: save without weights (confirms stripping), save with weights (confirms round-trip), live object immutability, FileNotFoundError handling — **all passed**
- `python -c "import nce.state; print('state module importable')"` — **passed** (slice-level check)
- Slice verification script (`scripts/verify_s04_state_preservation.py`) — not yet created (T03 scope)

## Diagnostics

- `state = load_state(path); state.keys()` — returns dict with per_bucket_training_log, config, logZ, elim_order, num_trained, bucket_complexities
- `save_state` TypeError includes the unpicklable attribute name in the message
- `load_state` FileNotFoundError and UnpicklingError include the path

## Deviations

None.

## Known Issues

None.

## Files Created/Modified

- `nce/state/__init__.py` — module init exporting save_state, load_state, undo_normalization
- `nce/state/state.py` — implementations of all three functions with error handling
