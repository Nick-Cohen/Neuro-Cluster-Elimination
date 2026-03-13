# S04: FastGM State Preservation

**Goal:** After running inference with NN-trained buckets, a FastGM's full training history (loss curves, epochs, hidden sizes, optionally NN weights) can be saved to disk, loaded in a fresh session, and inspected — including undo-normalization of NN outputs.

**Demo:** Run `save_state(fastgm, "test.pkl")` after inference, then `state = load_state("test.pkl")` in a fresh context — `state['per_bucket_training_log'][0]['losses']` contains the full loss curve, and when weights are saved, `undo_normalization(output, state['per_bucket_training_log'][0]['normalizing_constant'])` converts NN output back to log10 space.

## Must-Haves

- Loss curves (list of `(epoch, loss_value)` tuples) captured in `per_bucket_training_log` before bucket deletion
- Validation losses captured when available (empty list when not)
- Optional NN weight capture via `save_nn_weights` config flag (default: off, per D005)
- `normalizing_constant` saved per bucket when weights are saved
- `nce/state/` module with `save_state()` and `load_state()` functions (per D004)
- Standalone `undo_normalization()` in state module using saved scalar (per R011)
- CUDA tensors moved to CPU before serialization for portability
- Round-trip pickle fidelity verified on a real NN-trained FastGM

## Proof Level

- This slice proves: integration (real NN training → capture → save → load → inspect)
- Real runtime required: yes (must run actual NN training to generate loss curves/weights)
- Human/UAT required: no

## Verification

- `python scripts/verify_s04_state_preservation.py` — end-to-end test: trains FastGM with NN buckets on a small problem, saves state (with and without weights), loads state, asserts loss curves present, weights round-trip, and undo_normalization produces correct values
- `python -c "import nce.state; print('state module importable')"` — module existence check

## Observability / Diagnostics

- Runtime signals: `per_bucket_training_log` entries now include `losses` and `val_losses` keys — a future agent can inspect `len(entry['losses'])` to verify capture worked
- Inspection surfaces: `load_state(path)` returns a plain dict — inspectable with standard Python dict operations, no special tooling needed
- Failure visibility: `save_state` raises `TypeError` with context if unpicklable objects encountered; `load_state` raises `FileNotFoundError` or `pickle.UnpicklingError` with the path
- Redaction constraints: none (no secrets in model state)

## Integration Closure

- Upstream surfaces consumed: `nce/inference/bucket.py` (capture site at line 328), `nce/inference/graphical_model.py` (FastGM attributes), `nce/neural_networks/train.py` (Trainer.losses, Trainer.val_losses), `nce/data/data_preprocessor.py` (normalizing_constant, undo_normalization logic)
- New wiring introduced in this slice: `nce/state/__init__.py` exports `save_state`, `load_state`, `undo_normalization`; bucket.py capture site extended with loss curves and optional weights
- What remains before the milestone is truly usable end-to-end: S05 (visualization) consumes the state dict to produce plots; S06 (logging) and S07 (regression test) are independent

## Tasks

- [ ] **T01: Extend per_bucket_training_log to capture loss curves and optional NN weights** `est:45m`
  - Why: Currently only `{label, epochs_trained, hidden_sizes}` is saved before bucket deletion. Loss curves and NN state are lost forever. This is the foundation for all state preservation — without it, there's nothing to save.
  - Files: `nce/inference/bucket.py`, `nce/inference/graphical_model.py`
  - Do: At the append site (bucket.py:328), add `losses: t.losses`, `val_losses: t.val_losses` to the dict. When `config.get('save_nn_weights', False)` is true, also add `nn_state_dict: {k: v.cpu() for k, v in net.state_dict().items()}` and `normalizing_constant: t.data_preprocessor.normalizing_constant.cpu().item()`. In graphical_model.py, update the comment on `per_bucket_training_log` to reflect the new schema.
  - Verify: `python -c "from nce.inference.bucket import FastBucket; print('import ok')"` — no syntax errors. Manual inspection of the diff confirms all fields captured before `del self.buckets[var]`.
  - Done when: `per_bucket_training_log` entries contain `losses` and `val_losses` keys, and optionally `nn_state_dict` + `normalizing_constant` when flag is set.

- [ ] **T02: Create nce/state/ module with save_state, load_state, and undo_normalization** `est:45m`
  - Why: D004 mandates state preservation as a separate module. This is the user-facing API for saving, loading, and inspecting FastGM state. Also delivers the standalone undo_normalization function (R011).
  - Files: `nce/state/__init__.py`, `nce/state/state.py`
  - Do: `save_state(fastgm, path, save_weights=False)` extracts a dict from FastGM attributes (`per_bucket_training_log`, `config`, `logZ`, `elim_order`, `num_trained`, summary metadata) and pickles it. `load_state(path)` unpickles and returns the dict. `undo_normalization(outputs, normalizing_constant)` is a standalone function that adds back the normalizing constant and divides by ln(10), matching DataPreprocessor.undo_normalization() logic. CUDA tensors in state_dicts already CPU'd at capture time (T01). Handle edge cases: empty training log, missing optional fields.
  - Verify: `python -c "from nce.state import save_state, load_state, undo_normalization; print('all exports ok')"` — module importable with all three functions.
  - Done when: `nce/state/` module exists with `save_state`, `load_state`, `undo_normalization` functions that handle both metadata-only and weights-included modes.

- [ ] **T03: Round-trip verification script on a real NN-trained problem** `est:1h`
  - Why: Proves the entire pipeline works end-to-end — train NNs, capture state, save, load, inspect. Without this, T01 and T02 are untested plumbing. This is the slice's objective stopping condition.
  - Files: `scripts/verify_s04_state_preservation.py`
  - Do: Write a verification script that: (1) loads a small benchmark problem (BN_1 from small_problems or grid10x10 from nbe_sanity_check — pick whichever has NN-eligible buckets with low ecl), (2) runs `eliminate_variables(all=True)` with a config that forces a few NN buckets (low ecl, 2-5 epochs for speed), (3) calls `save_state(fastgm, path)` in metadata-only mode, (4) calls `load_state(path)` and asserts: training log is non-empty, each entry has `losses` key with non-empty list, `epochs_trained` > 0, (5) repeats with `save_weights=True` and asserts: `nn_state_dict` present, `normalizing_constant` is a finite float, (6) tests `undo_normalization` with a dummy tensor and the saved constant — asserts output is finite and different from input, (7) prints PASS/FAIL summary. The script must handle the case where the benchmark problem needs to be downloaded or generated. Use 2-5 epochs and a low ecl to keep runtime under 60 seconds.
  - Verify: `python scripts/verify_s04_state_preservation.py` exits 0 and prints all assertions passing.
  - Done when: Script runs to completion, exercises both save modes, and all assertions pass.

## Files Likely Touched

- `nce/inference/bucket.py` — extend capture site
- `nce/inference/graphical_model.py` — update per_bucket_training_log comment
- `nce/state/__init__.py` — new module exports
- `nce/state/state.py` — save_state, load_state, undo_normalization
- `scripts/verify_s04_state_preservation.py` — end-to-end verification
