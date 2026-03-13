---
id: T01
parent: S04
milestone: M001
provides:
  - per_bucket_training_log entries now include full loss curves and optional NN weights
key_files:
  - nce/inference/bucket.py
  - nce/inference/graphical_model.py
key_decisions:
  - Loss curves stored as-is (list of (epoch, loss_value) tuples) — plain Python, no serialization concerns
  - NN weights gated behind save_nn_weights config flag to avoid memory overhead by default
  - .cpu().clone() on state_dict values ensures no CUDA tensor leaks into the log
  - normalizing_constant stored as Python float via .cpu().item()
patterns_established:
  - Config-gated optional capture: check self.config.get('save_nn_weights', False) before adding heavy data
observability_surfaces:
  - per_bucket_training_log entries gain 'losses' and 'val_losses' keys — inspect via len(fastgm.per_bucket_training_log[i]['losses'])
  - When save_nn_weights=True, entries also contain 'nn_state_dict' and 'normalizing_constant'
duration: fast
verification_result: passed
completed_at: 2026-03-12
blocker_discovered: false
---

# T01: Extend per_bucket_training_log to capture loss curves and optional NN weights

**Extended per_bucket_training_log append to include losses, val_losses, and config-gated nn_state_dict + normalizing_constant — all captured before bucket deletion.**

## What Happened

Expanded the capture dict at `bucket.py` line ~328 to include `t.losses` and `t.val_losses` unconditionally, plus `nn_state_dict` (CPU-cloned) and `normalizing_constant` (as Python float) when `save_nn_weights=True` in config. Updated the schema comment in `graphical_model.py` to document the expanded fields. No new imports needed — all data is plain Python or CPU tensors.

## Verification

- `python -c "from nce.inference.bucket import FastBucket; print('ok')"` — **passed**, no syntax/import errors
- `python -c "from nce.inference.graphical_model import FastGM; print('ok')"` — **passed**
- `grep -A 20 'per_bucket_training_log' nce/inference/bucket.py` — confirmed expanded dict with losses, val_losses, conditional weight fields
- Confirmed capture site executes before `del self.buckets[var]` at graphical_model.py:309

### Slice-level checks (partial, expected for T01):
- `scripts/verify_s04_state_preservation.py` — does not exist yet (later task creates it)
- `python -c "import nce.state; print('ok')"` — expected failure, nce.state module is a later task

## Diagnostics

After running FastGM inference with NN buckets:
- `fastgm.per_bucket_training_log[i]['losses']` — list of (epoch, loss_value) tuples, empty list if no training occurred
- `fastgm.per_bucket_training_log[i]['val_losses']` — same format, may be empty
- When `save_nn_weights=True`: `fastgm.per_bucket_training_log[i]['nn_state_dict']` contains CPU tensor dict, `fastgm.per_bucket_training_log[i]['normalizing_constant']` is a Python float

## Deviations

None.

## Known Issues

None.

## Files Created/Modified

- `nce/inference/bucket.py` — expanded per_bucket_training_log append with losses, val_losses, and conditional nn_state_dict/normalizing_constant
- `nce/inference/graphical_model.py` — updated schema comment for per_bucket_training_log
