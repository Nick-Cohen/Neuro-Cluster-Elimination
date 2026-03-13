---
estimated_steps: 4
estimated_files: 2
---

# T01: Extend per_bucket_training_log to capture loss curves and optional NN weights

**Slice:** S04 — FastGM State Preservation
**Milestone:** M001

## Description

The capture site in `bucket.py:328-333` currently appends only `{label, epochs_trained, hidden_sizes}` to `per_bucket_training_log`. Loss curves live in `Trainer.losses` and `Trainer.val_losses` (lists of `(epoch, loss_value)` tuples) but are lost when the bucket is deleted at `graphical_model.py:309`. This task extends the capture dict to include full loss curves and, when a config flag is set, NN weights and normalizing constant — all before bucket deletion.

## Steps

1. **Extend the per_bucket_training_log append** (bucket.py:328-333): Add `'losses': t.losses` and `'val_losses': t.val_losses` to the dict. These are lists of `(epoch, loss_value)` tuples — plain Python data, no CUDA dependency.

2. **Add optional NN weight capture**: After the existing dict construction, if `self.config.get('save_nn_weights', False)` is true, add `'nn_state_dict': {k: v.cpu().clone() for k, v in net.state_dict().items()}` and `'normalizing_constant': t.data_preprocessor.normalizing_constant.cpu().item()` to the entry. The `.cpu()` ensures portability; `.item()` converts the scalar tensor to a Python float. Use `.clone()` to detach from the original tensor.

3. **Update the comment** on `self.per_bucket_training_log` in `graphical_model.py:57` to document the expanded schema: `# List of dicts per NN bucket: {label, epochs_trained, hidden_sizes, losses, val_losses, [nn_state_dict, normalizing_constant]}`.

4. **Verify no import or syntax issues**: Run a quick import check and confirm the diff is correct — all new fields are captured before `del self.buckets[var]` at graphical_model.py:309.

## Must-Haves

- [ ] `losses` key added to every per_bucket_training_log entry (list of (epoch, loss) tuples)
- [ ] `val_losses` key added to every entry (list, may be empty)
- [ ] `nn_state_dict` key added only when `save_nn_weights=True` in config
- [ ] `normalizing_constant` key added only when `save_nn_weights=True` in config
- [ ] CUDA tensors in state_dict moved to CPU via `.cpu().clone()`
- [ ] `normalizing_constant` stored as Python float via `.cpu().item()`
- [ ] All capture happens before `del self.buckets[var]`

## Verification

- `python -c "from nce.inference.bucket import FastBucket; print('ok')"` — no syntax errors
- Visual inspection: the append at bucket.py:328 now includes `losses`, `val_losses`, and conditional weight fields
- `grep -A 15 'per_bucket_training_log.append' nce/inference/bucket.py` shows the expanded dict

## Observability Impact

- Signals added/changed: `per_bucket_training_log` entries gain `losses` and `val_losses` keys — any downstream code can now inspect training trajectories without access to the (deleted) bucket or trainer
- How a future agent inspects this: `len(fastgm.per_bucket_training_log[i]['losses'])` after elimination
- Failure state exposed: If `t.losses` is unexpectedly empty, the entry will have `losses: []` — visible, not silent

## Inputs

- `nce/inference/bucket.py` — current capture site at lines 328-333
- `nce/inference/graphical_model.py` — `per_bucket_training_log` initialization at line 57
- `nce/neural_networks/train.py` — `self.losses` and `self.val_losses` structure (line 80-81)
- `nce/data/data_preprocessor.py` — `self.normalizing_constant` attribute

## Expected Output

- `nce/inference/bucket.py` — expanded per_bucket_training_log append with loss curves and optional weights
- `nce/inference/graphical_model.py` — updated comment documenting expanded schema
