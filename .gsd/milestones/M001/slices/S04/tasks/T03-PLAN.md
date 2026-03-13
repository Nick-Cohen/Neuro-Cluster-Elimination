---
estimated_steps: 5
estimated_files: 1
---

# T03: Round-trip verification script on a real NN-trained problem

**Slice:** S04 — FastGM State Preservation
**Milestone:** M001

## Description

This is the slice's proof task. It creates and runs a verification script that exercises the full pipeline: load a small benchmark problem, run NN inference (forcing a few NN buckets with low ecl, 2-5 epochs for speed), save state in both modes (metadata-only and with-weights), load state in a simulated fresh context, and assert all expected fields are present and correct. This proves R009, R010, R011, and R012 end-to-end.

## Steps

1. **Create `scripts/verify_s04_state_preservation.py`**: The script should:
   - Load a small benchmark problem. Use `small_problems` from `nce.benchmark_problems` — pick the first problem (alchemy/smokers_20) with a modified config: `ecl=4` (force NN buckets on modest-width messages), `num_epochs=3`, `hidden_sizes=[3,3]`, `device='cuda'`. If CUDA unavailable, fall back to CPU.
   - Run `FastGM(model=model, nn_config=config, device=config['device'])` then `fastgm.eliminate_variables(all=True)`.
   - Assert `fastgm.per_bucket_training_log` is non-empty (at least one NN bucket trained).

2. **Test metadata-only save/load**:
   - Call `save_state(fastgm, '/tmp/test_s04_meta.pkl', save_weights=False)`.
   - Call `state = load_state('/tmp/test_s04_meta.pkl')`.
   - Assert: `state` has key `per_bucket_training_log`, it's a non-empty list, each entry has `losses` key with at least one `(epoch, loss)` tuple, `epochs_trained > 0`, `hidden_sizes` is a list, `val_losses` is a list (may be empty).
   - Assert: no `nn_state_dict` key in entries (stripped by save_state in metadata-only mode).

3. **Test weights-included save/load**:
   - Set `config['save_nn_weights'] = True` before inference OR call `save_state(fastgm, path, save_weights=True)`. Since save_state in metadata-only mode strips weights, we need to ensure the capture happened. Modify the config to include `save_nn_weights=True`, re-run inference on the same problem.
   - Call `save_state(fastgm, '/tmp/test_s04_weights.pkl', save_weights=True)`.
   - Load and assert: entries have `nn_state_dict` (an OrderedDict), `normalizing_constant` (a finite float).

4. **Test undo_normalization**:
   - From a weight-included state entry, grab `normalizing_constant`.
   - Create a dummy tensor `torch.tensor([0.0, 1.0, -1.0])`.
   - Call `undo_normalization(dummy, normalizing_constant)`.
   - Assert: result is finite, different from input (normalizing_constant != 0), has same shape.

5. **Print summary**: Print PASS/FAIL for each assertion group. Exit 0 if all pass, exit 1 if any fail. Clean up temp files.

## Must-Haves

- [ ] Script loads a real benchmark problem and runs NN inference
- [ ] Verifies per_bucket_training_log has loss curves after elimination
- [ ] Tests metadata-only save/load round-trip
- [ ] Tests weights-included save/load round-trip
- [ ] Tests undo_normalization with saved normalizing constant
- [ ] Exits 0 on success, 1 on failure
- [ ] Runtime under 120 seconds (low epochs, small problem)

## Verification

- `python scripts/verify_s04_state_preservation.py` exits 0 with all assertions passing
- If any assertion fails, the script names the failing check and exits 1

## Observability Impact

- Signals added/changed: None (this is a verification script, not production code)
- How a future agent inspects this: Re-run the script — it's self-contained and prints PASS/FAIL per check
- Failure state exposed: Named assertion failures with context on what was expected vs found

## Inputs

- T01 output: extended per_bucket_training_log in bucket.py (losses, val_losses, optional weights)
- T02 output: `nce/state/` module with save_state, load_state, undo_normalization
- `nce/benchmark_problems/small_problems.py` — benchmark problems to test against
- `nce/inference/graphical_model.py` — FastGM class

## Expected Output

- `scripts/verify_s04_state_preservation.py` — self-contained verification script that proves the slice works end-to-end
