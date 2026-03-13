# S04 Post-Slice Roadmap Assessment

## Verdict: Roadmap unchanged

S04 delivered exactly what it promised. No slice reordering, merging, splitting, or scope changes needed.

## What S04 Retired

- **Pickle compatibility risk** — retired. Round-trip save/load verified on a real 20-NN-bucket problem (rbm_20) with 26 passing checks. pyGMs Var objects sidestepped by converting elim_order to plain int labels (D015). CUDA tensors handled via `.cpu().clone()`.
- **Loss curve capture risk** — retired. `per_bucket_training_log` entries now include `losses` and `val_losses` as `(epoch, value)` tuple lists, captured before bucket deletion.

## Boundary Contract S04→S05: Verified Accurate

The roadmap's boundary map matches what was built:
- `save_state(fastgm, path, save_weights=False)` / `load_state(path)` → exists as specified in `nce/state/`
- State dict keys: `per_bucket_training_log`, `config`, `logZ`, `elim_order`, `num_trained`, `bucket_complexities` → confirmed
- Training log entries: `{label, epochs_trained, hidden_sizes, losses, val_losses}` → confirmed, losses are `(epoch, value)` tuples
- Optional weights: `nn_state_dict` + `normalizing_constant` gated on `save_weights=True` → confirmed
- `undo_normalization(outputs, normalizing_constant)` standalone function → confirmed

S05 can consume these interfaces directly.

## Success Criteria Coverage

- User can write experiment configs using readable nested sections → S01 ✅
- Old flat config dicts continue to work without any changes → S01 ✅
- Dead config fields raise clear errors when used → S01 ✅
- A pickled FastGM preserves per-bucket training metadata and can be inspected/plotted in a fresh session → S04 ✅ (preservation), S05 (plotting)
- A one-command regression test proves nested and flat configs produce identical inference results → S07

All criteria have at least one remaining owning slice.

## Requirement Coverage

No changes. R009–R012 (S04's requirements) are implemented and verified. R013–R015 (S05), R016 (S06), R017 (S07) remain correctly mapped.

## Remaining Slice Order

S05 → S06 → S07 — no reordering needed. S05 depends on S04 (done). S06 is independent. S07 depends on S01+S02 (done). No new dependencies emerged.

## Notes

- All 24 `small_problems` models have incomplete `.vo` files (missing one variable). Pre-existing issue, doesn't affect remaining slices. Worth noting for M002 test suite planning.
