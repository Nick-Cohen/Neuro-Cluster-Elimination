---
estimated_steps: 5
estimated_files: 1
---

# T03: End-to-end verification script

**Slice:** S05 — Visualization Module
**Milestone:** M001

## Description

Create `scripts/verify_s05_visualization.py` following S04's verification pattern (check/summarize_and_exit). Exercises both `plot_learning_curves()` and `compare_experiments()` on real inference data from rbm_20, verifying correct behavior with live FastGM, loaded state dict, and file path inputs. Also tests edge cases (empty training log, entries with empty losses). This is the objective stopping condition for S05.

## Steps

1. Create `scripts/verify_s05_visualization.py` with check/summarize_and_exit helpers (same pattern as S04)
2. **Setup group**: Run inference on rbm_20 with same config as S04 verification (ecl=2^19, iB=20, 3 epochs). Save state via `save_state()`. This gives us a live FastGM, a state dict via `load_state()`, and a file path.
3. **plot_learning_curves checks** (~10 checks):
   - Returns matplotlib Figure (isinstance check)
   - Subplot count matches number of entries with non-empty losses
   - Each subplot has at least one line (train loss)
   - save_path produces a file on disk (PNG)
   - Accepts live FastGM → Figure returned
   - Accepts loaded state dict → Figure returned
   - Accepts file path (str) → Figure returned
   - Empty training log → Figure returned (no crash), zero or one axes
   - Entry with empty losses list → skipped (no subplot for it)
   - max_subplots=2 caps subplot count
4. **compare_experiments checks** (~8 checks):
   - Returns matplotlib Figure
   - Has at least 2 axes (summary + at least one overlay)
   - Accepts list of two state dicts → Figure
   - Accepts list of file paths → Figure
   - Labels appear in legend/title
   - save_path produces a file on disk
   - Single source (degenerate) → no crash
   - Two sources with no overlapping labels → Figure with summary only
5. Clean up temp files, print summary, exit 0/1

## Must-Haves

- [ ] Script follows check/summarize_and_exit pattern from S04
- [ ] Tests `plot_learning_curves` with FastGM, state dict, and file path inputs
- [ ] Tests `compare_experiments` with multiple sources
- [ ] Tests edge cases: empty log, empty losses, max_subplots cap, no overlapping labels
- [ ] Tests save_path produces file on disk
- [ ] All Figures are closed after checks (`plt.close(fig)`)
- [ ] Temp files cleaned up on exit
- [ ] Exit 0 if all pass, exit 1 if any fail

## Verification

- `python scripts/verify_s05_visualization.py` — all checks pass, OVERALL: PASS, exit 0

## Observability Impact

- Signals added/changed: None (verification-only)
- How a future agent inspects this: Re-run the script — self-contained, prints named PASS/FAIL per check
- Failure state exposed: Failing checks list their detail messages in the summary

## Inputs

- `nce/visualization/learning_curves.py` — `plot_learning_curves()` (from T01)
- `nce/visualization/comparison.py` — `compare_experiments()` (from T02)
- `nce/state/state.py` — `save_state()`, `load_state()`
- `nce/benchmark_problems/nbe_sanity_check.py` — rbm_20 model for real inference
- `scripts/verify_s04_state_preservation.py` — pattern reference for check/summarize_and_exit

## Expected Output

- `scripts/verify_s05_visualization.py` — end-to-end verification script, ~18 checks across 2 test groups, prints summary, exits 0/1
