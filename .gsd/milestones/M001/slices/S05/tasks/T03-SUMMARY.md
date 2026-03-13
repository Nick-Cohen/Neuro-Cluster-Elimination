---
id: T03
parent: S05
milestone: M001
provides:
  - scripts/verify_s05_visualization.py — end-to-end verification for S05 visualization module
key_files:
  - scripts/verify_s05_visualization.py
key_decisions:
  - Matched S04 verification pattern exactly (check/summarize_and_exit, temp file cleanup, same base config)
patterns_established:
  - 18 checks across 2 groups (plot_learning_curves: 10, compare_experiments: 8) covering all three input modes and edge cases
observability_surfaces:
  - Script prints named PASS/FAIL per check, summary count, OVERALL verdict, and lists failing checks on failure
duration: ~5m implementation + ~1m inference runtime
verification_result: passed
completed_at: 2026-03-12
blocker_discovered: false
---

# T03: End-to-end verification script

**Created `scripts/verify_s05_visualization.py` — 18/18 checks pass, proving S05 visualization module works on real inference data.**

## What Happened

Built the S05 verification script following S04's check/summarize_and_exit pattern. The script runs inference on rbm_20 (3 epochs, ecl=2^19, iB=20), saves state to disk, then exercises both `plot_learning_curves()` and `compare_experiments()` across all three input modes (live FastGM, loaded state dict, file path string) plus edge cases.

10 checks for `plot_learning_curves`:
- Returns Figure from FastGM, state dict, and file path inputs
- Subplot count matches entries with non-empty losses (20 NN buckets)
- Each subplot has at least one plotted line
- save_path produces a PNG file on disk
- Empty training log → Figure with ≤1 axis, no crash
- Entry with empty losses → skipped (no subplot created)
- max_subplots=2 caps visible subplots

8 checks for `compare_experiments`:
- Returns Figure from list of state dicts and file paths
- Has ≥2 visible axes (summary + overlays)
- Summary axis title contains "Summary"
- Experiment labels appear in legend
- save_path produces file on disk
- Single source (degenerate) → no crash
- No overlapping bucket labels → 1 visible axis (summary only)

## Verification

```
python scripts/verify_s05_visualization.py
# Results: 18/18 checks passed
# OVERALL: PASS
# Exit code: 0

python -c "from nce.visualization import plot_learning_curves, compare_experiments; print('ok')"
# ok
```

Both slice-level verification checks pass.

## Diagnostics

Re-run `python scripts/verify_s05_visualization.py` — self-contained, prints named PASS/FAIL per check. Failing checks list their detail messages in the summary section.

## Deviations

None.

## Known Issues

None.

## Files Created/Modified

- `scripts/verify_s05_visualization.py` — end-to-end verification script, 18 checks across 2 test groups
