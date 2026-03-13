---
id: T03
parent: S04
milestone: M001
provides:
  - End-to-end verification script proving S04 state preservation pipeline works
key_files:
  - scripts/verify_s04_state_preservation.py
key_decisions:
  - Used rbm_20 from nbe_sanity_check instead of small_problems — small_problems models all have incomplete .vo files (missing one variable from elimination order), causing bucket placement failures
  - Config tuned to ecl=2^19 with iB=20 to get exactly 20 NN-trained buckets on rbm_20, balancing coverage against runtime (~25s per inference pass)
  - Used dope_factors=True (matching nbe_sanity_check convention) for stable training with the rbm_20 model
patterns_established:
  - Verification script pattern: check() helper accumulates PASS/FAIL results, summarize_and_exit() prints summary and cleans temp files
observability_surfaces:
  - Re-run `python scripts/verify_s04_state_preservation.py` — self-contained, prints 26 named PASS/FAIL checks, exits 0/1
duration: 25min
verification_result: passed
completed_at: 2026-03-12
blocker_discovered: false
---

# T03: Round-trip verification script on a real NN-trained problem

**Created verification script that exercises the full S04 pipeline: NN inference → loss curve capture → save/load in both modes → undo_normalization — all 26 checks pass.**

## What Happened

Built `scripts/verify_s04_state_preservation.py` that runs two full inference passes on rbm_20 (40 binary vars, 20 NN-trained buckets with 3 epochs each). The original plan suggested using small_problems (alchemy/smokers_20), but all 24 models in that benchmark set have incomplete .vo files — each is missing one variable from the elimination order, causing a `ValueError` at bucket placement. Switched to nbe_sanity_check's rbm_20 which has a reliable ordering.

The script verifies four groups:
1. **Training log capture** (6 checks): log non-empty, losses present as (epoch, value) tuples, epochs_trained > 0, hidden_sizes is list, val_losses is list
2. **Metadata-only save/load** (12 checks): all state dict keys present (training_log, config, logZ, elim_order, num_trained, bucket_complexities), no nn_state_dict in entries, loss curves survive round-trip with correct format
3. **Weights-included save/load** (5 checks): nn_state_dict is dict with >0 keys, normalizing_constant present and finite
4. **undo_normalization** (3 checks): result is finite, differs from input (nc≠0), shape preserved

## Verification

```
$ python scripts/verify_s04_state_preservation.py
Results: 26/26 checks passed
OVERALL: PASS

$ python -c "import nce.state; print('state module importable')"
state module importable
```

Both slice-level verification checks pass.

## Diagnostics

Re-run the script — it's self-contained and prints named PASS/FAIL per check. On failure, the summary section lists only failing checks with their detail messages. Temp files (`/tmp/test_s04_*.pkl`) are cleaned up on exit.

## Deviations

- Used `nbe_sanity_check.problems[3]` (rbm_20) instead of `small_problems.problems[0]` (smokers_20) — all small_problems models have broken .vo files that are missing one variable from the elimination order. This is a pre-existing data issue, not a code bug.
- Used `ecl=2^19` instead of `ecl=4` — ecl=4 would make 37 of 40 buckets NN-eligible, causing runtime to exceed 120s. ecl=2^19 gives 20 NN buckets which is sufficient coverage and runs in ~25s per pass.
- Total runtime is ~50s (two inference passes), well within the 120s budget.

## Known Issues

- All 24 models in `small_problems` have incomplete .vo files (each missing one variable). This pre-dates S04 and doesn't affect the state preservation feature, but it means small_problems can't be used for quick integration tests without fixing the ordering files.

## Files Created/Modified

- `scripts/verify_s04_state_preservation.py` — end-to-end verification script, 26 assertions across 4 test groups
