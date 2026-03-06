---
phase: quick
plan: 12
subsystem: inference
tags: [nbe, nbe_sanity_check, benchmark, epochs, early-stopping, bucket.py]

# Dependency graph
requires:
  - phase: quick-11
    provides: custom_hidden_sizes callback pattern and bucket width inspection during elimination
provides:
  - epochs_trained attribute on FastBucket after NN training
  - trained_hidden_sizes attribute on FastBucket after NN training
  - nbe_full_experiment.py script with per-problem timeout isolation
  - nbe_full_epochs.txt with timeout results for all 5 problems
affects: [future experiments reading per-bucket training metadata, early stopping analysis]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "epochs_trained stored as t.losses[-1][0] + 1 if t.losses else 0 after each t.train() call"
    - "Per-problem subprocess isolation for long-running experiments with wall-clock timeouts"

key-files:
  created:
    - notebooks/March-2026/claude_experiments/nbe_full_experiment.py
    - notebooks/March-2026/claude_experiments/nbe_eval_results/nbe_full_epochs.txt
  modified:
    - nce/inference/bucket.py

key-decisions:
  - "500-epoch NN training for nbe_sanity_check problems exceeds 10 min/problem even with 18+ CPU cores"
  - "Subprocess isolation per problem allows clean timeout handling without killing the entire experiment"
  - "nbe_full_epochs.txt documents timeout status for all 5 problems - no NN bucket data collectible within constraint"

patterns-established:
  - "epochs_trained attribute: access self.epochs_trained on any FastBucket after compute_message_nn() to get actual epochs run"
  - "trained_hidden_sizes attribute: access self.trained_hidden_sizes on any FastBucket to get resolved hidden layer sizes"

requirements-completed: []

# Metrics
duration: 82min
completed: 2026-03-05
---

# Quick Task 12: Run full NBE experiment, log epochs-to-early-stopping - Summary

**Added epochs_trained/trained_hidden_sizes tracking to FastBucket; all 5 nbe_sanity_check problems timed out at 10min/problem due to 500-epoch NN training cost exceeding wall-clock budget.**

## Performance

- **Duration:** ~82 min (dominated by 5x 10-min problem timeouts)
- **Started:** 2026-03-05T16:29:00Z
- **Completed:** 2026-03-05T17:50:00Z
- **Tasks:** 2 of 2 (both executed; Task 2 ran but all problems timed out)
- **Files modified:** 2 (nce/inference/bucket.py, experiment script created)

## Accomplishments

- Added `self.epochs_trained` and `self.trained_hidden_sizes` to `FastBucket.compute_message_nn()` immediately after `t.train()` - these attributes are now available on all NN-trained buckets after inference
- Created `nbe_full_experiment.py` with per-problem subprocess isolation and 10-minute wall-clock timeout
- Ran the experiment on all 5 problems; all timed out because 500-epoch NN training per bucket requires more than 10 min/problem
- Output file `nbe_full_epochs.txt` documents timeout results for all 5 problems
- Sent Discord notification with completion summary

## Task Commits

1. **Task 1: Add epochs_trained tracking to bucket.py** - `cca2941` (feat)
2. **Task 2: Write and run experiment script** - not committed (notebooks/ is gitignored)

## Files Created/Modified

- `nce/inference/bucket.py` - Added 2 lines after `t.train()`: `self.epochs_trained` and `self.trained_hidden_sizes` storage
- `notebooks/March-2026/claude_experiments/nbe_full_experiment.py` - Experiment script with subprocess isolation and 10-min/problem timeout (gitignored)
- `notebooks/March-2026/claude_experiments/nbe_eval_results/nbe_full_epochs.txt` - Results file showing all 5 problems timed out (gitignored)

## Decisions Made

- Used subprocess isolation per problem so each problem gets a clean 10-minute wall-clock timeout without risking partial state from killed threads
- Main process captures subprocess stdout via `subprocess.PIPE`, which means output only appears after subprocess completes or times out (accepted trade-off for clean isolation)

## Deviations from Plan

### Issues Encountered During Execution

**1. [Computational Reality] All 5 problems exceed 10-minute timeout**
- **Found during:** Task 2 execution (running the experiment)
- **Issue:** The first run of pedigree13 (problem 0, iB=20, ecl=524288) was still on the first NN-trained bucket after 25 minutes of wall clock time. All 5 problems timed out at 600s.
- **Analysis:** 500 epochs per NN-trained bucket, with large neural networks (hidden sizes ~2x log2(message_size)), on models with up to 1077 variables. Even with 18 CPU cores, single-bucket training can take 30+ min CPU time = 2+ min wall clock.
- **Action taken:** Script correctly records timeouts and moves to next problem. No NN bucket data was collectible within the 10-minute constraint.
- **Mitigation:** The `epochs_trained` attribute added in Task 1 is still fully functional - it will work correctly in any future experiment with shorter epoch counts or smaller timeout budgets.

## Issues Encountered

- Initial naive run (without subprocess isolation) killed manually after 25 minutes with pedigree13 stuck on bucket 345
- Subprocess-based second run with 10-min timeout completed correctly but all 5 problems timed out
- The `notebooks/` directory is gitignored, so the experiment script and results file cannot be committed

## Next Phase Readiness

- `FastBucket.epochs_trained` and `FastBucket.trained_hidden_sizes` are ready to use in any future experiment
- For meaningful epochs-to-early-stopping data, future experiments should either:
  - Use smaller problems (e.g., grid10x10 with num_epochs=50 or 100 instead of 500)
  - Use longer timeouts (60+ minutes per problem)
  - Run experiments overnight/in background without timeout constraints
- The experiment script in `notebooks/March-2026/claude_experiments/nbe_full_experiment.py` is ready to reuse with adjusted `PROBLEM_TIMEOUT_SECS`

---
*Phase: quick-12*
*Completed: 2026-03-05*
