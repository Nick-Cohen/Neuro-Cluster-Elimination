---
phase: quick-7
plan: 1
subsystem: benchmark-problems
tags: [neurobe, config, evaluation, nbe, pre-smoke-test]

# Dependency graph
requires:
  - phase: quick-5
    provides: "NeuroBE num_samples function and nbe config structure"
provides:
  - "Corrected NBE configs with paper-accurate hyperparameters"
  - "Updated evaluation plan with Phase 0a/0b pre-smoke tests"
  - "Three runnable pre-smoke test scripts"
affects: [nbe-evaluation, benchmark-problems]

# Tech tracking
tech-stack:
  added: []
  patterns: [matching_var-for-int-to-Var-conversion, get_log_partition_function-over-run]

key-files:
  created:
    - notebooks/March-2026/claude_experiments/nbe_eval_phase0a_exact.py
    - notebooks/March-2026/claude_experiments/nbe_eval_phase0b_single_bucket.py
    - notebooks/March-2026/claude_experiments/nbe_eval_practice_1epoch.py
  modified:
    - nce/benchmark_problems/nbe_sanity_check.py
    - docs/nbe_evaluation_plan.md

key-decisions:
  - "Constructor auto-calls dope_factors() when config['dope_factors']=True, scripts rely on this"
  - "Scripts still document dope_factors behavior in comments for clarity"
  - "Phase 0b uses pedigree13 (not grid10x10) for large bucket testing"

patterns-established:
  - "matching_var() converts int label to Var before eliminate_variables(up_to=var)"
  - "get_log_partition_function() is the correct API, not run()"

requirements-completed: []

# Metrics
duration: 4min
completed: 2026-03-05
---

# Quick Task 7: Fix NBE Configs, Update Eval Plan, Create Pre-Smoke Scripts Summary

**Fixed 6 NeuroBE config bugs (loss_fn, backward_iB, dope_factors, epochs, lr, backward_ecl), rewrote evaluation plan with Phase 0a/0b pre-smoke phases, and created 3 runnable test scripts**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-05T20:00:08Z
- **Completed:** 2026-03-05T20:04:52Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments
- All 6 NBE config bugs fixed: loss_fn='weighted_logspace_mse', backward_iB matches per-model iB, dope_factors=True, num_epochs=500, lr=0.001, backward_ecl=None
- Evaluation plan rewritten with Phase 0a (exact baseline), Phase 0b (single bucket NN), corrected API usage, hidden_sizes explanation, simplified output table
- Three pre-smoke test scripts created with correct API patterns (matching_var for int-to-Var conversion)

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix all NBE config bugs in nbe_sanity_check.py** - `2f657f4` (fix)
2. **Task 2: Update docs/nbe_evaluation_plan.md with user feedback** - `21ca6ba` (docs)
3. **Task 3: Create three pre-smoke test scripts** - `5722360` (feat)

## Files Created/Modified
- `nce/benchmark_problems/nbe_sanity_check.py` - Fixed 6 config bugs, updated docstring with NeuroBE hyperparameters
- `docs/nbe_evaluation_plan.md` - Rewritten with Phase 0a/0b, API corrections, simplified output
- `notebooks/March-2026/claude_experiments/nbe_eval_phase0a_exact.py` - Exact computation baseline on grid10x10
- `notebooks/March-2026/claude_experiments/nbe_eval_phase0b_single_bucket.py` - Single bucket NN test on pedigree13
- `notebooks/March-2026/claude_experiments/nbe_eval_practice_1epoch.py` - Practice 1-epoch full pipeline run

## Decisions Made
- Verified that FastGM constructor auto-calls `dope_factors()` when `config['dope_factors']=True` (line 112-113 of graphical_model.py), so scripts don't need to call it manually
- Scripts document this behavior in comments for developer clarity
- Phase 0b uses pedigree13 (index 0) because it has 1077 vars and width 32, providing genuinely large buckets for NN testing
- Used `git add -f` for notebook scripts since `notebooks/` is in .gitignore

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- `notebooks/` directory is in `.gitignore`, required `git add -f` to commit the scripts. This is expected since notebooks are generally local-only.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Pre-smoke scripts are ready to run: start with Phase 0a, then 0b, then practice 1-epoch
- After pre-smoke tests pass, proceed to Phase 1 (full 500-epoch smoke test) and Phase 2 (rbm_20 ablation)
- All configs are corrected and ready for evaluation

## Self-Check: PASSED

All 5 files verified present. All 3 task commits verified in git log.

---
*Phase: quick-7*
*Completed: 2026-03-05*
