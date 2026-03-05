---
phase: quick-10
plan: 1
subsystem: inference
tags: [nbe, hidden-sizes, bucket-analysis, grid10x10]

requires: []
provides:
  - "Script nbe_log_hidden_sizes.py that iterates NN-eligible buckets and logs hidden sizes"
  - "Output file grid10x10_hidden_sizes.txt showing 0 NN-eligible buckets for grid10x10 with iB=10, ecl=512"
affects: []

tech-stack:
  added: []
  patterns:
    - "Collect bucket info BEFORE calling get_log_partition_function() — elimination destroys buckets"

key-files:
  created:
    - notebooks/March-2026/claude_experiments/nbe_log_hidden_sizes.py
    - notebooks/March-2026/claude_experiments/nbe_eval_results/grid10x10_hidden_sizes.txt
  modified: []

key-decisions:
  - "grid10x10 with iB=10 and ecl=512 has 0 NN-eligible buckets — max bucket width is 4, far below iB=10"

patterns-established:
  - "Bucket info collection pattern: iterate fastgm.elim_order, call bucket.get_width()/get_ec()/get_message_size() BEFORE elimination"

requirements-completed: [QUICK-10]

duration: 2min
completed: 2026-03-05
---

# Quick Task 10: Log Per-Bucket Hidden Sizes Summary

**Script to log NBE hidden sizes per bucket confirms grid10x10.f5.wrap has 0 NN-eligible buckets with iB=10, ecl=512 (max bucket width=4)**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-03-05T23:23:45Z
- **Completed:** 2026-03-05T23:25:47Z
- **Tasks:** 1
- **Files modified:** 2 (created)

## Accomplishments
- Created `nbe_log_hidden_sizes.py` script with exact content from `docs/task_log_hidden_sizes.md`
- Script correctly identifies NN-eligible buckets using condition `width > iB OR ec > ecl`
- Output written to `nbe_eval_results/grid10x10_hidden_sizes.txt` with header and empty bucket table
- Confirmed grid10x10 with iB=10, ecl=512 has NO NN-eligible buckets (max width=4, consistent with num_trained=0 findings)

## Task Commits

Files are in `notebooks/` directory which is gitignored (experiments excluded from git by design).

1. **Task 1: Create and run hidden sizes logging script** - files created on disk, excluded from git per `.gitignore`

## Files Created/Modified
- `notebooks/March-2026/claude_experiments/nbe_log_hidden_sizes.py` - Script to log per-bucket hidden sizes for grid10x10
- `notebooks/March-2026/claude_experiments/nbe_eval_results/grid10x10_hidden_sizes.txt` - Output file with header + empty bucket table (0 NN-eligible buckets)

## Decisions Made
- grid10x10 max bucket width is 4 (sparse graph), far below iB=10 — all 100 buckets are exact computation
- This is consistent with quick-15/16 finding that num_trained=0 for grid10x10 with this config

## Deviations from Plan
None - plan executed exactly as written.

## Issues Encountered
The output file is technically correct but contains no bucket rows. This is expected behavior, not an error:
- grid10x10 is a sparse graph — max bucket width during elimination is 4
- With iB=10, every bucket satisfies `width <= iB`
- With binary variables, ec = 2^width <= 2^4 = 16 << 512 = ecl
- All 100 buckets qualify for exact computation
- This is consistent with the established finding that num_trained=0 for this problem/config

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Hidden sizes analysis complete for grid10x10
- Result confirms NBE adaptive sizing logic works correctly but produces no NN buckets for this sparse graph
- If the goal is to see actual hidden sizes, a denser problem (wider buckets) or lower iB/ecl values are needed

---
*Phase: quick-10*
*Completed: 2026-03-05*

## Self-Check: PASSED

- FOUND: notebooks/March-2026/claude_experiments/nbe_log_hidden_sizes.py
- FOUND: notebooks/March-2026/claude_experiments/nbe_eval_results/grid10x10_hidden_sizes.txt
- FOUND: .planning/quick/10-log-per-bucket-hidden-sizes-to-file-for-/10-SUMMARY.md
