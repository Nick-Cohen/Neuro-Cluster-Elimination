---
phase: quick-4
plan: 4
subsystem: benchmark
tags: [benchmark, refactoring, api-design]

# Dependency graph
requires:
  - phase: quick-3
    provides: benchmark_problems module with neuro_be_sanity_check model set and configs
provides:
  - BenchmarkSet class wrapping .problems and .configs attributes
  - nbe_sanity_check module with fully-populated config dicts (42 keys each)
  - Cleaner single-object import pattern for benchmark sets
affects: [benchmark_problems, experiment-configs]

# Tech tracking
tech-stack:
  added: []
  patterns: [BenchmarkSet class for benchmark problem/config pairing]

key-files:
  created:
    - nce/benchmark_problems/nbe_sanity_check.py
  modified:
    - nce/benchmark_problems/__init__.py
    - notebooks/March-2026/test_benchmark_configs.py
    - CLAUDE.md
    - docs/creating_benchmark_sets.md

key-decisions:
  - "BenchmarkSet class consolidates problems list and configs dict into single importable object"
  - "Config dicts fully populated with all 42 fields from reference get_config() template"
  - "neuroBE-specific overrides: batch_size=256, iB=10, num_samples='nbe'"

patterns-established:
  - "BenchmarkSet pattern: single object with .problems and .configs['name'] attributes"

requirements-completed: [QUICK-4]

# Metrics
duration: 3min
completed: 2026-03-03
---

# Quick Task 4: Restructure benchmark_problems - Rename to nbe_sanity_check Summary

**BenchmarkSet class consolidating 4 models + fully-populated 42-key config dicts into single nbe_sanity_check import**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-04T00:10:25Z
- **Completed:** 2026-03-04T00:14:09Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Created BenchmarkSet class with .problems and .configs attributes, replacing triple-import pattern
- Fully populated config dicts with all 42 fields from reference get_config() (not just 3 fields)
- Renamed module from neuro_be_sanity_check to nbe_sanity_check for brevity
- Updated all documentation and test scripts to use new API pattern

## Task Commits

Each task was committed atomically:

1. **Task 1: Create nbe_sanity_check.py with BenchmarkSet class and fully-populated configs** - `8fb968a` (feat)
2. **Task 2: Update test script and documentation references** - `d474596` (docs)

## Files Created/Modified
- `nce/benchmark_problems/nbe_sanity_check.py` - BenchmarkSet class with 4 models and 42-key config dicts
- `nce/benchmark_problems/__init__.py` - Single import of nbe_sanity_check + BenchmarkSet
- `nce/benchmark_problems/neuro_be_sanity_check.py` - Deleted (replaced by nbe_sanity_check.py)
- `notebooks/March-2026/test_benchmark_configs.py` - Updated to use new .problems/.configs API
- `CLAUDE.md` - Updated benchmark section with new import pattern
- `docs/creating_benchmark_sets.md` - Updated examples to show BenchmarkSet pattern

## Decisions Made
- BenchmarkSet class consolidates problems list and configs dict into single importable object
- Config dicts fully populated with all 42 fields from reference get_config() template
- neuroBE-specific overrides: batch_size=256 (not 50000), iB=10, num_samples='nbe'

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- nbe_sanity_check is ready for use in experiment scripts
- BenchmarkSet pattern established for creating additional benchmark sets

## Self-Check: PASSED

- [x] nce/benchmark_problems/nbe_sanity_check.py exists
- [x] nce/benchmark_problems/neuro_be_sanity_check.py deleted
- [x] nce/benchmark_problems/__init__.py updated
- [x] notebooks/March-2026/test_benchmark_configs.py updated
- [x] docs/creating_benchmark_sets.md updated
- [x] Commit 8fb968a exists
- [x] Commit d474596 exists

---
*Quick Task: 4*
*Completed: 2026-03-03*
