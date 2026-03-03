---
phase: quick-3
plan: 1
subsystem: benchmark
tags: [benchmark, nn-config, neuroBE]

requires:
  - phase: quick-2
    provides: benchmark_problems module with neuro_be_sanity_check model set
provides:
  - neuroBE config dicts paired with neuro_be_sanity_check models
  - neuro_be_sanity_check_configs dict keyed by catalogue key
  - neuro_be_sanity_check_configs_list ordered list for zip() usage
affects: [benchmark, experiments, neuroBE]

tech-stack:
  added: []
  patterns: [benchmark config pairing via dict + ordered list exports]

key-files:
  created:
    - notebooks/March-2026/test_benchmark_configs.py
  modified:
    - nce/benchmark_problems/neuro_be_sanity_check.py
    - nce/benchmark_problems/__init__.py

key-decisions:
  - "Dual export: configs dict (keyed by catalogue key) + configs list (same order as models) for flexible usage"
  - "_MODEL_KEYS constant as single source of truth for model list and config keys"
  - "Per-model hidden_sizes multiplier stored in _HIDDEN_SIZES_MAP for clarity"

patterns-established:
  - "Benchmark config pairing: export both dict and list forms for configs"

requirements-completed: [QUICK-3]

duration: 2min
completed: 2026-03-03
---

# Quick Task 3: Add Optional NN Config Sets to Benchmark Summary

**NeuroBE config dicts paired with neuro_be_sanity_check models via dict and ordered list exports**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-03T23:53:03Z
- **Completed:** 2026-03-03T23:54:48Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Added neuroBE config dicts to neuro_be_sanity_check with per-model hidden_sizes multipliers
- Exported both `neuro_be_sanity_check_configs` (dict) and `neuro_be_sanity_check_configs_list` (list)
- Created interactive test script with 4 #%% cells verifying all configs match expected values

## Task Commits

Each task was committed atomically:

1. **Task 1: Add neuroBE config dicts and export from __init__.py** - `6a95ebf` (feat)
2. **Task 2: Create interactive test script with #%% cells** - `fbe2cd8` (test)

## Files Created/Modified
- `nce/benchmark_problems/neuro_be_sanity_check.py` - Added _MODEL_KEYS, _HIDDEN_SIZES_MAP, _build_configs(), and config exports
- `nce/benchmark_problems/__init__.py` - Added config imports and usage docstring
- `notebooks/March-2026/test_benchmark_configs.py` - Interactive test script with 4 cells for config verification

## Decisions Made
- Dual export pattern: configs dict (keyed by catalogue key) for lookup, configs list for zip() pairing
- Extracted _MODEL_KEYS as module-level constant to avoid duplication between _load_benchmark_set() and _build_configs()
- Used _HIDDEN_SIZES_MAP to clearly document per-model hidden_sizes multiplier values

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- notebooks/ directory is gitignored; used `git add -f` to force-add the test script as it is an explicit plan deliverable

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Config pairing pattern established for future benchmark sets
- num_samples='nbe' is a placeholder string; implementation of nbe-based sample count logic is a separate task

## Self-Check: PASSED

All files and commits verified:
- [x] nce/benchmark_problems/neuro_be_sanity_check.py
- [x] nce/benchmark_problems/__init__.py
- [x] notebooks/March-2026/test_benchmark_configs.py
- [x] Commit 6a95ebf (Task 1)
- [x] Commit fbe2cd8 (Task 2)

---
*Quick task: 3-add-optional-nn-config-sets-to-benchmark*
*Completed: 2026-03-03*
