---
phase: quick
plan: 2
subsystem: data
tags: [pyGMs, catalog, benchmark, UAI]

# Dependency graph
requires: []
provides:
  - nce.benchmark_problems module with named benchmark sets
  - get_catalog() utility for pyGMs UAI model access
  - neuro_be_sanity_check set (4 models)
affects: [experiments, notebooks]

# Tech tracking
tech-stack:
  added: [pyGMs.data.catalog]
  patterns: [lazy-load benchmark sets at import, catalogue index patching for extra model sets]

key-files:
  created:
    - nce/benchmark_problems/__init__.py
    - nce/benchmark_problems/catalog_utils.py
    - nce/benchmark_problems/neuro_be_sanity_check.py
    - docs/creating_benchmark_sets.md
  modified:
    - CLAUDE.md

key-decisions:
  - "Models loaded at import time via Catalog (metadata only, no .uai download)"
  - "Pedigree index patching handled automatically via _ensure_extra_sets()"
  - "Default cache dir computed from __file__ to resolve to project-level .model_cache"

patterns-established:
  - "Benchmark set pattern: _load_benchmark_set() function with module-level variable assignment"
  - "Extra model sets: add to _EXTRA_SETS dict in catalog_utils.py"

requirements-completed: [BENCH-01]

# Metrics
duration: 3min
completed: 2026-03-03
---

# Quick Task 2: Create Benchmark Problems Module Summary

**nce.benchmark_problems module with neuro_be_sanity_check set (4 UAI models) and comprehensive documentation for creating new benchmark sets**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-03T23:38:35Z
- **Completed:** 2026-03-03T23:41:43Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Created `nce.benchmark_problems` package with `catalog_utils.py`, `neuro_be_sanity_check.py`, and `__init__.py`
- neuro_be_sanity_check set provides 4 models: pedigree13 (1077 vars), grid40x40.f10 (1600 vars), grid20x20.f10 (400 vars), rbm_20 (40 vars)
- Created thorough `docs/creating_benchmark_sets.md` with step-by-step guide, code snippets, and catalogue reference
- Updated CLAUDE.md with Benchmark Problems section

## Task Commits

Each task was committed atomically:

1. **Task 1: Create benchmark_problems module with catalogue utilities and neuro_be_sanity_check set** - `a7ff31a` (feat)
2. **Task 2: Create benchmark set documentation and update CLAUDE.md** - `6a0ecbd` (docs)

## Files Created/Modified
- `nce/benchmark_problems/__init__.py` - Module entry point exporting neuro_be_sanity_check and get_catalog
- `nce/benchmark_problems/catalog_utils.py` - Shared catalogue initialization with cache dir, pedigree patching
- `nce/benchmark_problems/neuro_be_sanity_check.py` - Benchmark set with 4 models loaded from pyGMs catalogue
- `docs/creating_benchmark_sets.md` - Step-by-step guide for creating new benchmark sets
- `CLAUDE.md` - Added Benchmark Problems section with usage example and doc pointer
- `notebooks/March-2026/claude_experiments/test_benchmark_problems.py` - Test script (gitignored)

## Decisions Made
- Models loaded at import time via Catalog -- only metadata (CSV stats) is read, not .uai files
- Pedigree index patching via `_ensure_extra_sets()` is automatic and idempotent
- Default cache directory computed from `__file__` to always resolve to project-level `.model_cache`

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Module is ready for use in any experiment notebook
- New benchmark sets can be added by following docs/creating_benchmark_sets.md

## Self-Check: PASSED

All 8 checks passed: 5 files exist, 2 commits verified, CLAUDE.md reference confirmed.

---
*Quick Task: 2*
*Completed: 2026-03-03*
