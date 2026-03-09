---
phase: quick-19
plan: 19
subsystem: docs
tags: [documentation, model-cache, pyGMs, debugging, catalog]
dependency_graph:
  requires: []
  provides: [docs/model_cache_setup.md]
  affects: [playground.py, any script using pyGMs catalog models]
tech_stack:
  added: []
  patterns: [pre-caching guide, preflight check pattern]
key_files:
  created:
    - docs/model_cache_setup.md
  modified: []
decisions:
  - Evidence file format for no-evidence case is "0" (single integer), not an empty file
  - .model_cache/ is gitignored so cache files must be manually maintained on each machine
  - Root cause is requests.get() with no timeout in pyGMs catalog loader
metrics:
  duration: 3 min
  completed: 2026-03-09
  tasks_completed: 1
  files_created: 1
  files_modified: 0
---

# Quick Task 19: Fix playground.py Hanging on FastGM Creation — Summary

**One-liner:** Documented pyGMs catalog hang root cause (unreachable server + no timeout) and three-file cache fix for grid10x10.f10.wrap.

## What Was Built

Created `docs/model_cache_setup.md` (145 lines) documenting:

1. **Problem** — `playground.py` hung indefinitely on `FastGM` creation with no error output.
2. **Root Cause** — pyGMs catalog lazy-loader calls `requests.get()` with no timeout to download missing `.uai`, `.uai.ord`, and `.uai.evid` files from `sli.ics.uci.edu`, which is unreachable from this machine.
3. **Fix Applied (2026-03-09)** — Three files corrected in `.model_cache/grids/` for `grid10x10.f10.wrap`:
   - `.uai` — was absent; copied from `/home/cohenn1/SDBE/benchmark_problems/`
   - `.uai.ord` — was absent; computed via `wtminfill_order()` and written one integer per line
   - `.uai.evid` — had wrong content (contained elimination order); replaced with `0` (no evidence)
4. **Pre-Caching Guide** — Steps to verify and create all three required files for any future catalog model.
5. **Long-Term Recommendation** — Preflight check code snippet for `playground.py` that converts silent hangs into clear error messages.

## Decisions Made

- Evidence file format for an empty evidence set is `"0"` (a single integer indicating 0 observed variables), not an empty file and not a list of variable assignments.
- `.model_cache/` is gitignored; cache files must be maintained manually on each machine. Future agents should treat this as machine-local state.
- The hang occurs during pyGMs `Model` construction (before `FastGM` is called), making it look like a `FastGM` hang.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| Task 1: Create docs/model_cache_setup.md | e736006 | docs(quick-19): document model cache setup and pyGMs catalog hang fix |

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check: PASSED

- [x] `docs/model_cache_setup.md` exists (145 lines)
- [x] Commit e736006 exists
- [x] File covers: root cause, fix applied, pre-caching steps, long-term recommendation
