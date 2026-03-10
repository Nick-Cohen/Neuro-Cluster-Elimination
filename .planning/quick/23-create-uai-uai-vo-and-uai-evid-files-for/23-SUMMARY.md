---
phase: quick-23
plan: 1
subsystem: benchmark-problems
tags: [cache, sdbe, model-files, small-problems]
dependency_graph:
  requires: [.model_cache/{category}/{model}.uai.ord files for all 24 small_problems models]
  provides: [.uai.vo files (SDBE format) for all 24 models, .uai.evid files for all 24 models]
  affects: [SDBE-compatible cache access, pyGMs catalog offline loading]
tech_stack:
  added: []
  patterns: [SDBE vo-file format (# header + one var per line), idempotent cache generation]
key_files:
  created:
    - scripts/export_small_problems_cache.py
  modified: []
decisions:
  - Hardcode model list in script (no nce import) to avoid pyGMs catalog network hang
  - Preserve existing .evid files with real evidence (never overwrite)
  - Write "0" for new .evid files (no evidence, correct pyGMs format)
  - Idempotent script: skip files that already exist (safe to rerun)
metrics:
  duration: "3 min"
  completed_date: "2026-03-10"
---

# Phase quick-23 Plan 1: Create .uai.vo and .uai.evid Cache Files Summary

**One-liner:** SDBE-format .vo files and missing .evid files generated for all 24 small_problems models via idempotent export script.

## Objective

Create `.uai.vo` and `.uai.evid` files for all 24 small_problems benchmark instances so that the complete set of cache files (`.uai`, `.uai.ord`, `.uai.vo`, `.uai.evid`) is available in `.model_cache/{category}/`.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Create export script and generate all .uai.vo and .uai.evid files | 133a16b | scripts/export_small_problems_cache.py |

## What Was Built

### scripts/export_small_problems_cache.py

A standalone Python script that:
1. Reads existing `.uai.ord` files (format: `count var1 var2 ... varN` on one line)
2. Generates `.uai.vo` files in SDBE format (`#` header + one variable per line)
3. Generates `.uai.evid` files with `0` (no evidence) for models missing them
4. Preserves existing `.evid` files with real evidence (BN models + or_chain_10.fg)
5. Prints a summary table showing all 4 file status flags per model

The script does NOT import from `nce` to avoid triggering pyGMs catalog network calls.

### Files Generated

- **24 .uai.vo files created** (all 24 models were missing .vo files)
- **14 .uai.evid files created** with content `0` (no evidence):
  - alchemy/smokers_20
  - segmentation/10_14_s.binary, 10_16_s.binary, 11_17_s.binary, 11_4_s.binary
  - objdetect/deer_rescaled_0034.K15.F1.5.model, K15.F1.75.model, K10.F2.model, K20.F1.25.model, K20.F1.5.model
  - objdetect/deer_rescaled_0294.K10.F1.75.model
  - grids/grid10x10.f5.wrap
  - csp/29.wcsp, csp/404.wcsp
- **10 existing .evid files preserved** (BN_1..BN_11 + or_chain_10.fg with real evidence)

## Verification

- All 24 models have .uai.vo files: PASSED
- All 24 models have .uai.evid files: PASSED
- .vo format correct (# header + one int per line): PASSED
- BN_1.uai.evid preserved with real evidence (18 observed vars): PASSED
- or_chain_10.fg.uai.evid preserved with real evidence (8 observed vars): PASSED
- New .evid files contain "0": PASSED

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check

- [x] scripts/export_small_problems_cache.py exists and is committed (133a16b)
- [x] 24 .uai.vo files present across all category subdirectories
- [x] 24 .uai.evid files present across all category subdirectories
- [x] BN evidence preserved (18 vars in BN_1.uai.evid)

## Self-Check: PASSED
