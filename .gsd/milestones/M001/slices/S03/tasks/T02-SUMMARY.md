---
id: T02
parent: S03
milestone: M001
provides:
  - tests/test_config_docs.py — doc-sync enforcement test (4 test cases)
  - config_schema.py doc pointer comment
  - configs/example_nn_config.py — flat + nested example configs
key_files:
  - tests/test_config_docs.py
  - nce/config_schema.py
  - configs/example_nn_config.py
key_decisions:
  - Parsed 5-column tables (requiring backtick in 3rd column) to distinguish section field tables from 3-column Dead/Runtime/Legacy tables
patterns_established:
  - Doc-sync tests parse markdown tables with regex; scope by column count to avoid false positives from non-section tables
observability_surfaces:
  - pytest tests/test_config_docs.py -v — assertion failures name specific missing fields and drift direction
duration: 15m
verification_result: passed
completed_at: 2026-03-12
blocker_discovered: false
---

# T02: Add doc-sync test, schema pointer, and update example config

**Created `tests/test_config_docs.py` with 4 test cases enforcing bidirectional sync between `docs/config_reference.md` and `nce/config_schema.py`, plus schema doc pointer and updated example config.**

## What Happened

Wrote `tests/test_config_docs.py` with four tests: schema→docs sync, docs→schema sync, dead fields documented, legacy fields documented. The markdown parser uses regex on 5-column table rows (requiring a backtick-wrapped Type column) to distinguish the 6 main section tables from 3-column tables (Dead Fields, Runtime-Injected, Legacy). Initial run caught false positives from `sigma_g_global`/`rho_global`/`exact` rows in the Runtime and Legacy tables — fixed by tightening the regex to require a third backticked column. Added `# Field documentation: docs/config_reference.md` comment near the top of `config_schema.py`. Rewrote `configs/example_nn_config.py` with `flat_config` and `nested_config` examples using the same parameter values, with a module docstring explaining both formats are valid.

## Verification

- `pytest tests/test_config_docs.py -v` — 4/4 passed
- Sanity check: temporarily removed `dope_factors` from guide → test correctly failed naming `['dope_factors']`
- `python -c "from nce.config_schema import NESTED_SECTIONS; print('schema importable')"` — confirmed import works after adding comment
- Slice-level verification: all 3 checks pass (pytest green, schema importable, guide exists and is organized)

## Diagnostics

Run `pytest tests/test_config_docs.py -v`. On failure, assertion messages report exactly which fields are missing and which direction (schema→docs or docs→schema).

## Deviations

Initial regex parsed all 2-backtick table rows, catching `bool`/`float` from 3-column tables. Tightened to require a 3rd backtick column to scope to 5-column section tables only.

## Known Issues

None.

## Files Created/Modified

- `tests/test_config_docs.py` — new; 4 test cases enforcing doc-schema sync
- `nce/config_schema.py` — added doc pointer comment after module docstring
- `configs/example_nn_config.py` — rewritten with `flat_config` and `nested_config` examples
