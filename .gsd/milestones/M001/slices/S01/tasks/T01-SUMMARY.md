---
id: T01
parent: S01
milestone: M001
provides:
  - pytest test harness with 22 acceptance tests encoding R001–R006
  - shared fixtures with reference flat, nested, minimal configs
key_files:
  - tests/__init__.py
  - tests/conftest.py
  - tests/test_config_schema.py
key_decisions:
  - Dead field tests split into strict error tests (R002) and benchmark passthrough test (D011 warn+strip)
  - 22 tests total (exceeds 12+ requirement) to cover edge cases in each requirement
patterns_established:
  - Test classes grouped by requirement (TestNestedToFlatTranslation, TestDeadFields, etc.)
  - Fixtures in conftest.py provide reference_flat_config (42 fields matching nbe_sanity_check), equivalent_nested_config (6 sections with alias names), and minimal variants
observability_surfaces:
  - "python -m pytest tests/test_config_schema.py -v" shows pass/fail per requirement
duration: 15m
verification_result: passed
completed_at: 2026-03-12
blocker_discovered: false
---

# T01: Set up pytest and write failing acceptance tests

**Installed pytest, created 22-test acceptance suite encoding all S01 requirements against the not-yet-created `nce.config_schema` module.**

## What Happened

Installed pytest 9.0.2 in the project venv. Created `tests/` package with `conftest.py` (4 fixtures) and `test_config_schema.py` (22 test functions in 7 test classes). Tests import `prepare_config` from `nce.config_schema` — which doesn't exist yet — so they all fail with `ModuleNotFoundError`, confirming the stopping condition is clear.

Fixtures:
- `reference_flat_config`: 42-field dict matching nbe_sanity_check pedigree13 config (including dead fields)
- `equivalent_nested_config`: same values in 6 nested sections using readable alias names from D007
- `minimal_flat_config` / `minimal_nested_config`: bare minimum for nn mode

Test coverage by requirement:
- R001 (nested→flat translation): 4 tests including polymorphic types
- R002 (dead fields): 4 tests covering both dead fields, error messages, and nested context
- R003 (aliases): 3 tests for flat aliases, nested aliases, and internal name passthrough
- R004 (validation errors): 4 tests for unknown fields, unknown sections, missing required fields
- R005 (flat passthrough + detection): 3 tests for passthrough, flat detection, nested detection
- R006 (return type): 3 tests for dict type, None input, empty dict input
- Benchmark integration: 1 test importing real nbe_sanity_check config

## Verification

- `python -m pytest tests/test_config_schema.py --co` with temporary stub: 22 tests collected
- `python -m pytest tests/test_config_schema.py`: fails with `ModuleNotFoundError: No module named 'nce.config_schema'` (expected)
- AST parse confirms 22 test functions across 7 classes

Slice-level verification (partial — intermediate task):
- ✅ pytest installed
- ✅ 22 tests collected (exceeds 12+ requirement)
- ✅ All requirements R001–R006 have test coverage
- ⏳ All tests passing — blocked on T02 (implementation) and T03 (integration)

## Diagnostics

- Run `python -m pytest tests/test_config_schema.py -v` to see per-test pass/fail status
- Each test class maps to a requirement for easy traceability
- Fixture configs match the field inventory in S01-RESEARCH.md

## Deviations

- Plan said `--co` should collect tests. Top-level `from nce.config_schema import prepare_config` prevents collection when module doesn't exist. Verified collection works with a temporary stub (22 collected), then removed stub. The ImportError on run is the intended failure mode.

## Known Issues

None.

## Files Created/Modified

- `tests/__init__.py` — empty package init
- `tests/conftest.py` — shared fixtures with 4 reference config dicts
- `tests/test_config_schema.py` — 22 acceptance tests across 7 classes covering R001–R006
