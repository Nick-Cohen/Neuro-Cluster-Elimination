---
estimated_steps: 5
estimated_files: 4
---

# T01: Set up pytest and write failing acceptance tests

**Slice:** S01 — Config Schema & Flat Translation
**Milestone:** M001

## Description

Install pytest and create the complete test suite that defines S01's acceptance criteria. Every requirement (R001–R006) gets encoded as executable test assertions. Tests import from `nce.config_schema` which doesn't exist yet — they should all fail on import error, confirming the test harness works and the stopping condition is clear.

## Steps

1. Install pytest in the project venv: `source venv/bin/activate && pip install pytest`
2. Create `tests/__init__.py` (empty) and `tests/conftest.py` with shared fixtures:
   - `reference_flat_config`: a complete flat config dict matching the nbe_sanity_check template (all 42 fields)
   - `equivalent_nested_config`: the same config expressed in nested section format
   - `minimal_flat_config`: bare minimum flat config (just required fields for nn mode)
   - `minimal_nested_config`: bare minimum nested config
3. Create `tests/test_config_schema.py` with test functions covering:
   - `test_nested_to_flat_translation` — nested config flattens to correct internal keys (R001)
   - `test_flat_passthrough` — flat config passes through with all fields preserved (R005)
   - `test_auto_detect_nested` — nested config detected correctly (R005)
   - `test_auto_detect_flat` — flat config detected correctly (R005)
   - `test_dead_field_error` — `backward_ecl` and `num_batches_per_set` raise ValueError (R002)
   - `test_dead_field_error_message` — error message names the dead field and suggests alternative (R002)
   - `test_alias_resolution` — `learning_rate` resolves to `lr`, `exact_computation_limit` to `ecl`, etc. (R003)
   - `test_unknown_nested_field_error` — unknown field in a nested section raises ValueError naming the section (R004)
   - `test_unknown_section_error` — unknown top-level section in nested config raises ValueError (R004)
   - `test_required_fields_nn` — missing `loss_fn` or `num_epochs` when `approximation_method='nn'` raises ValueError (R004)
   - `test_polymorphic_types` — `hidden_sizes` as list/str/`'bias_only'`, `batch_size` as int/`'all'`, etc. all accepted (R001)
   - `test_prepare_config_returns_dict` — return value is a plain `dict` instance, mutable (R006)
   - `test_prepare_config_with_none` — `prepare_config(None)` returns empty dict or minimal defaults
   - `test_benchmark_config_passthrough` — import actual nbe_sanity_check config, run through `prepare_config`, verify all expected fields present with correct values (R005)
   - `test_nested_and_flat_produce_same_result` — equivalent nested and flat configs produce identical flat output (R001, R005)
4. Verify pytest collects all tests: `python -m pytest tests/test_config_schema.py --co`
5. Verify tests fail (expected — `nce.config_schema` doesn't exist): `python -m pytest tests/test_config_schema.py` should show import errors

## Must-Haves

- [ ] pytest installed and importable in venv
- [ ] `tests/test_config_schema.py` has 12+ test functions
- [ ] Every requirement R001–R006 has at least one test function
- [ ] Tests import from `nce.config_schema` (the module T02 will create)
- [ ] Fixtures provide reference flat and nested config dicts matching the research field inventory

## Verification

- `source venv/bin/activate && python -m pytest tests/test_config_schema.py --co` collects 12+ tests
- `python -m pytest tests/test_config_schema.py` fails with ImportError (expected)

## Observability Impact

- Signals added/changed: pytest test names serve as executable specification — each test name maps to a requirement
- How a future agent inspects this: `python -m pytest tests/test_config_schema.py -v` shows pass/fail per requirement
- Failure state exposed: pytest output shows exactly which requirement's test failed and why

## Inputs

- S01-RESEARCH.md field inventory (all 55 fields across 6 sections) — drives fixture construction
- `nce/benchmark_problems/nbe_sanity_check.py` — provides real flat config for benchmark passthrough test
- D007 — field name mappings for alias tests
- D009 — pytest as test framework

## Expected Output

- `tests/__init__.py` — empty package init
- `tests/conftest.py` — shared fixtures with reference configs
- `tests/test_config_schema.py` — 12+ failing test functions encoding R001–R006
