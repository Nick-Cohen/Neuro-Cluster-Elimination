---
estimated_steps: 5
estimated_files: 4
---

# T02: Add nested config builders with round-trip equality tests

**Slice:** S02 — Benchmark Config Migration
**Milestone:** M001

## Description

Add `_build_nbe_nested_configs()` and `_build_default_nested_configs()` functions that produce nested-format config dicts using the 6-section schema from `config_schema.NESTED_SECTIONS`. Register them in `BenchmarkSet.configs` under `'nbe_nested'` and `'default_nested'` keys. Add round-trip equality tests proving `prepare_config(nested) == prepare_config(flat)` for every model in both benchmark sets.

## Steps

1. Add `_build_nbe_nested_configs()` in `nce/benchmark_problems/nbe_sanity_check.py`:
   - Returns list of 5 nested config dicts (one per model, same order as `_MODEL_KEYS`).
   - Uses 6 sections: `inference`, `nn`, `training`, `sampling`, `backward`, `output`.
   - Uses readable names where available (e.g., `exact_computation_limit` not `ecl`, `learning_rate` not `lr`, `forward_diff_barrier` not `fdb`, `backward_ecl` not `bw_ecl` in backward section — per research: `backward_ecl` inside backward section is a live alias mapping to `bw_ecl`, NOT a dead field).
   - Parameterized by the same per-model maps (`_HIDDEN_SIZES_MAP`, `_NUM_SAMPLES_MAP`, `_IB_MAP`).
   - No dead fields. No `prepare_config()` call at module level (keep lightweight).
   - Use `tests/conftest.py::equivalent_nested_config` as structural reference.
2. Register `'nbe_nested': _build_nbe_nested_configs()` in the module-level `BenchmarkSet` constructor.
3. Add `_build_default_nested_configs()` in `nce/benchmark_problems/small_problems.py`:
   - Same pattern, 24 configs. Uses `_get_auto_ecl()` for per-model ecl values.
   - Uses readable names in nested sections.
4. Register `'default_nested': _build_default_nested_configs()` in the module-level `BenchmarkSet`.
5. Add round-trip equality tests in `tests/test_benchmark_configs.py`:
   - `TestNestedBuilderRoundTrip`: for each model in nbe_sanity_check, assert `prepare_config(configs['nbe_nested'][i]) == prepare_config(configs['nbe'][i])`. Same for small_problems `default_nested` vs `default`.
   - `TestNestedBuilderValidation`: each nested config passes `prepare_config(config, strict=True)` without error (nested configs have no dead fields and should pass strict validation).
   - Parametrize tests over model index for clear per-model failure messages.

## Must-Haves

- [ ] `_build_nbe_nested_configs()` exists and returns 5 nested config dicts
- [ ] `configs['nbe_nested']` accessible on `nbe_sanity_check` BenchmarkSet instance
- [ ] `_build_default_nested_configs()` exists and returns 24 nested config dicts
- [ ] `configs['default_nested']` accessible on `small_problems` BenchmarkSet instance
- [ ] Round-trip equality: `prepare_config(nested[i]) == prepare_config(flat[i])` for all models in both sets
- [ ] Nested configs pass `prepare_config(strict=True)` without error
- [ ] No `prepare_config()` calls at module level in benchmark modules

## Verification

- `python -m pytest tests/test_benchmark_configs.py -v -k nested` — all nested-related tests pass
- `python -m pytest tests/test_benchmark_configs.py tests/test_config_schema.py -v` — full suite still green

## Observability Impact

- Signals added/changed: None runtime. Test output clearly identifies which model index fails round-trip equality if any drift occurs between flat and nested builders.
- How a future agent inspects this: Run the round-trip tests — they're parametrized per model index, so failures name the exact model.
- Failure state exposed: pytest parametrize IDs show `[nbe-0]`, `[nbe-1]`, etc. and `[default-0]` through `[default-23]` for per-model identification.

## Inputs

- `nce/benchmark_problems/nbe_sanity_check.py` — T01-cleaned flat builder (no dead fields)
- `nce/benchmark_problems/small_problems.py` — T01-cleaned flat builder (no dead fields)
- `nce/config_schema.py` — `prepare_config()`, `NESTED_SECTIONS` (for section/field reference)
- `tests/conftest.py` — `equivalent_nested_config` fixture as structural template
- `tests/test_benchmark_configs.py` — T01-created test file to extend

## Expected Output

- `nce/benchmark_problems/nbe_sanity_check.py` — `_build_nbe_nested_configs()` added, registered in BenchmarkSet
- `nce/benchmark_problems/small_problems.py` — `_build_default_nested_configs()` added, registered in BenchmarkSet
- `tests/test_benchmark_configs.py` — round-trip equality tests and nested validation tests added
