# S02: Benchmark Config Migration

**Goal:** All benchmark sets offer nested config builders alongside flat; dead fields removed from flat builders; experiment_config.py validates nn_configs through config_schema.
**Demo:** Import `nbe_sanity_check`, access `configs['nbe_nested']`, pass through `prepare_config()` — get identical flat output to `configs['nbe']`. Worker's `build_nn_config()` output passes `prepare_config()` validation.

## Must-Haves

- Dead fields (`backward_ecl`, `num_batches_per_set`) removed from `_build_nbe_configs()` and `_build_default_configs()`
- `set_bw_ecl()` no longer sets `backward_ecl` (keeps `bw_ecl` and `populate_bw_factors`)
- `_build_nbe_nested_configs()` added to `nbe_sanity_check.py`, registered under `configs['nbe_nested']`
- `_build_default_nested_configs()` added to `small_problems.py`, registered under `configs['default_nested']`
- Nested builder output → `prepare_config()` → flat dict equals flat builder output → `prepare_config()` → flat dict (round-trip equality)
- Worker's `build_nn_config()` produces configs that pass `prepare_config()` without warnings (dead fields removed from worker too)
- Existing tests updated to reflect clean benchmark configs; dead field stripping mechanism tests kept with synthetic configs

## Proof Level

- This slice proves: integration
- Real runtime required: no (config validation is pure Python, no model loading needed for config tests)
- Human/UAT required: no

## Verification

- `python -m pytest tests/test_config_schema.py tests/test_benchmark_configs.py -v` — all pass
- New test file `tests/test_benchmark_configs.py` covers:
  - Dead fields absent from flat benchmark configs (no warnings emitted)
  - Nested builder configs validate through `prepare_config()` without error
  - Round-trip equality: nested→flat == flat→flat for both benchmark sets
  - `set_bw_ecl()` does not write `backward_ecl`
  - Worker's `build_nn_config()` passes `prepare_config()` without warnings
- Existing `tests/test_config_schema.py` still passes (dead field stripping tests use synthetic configs, not benchmark configs)

## Observability / Diagnostics

- Runtime signals: `prepare_config()` emits `UserWarning` for dead fields in non-strict mode — if any benchmark config still has dead fields after S02, warnings will surface in test output with `-W error`
- Inspection surfaces: `python -m pytest tests/test_benchmark_configs.py -v` shows per-test pass/fail for all benchmark config behaviors
- Failure visibility: `prepare_config(config, strict=True)` will raise `ValueError` naming the offending dead field if cleanup was incomplete
- Redaction constraints: none

## Integration Closure

- Upstream surfaces consumed: `nce/config_schema.py` → `prepare_config()`, `NESTED_SECTIONS`, `DEAD_FIELDS` (all from S01)
- New wiring introduced in this slice: benchmark modules register nested configs in `BenchmarkSet.configs`; worker.py imports and calls `prepare_config()` on assembled nn_config
- What remains before the milestone is truly usable end-to-end: S03 (config docs), S04 (state preservation), S05 (visualization), S06 (logging), S07 (regression verification)

## Tasks

- [x] **T01: Remove dead fields from benchmark configs and fix set_bw_ecl** `est:30m`
  - Why: Benchmark flat configs currently carry `backward_ecl` and `num_batches_per_set` (dead fields per D010). Cleaning them is prerequisite to T02's round-trip equality test and moves toward D014's eventual strict-by-default.
  - Files: `nce/benchmark_problems/nbe_sanity_check.py`, `nce/benchmark_problems/small_problems.py`, `tests/test_benchmark_configs.py`, `tests/test_config_schema.py`
  - Do: (1) Remove `backward_ecl` and `num_batches_per_set` lines from `_build_nbe_configs()`. (2) Same for `_build_default_configs()`. (3) Remove `cfg['backward_ecl'] = value` line from `set_bw_ecl()`, keep `bw_ecl` and `populate_bw_factors` assignments. (4) Create `tests/test_benchmark_configs.py` with tests asserting: dead fields absent from both builders' output, `set_bw_ecl()` does not write `backward_ecl`, all benchmark flat configs pass `prepare_config()` with no warnings. (5) Verify existing `tests/test_config_schema.py` tests still pass — the `TestBenchmarkPassthrough` tests import `_build_nbe_configs()` which will now produce clean configs; update assertions if they explicitly check for dead field stripping behavior on benchmark configs.
  - Verify: `python -m pytest tests/test_benchmark_configs.py tests/test_config_schema.py -v`
  - Done when: Both benchmark builders produce configs without dead fields; `set_bw_ecl()` only writes `bw_ecl` and `populate_bw_factors`; all tests pass.

- [ ] **T02: Add nested config builders with round-trip equality tests** `est:45m`
  - Why: Core deliverable — nested builders provide the new config format for benchmark sets, proving the S01 schema works with real configs. Round-trip equality is the primary correctness check.
  - Files: `nce/benchmark_problems/nbe_sanity_check.py`, `nce/benchmark_problems/small_problems.py`, `tests/test_benchmark_configs.py`
  - Do: (1) Add `_build_nbe_nested_configs()` to `nbe_sanity_check.py` — returns list of 5 nested config dicts using NESTED_SECTIONS format with readable names. Use `equivalent_nested_config` fixture from conftest.py as structural template. Per-model maps (`_HIDDEN_SIZES_MAP`, etc.) parameterize values just like the flat builder. (2) Register under `configs['nbe_nested']` in the module-level `BenchmarkSet`. (3) Add `_build_default_nested_configs()` to `small_problems.py` — same pattern, 24 configs. Register under `configs['default_nested']`. (4) Add round-trip equality tests in `tests/test_benchmark_configs.py`: for each model index, `prepare_config(nested_config) == prepare_config(flat_config)`. Test both benchmark sets. (5) Keep nested builders lightweight — no torch imports, no `prepare_config()` calls at module level (research constraint: module-level BenchmarkSet triggers at import time).
  - Verify: `python -m pytest tests/test_benchmark_configs.py -v -k nested`
  - Done when: `configs['nbe_nested']` and `configs['default_nested']` exist on both benchmark sets; round-trip equality tests pass for all models in both sets.

- [ ] **T03: Wire experiment_config worker to validate through config_schema** `est:30m`
  - Why: Integration closure — worker.py assembles nn_config dicts with dead fields. Cleaning and adding a `prepare_config()` validation step catches problems at config-build time rather than at FastGM init.
  - Files: `notebooks/_1-2026/worker.py`, `notebooks/_1-2026/experiment_config.py`, `tests/test_benchmark_configs.py`
  - Do: (1) Remove `backward_ecl` and `num_batches_per_set` from `build_nn_config()` in worker.py. (2) Add `from nce.config_schema import prepare_config` to worker.py. (3) Add a validation call at the end of `build_nn_config()`: `prepare_config(nn_config, strict=True)` — if the assembled config has dead fields or unknown fields, it fails fast with a clear error. Return the original `nn_config` dict (not the prepared one, since FastGM will call `prepare_config` again at init — double-preparation is idempotent but the worker config has fields like `error_tracking` that aren't in the schema and are consumed separately). Actually: the flat config may have fields unknown to the schema — validate with `strict=False` to catch only dead fields. (4) Remove `num_batches_per_set` from `validate_config()` defaults in experiment_config.py. (5) Add test in `tests/test_benchmark_configs.py`: build a sample nn_config via `build_nn_config()` with test inputs, pass through `prepare_config()`, assert no warnings emitted.
  - Verify: `python -m pytest tests/test_benchmark_configs.py -v -k worker`
  - Done when: Worker's `build_nn_config()` output contains no dead fields and passes `prepare_config()` validation without warnings.

## Files Likely Touched

- `nce/benchmark_problems/nbe_sanity_check.py`
- `nce/benchmark_problems/small_problems.py`
- `notebooks/_1-2026/worker.py`
- `notebooks/_1-2026/experiment_config.py`
- `tests/test_benchmark_configs.py` (new)
- `tests/test_config_schema.py` (minor updates)
