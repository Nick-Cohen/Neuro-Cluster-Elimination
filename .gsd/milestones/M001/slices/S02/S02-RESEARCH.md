# S02: Benchmark Config Migration — Research

**Date:** 2026-03-12

## Summary

S02's scope is narrow and low-risk: clean dead fields from the two benchmark modules, add nested config builder functions alongside the existing flat builders, and wire `experiment_config.py` to validate nn_configs through `config_schema.prepare_config()`.

The two benchmark modules (`nbe_sanity_check.py`, `small_problems.py`) each have a `_build_*_configs()` function producing flat dicts with 2 dead fields (`backward_ecl`, `num_batches_per_set`). These are the only two dead fields (per D010). Removing them is a one-line-per-field change. Adding nested builders means writing a parallel `_build_*_nested_configs()` function that uses the 6-section format from config_schema's `NESTED_SECTIONS`. The `set_bw_ecl()` helper in `small_problems.py` also sets `backward_ecl`, which needs fixing.

The `experiment_config.py` in `notebooks/_1-2026/` has its own `validate_config()` that knows nothing about `config_schema`. It validates experiment-level fields (problem, loss, architectures, gpus) not nn_config fields. The worker's `build_nn_config()` assembles flat nn_config dicts including dead fields (`backward_ecl`, `num_batches_per_set`). The experiment_config integration should add a step that validates the assembled nn_config through `prepare_config()` rather than replacing its experiment-level validation.

## Recommendation

Three tasks:

1. **Clean dead fields from benchmark configs** — Remove `backward_ecl` and `num_batches_per_set` from `_build_nbe_configs()` and `_build_default_configs()`. Fix `set_bw_ecl()` to stop setting `backward_ecl`. Update existing tests that assert dead field behavior on benchmark configs (they currently expect stripping; after cleanup they should verify absence without warnings).

2. **Add nested config builders** — Add `_build_nbe_nested_configs()` in `nbe_sanity_check.py` and `_build_default_nested_configs()` in `small_problems.py`. Register them in `BenchmarkSet.configs` under new keys (e.g., `'nbe_nested'`, `'default_nested'`). Add a round-trip test: nested builder output → `prepare_config()` → compare with flat builder output → `prepare_config()` → assert equality.

3. **Wire experiment_config.py to config_schema** — Add a validation step in `worker.py`'s `build_nn_config()` or in `experiment_config.py` that calls `prepare_config()` on the assembled nn_config. This catches dead fields and invalid keys at config-build time rather than at FastGM init time. The experiment-level validation (problem path, GPU list, etc.) stays in `experiment_config.py` since those aren't nn_config fields.

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| Config validation/flattening | `nce.config_schema.prepare_config()` | Already handles dead fields, aliases, nested→flat — don't duplicate |
| Nested section structure | `nce.config_schema.NESTED_SECTIONS` | Single source of truth for field placement in sections |
| Benchmark set pattern | `BenchmarkSet` class in `nbe_sanity_check.py` | Established pattern: `.problems` list + `.configs` dict of config lists |

## Existing Code and Patterns

- `nce/benchmark_problems/nbe_sanity_check.py` — `_build_nbe_configs()` returns list of 5 flat config dicts. Per-model maps (`_HIDDEN_SIZES_MAP`, `_NUM_SAMPLES_MAP`, `_IB_MAP`) parameterize configs. Dead fields: `backward_ecl` (line 113), `num_batches_per_set` (line 102).
- `nce/benchmark_problems/small_problems.py` — `_build_default_configs()` returns list of 24 flat config dicts. Uses `_AUTO_ECL` map. Dead fields: `backward_ecl` (line 140), `num_batches_per_set` (line 129). `set_bw_ecl()` helper sets `backward_ecl` (line 170) — dead field write that must be removed.
- `nce/config_schema.py` — `prepare_config()` is the public entry point (D001). `NESTED_SECTIONS` defines the 6-section schema. `flatten_config()` handles nested→flat. `DEAD_FIELDS` has the two dead field entries.
- `tests/conftest.py` — Reference flat config fixture includes dead fields; equivalent nested config fixture is clean. The `equivalent_nested_config` fixture is a good template for nested builder structure.
- `tests/test_config_schema.py` — `TestBenchmarkPassthrough` asserts dead fields are stripped. After cleanup, these tests need updating to not expect dead fields in input.
- `notebooks/_1-2026/worker.py` — `build_nn_config()` assembles flat nn_config with `backward_ecl` (dead) and `num_batches_per_set` (dead). Not part of nce package, but a consumer.
- `notebooks/_1-2026/experiment_config.py` — YAML config loader with its own `validate_config()`. Validates experiment-level fields (problem, loss, architectures, bw_ecl, gpus), not nn_config internals. No current import of `config_schema`.

## Constraints

- **Backward compat for flat configs is already handled** — `prepare_config(strict=False)` warn+strips dead fields. After S02 cleans benchmark configs, the dead fields simply won't be present. External code that still uses dead fields still gets warnings, not errors.
- **BenchmarkSet instantiation happens at module import** — Both benchmark modules create their module-level `BenchmarkSet` instance at import time, which triggers model catalogue loading. Tests must handle the case where model files aren't cached.
- **experiment_config.py lives in notebooks/, not nce/** — It's not part of the installed package. Changes there are local to the experiment runner workflow. It can import from `nce.config_schema` since NCE is on the path.
- **Nested builders must produce configs that round-trip identically through prepare_config** — The nested config → `prepare_config()` → flat dict must equal the flat config → `prepare_config()` → flat dict. This is the primary correctness check.
- **Per D014, `prepare_config` defaults to `strict=False`** — After cleaning benchmark configs of dead fields, we could flip to `strict=True` default, but that's a separate decision (D014 notes this possibility). S02 cleans configs; flipping the default is out of scope.

## Common Pitfalls

- **`backward_ecl` is a dead field name AND a live readable alias** — In the `backward` section of a nested config, `backward_ecl` is a readable name that maps to internal `bw_ecl`. It's only dead as a *flat top-level key*. The nested builder should use `backward_ecl` (or `bw_ecl`) inside the `backward` section — that's fine. The flat builder should use `bw_ecl`, not `backward_ecl`.
- **`set_bw_ecl()` modifies configs in-place** — It mutates the config dicts stored in `BenchmarkSet.configs`. The fix must remove the `backward_ecl` assignment (line 170) but keep the `bw_ecl` and `populate_bw_factors` assignments which are live.
- **Module-level BenchmarkSet triggers catalogue loading** — Adding nested configs to the `.configs` dict means the nested builder also runs at import time. Keep it lightweight (no torch imports, no validation calls at module level).
- **Test fixtures reference dead fields** — `conftest.py`'s `reference_flat_config` includes dead fields by design (to test stripping). After S02, the *benchmark config tests* should test clean configs, but the *dead field stripping tests* should keep their synthetic configs with dead fields to verify the stripping mechanism still works.

## Open Risks

- **Notebook code (`worker.py`, `config.py`) won't auto-update** — These files build flat configs with dead fields and aren't part of the nce package. S02 should update them for consistency but they're living notebooks — the user may have modified them since. Should document what changed and why, and make the changes conservative.
- **Nested config builder maintenance burden** — Two parallel config builders (flat + nested) per benchmark set doubles the maintenance surface. Mitigated by the fact that nested→flat round-trip tests will catch drift. Could later deprecate flat builders once all consumers migrate.

## Skills Discovered

| Technology | Skill | Status |
|------------|-------|--------|
| Python/pytest | N/A — standard library tooling, no skill needed | N/A |
| pyGMs | N/A — only used via existing catalogue utils | N/A |

No external technologies requiring specialized skills. This is pure Python config restructuring within the existing codebase patterns.

## Sources

- `nce/config_schema.py` — schema definition and prepare_config (S01 deliverable)
- `nce/benchmark_problems/nbe_sanity_check.py` — primary benchmark module
- `nce/benchmark_problems/small_problems.py` — secondary benchmark module  
- `tests/conftest.py` + `tests/test_config_schema.py` — existing test infrastructure
- `.gsd/DECISIONS.md` — D010 (dead field list), D011 (dead field handling), D014 (strict default)
- S01 task summaries (T01-T03) — what was delivered
