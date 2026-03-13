# S07: Regression Verification — Research

**Date:** 2026-03-12

## Summary

S07 owns R017: a one-command regression test that translates flat→nested config, runs inference on a reference problem, and confirms identical partition function estimates. The infrastructure for this is well-positioned: `prepare_config()` already handles both config formats, both benchmark sets (`nbe_sanity_check`, `small_problems`) already ship flat and nested config builders, and the existing test suite in `tests/` provides a pytest pattern to follow.

Empirical validation confirms the approach works: running `FastGM` with flat vs nested configs on `rbm_20` (the only nbe_sanity_check model that loads without evidence-related bucket errors) produces bitwise-identical `log_partition_function` values in both exact-only mode (~3.7s) and NN mode with 2 epochs + 500 samples (~6s first run). The deterministic seeding in `SampleGenerator` ensures identical training data paths.

The deliverable is two things: (1) a standalone `scripts/regression_test.py` script that can be run one-command from the terminal with pass/fail output, and (2) a pytest test in `tests/test_regression.py` for CI integration. Both verify the same thing: `prepare_config(flat) == prepare_config(nested)` at the config level AND `FastGM(flat).get_log_partition_function() == FastGM(nested).get_log_partition_function()` at the inference level.

## Recommendation

**Two-tier test: exact + NN, both using rbm_20 as reference model.**

- **Exact-only test** (high ecl, no NN buckets): ~4s, verifies config translation doesn't affect factor operations. This is the fast, always-passing sanity check.
- **NN test** (2 epochs, 500 samples): ~6s, verifies config translation doesn't affect training pipeline. Tests the full path through sampling, training, and message computation.

Both should assert bitwise equality (`==`), not approximate equality — verified empirically that identical configs through different paths produce identical floats. If CUDA non-determinism becomes an issue in the future, we can relax to a tolerance, but start strict.

Use `rbm_20` (nbe_sanity_check model index 3) as the reference problem: it's the smallest model that loads cleanly (40 vars, 20 NN buckets), runs quickly, and has known-good flat+nested config builders.

The standalone script should be runnable as `python scripts/regression_test.py` with zero arguments, printing PASS/FAIL and exiting with code 0/1. The pytest version should be runnable as `pytest tests/test_regression.py -v`.

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| Config translation | `prepare_config()` in `config_schema.py` | Already handles flat/nested detection, validation, alias resolution, flattening |
| Reference model loading | `nbe_sanity_check.problems[3]` / `nbe_sanity_check.configs['nbe'][3]` | Provides rbm_20 model object + matching flat AND nested configs |
| Nested config equivalents | `nbe_sanity_check.configs['nbe_nested'][3]` | Pre-built nested config that's been verified to round-trip identically to flat |
| Inference execution | `FastGM(model=m, nn_config=cfg).get_log_partition_function()` | Standard inference path, handles all bucketing and elimination |
| Config equality check | `prepare_config(flat) == prepare_config(nested)` | Already tested in `TestNestedBuilderRoundTrip` (S02); regression test extends to runtime |

## Existing Code and Patterns

- `nce/config_schema.py` — `prepare_config()` is the single entry point for config normalization. Auto-detects flat vs nested, validates, resolves aliases, flattens. Returns a plain dict with internal key names.
- `nce/benchmark_problems/nbe_sanity_check.py` — Ships both `_build_nbe_configs()` (flat) and `_build_nbe_nested_configs()` (nested) for 5 models. Module-level `nbe_sanity_check` instance provides `.problems` and `.configs['nbe']` / `.configs['nbe_nested']`.
- `nce/inference/graphical_model.py` — `FastGM.__init__` calls `prepare_config(nn_config)` at line 38. `get_log_partition_function()` runs full elimination and returns the scalar partition function.
- `tests/test_benchmark_configs.py::TestNestedBuilderRoundTrip` — Already verifies `prepare_config(flat) == prepare_config(nested)` for all 29 models. S07 extends this to verify inference output identity, not just config dict identity.
- `tests/conftest.py` — Existing fixtures for reference flat/nested configs (pedigree13 template). S07 tests will use `nbe_sanity_check` directly rather than fixtures since they need actual model objects.
- `nce/sampling/sample_generator.py` — Uses `_compute_seed()` with deterministic seed generation from `bucket_id + random_seed * 10000 + counter * 100`. This is why identical configs produce identical training data.

## Constraints

- **rbm_20 is the only usable nbe_sanity_check model** — pedigree13, grid40x40, grid20x20 work but are too large (134-308 NN buckets). grid10x10 fails with unplaced factors (evidence conditioning issue). All 24 small_problems models also fail with the same unplaced-factor error.
- **CUDA required** — benchmark configs specify `device='cuda'`. The test should check for CUDA availability and skip gracefully if unavailable.
- **NN test takes ~6s** — acceptable for a regression test run on demand, but should be marked appropriately if added to a fast test suite.
- **No `torch.use_deterministic_algorithms`** — the codebase doesn't set this globally. Bitwise equality works today without it, but is technically fragile across PyTorch versions. If it ever breaks, relax to tolerance `atol=1e-6`.
- **`prepare_config` defaults to `strict=False`** (D014) — regression test should test with default strictness, matching real usage.

## Common Pitfalls

- **Using `nbe,X` string formats for num_samples/hidden_sizes in custom configs** — these are resolved dynamically by bucket.py based on bucket width. If you construct a custom config, use explicit integer values to avoid resolution differences. The benchmark configs already handle this correctly.
- **Mutating shared config dicts** — `prepare_config()` makes a shallow copy, but the benchmark config builders return fresh dicts each call via `_build_nbe_configs()`. However, the module-level `nbe_sanity_check.configs['nbe']` list is shared state. Always `dict(cfg)` before modifying.
- **Forgetting to reset CUDA state between runs** — if the first run leaves GPU memory in a bad state, the second run could behave differently. This hasn't been observed in practice (both runs produce identical results), but worth noting.
- **Running on wrong Python** — must use `/home/cohenn1/NCE/venv/bin/python`, not system Python (lacks torch).

## Open Risks

- **Evidence-related bucket placement failures** — most benchmark models fail with "Some factors could not be placed in buckets" when evidence variables create single-variable factors that don't match the elimination order. This limits the reference model to rbm_20. If rbm_20 ever changes or becomes unavailable, the test needs a different reference.
- **CUDA non-determinism across hardware/driver versions** — the bitwise equality assertion works on the current setup but could fail on different GPU hardware or CUDA versions. The tolerance fallback strategy is documented but not yet needed.
- **Config coverage gap** — the test only proves flat→nested equivalence for one specific config (rbm_20's nbe config). It doesn't test all possible config combinations. This is acceptable for R017's scope — the config-level round-trip is already exhaustively tested in S02's `TestNestedBuilderRoundTrip`.

## Skills Discovered

| Technology | Skill | Status |
|------------|-------|--------|
| PyTorch | N/A | No relevant skill — standard usage, no special tooling needed |
| pytest | N/A | No relevant skill — standard test patterns sufficient |

## Sources

- Empirical validation via direct Python execution on the dev machine (all runtime numbers measured live)
- `nce/config_schema.py` source code — prepare_config API and validation logic
- `nce/benchmark_problems/nbe_sanity_check.py` — config builder patterns and model inventory
- `tests/test_benchmark_configs.py` — existing round-trip test pattern (TestNestedBuilderRoundTrip)
- `notebooks/_1-2026/test_reproducibility.py` — prior reproducibility testing pattern for reference
