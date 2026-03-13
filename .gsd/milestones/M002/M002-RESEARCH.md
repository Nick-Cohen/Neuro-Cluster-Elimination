# M002: Test Suite — Research

**Date:** 2026-03-12

## Summary

The test infrastructure from M001 is solid — 110 pytest tests covering config schema, regression, and benchmark configs, running in ~41 seconds. M002 needs to add a different *kind* of test: inference correctness, NN training functionality, convergence, and robustness. These test real algorithm execution paths rather than config plumbing.

The primary challenge is building fast, self-contained test fixtures. Hand-built `FastFactor` problems work well for exact inference (< 10ms per test), but NN training requires complete configs with all ~40 fields. The existing benchmark config builders (`_build_nbe_configs()`) provide complete configs but tie tests to catalog models that need network downloads. The recommended approach is to build a `conftest.py` fixture that returns a complete minimal config dict for CPU-based NN training, combined with hand-built `FastFactor` problems small enough to have known exact solutions.

Edge case testing (R022/R023) revealed that *all* major loss functions produce inf or NaN when given all-neg-inf targets — `logspace_mse_fdb` returns inf, while `linspace_mse_fdb`, `unnormalized_kl`, and `weighted_logspace_mse` return NaN. This is a real behavior that should be documented/tested rather than treated as a bug, since all-neg-inf targets represent deterministic messages where every assignment has zero probability. Tests should verify the code doesn't crash (no unhandled exceptions) and should document which loss functions handle this gracefully.

## Recommendation

Build a single slice (S01) delivering all 7 requirements in one pass, structured as 3-4 tasks:

1. **Test fixtures** — conftest.py with complete NN config fixture, hand-built factor problems (binary chain, ternary chain, star graph), and GPU skip markers
2. **Correctness + functional tests** (R018, R019, R020) — exact inference on hand-built problems, single-bucket NN training, multi-domain variable handling
3. **Convergence + robustness tests** (R021, R022, R023) — loss decrease over 50 epochs, inf-output checks, edge-case target handling
4. **Extensible pattern** (R024) — documented pattern for adding failure-mode regression tests

Total estimated test runtime: < 60 seconds (hand-built problems + CPU-only NN training with small epoch counts).

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| Test config completeness | Copy pattern from `_build_nbe_configs()` | Need all ~40 config fields or Trainer crashes on missing `lower_dim`, `debug`, etc. |
| GPU availability detection | `pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")` | Already used in test_regression.py |
| Small test problems | Hand-build `FastFactor` chains/trees with known Z values | Catalog models require network + disk I/O; synthetic problems are deterministic and instant |
| Config plumbing fixtures | Extend existing `tests/conftest.py` | Already has `reference_flat_config`, `minimal_flat_config`, etc. |

## Existing Code and Patterns

- `tests/conftest.py` — Existing fixtures for config schema tests. Has `reference_flat_config` (42 fields) and `minimal_flat_config` (7 fields). The minimal config is too minimal for NN training — needs `lower_dim`, `debug`, `optimizer`, `sampling_scheme`, etc. **Extend with a new `nn_training_config` fixture.**
- `tests/test_regression.py` — Best existing pattern for inference tests. Uses `FastGM(model=..., nn_config=..., device=...)` → `eliminate_variables(all=True)` → check `log_partition_function`. Uses `pytest.mark.skipif` for GPU tests. **Reuse this pattern.**
- `nce/inference/factor.py` — `FastFactor(tensor, labels)` with log-space tensors. Can build test problems as `FastFactor(torch.log10(tensor), labels)`. **Use for hand-built test problems.**
- `nce/inference/graphical_model.py` — `FastGM(factors=..., elim_order=..., nn_config=..., device=...)` accepts a list of `FastFactor` objects. No model file needed. **Use for constructing test GMs from hand-built factors.**
- `nce/neural_networks/losses.py` — Direct function calls: `loss_fn(outputs, targets)`. Can test robustness by calling loss functions directly with edge-case inputs. **Test R022/R023 at the loss function level.**
- `nce/neural_networks/train.py` — `Trainer.__init__` accesses `self.config['lower_dim']`, `self.config['debug']`, etc. with bare dict access (not `.get()`). **Any config used for NN training must include these fields.**
- `nce/benchmark_problems/catalog_utils.py` — `get_catalog()` returns model catalog. Models like `bn/BN_28` have domain size 10 (24 vars, width 5). **Can use for R020 if catalog is available, but hand-built ternary factors are more reliable for tests.**
- `nce/problems/test_problems.py` — Legacy test problem definitions with known Z values. References absolute paths. **Reference for known partition function values, but don't depend on file paths.**

## Constraints

- **Config completeness required** — `Trainer.__init__` uses `self.config['lower_dim']` (bare access), `self.config['debug']`, `self.config['optimizer']`, `self.config['inverse_time_decay_constant']`, and ~10 more fields without `.get()` fallbacks. A "minimal" config for NN training needs at minimum 25+ fields.
- **Log-space throughout** — All factor tensors are log10 values. `FastFactor.__mul__` performs addition (log-space multiplication). `eliminate()` does log-sum-exp. Test expectations must account for this.
- **`prepare_config` does NOT add defaults** — It validates, translates aliases, strips dead fields, but does not fill in missing optional fields. Tests must provide complete configs.
- **Elimination order required for factors** — `FastGM(factors=..., elim_order=None)` calls `wtminfill_order()` which has a bug with `None` variables_not_eliminated (line 72). Must provide explicit `elim_order` for hand-built problems.
- **No pytest-timeout** — The `--timeout` flag is not available (no `pytest-timeout` package installed). Tests must be self-limiting by design (small problems, few epochs).
- **Single venv** — All dependencies in `venv/`. Activate with `source venv/bin/activate`.

## Common Pitfalls

- **Missing config fields crash Trainer** — Using a minimal 7-field config for NN training will crash with `KeyError: 'lower_dim'`. Always use a complete config with all fields Trainer accesses. Build a fixture that mirrors `_build_nbe_configs()` but with CPU device and small epoch count.
- **Catalog model availability** — Tests that depend on `get_catalog()` may fail if `.model_cache/` is empty or network is unavailable. Use `pytest.skip()` for catalog-dependent tests (as M001 tests already do). But prefer hand-built factors for core correctness tests.
- **Log-space confusion** — Factor values are `log10(probability)`. A uniform binary factor is `log10(0.5) ≈ -0.301`, not `0.5`. Test a known partition function by computing Z manually in linear space, then `log10(Z)` for comparison.
- **All-neg-inf targets cause inf/nan** — Every major loss function produces inf or NaN when targets are all `-inf`. This is expected behavior (deterministic zero-probability messages). Tests should verify no *unhandled exception*, not that the loss value is finite. Document which loss functions handle this edge case gracefully vs. which produce inf/nan.
- **NN training is nondeterministic without seed control** — Even with `seed=42`, GPU training can have nondeterminism from cuDNN. Run convergence tests on CPU for reproducibility, or use tolerance-based assertions.
- **`wtminfill_order` bug with None** — `wtminfill_order(factors, variables_not_eliminated=None)` crashes at line 72. When using `FastGM(factors=...)`, always provide an explicit `elim_order` or the auto-computation will hit this path if factors have variables not in the computed elimination set.

## Open Risks

- **Test runtime on CPU** — NN training tests (R019, R021) will run on CPU to avoid GPU dependency in the default test path. Training 50 epochs on a small problem should take < 5 seconds, but this needs validation with the actual data pipeline overhead.
- **Edge case behavior may change** — Loss function behavior with all-neg-inf targets (R023) is currently inf/nan. If this is later fixed, tests need to be updated. Tests should assert "no crash" rather than specific nan/inf values.
- **Multi-domain test problem selection** — Hand-building a 3-state problem requires ternary factor tensors. This is straightforward but the `get_message_scope()` and sample generation code paths for domain>2 may have subtleties (one-hot encoding, domain size handling in `data_preprocessor.py`).

## Skills Discovered

| Technology | Skill | Status |
|------------|-------|--------|
| pytest | `github/awesome-copilot@pytest-coverage` | available (7K installs) — not needed, standard pytest knowledge sufficient |
| pytest | `bobmatnyc/claude-mpm-skills@pytest` | available (462 installs) — not needed |
| PyTorch | none directly relevant | none found for testing patterns |

## Requirement Analysis

### Table Stakes (must have, no debate)

- **R018** (Exact inference correctness) — Fundamental. If exact inference is wrong, everything downstream is broken. Hand-built problems with known Z values.
- **R019** (Single bucket training) — Core functional test. NN training is the product's purpose.
- **R021** (Convergence test) — Essential sanity check. Training that doesn't reduce loss is broken.

### Expected / Standard

- **R020** (Domain ≥3 handling) — Standard coverage. Most test problems are binary; multi-domain exercises different code paths in data preprocessing and one-hot encoding.
- **R022** (No-infinity output) — Standard robustness. Infinity propagation is a known failure mode in this domain.

### Good to Have / Advisory

- **R023** (All-neg-inf / all-zero target handling) — Edge case that occurs in real problems. Current behavior is inf/nan from all loss functions — tests should document this behavior, not necessarily fix it. This is more of a "known behavior" test than a "must pass" test.
- **R024** (Extensible failure-mode pattern) — Convention/structure requirement. Satisfied by clear test file organization, docstrings, and a PATTERN.md or section in the test file explaining how to add new regression tests.

### Candidate Requirements (from research, not in current requirements)

- **C001: Config completeness for NN training** — `prepare_config()` doesn't add defaults for missing optional fields, but `Trainer.__init__` uses bare dict access. This means any config used for NN training must include ~25+ fields. Consider whether `prepare_config` should add defaults for all fields, or whether this is working as intended and tests just need complete configs. **Advisory only — don't expand scope.**
- **C002: `wtminfill_order` None handling** — Line 72 crashes when `variables_not_eliminated=None`. This is a latent bug exposed only when constructing FastGM from factors without providing an elimination order. **Advisory — could be a separate bugfix, not a test requirement.**

## Sources

- Codebase exploration of `nce/` directory structure and key files
- Direct execution of hand-built factor problems and loss function edge cases
- Existing test suite analysis (110 tests, ~41s runtime)
- pyGMs catalog inspection for multi-domain test problem candidates
