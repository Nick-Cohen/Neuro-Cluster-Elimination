---
id: M002
provides:
  - pytest-based test suite covering exact inference correctness, NN training, convergence, and loss function robustness
  - Hand-built factor problem fixtures with analytically known partition function values
  - Complete NN training config fixture (42-field, CPU, 50 epochs) derived from reference_flat_config
  - Extensible failure-mode regression pattern documented in tests/PATTERN.md
key_decisions:
  - "D020: Single-slice M002 — all 7 requirements share fixtures, no inter-test dependencies"
  - "D021: Hand-built factors over catalog models — instant, deterministic, analytically verifiable"
  - "D022: CPU-only default for NN training tests — reproducible, no GPU dependency"
  - "D023: Tolerance-based convergence assertion — final_avg < 0.9 * initial_avg"
  - "D024: Direct loss function calls for edge-case tests — isolates function under test"
  - "D025: No-crash assertion for edge cases, not finite-output — inf/nan are expected for extreme inputs"
patterns_established:
  - "Hand-built factor fixtures return dicts with 'factors', 'elim_order', 'expected_log10_z' keys"
  - "NN training tests use star graph fixture with ecl=4 to force NN path on the hub bucket"
  - "Robustness tests parametrize over STANDARD_LOSS_FNS list for uniform edge-case coverage"
  - "Each test file maps to a testing concern; docstrings reference requirement IDs for traceability"
observability_surfaces:
  - "pytest tests/ --tb=short -v — full test suite with per-test pass/fail and convergence diagnostics"
  - "Convergence test prints per-bucket loss trajectory (initial_avg, final_avg, ratio) with -s flag"
requirement_outcomes:
  - id: R018
    from_status: active
    to_status: validated
    proof: "test_binary_chain_exact_z and test_star_graph_exact_z pass — exact inference matches analytic Z within 1e-5"
  - id: R019
    from_status: active
    to_status: validated
    proof: "test_nn_training_completes passes — star graph with ecl=4 triggers NN path, per_bucket_training_log is non-empty with loss data"
  - id: R020
    from_status: active
    to_status: validated
    proof: "test_ternary_chain_exact_z passes — domain-3 variables produce correct Z=4.5 within 1e-5"
  - id: R021
    from_status: active
    to_status: validated
    proof: "test_convergence_loss_decreases passes — final_avg < 0.9 * initial_avg for at least one NN-trained bucket over 50 epochs"
  - id: R022
    from_status: active
    to_status: validated
    proof: "test_loss_fn_inf_input_no_crash passes for logspace_mse_fdb, linspace_mse_fdb, from_logspace_mse — no unhandled exception on inf inputs"
  - id: R023
    from_status: active
    to_status: validated
    proof: "test_loss_fn_neg_inf_targets_no_crash and test_loss_fn_zero_targets_no_crash pass for all 3 loss functions — no crash on all-neg-inf or all-zero targets"
  - id: R024
    from_status: active
    to_status: validated
    proof: "tests/PATTERN.md documents the step-by-step process for adding failure-mode regression tests with template code, file routing, and conventions"
duration: "1 session"
verification_result: passed
completed_at: 2026-03-12
---

# M002: Test Suite

**pytest-based test suite validating exact inference correctness, NN training functionality, convergence, and loss function robustness on hand-built problems with known partition function values.**

## What Happened

A single slice (S01) delivered the full test suite across four tasks. T01 built the foundation: three hand-built factor problem fixtures (binary chain Z=2.0, ternary chain Z=4.5, star graph with analytic Z) and a complete 42-field NN training config fixture derived from `reference_flat_config` with CPU device and 50 epochs. T02 added exact inference correctness tests verifying partition function values within 1e-5 tolerance on all three problems, including the domain-3 ternary chain for R020 coverage. T03 implemented NN training tests using the star graph (whose hub bucket exceeds ecl=4, forcing the NN path), verifying both functional completion (R019) and convergence (R021) with a tolerance-based assertion (final avg loss < 90% of initial avg). T04 added loss function robustness tests parametrized over three standard loss functions, covering inf inputs (R022) and all-neg-inf/all-zero targets (R023), plus the PATTERN.md guide documenting the extensible regression test pattern (R024).

The suite runs in ~30 seconds alongside the 110 existing M001 config tests. All 125 tests pass. No test requires network access, GPU availability, or external file downloads.

## Cross-Slice Verification

Single slice — no cross-slice integration needed.

- **`pytest tests/` all green:** 125 passed in 30s (110 M001 + 15 M002 tests)
- **R018–R024 coverage:** Each requirement has ≥1 test calling real code with real tensors, verified by grep for requirement IDs in test docstrings
- **Hand-built problems produce correct Z:** binary chain (Z=2.0, log10≈0.30103), ternary chain (Z=4.5, log10≈0.65321), star graph (analytic Z from brute-force enumeration) — all within 1e-5
- **NN convergence:** Star graph training shows loss decrease from ~5.46 to ~2.64 over 50 epochs (seed=42), well exceeding the 10% threshold
- **Edge-case robustness:** 9 parametrized tests (3 loss fns × 3 edge cases) all pass without unhandled exceptions
- **Runtime:** 30s total, well under the 120s ceiling
- **PATTERN.md:** Documents file routing, fixture usage, parametrize conventions, and a copy-paste template for new regression tests

## Requirement Changes

- R018: active → validated — test_binary_chain_exact_z and test_star_graph_exact_z pass with analytic Z within 1e-5
- R019: active → validated — test_nn_training_completes verifies NN path is triggered and produces loss data
- R020: active → validated — test_ternary_chain_exact_z verifies domain-3 inference correctness
- R021: active → validated — test_convergence_loss_decreases confirms >10% loss reduction over 50 epochs
- R022: active → validated — test_loss_fn_inf_input_no_crash passes for all 3 standard loss functions
- R023: active → validated — test_loss_fn_neg_inf_targets_no_crash and test_loss_fn_zero_targets_no_crash pass for all 3 loss functions
- R024: active → validated — tests/PATTERN.md provides step-by-step guide with template code

## Forward Intelligence

### What the next milestone should know
- The `nn_training_config` fixture in conftest.py is a complete 42-field config. If new config fields are added to the codebase, this fixture must be updated or tests will crash with KeyError.
- The star graph fixture is specifically designed so that bucket 0's message scope exceeds ecl=4, forcing the NN training path. Changing the graph topology or ecl value may break this invariant.
- All M002 tests run on CPU with seed=42. If future tests need GPU, use `@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")`.

### What's fragile
- The convergence test (R021) uses a 10% relative decrease threshold. If the star graph problem or loss function behavior changes, this could become flaky. The current margin is large (~52% decrease), but it's worth monitoring.
- The `nn_training_config` fixture hardcodes `sampling_scheme='all'` and `num_samples=256` for fast CPU training. These values are not representative of production configs.

### Authoritative diagnostics
- `pytest tests/ -v --tb=short` — full test output with per-test names showing requirement coverage
- `pytest tests/test_nn_training.py -v -s` — convergence test prints per-bucket loss trajectories

### What assumptions changed
- D025 clarified that inf/nan outputs from loss functions on extreme inputs are expected behavior, not bugs. The test contract is "no crash," not "finite output."

## Files Created/Modified

- `tests/test_inference.py` — exact inference correctness tests (R018, R020): binary chain, ternary chain, star graph
- `tests/test_nn_training.py` — NN training functional and convergence tests (R019, R021)
- `tests/test_robustness.py` — loss function edge-case robustness tests (R022, R023)
- `tests/PATTERN.md` — extensible failure-mode regression test guide (R024)
- `tests/conftest.py` — extended with nn_training_config, binary_chain_factors, ternary_chain_factors, star_graph_factors fixtures
