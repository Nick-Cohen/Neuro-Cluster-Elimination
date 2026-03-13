---
estimated_steps: 4
estimated_files: 1
---

# T02: Exact inference correctness and domain≥3 tests

**Slice:** S01 — Inference & Training Test Suite
**Milestone:** M002

## Description

Create `tests/test_inference.py` with tests that verify exact inference (no NN approximation) produces known-correct partition function values on hand-built problems. Covers R018 (exact inference correctness) with binary and star graph problems, and R020 (domain≥3 handling) with the ternary chain problem.

## Steps

1. Create `tests/test_inference.py`. Import `FastGM`, `prepare_config`, math, torch, pytest. Import fixtures from conftest.
2. Write `test_binary_chain_exact_z(binary_chain_factors, nn_training_config)`: build a config with `ecl=2**30` (force exact path), `dope_factors=False`. Create `FastGM(factors=fixture['factors'], elim_order=fixture['elim_order'], nn_config=config, device='cpu')`. Call `gm.eliminate_variables(all=True)`. Assert `abs(gm.log_partition_function - fixture['expected_log10_z']) < 1e-5` with a diagnostic message showing both values.
3. Write `test_ternary_chain_exact_z(ternary_chain_factors, nn_training_config)`: same pattern, verifying domain-3 variables work through the exact inference path. Assert log10(Z) matches within 1e-5. This covers R020.
4. Write `test_star_graph_exact_z(star_graph_factors, nn_training_config)`: same pattern with star graph. Verifies multi-variable bucket exact computation. This provides a second R018 data point and validates the star graph fixture that T03 will use for NN training.

## Must-Haves

- [ ] Binary chain test passes with log10(Z) within 1e-5 of log10(2.0) (R018)
- [ ] Ternary chain test passes with log10(Z) within 1e-5 of log10(4.5) (R020)
- [ ] Star graph test passes with exact Z matching analytic value (R018)
- [ ] All tests use real FastGM inference (not mocks)
- [ ] Assertion messages include computed vs expected values for debugging

## Verification

- `source venv/bin/activate && python -m pytest tests/test_inference.py -v` — all 3 tests pass
- `source venv/bin/activate && python -m pytest tests/ -v --tb=short` — all tests (existing + new) pass

## Observability Impact

- Signals added/changed: None (pure test assertions)
- How a future agent inspects this: Run `pytest tests/test_inference.py -v` to see per-test results; failure messages show computed vs expected log10(Z)
- Failure state exposed: Assertion errors include both values and problem description

## Inputs

- `tests/conftest.py` — T01 fixtures: `binary_chain_factors`, `ternary_chain_factors`, `star_graph_factors`, `nn_training_config`
- `nce/inference/graphical_model.py` — `FastGM(factors=..., elim_order=..., nn_config=..., device=...)` API
- `nce/config_schema.py` — `prepare_config()` for config preparation

## Expected Output

- `tests/test_inference.py` — 3 passing tests covering R018 (2 tests) and R020 (1 test)
