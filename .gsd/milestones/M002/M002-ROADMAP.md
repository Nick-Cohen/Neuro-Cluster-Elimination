# M002: Test Suite

**Vision:** `pytest tests/` validates core inference correctness, NN training functionality, convergence, robustness, and edge-case handling — catching regressions before they reach experiment runs.

## Success Criteria

- `pytest tests/` passes with all tests green (including the 110 existing M001 tests)
- At least one test per requirement R018–R024 exercises real code paths (not mocks)
- Hand-built test problems produce known-correct partition function values within tolerance
- NN training on a small CPU problem shows measurable loss decrease over 50 epochs
- Loss functions handle edge-case inputs (inf, all-neg-inf targets) without unhandled exceptions
- Adding a new failure-mode regression test requires only creating a new test function following the established pattern

## Key Risks / Unknowns

- **CPU convergence reliability** — NN training on tiny hand-built problems might not converge reliably in 50 epochs, making R021 flaky. Mitigated by choosing a learnable problem (binary chain with non-uniform factors) and using tolerance-based assertions.
- **Config completeness for Trainer** — `Trainer.__init__` uses bare dict access for ~25+ fields. A training config fixture must include all of them or tests crash with KeyError, not a training failure. Mitigated by basing the fixture on `reference_flat_config`.

## Proof Strategy

- **CPU convergence** → retire in S01 by building a hand-built binary chain problem and training one bucket for 50 epochs on CPU, asserting final loss < initial loss with margin.
- **Config completeness** → retire in S01 by building the `nn_training_config` fixture from the existing `reference_flat_config` pattern (known-complete, 42 fields) with CPU device and small epoch count.

## Verification Classes

- Contract verification: `pytest tests/` — all tests pass, each R018–R024 requirement has ≥1 test
- Integration verification: tests exercise real `FastGM`, `FastBucket`, `Trainer`, and loss function code paths with real tensor operations
- Operational verification: none
- UAT / human verification: none

## Milestone Definition of Done

This milestone is complete only when all are true:

- All test functions pass via `pytest tests/`
- Each of R018–R024 has at least one passing test that exercises real inference/training code
- Test runtime is under 120 seconds total (including existing M001 tests)
- A `PATTERN.md` or equivalent docstring documents how to add new failure-mode regression tests
- No test depends on network access, GPU availability (or is properly skip-marked), or external file downloads

## Requirement Coverage

- Covers: R018, R019, R020, R021, R022, R023, R024
- Partially covers: none
- Leaves for later: none
- Orphan risks: none

## Slices

- [ ] **S01: Inference & Training Test Suite** `risk:medium` `depends:[]`
  > After this: `pytest tests/` runs all new tests — exact inference correctness on hand-built problems, single-bucket NN training, domain≥3 handling, convergence over 50 epochs, infinity/edge-case robustness — alongside the 110 existing config tests, all green.

<!--
  Single slice rationale:
  All 7 requirements share the same fixtures (hand-built factor problems + complete NN training config).
  No requirement produces output that another consumes. The demoable unit is atomic: `pytest tests/` green.
  Splitting into multiple slices would create artificial boundaries with no independent demo value.
  3-4 tasks within S01 will organize the work: fixtures first, then correctness/functional tests,
  then convergence/robustness tests, then the extensible pattern documentation.
-->

## Boundary Map

### S01 (single slice)

Produces:
- `tests/conftest.py` extended with `nn_training_config` fixture (complete 42-field config, CPU, small epochs) and hand-built factor problem fixtures (binary chain, ternary chain with known Z values)
- `tests/test_inference.py` — exact inference correctness tests (R018), domain≥3 tests (R020)
- `tests/test_nn_training.py` — single-bucket training (R019), convergence (R021)
- `tests/test_robustness.py` — infinity output checks (R022), all-neg-inf/all-zero target handling (R023), extensible failure-mode pattern (R024)
- `tests/PATTERN.md` or equivalent docstring documenting how to add new regression tests

Consumes:
- Existing `tests/conftest.py` fixtures from M001
- `nce/inference/factor.py` (`FastFactor`), `nce/inference/graphical_model.py` (`FastGM`)
- `nce/neural_networks/losses.py` (loss functions), `nce/neural_networks/train.py` (`Trainer`)
- `nce/config_schema.py` (`prepare_config`)
