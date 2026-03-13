---
estimated_steps: 4
estimated_files: 2
---

# T04: Robustness edge-case tests and extensibility pattern

**Slice:** S01 — Inference & Training Test Suite
**Milestone:** M002

## Description

Create `tests/test_robustness.py` with edge-case tests for loss function robustness (R022, R023) and `tests/PATTERN.md` documenting how to add new failure-mode regression tests (R024). Loss function tests call the functions directly — no Trainer needed.

## Steps

1. Create `tests/test_robustness.py`. Import loss functions from `nce.neural_networks.losses`, torch, pytest, math.
2. Write `test_loss_fn_inf_input_no_crash` parametrized over key loss functions (`logspace_mse_fdb`, `linspace_mse_fdb`, `from_logspace_mse`): create `outputs = torch.tensor([float('inf'), 0.0, -1.0])` and `targets = torch.tensor([1.0, 0.5, -0.5])`. Call the loss function. Assert no unhandled exception. Document whether result is finite, inf, or nan (but don't assert finiteness — the requirement is no crash, not finite output). Covers R022.
3. Write `test_loss_fn_neg_inf_targets_no_crash` parametrized over the same functions: create `outputs = torch.tensor([1.0, 0.5, -0.5])` and `targets = torch.tensor([float('-inf'), float('-inf'), float('-inf')])`. Call the loss function. Assert no unhandled exception. Covers R023. Also write `test_loss_fn_zero_targets_no_crash` with `targets = torch.tensor([0.0, 0.0, 0.0])` to verify zero-probability edge case.
4. Create `tests/PATTERN.md` documenting: (a) file-per-concern structure (`test_inference.py`, `test_nn_training.py`, `test_robustness.py`), (b) shared fixtures in `conftest.py`, (c) how to add a new failure-mode regression test (step-by-step: identify the failure, write a test that reproduces it, add to the appropriate file, use `pytest.mark.parametrize` for variants), (d) example template for a new robustness test. Covers R024.

## Must-Haves

- [ ] Inf-input tests pass without unhandled exception for ≥3 loss functions (R022)
- [ ] All-neg-inf target tests pass without unhandled exception for ≥3 loss functions (R023)
- [ ] All-zero target tests pass without crash
- [ ] PATTERN.md exists with clear step-by-step instructions for adding regression tests (R024)
- [ ] Tests use real loss function calls with real tensors (not mocks)

## Verification

- `source venv/bin/activate && python -m pytest tests/test_robustness.py -v` — all tests pass
- `test -f tests/PATTERN.md && echo "PATTERN.md exists"` — file exists
- `source venv/bin/activate && python -m pytest tests/ -v --tb=short` — full suite passes
- `source venv/bin/activate && python -m pytest tests/ --co -q | tail -1` — total count ≥ 125

## Observability Impact

- Signals added/changed: None (pure test assertions for edge cases)
- How a future agent inspects this: Read PATTERN.md for instructions on adding new tests; run `pytest tests/test_robustness.py -v` for edge-case status
- Failure state exposed: Parametrized test names include the loss function name, making it clear which function failed on which edge case

## Inputs

- `nce/neural_networks/losses.py` — loss function signatures: `fn(outputs, targets, bw_hat=None)`
- S01-RESEARCH.md — confirmed edge-case behavior: inf inputs → inf/nan (no crash), neg-inf targets → inf/nan (no crash), zero targets → handled
- `tests/test_inference.py`, `tests/test_nn_training.py` — existing test file structure to reference in PATTERN.md

## Expected Output

- `tests/test_robustness.py` — ~9 parametrized tests covering R022 (3+) and R023 (3+)
- `tests/PATTERN.md` — step-by-step guide for adding failure-mode regression tests (R024)
