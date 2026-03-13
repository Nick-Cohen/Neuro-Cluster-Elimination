# Test Suite Pattern Guide (R024)

How to navigate, extend, and add failure-mode regression tests to the NCE test suite.

## File Structure

Each file covers one testing concern:

| File | Concern | Requirements |
|---|---|---|
| `test_regression.py` | Config and inference regression against saved baselines | Pre-existing |
| `test_inference.py` | Exact inference correctness on hand-built problems | R018, R020 |
| `test_nn_training.py` | Single-bucket NN training completion and convergence | R019, R021 |
| `test_robustness.py` | Loss function edge-case inputs (inf, nan, zeros) | R022, R023 |
| `conftest.py` | Shared fixtures: configs, problem graphs, factors | — |

## Shared Fixtures (`conftest.py`)

- **`nn_training_config`** — Validated 40-field config dict for CPU NN training (via `prepare_config`). Use as a base config; override fields with a copy for specific tests.
- **`binary_chain_factors`** — Two domain-2 variables, uniform pairwise factor. Z=2.0.
- **`ternary_chain_factors`** — Two domain-3 variables, uniform pairwise factor. Z=4.5.
- **`star_graph_factors`** — Hub + 3 leaves with pairwise + 3-way factors. Message scope exceeds ecl=4, triggering NN path.

All problem fixtures return dicts with keys: `'factors'`, `'elim_order'`, `'expected_log10_z'`.

## Adding a New Failure-Mode Regression Test

### Step 1: Identify the failure

Document the exact failure: what input triggered it, which function failed, what the error message was. Example:

> `from_logspace_mse` raises `RuntimeError` when all targets are `-inf` because `torch.max` of an all-`-inf` tensor returns `-inf`, then `exp(-inf - (-inf))` produces `nan`.

### Step 2: Determine the right file

- Loss function behavior → `test_robustness.py`
- Inference correctness (wrong Z value) → `test_inference.py`
- Training process failures (crashes, bad logs) → `test_nn_training.py`
- New concern → create a new `test_<concern>.py` file

### Step 3: Write the test

Use `pytest.mark.parametrize` when the same edge case applies to multiple functions. Follow this template:

```python
import pytest
import torch

from nce.neural_networks.losses import logspace_mse_fdb, linspace_mse_fdb

LOSS_FNS = [
    pytest.param(logspace_mse_fdb, id="logspace_mse_fdb"),
    pytest.param(linspace_mse_fdb, id="linspace_mse_fdb"),
]

class TestMyEdgeCase:
    """Brief description of what edge case this covers."""

    @pytest.mark.parametrize("loss_fn", LOSS_FNS)
    def test_my_edge_case_no_crash(self, loss_fn):
        """Describe the specific scenario and expected behavior."""
        # 1. Create problematic inputs
        outputs = torch.tensor([...])
        targets = torch.tensor([...])

        # 2. Call the function — the test passes if no exception is raised
        result = loss_fn(outputs, targets)

        # 3. Optionally assert on the result (e.g., finiteness)
        # Only assert finiteness if the function *should* handle this gracefully.
        # For "no crash" tests, reaching this line is sufficient.
```

### Step 4: Run and verify

```bash
# Run just your new tests
source venv/bin/activate && python -m pytest tests/test_robustness.py -v -s

# Run the full suite to confirm no regressions
source venv/bin/activate && python -m pytest tests/ -v --tb=short
```

### Step 5: Document the edge case

Add a class-level or module-level docstring mapping the test to its requirement ID (e.g., R022, R023). This makes it easy to trace from requirements to tests.

## Conventions

- **Real calls only.** Tests call actual functions with real tensors. No mocks for the code under test.
- **Parametrize over variants.** When the same edge case applies to multiple loss functions, use `@pytest.mark.parametrize` instead of copy-pasting tests.
- **Diagnostic output.** Use `print()` for diagnostic info visible with `pytest -s`. Don't rely on it for assertions.
- **Assertion messages.** Always include a message showing computed vs expected values. Example:
  ```python
  assert abs(actual - expected) < 1e-5, (
      f"log10(Z): got {actual:.8f}, expected {expected:.8f} "
      f"(diff={abs(actual - expected):.2e})"
  )
  ```
- **Fixtures over setup.** Put reusable test data in `conftest.py` fixtures, not in test-file-level globals.
