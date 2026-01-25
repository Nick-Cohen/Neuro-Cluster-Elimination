# Testing Patterns

**Analysis Date:** 2026-01-25

## Test Framework

**Current State:** Manual, Notebook-based Testing (No Formal Test Framework)

**Runner:**
- Not detected (no pytest.ini, tox.ini, or conftest.py)
- Tests run via Jupyter notebooks and ad-hoc Python scripts
- No test discovery or automation in place

**Assertion Library:**
- Standard Python assertions: `assert condition, "message"`
- Manual equality checks: `torch.allclose()` for tensor comparison
- No unittest or pytest framework integrated

**Run Commands:**
```bash
# No standardized test execution command
# Tests run via:
python -m jupyter notebook                    # Interactive notebook testing
python nce/problems/test_problems.py          # Direct script execution
```

## Test File Organization

**Location:**
- Tests are scattered: `notebooks/` directories contain `.py` test scripts
- Specialized test file: `/home/cohenn1/NCE/nce/problems/test_problems.py` (test problem definitions, not unit tests)
- No co-located test structure (`src/module.py` with `src/module_test.py`)

**Naming:**
- Notebooks: `decision_tree_test.py`, `test_new_loss_NN.py`, `test_weighted_ls_single_problem.py`
- Pattern: `test_*.py` or `*_test.py` prefixes
- Paths: `notebooks/August-2025/decision_tree_test.py`, `notebooks/10-2025/claude/elp_least_squares_debug/`

**Directory Structure:**
```
notebooks/
├── July-2025/
│   ├── test_new_loss_NN.py
│   └── ...
├── August-2025/
│   └── decision_tree_test.py
├── 09-2025/
│   └── decision_tree_test2.py
├── 10-2025/
│   ├── claude/
│   │   └── elp_least_squares_debug/
│   │       ├── test_weighted_ls_single_problem.py
│   │       └── test_*.py (multiple)
│   └── single_bucket_elp_sq_tests.py
└── Older/
    ├── test_distance_metrics.py
    ├── test_loss_fns.py
    └── ...
```

## Test Structure

**Pattern from Real Code:**

Tests in `notebooks/` are typically standalone scripts testing specific components:

```python
# From nce/problems/test_problems.py (test problem definitions)
from nce.inference.graphical_model import FastGM

class TestProblem:
    def __init__(self, config, device='cuda') -> None:
        self.name = config['name']
        self.width = config['width']
        self.uai_file = config['uai_file']
        self.Z = config['Z']
        self.interesting_buckets = config['interesting_buckets']
        self.device = device
        self.gm = None

    def load(self, device=None, doping=-5):
        if device == None:
            device = self.device
        self.gm = FastGM(uai_file=self.uai_file, device=device)
        self.gm.dope_factors(doping)
```

**No formal test suite structure detected:**
- No setUp/tearDown methods
- No test fixtures (pytest style)
- No test inheritance patterns
- Tests are procedural and manual

## Mocking

**Framework:** Minimal/None

**Patterns:**
- No mocking framework used (no unittest.mock, pytest-mock, etc.)
- Tests use real objects:
  ```python
  # Actual test loads real graphical models from files
  gm = FastGM(uai_file=self.uai_file, device=device)
  ```

**What to Mock:**
- Not applicable (no mocking framework integrated)

**What NOT to Mock:**
- Tests prefer real implementations for graphical models and inference

## Fixtures and Factories

**Test Data:**

Test data is organized by problem category in `nce/problems/`:

```python
# From nce/problems/test_problems.py - problem configuration dictionaries
grid10x10_f10 = {
    "name": "grid10x10.f10",
    "width": 12,
    "nvars": 100,
    "uai_file": "/home/cohenn1/NCE/nce/problems/width_under_20_problems/grid10x10.f10.uai",
    "Z": 303.0858154296875,
    "interesting_buckets": [47, 59, 54],
    "8-5-benchmark": True
}

pedigree1 = {
    "name": "pedigree1",
    "width": 16,
    "nvars": 334,
    "uai_file": "/home/cohenn1/NCE/nce/problems/width_under_20_problems/pedigree1.uai",
    "Z": -14.107169,
    "interesting_buckets": []
}

rbm_20 = {
    "name": "rbm_20",
    "width": 20,
    "nvars": 40,
    "uai_file": "/home/cohenn1/NCE/nce/problems/width20-30/rbm_20.uai",
    "Z": 58.5306282043457,
    "interesting_buckets": [28, 30, 35, 27, 33, 29, 23, 36, 31, 25, 24, 39, 37, 21, 26, 38, 34, 22, 32, 2],
    "10-7-benchmark": True
}
```

**Location:**
- `nce/problems/test_problems.py` contains TestProblem class and problem dictionaries
- Actual test data files: UAI format files in subdirectories:
  - `nce/problems/width_under_20_problems/` (12-16 width problems)
  - `nce/problems/width20-30/` (20-26 width problems)
  - `nce/problems/width_over_30/` (32+ width problems)
  - `nce/problems/teeny_weenee/` (small test cases)
- Pickled benchmark results: `nce/problems/benchmarks_12_4_2025.pkl`, `nce/problems/nbe_benchmarks.pkl`

## Coverage

**Requirements:** Not enforced (no configuration detected)

**View Coverage:**
```bash
# No coverage tool configured or documented
```

## Test Types

**Unit Tests:**
- None formalized
- Ad-hoc validation: Component functions tested in isolation via notebooks
- Example pattern from factor validation:
  ```python
  def is_equal(self, other, rtol=1e-3, atol=1e-5):
      """Check if this FastFactor is approximately equal to another."""
      if set(self.labels) != set(other.labels):
          return False
      perm = [other.labels.index(label) for label in self.labels]
      other_tensor_permuted = other.tensor.permute(*perm)
      return torch.allclose(self.flat, other_flat, rtol=rtol, atol=atol)
  ```

**Integration Tests:**
- Primary testing mode: Full graphical model inference with real UAI files
- Tests verify end-to-end learning with specific loss functions
- Example from notebook structure:
  - Load problem (UAI file)
  - Initialize bucket/message training
  - Train neural network approximation
  - Verify convergence or accuracy metrics
  - Compare with exact or reference values

**E2E Tests:**
- Not formalized, but integration tests serve this purpose
- Real data used: Actual benchmark problems (grid, pedigree, RBM models)
- Success criteria: Loss convergence, message accuracy against exact values

## Manual Test Patterns in Codebase

**Inline Assertions in Production Code:**

While no formal test suite exists, production code includes self-checking:

```python
# From nce/inference/factor.py
def is_equal(self, other, rtol=1e-3, atol=1e-5):
    """Check if this FastFactor is approximately equal to another."""
    ...
    return torch.allclose(self_flat, other_flat, rtol=rtol, atol=atol)

# From nce/inference/bucket.py
assert self.device in str(factor.device), f"Factor device mismatch"
assert not (message.tensor is None and len(message.labels) > 0)
```

**Validation Methods in Classes:**

Classes include validation/testing methods:

```python
# From nce/neural_networks/linear_mse_solver.py
def _diagnose_matrix(self, XtX: torch.Tensor) -> Dict[str, Any]:
    """Diagnose the condition of the X^T X matrix."""
    # Returns detailed diagnostics: rank, condition number, eigenvalues, etc.
```

**Problem-Specific Test Organization:**

Benchmark tests organized by categories:

```python
# Benchmarks marked by characteristics
"8-5-benchmark": True      # Specific benchmark suite
"10-7-benchmark": True     # Another suite
"interesting_buckets": []  # Buckets to focus on for detailed testing
```

## Debugging and Validation Patterns

**Tensor Validation:**
- Shape assertions: `assert bw_message_values.dim() == 1`
- Device consistency checks at initialization
- NaN/Inf detection: `if torch.isnan(params).any() or torch.isinf(params).any()`

**Numerical Stability:**
- Documented in loss functions with explicit comments about normalization
- Example from `unnormalized_kl()`:
  ```python
  # CRITICAL: For batched training, max_val MUST be computed from full dataset
  # and passed in. Per-batch max_val causes gradient inconsistency.
  if max_val is None:
      max_val = torch.max(torch.max(targets), torch.max(outputs.detach()))
  ```

**Verbosity/Debug Modes:**
- Conditional printing via config: `if self.config.get('debug', True):`
- Stats collection: `self.tracked = {'parameters': [], 'gradients': []}`
- Message statistics: `get_message_stats()` utility function

---

*Testing analysis: 2026-01-25*
