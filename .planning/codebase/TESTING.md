# Testing Patterns

**Analysis Date:** 2026-02-21

## Test Framework

**Runner:**
- No formal framework configured (no pytest.ini, conftest.py, tox.ini)
- Tests run as standalone Python scripts in Jupyter notebooks
- Manual test execution via Python interpreter
- No automated test discovery

**Assertion Library:**
- Python's built-in `assert` statement
- Manual tensor comparison: `torch.allclose()`, direct equality checks
- Print statements for verification

**Run Commands:**
```bash
# No standardized test runner
# Tests executed via:
python /path/to/test_*.py
# Or in Jupyter notebooks via cell execution with %% markers
```

## Test File Organization

**Location:**
- `/home/cohenn1/NCE/notebooks/` - Primary test location
- `/home/cohenn1/NCE/nce/problems/test_problems.py` - Problem definitions (fixtures, not unit tests)
- Ad-hoc scripts in various notebook subdirectories

**Naming:**
- `test_*.py` or `*_test.py` pattern
- Examples: `test_new_loss_NN.py`, `decision_tree_test.py`, `test_weighted_ls_simple.py`

**Structure:**
```
notebooks/
├── 2025-07/
│   └── test_new_loss_NN.py
├── 2025-08/
│   └── decision_tree_test.py
├── 09-2025/
│   └── decision_tree_test.py
├── 10-2025/
│   ├── claude/elp_least_squares_debug/
│   │   ├── test_elp_8_5_benchmarks_ibound10.py
│   │   ├── test_weighted_ls_single_problem.py
│   │   └── test_weighted_ls_simple.py
│   └── single_bucket_elp_sq_tests.py
└── Older/
    ├── test_distance_metrics.py
    ├── test_loss_fns.py
    └── test.py
```

## Test Structure

**Suite Organization:**
Tests in notebooks use cell-based structure with `#%%` markers:

```python
#%% imports
if True:
    import time, copy, importlib
    from nce.utils import lse, get_message_gradient
    from nce.inference import FastGM, FastFactor
    from nce.problems import test_problems, TestProblem
    device = 'cuda'

#%% config
if True: # NN config
    traced_losses = ['weighted_logspace_mse', 'z_err', 'logspace_mse']
    gm_config = {
        'device': device,
        'iB': 3,
        'sampling_scheme': 'all',
        ...
    }

#%% test setup
if True:
    problem = test_problems[problem_name]
    gm = FastGM(uai_file=problem['uai_file'], device=device)
    bucket = gm.buckets[0]
    net = Net(bucket, hidden_sizes=nn_config['hidden_sizes'])

#%% test execution
if True:
    trainer = Trainer(net, bucket, loss_fn='mse')
    trainer.train_model(X, Y, batch_size=32, num_epochs=100)

#%% assertions
if True:
    assert trainer.losses[-1] < trainer.losses[0]
    assert not torch.isnan(net.network[0].weight.grad).any()
```

**Patterns:**
- `if True:` blocks for code organization and execution control
- Config dicts constructed before test execution
- Real objects used (no mocking)
- Assertions at end of cells
- Print statements for visual inspection

## Mocking

**Framework:** None detected

**Patterns:**
- Tests use actual objects: real `FastGM`, `FastBucket`, `Net` instances
- Memorizer class used as deterministic mock for networks:
  ```python
  class Memorizer(Net):
      """Lookup table that stores exact input-output pairs"""
      def __init__(self, bucket, all_x, all_y):
          super().__init__(bucket)
          self.memory = {}
          for i in range(self.nsamples):
              input_vector = tuple(all_x[i].tolist())
              self.memory[input_vector] = all_y[i].item()

      def forward(self, x):
          outputs = []
          for input_vector in x:
              input_tuple = tuple(input_vector.tolist())
              if input_tuple in self.memory:
                  outputs.append(self.memory[input_tuple])
              else:
                  outputs.append(self.linear(input_vector).item())
          return torch.tensor(outputs, device=self.device).view(-1, 1)
  ```

**What to Mock:**
- Use `Memorizer` for deterministic/ground-truth testing when exact values needed
- Use `LinearMSEOptimalSolver` for linear regression baseline comparison
- Use simplified tensor fixtures instead of real UAI files for basic validation

**What NOT to Mock:**
- Core factor operations (multiplication, elimination)
- Message computation pipeline
- Device-specific tensor behavior
- Graphical model loading from UAI files (use real test files instead)

## Fixtures and Factories

**Test Data:**
Problem configurations defined as dictionaries in `nce/problems/test_problems.py`:

```python
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
    "interesting_buckets": [...],
    "10-7-benchmark": True
}
```

**Factory Class:**
```python
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

**Location:**
- `/home/cohenn1/NCE/nce/problems/test_problems.py` - Problem fixtures
- UAI test files: `nce/problems/width_under_20_problems/`, `nce/problems/width20-30/`, `nce/problems/width_over_30/`
- Pickled benchmarks: `nce/problems/benchmarks_12_4_2025.pkl`, `nce/problems/nbe_benchmarks.pkl`

## Coverage

**Requirements:** None enforced

**View Coverage:**
- No coverage tool configured
- Manual inspection via print statements and assertions

## Test Types

**Unit Tests:**
- No formal unit test suite
- Ad-hoc component testing in notebooks
- Example pattern - testing factor operations:
  ```python
  f1 = FastFactor(torch.tensor([1.0, 2.0]), ['A'])
  f2 = FastFactor(torch.tensor([3.0, 4.0]), ['A'])
  result = f1 * f2
  assert result.tensor is not None
  ```

**Integration Tests:**
- Primary testing mode: full pipeline with real problems
- Load UAI problem → initialize bucket → train network → verify accuracy
- Example from `test_new_loss_NN.py`:
  ```python
  # 1. Load problem
  problem = test_problems['grid10x10.f10']
  gm = FastGM(uai_file=problem['uai_file'], device='cuda')

  # 2. Create and train network
  bucket = gm.buckets[0]
  net = Net(bucket, hidden_sizes=[64, 32])
  trainer = Trainer(net, bucket, loss_fn='mse')
  trainer.train_model(X, Y, batch_size=32, num_epochs=100)

  # 3. Verify convergence
  assert trainer.losses[-1] < trainer.losses[0]
  ```

**E2E Tests:**
- Not formally structured
- Inference end-to-end: load problem → compute messages → verify against benchmarks
- Validation criteria: loss convergence, message accuracy, Z partition function error

## Common Patterns

**Numerical Validation:**
```python
# Relative error bounds
z_estimate = net_result.item()
z_true = problem['Z']
relative_error = abs(z_estimate - z_true) / abs(z_true)
assert relative_error < 0.1, f"Error {relative_error} exceeds threshold"

# NaN detection
if torch.isnan(y_vals).any():
    print(f"[DataPreprocessor] ERROR: NaN in y_vals!")
assert not torch.isnan(outputs).any(), "Output contains NaN"

# Tensor gradient checks
loss.backward()
assert net.network[0].weight.grad is not None
assert torch.isfinite(net.network[0].weight.grad).all()
```

**Convergence Verification:**
```python
# Training improves loss
initial_loss = float('inf')
for epoch in range(num_epochs):
    loss = trainer.train_epoch()
    if epoch == 0:
        initial_loss = loss

final_loss = trainer.losses[-1]
assert final_loss < initial_loss, f"No convergence: {initial_loss} -> {final_loss}"
assert len(trainer.losses) == num_epochs
```

**Device Consistency:**
```python
# Verify all factors on correct device
for factor in bucket.factors:
    assert bucket.device in str(factor.device), \
        f"Factor device {factor.device} mismatch"

# Create tensors on correct device
tensor = torch.tensor([1.0, 2.0], device=self.device)
```

**Shape Validation:**
```python
# Batch dimensions
assert bw_message_values.dim() == 1, "Expected 1D tensor"
assert outputs.shape == targets.shape, "Shape mismatch"

# Reshaping documentation
# (num_samples, input_dim) -> (batch_size,)
expanded = outputs.unsqueeze(0)  # (1, len(outputs))
```

---

*Testing analysis: 2026-02-21*
