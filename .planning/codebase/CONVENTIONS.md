# Coding Conventions

**Analysis Date:** 2026-01-25

## Naming Patterns

**Files:**
- Snake case for module files: `data_loader.py`, `message_gradient.py`, `graphical_model.py`
- Snake case for class-specific files: `factor.py`, `bucket.py`
- Legacy/temporary/copy files use descriptive suffixes: `train_old.py`, `NN_Train_copy.py`, `linear_mse_solver.py`

**Classes:**
- PascalCase for all classes: `FastFactor`, `FastBucket`, `FastGM`, `Net`, `Memorizer`, `Trainer`, `LinearMSEOptimalSolver`
- Special purpose classes use descriptive names: `FactorNN`, `BitVectorLookup`, `SimpleConvexEarlyStopping`

**Functions:**
- Snake case for all functions: `compute_message_exact()`, `eliminate_variables()`, `get_message_stats()`, `shuffle_batches()`
- Private functions prefixed with underscore: `_make_dataloader()`, `_get_loss_fn()`, `_diagnose_matrix()`, `_solve_cholesky()`
- Helper methods in classes follow same convention: `_load_from_uai()`, `_create_buckets_from_factors()`

**Variables:**
- Snake case for instance and local variables: `hidden_sizes`, `sample_generator`, `device`, `nn_config`, `loss_fn`
- Class attributes use snake case: `self.bucket`, `self.gm`, `self.tensor`, `self.labels`
- Configuration dictionaries use lowercase keys: `'hidden_sizes'`, `'loss_fn'`, `'num_samples'`, `'lr'`, `'device'`
- Underscore-suffixed attributes for computed properties: `weights_`, `bias_`, `solution_method_`, `condition_number_`

**Constants/Config Keys:**
- Uppercase for boolean config flags: `'exact'`, `'debug'`, `'track_errors'`, `'use_memorizer'`, `'use_bw_approx'`, `'populate_bw_factors'`
- Lowercase hyphenated or underscore-separated for complex config keys: `'lower_dim'`, `'seed'`, `'inverse_time_decay_constant'`, `'8-5-benchmark'`
- Numeric parameters lowercase: `'iB'` (bucket i-bound), `'ecl'` (exact complexity limit), `'num_samples'`, `'num_epochs'`

## Code Style

**Formatting:**
- No enforced linter (no `.eslintrc`, `.pylintrc`, or `pyproject.toml` config in repo root)
- Default Python conventions: 4-space indentation, max line length ~100-120 characters (varies)
- Multi-line function definitions indent parameters naturally

**Import Organization:**
- Standard library first: `import torch`, `import math`, `import time`, `import sys`
- Third-party imports: `import torch.nn as nn`, `from torch.utils.data import DataLoader`, `import pyGMs as gm`
- Relative imports for internal modules: `from .losses import *`, `from .factor import FastFactor`, `from nce.inference import FastBucket`
- Wildcard imports used in some files: `from .losses import *` in `train.py`, `from nce.data import *`
- Type imports at function level or in comments: `from typing import List, Dict, Any, Tuple, IO`

**Import Patterns:**
- No path aliases detected (no jsconfig/tsconfig equivalents)
- Imports mixed throughout files (not always at top) when used conditionally:
  ```python
  def compute_message_nn(self, ...):
      from nce.neural_networks.net import Net, Memorizer
      from nce.utils.plots import plot_fastfactor_comparison
  ```

## Type Annotations

**Function Signatures:**
- Sparse type hints in some modules, dense in others
- Type hints used in function parameters and return types where present:
  ```python
  def load(self, num_samples: int = 0, all: bool = False, is_validation: bool = False) -> tuple:
  def _diagnose_matrix(self, XtX: torch.Tensor) -> Dict[str, Any]:
  def forward(self, x: torch.Tensor) -> torch.Tensor:
  ```
- Return type hints often use generic types: `-> tuple`, `-> torch.Tensor`, `-> Tuple[torch.Tensor, bool]`

**Variable Annotations:**
- Type hints in docstrings for parameters without inline annotations
- Tensor operations use PyTorch type system implicitly

## Error Handling

**Strategy:** Broad exception catching with optional logging/re-raising

**Patterns:**
- Generic `Exception` catching with conditional re-raise:
  ```python
  try:
      message = message.eliminate(self.elim_vars)
  except Exception as e:
      print(f"Warning: Elimination failed in bucket {self.label}...")
      raise e
  ```
- Silent exception suppression (try-except without handler):
  ```python
  try:
      if not other.labels:
          return FastFactor(self.tensor + other.tensor.item(), self.labels)
  except:
      print("got here")
      raise(ValueError("Other is not a FastFactor"))
  ```
- Silent exception handling in matrix diagnosis with fallback return:
  ```python
  except Exception as e:
      if self.verbose:
          print(f"Warning: Could not diagnose matrix condition: {e}")
      return {...error dict...}
  ```

## Logging

**Framework:** `print()` statements (no structured logging framework detected)

**Patterns:**
- Debug output: `print(f"Warning: {message}")`, `print("Computing backward message with bw ecl ", backward_ecl)`
- Status messages: `print(f"Bucket {self.label}: Using Memorizer (lookup table)")`
- Verbose conditionals: `if self.verbose:` or `if self.config.get('debug', True):`
- Deprecation notices in docstrings: `"""OLD VERSION - kept for reference."""`

## Comments

**When to Comment:**
- Complex mathematical operations: Comments explain loss functions, elimination orders, tensor operations
- Non-obvious control flow: Conditionals with business logic include inline explanation
- Configuration flags: Comments explain conditional behavior based on config settings
- Numerical stability: Comments explain max normalization, regularization choices

**JSDoc/TSDoc:**
- Comprehensive docstrings used in key methods:
  ```python
  def load(self, num_samples: int = 0, all: bool = False, is_validation: bool = False) -> tuple:
      """Load training data by sampling and computing message/backward values.

      Args:
          num_samples: Number of samples to generate (ignored if all=True)
          ...
      Returns:
          Tuple of (x, y, bw) where:
          - x: One-hot encoded assignments
          ...
      """
  ```
- Docstrings follow standard Python convention (triple quotes, parameter/return sections)
- Docstrings used in classes and public methods, sparse in private methods
- Parameter descriptions in comments above functions where docstrings absent

## Function Design

**Size:** Functions vary widely from 5-line getters to 300+ line training loops
- Small focused functions: 5-20 lines for utilities and accessors
- Medium functions: 30-100 lines for core algorithms
- Large functions: 200+ lines for orchestration methods (e.g., `Trainer.train()`)

**Parameters:**
- Typically 1-5 parameters for public methods
- Optional parameters use defaults: `def __init__(self, ..., hidden_sizes=None)`
- Config objects passed as `self.config` (dict) rather than individual parameters
- Variadic parameters rare; **kwargs not commonly used

**Return Values:**
- Single return values (scalar, tensor, object) most common
- Tuple returns for multi-value results: `Tuple[torch.Tensor, torch.Tensor]`, `(x, y, bw)`
- Optional returns: `None` returned on error paths or when not applicable

## Module Design

**Exports:**
- No explicit `__all__` definitions detected
- Classes and functions exported implicitly (all public)
- Modules used via direct import: `from nce.inference.graphical_model import FastGM`

**Barrel Files:**
- `__init__.py` files exist but mostly empty: `from nce.inference import *` style imports used
- Some initialization in `__init__.py` for package setup (`setup.py` minimal)

## Assertion Patterns

**Validation:**
- Assertions used for invariants in critical code:
  ```python
  assert self.device in str(factor.device), f"Factor device {factor.device}..."
  assert not (message.tensor is None and len(message.labels) > 0), f"{self.label}"
  assert not(factor.tensor is None and not factor.is_nn), f"{bucket.label}"
  ```
- Assertions check device consistency, tensor validity, factor structure
- Runtime errors for invalid user input: `raise ValueError(...)`, `raise Exception(...)`

## Tensor Operations

**Device Management:**
- Explicit device tracking: `device = self.device` or `device = self.config['device']`
- Device consistency checked in assertions
- Tensor creation specifies device: `torch.tensor(..., device=device)`
- CUDA support: `if device == 'cuda': torch.cuda.manual_seed(seed)`

**Tensor Shapes:**
- Shape comments in docstrings: `# (num_samples, input_dim)`, `# (batch_size,)`
- Reshaping and view operations documented with expected dimensions
- Broadcasting operations explicit with reshape/unsqueeze: `combined = torch.stack([...], dim=-1)`

---

*Convention analysis: 2026-01-25*
