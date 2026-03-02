# Coding Conventions

**Analysis Date:** 2026-02-21

## Naming Patterns

**Files:**
- Snake case for module files: `data_loader.py`, `message_gradient.py`, `graphical_model.py`, `factor.py`, `bucket.py`
- Legacy/temporary files use descriptive suffixes: `train_old.py`, `NN_Train_copy.py`, `linear_mse_solver.py`
- Test files: `test_*.py` or `*_test.py` in `/home/cohenn1/NCE/notebooks/` directories

**Classes:**
- PascalCase for all classes: `FastFactor`, `FastBucket`, `FastGM`, `Net`, `Memorizer`, `Trainer`, `DataPreprocessor`
- Special purpose classes: `FactorNN`, `BitVectorLookup`, `SimpleConvexEarlyStopping`, `DecisionTreeLossOptimizer`

**Functions:**
- Snake case: `compute_message_exact()`, `eliminate_variables()`, `get_message_stats()`, `shuffle_batches()`
- Private functions prefixed with underscore: `_make_dataloader()`, `_get_loss_fn()`, `_initialize_normalizing_constant()`
- Verb-first pattern common: `get_*`, `compute_*`, `load_*`, `create_*`, `populate_*`

**Variables:**
- Snake case for all variables: `hidden_sizes`, `sample_generator`, `device`, `nn_config`, `bucket_var`, `message_scope`
- Boolean flags with `is_` or `use_` prefix: `is_nn`, `is_primary`, `use_bw_approx`, `use_memorizer`, `use_linspace_bias`
- Config dictionary keys: lowercase with underscores: `'hidden_sizes'`, `'loss_fn'`, `'num_samples'`, `'device'`, `'iB'`, `'ecl'`
- Class attributes follow snake case: `self.bucket`, `self.gm`, `self.tensor`, `self.labels`, `self.device`

## Code Style

**Formatting:**
- No explicit linting tool configured (no .eslintrc, .prettierrc, or pylintrc in repo root)
- PEP 8 broadly followed: 4-space indentation
- Line length varies (some exceeding 100 chars)
- Import statements at file top, some conditional imports inside functions

**Linting:**
- Manual validation via assertions and type checking
- Type hints selectively used in function signatures
- Type imports: `from typing import List, Dict, Any, Tuple, IO`

## Import Organization

**Order:**
1. Standard library: `import sys`, `import os`, `import math`, `import time`, `import argparse`, `import xml.etree.ElementTree`
2. Third-party: `import torch`, `import torch.nn as nn`, `import torch.nn.functional as F`, `import numpy as np`, `import matplotlib.pyplot as plt`, `from pyGMs import *`, `from tqdm.notebook import tqdm`
3. Local imports: `from nce.inference import ...`, `from nce.neural_networks import ...`, `from nce.data import ...`
4. Relative imports: `from .factor import FastFactor`, `from .losses import *`

**Path Aliases:**
- No path aliases configured
- Absolute imports preferred: `from nce.inference.graphical_model import FastGM`
- Wildcard imports in some files: `from pyGMs.neuro import *`, `from .losses import *`
- Conditional imports used to avoid circular dependencies:
  ```python
  def compute_message_nn(self, ...):
      from nce.neural_networks.net import Net, Memorizer
      from nce.neural_networks.train import Trainer
  ```

## Error Handling

**Patterns:**
- Try-except blocks for critical operations with informative messages:
  ```python
  try:
      message = message.eliminate(self.elim_vars)
  except Exception as e:
      print(f"Warning: Elimination failed in bucket {self.label} with size {message.tensor.shape if message.tensor is not None else 'None'}: {e}")
      raise e
  ```
- Assertions for invariants and preconditions:
  ```python
  assert self.device in str(factor.device), f"Factor device {factor.device} does not match bucket device type {self.device}"
  assert message.tensor is not None
  assert not (message.tensor is None and len(message.labels) > 0), f"{self.label}"
  ```
- ValueError for invalid inputs:
  ```python
  raise ValueError(f"Cannot use compute_linear_mse_message: {e}")
  raise ValueError(f"NN exact message for bucket {self.label} contains NaN values")
  ```
- RuntimeError for state violations:
  ```python
  raise RuntimeError("Could not find Linear layer in the network")
  ```
- Print statements for warnings (common pattern):
  ```python
  print(f"Warning: Could not compute exact message for plotting: {e}")
  print(f"[DataPreprocessor] ERROR: NaN in y_vals input!")
  ```

## Logging

**Framework:** `print()` statements and console output

**Patterns:**
- Debug output via f-string print: `print(f"Bucket {self.label}: Using Memorizer (lookup table)")`
- Conditional logging: `if self.debug:` or `if self.config.get('debug', True):`
- Progress tracking: `from tqdm.notebook import tqdm` for notebook environments
- Status messages with context: `print("Computing backward message with bw ecl ", backward_ecl)`
- Statistics gathering through utility functions: `from nce.utils.stats import get_message_stats`, `get_gm_message_stats()`

## Comments

**When to Comment:**
- Critical correctness notes (marked with CRITICAL):
  ```python
  # CRITICAL: Create a NEW copy of config dict to avoid sharing with other GMs
  # Using dict() ensures each GM has its own independent config
  self.config = dict(nn_config) if nn_config else {}
  ```
- Numerical stability warnings:
  ```python
  # CRITICAL: For batched training, max_val MUST be computed from full dataset
  # and passed in. Per-batch max_val causes gradient inconsistency and training divergence
  ```
- Algorithm explanation:
  ```python
  # Multiply all factors
  # Eliminate variables
  # Handle edge cases
  ```
- Config interpretation:
  ```python
  # Handle 'bias_only' mode: learn only a single constant (bias term)
  self.bias_only = (hidden_sizes == 'bias_only')
  ```
- TODO/FIXME for known issues:
  ```python
  # TODO: Multiply factors in a more sensible order, e.g. subsumed multiplications first
  # TODO: will need to grab list of factors in the future
  ```

**JSDoc/TSDoc:**
- Docstrings use triple quotes with Args and Returns sections:
  ```python
  def load(self, num_samples: int = 0, all: bool = False, is_validation: bool = False) -> tuple:
      """Load training data by sampling and computing message/backward values.

      Args:
          num_samples: Number of samples to generate (ignored if all=True)
          all: If True, enumerate all assignments instead of sampling
          is_validation: If True, generates validation set

      Returns:
          Tuple of (x, y, bw) where...
      """
  ```
- Comprehensive docstrings in public methods, sparse in private methods
- Emphasis on Args, Returns; implementation details in body comments

## Function Design

**Size:**
- Small functions: 5-30 lines (getters, utilities)
- Medium functions: 30-150 lines (core algorithms like `forward()`, `load()`)
- Large functions: 200+ lines (orchestration like `compute_message_nn()`, `train()`)

**Parameters:**
- Config dictionaries passed as `bucket.config` or `self.config` (dict) rather than individual args
- Named parameters preferred over positional
- Optional parameters with None defaults: `def __init__(self, ..., hidden_sizes=None)`
- Type hints selective: `def forward(self, x: torch.Tensor) -> torch.Tensor:`

**Return Values:**
- Single returns common: `return torch.Tensor`, `return FastFactor`
- Tuple returns for multiple values: `return (bw_msg, message)` or `return (bw_msg, message, bw_partitions)`
- Dictionary returns for structured data: `return {'input': ..., 'target': ..., 'bw_hat': ...}`
- None returns on error paths or when not applicable

## Module Design

**Exports:**
- No `__all__` declarations detected
- Public classes/functions exported by default
- Private functions prefixed with underscore: `_initialize_normalizing_constant()`, `_get_backward_factors()`

**Barrel Files:**
- Package `__init__.py` files typically empty: `/home/cohenn1/NCE/nce/__init__.py`, `/home/cohenn1/NCE/nce/inference/__init__.py`
- No centralized re-exports; consumers import directly: `from nce.inference.graphical_model import FastGM`

---

*Convention analysis: 2026-02-21*
