# Codebase Structure

**Analysis Date:** 2026-02-21

## Directory Layout

```
nce/
├── __init__.py                           # Package root (empty)
├── inference/                            # Core elimination & bucket inference engine
│   ├── __init__.py                       # Public API exports (FastFactor, FactorNN, FastBucket, FastGM, etc.)
│   ├── graphical_model.py                # FastGM: main orchestration class (~1800 lines)
│   ├── bucket.py                         # FastBucket: elimination target containers with message computation
│   ├── factor.py                         # FastFactor: probability factors in log space with tensor ops
│   ├── factor_nn.py                      # FactorNN: neural network-based factors
│   ├── factor_qdecision_tree.py          # Decision tree factor variant
│   ├── fastElim.py                       # Fast elimination utilities (new)
│   ├── elimination_order.py              # Elimination order computation (wtminfill_order)
│   ├── nn_factors.py                     # Conversions: NN to FastFactor (nn_to_FastFactor)
│   ├── message_gradient_factors.py       # Backward message computation & WMB support
│   └── utils.py                          # Factor ordering and helper functions
│
├── neural_networks/                      # Network training & architecture
│   ├── __init__.py                       # Public API: Net, Memorizer, BitVectorLookup, DecisionTreeLossOptimizer, Trainer, losses
│   ├── net.py                            # Net: MLP with configurable hidden layers, bias_only mode, linspace bias
│   ├── train.py                          # Trainer: training orchestration (~500 lines)
│   ├── losses.py                         # Loss functions: KL divergence, MSE, UKF variants, message gradient variants
│   ├── simple_net.py                     # SimpleNet: simplified network variant
│   ├── decision_tree.py                  # DecisionTreeLossOptimizer: decision tree learning
│   ├── dt2.py                            # Decision tree variant 2
│   ├── linear_mse_solver.py              # Linear solver for MSE optimization
│   ├── message_trainer.py                # Message-specific training (legacy)
│   ├── ukf_helpers.py                    # Unscented Kalman Filter utilities
│   ├── quantization.py                   # Quantization utilities (new)
│   ├── NN_Train_copy.py                  # Legacy training script (copy for reference)
│   └── train_old.py                      # Deprecated training version
│
├── data/                                 # Training data generation & preprocessing
│   ├── __init__.py                       # Public API: DataLoader, create_data_loaders, DataPreprocessor
│   ├── data_loader.py                    # DataLoader: orchestrates sampling, message computation, normalization
│   └── data_preprocessor.py              # DataPreprocessor: normalization with logsumexp, bw-aware centering
│
├── sampling/                             # Sample generation
│   ├── __init__.py                       # Public API: SampleGenerator
│   └── sample_generator.py               # SampleGenerator: deterministic sampling with seed control
│
├── problems/                             # Benchmark problem collections (UAI format)
│   ├── __init__.py
│   ├── test_problems.py                  # TestProblem class & standard benchmark registry
│   ├── 8-5-benchmarks/                   # Benchmark suite (8-5-benchmark=True problems)
│   ├── new8-5 copy/                      # Benchmark suite copy
│   ├── width_under_20_problems/          # Tree-width < 20 problems
│   ├── width20-30/                       # Tree-width 20-30 problems (includes pedigree, rbm, grids)
│   ├── width_over_30/                    # Tree-width > 30 problems
│   ├── kings_and_princes/                # King's graph specific problems
│   ├── teeny_weenee/                     # Tiny test problems
│   ├── temp/                             # Temporary problem files
│   ├── benchmarks_12_4_2025.pkl          # Serialized benchmark metadata
│   ├── nbe_benchmarks.pkl                # Alternative benchmark metadata
│   └── *.uai, *.uai.vo, *.uai.evid       # Model files (PyGMs format)
│
└── utils/                                # Utilities & cross-cutting helpers
    ├── __init__.py                       # Public API: get_message_gradient, get_backward_message, plot_fastfactor_comparison, lse, etc.
    ├── message_gradient.py               # get_message_gradient(): compute backward message for gradient learning
    ├── backward_message.py               # get_backward_message(): compute backward factors (WMB support)
    ├── backward_sensitivity.py           # Sensitivity analysis utilities (new)
    ├── stats.py                          # Statistics collection & analysis
    ├── plots.py                          # Visualization utilities
    ├── pygms_conversion.py               # PyGMs compatibility utilities (new)
    ├── pygms_wmb_interface.py            # WMB interface to PyGMs (new)
    └── [others]                          # Additional helper modules
```

## Directory Purposes

**`nce/inference/`:**
- Purpose: Probabilistic inference engine - bucket elimination with approximate/exact computation
- Contains: Factor representations, bucket management, elimination algorithms
- Key files:
  - `graphical_model.py` (FastGM main orchestrator class)
  - `bucket.py` (elimination targets and message computation)
  - `factor.py` (core tensor operations in log space)
- Depends on: PyGMs library, torch
- No internal test files (testing in notebooks)

**`nce/neural_networks/`:**
- Purpose: Neural network approximation and training infrastructure
- Contains: Network architectures, training loops, loss functions, optimization
- Key files:
  - `net.py` (network definition with configurable layers)
  - `train.py` (training orchestration and learning rate scheduling)
  - `losses.py` (all loss variants: MSE, KL, message gradient)
- Variants: decision trees, linear solvers, quantization
- Depends on: inference, data, sampling, torch

**`nce/data/`:**
- Purpose: Training data preparation for neural network approximation
- Contains: Sample generation orchestration, data normalization, encoding
- Key files:
  - `data_loader.py` (main loader orchestrator with bw_factors support)
  - `data_preprocessor.py` (normalization using logsumexp with backward message support)
- Depends on: inference, sampling

**`nce/sampling/`:**
- Purpose: Training sample generation and factor evaluation
- Contains: Sample creation from message scope, forward/backward message computation
- Key files:
  - `sample_generator.py` (deterministic seeding, factor evaluation, message value computation)
- Seeding pattern: bucket_label + global_seed * 10000 + counter * 100 + validation_offset
- Depends on: inference

**`nce/problems/`:**
- Purpose: Benchmark problem instances for testing and evaluation
- Contains: Subdirectories by problem width/type, loadable as UAI files
- Key files:
  - `test_problems.py` (TestProblem class and registry for accessing benchmarks)
- File format: UAI files with .vo (variable order) and .evid (evidence) files
- Coverage: Grids, RBMs, pedigrees, constraint satisfaction problems
- Depends on: inference (FastGM for loading)

**`nce/utils/`:**
- Purpose: Cross-cutting utilities: gradients, statistics, visualization
- Contains: Message gradient, backward message helpers, stats collection, plotting
- Key files:
  - `message_gradient.py` (get_message_gradient for gradient-informed training)
  - `backward_message.py` (get_backward_message with WMB support)
  - `stats.py` (message statistics tracking)
- Depends on: inference, data

## Key File Locations

**Entry Points (User-facing):**
- `nce/inference/graphical_model.py::FastGM.__init__()` - Create graphical model from UAI file or factors
- `nce/inference/graphical_model.py::FastGM.eliminate_variables()` - Run inference
- `nce/inference/bucket.py::FastBucket.compute_message_nn()` - Train neural approximation
- `nce/neural_networks/train.py::Trainer.train()` - Run training loop

**Configuration:**
- Config passed as `nn_config` dict to `FastGM.__init__()`
- No separate config files; configuration is programmatic via Python dicts
- Config stored in `FastGM.config` (dict copy, not reference, to avoid sharing)
- Key config parameters:
  - `device`: 'cuda' or 'cpu'
  - `loss_fn`: Loss function name (string key)
  - `hidden_sizes`: Network architecture (list of ints, or 'bias_only')
  - `num_epochs`, `batch_size`, `lr`: Training hyperparameters
  - `sampling_scheme`: 'all' or 'uniform'
  - `num_samples`: Number of training samples to generate

**Core Logic:**
- Variable elimination: `nce/inference/graphical_model.py::FastGM.eliminate_variables()` (~230+ lines)
- Bucket processing: `nce/inference/graphical_model.py::FastGM.process_bucket()`
- Message computation (exact): `nce/inference/bucket.py::FastBucket.compute_message_exact()`
- Message computation (NN): `nce/inference/bucket.py::FastBucket.compute_message_nn()`
- Training loop: `nce/neural_networks/train.py::Trainer.train()`
- Loss functions: `nce/neural_networks/losses.py` (all variants)
- Sampling: `nce/sampling/sample_generator.py::SampleGenerator.sample_assignments()`
- Data loading: `nce/data/data_loader.py::DataLoader.load()`

**Testing & Benchmarking:**
- No standard test/ directory or pytest fixtures
- Benchmark problems: `nce/problems/test_problems.py` (predefined via TestProblem class)
- Ad-hoc testing: `notebooks/` directory (exploration, experimentation, not part of package)
- Example notebook: `notebooks/2025-07/test_new_loss_NN.py`

## Naming Conventions

**Files:**
- Module files: lowercase snake_case (`graphical_model.py`, `data_loader.py`, `sample_generator.py`)
- Legacy files: Uppercase or mixed case (`NN_Train_copy.py`, `train_old.py`)
- Marker for deprecated: `_old`, `_copy` suffix

**Functions:**
- Public methods: `eliminate_variables()`, `compute_message_exact()`, `get_message_scope()`
- Verb-noun pattern: `compute_`, `get_`, `create_`, `load_`, `eliminate_`
- Private methods: Leading underscore `_compute_seed()`, `_initialize_normalizing_constant()`
- Conversion methods: `to_exact()` prefix

**Variables:**
- Instance attributes: `self.config`, `self.device`, `self.buckets`, `self.elim_order` (all lowercase with underscores)
- Class names: PascalCase (`FastGM`, `FastBucket`, `FastFactor`, `FactorNN`, `Net`, `Trainer`, `SampleGenerator`, `DataLoader`, `DataPreprocessor`)
- Type indicators: Factor types have `Fast` prefix (exact) or `NN` infix (neural)
- Boolean flags: `is_nn`, `is_root`, `use_memorizer`, `populate_bw_factors`

**Types:**
- Factors: `FastFactor` (exact), `FactorNN` (neural network approximation)
- Models: `FastGM` (graphical model orchestrator)
- Containers: `FastBucket` (elimination targets)
- Networks: `Net` (PyTorch nn.Module), `SimpleNet` (variant)
- Training: `Trainer` (training orchestrator)
- Data: `SampleGenerator`, `DataLoader`, `DataPreprocessor`

## Where to Add New Code

**New Approximation Method (e.g., WMB, decision tree):**
- Implementation: `nce/inference/bucket.py` - Add `compute_message_xxx()` method alongside existing `compute_message_exact()`, `compute_message_nn()`, `compute_message_wmb()`, `compute_message_dt()`
- Configuration parsing: Check config flags in method (e.g., `self.config.get('use_wmb', False)`)
- Integration: Called from `FastBucket.compute_message_nn()` based on message scope and config
- Test: Create FastBucket with config flag and call `compute_message_xxx()`

**New Loss Function:**
- Implementation: `nce/neural_networks/losses.py` - Define function matching signature:
  ```python
  def my_loss(outputs, targets, **kwargs):
      # outputs: predicted messages (normalized)
      # targets: true messages (normalized)
      # kwargs may include: bw_hat, sigma_f, sigma_g, bw_normalizing_constant, max_val, etc.
      return loss_scalar
  ```
- Integration: Import in `nce/neural_networks/train.py`, add to loss function mapping in `Trainer._get_loss_fn()`
- Configuration: Reference in config `{'loss_fn': 'my_loss'}`
- Key: Losses work in normalized log space; targets and outputs are already centered

**New Neural Network Architecture:**
- Implementation: `nce/neural_networks/net.py` - Extend `Net` class or create new class inheriting from `nn.Module`
  - Accept `bucket`, `hidden_sizes`, `device`, `seed` parameters
  - Implement `forward(x)` taking one-hot encoded assignments
  - Return log space values as single output per sample
- Configuration: Modify `Net.__init__()` to accept new architecture selection via config
- Integration: Pass to `Trainer` via `bucket.compute_message_nn()`
- Example: `SimpleNet` class for alternative architecture

**New Utility Function (cross-cutting):**
- Shared statistics: `nce/utils/stats.py` - Add to existing file
- Backward message helpers: `nce/utils/backward_message.py` - Add to existing file
- Message gradient computation: `nce/utils/message_gradient.py` - Add to existing file
- Plotting/visualization: `nce/utils/plots.py` - Add to existing file
- If crosses multiple utilities: Create new utils submodule

**New Benchmark Problem:**
- Location: `nce/problems/[problem_type]/` - Create subdirectory per problem width/type
- Format: UAI model file format (`.uai` extension, `.vo` for variable order, `.evid` for evidence)
- API registration: Add entry to `test_problems.py` dict with config dict:
  ```python
  my_problem = {
      "name": "my_problem_name",
      "width": 12,
      "nvars": 100,
      "uai_file": "/path/to/my_problem.uai",
      "Z": 123.456,  # ground truth partition function
      "interesting_buckets": []
  }
  test_problems["my_problem_name"] = TestProblem(my_problem)
  ```
- Discovery: Access via `test_problems["my_problem_name"]`

## Special Directories

**`nce/__pycache__/`:**
- Purpose: Python bytecode cache
- Generated: Automatically by Python interpreter
- Committed: No (in .gitignore)
- Cleanup: Safe to delete; regenerated on next import

**`nce.egg-info/` / `nce_package_files/`:**
- Purpose: Package installation metadata (setuptools)
- Generated: Yes, created by `pip install -e .`
- Committed: No (in .gitignore)

**`notebooks/`:**
- Purpose: Exploration, experimentation, and benchmark scripts
- Structure: Subdirectories by date (`2025-07/`, `2025-08/`, `09-2025/`, `Older/`)
- Types: Python scripts (cells marked with `#%%` comment blocks for notebook compatibility)
- Committed: Yes (but separate from package code)
- Key examples:
  - `notebooks/2025-07/test_new_loss_NN.py` - NN training workflow example
  - `notebooks/2025-08/benchmark_linear_loss2.py` - Linear solver benchmarking
  - `notebooks/09-2025/decision_tree_test.py` - Decision tree evaluation

**`configs/`:**
- Purpose: Configuration files (if any)
- Status: Present in repo structure but minimal content observed
- Format: Depends on use case; typically JSON, YAML, or Python dicts

**`graphs/`:**
- Purpose: Graph visualization output
- Generated: User-created during exploration
- Committed: Yes

## Import Patterns

**Within package (circular dependency prevention):**
```python
# From inference module
from nce.inference import FastGM, FastBucket, FastFactor, FactorNN
from nce.inference import wtminfill_order, nn_to_FastFactor

# From neural networks
from nce.neural_networks import Net, Trainer, losses

# From data/sampling
from nce.data import DataLoader, DataPreprocessor, create_data_loaders
from nce.sampling import SampleGenerator

# From utils (high-level utilities)
from nce.utils import get_message_gradient, get_backward_message
```

**External dependencies:**
```python
import torch
import torch.nn as nn
import torch.optim as optim
from pyGMs import Var  # Variable type from PyGMs
import pyGMs as gm    # Full PyGMs module
from pyGMs.wmb import ...  # WMB approximation
from pyGMs.neuro import ...  # PyGMs neural utilities
```

## Dependency Graph

```
FastGM (orchestration)
  ├→ FastBucket (elimination targets)
  │  ├→ FastFactor (exact factors)
  │  ├→ FactorNN (learned factors)
  │  └→ FastBucket.compute_message_nn()
  │     ├→ Trainer (training loop)
  │     │  ├→ Net (network architecture)
  │     │  ├→ losses.py (loss functions)
  │     │  ├→ torch.optim (optimizer)
  │     │  └→ SampleGenerator
  │     │     ├→ DataLoader
  │     │     └→ DataPreprocessor
  │     └→ Optional: get_backward_message()
  │        └→ Creates downstream FastGM for backward factors
  │
  └→ FastGM.eliminate_variables()
     └→ process_bucket() for each bucket
        ├→ compute_message_exact() (default)
        ├→ compute_message_nn() (if config['approximation_method'] == 'nn')
        ├→ compute_message_wmb() (if config['approximation_method'] == 'wmb')
        └→ compute_message_dt() (if config['approximation_method'] == 'dt')
```

## Configuration File Locations

- **No dedicated config files** - configuration is programmatic
- **In-code configuration**: `nn_config` dict passed to `FastGM()`
- **Environment variables**: Not used in core code; setup via Python
- **Hyperparameter defaults**: Specified in function signatures with `.get()` fallbacks in `FastGM.__init__()`

---

*Structure analysis: 2026-02-21*
