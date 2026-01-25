# Codebase Structure

**Analysis Date:** 2026-01-25

## Directory Layout

```
nce/
├── __init__.py                           # Package root
├── inference/                            # Core elimination & bucket inference
│   ├── __init__.py                       # Public API exports
│   ├── graphical_model.py                # FastGM: main orchestration class (1800+ lines)
│   ├── bucket.py                         # FastBucket: elimination target containers
│   ├── factor.py                         # FastFactor: probability factors in log space
│   ├── factor_nn.py                      # FactorNN: neural network-based factors
│   ├── factor_qdecision_tree.py          # Decision tree factor variant
│   ├── fastElim.py                       # Fast elimination utilities
│   ├── elimination_order.py              # Elimination order computation (wtminfill)
│   ├── nn_factors.py                     # Conversions between NN and FastFactor
│   ├── message_gradient_factors.py       # Backward message computation
│   └── utils.py                          # Helper functions for inference
├── neural_networks/                      # Network training & architecture
│   ├── __init__.py
│   ├── net.py                            # Net: MLP/linear network definitions
│   ├── train.py                          # Trainer: training orchestration (500+ lines)
│   ├── losses.py                         # Loss functions (KL, MSE, UKF variants)
│   ├── simple_net.py                     # Simplified network architectures
│   ├── decision_tree.py                  # Decision tree learning
│   ├── dt2.py                            # Decision tree variant 2
│   ├── linear_mse_solver.py              # Linear solver for MSE optimization
│   ├── message_trainer.py                # Message-specific training
│   ├── ukf_helpers.py                    # Unscented Kalman Filter utilities
│   ├── NN_Train_copy.py                  # Legacy training script (for reference)
│   └── train_old.py                      # Deprecated training version
├── data/                                 # Training data generation
│   ├── __init__.py
│   ├── data_loader.py                    # DataLoader: orchestrates sampling & loading
│   └── data_preprocessor.py              # DataPreprocessor: normalization & encoding
├── sampling/                             # Sample generation
│   ├── __init__.py
│   └── sample_generator.py               # SampleGenerator: creates training samples
├── problems/                             # Benchmark problem collections
│   ├── test_problems.py                  # Standard test benchmark definitions
│   ├── 8-5-benchmarks/                   # Benchmark suite 1
│   ├── kings_and_princes/                # King's graph problem variants
│   ├── width_under_20_problems/          # Width-limited benchmarks
│   ├── width20-30/                       # Medium-width problems
│   └── width_over_30/                    # High-width problems
└── utils/                                # Utilities & helpers
    ├── __init__.py
    ├── message_gradient.py               # Message gradient computation
    ├── backward_message.py               # Backward message helpers
    ├── stats.py                          # Statistics collection & analysis
    └── plots.py                          # Visualization utilities
```

## Directory Purposes

**`nce/inference/`:**
- Purpose: Probabilistic inference engine - bucket elimination with approximate/exact computation
- Contains: Factor representations, bucket management, elimination algorithms
- Key files: `graphical_model.py` (FastGM main class), `bucket.py` (elimination targets), `factor.py` (core tensor operations)

**`nce/neural_networks/`:**
- Purpose: Neural network approximation and training infrastructure
- Contains: Network architectures, training loops, loss functions, optimization
- Key files: `net.py` (network definition), `train.py` (training orchestration), `losses.py` (all loss variants)

**`nce/data/`:**
- Purpose: Training data preparation for neural network approximation
- Contains: Sample generation orchestration, data normalization, encoding
- Key files: `data_loader.py` (main loader orchestrator), `data_preprocessor.py` (normalization logic)

**`nce/sampling/`:**
- Purpose: Training sample generation and evaluation
- Contains: Sample creation from message scope, forward/backward message computation
- Key files: `sample_generator.py` (deterministic seeding, factor evaluation)

**`nce/problems/`:**
- Purpose: Benchmark problem instances for testing and evaluation
- Contains: Subdirectories by problem width/type, loadable as UAI files
- Key files: `test_problems.py` (API for accessing benchmarks)

**`nce/utils/`:**
- Purpose: Cross-cutting utilities: gradients, statistics, visualization
- Contains: Message gradient, backward message helpers, stats collection, plotting
- Key files: `message_gradient.py` (gradient computation), `stats.py` (statistics tracking)

## Key File Locations

**Entry Points:**
- `nce/inference/graphical_model.py`: FastGM class - user creates instances here
- `nce/neural_networks/train.py`: Trainer class - training entry point (called by bucket)
- `nce/sampling/sample_generator.py`: SampleGenerator class - data generation entry point

**Configuration:**
- Config passed as `nn_config` dict to FastGM.__init__()
- No separate config files; configuration is programmatic
- See `nce/inference/graphical_model.py` lines 20-66 for config parsing

**Core Logic:**
- Variable elimination: `nce/inference/graphical_model.py` FastGM.eliminate_variables() (230+ line method)
- Bucket processing: `nce/inference/graphical_model.py` FastGM.process_bucket()
- Message computation: `nce/inference/bucket.py` compute_message_exact/nn/wmb()
- Training: `nce/neural_networks/train.py` Trainer.train()
- Loss functions: `nce/neural_networks/losses.py` (all loss variants)

**Testing:**
- Unit tests: Not organized in standard test/ directory
- Benchmark problems: `nce/problems/` subdirectories
- Ad-hoc testing: `notebooks/` directory (exploration, not part of package)

## Naming Conventions

**Files:**
- `graphical_model.py`: CamelCase class names in lowercase files
- `loss_functions.py`: snake_case module names
- `data_loader.py`: Underscores for multi-word filenames
- Legacy code: `NN_Train_copy.py`, `train_old.py` (uppercase + deprecated suffix)

**Functions:**
- `eliminate_variables()`: verb_noun pattern
- `compute_message_exact()`: compute_ prefix for expensive operations
- `get_message_scope()`: get_ for accessors
- `_create_mini_buckets()`: Leading underscore for private methods
- `to_exact()`: Conversion methods use to_ prefix

**Variables:**
- `bucket_width`, `message_scope`, `elim_order`: snake_case throughout
- `FastGM`, `FastBucket`, `FastFactor`: Pascal case for classes
- `self.config`, `self.device`, `self.is_nn`: Lowercase with underscores for instance attributes

**Types:**
- Factors: `FastFactor` (exact), `FactorNN` (neural network), `FastFactor` subclasses
- Models: `FastGM` (graphical model)
- Containers: `FastBucket` (elimination targets)
- Networks: `Net` (PyTorch nn.Module)

## Where to Add New Code

**New Approximation Method:**
- Implementation: `nce/inference/bucket.py` - add compute_message_xxx() method alongside compute_message_exact, compute_message_nn, compute_message_wmb
- Integration: `nce/inference/graphical_model.py` process_bucket() - add elif branch for new approximation_method config value
- Test: Call from eliminate_variables() with config['approximation_method'] = 'new_method'

**New Loss Function:**
- Implementation: `nce/neural_networks/losses.py` - define function matching signature: `def loss_name(outputs, targets, **kwargs)`
- Integration: `nce/neural_networks/train.py` Trainer._get_loss_fn() - add case in method mapping
- Configuration: Reference in config['loss_fn'] = 'loss_name'

**New Neural Network Architecture:**
- Implementation: `nce/neural_networks/net.py` - extend Net class or create new class inheriting from nn.Module
- Configuration: Create variant in Net.__init__() or use config to select architecture
- Integration: Pass to Trainer via bucket.compute_message_nn()

**New Utility Function:**
- Shared helpers: `nce/utils/*.py` - place in appropriate existing file (message_gradient, stats, plots)
- If crosses multiple utilities: Create new utils submodule

**New Benchmark:**
- Location: `nce/problems/[problem_type]/` - subdirectory per benchmark
- Format: UAI model files (.uai, .vo, .evid extensions per PyGMs standard)
- API: Register in `nce/problems/test_problems.py` for discovery

## Special Directories

**`nce/__pycache__/`:**
- Purpose: Python bytecode cache
- Generated: Automatically by Python interpreter
- Committed: No (.gitignore)

**`nce.egg-info/`:**
- Purpose: Package installation metadata (setuptools)
- Generated: Yes, created by pip install -e .
- Committed: No

**`notebooks/`:**
- Purpose: Exploration and experimental code
- Generated: User-created
- Committed: Yes (but separate from package code)
- Structure: Subdirectories by date/experiment (July-2025/, Older/)

## Architecture Notes

**Layer Independence:**
- Inference layer (`nce/inference/`) is mostly independent, can compute exact messages
- Neural network layer (`nce/neural_networks/`) depends on inference (gets training targets)
- Data layer (`nce/data/`, `nce/sampling/`) depends on both inference and NN

**Extension Points:**
1. Approximation methods: Add compute_message_xxx in `bucket.py`
2. Loss functions: Add function in `losses.py`
3. Neural architectures: Extend/modify `Net` in `net.py`
4. Optimizers: Handled via config, set in `Trainer.set_optimizer()`
5. Sampling schemes: Extend `SampleGenerator.sample_assignments()`

**Dependency Directions:**
```
FastGM (orchestration)
  ├→ FastBucket (elimination targets)
  │  ├→ FastFactor (exact factors)
  │  ├→ FactorNN (learned factors)
  │  └→ SampleGenerator (data for NN training)
  │     ├→ DataLoader
  │     └→ DataPreprocessor
  └→ Trainer (network training)
     └→ Net (network architecture)
     └→ Loss functions
     └→ Optimizer (configured via config)
```

---

*Structure analysis: 2026-01-25*
