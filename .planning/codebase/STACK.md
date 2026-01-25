# Technology Stack

**Analysis Date:** 2026-01-25

## Languages

**Primary:**
- Python 3.11.7 - All application code, inference engine, neural network training, and utilities

## Runtime

**Environment:**
- Python 3.11.7 (from `/home/cohenn1/python311/bin/python3.11`)
- Virtual environment at `venv/` with isolated package dependencies

**Package Manager:**
- pip (with setuptools 45.2.0)
- Lockfile: Not detected (no requirements.txt or lock file, uses setup.py)

## Frameworks

**Core:**
- PyTorch (torch) - Deep learning framework for neural network layers and training
- pyGMs - Graphical models library for factor operations and variable elimination

**Neural Networks:**
- torch.nn - Neural network module definitions (`nce/neural_networks/net.py`)
- torch.optim - Optimization (Adam, LBFGS optimizers in `nce/neural_networks/train.py`)
- adabelief-pytorch 0.2.1 - Alternative optimizer (commented out in `nce/neural_networks/net.py`)

**Visualization:**
- matplotlib.pyplot - Plotting and visualization in `nce/utils/plots.py`
- torchviz 0.0.3 - Neural network visualization (commented out)
- torchinfo 1.8.0 - Network summary utilities

**Data Processing:**
- scikit-learn - DecisionTreeRegressor used in `nce/neural_networks/decision_tree.py`
- numpy 1.26.4 - Numerical computing throughout
- scipy 1.10.1 - Scientific computing utilities

**Development:**
- tqdm.notebook - Progress bars in notebooks and training loops (imported in `nce/neural_networks/train.py`)
- jupyter - Interactive notebook environment for experiments

## Key Dependencies

**Critical:**
- PyTorch (torch) - Core tensor operations in log-space throughout inference layer
  - Imported in: `nce/inference/factor.py`, `nce/neural_networks/net.py`, `nce/neural_networks/train.py`, `nce/neural_networks/losses.py`, `nce/sampling/sample_generator.py`
- pyGMs - Graphical models operations and factor manipulation
  - Imported in: `nce/inference/graphical_model.py`, `nce/inference/fastElim.py`
- numpy - Array operations and mathematical computations
  - Used throughout for data preprocessing and manipulation

**Infrastructure:**
- adabelief-pytorch 0.2.1 - Advanced optimizer option (currently commented out)
- scikit-learn 0.23.2 - Decision tree algorithms for factor approximation
- matplotlib 3.3.2 - Visualization for debugging and analysis

## Configuration

**Environment:**
- Python virtual environment at `venv/` with custom python executable
- Device configuration: Runtime-controlled via `device` config parameter ('cuda' or 'cpu')
- Configuration passed as Python dictionaries (see `configs/example_nn_config.py`)

**Build:**
- `setup.py` - Minimal setuptools configuration with package discovery
- Package name: "nce"
- Version: 0.1
- Packages auto-discovered in `nce` namespace

**Key Configuration Parameters:**
- `iB`: Mini-bucket i-bound (limits bucket complexity)
- `ecl`: Exact computation limit (threshold for exact vs NN approximation)
- `loss_fn`: Loss function selection ('logspace_mse_fdb', 'linspace_mse_fdb', 'approx_smg', etc.)
- `sampling_scheme`: Sample generation mode ('uniform', 'mg', 'all')
- `hidden_sizes`: Neural network architecture (list or 'bias_only' for linear models)
- `device`: 'cuda' or 'cpu'
- `num_epochs`, `batch_size`, `lr`: Standard training hyperparameters

## Platform Requirements

**Development:**
- Linux environment (verified on Ubuntu-based system)
- Python 3.11.7
- CUDA-capable GPU (optional, fallback to CPU)
- Virtual environment with pip

**Production:**
- Python 3.11.7 runtime
- PyTorch with CPU or CUDA support
- Approximately 14990 lines of Python code across inference, neural networks, sampling, and utilities

**Data:**
- UAI format graphical model files (read via `pyGMs.filetypes.readEvidence14`)
- Pickle files for serialized models and benchmarks
- CSV/data files in root and subdirectories (e.g., `casino.csv`)

---

*Stack analysis: 2026-01-25*
