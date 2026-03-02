# Technology Stack

**Analysis Date:** 2026-02-21

## Languages

**Primary:**
- Python 3.11.7 - Core implementation for inference, neural networks, data handling, and sampling

## Runtime

**Environment:**
- CPython 3.11.7

**Package Manager:**
- pip
- Lockfile: Not detected (no requirements.txt or poetry.lock)

## Frameworks

**Core ML:**
- PyTorch 2.0.1+cu117 - Neural network training, inference, and GPU computation (CUDA support)
- PyGMs 0.1.1 - Probabilistic graphical model inference primitives
  - Location: `/home/cohenn1/SDBE/PyGMs` (local editable install)
  - Used by `nce/inference/graphical_model.py`, `nce/inference/bucket.py`, `nce/utils/pygms_conversion.py`

**Optimization:**
- muon-optimizer 0.1.0 - Advanced optimizer used in `nce/neural_networks/train.py`

**Scientific Computing:**
- NumPy 1.26.4 - Array operations and numerical utilities
- SciPy 1.15.2 - Scientific computation utilities

**Machine Learning Utilities:**
- scikit-learn (sklearn) - DecisionTreeRegressor in `nce/neural_networks/decision_tree.py` and `nce/neural_networks/dt2.py`

**Visualization:**
- Matplotlib 3.10.1 - Plotting and visualization used throughout codebase

**Development:**
- tqdm 4.67.1 - Progress bars in `nce/neural_networks/train.py`, `nce/inference/graphical_model.py`
- Jupyter/IPython - Notebook environment with tqdm.notebook integration

**Audio/Vision (installed but not directly used in core inference):**
- torchaudio 2.0.2+cu117
- torchvision 0.15.2+cu117

## Key Dependencies

**Critical:**
- PyTorch 2.0.1 with CUDA 11.7 - GPU acceleration for message approximation networks
- PyGMs 0.1.1 - Weighted Mini-Bucket elimination and factor/variable abstractions
- NumPy 1.26.4 - Underlying tensor operations and numerical stability

**Optimization:**
- muon-optimizer 0.1.0 - Alternative to Adam/SGD in message training loops (see `train.py` for usage)

**Alternative/Legacy:**
- adabelief-pytorch - Referenced in commented code (`nce/neural_networks/net.py` line 13, `NN_Train_copy.py`)
- scikit-learn - Decision tree approximators for bucket message inference (experimental)

## Configuration

**Environment:**
- Device selection: Controlled via `device` parameter (e.g., 'cuda', 'cpu') in config dictionaries
- No .env file or environment variable loading detected
- Configuration passed as Python dictionaries in notebooks and scripts

**Build:**
- `setup.py` - Minimal setuptools configuration
  - Location: `/home/cohenn1/NCE/setup.py`
  - Package name: `nce`
  - Version: 0.1
  - Auto-discover packages under `nce/` directory

- `pyproject.toml` - Build system metadata
  - Location: `/home/cohenn1/NCE/claude_files/pyproject.toml`
  - Build backend: setuptools.build_meta
  - Requires: setuptools>=64, wheel

## Platform Requirements

**Development:**
- Python 3.11+
- CUDA 11.7 toolkit (for GPU support)
- C++ compiler (for PyTorch compilation if building from source)

**Production:**
- Deployment target: GPU-accelerated systems with CUDA 11.7
- Fallback to CPU available but not optimized
- Tested primarily in Jupyter notebook environments

---

*Stack analysis: 2026-02-21*
