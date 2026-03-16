# NCE — Neural Cluster Elimination

Neural network-based approximate inference for probabilistic graphical models (PGMs). NCE replaces expensive exact computations in bucket elimination with learned neural network approximations, enabling scalable inference on large models.

## Overview

Bucket elimination is a foundational algorithm for exact inference in graphical models, but its complexity grows exponentially with treewidth. **NCE** addresses this by using Weighted Mini-Bucket elimination (WMB) and training neural networks to approximate the messages (intermediate factors) that arise during elimination.

The key idea: when a bucket's scope is too large for exact computation, NCE trains a small neural network on samples from the bucket's factor product, producing an approximate message that is passed forward through the elimination order. This trades exactness for tractability while preserving the structure of the original inference algorithm.

### How It Works

1. **Load a graphical model** from a UAI file or a pyGMs `Model` object
2. **Compute an elimination order** using weighted min-fill heuristic
3. **Organize factors into buckets** based on the elimination order
4. **For each bucket** (in elimination order):
   - If the bucket's message size is within the approximation threshold (`iB`/`iB2`), compute the message exactly
   - Otherwise, generate training samples, train a neural network, and use it as the approximate message
5. **Messages propagate** through the bucket tree, producing an estimate of the partition function (or other query)

All internal factor operations are performed in **log-space** for numerical stability.

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended; CPU mode available)
- [PyTorch](https://pytorch.org/get-started/locally/) with CUDA support

### Install

```bash
git clone https://github.com/Nick-Cohen/Neuro-Cluster-Elimination.git
cd Neuro-Cluster-Elimination

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install PyTorch (select the right CUDA version for your system)
# See https://pytorch.org/get-started/locally/ for the correct command
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Install NCE in development mode
pip install -e .
```

## Quick Start

```python
import pyGMs as gm
from nce.inference.graphical_model import FastGM

# Load a model from a UAI file
fastgm = FastGM(uai_file='path/to/model.uai', device='cuda', nn_config={
    'iB': 10,                # max message scope width before approximation
    'iB2': 14,               # max log2(message_size) before approximation (accounts for domain sizes)
    'hidden_sizes': [32, 32],
    'loss_fn': 'unnormalized_kl',
    'num_epochs': 500,
    'lr': 0.001,
    'num_samples': 5000,
    'sampling_scheme': 'uniform',
})

# Compute the log partition function (trains NNs for large buckets automatically)
log_z = fastgm.get_log_partition_function()
print(f"Log Z ≈ {log_z}")
```

### Using a pyGMs Model

```python
import pyGMs as gm
from nce.inference.graphical_model import FastGM

# Load from pyGMs
model = gm.GraphModel()
model.load('path/to/model.uai')

fastgm = FastGM(model=model, device='cuda', nn_config=config)
log_z = fastgm.get_log_partition_function()
```

## Configuration

NCE supports both flat and nested configuration formats. The nested format is recommended for readability:

```python
config = {
    'inference': {
        'iB': 10,                       # max message scope width before approximation
        'iB2': 14,                      # max log2(message_size) before approximation
        'device': 'cuda',
    },
    'nn': {
        'hidden_sizes': [32, 32],       # empty list = linear model
    },
    'training': {
        'loss_fn': 'unnormalized_kl',   # KL divergence
        'num_epochs': 500,
        'learning_rate': 0.001,
    },
    'sampling': {
        'num_samples': 5000,
        'sampling_scheme': 'uniform',   # 'uniform', 'mg', or 'all'
    },
}
```

See [`configs/example_nn_config.py`](configs/example_nn_config.py) for the full flat and nested config examples.

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `iB` | Max message scope width (number of variables) before using NN approximation | 0 |
| `iB2` | Max log₂(message size) before using NN approximation. Better than `iB` for non-binary domains since it accounts for actual domain sizes. Equivalent to `iB` when all variables are binary. (Alias: `ecl`) | 0 |
| `hidden_sizes` | NN hidden layer sizes (e.g., `[32, 32]`). Empty = linear | `[]` |
| `loss_fn` | Loss function for NN training | (required) |
| `num_epochs` | Training epochs per NN | (required) |
| `num_samples` | Training samples generated per bucket | 5000 |
| `device` | `'cuda'` or `'cpu'` | `'cuda'` |

### Available Loss Functions

- `unnormalized_kl` — KL divergence (recommended default)
- `logspace_mse_fdb` — MSE on log-values (forward diff barrier, use with `sampling_scheme: 'all'`)
- `linspace_mse_fdb` — MSE on normalized probabilities (forward diff barrier, use with `sampling_scheme: 'all'`)
- `neurobe_weighted_mse` — Weighted MSE (NeuroBE reproduction mode)

## Project Structure

```
nce/
├── inference/              # Core graphical model operations
│   ├── graphical_model.py  # FastGM — main entry point
│   ├── bucket.py           # FastBucket — bucket elimination
│   ├── factor.py           # FastFactor — tensor-based log-space factors
│   ├── factor_nn.py        # FactorNN — neural network as a factor
│   └── elimination_order.py
├── neural_networks/        # NN training
│   ├── train.py            # Trainer class
│   ├── net.py              # Network architectures
│   ├── losses.py           # Loss functions
│   └── decision_tree.py    # Decision tree alternative to NNs
├── sampling/               # Training data generation
│   └── sample_generator.py
├── config_schema.py        # Config validation and normalization
├── benchmark_problems/     # Curated problem sets for evaluation
├── state/                  # Checkpoint save/load
├── visualization/          # Plotting utilities
└── utils/
```

## Citation

If you use this code in your research, please cite:

```
@misc{nce2025,
  author = {Nick Cohen},
  title = {Neural Cluster Elimination},
  year = {2025},
  url = {https://github.com/Nick-Cohen/Neuro-Cluster-Elimination}
}
```
