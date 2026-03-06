# NCE Project - Claude Code Guidance

## Notifications

**MANDATORY:** Whenever you complete a task, finish a phase, encounter a blocker, or are waiting for user input, ping Nick on Discord:

```bash
~/.claude/ai-ops/scripts/ping_nick.sh "brief description of what finished or what's needed"
```

- Never skip this step — the user depends on these pings to know when to check in

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

NCE is a Python package implementing neural network-based inference for graphical models. It combines variable elimination with neural network approximations to perform approximate inference on probabilistic graphical models (PGMs), particularly using Weighted Mini-Bucket elimination (WMB).

## Core Architecture

### Three-Layer Design

The codebase follows a three-layer architecture:

1. **Inference Layer** (`nce/inference/`): Core graphical model operations
   - `graphical_model.py`: `FastGM` class orchestrates the entire inference process
   - `bucket.py`: `FastBucket` represents buckets in the bucket elimination algorithm
   - `factor.py`: `FastFactor` for tensor-based factor operations (works in log-space)
   - `factor_nn.py`: `FactorNN` wraps neural networks as factors
   - `elimination_order.py`: Computes elimination orders using weighted min-fill heuristic

2. **Neural Network Layer** (`nce/neural_networks/`): Training and loss functions
   - `train.py`: `Trainer` class handles NN factor training
   - `net.py`: Neural network architectures (`Net`, `SimpleNet`)
   - `losses.py`: Custom loss functions (KL, MSE variants, message gradient losses)
   - `decision_tree.py`: Decision tree-based factors as an alternative to NNs

3. **Sampling Layer** (`nce/sampling/`): Data generation for training
   - `sample_generator.py`: `SampleGenerator` creates training samples from buckets

### Key Execution Flow

1. Load graphical model from UAI file or factor list → `FastGM.__init__`
2. Compute elimination order → `wtminfill_order()`
3. Organize factors into buckets → `_create_buckets_from_factors()`
4. For each bucket (in elimination order):
   - Generate samples → `SampleGenerator.sample_assignments()`
   - Train neural network → `Trainer.train()`
   - Compute message → `FastBucket.compute_message_nn()` or `compute_message_exact()`
5. Messages propagate through the bucket tree

### Factor Operations (Log-Space)

All factor operations are performed in **log-space**:
- `FastFactor.__mul__` performs addition in log-space (multiplication in linear space)
- `FastFactor.__matmul__` is reserved for true multiplication in log-space
- `eliminate()` performs log-sum-exp marginalization

## Development Setup

### Installation

```bash
# Activate virtual environment
source venv/bin/activate

# Install package in development mode
pip install -e .
```

### Python Environment

- Uses virtual environment at `venv/`
- Python executable: `/home/cohenn1/NCE/venv/bin/python`
- Main dependencies: PyTorch, pyGMs, numpy

## Configuration System

Neural network configurations are Python dictionaries (see `configs/example_nn_config.py`). Key parameters:

- `iB`: Mini-bucket i-bound (limits bucket width)
- `ecl`: Exact computation limit (when to switch from exact to NN)
- `loss_fn`: Loss function name (e.g., 'logspace_mse_fdb', 'linspace_mse_fdb', 'approx_smg')
- `sampling_scheme`: How to generate training samples ('uniform', 'mg', 'all')
- `num_epochs`, `batch_size`, `lr`: Standard NN training hyperparameters
- `hidden_sizes`: List of hidden layer sizes (empty list = linear model)
- `device`: 'cuda' or 'cpu'

## Experiment Execution Rules

**NEVER invent timeouts, resource limits, or runtime assumptions.** If you don't know how long
an experiment will take:
1. Check existing run data in the project (notebooks/, nbe_eval_results/, lab_notebook.txt)
2. Estimate from the computation: num_epochs x num_buckets x per-epoch cost
3. If still uncertain, ASK the user via Discord ping before running
4. When in doubt, run WITHOUT a timeout rather than with an arbitrary one

**"Full" experiments mean full.** When the user says "run the full experiment" or "all epochs",
do not add timeouts, reduce epochs, or truncate in any way unless explicitly told to.

**Before running any experiment that trains neural networks:**
- Check the config's num_epochs value
- Count NN-eligible buckets (use get_large_message_buckets or prior run data)
- If total training iterations > 10,000, warn the user about expected runtime
- Run long experiments in background with no timeout; ping Discord when done

## Assumption Escalation

When you are about to hardcode a value that the user did not specify (timeout, batch size,
number of iterations, resource limits, etc.):

1. **Is this value observable?** Check existing data, configs, or prior runs.
2. **Is this value derivable?** Can you compute a reasonable value from known quantities?
3. **If neither:** STOP and ask the user. Ping Discord:
   `~/.claude/ai-ops/scripts/ping_nick.sh "Need input: [what you need to know and why]"`

**Never** silently invent operational parameters. A missing timeout is better than a wrong one.
The cost of asking is minutes. The cost of a wrong assumption is hours or days.

## Working with the Codebase

### Running Experiments

Experiments are run in the `notebooks/` folder, usually inside the subfolder of the current month.

### Testing

No formal test suite exists. Testing is done through notebooks:
- `notebooks/July-2025/test_new_loss_NN.ipynb`
- `notebooks/Older/test_loss_fns.py`

### Understanding Message Gradients

The system supports training with approximate message gradients:
- `message_gradient_factors.py`: Computes factors for gradient approximation
- Loss functions with 'approx_smg' use approximate message gradient statistics
- `gather_message_stats` config flag enables collection of gradient statistics

### Loss Functions

Multiple loss function variants in `nce/neural_networks/losses.py`:
- `logspace_mse_fdb`: MSE on log-values
- `linspace_mse_fdb`: MSE on normalized probabilities (forward diff through normalizer)
- `mg_sampled_loss_fdb`: Uses sampled message gradients
- `unnormalized_kl`: KL divergence variant

Loss function names ending in `_fdb` indicate "forward diff barrier" (stop-gradient on normalizers).

### Data/Statistics Handling

- `nce/utils/stats.py`: Utilities for computing message statistics
- `nce/utils/plots.py`: Plotting functions for comparing exact vs approximate messages
- Pre-computed statistics can be passed via `stats` parameter in `FastGM.__init__`

### Benchmark Problems

Benchmark problem sets are defined in `nce/benchmark_problems/`. Each set is a `BenchmarkSet`
object with `.problems` (list of pyGMs `Model` objects) and `.configs` (dict of config lists)
attributes.

```python
from nce.benchmark_problems import nbe_sanity_check
for model, config in zip(nbe_sanity_check.problems, nbe_sanity_check.configs['nbe']):
    fastgm = FastGM(model=model, nn_config=config, device=config['device'])
```

For instructions on creating new benchmark sets, see `docs/creating_benchmark_sets.md`.

### Important Implementation Details

1. **Device Management**: Factors and buckets must be on consistent devices (CUDA/CPU)
2. **NN vs Exact**: Buckets decide whether to use exact computation or NN based on `ecl` parameter
3. **Scope Tracking**: Message scopes are tracked in `FastGM.message_scopes` dictionary
4. **Elimination Variables**: Each bucket eliminates a subset of variables (`bucket.elim_vars`)

## Git Workflow

Recent development history shows:
- Active work on decision tree functionality
- Convergence debugging and loss function testing
- Message gradient calculation enhancements
- Sample generation from NN factors

## File Organization Notes

- Old package structure directories (NCE/, data/, inference/, etc.) have been deleted
- Current active codebase is in `nce/` directory
- Package uses setuptools with minimal configuration
- Data files (`.pkl`) and output images are in root directory (gitignored)

# ai-ops Framework
# If the import below fails, replace with: @/home/cohenn1/.claude/ai-ops/bootstrap.md
@~/.claude/ai-ops/bootstrap.md
