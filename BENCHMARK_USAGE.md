# NCE Benchmark Config Usage Guide

This guide demonstrates the cleaner config interface introduced in M005 using width-based threshold parameters for binary-domain problems.

## Width-Based Parameters for Binary Domains

**Old interface** (from `small_problems` benchmark, circa M001-M004):
```python
config = {
    'ecl': 524287,            # What width does this correspond to? (2^19-1 → width 19)
    'iB': 100,
    'approximation_method': 'nn',
    'bw_ecl': 8388607,        # What width does this correspond to? (2^23-1 → width 23)
    'backward_iB': 100,
    # ... rest of config
}
```

**New interface** (M005 width-based parameters):
```python
config = {
    'inference': {
        'ib2': 19,                    # Binary width 19 → ecl = 2^19-1 = 524287
        'i_bound': 100,
        'approximation_method': 'nn',
    },
    'backward': {
        'bw_ib2': 23,                 # Binary width 23 → bw_ecl = 2^23-1 = 8388607
        'backward_i_bound': 100,
    },
    # ... rest of config in nested sections
}
```

The width-based parameters (`ib2`, `bw_ib2`) make the config self-documenting: width 23 means "approximate buckets wider than 23 variables in binary domains." No mental arithmetic converting powers of 2.

## When to Use Width Parameters

- ✅ **Use `ib2`/`bw_ib2`** for binary-domain problems (all variables have 2 states)
- ❌ **Use `ecl`/`bw_ecl`** for multi-valued problems (domain size ≥ 3)

## Backward Compatibility

All existing configs using the old flat format with direct `ecl` values continue to work unchanged. The schema auto-detects flat vs nested format and handles both.

## Complete Example Config

This section provides a copy-paste-ready YAML config that works across all 10 hard buckets. The config uses the M005 width-based parameters (`bw_ib2`) and nested format for readability.

```yaml
# NCE Benchmark Config - Complete Example
# Use nested format with 6 sections for clarity

inference:
  device: 'cuda'                    # 'cuda' for GPU, 'cpu' for CPU-only
  i_bound: 100                      # Forward i-bound (limits induced width)
  approximation_method: 'nn'        # Use neural network factors ('nn' or 'dt' for decision trees)
  # Note: ecl (exact computation limit) is omitted - bucket_benchmark.py
  # auto-sets it per bucket using the manifest's auto_ecl values

nn:
  hidden_sizes: [32, 32]            # Two hidden layers with 32 units each
                                    # Empty list [] = linear model
                                    # Larger networks (e.g. [64, 64]) increase capacity but training time

training:
  num_epochs: 500                   # Number of training epochs per bucket
                                    # 500 is typical for hard buckets
                                    # Reduce to 100-200 for quick experiments
  loss_fn: 'logspace_mse_fdb'       # Loss function: log-space MSE with forward diff barrier
                                    # Other options: 'linspace_mse_fdb', 'unnormalized_kl'
  batch_size: 256                   # Batch size - reduce if you hit GPU memory limits
                                    # Smaller = slower but lower memory usage
  learning_rate: 0.001              # Initial learning rate (Adam optimizer default)
  optimizer: 'adam'                 # Optimizer: 'adam' (recommended) or 'sgd'
  skip_early_stopping: true         # Disable loss-based early stopping (train full num_epochs)
  seed: 42                          # Random seed for reproducibility

sampling:
  sampling_scheme: 'all'            # Sample all possible assignments ('all', 'uniform', or 'mg')
                                    # 'all' exhaustively enumerates (works for width ≤ 20)
  num_samples: 100000               # Max samples to generate (ignored when scheme='all')
  set_size: 100000                  # Training set size cap
  val_set: true                     # Create validation set for tracking generalization

backward:
  use_backward_approximation: true  # Enable backward message approximation
  bw_ib2: 23                        # Backward width threshold (binary domain)
                                    # Width 23 = buckets wider than 23 binary variables
                                    # Translates to bw_ecl = 2^23-1 = 8388607
                                    # Increase for tighter backward approx (slower)
                                    # Decrease for faster backward pass (looser approx)
  backward_i_bound: 100             # Backward i-bound (limits backward induced width)

output:
  debug: false                      # Enable verbose debug output (set true for troubleshooting)
  display_intermediate: false       # Print intermediate results during training
  track_errors: false               # Track detailed error metrics (increases overhead)
```

### Quick Start

1. **Copy the config above** to a file named `config.yaml` in your working directory
2. **Modify parameters** as needed (see Customization below)
3. **Run the benchmark:**
   ```bash
   python scripts/bucket_benchmark.py config.yaml fast --gpus 0,1,2,3
   ```
   This runs the fast (per-bucket parallelized) mode using GPUs 0-3.

### Customization

Common modifications for different use cases:

**For quick experiments:**
- Reduce `num_epochs: 100` (faster, less accurate)
- Reduce `hidden_sizes: [16]` (smaller model, faster training)

**For memory-constrained GPUs:**
- Reduce `batch_size: 128` or `batch_size: 64`
- Smaller batch sizes train slower but use less GPU memory

**For tighter backward approximation:**
- Increase `bw_ib2: 25` or `bw_ib2: 27`
- Higher width thresholds = more exact backward messages = slower but more accurate

**For larger model capacity:**
- Increase `hidden_sizes: [64, 64]` or `hidden_sizes: [32, 32, 32]`
- Deeper/wider networks can fit more complex bucket relationships
