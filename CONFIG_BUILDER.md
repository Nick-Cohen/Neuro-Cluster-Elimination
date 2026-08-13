# NCE Config Builder - Simple Guide

**Making config creation easy: from 7/10 difficulty → 1/10 difficulty**

This guide shows you how to build configs for common use cases with copy-paste templates. No need to understand every field — just start with a template and modify what you need.

---

## Quick Start: 3 Steps

1. **Copy a template** from the Examples section below
2. **Change 1-3 values** for your experiment (usually just: epochs, batch_size, loss_fn)
3. **Run it**: `python -c "from nce.graphical_model import FastGM; ..." # see Running Configs below`

That's it. You don't need to understand all 50+ config fields. The templates handle the details.

---

## What You Actually Need to Know

### Required Fields (only 3!)

| Field | What it does | Typical values |
|---|---|---|
| `num_epochs` | How long to train | `100` (quick), `500` (normal), `5000` (thorough) |
| `loss_fn` | What to optimize | `'logspace_mse_fdb'` (standard), `'unnormalized_kl'` (alternative) |
| `num_samples` | Training data size | `100000` (typical), `'all'` (exhaustive) |

### Fields You'll Change Often (optional)

| Field | What it does | When to change |
|---|---|---|
| `batch_size` | Training batch size | Default 256. Use 2048+ for fast training. Use 64 if GPU memory limited. |
| `skip_early_stopping` | Train full num_epochs | Set `true` for controlled experiments |
| `device` | CPU or GPU | `'cuda'` (default), `'cpu'` (for debugging) |
| `hidden_sizes` | Network architecture | `[32, 32]` (default), `[]` (linear model), `[64, 64]` (larger capacity) |
| `bw_ib2` | Backward approximation width | Set to control backward message approximation (see templates) |

### Everything Else

You can ignore it. The templates set sensible defaults. If you need something special, see [docs/config_reference.md](config_reference.md) for the full field list.

---

## Templates for Common Use Cases

### Template 1: Quick 1-Minute Test (NeuroBE Mode)

**Use this when**: You want to test if everything works, fast feedback loop.

```python
config = {
    'inference': {
        'device': 'cuda',
        'i_bound': 100,
        'approximation_method': 'nn',
        'neurobe_mode': True,  # NeuroBE-faithful training (ReLU, min-max norm, patience stopping)
    },
    'nn': {
        'hidden_sizes': [32, 32],
    },
    'training': {
        'num_epochs': 50,  # Short for 1-minute test
        'loss_fn': 'logspace_mse_fdb',
        'batch_size': 2048,  # Large batch = fast training
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'skip_early_stopping': True,  # No early stopping - train all 50 epochs
    },
    'sampling': {
        'sampling_scheme': 'all',
        'num_samples': 100000,
    },
    'output': {
        'debug': False,
    },
}
```

**What to change**:
- `num_epochs`: Increase to 100-200 for more thorough testing
- `batch_size`: Decrease to 1024 if GPU memory limited
- `device`: Change to `'cpu'` if no GPU available

---

### Template 2: Standard Training with Backward Approximation

**Use this when**: You want controlled training with backward message approximation for loss functions that need it.

```python
config = {
    'inference': {
        'device': 'cuda',
        'ib2': 19,  # Binary width 19 (ecl = 2^19-1 = 524287)
        'approximation_method': 'nn',
    },
    'nn': {
        'hidden_sizes': [32, 32],
    },
    'training': {
        'num_epochs': 500,
        'loss_fn': 'unnormalized_kl',  # Requires backward info
        'batch_size': 256,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'skip_early_stopping': True,
    },
    'sampling': {
        'sampling_scheme': 'all',
        'num_samples': 100000,
    },
    'backward': {
        'use_backward_approximation': True,
        'bw_ib2': 10,  # Binary width 10 (bw_ecl = 2^10-1 = 1023)
        'backward_i_bound': 100,
    },
    'output': {
        'debug': False,
    },
}
```

**What to change**:
- `bw_ib2`: Controls backward approximation tightness. Higher = more accurate, slower. `10` (loose), `15` (medium), `20` (tight).
- `loss_fn`: Try `'logspace_mse_fdb'` (standard), `'weighted_logspace_mse'` (NeuroBE-style), `'unnormalized_kl'` (alternative)
- `num_epochs`: `500` is typical, use `100` for quick experiments or `5000` for thorough training

---

### Template 3: Long Training Run (Benchmark Mode)

**Use this when**: Running serious benchmarks, want full control, no shortcuts.

```python
config = {
    'inference': {
        'device': 'cuda',
        'ib2': 19,
        'approximation_method': 'nn',
    },
    'nn': {
        'hidden_sizes': [32, 32],
    },
    'training': {
        'num_epochs': 5000,
        'loss_fn': 'logspace_mse_fdb',
        'batch_size': 256,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'skip_early_stopping': True,  # Train all 5000 epochs no matter what
    },
    'sampling': {
        'sampling_scheme': 'all',
        'num_samples': 100000,
    },
    'backward': {
        'use_backward_approximation': True,
        'bw_ib2': 23,  # Tight backward approximation
        'backward_i_bound': 100,
    },
    'output': {
        'debug': False,
        'error_tracking': True,  # Track convergence over time
    },
}
```

**What to change**:
- `num_epochs`: This is the main knob. 5000 is for serious benchmarks.
- `bw_ib2`: Higher = more backward info = potentially better convergence. 23 is tight.

---

## Running Your Config

### Option 1: Direct Python Script

```python
from nce.graphical_model import FastGM
from pyGMs import Model

# Load a problem
model = Model.load_uai('path/to/problem.uai')

# Your config (from template above)
config = {
    'inference': {'device': 'cuda', 'ib2': 19, 'approximation_method': 'nn'},
    'nn': {'hidden_sizes': [32, 32]},
    'training': {'num_epochs': 500, 'loss_fn': 'logspace_mse_fdb', 'batch_size': 256,
                 'learning_rate': 0.001, 'optimizer': 'adam', 'skip_early_stopping': True},
    'sampling': {'sampling_scheme': 'all', 'num_samples': 100000},
}

# Run inference
fastgm = FastGM(model=model, nn_config=config, device='cuda')
log_Z = fastgm.infer()
print(f"Log partition function: {log_Z}")
```

### Option 2: Benchmark Problems (Small Test Problems)

```python
from nce.benchmark_problems.small_problems import small_problems

# Pick a problem
model = small_problems.problems[0]  # smokers_5.uai

# Your config
config = { ... }  # from template

# Run
from nce.graphical_model import FastGM
fastgm = FastGM(model=model, nn_config=config, device='cuda')
log_Z = fastgm.infer()
```

### Option 3: Multi-Bucket Benchmark Script

For running experiments across many problems/buckets, use the benchmark harness:

```bash
# Create config YAML file
cat > my_config.yaml << 'EOF'
inference:
  device: cuda
  ib2: 19
  approximation_method: nn
nn:
  hidden_sizes: [32, 32]
training:
  num_epochs: 500
  loss_fn: logspace_mse_fdb
  batch_size: 256
  learning_rate: 0.001
  optimizer: adam
  skip_early_stopping: true
sampling:
  sampling_scheme: all
  num_samples: 100000
backward:
  use_backward_approximation: true
  bw_ib2: 10
  backward_i_bound: 100
EOF

# Run benchmark
python scripts/bucket_benchmark.py my_config.yaml fast --gpus 0,1,2,3
```

---

## Common Modifications Cheat Sheet

### "I want faster training"

```python
'training': {
    'num_epochs': 100,        # ← Reduce this
    'batch_size': 4096,       # ← Increase this
    'skip_early_stopping': True,
}
```

### "I want more accurate backward approximation"

```python
'backward': {
    'use_backward_approximation': True,
    'bw_ib2': 23,  # ← Increase this (10→15→20→23)
    'backward_i_bound': 100,
}
```

### "I want to try a different loss function"

```python
'training': {
    'loss_fn': 'unnormalized_kl',  # ← Change this
    # Options: 'logspace_mse_fdb', 'unnormalized_kl', 
    #          'weighted_logspace_mse', 'linspace_mse_fdb'
}
```

### "I'm running out of GPU memory"

```python
'training': {
    'batch_size': 64,  # ← Reduce this (256→128→64)
}
'nn': {
    'hidden_sizes': [16, 16],  # ← Smaller network
}
```

### "I want linear model (no hidden layers)"

```python
'nn': {
    'hidden_sizes': [],  # ← Empty list = linear model
}
```

---

## What Each Section Does (High-Level)

- **inference**: How the graphical model elimination works (width limits, device)
- **nn**: Neural network architecture
- **training**: Optimization settings (epochs, loss, batch size, optimizer)
- **sampling**: How training data is generated
- **backward**: Backward message approximation (for certain loss functions)
- **output**: Debugging and diagnostics

You don't need to understand the internals. Just use the templates and modify the fields in the "What to change" sections.

---

## Troubleshooting

### "I don't know which loss function to use"

Start with `'logspace_mse_fdb'`. It works well in most cases. If you want to experiment, try `'unnormalized_kl'` (requires backward approximation).

### "I don't know what bw_ib2 value to use"

For quick experiments: `bw_ib2: 10` (loose, fast)  
For typical use: `bw_ib2: 15` (balanced)  
For tight approximation: `bw_ib2: 20` or `23` (slow, accurate)

### "What's the difference between ib2 and bw_ib2?"

- `ib2`: Forward pass width limit (how wide buckets can be before using NN approximation)
- `bw_ib2`: Backward pass width limit (for loss functions that need backward messages)

Start with `ib2: 19` and `bw_ib2: 10-15`. Adjust if needed.

### "Do I need to set all these fields?"

No! Only 3 are required: `num_epochs`, `loss_fn`, `num_samples`. Everything else has defaults. The templates show good starting values.

---

## See Also

- Full field reference: [config_reference.md](config_reference.md)
- Benchmark usage examples: [BENCHMARK_USAGE.md](../BENCHMARK_USAGE.md)
- Training harness: [scripts/bucket_benchmark.py](../scripts/bucket_benchmark.py)
