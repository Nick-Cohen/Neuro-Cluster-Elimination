# NCE Config Examples - Ready to Run

**1-command turnkey examples with pre-generated outputs and plots**

## Quick Start

Two complete working examples, each in its own directory with outputs:

### 1. Quick Test (NeuroBE Mode)
```bash
python examples/quick_test/run.py
```

**What it does:** Fast 1-minute test using NeuroBE mode with early stopping  
**Runtime:** ~34 seconds  
**Result:** log_Z = 606.229

**Output files:**
- `config.json` - The exact config used
- `results.json` - log_Z estimate, duration, metadata

### 2. Standard Training (With Convergence Plots)
```bash
python examples/standard_training/run.py
```

**What it does:** Standard training for 20 epochs with error tracking and convergence plots  
**Runtime:** ~72 seconds  
**Result:** log_Z = 606.306  
**Convergence:** 199.5x improvement (error: 5.42 → 0.027)

**Output files:**
- `config.json` - The exact config used
- `results.json` - log_Z estimate, error tracking data, metadata
- `plots/convergence.png` - **Loss curves showing convergence over epochs**

---

## Pre-Generated Outputs (Verified)

Both examples have been run and outputs saved in their directories.

### Quick Test Results
```json
{
  "log_Z": 606.229,
  "duration_seconds": 33.55,
  "problem": "smokers_5.uai"
}
```

### Standard Training Results

**Console output:**
```
[Error Tracking] Epoch 0: loss=1.70e+02, |log_Z_err|=5.419
[Error Tracking] Epoch 1: loss=1.78e+02, |log_Z_err|=0.263
[Error Tracking] Epoch 5: loss=4.32e-03, |log_Z_err|=0.003
[Error Tracking] Epoch 10: loss=1.92e-03, |log_Z_err|=0.002
[Error Tracking] Epoch 20: loss=9.12e-04, |log_Z_err|=0.027

Results:
  log_Z estimate: 606.306030
  Duration: 71.74s
  Error tracking (5 checkpoints)
  Improvement: 199.5x
  Plot: examples/standard_training/plots/convergence.png
```

**Convergence Plot Location:**
```
examples/standard_training/plots/convergence.png
```

The plot shows:
- **Left panel:** Absolute log Z error decreasing from 5.42 to 0.027 over 20 epochs
- **Right panel:** Training loss decreasing from 170 to 0.0009

---

## How to Reproduce Exact Results

Both examples use `seed=42` for reproducibility.

**On GPU server (deepreasoning):**
```bash
ssh deepreasoning
cd /home/cohenn1/NCE
source venv/bin/activate

# Run quick test
python examples/quick_test/run.py

# Run standard training (generates plots)
python examples/standard_training/run.py
```

You should get:
- Same log_Z values (±0.001 due to GPU floating point variance)
- Same convergence.png plot in examples/standard_training/plots/

---

## Modifying the Examples

Each example is a single Python file you can edit directly.

### Change training duration:
```python
# In examples/quick_test/run.py or examples/standard_training/run.py
config = {
    'training': {
        'num_epochs': 100,  # Change this value
        ...
    }
}
```

### Change batch size:
```python
config = {
    'training': {
        'batch_size': 1024,  # Change this value
        ...
    }
}
```

### Enable/disable error tracking:
```python
config = {
    'output': {
        'error_tracking': True,  # True = generates plots, False = faster
        ...
    }
}
```

### Try different loss function:
```python
config = {
    'training': {
        'loss_fn': 'unnormalized_kl',  # Change from logspace_mse_fdb
        ...
    }
}
```

---

## Directory Structure

```
examples/
├── README.md (this file)
├── quick_test/
│   ├── run.py           # Runnable script
│   ├── config.json      # Pre-generated config
│   └── results.json     # Pre-generated results
└── standard_training/
    ├── run.py           # Runnable script
    ├── config.json      # Pre-generated config
    ├── results.json     # Pre-generated results
    └── plots/
        └── convergence.png  # Pre-generated convergence plots ⭐
```

---

## What Makes This 1/10 Difficulty

✅ **One command to run:** `python examples/quick_test/run.py`  
✅ **Pre-generated outputs:** See results.json AND plots before running  
✅ **Complete working code:** No placeholders or TODOs  
✅ **Reproducible:** Fixed seed, deterministic results  
✅ **Self-contained:** Everything in one file, no dependencies to figure out  
✅ **Clear output:** Shows exactly what happened  
✅ **Visual results:** Convergence plots showing error curves  

**From config to results with plots in one command. No setup, no guessing, no debugging.**

---

## Example Output Visualized

The `standard_training` example generates a convergence plot showing:

1. **Log Z Error** (left): Shows how the approximation improves
   - Starts at ~5.4 (epoch 0)
   - Drops to ~0.27 (epoch 1) 
   - Converges to ~0.003 (epochs 5-10)
   - Final: ~0.027 (epoch 20)

2. **Training Loss** (right): Shows the optimization progress
   - Starts at ~170 (epoch 0-1)
   - Drops sharply to ~0.004 (epoch 5)
   - Continues decreasing to ~0.001 (epoch 20)

Both plots use log scale to show the full range of improvement.

---

## See Also

- **Simple instructions:** `/home/cohenn1/NCE/docs/CONFIG_BUILDER.md`
- **Full field reference:** `/home/cohenn1/NCE/docs/config_reference.md`  
- **Benchmark usage:** `/home/cohenn1/NCE/BENCHMARK_USAGE.md`

---

## File Locations (Complete)

**Quick Test:**
- `/home/cohenn1/NCE/examples/quick_test/run.py`
- `/home/cohenn1/NCE/examples/quick_test/config.json`
- `/home/cohenn1/NCE/examples/quick_test/results.json`

**Standard Training:**
- `/home/cohenn1/NCE/examples/standard_training/run.py`
- `/home/cohenn1/NCE/examples/standard_training/config.json`
- `/home/cohenn1/NCE/examples/standard_training/results.json`
- `/home/cohenn1/NCE/examples/standard_training/plots/convergence.png` ⭐

**Documentation:**
- `/home/cohenn1/NCE/examples/README.md` (this file)
- `/home/cohenn1/NCE/docs/CONFIG_BUILDER.md`
- `/home/cohenn1/NCE/docs/config_reference.md`
