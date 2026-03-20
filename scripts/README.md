# NCE GPU Execution Scripts

This directory contains utilities for seamless GPU experiment execution on the `deepreasoning` server.

## The Problem

- GSD 2.x runs locally on `circinus-6`
- GPU resources are on `deepreasoning` server
- All CUDA experiments must execute remotely

## Solutions

### Option 1: GPU Guard (Recommended)

**Best for:** Python scripts you're developing/iterating on

Add this boilerplate to the top of any GPU-requiring script:

```python
import sys
import os
sys.path.insert(0, '/home/cohenn1/NCE')
from scripts.gpu_guard import ensure_gpu_server
ensure_gpu_server()  # Auto-redirects if not on GPU server

# Now safe to import torch, NCE, etc.
import torch
from nce.inference.graphical_model import FastGM
```

**How it works:**
1. Detects if running on `deepreasoning` 
2. If not, re-executes the script via SSH using `run_on_gpu.sh`
3. Syncs code before and results after execution

**See:** `scripts/example_gpu_experiment.py` for a complete working example

### Option 2: Manual Wrapper

**Best for:** One-off runs, debugging, explicit control

```bash
./scripts/run_on_gpu.sh path/to/experiment.py [args...]
```

This wrapper:
- Syncs code to deepreasoning (excluding .git, venv, cache)
- Executes remotely with venv activated
- Syncs results back (experiment outputs, pickles, etc.)

### Option 3: Direct SSH

**Best for:** Interactive debugging, monitoring GPU usage

```bash
ssh deepreasoning
cd /home/cohenn1/NCE
source venv/bin/activate
python3 your_script.py
```

Use `nvidia-smi` to check GPU status/memory.

## Files

- **`gpu_guard.py`**: Auto-redirect module for Python scripts
- **`run_on_gpu.sh`**: Bash wrapper for manual invocation
- **`test_gpu_guard.py`**: Test script verifying auto-redirect works
- **`example_gpu_experiment.py`**: Complete example showing recommended pattern

## Testing

```bash
# Test auto-redirect (should run on deepreasoning automatically)
python3 scripts/test_gpu_guard.py

# Test example experiment
python3 scripts/example_gpu_experiment.py

# Test manual wrapper
./scripts/run_on_gpu.sh scripts/test_gpu_guard.py
```

## Pre-Flight Checklist

Before running ANY GPU experiment:

1. ✅ Config specifies `device='cuda'`
2. ✅ Script uses GPU guard OR manual wrapper
3. ✅ Never override to `device='cpu'` without user approval
4. ✅ Verify you're not already on deepreasoning (check `hostname`)

## Troubleshooting

**Error: "libffi.so.7: cannot open shared object file"**
- This means torch is trying to load locally on circinus-6
- Add GPU guard BEFORE importing torch/NCE

**Error: "Wrapper script not found"**
- Ensure you're running from project root: `cd /home/cohenn1/NCE`
- Check wrapper is executable: `chmod +x scripts/run_on_gpu.sh`

**Script runs locally but should run on GPU**
- Add `ensure_gpu_server()` at the top
- Or use the manual wrapper: `./scripts/run_on_gpu.sh your_script.py`

**SSH hangs or times out**
- Check SSH works: `ssh deepreasoning hostname`
- Verify server is up and accessible

## For Claude/GSD

When instructed to run a GPU experiment:

1. Check if script already has GPU guard
2. If not, add the boilerplate at the top
3. Execute with `python3 path/to/script.py` (guard handles redirect)
4. Monitor output for "Auto-redirecting to deepreasoning" confirmation

Never add timeouts to GPU experiments without explicit user approval.
