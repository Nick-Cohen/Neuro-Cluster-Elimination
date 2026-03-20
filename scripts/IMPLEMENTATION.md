# GPU Execution Workflow - Implementation Summary

## What Was Built

A seamless GPU execution system that auto-redirects experiments from the local machine (`circinus-6`) to the GPU server (`deepreasoning`) without manual SSH intervention.

## Components

### 1. GPU Guard Module (`scripts/gpu_guard.py`)

Python module that:
- Detects current hostname
- Auto-redirects to deepreasoning if not already there
- Preserves all command-line arguments
- Works via `os.execvp()` to replace the current process

**Usage in scripts:**
```python
import sys, os
sys.path.insert(0, '/home/cohenn1/NCE')
from scripts.gpu_guard import ensure_gpu_server
ensure_gpu_server()  # Must be before torch/NCE imports
```

### 2. SSH Wrapper Script (`scripts/run_on_gpu.sh`)

Bash script that:
- Syncs code to deepreasoning (excludes .git, venv, cache)
- Executes script remotely with venv activated
- Syncs results back to local machine

**Usage:**
```bash
./scripts/run_on_gpu.sh path/to/script.py [args...]
```

### 3. Test Scripts

- **`test_gpu_guard.py`**: Verifies auto-redirect works
- **`example_gpu_experiment.py`**: Complete working example showing recommended pattern

### 4. Documentation

- **`scripts/README.md`**: Complete guide with examples and troubleshooting
- **`CLAUDE.md`**: Updated with GPU execution requirements and workflow

## How It Works

### Auto-Redirect Flow

1. User/Claude runs: `python3 my_experiment.py`
2. `ensure_gpu_server()` checks hostname
3. If not on deepreasoning:
   - Constructs wrapper command
   - Calls `os.execvp()` to replace current process
   - Wrapper syncs code, runs remotely, syncs results
4. If on deepreasoning:
   - Returns immediately, script continues normally

### Key Design Decisions

1. **Standalone module**: `gpu_guard.py` lives in `scripts/` not `nce/utils/` to avoid triggering torch imports before redirect
2. **Process replacement**: Uses `os.execvp()` instead of subprocess to preserve terminal interaction
3. **Bi-directional sync**: Syncs both before (code) and after (results) execution
4. **Python3 explicit**: Uses `python3` on remote to avoid Python 2.7 default

## Verification

All components tested and working:

```bash
# Test 1: Auto-redirect
$ python3 scripts/test_gpu_guard.py
==> Not on GPU server (current: circinus-6.ics.uci.edu)
==> Auto-redirecting to deepreasoning...
[runs on deepreasoning, shows 4x TITAN RTX GPUs]

# Test 2: Full experiment
$ python3 scripts/example_gpu_experiment.py
[auto-redirects, initializes 2 FastGM models on CUDA, succeeds]

# Test 3: Manual wrapper
$ ./scripts/run_on_gpu.sh scripts/test_gpu_guard.py
[works identically to auto-redirect]
```

## Integration with CLAUDE.md

Updated the experiment execution rules section:

- GPU execution is now **mandatory** for all CUDA experiments
- Three methods documented (auto-guard, manual wrapper, direct SSH)
- Pre-flight checklist includes device verification
- Explicit instruction: never override `device='cuda'` to `device='cpu'`

## For Future Experiments

**Claude/AI should:**
1. Add GPU guard boilerplate to any new experiment script
2. Let the guard handle SSH redirection automatically
3. Never add manual `ssh` commands in experiment runners
4. Never override configs to use CPU without user approval

**Pattern to follow:**
```python
# Always at the top, before any imports
import sys, os
sys.path.insert(0, '/home/cohenn1/NCE')
from scripts.gpu_guard import ensure_gpu_server
ensure_gpu_server()

# Now safe to import
import torch
from nce.inference.graphical_model import FastGM
# ... rest of experiment
```

## What This Solves

✅ No more manual SSH for every experiment  
✅ No more "forgot to SSH" failures  
✅ No more device='cpu' overrides breaking GPU experiments  
✅ Code and results automatically synced  
✅ Works seamlessly from GSD/Claude context  
✅ Explicit error messages when setup is wrong  

## Files Created/Modified

**New files:**
- `scripts/gpu_guard.py`
- `scripts/run_on_gpu.sh`
- `scripts/test_gpu_guard.py`
- `scripts/example_gpu_experiment.py`
- `scripts/README.md`
- `scripts/IMPLEMENTATION.md` (this file)

**Modified files:**
- `CLAUDE.md` (added GPU execution requirements section)
- `prompts/prompt-3-18-26-1.txt` (original requirement - now implemented)
