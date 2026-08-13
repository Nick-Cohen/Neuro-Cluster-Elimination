# Single-Bucket Benchmark - BLOCKED BY SYSTEM LIBRARY ISSUE

## Problem

All Python environments on this machine have a `libffi.so.7` dependency issue:
- System has `libffi.so.8.1.0` at `/usr/lib/x86_64-linux-gnu/`
- Python's ctypes module requires `libffi.so.7` (older version)
- This blocks all torch imports and therefore all benchmark execution

## Solution Required

Create a symlink (requires sudo):
```bash
cd /usr/lib/x86_64-linux-gnu
sudo ln -s libffi.so.8.1.0 libffi.so.7
```

After this is done, the benchmark will run with:

**Config file**: `/home/cohenn1/NCE/benchmark_config.yaml`
**Command**: `python /home/cohenn1/NCE/train_buckets_simple.py --device cuda --fast`

The config uses:
- Loss function: `unnormalized_kl`
- Epochs: 10000
- Hidden layers: [64, 64]
- Time limit: 60 seconds per bucket (fast mode)
- Device: CUDA

## What the benchmark will do

1. Train all 10 hard buckets sequentially
2. Generate plots (loss.png, local_error.png) for each bucket
3. Save metrics.json with training results
4. Output directory: `/home/cohenn1/NCE/data/hard_buckets/benchmark_results/`

## Files created

- `/home/cohenn1/NCE/benchmark_config.yaml` - Training configuration
- `/home/cohenn1/NCE/train_buckets_simple.py` - Simple sequential training script (no multi-GPU complexity)
- `/home/cohenn1/NCE/data/hard_buckets/bucket_list.json` - Updated with actual 10 hard buckets

## Alternative: Run on a different machine

If you have access to another machine where torch works, copy these files and run there.
