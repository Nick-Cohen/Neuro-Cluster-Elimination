#!/usr/bin/env python3
"""
Example: How to write GPU experiments that auto-redirect to deepreasoning

This demonstrates the recommended pattern for NCE experiments.
"""

# Step 1: Add GPU guard at the very top (before any heavy imports)
import sys
import os
sys.path.insert(0, '/home/cohenn1/NCE')
from scripts.gpu_guard import ensure_gpu_server

# This will auto-redirect to deepreasoning if running locally
ensure_gpu_server()

# Step 2: Now safe to import torch and NCE modules
# (these will only load on the GPU server after redirect)
import torch
from nce.inference.graphical_model import FastGM
from nce.benchmark_problems import nbe_sanity_check

print("=" * 70)
print("Example GPU Experiment")
print("=" * 70)

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nDevice: {device}")

if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Example configuration
nn_config = {
    'iB': 4,
    'ecl': 10,
    'loss_fn': 'logspace_mse_fdb',
    'sampling_scheme': 'uniform',
    'num_epochs': 5,  # Small for demo
    'batch_size': 100,
    'lr': 0.001,
    'hidden_sizes': [64, 32],
    'device': device,
}

print(f"\nConfiguration:")
for k, v in nn_config.items():
    print(f"  {k}: {v}")

# Run a tiny experiment
print(f"\nRunning mini experiment on {len(nbe_sanity_check.problems)} problems...")

for i, (model, config) in enumerate(zip(nbe_sanity_check.problems[:2], 
                                         nbe_sanity_check.configs['nbe'][:2])):
    print(f"\n  Problem {i+1}/{2}:")
    
    # Merge with our test config
    test_config = {**config, **nn_config}
    test_config['num_epochs'] = 2  # Very quick test
    
    print(f"    Variables: {model.num_vars}, Factors: {model.num_factors}")
    
    # Initialize but don't run full inference (just for testing the setup)
    fastgm = FastGM(model=model, nn_config=test_config, device=test_config['device'])
    print(f"    Buckets created: {len(fastgm.buckets)}")
    print(f"    ✓ Initialization successful")

print("\n" + "=" * 70)
print("✓ Example complete - GPU guard and experiment setup working correctly")
print("=" * 70)
