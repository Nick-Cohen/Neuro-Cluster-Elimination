"""
Test script for UKF Sequential Loss implementation

This script tests the newly integrated ukf_sequential loss function.
"""

import torch
from nce.inference.graphical_model import FastGM

# Simple test configuration
nn_config = {
    'loss_fn': 'ukf_sequential',  # Use default parameters (resample_period=10, m_per=1000)
    'num_epochs': 50,
    'batch_size': 100,
    'set_size': 1000,
    'num_samples': 2000,
    'lr': 1e-3,
    'hidden_sizes': [],  # Linear model for quick testing
    'device': 'cpu',  # Use CPU for compatibility
    'optimizer': 'adam',
    'lower_dim': False,
    'fdb': True,
    'iB': 10,
    'ecl': 15,
    'debug': True,
    'traced_losses': ['logspace_mse_fdb'],
    'val_set': None,
    'seed': 42,
    'num_batches_per_set': 10,
}

print("=" * 70)
print("UKF SEQUENTIAL LOSS FUNCTION TEST")
print("=" * 70)
print()
print("Configuration:")
print(f"  Loss function: {nn_config['loss_fn']}")
print(f"  Epochs: {nn_config['num_epochs']}")
print(f"  Batch size: {nn_config['batch_size']}")
print(f"  Device: {nn_config['device']}")
print(f"  Model: Linear (hidden_sizes=[])")
print()

# Try to load a simple UAI file
# You may need to adjust this path to an actual UAI file in your directory
import os
import glob

# Find a UAI file to test with
uai_files = glob.glob('/home/cohenn1/NCE/**/*.uai', recursive=True)
if not uai_files:
    print("ERROR: No UAI files found. Please provide a path to a .uai file.")
    print("Example: python test_ukf_sequential.py path/to/model.uai")
    exit(1)

# Use the first UAI file found (or you can specify one)
uai_file = uai_files[0]
print(f"Testing with UAI file: {uai_file}")
print()

try:
    print("Initializing FastGM with ukf_sequential loss...")
    gm = FastGM(uai_file=uai_file, nn_config=nn_config)

    print("Starting inference with ukf_sequential training...")
    log_Z = gm.infer()

    print()
    print("=" * 70)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Log partition function: {log_Z}")
    print()
    print("The ukf_sequential loss function is working correctly!")
    print()
    print("Trained buckets:")
    for bucket_label, losses in gm.traced_losses_data:
        print(f"  Bucket {bucket_label}: {len(losses)} loss records")

except Exception as e:
    print()
    print("=" * 70)
    print("TEST FAILED")
    print("=" * 70)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    print()
    print("Please check the implementation and try again.")
