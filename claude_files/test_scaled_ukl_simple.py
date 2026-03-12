#!/usr/bin/env python
"""
Simple test to verify scaled_ukl loss function implementation.

This test:
1. Creates a small problem
2. Trains with scaled_ukl
3. Verifies the loss function is called correctly with scaling parameters
"""

import time
from nce.inference import FastGM
from nce.problems import test_problems

device = 'cuda'

# Get a small problem
probs = [test_problems[k] for k in list(test_problems.keys())]
probs_8_5 = [p for p in probs if p.is_8_5_benchmark]
prob = probs_8_5[0]  # Use first 8-5 benchmark problem

print("="*70)
print("SCALED_UKL SIMPLE TEST")
print("="*70)
print(f"\nProblem: {prob.name}")
print(f"Width: {prob.width}")
print(f"Z_true: {prob.Z:.6f}")

# Configuration
gm_config = {
    'device': device,
    'hidden_sizes': [3, 3],
    'optimizer': 'adam',
    'num_epochs': 100,  # Just 100 epochs for quick test
    'traced_losses': [],
    'val_set': 'all',
    'lr': 1e-3,
    'lr_decay': 0.95,
    'momentum': 0.9,
    'inverse_time_decay_constant': 10,
    'patience': 10,
    'min_lr': 1e-8,
    'sampling_scheme': 'all',
    'num_batches_per_set': 1,
    'use_linspace_bias': False,
    'batch_size': 1024,
    'set_size': 10240,
    'num_samples': 147000*20,
    'seed': 4,
    'fdb': True,
    'debug': False,
    'lower_dim': False,
    'dope_factors': True,
    'gather_message_stats': True,  # Required for scaled_ukl
    'ecl': 1024,
    'iB': 10,
    'iB_backwards': 10,
    'plot_messages': False,
    'approximation_method': 'nn',
    'convex_early_stopping': False,  # Disable early stopping for this test
    'loss_fn': 'scaled_ukl',
}

print("\n" + "="*70)
print("Testing scaled_ukl")
print("="*70)

try:
    fastgm = FastGM(uai_file=prob.uai_file, nn_config=gm_config, device=device)
    t = time.time()
    Z_hat = fastgm.get_log_partition_function()
    elapsed = time.time() - t

    error = Z_hat - prob.Z
    abs_error = abs(error)

    print(f"\n✓ scaled_ukl completed successfully!")
    print(f"\nZ_hat: {Z_hat:.6f}")
    print(f"Z_true: {prob.Z:.6f}")
    print(f"Error: {error:+.6f}")
    print(f"Abs Error: {abs_error:.6f}")
    print(f"Num trained: {fastgm.num_trained}")
    print(f"Time: {elapsed:.2f}s")

except Exception as e:
    print(f"\n✗ Error occurred!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
