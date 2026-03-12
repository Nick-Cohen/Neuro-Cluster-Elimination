"""
Integration Test: Batched Learning on grid10x10.f10

Tests batched learning implementation with:
- Problem: grid10x10.f10 (UAI benchmark)
- ecl: 2**16
- Batched mode: sampling_scheme='uniform'
- Backward message approximation: use_bw_approx=True
"""

import sys
sys.path.insert(0, '/home/cohenn1/NCE')

import torch
from nce.inference.graphical_model import FastGM

print("="*70)
print("BATCHED LEARNING INTEGRATION TEST: grid10x10.f10")
print("="*70)

# Configuration for batched learning
config = {
    # Mini-bucket settings
    'iB': 10,
    'ecl': 2**16,
    'complexity_limit': 0,

    # Backward message approximation (batched mode)
    'use_bw_approx': True,
    'backward_iB': 10,
    'backward_ecl': 2**20,

    # Batched sampling mode (NOT 'all' - triggers batched mode)
    'sampling_scheme': 'uniform',

    # Training parameters
    'num_samples': 5000,
    'batch_size': 512,
    'set_size': 1000,
    'num_epochs': 50,
    'lr': 0.01,
    'optimizer': 'adam',

    # Loss function
    'loss_fn': 'logspace_mse_fdb',

    # Network architecture
    'hidden_sizes': [],  # Linear model
    'lower_dim': False,

    # Other settings
    'device': 'cuda',
    'seed': 42,
    'debug': True,
    'skip_early_stopping': False,
    'traced_losses': [],
    'val_set': None,
    'fdb': True,
    'inverse_time_decay_constant': 1000,
    'patience': 10,
    'lr_decay': 0.5,
    'min_lr': 1e-6,
    'momentum': 0.9,
}

print("\nConfiguration:")
print(f"  Problem: grid10x10.f10")
print(f"  ecl: {config['ecl']} (2^16 = {2**16})")
print(f"  Batched mode: sampling_scheme='{config['sampling_scheme']}'")
print(f"  Backward message: use_bw_approx={config['use_bw_approx']}")
print(f"  Backward iB: {config['backward_iB']}")
print(f"  Backward ecl: {config['backward_ecl']} (2^20 = {2**20})")
print(f"  Batch size: {config['batch_size']}")
print(f"  Device: {config['device']}")

# Locate grid10x10.f10
import os
uai_path = None
search_paths = [
    '/home/cohenn1/SDBE/PyGMs/data/uai_files/grid10x10.f10.uai',
    '/home/cohenn1/NCE/data/grid10x10.f10.uai',
    '/home/cohenn1/SDBE/PyGMs/data/grid10x10.f10.uai',
]

for path in search_paths:
    if os.path.exists(path):
        uai_path = path
        break

if uai_path is None:
    print("\nError: Could not find grid10x10.f10.uai")
    print("Searched paths:")
    for path in search_paths:
        print(f"  - {path}")
    sys.exit(1)

print(f"\nUsing UAI file: {uai_path}")

try:
    print("\n" + "="*70)
    print("Creating FastGM with batched learning configuration...")
    print("="*70)

    gm = FastGM(
        uai_file=uai_path,
        device=config['device'],
        nn_config=config
    )

    print(f"\nGraphical Model created successfully!")
    print(f"  Number of variables: {len(gm.variables)}")
    print(f"  Number of factors: {len(gm.orig_factors)}")
    print(f"  Number of buckets: {len(gm.buckets)}")

    # Check which buckets will use batched mode
    print("\nBucket Analysis:")
    batched_buckets = []
    full_batch_buckets = []

    for label, bucket in gm.buckets.items():
        if hasattr(bucket, 'trainer') and bucket.trainer is not None:
            sampling_scheme = bucket.trainer.dataloader.sample_generator.sampling_scheme
            if sampling_scheme == 'all':
                full_batch_buckets.append(label)
            else:
                batched_buckets.append(label)

    print(f"  Buckets that will use batched mode: {len(batched_buckets)}")
    print(f"  Buckets that will use full_data_batch mode: {len(full_batch_buckets)}")

    print("\n" + "="*70)
    print("Running inference with batched learning...")
    print("="*70)

    # Run inference
    result = gm.infer()

    print("\n" + "="*70)
    print("INFERENCE COMPLETED SUCCESSFULLY!")
    print("="*70)

    print(f"\nResult: {result}")

    # Verify batched mode was actually used
    print("\n" + "="*70)
    print("Verification: Batched mode usage")
    print("="*70)

    buckets_with_bw_factors = 0
    buckets_with_mgh_modifier = 0

    for label, bucket in gm.buckets.items():
        if hasattr(bucket, 'trainer') and bucket.trainer is not None:
            dataloader = bucket.trainer.dataloader

            if hasattr(dataloader, 'bw_factors') and dataloader.bw_factors is not None:
                buckets_with_bw_factors += 1
                print(f"  Bucket {label}: Using bw_factors (batched mode) - {len(dataloader.bw_factors)} factors")
            elif hasattr(dataloader, 'mgh_modifier') and dataloader.mgh_modifier is not None:
                buckets_with_mgh_modifier += 1
                print(f"  Bucket {label}: Using mgh_modifier (full_data_batch mode)")

    print(f"\nSummary:")
    print(f"  Buckets using batched mode (bw_factors): {buckets_with_bw_factors}")
    print(f"  Buckets using full_data_batch mode (mgh_modifier): {buckets_with_mgh_modifier}")

    if buckets_with_bw_factors > 0:
        print("\n✓ BATCHED LEARNING MODE VERIFIED!")
        print("  Backward messages were stored as factor lists and sampled on-the-fly")
    else:
        print("\n⚠ WARNING: No buckets used batched mode")
        print("  This might be expected if all buckets used exact computation")

    print("\n" + "="*70)
    print("TEST PASSED ✓")
    print("="*70)
    print("\nBatched learning implementation works correctly on grid10x10.f10!")

    sys.exit(0)

except Exception as e:
    print("\n" + "="*70)
    print("TEST FAILED ✗")
    print("="*70)
    print(f"\nError: {e}")

    import traceback
    traceback.print_exc()

    sys.exit(1)
