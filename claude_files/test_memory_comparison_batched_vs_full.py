"""
Memory Comparison: Batched vs Full Data Batch Mode

This test compares memory usage between:
1. Batched mode (sampling_scheme='uniform')
2. Full data batch mode (sampling_scheme='all')

Uses small problem with 2 epochs and few batches to emphasize the difference.
"""

import sys
sys.path.insert(0, '/home/cohenn1/NCE')

import torch
import gc
from nce.inference.graphical_model import FastGM

def get_memory_stats():
    """Get current GPU memory statistics."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
        reserved = torch.cuda.memory_reserved() / (1024**2)    # MB
        return allocated, reserved
    return 0, 0

def print_memory(label):
    """Print memory usage with label."""
    allocated, reserved = get_memory_stats()
    print(f"  {label:40s} | Allocated: {allocated:8.2f} MB | Reserved: {reserved:8.2f} MB")

def clear_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def create_test_problem():
    """Create a simple graphical model for testing."""
    from nce.inference.factor import FastFactor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nUsing device: {device}")

    # Create 5 variables, each with domain size 3
    # This creates a small but non-trivial problem
    factors = []

    # Factor 1: f(0, 1)
    tensor1 = torch.rand(3, 3, device=device)
    factors.append(FastFactor(torch.log10(tensor1), [0, 1]))

    # Factor 2: f(1, 2)
    tensor2 = torch.rand(3, 3, device=device)
    factors.append(FastFactor(torch.log10(tensor2), [1, 2]))

    # Factor 3: f(2, 3)
    tensor3 = torch.rand(3, 3, device=device)
    factors.append(FastFactor(torch.log10(tensor3), [2, 3]))

    # Factor 4: f(3, 4)
    tensor4 = torch.rand(3, 3, device=device)
    factors.append(FastFactor(torch.log10(tensor4), [3, 4]))

    # Factor 5: f(4)
    tensor5 = torch.rand(3, device=device)
    factors.append(FastFactor(torch.log10(tensor5), [4]))

    elim_order = [0, 1, 2, 3, 4]

    return factors, elim_order, device

def run_batched_mode():
    """Run with batched mode (sampling_scheme='uniform')."""
    print("\n" + "="*80)
    print("BATCHED MODE (sampling_scheme='uniform')")
    print("="*80)

    factors, elim_order, device = create_test_problem()

    config = {
        'iB': 10,
        'ecl': 2**20,  # High enough to force NN learning
        'complexity_limit': 0,

        # Batched mode settings
        'use_bw_approx': True,
        'backward_iB': 10,
        'backward_ecl': 2**20,
        'sampling_scheme': 'uniform',  # BATCHED MODE

        # Training parameters (small for quick test)
        'num_samples': 200,
        'batch_size': 50,
        'set_size': 100,
        'num_epochs': 2,  # Just 2 epochs
        'lr': 0.01,
        'optimizer': 'adam',

        'loss_fn': 'logspace_mse_fdb',
        'hidden_sizes': [],
        'lower_dim': False,
        'device': device,
        'seed': 42,
        'debug': False,
        'skip_early_stopping': True,  # Don't stop early
        'traced_losses': [],
        'val_set': None,
        'fdb': True,
    }

    clear_memory()
    print_memory("Before creating FastGM")

    gm = FastGM(factors=factors, elim_order=elim_order, device=device, nn_config=config)

    print_memory("After creating FastGM")

    # Check which mode is being used
    for label, bucket in gm.buckets.items():
        if hasattr(bucket, 'trainer') and bucket.trainer is not None:
            dataloader = bucket.trainer.dataloader
            if hasattr(dataloader, 'bw_factors') and dataloader.bw_factors is not None:
                print(f"\n✓ Bucket {label}: Using batched mode (bw_factors with {len(dataloader.bw_factors)} factors)")
            elif hasattr(dataloader, 'mgh_modifier') and dataloader.mgh_modifier is not None:
                print(f"\n✗ Bucket {label}: Using full_data_batch mode (mgh_modifier)")

    print_memory("Before inference")

    # Run inference
    result = gm.infer()

    print_memory("After inference")

    # Get peak memory
    allocated, reserved = get_memory_stats()

    print(f"\n{'='*80}")
    print(f"BATCHED MODE PEAK MEMORY:")
    print(f"  Allocated: {allocated:.2f} MB")
    print(f"  Reserved:  {reserved:.2f} MB")
    print(f"{'='*80}")

    return allocated, reserved

def run_full_data_batch_mode():
    """Run with full data batch mode (sampling_scheme='all')."""
    print("\n" + "="*80)
    print("FULL DATA BATCH MODE (sampling_scheme='all')")
    print("="*80)

    factors, elim_order, device = create_test_problem()

    config = {
        'iB': 10,
        'ecl': 2**20,
        'complexity_limit': 0,

        # Full data batch mode settings
        'use_bw_approx': True,
        'backward_iB': 10,
        'backward_ecl': 2**20,
        'sampling_scheme': 'all',  # FULL DATA BATCH MODE

        # Training parameters
        'num_batches_per_set': 2,  # Just 2 batches per set for quick test
        'num_epochs': 2,  # Just 2 epochs
        'lr': 0.01,
        'optimizer': 'adam',

        'loss_fn': 'logspace_mse_fdb',
        'hidden_sizes': [],
        'lower_dim': False,
        'device': device,
        'seed': 42,
        'debug': False,
        'skip_early_stopping': True,
        'traced_losses': [],
        'val_set': None,
        'fdb': True,
    }

    clear_memory()
    print_memory("Before creating FastGM")

    gm = FastGM(factors=factors, elim_order=elim_order, device=device, nn_config=config)

    print_memory("After creating FastGM")

    # Check which mode is being used
    for label, bucket in gm.buckets.items():
        if hasattr(bucket, 'trainer') and bucket.trainer is not None:
            dataloader = bucket.trainer.dataloader
            if hasattr(dataloader, 'bw_factors') and dataloader.bw_factors is not None:
                print(f"\n✗ Bucket {label}: Using batched mode (bw_factors)")
            elif hasattr(dataloader, 'mgh_modifier') and dataloader.mgh_modifier is not None:
                print(f"\n✓ Bucket {label}: Using full_data_batch mode (mgh_modifier)")

    print_memory("Before inference")

    # Run inference
    result = gm.infer()

    print_memory("After inference")

    # Get peak memory
    allocated, reserved = get_memory_stats()

    print(f"\n{'='*80}")
    print(f"FULL DATA BATCH MODE PEAK MEMORY:")
    print(f"  Allocated: {allocated:.2f} MB")
    print(f"  Reserved:  {reserved:.2f} MB")
    print(f"{'='*80}")

    return allocated, reserved

def main():
    print("\n" + "#"*80)
    print("# MEMORY COMPARISON TEST: Batched vs Full Data Batch Mode")
    print("#"*80)

    if not torch.cuda.is_available():
        print("\n⚠ WARNING: CUDA not available. Memory comparison may not be meaningful.")
        print("Running on CPU...\n")

    try:
        # Run batched mode
        batched_alloc, batched_reserved = run_batched_mode()

        # Clear memory between runs
        clear_memory()
        print("\n" + "-"*80)
        print("Clearing memory between runs...")
        print("-"*80)

        # Run full data batch mode
        full_alloc, full_reserved = run_full_data_batch_mode()

        # Compare results
        print("\n" + "="*80)
        print("MEMORY COMPARISON SUMMARY")
        print("="*80)

        print(f"\nBatched Mode:")
        print(f"  Allocated: {batched_alloc:8.2f} MB")
        print(f"  Reserved:  {batched_reserved:8.2f} MB")

        print(f"\nFull Data Batch Mode:")
        print(f"  Allocated: {full_alloc:8.2f} MB")
        print(f"  Reserved:  {full_reserved:8.2f} MB")

        print(f"\nMemory Savings (Batched vs Full):")
        if full_alloc > 0:
            savings_alloc = ((full_alloc - batched_alloc) / full_alloc) * 100
            print(f"  Allocated: {savings_alloc:6.2f}% reduction")
        if full_reserved > 0:
            savings_reserved = ((full_reserved - batched_reserved) / full_reserved) * 100
            print(f"  Reserved:  {savings_reserved:6.2f}% reduction")

        print("\n" + "="*80)

        if batched_alloc < full_alloc:
            print("✓ BATCHED MODE USES LESS MEMORY!")
            print(f"  Reduction: {full_alloc - batched_alloc:.2f} MB")
        elif batched_alloc > full_alloc:
            print("⚠ WARNING: Batched mode used MORE memory than expected")
            print(f"  Increase: {batched_alloc - full_alloc:.2f} MB")
        else:
            print("⚠ Memory usage is identical (problem may be too small)")

        print("="*80)

        print("\n✓ TEST COMPLETED SUCCESSFULLY")
        return True

    except Exception as e:
        print("\n" + "="*80)
        print("TEST FAILED")
        print("="*80)
        print(f"\nError: {e}")

        import traceback
        traceback.print_exc()

        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
