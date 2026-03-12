#!/usr/bin/env python
"""
Test script for elp_least_squares loss function.

This script tests the new loss function and compares partition function
errors against unnormalized_kl loss.
"""

import sys
import torch
import numpy as np
from nce.inference import FastGM
from nce.utils.message_gradient import get_message_gradient
from nce.problems import test_problems

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Configuration for testing
gm_config = {
    'device': device,
    'hidden_sizes': [],  # Linear model for fair comparison
    'optimizer': 'adam',
    'num_epochs': 5000,
    'traced_losses': [],
    'val_set': 'all',
    'lr': 1e-3,
    'lr_decay': 1,
    'momentum': 0.9,
    'inverse_time_decay_constant': 10,
    'patience': 1,
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
    'gather_message_stats': False,
    'iB': 100,
    'ecl': 256,
    'iB_backwards': 100,
    'plot_messages': False,
    'approximation_method': 'nn'
}

def test_bucket_with_loss(fastgm, bucket_idx, loss_fn_name):
    """Train a single bucket with specified loss and return partition function error."""
    print(f"\n{'='*60}")
    print(f"Testing bucket {bucket_idx} with loss: {loss_fn_name}")
    print(f"{'='*60}")

    # Set the loss function
    fastgm.config['loss_fn'] = loss_fn_name

    # Get bucket and train
    bucket = fastgm.buckets[bucket_idx]

    # Compute exact message and message gradient for comparison
    mg, m_exact = get_message_gradient(fastgm, bucket_idx)

    # Compute approximate message using neural network
    m_hat = bucket.compute_message_nn()

    # Compute partition function estimates
    # Z_exact = sum over all states of (exact_message * message_gradient)
    Z_exact = (m_exact * mg).sum_all_entries()
    Z_approx = (m_hat.to_exact() * mg).sum_all_entries()

    # Compute error
    Z_error = abs(Z_exact - Z_approx)

    print(f"Z_exact:  {Z_exact:.6f}")
    print(f"Z_approx: {Z_approx:.6f}")
    print(f"Z_error:  {Z_error:.6f}")
    print(f"Relative error: {(Z_error / abs(Z_exact) * 100):.4f}%")

    return {
        'loss_fn': loss_fn_name,
        'Z_exact': float(Z_exact),
        'Z_approx': float(Z_approx),
        'Z_error': float(Z_error),
        'relative_error': float(Z_error / abs(Z_exact))
    }

def main():
    # Get a test problem (using grid problem as it's relatively small)
    probs = [test_problems[k] for k in list(test_problems.keys())]
    probs = [p for p in probs if p.is_10_7_benchmark]

    # Use first grid problem (index 0)
    test_prob = probs[0]
    print(f"\nTesting on problem: {test_prob.name}")
    print(f"Problem width: {test_prob.width}")

    # Loss functions to test
    loss_functions = [
        'elp_least_squares,10',
        'elp_least_squares,50',
        'elp_least_squares,100',
        'unnormalized_kl',
        'mg_sampled_loss_fdb_recompute,10'
    ]

    results = []

    for loss_fn in loss_functions:
        try:
            print(f"\n\n{'#'*70}")
            print(f"# Testing loss function: {loss_fn}")
            print(f"{'#'*70}")

            # Create fresh FastGM for each test
            gm_config['loss_fn'] = loss_fn
            fastgm = FastGM(uai_file=test_prob.uai_file, nn_config=gm_config, device=device)

            # Test on a bucket that needs approximation (find one with width > ecl)
            bucket_idx = None
            for idx, var in enumerate(fastgm.elim_order):
                bucket = fastgm.buckets[var]
                message_scope = bucket.get_message_scope()
                if len(message_scope) >= 3:  # Find a non-trivial bucket
                    bucket_idx = var
                    break

            if bucket_idx is None:
                print("Could not find suitable bucket for testing")
                continue

            # Test the bucket
            result = test_bucket_with_loss(fastgm, bucket_idx, loss_fn)
            results.append(result)

        except Exception as e:
            print(f"Error testing {loss_fn}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary
    print(f"\n\n{'='*70}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*70}")
    print(f"{'Loss Function':<40} {'Z Error':<15} {'Rel Error %':<15}")
    print(f"{'-'*70}")

    for result in results:
        print(f"{result['loss_fn']:<40} {result['Z_error']:<15.6f} {result['relative_error']*100:<15.4f}")

    # Find best loss
    if results:
        best_result = min(results, key=lambda x: x['Z_error'])
        print(f"\nBest performing loss: {best_result['loss_fn']}")
        print(f"Z Error: {best_result['Z_error']:.6f}")

if __name__ == "__main__":
    main()
