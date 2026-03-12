#!/usr/bin/env python
"""
Benchmark test for scaled_mse and unnormalized_kl on 8-5 problems.

Tests:
- Loss functions: scaled_mse, unnormalized_kl
- Hidden sizes: [3, 3]
- No pretraining
- 100k epochs max with early stopping
- Complexity bound (ecl): 1024
- Validates partition function results are reasonable
"""

import time
import pickle
from datetime import datetime
from nce.inference import FastGM
from nce.problems import test_problems

device = 'cuda'

# Filter for 8-5 benchmark problems
probs = [test_problems[k] for k in list(test_problems.keys())]
probs_8_5 = [p for p in probs if p.is_8_5_benchmark]

print(f"Found {len(probs_8_5)} problems with 8-5-benchmark flag")
print(f"Problems: {[p.name for p in probs_8_5]}")

# Configuration for benchmark
gm_config = {
    'device': device,
    'hidden_sizes': [3, 3],  # Two hidden layers with 3 units each
    'optimizer': 'adam',
    'num_epochs': 100000,  # 100k max epochs
    'traced_losses': [],
    'val_set': 'all',
    'lr': 1e-3,
    'lr_decay': 0.95,  # Decay learning rate
    'momentum': 0.9,
    'inverse_time_decay_constant': 10,
    'patience': 10,  # For learning rate scheduler
    'min_lr': 1e-8,
    'sampling_scheme': 'all',
    'num_batches_per_set': 1,
    'use_linspace_bias': False,
    'batch_size': 1024,
    'set_size': 10240,
    'num_samples': 147000*20,
    'seed': 4,
    'fdb': True,
    'debug': False,  # Disable debug output to avoid gradient tracking issues
    'lower_dim': False,
    'dope_factors': True,
    'gather_message_stats': True,  # Required for scaled_mse
    'ecl': 1024,  # Complexity bound (exact computation limit)
    'iB': 10,  # i-bound for mini-bucket
    'iB_backwards': 10,
    'plot_messages': False,
    'approximation_method': 'nn',
    'convex_early_stopping': True,  # Enable early stopping
    'convex_patience': 50,  # Stop if no improvement for 50 epochs
    'convex_min_delta': 1e-8,  # Minimum improvement threshold
}

def validate_partition_result(Z_hat, Z_true, prob_name, loss_fn):
    """
    Validate that the partition function result is reasonable.

    Args:
        Z_hat: Estimated log partition function
        Z_true: True log partition function
        prob_name: Name of the problem
        loss_fn: Loss function used

    Returns:
        bool: True if result looks reasonable, False otherwise
    """
    error = Z_hat - Z_true
    abs_error = abs(error)

    # Check if result is completely unreasonable (e.g., negative when should be positive)
    if Z_true > 0 and Z_hat < 0:
        print(f"WARNING: {prob_name} with {loss_fn}: Z_hat={Z_hat:.2f} is negative but Z_true={Z_true:.2f} is positive!")
        return False

    # Check if error is extremely large (more than 100 in log space)
    if abs_error > 100:
        print(f"WARNING: {prob_name} with {loss_fn}: Absolute error {abs_error:.2f} is very large!")
        return False

    # Result looks reasonable
    return True

def test_single_problem(prob, loss_fn):
    """Test a single problem with a specific loss function."""
    print(f"\n{'='*70}")
    print(f"Problem: {prob.name}")
    print(f"Width: {prob.width}, Loss: {loss_fn}")
    print(f"Z_true: {prob.Z:.6f}")
    print(f"{'='*70}")

    gm_config['loss_fn'] = loss_fn

    try:
        fastgm = FastGM(uai_file=prob.uai_file, nn_config=gm_config, device=device)
        t = time.time()
        Z_hat = fastgm.get_log_partition_function()
        elapsed = time.time() - t

        error = Z_hat - prob.Z
        abs_error = abs(error)

        # Validate result
        is_valid = validate_partition_result(Z_hat, prob.Z, prob.name, loss_fn)

        data_point = {
            'prob': prob.name,
            'width': prob.width,
            'approximation_method': gm_config['approximation_method'],
            'loss_fn': loss_fn,
            'ecl': gm_config['ecl'],
            'iB': gm_config['iB'],
            'hidden_sizes': gm_config['hidden_sizes'],
            'Z_hat': Z_hat,
            'Z': prob.Z,
            'num_trained': fastgm.num_trained,
            'err': error,
            'abs_err': abs_error,
            'time': round(elapsed, 2),
            'is_valid': is_valid
        }

        print(f"Z_hat: {Z_hat:.6f}")
        print(f"Z_true: {prob.Z:.6f}")
        print(f"Error: {error:+.6f}")
        print(f"Abs Error: {abs_error:.6f}")
        print(f"Num trained: {fastgm.num_trained}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Valid result: {'✓' if is_valid else '✗'}")

        return data_point

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_experiments():
    """Run experiments comparing scaled_mse and unnormalized_kl."""

    # Loss functions to compare
    losses_to_test = [
        'scaled_mse',
        'unnormalized_kl',
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'scaled_mse_ukl_benchmark_8_5_{timestamp}.pkl'

    all_results = {}

    for loss in losses_to_test:
        print(f"\n\n{'#'*70}")
        print(f"# Testing loss: {loss}")
        print(f"{'#'*70}\n")

        all_results[loss] = []

        for prob in probs_8_5:
            result = test_single_problem(prob, loss)
            if result:
                all_results[loss].append(result)

    # Save results
    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)

    print(f"\n\n{'='*70}")
    print(f"SUMMARY OF RESULTS")
    print(f"{'='*70}\n")

    # Print summary table
    print(f"{'Loss':<25} {'Avg Abs Error':<15} {'Total Time (s)':<15} {'Valid Results':<15}")
    print(f"{'-'*70}")

    for loss in losses_to_test:
        results = all_results[loss]
        if results:
            avg_abs_err = sum(r['abs_err'] for r in results) / len(results)
            total_time = sum(r['time'] for r in results)
            valid_count = sum(1 for r in results if r['is_valid'])
            print(f"{loss:<25} {avg_abs_err:<15.6f} {total_time:<15.2f} {valid_count}/{len(results)}")

    print(f"\n{'='*70}")
    print(f"Detailed results by problem:")
    print(f"{'='*70}\n")

    # Print detailed results for each problem
    for prob in probs_8_5:
        print(f"\n{prob.name} (width={prob.width}, Z_true={prob.Z:.4f}):")
        print(f"  {'Loss':<25} {'Z_hat':<12} {'Error':<12} {'Abs Err':<12} {'Time (s)':<10} {'Valid':<6}")
        print(f"  {'-'*80}")
        for loss in losses_to_test:
            results = [r for r in all_results[loss] if r['prob'] == prob.name]
            if results:
                r = results[0]
                valid_str = '✓' if r['is_valid'] else '✗'
                print(f"  {loss:<25} {r['Z_hat']:<12.4f} {r['err']:<+12.4f} {r['abs_err']:<12.6f} {r['time']:<10.2f} {valid_str:<6}")

    print(f"\n\nResults saved to: {results_file}")

    return all_results

if __name__ == '__main__':
    print(f"Starting scaled_mse vs unnormalized_kl benchmark on 8-5 problems")
    print(f"\nConfiguration:")
    print(f"  Hidden sizes: {gm_config['hidden_sizes']}")
    print(f"  Max epochs: {gm_config['num_epochs']}")
    print(f"  Early stopping: {gm_config['convex_early_stopping']}")
    print(f"  Patience: {gm_config['convex_patience']}")
    print(f"  Min delta: {gm_config['convex_min_delta']}")
    print(f"  Complexity bound (ecl): {gm_config['ecl']}")
    print(f"  i-bound: {gm_config['iB']}")
    print(f"  Device: {device}")
    print(f"  Number of problems: {len(probs_8_5)}")
    print(f"\nProblems to test:")
    for i, p in enumerate(probs_8_5, 1):
        print(f"  {i}. {p.name} (width={p.width}, Z={p.Z:.4f})")

    results = run_experiments()

    print("\n\nExperiments complete!")
