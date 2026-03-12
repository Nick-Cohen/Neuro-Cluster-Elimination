#!/usr/bin/env python3
"""
Test script comparing:
1. unnormalized_kl training only (num_epochs2=0)
2. scaled_mse training
Using same parameters as previous benchmark tests
"""

import time
import pickle
import csv
from nce.inference import FastGM
from nce.problems import test_problems
import torch

device = 'cuda'

# Base configuration (matching previous tests)
gm_config = {
    'device': device,
    'hidden_sizes': [3, 3],
    'optimizer': 'adam',
    'num_epochs': 100000,
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
    'ecl': 1e6,
    'iB_backwards': 100,
    'plot_messages': False,
    'approximation_method': 'nn',
    'num_leaves': 25,
    'num_iterations': 100,
    'dt_lr': 0.25,
    'dt_momentum': 0.25,
    'dt_random_seed': 0,
    'dt_convergence_threshold': 0.01,
}

# Get benchmark problems
probs = [test_problems[k] for k in list(test_problems.keys())]
probs = [p for p in probs if p.is_10_7_benchmark]

# Problem indices
grids = [0, 6]
pedigree = [1]
rbm = [2, 3]
BNs = [4]
all_indices = grids + pedigree + rbm + BNs

def run_single_test(prob, config_name, config):
    """Run a single test with given configuration"""
    print(f"\n{'='*60}")
    print(f"Testing {prob.name} with {config_name}")
    print(f"{'='*60}")

    try:
        fastgm = FastGM(uai_file=prob.uai_file, nn_config=config, device=device)
        t_start = time.time()
        Z_hat = fastgm.get_log_partition_function()
        elapsed = time.time() - t_start

        result = {
            'problem': prob.name,
            'config': config_name,
            'Z_hat': Z_hat,
            'Z_true': prob.Z,
            'error': Z_hat - prob.Z,
            'abs_error': abs(Z_hat - prob.Z),
            'num_trained': fastgm.num_trained,
            'time': round(elapsed, 2)
        }

        print(f"Z_hat: {Z_hat:.6f}")
        print(f"Z_true: {prob.Z:.6f}")
        print(f"Abs Error: {result['abs_error']:.6f}")
        print(f"Num trained buckets: {fastgm.num_trained}")
        print(f"Time: {elapsed:.2f}s")

        del fastgm
        return result

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            'problem': prob.name,
            'config': config_name,
            'Z_hat': None,
            'Z_true': prob.Z,
            'error': None,
            'abs_error': None,
            'num_trained': None,
            'time': None,
            'error_msg': str(e)
        }

def main():
    results = []

    # Test all benchmark problems
    test_indices = all_indices

    # Configuration 1: unnormalized_kl only
    config_ukl_only = gm_config.copy()
    config_ukl_only['loss_fn'] = 'unnormalized_kl'
    config_ukl_only['num_epochs'] = 100000

    # Configuration 2: scaled_mse
    config_scaled_mse = gm_config.copy()
    config_scaled_mse['loss_fn'] = 'scaled_mse'
    config_scaled_mse['num_epochs'] = 100000

    print(f"Testing {len(test_indices)} benchmark problems...")

    for idx in test_indices:
        prob = probs[idx]

        # Test 1: UKL only
        result_ukl = run_single_test(prob, 'unnormalized_kl_only', config_ukl_only)
        results.append(result_ukl)

        # Test 2: scaled_mse
        result_scaled = run_single_test(prob, 'scaled_mse', config_scaled_mse)
        results.append(result_scaled)

    # Print summary
    print("\n" + "="*120)
    print("BENCHMARK TEST RESULTS - UKL vs scaled_mse")
    print("="*120)
    print(f"{'Problem':<30} {'UKL Only Z_hat':<18} {'Scaled MSE Z_hat':<18} {'Z_true':<18} {'UKL Abs Err':<15} {'Scaled Abs Err':<15}")
    print("-"*120)

    for idx in test_indices:
        prob = probs[idx]
        ukl_result = next((r for r in results if r['problem'] == prob.name and r['config'] == 'unnormalized_kl_only'), None)
        scaled_result = next((r for r in results if r['problem'] == prob.name and r['config'] == 'scaled_mse'), None)

        ukl_z = f"{ukl_result['Z_hat']:.6f}" if ukl_result and ukl_result['Z_hat'] is not None else "ERROR"
        scaled_z = f"{scaled_result['Z_hat']:.6f}" if scaled_result and scaled_result['Z_hat'] is not None else "ERROR"
        z_true = f"{prob.Z:.6f}"
        ukl_err = f"{ukl_result['abs_error']:.6f}" if ukl_result and ukl_result['abs_error'] is not None else "ERROR"
        scaled_err = f"{scaled_result['abs_error']:.6f}" if scaled_result and scaled_result['abs_error'] is not None else "ERROR"

        print(f"{prob.name:<30} {ukl_z:<18} {scaled_z:<18} {z_true:<18} {ukl_err:<15} {scaled_err:<15}")

    print("="*120)

    # Save results to pickle
    output_pickle = '/home/cohenn1/NCE/ukl_vs_scaled_mse_results.pkl'
    with open(output_pickle, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: {output_pickle}")

    # Create CSV spreadsheet
    output_csv = '/home/cohenn1/NCE/ukl_vs_scaled_mse_results.csv'
    with open(output_csv, 'w', newline='') as f:
        fieldnames = ['problem', 'ukl_only_Z_hat', 'scaled_mse_Z_hat', 'Z_true',
                      'ukl_only_error', 'scaled_mse_error',
                      'ukl_only_abs_error', 'scaled_mse_abs_error',
                      'ukl_only_time', 'scaled_mse_time']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx in test_indices:
            prob = probs[idx]
            ukl_result = next((r for r in results if r['problem'] == prob.name and r['config'] == 'unnormalized_kl_only'), None)
            scaled_result = next((r for r in results if r['problem'] == prob.name and r['config'] == 'scaled_mse'), None)

            row = {
                'problem': prob.name,
                'ukl_only_Z_hat': ukl_result['Z_hat'] if ukl_result else None,
                'scaled_mse_Z_hat': scaled_result['Z_hat'] if scaled_result else None,
                'Z_true': prob.Z,
                'ukl_only_error': ukl_result['error'] if ukl_result else None,
                'scaled_mse_error': scaled_result['error'] if scaled_result else None,
                'ukl_only_abs_error': ukl_result['abs_error'] if ukl_result else None,
                'scaled_mse_abs_error': scaled_result['abs_error'] if scaled_result else None,
                'ukl_only_time': ukl_result['time'] if ukl_result else None,
                'scaled_mse_time': scaled_result['time'] if scaled_result else None,
            }
            writer.writerow(row)

    print(f"CSV spreadsheet saved to: {output_csv}")

if __name__ == '__main__':
    main()
