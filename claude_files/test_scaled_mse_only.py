#!/usr/bin/env python3
"""
Test script for scaled_mse loss function only
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
    'loss_fn': 'logspace_mse_fdb',
    'loss_fn2': 'scaled_mse',
    'num_epochs2': 100000,
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

def run_single_test(prob):
    """Run a single test with scaled_mse"""
    print(f"\n{'='*60}")
    print(f"Testing {prob.name} with scaled_mse")
    print(f"{'='*60}")

    try:
        fastgm = FastGM(uai_file=prob.uai_file, nn_config=gm_config, device=device)
        t_start = time.time()
        Z_hat = fastgm.get_log_partition_function()
        elapsed = time.time() - t_start

        result = {
            'problem': prob.name,
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

    print(f"Testing scaled_mse on {len(all_indices)} benchmark problems...")

    for idx in all_indices:
        prob = probs[idx]
        result = run_single_test(prob)
        results.append(result)

    # Print summary
    print("\n" + "="*90)
    print("BENCHMARK TEST RESULTS - scaled_mse")
    print("="*90)
    print(f"{'Problem':<30} {'Z_hat':<18} {'Z_true':<18} {'Abs Error':<15} {'Time (s)':<10}")
    print("-"*90)

    for result in results:
        z_hat = f"{result['Z_hat']:.6f}" if result['Z_hat'] is not None else "ERROR"
        z_true = f"{result['Z_true']:.6f}"
        abs_err = f"{result['abs_error']:.6f}" if result['abs_error'] is not None else "ERROR"
        elapsed = f"{result['time']:.2f}" if result['time'] is not None else "ERROR"

        print(f"{result['problem']:<30} {z_hat:<18} {z_true:<18} {abs_err:<15} {elapsed:<10}")

    print("="*90)

    # Save results to pickle
    output_pickle = '/home/cohenn1/NCE/scaled_mse_logspace_pretrain_results.pkl'
    with open(output_pickle, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: {output_pickle}")

    # Create CSV spreadsheet
    output_csv = '/home/cohenn1/NCE/scaled_mse_logspace_pretrain_results.csv'
    with open(output_csv, 'w', newline='') as f:
        fieldnames = ['problem', 'Z_hat', 'Z_true', 'error', 'abs_error', 'num_trained', 'time']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            row = {
                'problem': result['problem'],
                'Z_hat': result['Z_hat'],
                'Z_true': result['Z_true'],
                'error': result['error'],
                'abs_error': result['abs_error'],
                'num_trained': result['num_trained'],
                'time': result['time'],
            }
            writer.writerow(row)

    print(f"CSV spreadsheet saved to: {output_csv}")

if __name__ == '__main__':
    main()
