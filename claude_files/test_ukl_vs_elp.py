#!/usr/bin/env python3
"""
Test script comparing:
1. unnormalized_kl training only (num_epochs2=0)
2. unnormalized_kl followed by elp_recompute,10 training
"""

import time
import pickle
import csv
from nce.inference import FastGM
from nce.problems import test_problems
import torch

device = 'cpu'  # Use CPU to avoid CUDA memory errors

# Base configuration (keeping all values from original)
gm_config = {
    'device': device,
    # ------------for nn training -------------------
    'hidden_sizes': [3, 3],  # From original test setup
    'optimizer': 'adam',
    'num_epochs': 10,  # Reduced from 50000 for faster testing
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
    # -------------------------------------------------
    'fdb': True,
    'debug': False,
    'lower_dim': False,
    'dope_factors': True,
    'gather_message_stats': False,
    'iB': 100,
    'ecl': 256,
    'iB_backwards': 100,
    'plot_messages': False,  # Turn off plotting for batch processing
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

# Problem indices (from original)
grids = [0, 6]
pedigree = [1]
rbm = [2, 3]
BNs = [4]
promedas = [7]
all_indices = grids + pedigree + rbm + BNs  # + promedas

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

        # Clean up to free memory
        del fastgm
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return result

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

        # Clean up on error
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

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

    # Configuration 1: unnormalized_kl only (num_epochs2=0)
    config_ukl_only = gm_config.copy()
    config_ukl_only['loss_fn'] = 'unnormalized_kl'
    config_ukl_only['loss_fn2'] = 'elp_recompute,10'
    config_ukl_only['num_epochs2'] = 0  # This makes it only train with unnormalized_kl

    # Configuration 2: unnormalized_kl followed by elp_recompute,10
    config_ukl_elp = gm_config.copy()
    config_ukl_elp['loss_fn'] = 'unnormalized_kl'
    config_ukl_elp['loss_fn2'] = 'elp_recompute,10'
    config_ukl_elp['num_epochs'] = 10  # UKL training epochs
    config_ukl_elp['num_epochs2'] = 10  # ELP training epochs after ukl

    # Run tests on all benchmark problems
    for idx in all_indices:
        prob = probs[idx]

        # Test 1: UKL only
        result_ukl = run_single_test(prob, 'unnormalized_kl_only', config_ukl_only)
        results.append(result_ukl)

        # Test 2: UKL + ELP
        result_elp = run_single_test(prob, 'unnormalized_kl+elp_recompute', config_ukl_elp)
        results.append(result_elp)

    # Save detailed results to pickle
    output_pickle = '/home/cohenn1/NCE/ukl_vs_elp_results.pkl'
    with open(output_pickle, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nDetailed results saved to: {output_pickle}")

    # Create CSV spreadsheet
    output_csv = '/home/cohenn1/NCE/ukl_vs_elp_results.csv'
    with open(output_csv, 'w', newline='') as f:
        fieldnames = ['problem', 'ukl_only_Z_hat', 'ukl_elp_Z_hat', 'Z_true',
                      'ukl_only_error', 'ukl_elp_error',
                      'ukl_only_abs_error', 'ukl_elp_abs_error',
                      'ukl_only_time', 'ukl_elp_time']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Group results by problem
        for idx in all_indices:
            prob = probs[idx]
            ukl_result = next((r for r in results if r['problem'] == prob.name and r['config'] == 'unnormalized_kl_only'), None)
            elp_result = next((r for r in results if r['problem'] == prob.name and r['config'] == 'unnormalized_kl+elp_recompute'), None)

            row = {
                'problem': prob.name,
                'ukl_only_Z_hat': ukl_result['Z_hat'] if ukl_result else None,
                'ukl_elp_Z_hat': elp_result['Z_hat'] if elp_result else None,
                'Z_true': prob.Z,
                'ukl_only_error': ukl_result['error'] if ukl_result else None,
                'ukl_elp_error': elp_result['error'] if elp_result else None,
                'ukl_only_abs_error': ukl_result['abs_error'] if ukl_result else None,
                'ukl_elp_abs_error': elp_result['abs_error'] if elp_result else None,
                'ukl_only_time': ukl_result['time'] if ukl_result else None,
                'ukl_elp_time': elp_result['time'] if elp_result else None,
            }
            writer.writerow(row)

    print(f"\nCSV spreadsheet saved to: {output_csv}")

    # Print summary table
    print("\n" + "="*120)
    print("SUMMARY TABLE")
    print("="*120)
    print(f"{'Problem':<30} {'UKL Only Z_hat':<18} {'UKL+ELP Z_hat':<18} {'Z_true':<18} {'UKL Abs Err':<15} {'ELP Abs Err':<15}")
    print("-"*120)

    for idx in all_indices:
        prob = probs[idx]
        ukl_result = next((r for r in results if r['problem'] == prob.name and r['config'] == 'unnormalized_kl_only'), None)
        elp_result = next((r for r in results if r['problem'] == prob.name and r['config'] == 'unnormalized_kl+elp_recompute'), None)

        ukl_z = f"{ukl_result['Z_hat']:.6f}" if ukl_result and ukl_result['Z_hat'] is not None else "ERROR"
        elp_z = f"{elp_result['Z_hat']:.6f}" if elp_result and elp_result['Z_hat'] is not None else "ERROR"
        z_true = f"{prob.Z:.6f}"
        ukl_err = f"{ukl_result['abs_error']:.6f}" if ukl_result and ukl_result['abs_error'] is not None else "ERROR"
        elp_err = f"{elp_result['abs_error']:.6f}" if elp_result and elp_result['abs_error'] is not None else "ERROR"

        print(f"{prob.name:<30} {ukl_z:<18} {elp_z:<18} {z_true:<18} {ukl_err:<15} {elp_err:<15}")

    print("="*120)

if __name__ == '__main__':
    main()
