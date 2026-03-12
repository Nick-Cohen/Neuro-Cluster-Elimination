#!/usr/bin/env python
"""
Example usage of elp_least_squares loss function in your existing workflow.

This shows how to integrate the new loss into your benchmark testing.
"""

import time
import pickle
from nce.inference import FastGM
from nce.problems import test_problems

device = 'cuda'

# Get test problems (same as your gather_nn_data_width20s.py)
probs = [test_problems[k] for k in list(test_problems.keys())]
probs = [p for p in probs if p.is_10_7_benchmark]

# Your existing config (from gather_nn_data_width20s.py)
gm_config = {
    'device': device,
    'hidden_sizes': [],
    'optimizer': 'adam',
    'num_epochs': 50000,
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
    'approximation_method': 'nn',
}

def gather_benchmark_data_single(prob, loss_fn, save_location=None):
    """Test a single problem with a specific loss function."""
    print(f"\nProblem: {prob.name}, Loss: {loss_fn}")

    gm_config['loss_fn'] = loss_fn

    fastgm = FastGM(uai_file=prob.uai_file, nn_config=gm_config, device=device)
    t = time.time()
    Z_hat = fastgm.get_log_partition_function()

    data_point = {
        'prob': prob.name,
        'approximation_method': gm_config['approximation_method'],
        'loss_fn': loss_fn,
        'Z_hat': Z_hat,
        'Z': prob.Z,
        'num_trained': fastgm.num_trained,
        'err': Z_hat - prob.Z,
        'abs_err': abs(Z_hat - prob.Z),
        'time': round(time.time() - t, 2)
    }

    print(f"Z_hat: {Z_hat:.6f}, Z_true: {prob.Z:.6f}")
    print(f"Error: {data_point['abs_err']:.6f}")
    print(f"Time: {data_point['time']}s")

    if save_location:
        with open(save_location, 'wb') as f:
            pickle.dump(data_point, f)

    return data_point

# Example 1: Test elp_least_squares with different sample counts
print("="*70)
print("EXAMPLE 1: Testing elp_least_squares with different sample counts")
print("="*70)

losses_to_test = [
    'elp_least_squares,10',   # Fast, fewer samples
    'elp_least_squares,50',   # Balanced
    'elp_least_squares,100',  # More accurate
]

# Test on first problem (grid)
test_prob = probs[0]
results = []

for loss in losses_to_test:
    try:
        result = gather_benchmark_data_single(test_prob, loss)
        results.append(result)
    except Exception as e:
        print(f"Error with {loss}: {e}")

print("\nResults:")
for r in results:
    print(f"{r['loss_fn']:<30} Error: {r['abs_err']:.6f}  Time: {r['time']}s")

# Example 2: Compare elp_least_squares vs existing losses
print("\n" + "="*70)
print("EXAMPLE 2: Comparing elp_least_squares with existing losses")
print("="*70)

comparison_losses = [
    'elp_least_squares,50',
    'unnormalized_kl',
    'mg_sampled_loss_fdb_recompute,50',
    'logspace_mse_fdb',
    'linspace_mse_fdb',
]

results_comparison = []
for loss in comparison_losses:
    try:
        result = gather_benchmark_data_single(test_prob, loss)
        results_comparison.append(result)
    except Exception as e:
        print(f"Error with {loss}: {e}")

print("\nComparison Results:")
print(f"{'Loss Function':<40} {'Error':<12} {'Time (s)':<10}")
print("-"*70)
for r in sorted(results_comparison, key=lambda x: x['abs_err']):
    print(f"{r['loss_fn']:<40} {r['abs_err']:<12.6f} {r['time']:<10.2f}")

# Example 3: Integration with your existing workflow
print("\n" + "="*70)
print("EXAMPLE 3: Using in your gather_benchmark_data_multiple_losses")
print("="*70)

def gather_benchmark_data_multiple_losses_with_elp(prob_indices, losses, save_folder=None):
    """
    Modified version of your gather_benchmark_data_multiple_losses function
    that includes elp_least_squares.
    """
    for loss in losses:
        gm_config['loss_fn'] = loss
        save_location = f'{save_folder}/{loss}.pkl' if save_folder else None

        try:
            # Handle two-stage training for some losses
            if 'mg_sampled' in loss or 'approx_smg' in loss or 'linspace' in loss or 'elp_least_squares' in loss:
                gm_config['loss_fn'] = 'unnormalized_kl'  # pretraining
                gm_config['loss_fn2'] = loss  # actual loss
                gm_config['num_epochs2'] = gm_config['num_epochs']
            else:
                gm_config['loss_fn'] = loss
                if 'loss_fn2' in gm_config:
                    del gm_config['loss_fn2']
                if 'num_epochs2' in gm_config:
                    del gm_config['num_epochs2']

            data = {}
            for prob in [probs[i] for i in prob_indices]:
                result = gather_benchmark_data_single(prob, loss, save_location)
                data[prob.name] = result

        except Exception as e:
            print(f"Error processing loss {loss}: {e}")
            continue

# Add elp_least_squares to your existing loss list
your_losses = [
    'logspace_mse_fdb',
    'linspace_mse_fdb',
    'mg_sampled_loss_fdb_recompute,10',
    'approx_smg,10',
    'unnormalized_kl',
    'weighted_logspace_mse',
    # NEW: Add these
    'elp_least_squares,10',
    'elp_least_squares,50',
    'elp_least_squares,100',
]

print("\nYou can now add these to your losses_to_test in gather_nn_data_width20s.py:")
for loss in your_losses:
    if 'elp' in loss:
        print(f"  '{loss}',  # NEW")
    else:
        print(f"  '{loss}',")

print("\n" + "="*70)
print("Examples complete! Use these patterns in your existing scripts.")
print("="*70)
