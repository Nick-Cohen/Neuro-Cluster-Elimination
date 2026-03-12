#!/usr/bin/env python
"""
Test partition function errors with elp_least_squares vs other losses.
Mimics the gather_nn_data_width20s.py testing approach.
"""

import time
import torch
from nce.inference import FastGM
from nce.utils.message_gradient import get_message_gradient
from nce.problems import test_problems
from nce.utils.plots import plot_fastfactor_comparison

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Base configuration
gm_config = {
    'device': device,
    'hidden_sizes': [],  # Linear model
    'optimizer': 'adam',
    'num_epochs': 2000,
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
    'ecl': 256,  # Only approximate messages with width >= this
    'iB_backwards': 100,
    'plot_messages': False,
    'approximation_method': 'nn'
}

def train_bucket(fastgm, idx, plot_mg=False):
    """Train a single bucket and compute partition function error."""
    b = fastgm.buckets[idx]
    mg, m = get_message_gradient(fastgm, idx)
    m_hat = b.compute_message_nn()

    if plot_mg:
        plot_fastfactor_comparison(m, m_hat, mg)
    else:
        plot_fastfactor_comparison(m, m_hat)

    Z_hat = (m_hat.to_exact() * mg).sum_all_entries()
    Z = (m * mg).sum_all_entries()

    print(f'Z_hat: {Z_hat}')
    print(f'Z:     {Z}')
    print(f'Error: {abs(Z_hat - Z)}')

    return m, m_hat, mg, float(Z), float(Z_hat)

def test_problem_with_loss(prob, loss_fn):
    """Test a problem with a specific loss function."""
    print(f"\n{'='*70}")
    print(f"Problem: {prob.name}, Loss: {loss_fn}")
    print(f"{'='*70}")

    gm_config['loss_fn'] = loss_fn

    try:
        t_start = time.time()
        fastgm = FastGM(uai_file=prob.uai_file, nn_config=gm_config, device=device)
        Z_hat = fastgm.get_log_partition_function()
        t_elapsed = time.time() - t_start

        Z_true = prob.Z
        error = abs(Z_hat - Z_true)

        print(f"Z_true:    {Z_true:.6f}")
        print(f"Z_hat:     {Z_hat:.6f}")
        print(f"Error:     {error:.6f}")
        print(f"Rel error: {(error/abs(Z_true)*100):.4f}%")
        print(f"Time:      {t_elapsed:.2f}s")
        print(f"Num NN trained: {fastgm.num_trained}")

        return {
            'prob': prob.name,
            'loss_fn': loss_fn,
            'Z_true': Z_true,
            'Z_hat': Z_hat,
            'error': error,
            'rel_error': error / abs(Z_true),
            'time': t_elapsed,
            'num_trained': fastgm.num_trained
        }
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Get test problems
    probs = [test_problems[k] for k in list(test_problems.keys())]
    probs = [p for p in probs if p.is_10_7_benchmark]

    # Select a small problem for quick testing
    # Index 0 is grid10x10.f10.wrap (width 21)
    test_prob = probs[0]

    print(f"\nTesting on: {test_prob.name}")
    print(f"Width: {test_prob.width}")
    print(f"True log Z: {test_prob.Z}")

    # Loss functions to compare
    loss_functions = [
        'elp_least_squares,10',
        'elp_least_squares,50',
        'unnormalized_kl',
        'mg_sampled_loss_fdb_recompute,10',
    ]

    results = []
    for loss_fn in loss_functions:
        result = test_problem_with_loss(test_prob, loss_fn)
        if result:
            results.append(result)

    # Print summary
    print(f"\n\n{'='*90}")
    print("SUMMARY - PARTITION FUNCTION ERRORS")
    print(f"{'='*90}")
    print(f"{'Loss Function':<40} {'Error':<12} {'Rel Error %':<12} {'Time (s)':<10}")
    print(f"{'-'*90}")

    for r in results:
        print(f"{r['loss_fn']:<40} {r['error']:<12.6f} {r['rel_error']*100:<12.4f} {r['time']:<10.2f}")

    # Find best
    if results:
        best = min(results, key=lambda x: x['error'])
        print(f"\n{'='*90}")
        print(f"Best loss function: {best['loss_fn']}")
        print(f"Error: {best['error']:.6f} ({best['rel_error']*100:.4f}%)")
        print(f"{'='*90}")

if __name__ == "__main__":
    main()
