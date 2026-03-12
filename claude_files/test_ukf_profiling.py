"""
Quick UKF Sequential Loss Profiling Test
Tests on a single bucket with 256 states to identify bottlenecks
"""

import time
from nce.neural_networks.ukf_helpers import print_profiling_stats, reset_profiling_stats
from nce.inference import FastGM
from nce.problems import test_problems

# Device
device = 'cuda'

# GM Config
gm_config = {
    'device': device,
    'hidden_sizes': [3, 3],  # Small network for faster testing
    'optimizer': 'adam',
    'num_epochs': 100,  # Reduced for quick test
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
    'ecl': 1e6,  # Force NN training even for small buckets
    'iB_backwards': 100,
    'plot_messages': False,
    'approximation_method': 'nn',
    'display_intermediate': True,  # Show training progress

    # UKF Sequential Loss with default parameters (optimized)
    'loss_fn': 'ukf_sequential',  # Uses defaults: recompute every 10 batches, 500 samples
}

print("="*70)
print("UKF SEQUENTIAL LOSS - PROFILING TEST")
print("="*70)
print(f"Configuration:")
print(f"  Loss function: {gm_config['loss_fn']}")
print(f"  Hidden sizes: {gm_config['hidden_sizes']}")
print(f"  Num epochs: {gm_config['num_epochs']}")
print(f"  Device: {device}")
print("="*70)

# Get problem (using grid problem which has buckets with ~256 states)
probs = [test_problems[k] for k in list(test_problems.keys())]
probs = [p for p in probs if p.is_10_7_benchmark]
prob = probs[6]  # Grid problem

print(f"\nLoading problem: {prob.name}")
print(f"Width: {prob.width}")

# Reset profiling stats
reset_profiling_stats()

# Create FastGM
print("\nCreating FastGM...")
fastgm = FastGM(uai_file=prob.uai_file, nn_config=gm_config, device=device)

# Use bucket 5 (as in original test file)
bucket_idx = 5
print(f"\nTraining bucket {bucket_idx}...")

# Time the training
start_time = time.time()

# Train the bucket (this will use UKF sequential loss)
try:
    from nce.utils.message_gradient import get_message_gradient
    mg, m = get_message_gradient(fastgm, bucket_idx)
    m_hat = fastgm.buckets[bucket_idx].compute_message_nn()

    training_time = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Total training time: {training_time:.2f}s")
    print(f"Z_hat: {(m_hat.to_exact() * mg).sum_all_entries()}")
    print(f"Z: {(m * mg).sum_all_entries()}")

except Exception as e:
    print(f"\nError during training: {e}")
    import traceback
    traceback.print_exc()

# Print profiling results
print(f"\n{'='*70}")
print("PROFILING RESULTS")
print(f"{'='*70}")
print_profiling_stats()

print("\nDone!")
