# Test that different backward_ecl values produce different backward messages
import torch
from nce.utils.message_gradient import get_message_gradient
from nce.inference import FastGM
from nce.problems import test_problems

device = 'cuda'

# Base config
gm_config = {
    'device': device,
    'hidden_sizes': [3, 3],
    'optimizer': 'adam',
    'num_epochs': 1,
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
    'batch_size': 256,
    'set_size': 10240,
    'num_samples': 147000*20,
    'seed': 4,
    'fdb': True,
    'debug': False,
    'lower_dim': False,
    'dope_factors': True,
    'gather_message_stats': False,
    'iB': 100,
    'ecl': 2**20-1,
    'iB_backwards': 100,
    'plot_messages': False,
    'approximation_method': 'nn',
    'num_leaves': 25,
    'num_iterations': 100,
    'dt_lr': 0.25,
    'dt_momentum': 0.25,
    'dt_random_seed': 0,
    'dt_convergence_threshold': 0.01,
    'display_intermediate': False,
    'loss_fn': 'unnormalized_kl',
    'num_epochs2': 0,
}

# Get the first problem (grid10x10.f10.wrap)
probs = [test_problems[k] for k in list(test_problems.keys())]
probs = [p for p in probs if p.is_10_7_benchmark]
prob = probs[0]
print(f"Testing with problem: {prob.name}")

# Create FastGM
fastgm = FastGM(uai_file=prob.uai_file, nn_config=gm_config, device=device)

# Test bucket 18
bucket_idx = 18

# Using default iB=100 - backward_ecl should derive effective iB automatically
default_iB = 100

print("\n" + "="*70)
print("TEST 1: Exact backward message (very high ecl, iB=100)")
print("="*70)
mg_exact, m = get_message_gradient(fastgm, bucket_idx, iB=default_iB, backward_ecl=2**30, approximation_method='wmb')
print(f"Shape: {mg_exact.tensor.shape}")
exact_values = mg_exact.tensor.flatten()[:10].tolist()
print(f"First 10 values: {exact_values}")

# Reload fastgm to reset state
fastgm = FastGM(uai_file=prob.uai_file, nn_config=gm_config, device=device)

print("\n" + "="*70)
print(f"TEST 2: Approximate via ecl=1024 only (iB={default_iB}, effective_iB should be 10)")
print("="*70)
mg_1024, m = get_message_gradient(fastgm, bucket_idx, iB=default_iB, backward_ecl=1024, approximation_method='wmb')
print(f"Shape: {mg_1024.tensor.shape}")
values_1024 = mg_1024.tensor.flatten()[:10].tolist()
print(f"First 10 values: {values_1024}")

# Reload fastgm to reset state
fastgm = FastGM(uai_file=prob.uai_file, nn_config=gm_config, device=device)

print("\n" + "="*70)
print(f"TEST 3: More approximate via ecl=64 only (iB={default_iB}, effective_iB should be 6)")
print("="*70)
mg_64, m = get_message_gradient(fastgm, bucket_idx, iB=default_iB, backward_ecl=64, approximation_method='wmb')
print(f"Shape: {mg_64.tensor.shape}")
values_64 = mg_64.tensor.flatten()[:10].tolist()
print(f"First 10 values: {values_64}")

# Reload fastgm to reset state
fastgm = FastGM(uai_file=prob.uai_file, nn_config=gm_config, device=device)

print("\n" + "="*70)
print(f"TEST 4: Very approximate via ecl=8 only (iB={default_iB}, effective_iB should be 3)")
print("="*70)
mg_8, m = get_message_gradient(fastgm, bucket_idx, iB=default_iB, backward_ecl=8, approximation_method='wmb')
print(f"Shape: {mg_8.tensor.shape}")
values_8 = mg_8.tensor.flatten()[:10].tolist()
print(f"First 10 values: {values_8}")

print("\n" + "="*70)
print("COMPARISON")
print("="*70)

# Check if values are different
def arrays_equal(a, b, tol=1e-6):
    return all(abs(x - y) < tol for x, y in zip(a, b))

print(f"\nExact vs ecl=1024: {'SAME' if arrays_equal(exact_values, values_1024) else 'DIFFERENT'}")
print(f"Exact vs ecl=64:   {'SAME' if arrays_equal(exact_values, values_64) else 'DIFFERENT'}")
print(f"Exact vs ecl=8:    {'SAME' if arrays_equal(exact_values, values_8) else 'DIFFERENT'}")
print(f"ecl=1024 vs ecl=64: {'SAME' if arrays_equal(values_1024, values_64) else 'DIFFERENT'}")
print(f"ecl=1024 vs ecl=8:  {'SAME' if arrays_equal(values_1024, values_8) else 'DIFFERENT'}")
print(f"ecl=64 vs ecl=8:    {'SAME' if arrays_equal(values_64, values_8) else 'DIFFERENT'}")

print("\n" + "="*70)
if arrays_equal(exact_values, values_1024) and arrays_equal(exact_values, values_64) and arrays_equal(exact_values, values_8):
    print("BUG: All values are the same regardless of backward_ecl!")
else:
    print("SUCCESS: Different backward_ecl values produce different results!")
print("="*70)
