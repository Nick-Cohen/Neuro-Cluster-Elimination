"""
Diagnostic test for scaled_mse conversion issue
Tests whether the DataPreprocessor properly stores and uses scaling_factor
"""

import torch
from nce.inference import FastGM
from nce.problems import test_problems

device = 'cuda'

# Config for scaled_mse
gm_config = {
    'device': device,
    'hidden_sizes': [],  # Linear model
    'optimizer': 'sgd',
    'num_epochs': 100,
    'traced_losses': [],
    'val_set': 'all',
    'lr': 1e-3,
    'lr_decay': 1,
    'momentum': 0.9,
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
    'ecl': 1e6,
    'iB': 100,
    'loss_fn': 'scaled_mse',
}

print("="*70)
print("DIAGNOSTIC TEST: scaled_mse conversion")
print("="*70)

# Get a small problem
probs = [test_problems[k] for k in list(test_problems.keys())]
probs = [p for p in probs if p.is_10_7_benchmark]
prob = probs[0]  # Small grid problem

print(f"\nTesting on: {prob.name}")
print(f"Loss function: {gm_config['loss_fn']}")

# Create FastGM
fastgm = FastGM(uai_file=prob.uai_file, nn_config=gm_config, device=device)

# Just force training a specific bucket by lowering ecl
old_ecl = gm_config['ecl']
gm_config['ecl'] = 100  # Force training

# Get bucket 10 (a medium-sized bucket in the grid problem)
first_trained_bucket = fastgm.bucket_dict[10]

print(f"\nTraining bucket: {first_trained_bucket.label}")
print(f"Bucket complexity: {first_trained_bucket.get_ec()}")

# Get the trainer that will be created for this bucket
from nce.neural_networks.net import Net
from nce.neural_networks.train import Trainer

net = Net(first_trained_bucket, hidden_sizes=gm_config['hidden_sizes'])
trainer = Trainer(net=net, bucket=first_trained_bucket)

print(f"\n{'='*70}")
print("BEFORE TRAINING")
print(f"{'='*70}")
print(f"data_preprocessor.exp_preprocessing: {trainer.data_preprocessor.exp_preprocessing}")
print(f"data_preprocessor.scaling_factor: {trainer.data_preprocessor.scaling_factor}")
print(f"data_preprocessor.y_max_after_scaling: {trainer.data_preprocessor.y_max_after_scaling}")

# Train
print(f"\n{'='*70}")
print("TRAINING...")
print(f"{'='*70}")
trainer.train()

print(f"\n{'='*70}")
print("AFTER TRAINING")
print(f"{'='*70}")
print(f"data_preprocessor.exp_preprocessing: {trainer.data_preprocessor.exp_preprocessing}")
print(f"data_preprocessor.scaling_factor: {trainer.data_preprocessor.scaling_factor}")
print(f"data_preprocessor.y_max_after_scaling: {trainer.data_preprocessor.y_max_after_scaling}")

# Create FactorNN
from nce.inference.factor_nn import FactorNN
nn_message_factor = FactorNN(net, trainer.data_preprocessor)

print(f"\n{'='*70}")
print("CONVERTING TO FastFactor")
print(f"{'='*70}")

# Try conversion
try:
    fast_factor = nn_message_factor.to_exact()
    print("✓ Conversion successful!")
    print(f"FastFactor shape: {fast_factor.tensor.shape}")
    print(f"FastFactor has NaNs: {fast_factor.tensor.isnan().any()}")
    print(f"FastFactor min: {fast_factor.tensor.min().item():.6f}")
    print(f"FastFactor max: {fast_factor.tensor.max().item():.6f}")
    print(f"FastFactor mean: {fast_factor.tensor.mean().item():.6f}")
except Exception as e:
    print(f"✗ Conversion failed: {e}")
    import traceback
    traceback.print_exc()

# Compare with exact message
print(f"\n{'='*70}")
print("COMPARING WITH EXACT MESSAGE")
print(f"{'='*70}")

exact_message = first_trained_bucket.compute_message_exact()
print(f"Exact message min: {exact_message.tensor.min().item():.6f}")
print(f"Exact message max: {exact_message.tensor.max().item():.6f}")
print(f"Exact message mean: {exact_message.tensor.mean().item():.6f}")

# Compute error
if not fast_factor.tensor.isnan().any():
    diff = torch.abs(fast_factor.tensor - exact_message.tensor)
    print(f"\nAbsolute difference:")
    print(f"  Min: {diff.min().item():.6f}")
    print(f"  Max: {diff.max().item():.6f}")
    print(f"  Mean: {diff.mean().item():.6f}")
    print(f"  Median: {diff.median().item():.6f}")

print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
