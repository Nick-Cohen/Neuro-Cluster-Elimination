"""
Test scaled_mse conversion using Memorizer
This eliminates NN approximation error to isolate conversion issues
"""

import torch
from nce.inference import FastGM, FastBucket, FastFactor
from nce.inference.factor_nn import FactorNN
from nce.neural_networks.net import Memorizer
from nce.neural_networks.train import Trainer
from nce.data import DataPreprocessor
from nce.problems import test_problems

device = 'cuda'

print("="*70)
print("MEMORIZER TEST: scaled_mse conversion")
print("="*70)

# Get a small problem
probs = [test_problems[k] for k in list(test_problems.keys())]
probs = [p for p in probs if p.is_10_7_benchmark]
prob = probs[0]

# Simple config
gm_config = {
    'device': device,
    'hidden_sizes': [],
    'optimizer': 'sgd',
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
    'batch_size': 1024,
    'set_size': 10240,
    'num_samples': 147000*20,
    'seed': 4,
    'fdb': True,
    'debug': False,
    'lower_dim': False,
    'dope_factors': True,
    'gather_message_stats': False,
    'ecl': 100,
    'iB': 100,
    'iB_backwards': 100,
    'plot_messages': False,
    'approximation_method': 'nn',
    'loss_fn': 'scaled_mse',
}

print(f"\nProblem: {prob.name}")
print(f"Loss function: scaled_mse")

# Create FastGM
fastgm = FastGM(uai_file=prob.uai_file, nn_config=gm_config, device=device)

# Get a small bucket manually
print("\n" + "="*70)
print("CREATING TEST BUCKET")
print("="*70)

# Use bucket 26 (has 16 states - good size for testing)
test_bucket = fastgm.buckets[26]

print(f"Test bucket: {test_bucket.label}")
print(f"Message scope: {test_bucket.get_message_scope()}")
print(f"Message size: {test_bucket.get_message_size()}")

# Compute exact message
print("\n" + "="*70)
print("COMPUTING EXACT MESSAGE")
print("="*70)

exact_message = test_bucket.compute_message_exact()
print(f"Exact message shape: {exact_message.tensor.shape}")
print(f"Exact message stats:")
print(f"  Min: {exact_message.tensor.min().item():.6f}")
print(f"  Max: {exact_message.tensor.max().item():.6f}")
print(f"  Mean: {exact_message.tensor.mean().item():.6f}")

# Create a Trainer to get data loading infrastructure
print("\n" + "="*70)
print("CREATING TRAINER AND LOADING DATA")
print("="*70)

from nce.neural_networks.net import Net
net = Net(test_bucket, hidden_sizes=[])
trainer = Trainer(net=net, bucket=test_bucket)

# Check trainer's data preprocessor
print(f"\nTrainer data_preprocessor settings:")
print(f"  exp_preprocessing: {trainer.data_preprocessor.exp_preprocessing}")
print(f"  fdb: {trainer.data_preprocessor.fdb}")
print(f"  is_logspace: {trainer.data_preprocessor.is_logspace}")

# Load all data with scaling_factor
print("\n" + "="*70)
print("LOADING DATA WITH SCALING_FACTOR")
print("="*70)

# Get scaling factor
try:
    sigma_f, sigma_g, rho = test_bucket.get_fw_bw_stats()
    scaling_factor = (sigma_f ** 2) / (sigma_f ** 2 + sigma_g ** 2 + 1e-12)
    print(f"Computed scaling_factor: {scaling_factor}")
    print(f"  sigma_f: {sigma_f}")
    print(f"  sigma_g: {sigma_g}")
    print(f"  rho: {rho}")
except Exception as e:
    print(f"Error getting scaling_factor: {e}")
    scaling_factor = None

# Load data
x_all, y_all, _ = trainer.dataloader.load(all=True, scaling_factor=scaling_factor)

print(f"\nData loaded:")
print(f"  x_all shape: {x_all.shape}")
print(f"  y_all shape: {y_all.shape}")
print(f"  y_all min: {y_all.min().item():.6f}")
print(f"  y_all max: {y_all.max().item():.6f}")
print(f"  y_all mean: {y_all.mean().item():.6f}")

# Check data_preprocessor after loading
print(f"\nData_preprocessor after loading:")
print(f"  exp_preprocessing: {trainer.data_preprocessor.exp_preprocessing}")
print(f"  scaling_factor: {trainer.data_preprocessor.scaling_factor}")
print(f"  y_max_after_scaling: {trainer.data_preprocessor.y_max_after_scaling}")

# Create Memorizer
print("\n" + "="*70)
print("CREATING MEMORIZER")
print("="*70)

mem = Memorizer(test_bucket, x_all, y_all)
print(f"Memorizer created with {len(x_all)} entries")

# Verify memorizer outputs match targets
print("\nVerifying memorizer...")
with torch.no_grad():
    mem_outputs = mem(x_all[:10])
    print(f"First 10 targets:  {y_all[:10].cpu().numpy()}")
    print(f"First 10 mem outputs: {mem_outputs.cpu().numpy()}")
    print(f"Match: {torch.allclose(mem_outputs, y_all[:10])}")

# Create FactorNN with memorizer
print("\n" + "="*70)
print("CREATING FACTORNN WITH MEMORIZER")
print("="*70)

factor_nn = FactorNN(mem, trainer.data_preprocessor)
print(f"FactorNN created")
print(f"  labels: {factor_nn.labels}")
print(f"  is_nn: {factor_nn.is_nn}")

# Check data_preprocessor in FactorNN
print(f"\nFactorNN data_preprocessor:")
print(f"  exp_preprocessing: {factor_nn.data_processor.exp_preprocessing}")
print(f"  scaling_factor: {factor_nn.data_processor.scaling_factor}")
print(f"  y_max_after_scaling: {factor_nn.data_processor.y_max_after_scaling}")

# Convert to FastFactor
print("\n" + "="*70)
print("CONVERTING TO FASTFACTOR")
print("="*70)

try:
    converted_factor = factor_nn.to_exact()
    print("✓ Conversion successful!")

    print(f"\nConverted factor shape: {converted_factor.tensor.shape}")
    print(f"Converted factor stats:")
    print(f"  Min: {converted_factor.tensor.min().item():.6f}")
    print(f"  Max: {converted_factor.tensor.max().item():.6f}")
    print(f"  Mean: {converted_factor.tensor.mean().item():.6f}")
    print(f"  Has NaNs: {converted_factor.tensor.isnan().any()}")
    print(f"  Has Infs: {converted_factor.tensor.isinf().any()}")

except Exception as e:
    print(f"✗ Conversion failed!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Compare with exact message
print("\n" + "="*70)
print("COMPARING WITH EXACT MESSAGE")
print("="*70)

if converted_factor.tensor.shape != exact_message.tensor.shape:
    print(f"✗ Shape mismatch!")
    print(f"  Converted: {converted_factor.tensor.shape}")
    print(f"  Exact: {exact_message.tensor.shape}")
    exit(1)

diff = torch.abs(converted_factor.tensor - exact_message.tensor)
rel_diff = diff / (torch.abs(exact_message.tensor) + 1e-12)

print(f"Absolute difference:")
print(f"  Min: {diff.min().item():.6e}")
print(f"  Max: {diff.max().item():.6e}")
print(f"  Mean: {diff.mean().item():.6e}")
print(f"  Median: {diff.median().item():.6e}")

print(f"\nRelative difference:")
print(f"  Mean: {rel_diff.mean().item():.6e}")
print(f"  Median: {rel_diff.median().item():.6e}")
print(f"  Max: {rel_diff.max().item():.6e}")

# Check if they match (should be exact with Memorizer!)
max_abs_diff = diff.max().item()
if max_abs_diff < 1e-5:
    print(f"\n✓ SUCCESS: Conversion is correct! (max diff: {max_abs_diff:.6e})")
else:
    print(f"\n✗ FAILURE: Conversion has errors! (max diff: {max_abs_diff:.6e})")

    # Show worst mismatches
    print(f"\nWorst 10 mismatches:")
    flat_diff = diff.flatten()
    flat_converted = converted_factor.tensor.flatten()
    flat_exact = exact_message.tensor.flatten()

    worst_indices = torch.topk(flat_diff, min(10, len(flat_diff))).indices
    for i, idx in enumerate(worst_indices):
        print(f"  {i+1}. Index {idx.item():6d}: converted={flat_converted[idx].item():.6f}, exact={flat_exact[idx].item():.6f}, diff={flat_diff[idx].item():.6e}")

print("\n" + "="*70)
print("DONE")
print("="*70)
