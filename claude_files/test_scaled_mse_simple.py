"""
Simple test: Run inference with scaled_mse and check if it works
"""

import torch
from nce.inference import FastGM
from nce.problems import test_problems

device = 'cuda'

gm_config = {
    'device': device,
    'hidden_sizes': [],
    'optimizer': 'sgd',
    'num_epochs': 50,
    'traced_losses': [],
    'val_set': 'all',
    'lr': 1e-3,
    'sampling_scheme': 'all',
    'num_batches_per_set': 1,
    'batch_size': 256,
    'set_size': 10240,
    'num_samples': 147000*20,
    'seed': 4,
    'fdb': True,
    'lower_dim': False,
    'gather_message_stats': False,
    'ecl': 1000,  # Force training for buckets > 1000 states
    'iB': 100,
    'loss_fn': 'scaled_mse',
}

print("="*70)
print("SIMPLE TEST: scaled_mse inference")
print("="*70)

probs = [test_problems[k] for k in list(test_problems.keys())]
probs = [p for p in probs if p.is_10_7_benchmark]
prob = probs[0]  # grid10x10.f10.wrap

print(f"\nProblem: {prob.name}")
print(f"True Z: {prob.Z}")

# Run inference
fastgm = FastGM(uai_file=prob.uai_file, nn_config=gm_config, device=device)
Z_hat = fastgm.get_log_partition_function()

print(f"\nZ_hat: {Z_hat}")
print(f"Z_true: {prob.Z}")
print(f"Abs Error: {abs(Z_hat - prob.Z)}")
print(f"Num trained buckets: {fastgm.num_trained}")

print("\n" + "="*70)
print("CHECKING FIRST TRAINED BUCKET")
print("="*70)

# Find a trained bucket
for bucket in fastgm.ordering:
    if hasattr(bucket, 'factors'):
        # Check if any factor is a FactorNN
        for factor in bucket.factors:
            if factor.is_nn:
                print(f"\nBucket {bucket.label} has NN factor")
                print(f"Checking data_preprocessor attributes:")
                data_proc = factor.data_processor
                print(f"  exp_preprocessing: {data_proc.exp_preprocessing}")
                print(f"  scaling_factor: {data_proc.scaling_factor}")
                print(f"  y_max_after_scaling: {data_proc.y_max_after_scaling}")

                # Try converting to exact
                try:
                    exact_factor = factor.to_exact()
                    print(f"✓ Conversion successful")
                    print(f"  Has NaNs: {exact_factor.tensor.isnan().any()}")
                    print(f"  Min: {exact_factor.tensor.min().item():.6f}")
                    print(f"  Max: {exact_factor.tensor.max().item():.6f}")
                except Exception as e:
                    print(f"✗ Conversion failed: {e}")
                break
        break

print("\nDone!")
