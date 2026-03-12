#!/usr/bin/env python
"""
Simple test for elp_least_squares loss function.
"""

import torch
import numpy as np
from nce.neural_networks.losses import elp_least_squares, unnormalized_kl, mg_sampled_loss_fdb

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Create synthetic test data
torch.manual_seed(42)
batch_size = 100

# Create target message values (in log space)
targets = torch.randn(batch_size, device=device) * 2.0

# Create outputs that are slightly perturbed from targets
outputs = targets + torch.randn(batch_size, device=device) * 0.5

# Synthetic statistics
sigma_f = 2.0
sigma_g = 1.5
rho = 0.7
num_bw_samples = 50

print("\nTesting loss functions with synthetic data:")
print(f"Batch size: {batch_size}")
print(f"sigma_f: {sigma_f}, sigma_g: {sigma_g}, rho: {rho}")
print(f"num_bw_samples: {num_bw_samples}")

# Test elp_least_squares
print("\n" + "="*60)
print("Testing elp_least_squares")
print("="*60)
try:
    loss_elp = elp_least_squares(outputs, targets, None, sigma_f, sigma_g, rho, num_bw_samples)
    print(f"Loss value: {loss_elp.item():.6f}")
    print(f"Loss shape: {loss_elp.shape}")
    print(f"Loss requires grad: {loss_elp.requires_grad}")

    # Test gradient computation
    outputs_grad = outputs.clone().requires_grad_(True)
    loss_elp_grad = elp_least_squares(outputs_grad, targets, None, sigma_f, sigma_g, rho, num_bw_samples)
    loss_elp_grad.backward()
    print(f"Gradient computed successfully")
    print(f"Gradient norm: {outputs_grad.grad.norm().item():.6f}")
    print("✓ elp_least_squares PASSED")
except Exception as e:
    print(f"✗ elp_least_squares FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test mg_sampled_loss_fdb for comparison
print("\n" + "="*60)
print("Testing mg_sampled_loss_fdb (for comparison)")
print("="*60)
try:
    loss_mg = mg_sampled_loss_fdb(outputs, targets, None, sigma_f, sigma_g, rho, num_bw_samples)
    print(f"Loss value: {loss_mg.item():.6f}")
    print("✓ mg_sampled_loss_fdb PASSED")
except Exception as e:
    print(f"✗ mg_sampled_loss_fdb FAILED: {e}")

# Test unnormalized_kl for comparison
print("\n" + "="*60)
print("Testing unnormalized_kl (for comparison)")
print("="*60)
try:
    loss_kl = unnormalized_kl(outputs, targets, None)
    print(f"Loss value: {loss_kl.item():.6f}")
    print("✓ unnormalized_kl PASSED")
except Exception as e:
    print(f"✗ unnormalized_kl FAILED: {e}")

# Compare loss values
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"elp_least_squares:     {loss_elp.item():.6f}")
print(f"mg_sampled_loss_fdb:   {loss_mg.item():.6f}")
print(f"unnormalized_kl:       {loss_kl.item():.6f}")

# Test reproducibility with seed
print("\n" + "="*60)
print("Testing reproducibility with seed")
print("="*60)
loss1 = elp_least_squares(outputs, targets, None, sigma_f, sigma_g, rho, num_bw_samples, seed=123)
loss2 = elp_least_squares(outputs, targets, None, sigma_f, sigma_g, rho, num_bw_samples, seed=123)
loss3 = elp_least_squares(outputs, targets, None, sigma_f, sigma_g, rho, num_bw_samples, seed=456)

print(f"Loss with seed=123 (run 1): {loss1.item():.6f}")
print(f"Loss with seed=123 (run 2): {loss2.item():.6f}")
print(f"Loss with seed=456:         {loss3.item():.6f}")

if torch.allclose(loss1, loss2):
    print("✓ Reproducibility test PASSED (same seed gives same result)")
else:
    print("✗ Reproducibility test FAILED")

if not torch.allclose(loss1, loss3):
    print("✓ Different seeds give different results (as expected)")
else:
    print("✗ Different seeds should give different results")

print("\n" + "="*60)
print("ALL TESTS COMPLETED")
print("="*60)
