---
type: concept
title: Loss Functions
created: 2026-03-01
tags: [training, neural-networks, optimization]
---

# Loss Functions

NCE implements a large library of loss functions for training neural network message approximations. The choice of loss function significantly affects approximation quality and training dynamics.

## Core Loss Functions

### Unnormalized KL Divergence (`unnormalized_kl`)
The primary loss function. Measures the KL divergence between the exact and approximate messages in unnormalized log space. Heavily weights high-probability assignments. See [[unnormalized-kl-divergence]].

### Log-Space MSE (`logspace_mse`)
Mean squared error computed directly in log10 space: `mean((f - f_hat)^2)`. Simple and stable but treats all assignments equally regardless of their probability mass.

### Linear-Space MSE (`linspace_mse`)
Converts from log10 to linear space before computing MSE: `mean((10^f - 10^f_hat)^2)`. Strongly emphasizes high-probability assignments.

### From-Logspace MSE (`from_logspace_mse`)
Variant that converts predictions from log to linear space before MSE.

## Weighted Loss Functions

### Weighted Logspace MSE (`weighted_logspace_mse`)
Weights each sample by its backward message value, emphasizing assignments that matter most for the partition function.

### Scaled MSE (`scaled_mse`)
MSE with a learned scaling factor.

## Message Gradient Losses

### Sampled Message Gradient (`mg_sampled`)
Uses backward message values as importance weights for the loss. Key for the research hypothesis that backward-message-weighted training improves approximation.

## Expected Log Probability (ELP) Variants
- `elp_sampled`: Expected log probability with sampling
- `elp_exact`: Exact expected log probability
- Various combinations with other losses

## GIL (Generalized Information Loss) Variants
- `gil1`, `gil1c`, `gil2c`: Different formulations of generalized information loss
- `huber_gil1c`: Huber-smoothed version for stability

## UKF (Unscented Kalman Filter) Loss
- `ukf_sequential`: Treats the NN training as a state estimation problem
- Uses sigma points to propagate uncertainty
- Implemented via helpers in `nce/neural_networks/ukf_helpers.py`

## Combined Losses
- `combined_gil1_ls_mse`: Weighted combination of GIL1 and logspace MSE
- Various other combinations for exploring multi-objective training

## Implementation

All loss functions are defined in `nce/neural_networks/losses.py` (~1200 lines, 30+ functions).

The `Trainer._get_loss_fn()` method in `nce/neural_networks/train.py` dispatches to the appropriate function based on `config['loss_fn']`.

## Related

- [[unnormalized-kl-divergence]]
- [[backward-messages]]
- [[neural-network-factors]]
