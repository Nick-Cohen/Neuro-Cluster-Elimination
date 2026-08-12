---
type: pattern
title: One-Hot Encoding Convention
related: [[neural-network-factors]], [[factor-operations]]
updated: 2026-03-01
---

# One-Hot Encoding Convention

## Two Modes

NCE supports two one-hot encoding modes controlled by the `lower_dim` flag on the graphical model:

### Lower-dim mode (n-1 encoding, `lower_dim=True`)
- Drops the first category from each variable's one-hot vector
- A variable with k states gets a (k-1)-dimensional encoding
- The first state (0) is represented by all zeros
- This is the traditional "dummy variable" encoding

### Full mode (`lower_dim=False`)
- Includes all categories in the one-hot encoding
- A variable with k states gets a k-dimensional encoding
- More parameters but avoids the reference-category asymmetry

## Implementation Locations

One-hot encoding happens in three places (which must be kept in sync):
1. `DataPreprocessor.one_hot_encode()` -- for training data
2. `FactorNN._get_slices()` -- for inference queries
3. `FactorNN.nn_to_FastFactor()` -- for materializing NN to tensor

All three check `gm.lower_dim` to determine which mode to use.

## Design Note

The `lower_dim` flag is set on the `FastGM` instance and propagates through bucket, data loader, and preprocessor. Changing it mid-experiment would cause dimension mismatches.
