---
type: pattern
title: Data Normalization Pipeline
related: [[neural-network-factors]], [[loss-functions]], [[three-layer-architecture]]
updated: 2026-03-01
---

# Data Normalization Pipeline

## Overview

NCE uses a two-class normalization system (DataPreprocessor and DataPreprocessor_old) to transform message values from log10 space into a form suitable for neural network training.

## Current System (DataPreprocessor)

The new preprocessor converts all values to natural log space and centers using a normalizing constant:

- **Without backward messages:** `normalizing_constant = logsumexp(y)` (effectively the weighted mean)
- **With backward messages:** `normalizing_constant = logsumexp(y + bw) - logsumexp(bw)` (weighted mean of y, weighted by exp(bw))

This ensures that a zero-output NN produces the constant estimator, giving near-zero log Z error.

## Old System (DataPreprocessor_old)

The old preprocessor supports multiple normalization modes:
- **Max-based normalization:** Subtract max value, convert to natural log
- **FDB (mean-based) normalization:** Subtract mean instead of max
- **Exponential preprocessing:** For scaled_mse/linspace_mse losses
- **Linear space normalization:** Convert to linear space, standardize

## Key Design Constraint

The normalizing constant MUST be computed ONCE from all training data and reused for all batches. Using per-batch normalization causes gradient inconsistency and training divergence.

## Undo Normalization

Both classes provide `undo_normalization()` to convert NN outputs back to log10 space. This is called by `FactorNN._get_slices()` and `FactorNN.to_exact()` during inference.
