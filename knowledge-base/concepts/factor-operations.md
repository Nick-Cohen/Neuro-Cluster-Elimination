---
type: concept
title: Factor Operations (Log-Space)
created: 2026-03-01
tags: [inference, data-structure, graphical-models]
---

# Factor Operations (Log-Space)

The `FastFactor` class is the core data structure in NCE. It represents a factor (potential function) in a graphical model as a labeled tensor, with all values stored in **log10 space**.

## Data Structure

A `FastFactor` has two components:
- `tensor` (torch.Tensor): Multi-dimensional array of values in log10 space.
- `labels` (list[str]): Variable names corresponding to each tensor dimension.

Example: A factor over variables `[X0, X1]` where each has 3 states is a 3x3 tensor with `labels = ['X0', 'X1']`.

## Log-Space Convention

All factor values are in log10 space. This means:
- **Factor product** = element-wise **addition** of tensors (after dimension alignment)
- **Factor division** = element-wise **subtraction**
- **Marginalization** (summing out a variable) = **logsumexp** along the corresponding dimension
- **Partition function** = logsumexp over all dimensions

This convention avoids numerical underflow/overflow that would occur with very small/large probability values.

## Key Operations

### Multiplication (`__mul__`)
Aligns tensor dimensions by label matching, broadcasts, and adds values. This is the most common operation -- used to combine factors in a bucket before elimination.

### Elimination (`eliminate`)
Removes a variable by applying logsumexp along its dimension. Custom implementation using `_log10_sum_exp` to handle log10 space (torch's built-in logsumexp assumes natural log).

### Batch Value Lookup (`_get_slices`, `_get_values`)
Given a batch of variable assignments, returns the corresponding factor values. This is the workhorse for training data generation:
- `_get_slices`: For assignments that match the factor's scope exactly or are a subset.
- `_get_values`: More general, handles partial overlap and marginalization.

### Dimension Management
- `order_indices(order)`: Permute dimensions to match a given label order.
- `shuffle(other)`: Align dimensions to match another factor's labels.
- `inverse()`: Negate all values (multiplicative inverse in log space).

## Special Variants

- **FactorNN** (`factor_nn.py`): A "lazy" factor backed by a neural network. Does not store a tensor -- computes values on demand via NN queries. Can be materialized to a regular FastFactor via `to_exact()`.
- **Scalar factors**: Factors with empty `labels` list. Represent constants (e.g., accumulated partition function contributions).

## Implementation Files

- `nce/inference/factor.py` -- FastFactor class
- `nce/inference/factor_nn.py` -- FactorNN subclass

## Related

- [[variable-elimination]]
- [[neural-network-factors]]
- [[log-space-convention]]
