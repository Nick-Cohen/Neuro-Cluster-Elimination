---
type: concept
title: Sample Generation
created: 2026-03-01
tags: [training, data, sampling]
---

# Sample Generation

Sample generation is the process of creating training data for neural network message approximation. The `SampleGenerator` class (`nce/sampling/sample_generator.py`) handles this.

## Process

1. **Generate assignments**: Create variable assignment vectors in the message scope.
2. **Compute target values**: Evaluate the exact message at each assignment.
3. **Compute backward values** (optional): Evaluate the backward message for loss weighting.

## Sampling Methods

### Uniform Sampling (`sample_uniform`)
Randomly sample assignments uniformly from the joint state space. Each variable's assignment is drawn independently and uniformly from its domain.

### Exhaustive Enumeration (`sample_all`)
Generate all possible assignments. Only feasible for small message scopes. Used when `sample_method='all'` and the scope is small enough.

### Deterministic Seeding
`_compute_seed()` generates a reproducible random seed from the bucket variable name and scope. This ensures that repeated calls with the same bucket produce the same samples, aiding reproducibility.

## Target Computation

### `compute_message_values(assignments)`
For each assignment, computes the exact message value:
1. Look up each factor's value at the assignment (handling partial overlap via `_get_values`).
2. Multiply all factor values together (sum in log space).
3. Marginalize over the elimination variable (logsumexp).

This is done via `sample_tensor_product_elimination()` which performs:
- For each assignment to the message scope variables: enumerate all states of the elimination variable, evaluate the product of all factors, and take logsumexp.

### `compute_backward_values(assignments)`
For each assignment, evaluates the backward message. Uses the pre-populated backward factors from `get_backward_message()`.

## Data Flow

```
SampleGenerator
  |
  v
DataLoader.load()     -- orchestrates sampling + value computation
  |
  v
DataPreprocessor      -- normalizes for NN training
  |
  v
Trainer._make_dataloader() -- creates PyTorch DataLoader for batched training
```

## Batch Loading

`DataLoader.load_batches()` generates data in batches for memory efficiency. Each batch has `batch_size` samples. Multiple batches can be loaded and optionally shuffled across batches.

## Related

- [[neural-network-factors]]
- [[factor-operations]]
- [[variable-elimination]]
- [[backward-messages]]
