---
type: pattern
title: Log-Space Convention
created: 2026-03-01
tags: [convention, implementation]
---

# Pattern: Log-Space Convention

## Description

All factor values throughout the NCE codebase are stored in **log10 space**. This is a fundamental design decision that affects every computation in the system.

Consequences:
- Factor multiplication becomes tensor addition: `(f1 * f2).tensor = f1.tensor + f2.tensor`
- Factor division becomes tensor subtraction
- Marginalization (summing out a variable) uses logsumexp: `log10(sum(10^values))`
- The partition function is computed as logsumexp over all dimensions
- Neural networks predict log10 values, not linear probabilities
- Loss functions must account for log-space representation

## Example

```python
# Factor multiplication (in FastFactor.__mul__)
# Instead of: result = f1 * f2 (element-wise multiplication)
# We do:      result = f1 + f2 (element-wise addition in log space)

# Marginalization (in FastFactor.eliminate)
# Instead of: result = sum(f, axis=var_dim)
# We do:      result = logsumexp(f, axis=var_dim)  # log10(sum(10^f))

# Custom log10 logsumexp:
def _log10_sum_exp(tensor, dim):
    max_val = tensor.max(dim=dim, keepdim=True).values
    return max_val.squeeze(dim) + torch.log10(torch.sum(10**(tensor - max_val), dim=dim))
```
