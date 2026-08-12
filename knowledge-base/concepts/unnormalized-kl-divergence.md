---
type: concept
title: Unnormalized KL Divergence
created: 2026-03-01
tags: [loss-function, training, information-theory]
---

# Unnormalized KL Divergence

The unnormalized KL divergence is a key loss function in the NCE project for training neural network approximations of bucket messages. Unlike standard KL divergence which operates on normalized probability distributions, the unnormalized variant works directly with unnormalized log-space factor values.

## Definition

For exact message values `f(x)` and approximate values `f_hat(x)` (both in log10 space), the unnormalized KL divergence is:

```
UKL = sum_x [ 10^f(x) * (f(x) - f_hat(x)) ] - (Z - Z_hat)
```

where:
- `Z = sum_x 10^f(x)` is the partition function of the exact message
- `Z_hat = sum_x 10^f_hat(x)` is the partition function of the approximate message
- The sum is over all assignments `x` in the message scope

## Properties

- Always non-negative (UKL >= 0)
- Equals zero if and only if `f(x) = f_hat(x)` for all `x`
- Does not require normalization of the factors
- Naturally handles log-space representations
- Sensitive to errors on high-probability assignments (the `10^f(x)` weighting)

## In NCE

Implemented in `nce/neural_networks/losses.py` as `unnormalized_kl()`.

The loss function is one of many options available through the Trainer's `_get_loss_fn()` dispatch in `nce/neural_networks/train.py`. It is selected via `config['loss_fn'] = 'unnormalized_kl'`.

## Why Unnormalized KL?

The goal query in the NCE project is to minimize error to the **log partition function**. The unnormalized KL directly measures how much the approximation affects the partition function computation:
- Errors in high-probability regions are penalized more heavily (they contribute more to the partition function).
- The partition function gap `(Z - Z_hat)` ensures that the total "mass" is preserved.

This makes UKL a natural objective when the downstream task is partition function estimation.

## Related

- [[loss-functions]]
- [[backward-messages]]
- [[neural-network-factors]]
- [[variable-elimination]]
