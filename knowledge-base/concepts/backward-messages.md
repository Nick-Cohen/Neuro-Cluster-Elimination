---
type: concept
title: Backward Messages (Message Gradients)
created: 2026-03-01
tags: [inference, training, loss-weighting, graphical-models]
---

# Backward Messages (Message Gradients)

Backward messages represent the **downstream impact** of a bucket's message on the overall partition function. They are the product of all factors that appear in buckets processed *after* the current bucket in the elimination order. In optimization terms, they function as the "gradient" or "sensitivity" of the partition function with respect to the message at a given bucket.

## Intuition

In variable elimination, each bucket produces a message that flows forward. The backward message for a bucket captures "how important is this message to the final answer?" It is the product of all factors downstream of the current bucket.

- If the backward message has high variance, the bucket's message strongly affects the partition function.
- If it is nearly uniform, the bucket's contribution is less critical.

## Formal Definition

For a bucket eliminating variable `X_i`:
- **Forward message**: The result of multiplying all factors in the bucket and summing out `X_i`.
- **Backward message**: The product of all factors from buckets that come *after* `X_i` in the elimination order, marginalized down to the scope of the forward message.

The backward message has the same scope as the forward message (the message scope of bucket `X_i`).

## In NCE

Implemented in `nce/utils/backward_message.py`:
- `get_backward_message(gm, bucket_var, ...)`: Main function.
- `_get_backward_factors(gm, bucket_var, ...)`: Collects downstream factors.

### Computation Steps

1. Collect all factors from downstream buckets (everything after `bucket_var` in the elimination order).
2. Create a **downstream FastGM** from these factors.
3. Compute a new elimination order for the downstream factors using `wtminfill_order`, preserving the message scope variables.
4. Eliminate all variables in the downstream GM **except** the message scope variables.
5. The remaining factor is the backward message.

### Approximation

When the downstream GM is too large for exact computation:
- WMB is used with `backward_ecl` and `iB` parameters.
- The `approximation_method` parameter can override the default method.
- A dedicated downstream GM config is created with `populate_bw_factors=False` to avoid recursion.

### Modes

- **Standard mode** (`return_factor_list=False`): Returns the full backward message as a single `FastFactor`.
- **Batched mode** (`return_factor_list=True`): Returns a list of factors (avoids materializing the full product, used for batched learning).
- **Partition tracking** (`return_partitions=True`): Also returns the number of WMB partitions used.

## Role in Training

Backward messages are used to **weight the training loss** for neural network approximation:
- The hypothesis is that using backward messages to weight training samples improves approximation quality.
- Better backward message approximations (exact > WMB with moment matching > WMB > none) should lead to better trained neural networks.
- This is the core research hypothesis of the NCE project.

## Historical Note

Previously called "message gradients" in the codebase. The alias `get_message_gradient` is maintained in `backward_message.py` for backward compatibility. The old verbose implementation lives in `nce/utils/message_gradient.py` (superseded, see dead code report).

## Related

- [[variable-elimination]]
- [[weighted-mini-bucket]]
- [[unnormalized-kl-divergence]]
- [[loss-functions]]
- [[neural-network-factors]]
