---
type: concept
title: Backward Sensitivity Analysis
created: 2026-03-01
tags: [analysis, backward-messages, wmb, evaluation]
---

# Backward Sensitivity Analysis

Backward sensitivity analysis measures how much the quality of WMB backward message approximation matters for a given bucket. It provides a principled way to determine the minimum `bw_ecl` (and corresponding `backward_iB`) needed for backward messages to be useful.

## Definition

The sensitivity of a bucket is:

```
sensitivity = lse(f + b_wmb) - lse(b_wmb) - lse(f + b_exact) + lse(b_exact)
```

Where:
- `f` = exact forward message
- `b_wmb` = WMB approximate backward message at a given `bw_ecl`
- `b_exact` = exact backward message
- `lse` = log-sum-exp (i.e., log partition function of the argument)

This measures the **error in the weighted partition function** caused by using an approximate backward message instead of the exact one. It is zero when `b_wmb = b_exact` (exact backward message), and larger when the approximation is poor.

## Implementation

In `nce/utils/backward_sensitivity.py`:

- **`compute_bucket_sensitivity(forward_message, backward_wmb, backward_exact)`**: Computes the sensitivity formula for a single bucket.
- **`compute_sensitivity_sweep(model, bw_ecl_values, ...)`**: Sweeps multiple `bw_ecl` values and returns a per-bucket sensitivity dictionary.
- **`compute_sensitivity_sweep_efficient(model, bw_ecl_values, ...)`**: More efficient version: processes each bucket once through all `bw_ecl` values (avoids re-running VE per bw_ecl per bucket).
- **`compute_total_sensitivity(model, bw_ecl, ...)`**: Returns sum of absolute sensitivities across all buckets at a given `bw_ecl`.

## How to Use

```python
from nce.utils.backward_sensitivity import compute_sensitivity_sweep_efficient

results = compute_sensitivity_sweep_efficient(
    model,
    bw_ecl_values=[0] + [2**i for i in range(1, 20)],
    verbose=True
)

# results: {bucket_label: {bw_ecl: sensitivity_value}}
```

## Early Exit Logic

Both sweep functions implement **convergence detection**:
- If `|sensitivity| < zero_threshold`: mark bucket as converged (sensitivity is effectively zero for all higher bw_ecl values).
- If the change across consecutive bw_ecl values is smaller than `plateau_threshold` for `plateau_count` steps: mark as plateaued.
- This can significantly speed up the sweep for buckets whose backward messages are not sensitive to the approximation quality.

## Interpretation

- **Zero sensitivity**: The backward message does not matter for this bucket — using a uniform (or even zero) backward message has the same effect as the exact one. No benefit from improving `bw_ecl` for this bucket.
- **High sensitivity**: The backward message strongly influences the loss weighting. Better backward message approximation will improve training for this bucket.
- **Convergence at low bw_ecl**: The backward message is relatively insensitive; a coarse WMB approximation suffices.
- **No convergence**: The bucket requires a high-quality backward message; use large `bw_ecl` or exact backward messages.

## Practical Use

Sensitivity sweeps help:
1. Identify which buckets benefit most from backward message improvements.
2. Choose the minimum `bw_ecl` for a given accuracy requirement.
3. Understand how much exact vs. WMB backward messages differ across a model's structure.

## Related

- [[backward-messages]]
- [[weighted-mini-bucket]]
- [[iB-parameter]]
- [[variable-elimination]]
