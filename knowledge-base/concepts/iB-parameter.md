---
type: concept
title: iB Parameter (i-bound / Mini-Bucket Width)
created: 2026-03-01
tags: [inference, configuration, tradeoff, wmb, mini-bucket]
---

# iB Parameter (i-bound / Mini-Bucket Width)

The iB parameter (also called the i-bound or induced width bound) is the most important configuration knob in WMB elimination. It directly controls the **memory/accuracy tradeoff** in approximate inference.

## Definition

`iB` is the maximum number of variables allowed in the scope of any single mini-bucket during WMB partitioning. It bounds the width of each mini-bucket.

- `iB = 0`: No WMB (all exact, or full WMB with zero width = uniform messages).
- Small `iB`: Many mini-buckets per bucket, each very small. Fast but loose approximation.
- Large `iB`: Fewer mini-buckets per bucket, closer to exact. Slow but tight approximation.
- `iB >= induced_width`: No partitioning needed; reduces to exact elimination.

## Effect on Computation

### Memory
Each mini-bucket produces a message over its scope. The largest mini-bucket has at most `iB` variables and thus at most `d^iB` entries (where `d` is the average domain size). Memory is exponential in `iB`.

### Accuracy
With larger `iB`, mini-buckets can be larger and thus more accurate (fewer partitions, less approximation error). The bound on the partition function tightens as `iB` increases.

### WMB Partitioning
`FastBucket._get_wmb_partitions(iB)` greedily assigns factors to mini-buckets:
1. Start with an empty set of mini-buckets.
2. For each factor, try to add it to an existing mini-bucket whose scope union would not exceed `iB` variables.
3. If no existing mini-bucket can accommodate it, create a new mini-bucket.
This is a greedy first-fit algorithm — not necessarily optimal.

## In NCE Configuration

```python
config = {
    'iB': 4,           # Forward mini-bucket width
    'backward_iB': 4,  # Backward mini-bucket width (optional, defaults to iB)
    'ecl': 2**20,      # Exact complexity limit
    'bw_ecl': 2**16,   # Backward exact complexity limit
}
```

- `iB` controls forward WMB (for NN training with `approximation_method='wmb'`).
- `backward_iB` controls backward message WMB approximation (falls back to `iB` if not set).
- `bw_ecl` controls when WMB is applied to backward messages (vs. exact backward messages).

## Relationship to ECL

The ECL (exact complexity limit) and iB interact:
- A bucket with scope size <= ECL → exact computation regardless of iB.
- A bucket with scope size > ECL → use approximation. If `approximation_method='wmb'`, WMB with the given `iB` is used.
- Setting `iB` to the full induced width and `ECL = infinity` gives exact VE.
- Setting `iB = 1` gives the loosest possible WMB bound (all factors partitioned individually).

## Typical Values

For the NCE research workflow:
- Small test problems: `iB = 4-8` is often sufficient for meaningful approximations.
- Medium problems: `iB = 10-15` gives better bounds.
- Backward messages: typically use a smaller `backward_iB` (e.g., 4-8) since backward computation is more expensive.

## Backward Sensitivity and iB

The `compute_sensitivity_sweep()` function in `nce/utils/backward_sensitivity.py` sweeps over `bw_ecl` values (which implicitly set `bw_iB = log2(bw_ecl)`) to measure how much the backward message approximation degrades as `iB` decreases. This helps determine the minimum `iB` needed for backward messages to be useful.

## Related

- [[weighted-mini-bucket]]
- [[bucket-structure]]
- [[variable-elimination]]
- [[backward-messages]]
- [[backward-sensitivity]]
