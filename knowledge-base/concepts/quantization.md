---
type: concept
title: Quantization (Message Quantization)
created: 2026-03-01
tags: [approximation, neural-networks, optimization]
---

# Quantization (Message Quantization)

Quantization in the NCE context refers to approximating a message (factor) by mapping its values to a smaller set of discrete levels. This reduces the effective information content of the message while preserving its most important structure.

## Optimal K-Segment Quantization

The NCE implementation uses **dynamic programming** to find the optimal K-segment quantization:

1. Sort all factor values.
2. Partition the sorted values into K contiguous segments.
3. Within each segment, use the segment mean as the representative value.
4. The optimal partition minimizes the total squared error.

### SMAWK Optimization

The DP uses a **divide-and-conquer SMAWK** algorithm for efficiency:
- Standard DP for K-segment quantization is O(n^2 * K).
- SMAWK optimization reduces the inner loop by exploiting the concavity of the cost matrix.

## Implementation

In `nce/neural_networks/quantization.py`:

- **`QuantizationSolver`**: Takes sorted values and K (number of segments). Solves via DP.
  - `solve()`: Fills the DP table and finds optimal segment boundaries.
  - `get_quantized_values()`: Maps each original value to its segment mean.

- **`quantize_message(factor, K)`**: Entry point.
  1. Flatten the factor tensor.
  2. Sort values.
  3. Solve for optimal K segments.
  4. Map values back to segment means.
  5. Return a new FastFactor with quantized values.

- **`verify_monotonicity()`**: Sanity check that segment means are monotonically increasing.

## Use Case

Quantization provides a simple, non-parametric approximation alternative to neural networks:
- No training required -- just a DP solve.
- Produces a FastFactor directly (no lazy evaluation needed).
- May be useful for initializing neural network training or as a baseline.
- Could be combined with decision tree approximation.

## Status

Recently implemented. Per `prompt.txt`, the implementation needs review for correctness and testing. Research is needed on:
- How quantization quality compares to NN approximation for different K values.
- Whether quantized messages can serve as good initializations for NN training.
- How quantization interacts with backward message computation.

## Sources

- [[@lloyd1982quantization]] · [[@max1960quantizing]] — the Lloyd–Max MMSE scalar quantizer
  (centroid + nearest-neighbor optimality), the theory behind segment-mean representatives.
- [[@wu1991optimalquant]] — exact DP for optimal 1-D K-segment quantization reduced to O(KN)
  via matrix searching (SMAWK); the algorithm NCE's `QuantizationSolver` implements. (No
  single canonical paper for DP optimal quantization; Wu is the cleanest algorithmic anchor.)

## Related

- [[neural-network-factors]]
- [[factor-operations]]
- [[loss-functions]]
- [[decision-tree-approximation]]
