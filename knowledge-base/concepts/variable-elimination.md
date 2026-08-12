---
type: concept
title: Variable Elimination
created: 2026-03-01
tags: [inference, exact, graphical-models]
---

# Variable Elimination

Variable elimination (VE) is the fundamental exact inference algorithm for probabilistic graphical models. It computes marginals, MAP assignments, or the partition function by systematically summing out (eliminating) variables one at a time according to a specified elimination order.

## How It Works

1. Assign factors to **buckets** based on the elimination order. Each bucket corresponds to a variable to be eliminated.
2. For each bucket (in elimination order):
   - Multiply all factors in the bucket together (in log-space: sum them).
   - Sum out (marginalize) the bucket's variable from the product.
   - The result is a **message** -- a new factor over the remaining variables.
   - Route this message to the next appropriate bucket.
3. After all variables are eliminated, the remaining scalar is the log partition function.

## In NCE

- Implemented in `FastGM.eliminate_variables()` and `FastGM.process_bucket()` in `nce/inference/graphical_model.py`.
- Each bucket is a `FastBucket` object (`nce/inference/bucket.py`).
- The elimination decision depends on the **exact complexity (EC)** of the bucket:
  - If `EC <= ecl` (exact complexity limit): use exact elimination via `FastBucket.compute_message_exact()`
  - If `EC > ecl`: use approximate elimination (neural network, decision tree, or WMB)
- The **induced width** of a bucket is the number of variables in its message scope. The maximum induced width across all buckets determines the computational complexity.

## Key Quantities

- **Elimination Order**: Sequence in which variables are eliminated. Determined by the `wtminfill_order` heuristic (weighted min-fill) in `nce/inference/elimination_order.py`.
- **Message Scope**: The variables remaining in a message after elimination (scope of the product minus the eliminated variable).
- **Exact Complexity (EC)**: Product of state counts of variables in the bucket scope. Determines whether exact computation is feasible.
- **ECL (Exact Complexity Limit)**: Threshold above which approximation is used instead of exact computation.

## Sources

- [[@zhang1996exploiting]] (VE for Bayesian networks; conf. version `zhang1994simple`).
- [[@dechter1999bucket]] — VE organized as [[bucket-elimination]] (same algorithm).
- Textbooks: [[@koller2009pgm]], [[@darwiche2009modeling]].

> Terminology: NCE's "EC / ecl" are project-specific names; see [[terminology-map]].
> "Induced width" is the standard term for the complexity parameter — see [[induced-width]].

## Related

- [[bucket-elimination]] · [[induced-width]] · [[elimination-ordering]]
- [[weighted-mini-bucket]] · [[mini-bucket-elimination]]
- [[factor-operations]] · [[neural-network-factors]] · [[backward-messages]]
- [[partition-function]] · [[discrete-graphical-model]]
