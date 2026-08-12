---
type: concept
title: Induced Width (Treewidth)
status: growing
tags: [inference, complexity, graphical-models, foundations]
created: 2026-06-12
updated: 2026-06-12
---

# Induced Width (Treewidth)

The **induced width** of a graph along an [[elimination-ordering|elimination order]] is
the maximum number of earlier-neighbors any variable has when eliminated, after adding
"fill" edges created by elimination. The minimum induced width over all orders is the
graph's **treewidth**. It is the single parameter that governs exact-inference cost.

## Key points

- [[bucket-elimination]] / [[variable-elimination]] run in time and space
  **exponential in the induced width** — small width = tractable, large width = infeasible.
- Finding the minimum-width order is NP-hard; in practice we use heuristics such as
  **weighted min-fill** ([[elimination-ordering]]).
- "Induced width", "treewidth", and "(max clique size − 1) of the chordal completion"
  coincide; a [[tree-decomposition]] of width $w$ exists iff treewidth $\le w$.
- The [[iB-parameter|i-bound]] of [[mini-bucket-elimination]] caps the *effective* width
  at $i$, trading exactness for cost exponential only in $i$.

## In NCE

- A bucket's width = size of its message scope plus eliminated variables; the per-bucket
  exact cost ≈ $2^{\text{width}}$ for binary domains (more generally the product of domain
  sizes). NCE neural-approximates exactly those buckets whose width/table exceeds the
  exact limits.
- [[bucket-merging]] deliberately raises a cluster's local width (up to the
  [[merge-bound]]) to reduce the *number* of approximated buckets.

## Sources

- [[@dechter1999bucket]] · [[@kask2005unifying]] (tree decompositions) · [[@koller2009pgm]].

## Related

- [[elimination-ordering]] · [[tree-decomposition]] · [[bucket-elimination]] · [[merge-bound]]
