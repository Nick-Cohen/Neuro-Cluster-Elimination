---
type: concept
title: Tree Decomposition
status: growing
tags: [inference, complexity, graphical-models, foundations]
created: 2026-06-12
updated: 2026-06-12
---

# Tree Decomposition

A **tree decomposition** maps a graph onto a tree of clusters (bags) such that every edge
is inside some cluster and the clusters satisfy the [[running-intersection-property]]. Its
**width** is the largest cluster size minus one; the minimum over all decompositions is the
treewidth ([[induced-width]]). It is the unifying object behind
[[junction-tree|junction trees]], bucket trees, and cluster-tree elimination.

## Key points

- Many inference schemes are instances of message passing on a tree decomposition:
  [[bucket-elimination]] (bucket tree), junction-tree propagation, cluster-tree elimination
  (CTE), and — with bounded cluster size $i$ — [[mini-bucket-elimination]] and
  [[iterative-join-graph-propagation]].
- Kask, Dechter, Larrosa & Dechter ([[@kask2005unifying]]) unify these and introduce the
  super-cluster machinery underlying [[super-bucket|super-buckets]].
- Exact cost is exponential in the width; bounding the width by $i$ is the central
  approximation lever in this whole family.

## In NCE

- [[bucket-merging]] re-partitions the bucket tree into coarser clusters (up to the
  [[merge-bound]]), i.e. it chooses a different tree decomposition that trades a few wider
  exact clusters for fewer neural approximations.

## Sources

- [[@kask2005unifying]] · [[@dechter1999bucket]] · [[@koller2009pgm]].

## Related

- [[induced-width]] · [[junction-tree]] · [[super-bucket]] · [[bucket-merging]]
