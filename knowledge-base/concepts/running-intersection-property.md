---
type: concept
title: Running Intersection Property
status: growing
tags: [inference, graphical-models, foundations]
created: 2026-06-12
updated: 2026-06-12
---

# Running Intersection Property

The **running intersection property (RIP)** is the defining condition of a
[[junction-tree|junction]]/[[join-tree|join]] tree: for every variable $X$, the set of
clusters that contain $X$ forms a **connected subtree**. Equivalently, if $X$ is in two
clusters, it is in every cluster on the path between them.

## Key points

- RIP is what makes local message passing globally correct — a variable's information can
  never "skip" across a gap in the tree.
- A tree of clusters satisfies RIP **iff** it is a valid [[tree-decomposition]] of the graph.
- The separator on an edge is the intersection of its two clusters; RIP guarantees each
  variable's separators are consistent along its subtree.

## In NCE

- The **subsumption** merge condition — a child bucket whose elimination-time scope
  *contains* its parent's — is exactly a RIP/join-tree condition, which is why such merges
  add no new variables to a cluster (they are "free"). See [[bucket-merging]].

## Sources

- [[@lauritzen1988local]] · [[@koller2009pgm]] · [[@kask2005unifying]].

## Related

- [[junction-tree]] · [[join-tree]] · [[clique-tree]] · [[tree-decomposition]] · [[bucket-merging]]
