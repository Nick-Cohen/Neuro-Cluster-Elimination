---
type: concept
title: Junction Tree
status: growing
tags: [inference, exact, graphical-models, foundations]
created: 2026-06-12
updated: 2026-06-12
---

# Junction Tree

A **junction tree** is a tree whose nodes are *clusters* (subsets) of variables that
(1) cover every factor's scope and (2) satisfy the **[[running-intersection-property]]**:
for any variable, the clusters containing it form a connected subtree. Exact inference
runs as two-pass message passing (collect / distribute) over this tree; each edge carries
a **separator** (the intersection of its two clusters). It is the clustering dual of
[[bucket-elimination]] and the basis of the HUGIN expert-system architecture
([[@lauritzen1988local]]).

## Key points

- Built by **moralizing** then **triangulating** (chordal completion) the graph and
  taking a [[clique-tree|clique tree]] of the result; cluster size − 1 = [[induced-width]].
- Cost is exponential in the largest cluster, i.e. in treewidth.
- "Junction tree", "[[join-tree|join tree]]", and "[[clique-tree|clique tree]]" are
  largely synonymous — see [[terminology-map]] for the community conventions.
- Two classical propagation architectures: **Shenoy–Shafer** (message = combine then
  marginalize) and **HUGIN** (potentials on clusters/separators with division). See
  [[@lauritzen1988local]].

## In NCE

- NCE uses the **bucket-tree** form of this idea, not explicit clique propagation.
  Subsumption-based [[bucket-merging]] (`merge_join_tree`) absorbs a child bucket whose
  scope contains its parent's — exactly the running-intersection / join-tree condition —
  so merging is "free" (adds no new variable). See [[bucket-merging]].

## Sources

- [[@lauritzen1988local]] (foundational; with the Shenoy–Shafer 1990 and
  Jensen–Lauritzen–Olesen 1990 architecture papers) · [[@koller2009pgm]] ("clique tree").

## Related

- [[join-tree]] · [[clique-tree]] · [[running-intersection-property]] ·
  [[tree-decomposition]] · [[bucket-elimination]]
