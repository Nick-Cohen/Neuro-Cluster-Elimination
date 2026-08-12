---
type: concept
title: Clique Tree
status: growing
tags: [inference, graphical-models, terminology]
created: 2026-06-12
updated: 2026-06-12
---

# Clique Tree

**Clique tree** is the term Koller & Friedman ([[@koller2009pgm]]) use for a
**[[junction-tree|junction tree]]** whose clusters are the **maximal cliques** of a chordal
(triangulated) graph, arranged to satisfy the [[running-intersection-property]].

## Key points

- The only substantive distinction from "[[join-tree|join tree]]"/"junction tree" is the
  *maximal cliques* requirement; otherwise interchangeable. See [[terminology-map]].
- Each clique's size minus one lower-bounds the [[induced-width]] achievable for that
  triangulation.
- Sum-product message passing over a clique tree computes all clique marginals and the
  [[partition-function]].

## Sources

- [[@koller2009pgm]] (canonical use of "clique tree").

## Related

- [[junction-tree]] · [[join-tree]] · [[running-intersection-property]] · [[tree-decomposition]]
