---
type: concept
title: Iterative Join-Graph Propagation (IJGP)
status: growing
tags: [inference, approximate, graphical-models]
created: 2026-06-12
updated: 2026-06-12
---

# Iterative Join-Graph Propagation (IJGP)

**IJGP** runs [[junction-tree|junction-tree]]-style message passing on a **join-*graph***
(a join tree with extra cycles), **iteratively**. It blends two approximation ideas:
bounded cluster size (the [[iB-parameter|i-bound]], from mini-clustering /
[[mini-bucket-elimination]]) and iteration (loopy / generalized belief propagation). The
i-bound interpolates between cheap, loose mini-clustering and exact join-tree inference
([[@mateescu2010jgp]]).

## Key points

- Sits in the **Generalized Belief Propagation** family; messages are passed until
  convergence rather than in a single sweep.
- Same accuracy/cost knob (`i`) as [[weighted-mini-bucket]], but trades the WMB *bound*
  guarantee for iterative refinement on a graph.
- Origin: Dechter, Kask & Mateescu, IJGP, UAI 2002 (`dechter2002ijgp`); definitive
  treatment: [[@mateescu2010jgp]] (JAIR 2010, Gogate added).

## Relation to NCE

- Another classical approximate-inference family for the related-work map; NCE uses
  single-pass elimination with learned messages, not iterative graph propagation.

## Sources

- [[@mateescu2010jgp]].

## Related

- [[mini-bucket-elimination]] · [[weighted-mini-bucket]] · [[junction-tree]] · [[tree-decomposition]]
