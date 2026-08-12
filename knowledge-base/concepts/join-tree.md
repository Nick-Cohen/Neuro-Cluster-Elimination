---
type: concept
title: Join Tree
status: growing
tags: [inference, graphical-models, terminology]
created: 2026-06-12
updated: 2026-06-12
---

# Join Tree

**Join tree** is another name for a **[[junction-tree|junction tree]]**: a tree of
variable clusters satisfying the **[[running-intersection-property]]** (the "join"/junction
property). The term comes from the relational-database / acyclic-hypergraph tradition
(Beeri–Fagin–Maier–Yannakakis) and the Shenoy–Shafer propagation lineage, whereas
"junction tree" comes from the Lauritzen–Spiegelhalter/HUGIN lineage.

## Key points

- **Join tree ≈ junction tree ≈ [[clique-tree|clique tree]]** — same structure, different
  communities. A clique tree specifically uses *maximal* cliques of the chordal graph; a
  join/junction tree may use any clusters with running intersection. See [[terminology-map]].
- A join tree of width $w$ is a [[tree-decomposition]] of width $w$.
- Darwiche's textbook ([[@darwiche2009modeling]]) writes "jointree".

## In NCE

- NCE's bucket tree is a join tree; the subsumption merge `merge_join_tree` is named for
  this condition. See [[bucket-merging]], [[junction-tree]].

## Sources

- [[@lauritzen1988local]] · [[@darwiche2009modeling]] · [[@kask2005unifying]].

## Related

- [[junction-tree]] · [[clique-tree]] · [[running-intersection-property]] · [[tree-decomposition]]
