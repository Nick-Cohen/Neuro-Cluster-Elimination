---
type: moc
title: MOC — Exact Inference
status: evergreen
tags: [moc, exact, foundations]
created: 2026-06-12
updated: 2026-06-12
---

# MOC — Exact Inference

The exact backbone NCE accelerates. Reading order top→bottom.

## Objects
- [[discrete-graphical-model]] — factors, scopes, the model NCE consumes.
- [[partition-function]] — $Z$, the quantity we estimate.
- [[factor-operations]] — log-space factor product / marginalization (`FastFactor`).

## Algorithms
- [[variable-elimination]] — the base algorithm (Zhang & Poole, [[@zhang1996exploiting]]).
- [[bucket-elimination]] — Dechter's unifying organization ([[@dechter1999bucket]]).
- [[elimination-ordering]] — weighted min-fill; determines cost.
- [[induced-width]] — the complexity parameter (treewidth).

## Clustering view
- [[tree-decomposition]] — the unifying structure ([[@kask2005unifying]]).
- [[junction-tree]] · [[join-tree]] · [[clique-tree]] — same idea, three names
  ([[@lauritzen1988local]]); see [[terminology-map]].
- [[running-intersection-property]] — what makes local message passing correct.

## Alternative paradigm
- [[and-or-search]] — search instead of elimination ([[@dechter2007andor]]).

→ Continue to [[MOC-approximate-inference]].
