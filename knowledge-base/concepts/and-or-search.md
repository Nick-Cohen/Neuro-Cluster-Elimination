---
type: concept
title: AND/OR Search Spaces
status: growing
tags: [inference, search, graphical-models]
created: 2026-06-12
updated: 2026-06-12
---

# AND/OR Search Spaces

**AND/OR search** is an alternative paradigm to elimination/clustering for the same
#P-hard graphical-model tasks. It searches a space where **OR** nodes choose a variable's
value and **AND** nodes split the problem into independent subproblems (guided by a
**pseudo-tree** that exposes conditional independence). Exploiting decomposition makes the
AND/OR space exponentially smaller than a flat OR search, bounded by the pseudo-tree depth
/ [[induced-width]] ([[@dechter2007andor]]).

## Key points

- Trades the *memory* of elimination for *time* of search; the two are complementary views
  of the same problem structure.
- Underlies AND/OR Branch-and-Bound (AOBB) for optimization (MPE/WCSP) — Marinescu &
  Dechter, `marinescu2009aobb`.
- Caching on the AND/OR graph recovers elimination-like complexity (context-minimal graph).

## Relation to NCE

- NCE is in the **elimination** family, not the search family; AND/OR search is the main
  alternative paradigm and belongs in the paper's related work for completeness. Not
  currently implemented in `nce/`.

## Sources

- [[@dechter2007andor]] (and the AOBB companion `marinescu2009aobb`).

## Related

- [[bucket-elimination]] · [[induced-width]] · [[partition-function]]
