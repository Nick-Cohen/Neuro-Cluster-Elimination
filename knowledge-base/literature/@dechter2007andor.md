---
type: literature
title: Dechter & Mateescu 2007 — AND/OR Search Spaces for Graphical Models
citekey: dechter2007andor
authors: [Dechter, Rina, Mateescu, Robert]
year: 2007
venue: Artificial Intelligence
status: growing
tags: [literature]
created: 2026-06-12
updated: 2026-06-12
---

# Dechter & Mateescu (2007) — AND/OR Search Spaces for Graphical Models

> **Citation.** Rina Dechter and Robert Mateescu. "AND/OR Search Spaces for Graphical Models." Artificial Intelligence, 171(2–3):73–106, 2007.
> **BibTeX key.** `dechter2007andor` (see [`../references.bib`](../references.bib))
> **Link.** DOI 10.1016/j.artint.2006.11.003

## Contribution (in our words)
Establishes the AND/OR search space as a unifying framework where AND nodes capture problem decomposition (conditional independence via a pseudo-tree) and OR nodes capture variable value choices. Exploiting decomposition makes the AND/OR space exponentially smaller than the traditional OR space, bounded by pseudo-tree depth / induced width. The optimization companion line is Marinescu & Dechter, "AND/OR Branch-and-Bound search for combinatorial optimization in graphical models," AI 173(16–17):1457–1491, 2009 ([[@marinescu2009aobb]]).

## Why it matters to NCE
AND/OR search is the major alternative paradigm to elimination/clustering for the same #P-hard inference tasks, and so is relevant related work for situating NCE. Where NCE pursues elimination with learned bucket messages, AND/OR methods exploit decomposition through search, offering a contrasting route to the same inference targets.

## Key terms introduced
- **AND/OR search space** — a search space with AND nodes (decomposition) and OR nodes (value choices).
- **pseudo-tree** — the structure capturing conditional independence used to decompose the problem.
- **decomposition** — exploiting conditional independence to shrink the search space exponentially.
- **OR space** — the traditional search space that AND/OR improves upon.

## Citation confidence
High on both. Marinescu & Dechter's companion "Memory intensive AND/OR search..." (same AI volume 173(16–17)) page range is unconfirmed — do not cite its pages.

## Related
- [[and-or-search]]
- [[induced-width]]
- [[bucket-elimination]]
