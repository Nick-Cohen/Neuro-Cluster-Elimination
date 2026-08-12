---
type: literature
title: Mateescu, Kask, Gogate & Dechter 2010 — Join-Graph Propagation Algorithms
citekey: mateescu2010jgp
authors: [Mateescu, Robert, Kask, Kalev, Gogate, Vibhav, Dechter, Rina]
year: 2010
venue: Journal of Artificial Intelligence Research
status: growing
tags: [literature]
created: 2026-06-12
updated: 2026-06-12
---

# Mateescu, Kask, Gogate & Dechter (2010) — Join-Graph Propagation Algorithms

> **Citation.** Robert Mateescu, Kalev Kask, Vibhav Gogate, and Rina Dechter. "Join-Graph Propagation Algorithms." Journal of Artificial Intelligence Research, 37:279–328, 2010.
> **BibTeX key.** `mateescu2010jgp` (see [`../references.bib`](../references.bib))
> **Link.** DOI 10.1613/jair.2842

## Contribution (in our words)
The definitive journal treatment of the parameterized family of bounded-inference algorithms ranging from mini-clustering up to Iterative Join-Graph Propagation (IJGP), which applies join-tree-style message passing to join-graphs iteratively — blending bounded inference (the i-bound of mini-clustering) with iteration (loopy/generalized belief propagation). The origin paper is Dechter, Kask & Mateescu, "Iterative Join-Graph Propagation," UAI 2002, pp. 128–136 ([[@dechter2002ijgp]]); note the author order differs, with Gogate added in the 2010 journal version.

## Why it matters to NCE
IJGP/join-graph propagation is the other main classical approximate-inference family in this lineage and a natural related-work anchor for NCE. It shares the [[iB-parameter]] knob with [[weighted-mini-bucket]], so it sits alongside WMB as a bounded-cost baseline against which NCE's neural message approximation can be situated.

## Key terms introduced
- **join graph** — a graph of clusters over which messages are passed, generalizing the join tree.
- **iterative join-graph propagation (IJGP)** — iterative message passing on a join graph blending bounded inference with loopy iteration.
- **mini-clustering** — the bounded-inference building block parameterized by an i-bound.
- **i-bound** — the parameter bounding cluster width.
- **generalized belief propagation** — the loopy/iterative message-passing regime that IJGP generalizes.

## Citation confidence
High. Cite UAI 2002 (Dechter-Kask-Mateescu) for origin and JAIR 2010 (Mateescu-Kask-Gogate-Dechter) as definitive.

## Related
- [[junction-tree]]
- [[tree-decomposition]]
- [[iterative-join-graph-propagation]]
- [[weighted-mini-bucket]]
- [[iB-parameter]]
