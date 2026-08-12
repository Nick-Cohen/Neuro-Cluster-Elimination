---
type: literature
title: Lauritzen & Spiegelhalter 1988 — Local Computations with Probabilities on Graphical Structures
citekey: lauritzen1988local
authors: [Lauritzen, S. L., Spiegelhalter, D. J.]
year: 1988
venue: Journal of the Royal Statistical Society, Series B
status: growing
tags: [literature]
created: 2026-06-12
updated: 2026-06-12
---

# Lauritzen & Spiegelhalter (1988) — Local Computations with Probabilities on Graphical Structures and Their Application to Expert Systems

> **Citation.** S. L. Lauritzen and D. J. Spiegelhalter. "Local Computations with Probabilities on Graphical Structures and Their Application to Expert Systems." Journal of the Royal Statistical Society, Series B, 50(2):157–224, 1988 (with discussion).
> **BibTeX key.** `lauritzen1988local` (see [`../references.bib`](../references.bib))
> **Link.** DOI [10.1111/j.2517-6161.1988.tb01721.x](https://doi.org/10.1111/j.2517-6161.1988.tb01721.x)

## Contribution (in our words)
The foundational junction-tree inference paper: local computation of marginals on a triangulated (chordal) graph via moralization and triangulation, then propagation over a tree of cliques. It is the basis of the HUGIN system. Two companion architecture papers define this family: Shenoy & Shafer 1990, "Axioms for Probability and Belief-Function Propagation" (the abstract Shenoy–Shafer propagation architecture; citekey `shenoy1990axioms`), and Jensen, Lauritzen & Olesen 1990, "Bayesian Updating in Causal Probabilistic Networks by Local Computations" (the HUGIN collect/distribute architecture with separator division; citekey `jensen1990bayesian`).

## Why it matters to NCE
Junction-tree / clique-tree inference is the exact-inference backbone against which NCE's bucket-elimination engine is understood: a [[junction-tree]] is the cluster-tree dual of an elimination order, and the cliques NCE cannot represent exactly (because their tables are exponential in [[induced-width]]) are precisely where neural-network message approximations enter. The [[running-intersection-property]] and [[tree-decomposition]] structure formalized here underpins NCE's contribution, [[bucket-merging]], which merges adjacent buckets into larger clusters.

## Key terms introduced
- **junction tree** — a tree of cliques of a triangulated graph satisfying the running-intersection property, over which marginals propagate.
- **clique tree** — synonymous with junction tree, but specifically built from the *maximal* cliques of the triangulated graph.
- **join tree** — another largely synonymous term for the same cluster-tree structure (community convention).
- **separator** — the intersection of two adjacent cliques in the tree; messages divide/multiply through separators during propagation.
- **triangulation** — adding fill edges to make the (moralized) graph chordal so a junction tree exists.
- **running-intersection property** — the requirement that any variable shared by two cliques appears in every clique on the path between them.

## Citation confidence
High. One legitimate ambiguity worth noting: L&S pages are sometimes given as 157–194 (article body) and sometimes 157–224 (the full read-paper item with discussion); we cite 157–224. Jensen–Lauritzen–Olesen appeared in a now-defunct journal (Computational Statistics Quarterly, 4:269–282) with no DOI — moderate-high confidence on pages.

## Related
- [[junction-tree]]
- [[join-tree]]
- [[clique-tree]]
- [[running-intersection-property]]
- [[tree-decomposition]]
- [[variable-elimination]]
- [[bucket-merging]]
