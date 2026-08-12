---
type: literature
title: Kask, Dechter, Larrosa & Dechter 2005 — Unifying Tree Decompositions for Reasoning in Graphical Models
citekey: kask2005unifying
authors: [Kask, Kalev, Dechter, Rina, Larrosa, Javier, Dechter, Avi]
year: 2005
venue: Artificial Intelligence
status: growing
tags: [literature]
created: 2026-06-12
updated: 2026-06-12
---

# Kask, Dechter, Larrosa & Dechter (2005) — Unifying Tree Decompositions for Reasoning in Graphical Models

> **Citation.** Kalev Kask, Rina Dechter, Javier Larrosa, and Avi Dechter. "Unifying Tree Decompositions for Reasoning in Graphical Models." Artificial Intelligence, 166(1–2):165–193, 2005.
> **BibTeX key.** `kask2005unifying` (see [`../references.bib`](../references.bib))
> **Link.** — (Artificial Intelligence journal)

## Contribution (in our words)
Unifies cluster/tree-decomposition schemes (join-tree clustering, bucket-tree elimination, cluster-tree elimination) for reasoning in graphical models, providing a common framework and the machinery of super-clusters / merged clusters in a tree decomposition. We use it as the anchor reference for the notion of merging adjacent buckets into larger clusters. Note that the term "super-bucket" itself is non-canonical/loosely used in the literature and has no single origin paper; this paper plus [[@dechter1999bucket]] are the right anchors.

## Why it matters to NCE
This is the closest formal grounding for [[super-bucket]] and for [[bucket-merging]] (our contribution): merging adjacent buckets into a super-cluster is exactly the tree-decomposition operation this paper formalizes. NCE greedily merges adjacent NN-eligible buckets to cut the number of neural approximations needed to estimate the [[partition-function]], and the cluster-tree machinery here justifies that merge as a valid tree-decomposition transformation rather than an ad-hoc heuristic.

## Key terms introduced
- **tree decomposition** — a cluster tree over the variables satisfying the running-intersection property, generalizing junction trees and elimination orders.
- **cluster-tree elimination (CTE)** — message passing over the clusters of a tree decomposition for reasoning tasks.
- **bucket-tree elimination (BTE)** — the bucket-elimination view of message passing over a tree of buckets.
- **super-cluster** — a cluster formed by merging adjacent clusters/buckets of a tree decomposition (the formal basis for "super-bucket").

## Citation confidence
High on authors/venue/volume/pages/year. The specific term "super-bucket" is not coined here; treat this as the tree-decomposition/super-cluster anchor, not the origin of that word.

## Related
- [[super-bucket]]
- [[bucket-merging]]
- [[tree-decomposition]]
- [[bucket-elimination]]
- [[junction-tree]]
- [[@dechter1999bucket]]
- [[nce-method-overview]]
