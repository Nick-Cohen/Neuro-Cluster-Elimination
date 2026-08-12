---
type: concept
title: Super-Bucket
status: growing
tags: [inference, graphical-models, terminology, this-project]
created: 2026-06-12
updated: 2026-06-12
---

# Super-Bucket

A **super-bucket** is a bucket that eliminates **several variables at once** — formed by
merging adjacent buckets / clusters in a [[tree-decomposition]] so a group of variables is
eliminated together. It trades more memory (a wider local computation) for fewer, coarser
messages.

> ⚠️ **Terminology caution.** "Super-bucket" is **not** a single-origin canonical term in
> the literature; it is used loosely. Two related senses appear: (A) simultaneously
> eliminating a *set* of variables (clustering them into a meta-variable), and (B) merging
> adjacent buckets into a "super-cluster" (super-bucket-tree), trading space for time. The
> right anchors are Dechter's bucket framework ([[@dechter1999bucket]]) and the
> tree-decomposition / super-cluster machinery of [[@kask2005unifying]] — not a paper that
> "coins" the word. When writing, prefer **"merged cluster"** / **"super-cluster"** unless
> quoting. See [[terminology-map]].

## Key points

- Closest formal concepts: [[tree-decomposition]], cluster-tree elimination, bucket-tree
  elimination, super-clusters in a join tree, and mini-bucket *merging* (inverse of
  [[mini-bucket-elimination]] splitting).
- Local exact cost grows as ≈ $2^{(\text{variables eliminated})}$ — the reason a cap is
  needed (see [[merge-bound]]).

## In NCE

- A `FastBucket` with `len(elim_vars) > 1` is a super-bucket. They are produced by
  [[bucket-merging]] (`merge_join_tree`, `merge_by_degree`, `reduce_nn_merge` in
  [`nce/inference/graphical_model.py`](../../nce/inference/graphical_model.py)). A merged
  super-bucket emits **one** message — exact if its merged table now fits under `ecl`,
  otherwise a single trained network. See [[cluster-elimination]].

## Sources

- [[@dechter1999bucket]] · [[@kask2005unifying]] (super-cluster / tree decompositions).

## Related

- [[bucket-merging]] · [[merge-bound]] · [[tree-decomposition]] · [[cluster-elimination]] · [[bucket-elimination]]
