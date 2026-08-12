---
type: literature
title: Dechter & Rish 2003 — Mini-buckets: A General Scheme for Bounded Inference
citekey: dechter2003minibuckets
authors: [Dechter, Rina, Rish, Irina]
year: 2003
venue: Journal of the ACM
status: growing
tags: [literature]
created: 2026-06-12
updated: 2026-06-12
---

# Dechter & Rish (2003) — Mini-buckets: A General Scheme for Bounded Inference

> **Citation.** Rina Dechter and Irina Rish. "Mini-buckets: A general scheme for bounded inference." Journal of the ACM, 50(2):107–153, 2003.
> **BibTeX key.** `dechter2003minibuckets` (see [`../references.bib`](../references.bib))
> **Link.** (JACM)

## Contribution (in our words)
Mini-bucket elimination is a bounded-complexity approximation of full bucket elimination controlled by a parameter i (the i-bound). When a bucket's combined function would exceed i variables, the functions are partitioned into "mini-buckets" of at most i variables, each processed separately. This trades exactness for complexity that is exponential only in i rather than in the induced width, and yields upper and lower bounds on the quantity of interest. Larger i gives tighter approximations and recovers exact inference at i = induced width. An earlier conference version appeared as Dechter, "Mini-Buckets: A General Scheme for Generating Approximations in Automated Reasoning," IJCAI-97, pp. 1297–1302 ([[@dechter1997minibuckets]]).

## Why it matters to NCE
This is the direct ancestor of [[weighted-mini-bucket]] and the [[iB-parameter]] used throughout NCE. NCE's [[bucket-merging]] is framed precisely as the inverse of mini-bucket splitting: instead of partitioning a wide bucket into smaller mini-buckets, NCE merges adjacent NN-eligible buckets to reduce the number of neural approximations. Understanding the splitting operation here clarifies what the merge operation undoes.

## Key terms introduced
- **mini-bucket** — a partition of a bucket's functions into subsets of at most i variables, each processed independently.
- **i-bound** — the parameter bounding mini-bucket width, controlling the accuracy/complexity tradeoff.
- **bounded inference** — approximate inference with complexity exponential only in i, not in the induced width.
- **upper/lower bound** — the mini-bucket scheme produces both upper and lower bounds on the target quantity.

## Citation confidence
High. Authors are Dechter & Rish (Dechter first). Exact conference title is "...for Generating Approximations in Automated Reasoning".

## Related
- [[bucket-elimination]]
- [[mini-bucket-elimination]]
- [[weighted-mini-bucket]]
- [[iB-parameter]]
- [[bucket-merging]]
- [[induced-width]]
