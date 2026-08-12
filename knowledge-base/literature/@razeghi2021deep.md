---
type: literature
title: Razeghi 2021 — Deep Bucket Elimination
citekey: razeghi2021deep
authors: [Razeghi, Yasaman, Kask, Kalev, Lu, Yadong, Baldi, Pierre, Agarwal, Sakshi, Dechter, Rina]
year: 2021
venue: IJCAI-21
status: growing
tags: [literature]
created: 2026-06-12
updated: 2026-06-12
---

# Razeghi (2021) — Deep Bucket Elimination

> **Citation.** Yasaman Razeghi, Kalev Kask, Yadong Lu, Pierre Baldi, Sakshi Agarwal, and Rina Dechter. "Deep Bucket Elimination." Proc. 30th IJCAI (IJCAI-21), pp. 4235–4242, 2021.
> **BibTeX key.** `razeghi2021deep` (see [`../references.bib`](../references.bib))
> **Link.** DOI 10.24963/ijcai.2021/582

## Contribution (in our words)
Bucket Elimination is a universal exact inference scheme but requires memory exponential in the induced width, which is often infeasible. Deep Bucket Elimination (DBE) replaces the messages BE generates with neural-network approximations whenever a bucket's memory requirement is too large, applied to computing the partition function. The empirical proof-of-concept shows DBE can be more accurate than state-of-the-art approximations.

## Why it matters to NCE
DBE is the FIRST step of NCE's direct lineage — the original idea of replacing high-width bucket messages with neural networks. [[neural-bucket-elimination]] ([[@agarwal2022neurobe]]) builds directly on it, and NCE reproduces and extends that line.

## Key terms introduced
- **Deep bucket elimination** — using neural networks to approximate BE messages when a bucket is too large to represent exactly.
- **Learned message** — a bucket message represented by a trained neural network rather than an explicit table.
- **Message approximation** — substituting an approximate representation for a high-width exact message.

## Citation confidence
High — full six-author list and order, venue, year, pages confirmed via official IJCAI proceedings and DBLP. NOTE: Radu Marinescu and Alexander Ihler are NOT authors of DBE (a common mistake) — do not add them.

## Related
- [[bucket-elimination]]
- [[induced-width]]
- [[partition-function]]
- [[deep-bucket-elimination]]
- [[neural-network-factors]]
- [[@agarwal2022neurobe]]
