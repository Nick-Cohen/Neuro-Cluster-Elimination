---
type: literature
title: Lloyd 1982 — Least Squares Quantization in PCM
citekey: lloyd1982quantization
authors: [Lloyd, Stuart P.]
year: 1982
venue: IEEE Transactions on Information Theory, 28(2)
status: growing
tags: [literature]
created: 2026-06-13
updated: 2026-06-13
---

# Lloyd (1982) — Least Squares Quantization in PCM

> **Citation.** Stuart P. Lloyd. "Least Squares Quantization in PCM." IEEE Transactions on Information Theory, 28(2):129–137, 1982. DOI 10.1109/TIT.1982.1056489.
> **BibTeX key.** `lloyd1982quantization` (see [`../references.bib`](../references.bib))
> **Link.** DOI 10.1109/TIT.1982.1056489

## Contribution (in our words)
Derives the centroid and nearest-neighbor optimality conditions for minimum mean-squared-error scalar quantization, and the iterative "Lloyd's algorithm" that alternates between them (originally a 1957 Bell Labs memo). This is the foundation of the Lloyd–Max quantizer and of k-means.

## Why it matters to NCE
Provides theoretical grounding for the [[quantization]] surrogate in NCE — approximating a bucket message by K representative levels. The MMSE optimality conditions justify quantization as a principled, low-parameter alternative to [[neural-network-factors]] for compressing messages.

## Citation confidence
Fully verified.

## Related
- [[quantization]]
- [[@max1960quantizing]]
- [[@wu1991optimalquant]]
