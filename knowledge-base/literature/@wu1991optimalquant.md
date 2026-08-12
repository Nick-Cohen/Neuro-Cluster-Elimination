---
type: literature
title: Wu 1991 — Optimal Quantization by Matrix Searching
citekey: wu1991optimalquant
authors: [Wu, Xiaolin]
year: 1991
venue: Journal of Algorithms, 12(4)
status: growing
tags: [literature]
created: 2026-06-13
updated: 2026-06-13
---

# Wu (1991) — Optimal Quantization by Matrix Searching

> **Citation.** Xiaolin Wu. "Optimal Quantization by Matrix Searching." Journal of Algorithms, 12(4):663–673, 1991. DOI 10.1016/0196-6774(91)90039-2.
> **BibTeX key.** `wu1991optimalquant` (see [`../references.bib`](../references.bib))
> **Link.** DOI 10.1016/0196-6774(91)90039-2

## Contribution (in our words)
Gives an exact dynamic-programming solution for globally optimal 1-D K-level quantization, reducing the classical O(KN^2) DP to O(KN) via matrix searching (SMAWK). Unlike Lloyd's algorithm it guarantees the global optimum rather than a local one.

## Why it matters to NCE
This is the algorithm behind NCE's optimal K-segment [[quantization]] (DP + SMAWK in `quantization.py`). It yields globally optimal level assignments for a message, providing an exact, fast surrogate to compare against learned [[neural-network-factors]].

## Citation confidence
There is no single canonical paper for DP optimal 1-D quantization (it originates with Bellman/Bruce 1964); Wu (1991) is the cleanest citable algorithmic reference and should be cited as such.

## Related
- [[quantization]]
- [[@lloyd1982quantization]]
