---
type: literature
title: Dechter 1999 — Bucket Elimination: A Unifying Framework for Reasoning
citekey: dechter1999bucket
authors: [Dechter, Rina]
year: 1999
venue: Artificial Intelligence
status: growing
tags: [literature]
created: 2026-06-12
updated: 2026-06-12
---

# Dechter (1999) — Bucket Elimination: A Unifying Framework for Reasoning

> **Citation.** Rina Dechter. "Bucket elimination: A unifying framework for reasoning." Artificial Intelligence, 113(1–2):41–85, 1999.
> **BibTeX key.** `dechter1999bucket` (see [`../references.bib`](../references.bib))
> **Link.** — (Artificial Intelligence journal; no open DOI listed)

## Contribution (in our words)
Bucket elimination is an algorithmic framework that generalizes nonserial dynamic programming to unify a wide range of inference and reasoning tasks under variable elimination. Variables are processed in a fixed order; each variable's "bucket" collects all functions mentioning it, which are combined and the variable marginalized out, producing a message passed to a later bucket. It subsumes belief updating, MPE, MAP and MEU for probabilistic inference, plus adaptive consistency, directional resolution and Fourier elimination for constraints. Complexity is exponential in the induced width of the ordered graph.

Note on closely-related earlier items by the same author, which carry the variant title "...for probabilistic inference": the UAI-96 conference paper (pp. 211–219, citekey `dechter1996bucket`) and the 1998 book chapter in *Learning in Graphical Models* (pp. 75–104, citekey `dechter1998bucket`). The 1999 AI journal article is the canonical citation.

## Why it matters to NCE
This is the foundational reference for [[bucket-elimination]], [[variable-elimination]], and [[induced-width]] — the exact algorithm NCE accelerates with neural approximations. NCE runs bucket elimination and replaces the messages of high-width buckets (those whose tables are exponential in [[induced-width]] and too large to represent exactly) with trained neural-network approximations. Understanding bucket structure and message passing here is prerequisite to understanding both NCE and its contribution, [[bucket-merging]].

## Key terms introduced
- **bucket** — the set of all functions mentioning a given variable, collected when that variable is processed in the elimination order.
- **message** — the function produced by combining a bucket's functions and marginalizing out its variable, then passed to a later bucket.
- **induced width** — the maximum bucket width along the ordering; bounds the algorithm's exponential time and space complexity.
- **elimination order** — the fixed ordering of variables that determines bucket structure and induced width.

## Citation confidence
High on all fields. The "reasoning" vs "probabilistic inference" title difference between the 1999 journal version and the 1996/1998 versions is real, not an error.

## Related
- [[bucket-elimination]]
- [[variable-elimination]]
- [[induced-width]]
- [[elimination-ordering]]
- [[partition-function]]
- [[neural-bucket-elimination]]
- [[nce-method-overview]]
