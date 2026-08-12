---
type: literature
title: Bengio 2009 — Curriculum Learning
citekey: bengio2009curriculum
authors: [Bengio, Yoshua, Louradour, Jérôme, Collobert, Ronan, Weston, Jason]
year: 2009
venue: Proc. 26th International Conf. on Machine Learning (ICML 2009)
status: growing
tags: [literature]
created: 2026-06-13
updated: 2026-06-13
---

# Bengio et al. (2009) — Curriculum Learning

> **Citation.** Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. "Curriculum Learning." Proc. 26th International Conf. on Machine Learning (ICML 2009), Montreal, pp. 41–48. ACM. DOI 10.1145/1553374.1553380.
> **BibTeX key.** `bengio2009curriculum` (see [`../references.bib`](../references.bib))
> **Link.** DOI 10.1145/1553374.1553380

## Contribution (in our words)
Proposes presenting training examples in a meaningful easy→hard order rather than randomly. It formalizes curriculum learning as a continuation method that smooths a non-convex objective, gradually annealing toward the harder target task. The paper shows empirically that curricula can speed convergence and find better minima.

## Why it matters to NCE
Grounds the [[curriculum-learning]] research direction for NCE: training easy or small-scope buckets before hard, high-width ones could smooth the optimization of [[neural-network-factors]] and improve message approximation. The continuation-method framing is a natural fit for the staged, bucket-by-bucket structure of [[neural-bucket-elimination]].

## Citation confidence
Fully verified (pages 41–48 + DOI 10.1145/1553374.1553380).

## Related
- [[curriculum-learning]]
- [[neural-network-factors]]
- [[neural-bucket-elimination]]
