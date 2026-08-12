---
type: literature
title: Mnih 2014 — Neural Variational Inference and Learning in Belief Networks
citekey: mnih2014nvil
authors: [Mnih, Andriy, Gregor, Karol]
year: 2014
venue: Proc. 31st International Conf. on Machine Learning (ICML 2014), PMLR 32
status: growing
tags: [literature]
created: 2026-06-13
updated: 2026-06-13
---

# Mnih & Gregor (2014) — Neural Variational Inference and Learning in Belief Networks

> **Citation.** Andriy Mnih and Karol Gregor. "Neural Variational Inference and Learning in Belief Networks." Proc. 31st International Conf. on Machine Learning (ICML 2014), PMLR 32, pp. 1791–1799. arXiv:1402.0030.
> **BibTeX key.** `mnih2014nvil` (see [`../references.bib`](../references.bib))
> **Link.** arXiv:1402.0030

## Contribution (in our words)
Trains a feedforward inference network jointly with a directed belief network by maximizing a variational lower bound, using variance-reduction baselines for the REINFORCE gradient. It is an early amortized-inference method specifically for discrete latent-variable models.

## Why it matters to NCE
A related-work anchor for [[learned-inference]] over discrete models — the regime NCE operates in. Its emphasis on discrete latent variables and gradient variance reduction is directly relevant to training [[neural-network-factors]] on discrete bucket assignments.

## Citation confidence
Fully verified (PMLR v32, pp. 1791–1799).

## Related
- [[learned-inference]]
- [[@kingma2014vae]]
