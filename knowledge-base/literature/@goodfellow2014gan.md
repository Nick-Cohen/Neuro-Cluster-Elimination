---
type: literature
title: Goodfellow 2014 — Generative Adversarial Nets
citekey: goodfellow2014gan
authors: [Goodfellow, Ian J., Pouget-Abadie, Jean, Mirza, Mehdi, Xu, Bing, Warde-Farley, David, Ozair, Sherjil, Courville, Aaron, Bengio, Yoshua]
year: 2014
venue: Advances in Neural Information Processing Systems 27 (NIPS 2014)
status: growing
tags: [literature]
created: 2026-06-13
updated: 2026-06-13
---

# Goodfellow et al. (2014) — Generative Adversarial Nets

> **Citation.** Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. "Generative Adversarial Nets." Advances in Neural Information Processing Systems 27 (NIPS 2014), pp. 2672–2680.
> **BibTeX key.** `goodfellow2014gan` (see [`../references.bib`](../references.bib))
> **Link.** —

## Contribution (in our words)
Introduces GANs — a generator/discriminator minimax game in which the generator learns to produce samples that fool a discriminator trained to distinguish real from generated data. At the optimum the generator recovers the data distribution. This is the foundational adversarial-training framework.

## Why it matters to NCE
Provides the conceptual basis for the [[adversarial-learning]] direction in NCE: a generator proposes high-loss assignments while the message-approximation network ([[neural-network-factors]]) adapts to them. Unlike a GAN, NCE uses an exact target oracle rather than a learned discriminator, and the adversarial signal can guide [[sample-generation]] toward regions where the approximation is weak.

## Citation confidence
Fully verified (8 authors, pages 2672–2680; canonical NIPS 2014).

## Related
- [[adversarial-learning]]
- [[neural-network-factors]]
- [[importance-sampling]]
- [[sample-generation]]
