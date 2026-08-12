---
type: literature
title: Kingma 2014 — Auto-Encoding Variational Bayes
citekey: kingma2014vae
authors: [Kingma, Diederik P., Welling, Max]
year: 2014
venue: Proc. 2nd International Conf. on Learning Representations (ICLR 2014)
status: growing
tags: [literature]
created: 2026-06-13
updated: 2026-06-13
---

# Kingma & Welling (2014) — Auto-Encoding Variational Bayes

> **Citation.** Diederik P. Kingma and Max Welling. "Auto-Encoding Variational Bayes." Proc. 2nd International Conf. on Learning Representations (ICLR 2014). arXiv:1312.6114.
> **BibTeX key.** `kingma2014vae` (see [`../references.bib`](../references.bib))
> **Link.** arXiv:1312.6114

## Contribution (in our words)
Introduces the reparameterization trick and an amortized inference network (encoder) that maps each datapoint to its approximate posterior, enabling scalable stochastic variational inference in deep latent-variable models. It is the canonical amortized-inference reference.

## Why it matters to NCE
Serves as a related-work anchor for amortized [[learned-inference]] — learning an inference computation once rather than running exact inference per query. NCE similarly amortizes work by learning bucket messages, though it targets discrete PGMs and the partition function rather than continuous latent posteriors.

## Citation confidence
The title is "Auto-Encoding Variational Bayes" (not "Variational Autoencoders"); ICLR 2014 had no archival proceedings, so cite via arXiv:1312.6114.

## Related
- [[learned-inference]]
- [[neural-bucket-elimination]]
- [[@mnih2014nvil]]
