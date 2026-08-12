---
type: concept
title: Learned / Amortized / Neural Inference
status: growing
tags: [inference, neural, related-work]
created: 2026-06-13
updated: 2026-06-13
---

# Learned / Amortized / Neural Inference

**Learned inference** is the broad family of methods that *train a neural network to perform
or accelerate* probabilistic inference, instead of running an exact or fixed approximate
algorithm. NCE belongs to this family: it learns the **messages** of
[[bucket-elimination]]. This note collects the neighboring approaches for related-work
positioning.

## Strands

- **Amortized variational inference** — learn an *encoder* that maps each input to its
  approximate posterior, paying training cost once to make per-query inference cheap.
  Canonical: [[@kingma2014vae]] (VAE / reparameterization); for discrete latent models,
  [[@mnih2014nvil]] (NVIL).
- **Neural message passing on the graph** — train a GNN to do marginal/MAP inference by
  learning a message-passing scheme: [[@yoon2019gnninference]]; or *correct* belief-propagation
  messages with a GNN: [[@satorras2021nebp]].
- **Learned elimination messages (NCE's strand)** — replace high-width
  [[bucket-elimination]] messages with trained NNs: [[deep-bucket-elimination]]
  ([[@razeghi2021deep]]) → [[neural-bucket-elimination]] ([[@agarwal2022neurobe]]) → NCE.

## How NCE differs

NCE learns **per-bucket elimination messages** for the **partition function**, not a global
amortized posterior (VAE/NVIL) or a learned graph message-passer (GNN/NEBP). Its contribution,
[[bucket-merging]], is *structural* (re-partition the bucket tree) and composes with the
learned-message backend. The shared risk across the elimination strand is [[error-accumulation]].

## Sources

- Amortized: [[@kingma2014vae]] · [[@mnih2014nvil]]. GNN-based: [[@yoon2019gnninference]] ·
  [[@satorras2021nebp]]. Elimination strand: [[@razeghi2021deep]] · [[@agarwal2022neurobe]].

## Related

- [[neural-bucket-elimination]] · [[deep-bucket-elimination]] · [[nce-method-overview]] ·
  [[neural-network-factors]]
