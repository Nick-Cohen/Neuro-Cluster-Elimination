---
type: moc
title: MOC — Neural Inference
status: evergreen
tags: [moc, neural]
created: 2026-06-12
updated: 2026-06-12
---

# MOC — Neural Inference

Replacing wide bucket messages with learned surrogates — NCE's direct lineage.

## Lineage
- [[deep-bucket-elimination]] — first to learn high-width messages ([[@razeghi2021deep]], 2021).
- [[neural-bucket-elimination]] — personalized per-message NNs ([[@agarwal2022neurobe]], 2022).
  ⚠️ cite as **UAI 2022**, not AAAI — see [[terminology-map]].

## Mechanics
- [[neural-network-factors]] — the NN-as-factor wrapper (`FactorNN`).
- [[sample-generation]] · [[importance-sampling]] — training data for a message.
- [[loss-functions]] — log/linear MSE, weighted MSE, [[unnormalized-kl-divergence]].
- [[data-normalization]] · [[one-hot-encoding]] — input/label handling.
- Alternatives: [[memorizer-nn]] · [[decision-tree-approximation]] · [[quantization]].

## The problem NCE attacks
- [[error-accumulation]] — final $Z$ error scales with the **number** of learned messages.
  → [[MOC-this-project]] / [[bucket-merging]].

→ Continue to [[MOC-this-project]].
