---
type: concept
title: Deep Bucket Elimination (DBE)
status: growing
tags: [inference, neural, graphical-models]
created: 2026-06-12
updated: 2026-06-12
---

# Deep Bucket Elimination (DBE)

**Deep Bucket Elimination** is the first method to replace the messages of high-width
buckets in [[bucket-elimination]] with **trained neural networks**, used to estimate the
[[partition-function]]. When a bucket's exact message table is too large to store (memory
exponential in [[induced-width]]), DBE fits a neural network to the message function
instead of computing it exactly ([[@razeghi2021deep]]).

## Key points

- Proof-of-concept that learned messages can beat state-of-the-art approximations on $Z$.
- The network is queried by downstream buckets like any other factor — a
  [[neural-network-factors|neural network factor]].
- **Direct ancestor of [[neural-bucket-elimination]] (NeuroBE)**, which personalizes the
  per-message architecture, training, and loss.

## Relation to NCE

- DBE is the first step of NCE's lineage (DBE → NeuroBE → NCE). NCE reproduces the NeuroBE
  refinement and adds [[bucket-merging]] to cut the *number* of learned messages (the
  driver of [[error-accumulation]]).

## Sources

- [[@razeghi2021deep]] (IJCAI 2021; six authors — note Marinescu and Ihler are **not**
  DBE authors).

## Related

- [[neural-bucket-elimination]] · [[bucket-elimination]] · [[neural-network-factors]] ·
  [[error-accumulation]] · [[nce-method-overview]]
