---
type: concept
title: Neural Bucket Elimination (NeuroBE)
status: growing
tags: [inference, neural, graphical-models, this-project]
created: 2026-06-12
updated: 2026-06-12
---

# Neural Bucket Elimination (NeuroBE)

**NeuroBE** ("Neural Bucket Elimination") performs approximate inference by running
[[bucket-elimination]] and replacing each high-width bucket's outgoing message with a
**personalized trained neural network** ([[@agarwal2022neurobe]]). It builds directly on
[[deep-bucket-elimination]], adding per-message customization: the architecture, training
process, and especially the **loss function** are tailored to each message's predicted
size and value distribution. The trained network *is* the bucket's message — downstream
buckets query it as a factor.

## The per-bucket pipeline

1. **Sample** configurations of the message's output scope ([[importance-sampling]] in the
   faithful mode; see [[sample-generation]]).
2. **Label** each sample with its exact log-message value (sum out the bucket's eliminated
   variables for that fixed output).
3. **Normalize** labels to $[0,1]$ by min–max scaling (see [[data-normalization]]).
4. **Train** a small MLP (ReLU, weighted MSE, early stopping) to predict the label.
5. **Denormalize** at query time.

Buckets whose exact table fits under `ecl` are computed exactly; the rest are learned.

## Key points

- The dominant error driver is the **number** of learned buckets, not per-network loss —
  see [[error-accumulation]]. This motivates NCE's [[bucket-merging]].
- ⚠️ **Cite correctly:** UAI 2022 (PMLR 180, pp. 11–21), title "Escalating *Neural
  Network* Approximations *of* Bucket Elimination", authors Agarwal, Kask, Ihler, Dechter —
  **not** AAAI 2022. See [[@agarwal2022neurobe]] and [[terminology-map]].

## In NCE

- Reproduced faithfully under the `neurobe_mode` config flag (milestone M003), matching
  NeuroBE's NN dispatch counts on all 15 working binary-domain problems. See
  [[nce-method-overview]] and [[neural-network-factors]].

## Sources

- [[@agarwal2022neurobe]] · builds on [[@razeghi2021deep]].

## Related

- [[deep-bucket-elimination]] · [[bucket-elimination]] · [[neural-network-factors]] ·
  [[importance-sampling]] · [[error-accumulation]] · [[bucket-merging]] · [[nce-method-overview]]
