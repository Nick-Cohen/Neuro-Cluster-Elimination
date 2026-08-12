---
type: literature
title: Agarwal 2022 — NeuroBE
citekey: agarwal2022neurobe
authors: [Agarwal, Sakshi, Kask, Kalev, Ihler, Alexander, Dechter, Rina]
year: 2022
venue: UAI 2022 (PMLR vol. 180)
status: growing
tags: [literature]
created: 2026-06-12
updated: 2026-06-12
---

# Agarwal (2022) — NeuroBE: Escalating Neural Network Approximations of Bucket Elimination

> **Citation.** Sakshi Agarwal, Kalev Kask, Alexander Ihler, and Rina Dechter. "NeuroBE: Escalating Neural Network Approximations of Bucket Elimination." Proc. 38th Conf. on Uncertainty in Artificial Intelligence (UAI 2022), PMLR vol. 180, pp. 11–21, 2022.
> **BibTeX key.** `agarwal2022neurobe` (see [`../references.bib`](../references.bib))
> **Link.** PMLR vol. 180

## Contribution (in our words)
NeuroBE builds directly on Deep Bucket Elimination, using neural networks to approximate BE messages in high-memory buckets to estimate the partition function. Its contribution is to PERSONALIZE the NN construction and training using prior information about each message's size and distribution — customizing the architecture, the learning process, and especially the loss function to the form of each message. It reports significant accuracy and time improvements over DBE.

## Why it matters to NCE
NeuroBE is the method NCE reproduces faithfully (project milestone M003) and extends with [[bucket-merging]]. It is THE central reference for the project.

## Key terms introduced
- **NeuroBE** — escalating, per-message-personalized neural approximation of bucket-elimination messages.
- **Escalating approximations** — progressively investing more NN capacity/effort as message difficulty grows.
- **Per-message customization** — tailoring architecture, training, and loss to each message's size and distribution.
- **Importance sampling** — sampling assignments to focus training on high-weight regions of a message.
- **Min-max label normalization** — rescaling message labels to stabilize NN training.
- **Weighted MSE loss** — a customized loss weighting errors according to message structure.

## Citation confidence
VERY IMPORTANT — this paper is frequently mis-cited. The CORRECT venue is UAI 2022 (PMLR vol. 180, pp. 11–21), NOT AAAI 2022. The CORRECT title is "Escalating Neural Network Approximations of Bucket Elimination" (note "Neural Network", not "NN"; "of", not "to"). Four authors: Agarwal, Kask, Ihler, Dechter (this differs from the DBE author set — no Razeghi, Lu, or Baldi).

## Related
- [[deep-bucket-elimination]]
- [[neural-bucket-elimination]]
- [[neural-network-factors]]
- [[importance-sampling]]
- [[error-accumulation]]
- [[nce-method-overview]]
- [[@razeghi2021deep]]
