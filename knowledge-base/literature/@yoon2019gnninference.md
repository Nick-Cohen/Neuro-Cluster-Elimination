---
type: literature
title: Yoon 2019 — Inference in PGMs by Graph Neural Networks
citekey: yoon2019gnninference
authors: [Yoon, KiJung, Liao, Renjie, Xiong, Yuwen, Zhang, Lisa, Fetaya, Ethan, Urtasun, Raquel, Zemel, Richard S., Pitkow, Xaq]
year: 2019
venue: Proc. 53rd Asilomar Conf. on Signals, Systems, and Computers (ACSSC)
status: growing
tags: [literature]
created: 2026-06-13
updated: 2026-06-13
---

# Yoon et al. (2019) — Inference in Probabilistic Graphical Models by Graph Neural Networks

> **Citation.** KiJung Yoon, Renjie Liao, Yuwen Xiong, Lisa Zhang, Ethan Fetaya, Raquel Urtasun, Richard S. Zemel, and Xaq Pitkow. "Inference in Probabilistic Graphical Models by Graph Neural Networks." Proc. 53rd Asilomar Conf. on Signals, Systems, and Computers (ACSSC), pp. 868–875, 2019. arXiv:1803.07710.
> **BibTeX key.** `yoon2019gnninference` (see [`../references.bib`](../references.bib))
> **Link.** arXiv:1803.07710

## Contribution (in our words)
Trains graph neural networks to perform marginal and MAP inference on PGMs by learning a message-passing scheme over the graph. The learned GNNs outperform classical belief propagation, especially on loopy graphs.

## Why it matters to NCE
A direct related-work anchor for "neural networks approximating PGM inference" ([[learned-inference]]). The contrast sharpens NCE's positioning: NCE learns elimination messages within [[neural-bucket-elimination]], rather than a generic GNN message-passer over the graph.

## Citation confidence
Most often cited as arXiv:1803.07710; the Asilomar 2019 page numbers (868–875) should be verified on IEEE Xplore if citing the published version.

## Related
- [[learned-inference]]
- [[neural-bucket-elimination]]
- [[@satorras2021nebp]]
