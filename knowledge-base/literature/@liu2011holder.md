---
type: literature
title: Liu & Ihler 2011 — Bounding the Partition Function using Hölder's Inequality
citekey: liu2011holder
authors: [Liu, Qiang, Ihler, Alexander]
year: 2011
venue: ICML 2011
status: growing
tags: [literature]
created: 2026-06-12
updated: 2026-06-12
---

# Liu & Ihler (2011) — Bounding the Partition Function using Hölder's Inequality

> **Citation.** Qiang Liu and Alexander Ihler. "Bounding the Partition Function using Hölder's Inequality." Proc. 28th ICML (ICML 2011), Bellevue, WA, pp. 849–856, 2011.
> **BibTeX key.** `liu2011holder` (see [`../references.bib`](../references.bib))
> **Link.** (PMLR/ICML 2011)

## Contribution (in our words)
Introduces an approximate-inference algorithm based on Hölder's inequality that produces both upper and lower bounds on the partition function, unifying and generalizing mini-bucket elimination with variational methods (tree-reweighted BP, conditional entropy decomposition). This is the foundational reference for the weighted mini-bucket (WMB) bound — the "weights" come from the Hölder exponents. Related follow-ups extend the line: Liu & Ihler, "Variational Algorithms for Marginal MAP," JMLR 14:3165–3200, 2013 ([[@liu2013marginalmap]]), and Forouzan & Ihler, "Incremental Region Selection for Mini-bucket Elimination Bounds," UAI 2015, pp. 268–277 ([[@forouzan2015incremental]]).

## Why it matters to NCE
This is THE citation for [[weighted-mini-bucket]], NCE's classical baseline and the source of meaning for the per-mini-bucket weights and the [[iB-parameter]]. WMB is the bounded-cost alternative against which NCE's learned-message approach is compared, so the Hölder-based bound on the [[partition-function]] is the precise quantity NCE aims to estimate more tightly with neural networks.

## Key terms introduced
- **weighted mini-bucket** — a mini-bucket scheme whose bound is controlled by Hölder weights.
- **Hölder's inequality** — the inequality underlying the partition-function bound.
- **power sum / weighted log-sum-exp** — the weighted marginalization operation generalizing log-sum-exp.
- **partition-function bound** — upper and lower bounds on Z produced by the method.
- **cost shifting** — reparameterization used to tighten the bound.

## Citation confidence
High on the Hölder paper (DBLP conf/icml/LiuI11, pp. 849–856; ICML 2011 was in Bellevue, WA). The marginal-MAP UAI 2011 page numbers are medium-confidence — prefer the JMLR 2013 version.

## Related
- [[weighted-mini-bucket]]
- [[partition-function]]
- [[mini-bucket-elimination]]
- [[iB-parameter]]
