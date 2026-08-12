---
type: moc
title: MOC — This Project (NCE)
status: evergreen
tags: [moc, this-project]
created: 2026-06-12
updated: 2026-08-12
---

# MOC — This Project (NCE)

## The method
- [[nce-method-overview]] — pipeline end-to-end.
- [[bucket-merging]] — the contribution (3 strategies, headline results).
- [[merge-bound]] — the single knob ($e_{\max}$, formerly $D$).
- [[error-accumulation]] — the problem it solves.
- [[time-optimal-merge-bound]] — ⚠️ the sweet spot is a property of a **code version**.
- [[wmb-residual-learning]] — learn the residual against a WMB estimate.

## Correctness findings (2026-08, read before trusting an old number)
- [[bit-exact-reproducibility]] — runs were not reproducible at a fixed seed; now they are.
- [[num-samples-freeze]] — the per-cluster sample count was frozen and arm-confounded.
- [[nn-elim-enumeration-redundancy]] — the NN path evaluated $k^{e-e_f}$ redundant rows.
- [[backward-factor-population-under-merging]] — broken or silently wrong under merging.
- [[convergence-diagnostic-gap]] — no usable convergence signal existed until 2026-08-11.
- [[adam-eps-and-loss-scale]] — `eps` is a cost knob, parameterisation is an accuracy knob.

## Navigating the code & terms
- [[codebase-map]] — concept → `nce/*.py:line`.
- [[terminology-map]] — NCE term ↔ standard term, and citation gotchas.
- [[glossary]] — definitions.
- Architecture: [[three-layer-architecture]] · [[graphical-model-structure]] ·
  [[bucket-structure]] · [[log-space-convention]].

## Research directions (migrated)
- [[curriculum-learning]] · [[adversarial-learning]] · [[mini-bucket-sampling]] ·
  [[backward-messages]] · [[backward-sensitivity]].

## Status & writing
- [[open-questions]] — what's blocking the paper / live unknowns.
- Paper: [`../../writeup/paper.md`](../../writeup/paper.md) · provenance
  [`../../writeup/NOTES.md`](../../writeup/NOTES.md).
- Progress: [`../progress/`](../progress/) (weekly summaries, advisor updates) ·
  `lab_notebook.txt`. Latest: [[2026-W33]].

## Related literature
- [[@agarwal2022neurobe]] · [[@razeghi2021deep]] · [[@liu2011holder]] ·
  [[@dechter2003minibuckets]] · [[@dechter1999bucket]] · [[@kask2005unifying]].
