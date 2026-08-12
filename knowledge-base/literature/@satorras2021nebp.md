---
type: literature
title: Satorras 2021 — Neural Enhanced Belief Propagation on Factor Graphs
citekey: satorras2021nebp
authors: [Satorras, Victor Garcia, Welling, Max]
year: 2021
venue: Proc. 24th International Conf. on Artificial Intelligence and Statistics (AISTATS 2021), PMLR 130
status: growing
tags: [literature]
created: 2026-06-13
updated: 2026-06-13
---

# Satorras & Welling (2021) — Neural Enhanced Belief Propagation on Factor Graphs

> **Citation.** Victor Garcia Satorras and Max Welling. "Neural Enhanced Belief Propagation on Factor Graphs." Proc. 24th International Conf. on Artificial Intelligence and Statistics (AISTATS 2021), PMLR 130, pp. 685–693. arXiv:2003.01998.
> **BibTeX key.** `satorras2021nebp` (see [`../references.bib`](../references.bib))
> **Link.** arXiv:2003.01998

## Contribution (in our words)
Proposes a hybrid that runs a factor-graph GNN alongside belief propagation, with the GNN correcting BP messages at each iteration. This combines BP's structural inductive bias with the flexibility of a learned correction.

## Why it matters to NCE
A related-work anchor for hybrid learned+classical inference ([[learned-inference]]). It parallels NCE's strategy of keeping a classical inference backbone (bucket elimination) while injecting learned components ([[neural-network-factors]]) where exact computation is infeasible.

## Citation confidence
Pages 685–693 (PMLR v130) are high confidence.

## Related
- [[learned-inference]]
- [[@yoon2019gnninference]]
