---
type: concept
title: Discrete Graphical Model
status: growing
tags: [inference, graphical-models, foundations]
created: 2026-06-12
updated: 2026-06-12
---

# Discrete Graphical Model

A discrete graphical model compactly represents a function over many discrete
variables as a **product of factors**, each depending on only a small subset of the
variables. Over variables $X = \{X_1,\dots,X_n\}$ with finite domains, the model
defines an unnormalized distribution
$$ \tilde p(x) = \prod_{j} f_j(x_{S_j}), $$
where each factor $f_j$ has **scope** $S_j \subseteq X$. Bayesian networks (directed,
factors are CPTs) and Markov random fields / factor graphs (undirected, factors are
potentials) are the two standard families.

## Key points

- The graph encodes **conditional independence**: structure is what makes inference
  tractable when treewidth is small.
- The central inference quantities are the **[[partition-function]]** $Z=\sum_x \tilde p(x)$,
  marginals, and the most-probable explanation (MPE/MAP).
- Exact inference is #P-hard in general; cost is governed by [[induced-width]].
- NCE works with models loaded from the **UAI text format** (see [[@uaifileformat]] in
  `references.bib`), via the `pyGMs` library.

## In NCE

- A factor is a `FastFactor` ([`nce/inference/factor.py`](../../nce/inference/factor.py)),
  a tensor of **log-space** values with variable labels — see [[factor-operations]] and
  [[log-space-convention]].
- The whole model is a `FastGM` ([`nce/inference/graphical_model.py`](../../nce/inference/graphical_model.py));
  see [[graphical-model-structure]].
- Benchmark instances (grids, pedigrees, RBMs) come from the UCI/UAI repositories.

## Sources

- [[@koller2009pgm]], [[@darwiche2009modeling]], [[@pearl1988probabilistic]] — textbook foundations.

## Related

- [[partition-function]] · [[factor-operations]] · [[variable-elimination]] · [[bucket-elimination]]
