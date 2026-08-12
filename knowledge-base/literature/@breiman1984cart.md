---
type: literature
title: Breiman 1984 — Classification and Regression Trees
citekey: breiman1984cart
authors: [Breiman, Leo, Friedman, Jerome H., Olshen, Richard A., Stone, Charles J.]
year: 1984
venue: Wadsworth & Brooks/Cole, Monterey, CA
status: growing
tags: [literature]
created: 2026-06-13
updated: 2026-06-13
---

# Breiman et al. (1984) — Classification and Regression Trees

> **Citation.** Leo Breiman, Jerome H. Friedman, Richard A. Olshen, and Charles J. Stone. "Classification and Regression Trees." Wadsworth & Brooks/Cole, Monterey, CA, 1984.
> **BibTeX key.** `breiman1984cart` (see [`../references.bib`](../references.bib))
> **Link.** —

## Contribution (in our words)
Foundational monograph introducing CART: binary decision trees for classification and regression built via recursive partitioning, impurity-based splitting, and cost-complexity pruning. It established decision trees as a principled, interpretable nonparametric method.

## Why it matters to NCE
Grounds the [[decision-tree-approximation]] surrogate in NCE, where a regression tree is fit to a bucket message and its leaf values are then optimized. CART's recursive partitioning offers a piecewise-constant alternative to [[neural-network-factors]] that connects naturally to [[quantization]].

## Citation confidence
Authors, title, and year are high confidence; publisher metadata is inconsistent across sources (Wadsworth Belmont vs Monterey vs Chapman & Hall reprint, ISBN 9780412048418) — treat the publisher field cautiously.

## Related
- [[decision-tree-approximation]]
- [[quantization]]
- [[neural-network-factors]]
