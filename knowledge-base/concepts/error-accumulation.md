---
type: concept
title: Error Accumulation
status: growing
tags: [neural, analysis, this-project]
created: 2026-06-12
updated: 2026-06-12
---

# Error Accumulation

**Error accumulation** is the core failure mode of [[neural-bucket-elimination]]: because
messages **propagate**, each learned (approximate) bucket message injects an error that
every downstream elimination folds in, so the final [[partition-function]] error grows with
the **number** of neural-approximated buckets — not merely the quality of each network.

## Key points

- A learned message $k$ carries error $\varepsilon_k$; bucket $k$'s output feeds bucket
  $k' > k$, whose own message now sits on top of $\varepsilon_k$. To first order the
  $\log Z$ error accumulates across the chain of approximated buckets.
- **Empirically the dominant predictor of final error is the count of NN buckets**, not the
  per-network training loss (NCE experiments, lab notebook 2026-05/06).
- This is the central motivation for [[bucket-merging]]: reduce the count of
  error-injection sites by fusing adjacent NN buckets into fewer (and sometimes exact)
  clusters.

## In NCE

- Quantified by the merge experiments: cutting NN count from ~30–130 down to ~20–30 cut
  $\log_{10}Z$ error ~6–7× (up to ~100× on the hardest grid). See [[bucket-merging]],
  [[merge-bound]], and `writeup/paper.md` §1, §2.4, §5.

## Sources

- [[@agarwal2022neurobe]] · [[@razeghi2021deep]] (the scheme whose error this analyzes).

## Related

- [[bucket-merging]] · [[merge-bound]] · [[neural-bucket-elimination]] · [[partition-function]]
