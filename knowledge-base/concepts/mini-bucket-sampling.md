---
type: concept
title: Mini-Bucket Sampling
created: 2026-03-01
tags: [inference, sampling, mini-bucket, anytime, graphical-models]
---

# Mini-Bucket Sampling

Mini-bucket sampling refers to anytime algorithms that augment WMB elimination with stochastic sampling to produce increasingly accurate estimates of partition functions and marginals as computation time increases.

## Core Idea

WMB elimination produces an upper bound on the partition function by independently eliminating variables from each mini-bucket. The approximation error comes from ignoring correlations between mini-buckets. Sampling can correct this error by:

1. Using WMB as a proposal distribution or importance sampler.
2. Generating samples from the WMB-defined distribution.
3. Using these samples to estimate the true partition function with provably decreasing variance as sample count grows.

The combined approach is **anytime**: the WMB bound is available immediately; sampling progressively tightens the estimate.

## Variants

### Abstraction Sampling (Dechter et al.)
Abstraction sampling uses mini-bucket (or WMB) as an "abstraction function" that stochastically compacts the state space. It combines:
- **Search**: Systematic traversal of states using the bucket elimination structure.
- **Stratified importance sampling**: Random samples from WMB-defined strata.

Key insight: the abstraction function defines a partition of the state space into strata. Within each stratum, exact or sampled evaluation is performed. This balances systematic accuracy with random exploration.

Reference: [Abstraction Sampling in Graphical Models](https://ics.uci.edu/~dechter/publications/r240.pdf)

### WMB + IS (Importance Sampling)
Use WMB messages as proposal distributions for importance sampling. The WMB distribution is a valid (but biased) distribution over variable assignments. Sampling from it and correcting with importance weights gives an unbiased estimator of the partition function.

### Gauged Mini-Bucket Elimination (WMBE-G)
Combines gauge transformations (local reparameterizations that preserve the partition function) with WMB. After running WMB forward, a backward sweep calibrates gauge parameters to tighten the bound. This can produce both upper and lower bounds.

Reference: [Gauged Mini-Bucket Elimination for Approximate Inference](https://arxiv.org/abs/1801.01649)

## Relevance to NCE

NCE is currently focused on replacing bucket messages with neural network approximations (NeuroBE-style). Mini-bucket sampling is relevant because:

1. **Backward messages**: The backward message computation uses WMB; sampling could provide better backward message estimates (analogous to using sampling to correct WMB bias in the forward pass).
2. **Training data**: Mini-bucket sampling could generate better-distributed training samples than uniform random sampling — samples from the WMB distribution concentrate on higher-probability assignments.
3. **Anytime approximation**: For very large models where even NN training is too slow, WMB + sampling could serve as a fast initial approximation before NN training begins.

## Cost Shifting and Moment Matching

Two techniques that improve WMB before sampling:

### Cost Shifting (Iterative Reparameterization)
Reparameterizes the factors across mini-buckets so the WMB bound is tighter. Factors are jointly modified in a way that preserves the true partition function, then WMB is run on the reparameterized model. Multiple iterations progressively tighten the bound.

Reference: [Beyond Static Mini-Bucket: Towards Integrating with Iterative Cost-Shifting](https://ics.uci.edu/~dechter/publications/r212.pdf)

### Moment Matching (WMB-MM)
After WMB partitioning, enforces that the marginal beliefs from different mini-buckets of the same bucket agree (match moments). This is a local optimization over reparameterization parameters within each bucket. The result, WMB-MM, is often the tightest single-pass WMB variant.

Reference: [Mini-bucket Elimination with Moment Matching](https://ics.uci.edu/~ihler/papers/discml11.pdf)

## Relationship to pyGMs WMB Interface

The NCE codebase includes `nce/utils/pygms_wmb_interface.py`, which wraps pyGMs' WMB implementation. The pyGMs WMB supports:
- Weight optimization (GDD: Generalized Dual Decomposition).
- Entropy-based learning.
- Upper and lower bound computation.

This interface provides access to more advanced WMB variants (including moment matching) than the basic NCE WMB implementation.

## Related

- [[weighted-mini-bucket]]
- [[variable-elimination]]
- [[backward-messages]]
- [[iB-parameter]]
- [[sample-generation]]
