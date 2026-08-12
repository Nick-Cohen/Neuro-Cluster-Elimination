---
type: concept
title: Weighted Mini-Bucket Elimination
created: 2026-03-01
tags: [inference, approximate, graphical-models, wmb]
---

# Weighted Mini-Bucket Elimination (WMB)

Weighted Mini-Bucket (WMB) elimination is an approximate inference algorithm that extends mini-bucket elimination with cost-shifting weights to produce tighter bounds on the partition function.

## Mini-Bucket Elimination

When exact variable elimination is infeasible (bucket scope too large), mini-bucket elimination **partitions** the factors in a bucket into smaller groups called **mini-buckets**. Each mini-bucket is eliminated independently, producing multiple smaller messages instead of one large exact message.

- The **i-bound (iB)** controls the maximum width of each mini-bucket. Factors are partitioned so that no mini-bucket has more than `iB` variables in its scope.
- This produces an **upper bound** on the partition function (or a lower bound, depending on formulation).

## Weighted Mini-Bucket (WMB)

WMB improves on basic mini-bucket by introducing **weights** on the mini-buckets. Instead of simply partitioning and eliminating, WMB uses a weighted power-sum operation:

- Each mini-bucket gets a weight `w_i` where the weights sum to 1.
- Instead of regular logsumexp for elimination, WMB uses a weighted logsumexp: `(1/w_i) * logsumexp(w_i * f)`.
- The weights can be optimized to tighten the bound.

## Variants

1. **Regular Mini-Bucket**: No weights, simple partitioning. Loosest bound.
2. **Weighted Mini-Bucket (WMB)**: Adds optimization weights. Tighter bound.
3. **WMB with Cost Shifting**: Redistributes potential across mini-buckets to tighten the bound further. Uses reparameterization to shift "costs" between factors.
4. **WMB with Moment Matching**: Enforces that the marginals from different mini-buckets agree (match moments). Tightest among these variants.

## In NCE

- Implemented in `FastBucket.compute_wmb_message()` in `nce/inference/bucket.py` and `FastGM._wmb_eliminate()` in `nce/inference/graphical_model.py`.
- The i-bound is set via `config['iB']`.
- WMB is used both in the forward pass (approximating messages) and in backward message computation.
- The number of WMB partitions is tracked in `FastGM.wmb_fw_partitions`.
- pyGMs provides an alternative WMB implementation accessed via `nce/utils/pygms_wmb_interface.py`, which supports weight optimization, GDD (Generalized Dual Decomposition), and entropy-based learning.

## Gotcha: `compute_wmb_message` under *approximate* upstream context

Two pre-existing crash bugs found 2026-08-11 while building [[wmb-residual-learning]]
(branch `feat/wmb-residual`; docs 14 §1.2, 16 §7). Both matter for **any** work that calls WMB
inside a real elimination sweep, not just residual learning:

1. **`compute_wmb_message` crashes on incoming `FactorNN` messages.** `_create_mini_buckets`
   partitions on `f.tensor.numel()`, and a `FactorNN` has `tensor is None` →
   `AttributeError: 'NoneType' object has no attribute 'numel'`. This fires on the **second
   and later NN clusters of any real elimination**, so any probe that only touched the first
   NN cluster silently ran with *exact* upstream context and never hit it. Fix: densify the
   elim-var-touching NN factors before partitioning, exactly as `compute_message_exact`
   already does. Pass-through factors ride along untouched.
   Note "WMB is merge-safe" is a **different** claim from "WMB is safe under approximate
   upstream context" — the first was verified, the second was false.
2. **`FactorNN` had no `_get_values`.** It inherited `FastFactor`'s, which indexes
   `self.tensor` (`None`). Needed by `SampleGenerator.sample_tensor_product` whenever a
   factor being evaluated is an NN pass-through — an upstream NN message that does not
   mention an elim var and so is returned untouched. Related:
   `sample_tensor_product` must **not** call `order_indices()` on NN factors, whose label
   order defines the network's input layout.

Both fixes live on `feat/wmb-residual`; they are pure crash fixes on existing code paths and
are worth landing independently of whether residual learning ships. **Still absent from
`perf/nn-eval-fixes`** as of 2026-08-12. In the current checkout the offending line is
`sorted(self.factors, key=lambda f: f.tensor.numel(), reverse=True)` in
`FastBucket._create_mini_buckets`, `bucket.py` ~L1248; the densification pattern to copy is in
`compute_message_exact`, `bucket.py` ~L58–69.

They are load-bearing and stress-tested: they fire on every run past the first NN cluster, and
survived a 400-variable model with **108 NN clusters**, and 11 NN clusters chained under
approximate upstream context *and* merging (docs 16 §6, 18 §2.5).

> The *backward* direction has its own, separate and much larger failure under merging — see
> [[backward-factor-population-under-merging]]. "WMB is merge-safe" was verified for the forward
> per-cluster estimate only.

## Key Parameters

- **iB (i-bound)**: Maximum mini-bucket width. Lower iB = more partitions = looser bound.
- **ECL**: When a bucket's exact complexity exceeds ECL, WMB (or NN) is used instead.
- **Weights**: Per-mini-bucket weights that can be optimized.

## Connection to Backward Messages

WMB is also used to approximate backward messages when the downstream graphical model is too large for exact computation. The `backward_ecl` parameter controls when WMB kicks in for backward message computation. See [[backward-messages]].

## Sources

- [[@liu2011holder]] — **the** WMB / Hölder-bound reference; the weights are Hölder
  exponents. Region-selection follow-up: `forouzan2015incremental`.
- [[@dechter2003minibuckets]] — the underlying [[mini-bucket-elimination]] scheme (Dechter
  & Rish). See also [[iterative-join-graph-propagation]] for the join-graph cousin.

> NCE's `iB` is the standard **i-bound**; `ecl` (exact table-size limit) is
> project-specific and can override `iB` when partitioning — see [[terminology-map]].
> [[bucket-merging]] is the inverse operation to mini-bucket splitting.

## Related

- [[mini-bucket-elimination]] · [[bucket-elimination]] · [[variable-elimination]]
- [[iB-parameter]] · [[partition-function]] · [[iterative-join-graph-propagation]]
- [[backward-messages]] · [[factor-operations]] · [[cluster-elimination]] · [[bucket-merging]]
- [[wmb-residual-learning]] · [[backward-factor-population-under-merging]]
- [[terminology-map]] — in NCE prose `iB` is written **s-bound (sB)**, not i-bound
