---
type: project
title: Merge Bound (max_merge_bound, e_max)
status: growing
tags: [this-project, parameter]
created: 2026-06-12
updated: 2026-08-12
---

# Merge Bound (`max_merge_bound`, `e_max`)

The **merge bound** $e_{\max}$ is the single control on [[bucket-merging]]: the maximum number
of variables a merged cluster may eliminate. Config key **`max_merge_bound`**.

> **Notation.** Written $D$ before 2026-08 and $e_{\max}$ after; the predicted cost-optimal
> value is $e_{\max}^{*}$ (was $D^{*}$). The code was **not** renamed. It caps a *variable
> count*, not a state-space product — see [[terminology-map]] for the full rename block and
> the recorded disagreement about which semantics were intended.

## Key points

- $D=1$ → no merging (plain [[neural-bucket-elimination]]); $D\to\infty$ → unbounded merges
  that reintroduce the $2^{\text{width}}$ blow-up NeuroBE exists to avoid.
- **Sweet spot $D \approx 8$–$16$**: captures essentially all the accuracy gain before
  per-cluster exact cost (≈ $2^{D}$ sample/table complexity) dominates runtime. See the
  cost-cliff table (paper §5.3): for pedigree13 the NN-count floor (26) is reached by
  $D\approx12$, while $\log_2$ sample complexity climbs 11 → 31 as $D$ goes 1 → ∞.
- Interacts with `ecl`: a cluster that grows past `ecl` stays neural; one that *collapses*
  under `ecl` becomes exact. See [[terminology-map]].

## Naming history

⚠️ Renamed from `max_cluster_size` → **`max_merge_bound`** on 2026-06-04 (backward-compat
alias kept). Prefer "**merge bound**", not "merge degree" / "cluster size", in writing —
see project memory `feedback_merge_degree_naming` and [[terminology-map]].
⚠️ 2026-08: prose symbol $D \to e_{\max}$, $D^{*} \to e_{\max}^{*}$ (docs 25, 27).

## 2026-08 caveat: the sweet spot is not a property of the method

The $D \approx 8$–$16$ band above was measured on pre-August code. The
**time-optimal merge bound is code-version-dependent** and moved after the August fixes —
see [[time-optimal-merge-bound]] (doc 17). Any $e_{\max}$ recommendation must name the
code version it was measured on.

## Related

- [[bucket-merging]] · [[super-bucket]] · [[error-accumulation]] · [[iB-parameter]] · [[induced-width]]
- [[time-optimal-merge-bound]] · [[terminology-map]]
