---
type: concept
title: Mini-Bucket Elimination
status: growing
tags: [inference, approximate, graphical-models]
created: 2026-06-12
updated: 2026-06-12
---

# Mini-Bucket Elimination

**Mini-bucket elimination (MBE)** is the bounded-cost approximation of
[[bucket-elimination]]: when a bucket's combined function would exceed $i$ variables, its
factors are **partitioned into "mini-buckets"** of scope $\le i$, and each mini-bucket is
eliminated independently. Cost becomes exponential only in the **[[iB-parameter|i-bound]]
$i$** instead of the [[induced-width]], and the result is a bound on the
[[partition-function]] (upper or lower depending on the operators). At $i=$ induced width
it recovers exact inference ([[@dechter2003minibuckets]]).

## Key points

- Splitting a bucket *decouples* shared variables, which is what loosens the bound;
  smaller $i$ = more splits = looser bound, cheaper compute.
- The weighted refinement, [[weighted-mini-bucket]] (WMB), adds Hölder weights to tighten
  the bound ([[@liu2011holder]]).
- **[[bucket-merging]] is the inverse operation** to mini-bucket splitting: NCE merges
  adjacent buckets to reduce approximations, where MBE splits them to bound cost.

## In NCE

- WMB partitioning lives in `FastBucket` (`compute_wmb_message`, `_create_mini_buckets`) —
  see [[bucket-structure]] and the migrated [[weighted-mini-bucket]] note. NCE's `ecl`
  (exact table-size limit) can override `iB` when partitioning; see [[terminology-map]].

## Sources

- [[@dechter2003minibuckets]] (JACM; Dechter & Rish) — origin. Conference version
  `dechter1997minibuckets`.

## Related

- [[bucket-elimination]] · [[weighted-mini-bucket]] · [[iB-parameter]] ·
  [[iterative-join-graph-propagation]] · [[bucket-merging]]
