---
type: moc
title: MOC — Approximate Inference
status: evergreen
tags: [moc, approximate]
created: 2026-06-12
updated: 2026-06-12
---

# MOC — Approximate Inference

Bounded-cost approximations of [[bucket-elimination]] — NCE's classical baselines and
neighbors.

## Mini-bucket family
- [[mini-bucket-elimination]] — partition wide buckets to bound cost ([[@dechter2003minibuckets]]).
- [[iB-parameter]] — the i-bound knob.
- [[weighted-mini-bucket]] — Hölder-weighted bound ([[@liu2011holder]]); NCE's WMB path.

## Join-graph family
- [[iterative-join-graph-propagation]] — iterative message passing on a join-graph
  ([[@mateescu2010jgp]]).

## Clustering / merging
- [[super-bucket]] — eliminating several variables at once (merged cluster).
- [[cluster-elimination]] — NCE's multi-variable elimination.
- [[tree-decomposition]] — the structure these all re-partition.

Note: **[[bucket-merging]] (NCE) is the inverse of mini-bucket splitting** — merge to
reduce learned-message count, where MBE splits to bound cost.

→ Continue to [[MOC-neural-inference]].
