---
type: project
title: Bucket Merging (NCE contribution)
status: growing
tags: [this-project, method, contribution]
created: 2026-06-12
updated: 2026-06-12
---

# Bucket Merging (NCE contribution)

**Bucket merging** is NCE's contribution to [[neural-bucket-elimination]]: *before* training,
greedily combine adjacent NN-eligible buckets into larger clusters
([[super-bucket|super-buckets]]), so several buckets emit **one** learned-or-exact message
instead of several. This directly reduces the number of neural approximations — the driver
of [[error-accumulation]] — at the cost of wider exact sub-computations inside each cluster.
A single knob, the **[[merge-bound]]** `max_merge_bound = D`, caps how many variables a
cluster may eliminate ($D=1$ = plain NeuroBE; $D\to\infty$ = unbounded, reintroduces the
exponential blow-up). It is the **inverse operation** to [[mini-bucket-elimination]] splitting.

## Three strategies (in `graphical_model.py`)

1. **Subsumption merging — `merge_join_tree`.** Absorb a child whose elimination-time scope
   *contains* its parent's — the [[running-intersection-property|running-intersection]] /
   [[join-tree]] condition. Adds **no** new variable → "free" accuracy gain.
2. **Degree-bounded greedy — `merge_by_degree`.** Greedily merge the adjacent pair adding
   the fewest new scope variables; stop a cluster at $D$ eliminated variables.
3. **NN-count reduction (+ backtracking) — `reduce_nn_merge`.** For each NN bucket, absorb
   its ancestor chain until the cluster collapses under `ecl` (becomes exact) or merges into
   another NN cluster; backtracking re-runs at the *smallest* cap achieving the minimum NN count.

## Headline result (lab notebook 2026-06-03/04; paper §5.2)

| Problem | iB | No-merge (D=1) | Sweet spot | Gain |
|---|---|---|---|---|
| grid10x10.f10.wrap | 10 | err 3.81 | err 0.11 @ D≈8–16 (29 NN) | 6.7×, faster |
| grid20x20.f10 | 10 | err 15.84 | err 0.16 @ D≈8–16 (7 NN) | ~100×, faster |
| pedigree13 | 20 | err 2.49 (126 NN) | err 0.37 @ D=16 (26 NN) | 6.7× |
| rbm_22 | 10 | err 0.648 (33 NN) | err 0.094 @ D=6 (23 NN) | 7×, time halved |

No-merge NeuroBE is the **worst** accuracy point on every curve.

## Why a sweet spot

Per-cluster exact cost ≈ $2^{(\text{eliminated vars})}$, so past $D\approx12$–$16$ the NN
count already hit its floor while cost explodes — "merge just enough to hit the NN-count
floor, no further". See [[merge-bound]] and the cost-cliff table in paper §5.3.

## Systems work required (iB=20)

- streaming dense materialization (`factor_nn.py::nn_to_FastFactor`),
- vectorized sample slicing (`sample_generator.py::_get_slices`),
- memory-bounded exact elimination (`bucket.py::_compute_message_exact_chunked`).
See [[codebase-map]].

## Sources

- Builds on [[@agarwal2022neurobe]]; structural grounding [[@kask2005unifying]]
  (tree decompositions / super-clusters); inverse of [[@dechter2003minibuckets]] splitting.

## Related

- [[merge-bound]] · [[super-bucket]] · [[error-accumulation]] · [[neural-bucket-elimination]] ·
  [[nce-method-overview]] · [[cluster-elimination]]
