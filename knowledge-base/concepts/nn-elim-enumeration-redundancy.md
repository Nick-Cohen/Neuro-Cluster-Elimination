---
type: concept
title: NN Elimination-Enumeration Redundancy
status: budding
tags: [this-project, performance, sampling, neural-network-factors]
created: 2026-08-12
updated: 2026-08-12
---

# NN Elimination-Enumeration Redundancy

When [[sample-generation]] evaluates a factor over a cluster's elimination grid, it must produce a
value for every assignment to the cluster's $e$ eliminated variables. A factor whose scope contains
only $e_f < e$ of them is **constant along the other $e - e_f$ axes**. The table implementation
exploits this; the [[neural-network-factors|NN]] implementation did not, and instead pushed
$k^{e}$ rows through the network where $k^{e_f}$ distinct rows exist. The waste factor is
$k^{\,e - e_f}$ — unbounded, and it grows with exactly the [[merge-bound]] that [[bucket-merging]]
pushes up.

## The asymmetry, in code

- **Table path** — `FastFactor._get_slices`, `nce/inference/factor.py` ~L271: builds
  `unexpanded_slice_shape = (len(assignments),) + tuple(v.states if v.label in tensor_labels else 1 for v in elim_vars)`,
  then `.expand()`s. Absent elimination variables get size 1 and are handed to a **stride-0 view,
  which costs nothing**.
- **NN path** — `FactorNN._get_slices`, `nce/inference/factor_nn.py` ~L91: does
  `torch.cartesian_prod(*[torch.arange(size) for size in elim_domain_sizes])` where
  `elim_domain_sizes` is the **whole cluster's** list. All $k^e$ configurations get written into an
  int64 coordinate cube, one-hot encoded and pushed through the net — including configurations that
  differ only in variables the network has never heard of. `FactorNN._eval_elim_block` has the same
  shape and the same problem.

Doc 03 verified the redundancy is exactly that and nothing subtler: it wrapped a real run and
checked with `torch.equal` that the returned tensor is **exactly constant** along each absent
elimination axis. NN scope 16 with $e=8, e_f=2$ → 256 rows for 4 distinct (64×); scope 21 with
$e_f=5$ → 8×. Pooled over the run: **15,773,696 network rows evaluated, 1,109,088 distinct =
14.2× redundant, every duplicate bit-identical.**

## Why it dominated the profile

At `e_max` = 12, sample generation is **83.8%** of elimination wall time, `_get_slices` is **93.4%**
of that, and the network's own arithmetic is **5.9%** (doc 03). Per-factor with
`cuda.synchronize()`: `FactorNN._get_slices` 701.3 ms against `FastFactor._get_slices` 31.0 ms.
The path is **CPU-dispatch bound** (191 ms CPU vs 88 ms CUDA), not FLOP bound.

## It is a merging bug, not a general one

Doc 06 answered "does 14.2× generalise?" with a structure-only CPU pass — 261 structures
(29 problem/sB configs × nomerge + reduce-NN/subsumption at `e_max` 4/8/12/16), 158,359 clusters,
10,029 NN messages, ~11 min, 0 failures, method validated three ways including reproducing doc 03's
14.22× exactly. The answer is **no**:

- **86% of NN factors have $e_f = e$** (74% among clusters eliminating ≥2 vars) and gain nothing.
  Per-factor redundancy is a spike at 1× with a heavy tail: median 1.0, harmonic 1.13, p90 4,
  max 82,944.
- Aggregate: **median merged config 1.50×, pooled 2.07×**. 30% of merged configs gain <1.05×;
  18% gain ≥10×.
- The gain is entirely a function of the merge arm: at `nomerge`, $e=1$ forces $e_f=e$ so the bug
  **cannot exist**; subsumption-only gains 1.01–1.07× throughout; reduce-NN gains
  **1.25 / 1.69 / 2.64 / 4.06×** at `e_max` 4 / 8 / 12 / 16. RBM is the extreme (32,768× at
  `e_max`=16); pedigree and grid40x40 ≈4×.

So 14.2× is real but roughly the **80th percentile**, not typical. It is "a merging bug, and
specifically a reduce-NN-at-high-`e_max` bug".

## The fix and what it actually bought

Branch `perf/nn-eval-fixes` (not merged, not pushed), two commits: `2b99709` builds the one-hot with
a single `scatter_`, `280eba7` restricts the NN elim enumeration to the elim vars actually in scope
(doc 09).

| config | before | after | speedup |
|---|---|---|---|
| wrap sB10 `e_max`=8, fix 1 only | 112.48 ms/call | 47.19 | 2.38× |
| wrap sB10 `e_max`=8, both | 115.95 | 4.54 | **25.5×** (rows 14.22×) |
| wrap sB10 rnn12 (`e_max`=12), both | 204.77 | 10.53 | 19.6× |
| wrap sB10 **sub20**, both | 2.299 | 1.519 | 1.51× (rows **1.00×**) |

`sub20` is the honest negative: subsumption-only leaves $e_f = e$ for every NN factor, so fix 2
removes zero rows and the whole 1.51× is fix 1 — exactly as doc 06 predicted. `_eval_elim_block`
was deliberately **not** given fix 2 (it receives a caller-supplied block, so restricting it needs
projection + `torch.unique`); it measured only 1.34× and had **0 calls** in all four real configs
probed. End-to-end at `num_epochs=1` the fixes are worth 1.29× on the whole elimination; at
production epoch counts training dominates and the wall-clock difference is not attributable to
the fix.

⚠️ **Two documents disagree on the measured numbers.** Doc 09 measured 25.5× at `e_max`=8 with
26/27 calls bit-exact and **one call off by exactly 1 float32 ULP**; doc 13's re-run of the same
harness reports **8/8 bit-exact, max diff 0.0, at 31.7×, redundancy 18.40×**. The range
"19.6–31.7×" quoted downstream (docs 15, 17, 25) spans both sessions.

⚠️ **Doc 09 corrects doc 03 on numerical risk.** Doc 03 §4.1 claimed *zero* numerical risk. Fix 1
is bit-identical everywhere. Fix 2 is not: the ULP deviation is caused by `chunk_size =
MAX_QUERY_ROWS // n_elim` now dividing by the *restricted* `n_elim`, which enlarges the GEMM's M
dimension so cuBLAS picks a different tile. This was proven independently — feeding the *same*
65,536 rows through the *same* weights at different batch sizes already disagrees by up to 8 ULP.
"Zero numerical risk" holds for the enumeration change but **not for the batching change it
implies**. This matters for [[bit-exact-reproducibility]] goldens.

## Downstream

Removing this cost collapsed sample generation as a share of runtime (SETUP 8.7–11.9× faster; on
grid40 at `e_max`=16 generation fell from 89% to 17% of the run), which is one of the two forces
that moved the [[time-optimal-merge-bound]].

## Related

- [[sample-generation]] · [[neural-network-factors]] · [[factor-operations]] · [[bucket-merging]]
- [[time-optimal-merge-bound]] · [[bit-exact-reproducibility]] · [[codebase-map]] · [[2026-W33]]
