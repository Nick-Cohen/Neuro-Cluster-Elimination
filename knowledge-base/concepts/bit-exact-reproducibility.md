---
type: concept
title: Bit-Exact Reproducibility
status: budding
tags: [this-project, correctness, reproducibility, determinism, tooling]
created: 2026-08-12
updated: 2026-08-12
---

# Bit-Exact Reproducibility

Until 2026-08-11 an NCE run was **not reproducible at a fixed seed**, and the cause was not
floating-point associativity in the usual sense — it was Python's `set` iterating
**identity-hashed objects**. Because the perturbation entered *before* training and was then
amplified by early stopping, it produced a $\log Z$ spread large enough to be mistaken for
seed variance. Fixing it turned the project's headline metric into something a regression test
can pin.

## The mechanism, in four steps

1. **Identity-hashed containers set the factor order.** `FastGM._create_buckets_from_factors`
   did `unplaced_factors = set(factors)`. [[factor-operations|FastFactor]] defines neither
   `__hash__` nor `__eq__`, so `set` falls back to `id()`-derived hashing and iterates in
   **memory-address order** — which differs between processes *and* between two builds inside
   one process. The same pattern existed in `PyGMsWMBBackward`, where
   `list(set(A) | set(B))` ordered identity-hashed pyGMs mini-bucket nodes (doc 21).
2. **Factor order is the order of a log-space sum.** `bucket.factors` order is the order in
   which the log-space product is accumulated, so reordering changes the result in the last
   bits — measured at **3–4 float32 ULP** on training targets (doc 21). It can also change the
   [[mini-bucket-elimination|mini-bucket]] partition outright, because
   `_partition_into_miniBuckets` sorts stably.
3. **SGD chaos amplifies it.** A 3–4 ULP change to the targets diverges over training. Doc 21
   quantified the amplification on the validation trajectory: run-to-run relative noise is
   157× *smaller* than the per-epoch improvement over epochs 1–50, but crosses over around
   epoch 100 and by epochs 200–354 **96% of epochs improve by less than the noise**.
4. **The end result is a spread that looks like seed variance.** Five runs of unmodified code,
   same config, same seed 42, spanned `333.4752`–`333.9194` — a **0.44 $\log Z$ spread**, ~78×
   the GPU-level nondeterminism previously blamed on gpu2, and present on gpu0/1/3 (doc 09).
   Doc 21's own direct replication of that configuration measured **0.350** at $n=3$; the
   0.44/0.35 gap is across code bases and was never reconciled. Quote 0.35 if you want a number
   doc 21 measured itself.

## Early stopping is *not* the amplifier — measured, against expectation

The obvious story is that the NeuroBE patience rule's bare `val < prev_best` lets the last bit
decide the stopping epoch. Doc 21 tested it and the story is **false**:

| cell | ordering | `neurobe_es_min_delta` | n | $\log Z$ spread |
|---|---|---|---|---|
| U0 status quo | legacy | 0.0 | 6 | 0.0410 |
| U1 hardened rule | legacy | 1e-4 | 6 | **0.0804** (worse) |
| E stopping removed | legacy | — | 5 | **0.1135** (worse still) |
| F0 / F1 ordering fixed | fixed | 0.0 / 1e-4 | 6 / 6 | **0.000000** |

Removing early stopping *increases* the spread; hardening the rule increases it. And there is no
threshold that would have worked: across 353 epochs of three runs, **zero** epochs had a relative
improvement in $(0, 10^{-4})$, and a binding $\delta$ would have to be ≈5e-2 — larger than the
median per-epoch improvement in every band past epoch 50. Doc 21's verdict on
`neurobe_es_min_delta` is that it removes **nothing**.

Fixing the ordering, by contrast, takes the spread to exactly zero — 12/12 identical on CPU
(`303.0346984863281`) and 3/3 on CUDA (`333.630249`) — at **no measurable wall-clock cost**
(225 s → 219 s on gpu1).

## The fix

Branch `fix/determinism`, commit `d5cca0e` (not merged as of 2026-08-12; branched from
`safety/worktree-snapshot-2026-08-11` = `4a1f8b6`). Four files, +66/−17:

- `graphical_model.py` — `set(factors)` → `list(factors)` with an order-preserving placement loop.
  **This is the whole fix.**
- `pygms_wmb_interface.py` — ordered de-duplication (`id()` in a `_seen` set, append to a list)
  in place of `list(set(A) | set(B))`. Same bug; it did not bite in the probed config (`bw` was
  `None` on that load) but it orders the backward `factor_list` and the `sum(...)` over it.
- `train.py` + `config_schema.py` — new `neurobe_es_min_delta` (relative improvement threshold,
  default `0.0` = historical behaviour). Shipped for completeness and measured **useless for
  determinism**; keep it, don't rely on it. Also re-applies the `val_losses` instrumentation from
  `2ae8cc1` — see [[convergence-diagnostic-gap]].

**Verified**: 12/12 replicates bit-identical at a fixed seed — $\log Z$ *and* every per-bucket
epoch count — on cells with 1, 1, 3 and 12 NN buckets (doc 23); 10/10 more across pedigree and
`masked_net` arms with 1–15 NN clusters, CPU and CUDA, doped and undoped (doc 28).

## How much of the published error bars was this?

The honest answer, measured on 4 cells of the June reduce-NN study (doc 23):

- Artefact share of $\log Z$ **variance**: 1% / 2% / 5% on grid cells, **31%** on `rbm_21 rnn10`.
  Pooled ≈ 10% of variance, ≈ 6% sd inflation.
- **Seed variance dominates the artefact in all four cells**, so the study's error bars were
  mostly honest. Re-running would tighten intervals and buy reproducibility, not rewrite
  conclusions — *but* it moves point estimates by $O(\text{artefact sd})$, so **gaps below
  ~0.24 $\log Z$ were never actually decided**.
- Caveats stated in doc 23: $n=3$ everywhere, 4 cells of ~500, no extrapolation to all 27 problems.
- The RBM cell shows how large a single-cell artefact can get: three pre-fix replicates *at the
  same seed* gave 62.6371 / 63.1341 / 63.1372 — **a 0.50 $\log Z$ gap opened by nothing but the
  factor-product association order** — with the last bucket's epoch count landing on 164 / 346 / 137.
- ⚠️ Doc 23 is internally inconsistent about the percentages: its TL;DR quotes 1/2/5/31% (the
  $\log Z$ table) while its abs-error table gives 2/3/5/31% for the same four cells, and the
  pooled sd inflation is "~6%" in the TL;DR vs 5.6% in the body. Cite the $\log Z$ row and say so.

## The bug is fixed; the *class* of bug is not

`FastFactor` **still has no `__hash__`/`__eq__`** (docs 21, 28, 30). Any future `set()` or `dict`
over NCE factor objects silently reintroduces this. No stable key and no lint/CI rule was
implemented. Doc 21 audited the ~40 other `set(...)` sites in `graphical_model.py`, `bucket.py`
and `elimination_order.py` and found them safe because they hold **ints** (`hash(x) == x`), and
`self.buckets` is keyed by a pyGMs `Var`, which hashes on its label. That audit is a snapshot,
not a guarantee.

Separately, doc 21 §6.2 lists nondeterminism hazards that were **audited but never exercised** by
the tested configs: float `scatter_add_` in the no-replacement samplers, `nn.Embedding` backward
in `net.py`, ungenerated `torch.randperm`/`torch.randint` on the global RNG in `data_loader.py` /
`factor.py` / `graphical_model.py`, and several `torch.manual_seed` calls made *mid-training-loop*
inside `losses.py`. Any of these can bite a config outside the tested set.

## Loose ends and corrections

- **Doc 21/23 misattributed the residual suspicion to `masked_net`.** `masked_net` is a Linear
  value head + Linear mask head over a Linear trunk and contains **no `nn.Embedding`**; the
  tree's only `nn.Embedding` is in `BitVectorLookup`, which is dead code. The masked branch also
  calls `loss.backward()` directly, bypassing the AMP `GradScaler`, so doc 21's AMP-amplifier
  concern does not apply to it either (doc 28).
- **Documents disagree on `torch.use_deterministic_algorithms(True)`.** Doc 21 reports it "raises
  nothing — on CPU *or* on CUDA". Doc 28 measured it **raising** on CUDA, a `RuntimeError` on
  `F.linear` demanding `CUBLAS_WORKSPACE_CONFIG=:4096:8`; with the variable set the run completed
  and returned an identical $\log Z$ at **2.52×** wall time. Doc 28's reading is that doc 21 had
  the variable set in its environment but did not record it. Doc 30 sides with doc 28, hard-codes
  a `require_cublas_workspace_config()` guard, and independently measures **2.59×** on a different
  cell. **Practical upshot either way**: set the env var, or the check dies before reaching any
  interesting kernel and tests nothing.
- **Multi-threaded CPU is its own axis**: `cpu_pedigree_plain` moves by 2 float32 ULP between 1
  thread and 4/8 threads (stable *within* a thread count), so the CPU regression tier pins
  `torch.set_num_threads(1)`. `PYTHONHASHSEED` and the OMP/MKL env vars turned out **not** to be
  needed (doc 30).
- **Scope**: complete for what is tested (3 problem families, both net arms, 1–15 NN clusters,
  CPU+CUDA); **partial as a codebase claim** — grid20/grid40, `sub*`/nomerge, multi-threaded CPU
  and the `scatter_add_` sampler paths are untested (doc 28).

## What it unlocked: the regression suite

Doc 19 recorded the consequence plainly — "the project cannot currently write an exact
regression test for ANY change". Doc 30 wrote it once the fix landed: branch `fix/determinism`
`f212869`, `tests/test_determinism_regression.py` + `tests/goldens/determinism_goldens.json`,
**no `nce/` file touched**. 13 CPU tests in 23.8 s (single-threaded, no GPU); 19 CUDA tests
(`--gpu`) in ~8.8 min; 9 pinned goldens each carrying provenance (branch, commit, GPU model,
torch build, thread count, host, date). Beyond the goldens it pins the *structural* invariant —
a two-build bucket-factor-**order** digest, which is the doc-21 bug's own cheapest reproducer at
<2 s, build only. See [[codebase-map]].

## Why this note exists

End-to-end $\log Z$ was never a valid regression target for trained configs before this fix
(doc 09), and several 2026-08 experiments (docs 18, 22) explicitly ran *before* `d5cca0e` and
carry that caveat. When reading any pre-2026-08-11 number, assume a run-to-run floor of the
order of 0.2 $\log Z$ unless the doc says otherwise.

## Related

- [[error-accumulation]] · [[factor-operations]] · [[bucket-structure]] · [[weighted-mini-bucket]]
- [[convergence-diagnostic-gap]] — the other reason early stopping was hard to reason about
- [[codebase-map]] · [[2026-W33]]
