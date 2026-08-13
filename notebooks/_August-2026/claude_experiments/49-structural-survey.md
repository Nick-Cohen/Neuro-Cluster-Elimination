# 49 — Structural survey: shared encoding into streaming (Q30), cross-block NN reevaluation (Q31)

Build-only survey of the whole 29-config paper benchmark set × 3 merge strategies × merge
bounds D ∈ {8,16,24} = **203 structures, 5824 clusters**. No training, no message computation,
no GPU: 0.2–30 s per structure. One 6-second GPU microbenchmark on gpu0 supplies the cost
constant. Zero failures; nothing excluded.

---

## Decision 1 (Q30) — shared encoding into streaming: **DO NOT FUND**

**The population exists but is the wrong shape.**

| | clusters |
|---|---|
| all clusters surveyed | 5824 |
| with ≥2 co-resident NN factors | 1004 |
| triggering a chunked-streaming site | 224 |
| **satisfying BOTH** | **62** (1.1% of all; 6.2% of the ≥2-NN population) |

The 62 are worthless for this optimisation, for three independent reasons:

1. **Co-residency collapses to 2–3 wherever streaming triggers.** Max co-resident NN in *any*
   streaming cluster is **3**; the distribution is 53×2NN, 9×3NN. Shared encoding saves at best
   (n−1)/n of the encode cost, i.e. 50–67%. The big co-residency lives where streaming never
   fires: rbm_22 under subsumption has a **21-NN cluster** — but its whole joint is 2^23, an
   order of magnitude under the 2^28 chunk trigger, so it takes the dense path.
2. **Encoding is only 17.9–27.2% of the eval.** Measured on gpu0 (TITAN RTX) over eight shapes
   spanning the surveyed clusters. So the best case is ~0.6 × 0.22 ≈ **13% of NN-eval time, in
   62 of 5824 clusters** — and the true figure is lower, because in all 62 the co-resident nets
   have **pairwise-distinct input scopes** (n_encoding_groups == n_NN, always), so a shared
   cluster-level one-hot still needs a per-net column gather.
3. **They only appear at merge bounds the project does not run.** 30/62 at reduce-NN D=24,
   22/62 at subsumption D=24. At the canonical **D=16 the count is 10** (9 subsumption +
   1 reduce-NN), all exactly 2 NN factors. At D=8 and no-merge it is **0**.

**Where shared encoding *would* pay is the dense path, not streaming.** 942 non-streaming
clusters hold ≥2 NN factors, 35 of them hold ≥10, and 42 clusters have all their co-resident
nets on an *identical* input scope (the RBM 19/20/21-NN clusters). Those go through
`FactorNN._get_slices`, which builds a separate one-hot per factor over the same assignments —
21 identical encodings built 21 times. If shared encoding is funded at all, fund it there.

## Decision 2 (Q31) — cross-block NN reevaluation: **FIX IT** (but not with a cache)

**Frequency.** Of 150 (streaming cluster × NN factor) pairs in the sample-generation streaming
path, **77% are redundant**, **51% by ≥1000×**. Median redundancy **1024×**, max **3.98 × 10⁶×**.
In the chunked-exact path it is immaterial: median 1×, max 324×, ~0.3 h total.

**Cost, measured not guessed.** `FactorNN._eval_elim_block` costs **14–22 ns per
(assignment × elim-point) row** on a TITAN RTX (`49_bench_eval_block.py`). Worst single cluster,
grid40x40.f10.wrap under reduce-NN D=24: 8.31 × 10¹² rows for one net, of which 1 in 4096 is
distinct → **≈32 GPU-hours of pure redundant forward passes, in one cluster**. Summed over every
streaming cluster in the survey: **≈1125 GPU-hours** of redundant NN forwards. Even at the
canonical D=16, pedigree7's reduce-NN cluster wastes 2 × ~5 min.

**Root cause — and it is not a missing cache, it is a missing projection.**
`SampleGenerator.sample_tensor_product_elimination` hands `_eval_elim_block` the coordinates of
*all* of the cluster's elimination variables. A factor only reads the elim vars in its own scope,
`E_f = labels(f) ∩ elim_labels`, so its value repeats `elim_prod / k^|E_f|` times across the block
sweep. **`FastFactor._eval_elim_block` already handles this** (nce/inference/factor.py:281 —
"This factor only depends on the elim vars it actually contains", builds the small slice and
flat-indexes it), and so does `FactorNN._get_slices` (factor_nn.py:158–175, "enumerate ONLY the
elimination variables this factor's scope actually contains"). Only
**`FactorNN._eval_elim_block` (factor_nn.py:241) omits it** — it builds the full (B, n_labels)
coordinate cube and runs the net on every row. The fix is to port the existing projection: run
the net over the `k^|E_f|` distinct points, then gather back to (A, B). No cache, no memory
growth, no new approximation — and bit-identical output.

Reported, not fixed: this is shared inference code and `perf/*` branches are live in it.

**Order of work.** Q31 first. It removes 3–6 orders of magnitude from exactly the clusters Q30
was aiming at, which shrinks any remaining shared-encoding win in the streaming path further.

---

## Survey table — joint condition per merge strategy (summed over the 29 benchmark configs)

| strategy | D | NN clusters | max co-res NN | clusters ≥2 NN | S1 stream | S4 stream | **S1 ∧ ≥2NN** | **S4 ∧ ≥2NN** |
|---|---|---|---|---|---|---|---|---|
| no-merge | – | 3899 | 21 | 210 | 0 | 0 | **0** | **0** |
| subsumption | 8 | 926 | 21 | 207 | 0 | 0 | **0** | **0** |
| subsumption | 16 | 809 | 21 | 202 | 0 | 9 | **0** | **9** |
| subsumption | 24 | 788 | 21 | 202 | 16 | 14 | **12** | **10** |
| reduce-NN | 8 | 549 | 14 | 91 | 0 | 1 | **0** | **0** |
| reduce-NN | 16 | 266 | 6 | 58 | 5 | 18 | **1** | **0** |
| reduce-NN | 24 | 155 | 2 | 34 | 136 | 25 | **29** | **1** |

Joint clusters by family: grid40x40 48, pedigree 14, **grid-small 0, RBM 0, no other family**.
Sanity checks against known data: reduce-NN leaves 3–8 NN clusters on pedigree at D=16 and 1–5 at
D=24, and 0 on the cheap RBMs; rbm_22 under subsumption reproduces the known 21-network cluster
exactly (and reduce-NN drives rbm_21/22 to 0 NN at D=24); NN counts
under no-merge match the NeuroBE reference (BN_1=2, BN_3=1, 10_14_s.binary=3,
grid10x10.f5.wrap=1).

## The five chunk/stream sites (the "real trigger" Q30 asked for)

There is no single streaming switch. Only S1 and S4 ever have several NN factors alive at once.

| | site | trigger | multiple NNs co-evaluated? |
|---|---|---|---|
| S1 | `sample_tensor_product_elimination` large path | `prod(dom(elim vars)) > 2**18` | **yes** — every factor per block, same coords |
| S2 | same, small path → `FactorNN._get_slices` | otherwise | yes, but already scope-projected |
| S3 | `_get_slices` internal batching | `n_assign × n_elim > 65536` | no (one net) |
| S4 | `compute_message_exact` → `_compute_message_exact_chunked` | `prod(dom(scope)) > 2**28`, exact branch only | **yes**, via `_nn_factor_slice` when `stream_nn_exact` |
| S5 | `nn_to_FastFactor` | `prod(dom(net scope)) > 65536` | no (one net) |

`MAX_QUERY_ROWS = 65536` (S3/S5) is a per-network batching cap, not the cluster-level streaming
trigger; assuming it would have surveyed the wrong thing.

## Method — and where in the lifecycle this is measured

Bucket state changes during elimination, so co-residency and cluster scope are properties of the
**merged, mid-elimination** structure. The survey measures each cluster **at the instant
`process_bucket` dispatches on it** — after every earlier cluster has delivered its message —
by replaying `FastGM.eliminate_variables`' bookkeeping with the real code: the real merge passes
(they run in `FastGM.__init__`), the real `process_bucket` exact/NN predicate, the real
`find_next_bucket` routing, the real `FastBucket` scope/width/complexity accessors. Only the
*numeric content* of a message is replaced, by a stub carrying its label set, its `is_nn` flag,
and the same `get_factor_complexity()` a real message would report. Nothing else is simulated,
so nothing else can drift. Static join-tree message scopes and pre-elimination widths were not
used anywhere.

Estimates and their assumptions: redundancy factors and cluster counts are **exact**. The wall
times use the measured 14 ns/row and NeuroBE's `num_samples` formula at ε=0.1 with l=2, which
*understates* pedigree (domain 5). Merge flags are the only config values varied; `iB`/`ecl`
come from the benchmark set (ecl = 2^iB + 1), `device` is irrelevant (nothing is computed) and
the survey runs on CPU.

## Reusable machinery — where it lives, how to call it

Branch `tool/structural-survey`, worktree `/home/cohenn1/NCE-wt-survey`, commit 31bb561.
Not merged, not pushed. Additive only — no shared inference code touched.

- **`nce/analysis/structure_survey.py`** — the machinery. `survey_problem(key, iB, ecl,
  strategy, max_merge_bound)` builds and surveys in one call (`strategy` ∈ `nomerge`,
  `subsumption`, `reduce_nn`, `nonsubsumption`); `survey_gm(gm)` surveys an already-built FastGM
  (consumes it, as a real elimination does); `plan_exact_blocks(...)` mirrors the chunked-exact
  block planner. Returns `StreamingSurvey` → `.clusters` (`ClusterRecord`, one per cluster, with
  `.nn_factors`, `.s1_joint`, `.s4_joint`, `.n_shared_encoding_groups`) and `.summary()`.
  The module docstring documents the five streaming sites and the lifecycle point.
- **`notebooks/_August-2026/claude_experiments/49_run_structural_survey.py`** — the sweep runner
  (`… out.json [group_idx] [problem_idx]` for process-level parallelism).
- **`notebooks/_August-2026/claude_experiments/49_bench_eval_block.py`** — the gpu0 cost split
  (encode vs forward) of the real `_eval_elim_block`.

Raw output: `/tmp/claude-58902/-home-cohenn1-NCE/4a4f80a3-b352-42de-90d5-8b50eff6b71a/scratchpad/survey49_g*.json`.

Environment note: `.model_cache` resolves relative to the *package* directory, so a worktree gets
its own empty cache and silently re-downloads (rbm_22 then fails — the DBN `statistics.csv` in
the cache index no longer lists the rbm models). Symlinked the worktree cache to the main repo's.
