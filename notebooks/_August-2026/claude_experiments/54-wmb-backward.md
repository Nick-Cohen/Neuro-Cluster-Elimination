# WMB backward approximation over the cluster tree — is it correct under every merge strategy?

> ## Verdict
> **Yes — now. It was not before this branch, and not for the reason anyone was looking at.**
>
> Doc 10 fixed the *population* of backward factors. Nobody had checked the *consumption* of
> them. Four more defects of the same class were found, all downstream of doc 10's fixes:
>
> **The one that matters:** `get_backward_message` rebuilt its downstream GM from a verbatim
> `deepcopy(gm.config)`, so all four merge passes re-ran **inside the backward GM**. Measured on
> `grids/grid10x10.f10` under subsumption, cluster 0: the returned backward message spanned
> **11 variables against a 2-variable separator** — nine variables that must be summed out were
> not. **68/69 populated clusters affected under subsumption, 27/27 under sub+nonsub.** It did
> not crash. It fed a wrong-shaped, wrong-valued backward message straight into NN training.
>
> Also: `_get_backward_factors` raised `KeyError` under *every* merge strategy, which is why
> `use_bw_approx` could never be run under merging at all.
>
> After the fixes: build matrix **36/36**; upstream/downstream pointwise-correct against
> independent ground truth on **20/20** strategy × route combinations; the backward message as
> actually consumed during a live elimination sweep is exact to float32 round-off on **12/12**
> combinations across two problems. **41 new regression tests fail on `9dbc2d0` and pass here.**
>
> `use_bw_approx` coupling: **was real, now broken** — `FastBucket.get_backward_factor_list()`.
> The `fw_bw` arm is unblocked.
>
> **Remaining, not fixed:** merge-quality bug in `merge_non_subsumption` (doc 10 §5.3, unchanged,
> not a correctness bug); no trained end-to-end run (see §6).

**Date:** 2026-08-14
**Branch:** `fix/wmb-backward-cluster-tree`, worktree `/tmp/claude-58902/wt-wmbbw`, forked from
`integration/aug11-fixes` (`9dbc2d0`). Commits `c21c711` (fixes) + `5a87d29` (tests + accessor).
**Not merged, not pushed.**
**No GPU was used** — every result here is CPU and deterministic. cuda:2 was never touched.
**Collision check:** the perf branches touch `factor.py` / `factor_nn.py` / `sample_generator.py`;
this branch touches `backward_message.py` / `message_gradient_factors.py` / `graphical_model.py` /
`bucket.py`. **Zero file-level overlap.**

---

## 1. How the search was run

Doc 10 fixed four defects at the sites it inspected; doc 44 §5.2 found a fifth of the same class at
a site doc 10 did not inspect. Rather than wait for a sixth to surface, the class itself was
enumerated: **every construction of a temporary/derived `FastGM` from a config the caller did not
sanitise.** `grep -n "FastGM(" nce/` gives eleven sites. Six are user-facing entry points. Of the
remaining five derived-GM sites, doc 10 had fixed two and doc 44 one. **The other two had never
been looked at, and both were broken.** A fourth site — `_get_backward_factors` — came out of the
second hazard class (per-variable indexing of `gm.buckets` while walking `gm.elim_order`).

This is the whole method. It is cheap and it should be re-run whenever a new derived GM is added.

---

## 2. The four defects

All confirmed by measurement on `grids/grid10x10.f10`, `iB=10`, `ecl=1025`, `max_merge_bound=10`.

### D1 — `get_backward_message`'s downstream GM re-merged  *(silent, worst)*

`nce/utils/backward_message.py:97`. `downstream_config = copy.deepcopy(gm.config)` set
`populate_bw_factors = False` and nothing else. All four merge flags survived into a GM built from
a **factor list**, not from the primary bucket tree. The merge passes then merged clusters that
`eliminate_variables(all_but=bucket_scope)` had no way to reconcile with the target scope.

Measured, subsumption, cluster 0:

```
true separator                    [1, 10]
pre-fix  backward message labels  [1, 4, 10, 13, 23, 32, 34, 42, 43, 51, 60]
post-fix backward message labels  [1, 10]
```

Nine variables never summed out. Across 263 populated clusters × 5 merge strategies, comparing the
backward message computed with the merge flags left on against the same computation with them off,
from the *same* `approximate_downstream_factors` and the *same* `backward_ecl`:

| strategy | clusters | changed >1e-4 log₁₀ | max \|Δ\| | notes |
|---|---|---|---|---|
| subsumption | 69 | 68 | **wrong scope** | 68/69 returned a message over the wrong variable set |
| sub+nonsub | 27 | 27 | **wrong scope** | 27/27 |
| non_subsumption | 28 | 15 | 6.71 | |
| reduce_nn | 69 | 42 | 9.25 | see below |
| merge_degree | 70 | 3 | 3.10 | |

**155 of 263 clusters had their backward message changed by the defect.**

The `reduce_nn` row is worth its own sentence: there the re-merge did *not* break the scope, it
made the backward GM **ignore `bw_ecl`** (max error against exact was 1.2e-4 with the defect vs
9.25 without it). `reduce_nn_merge`'s whole purpose is to merge away clusters that exceed the
limit, so running it inside the backward GM systematically removed the WMB partitioning that
`bw_ecl` asks for, computing a near-exact backward message at unbounded cost. Both behaviours are
wrong — the config said `bw_ecl=1024`.

### D2 — `_get_backward_factors` was not merge-aware  *(hard crash)*

`nce/utils/backward_message.py:19`. Indexed `gm.buckets[var]` for every `var` in `gm.elim_order`.
Absorbed members are removed from `gm.buckets` but **stay** in `gm.elim_order` — doc 10 fixed
exactly this in `get_senders_receivers` and the identical pattern survived here.

This is the path taken whenever `use_bw_approx: True` and pre-populated factors are unavailable —
i.e. `populate_bw_factors: False`, or `populate_bw_skip_non_nn` skipped the cluster. Measured:

```
reduce_nn     KeyError: Var (45,2)
subsumption   KeyError: Var (26,2)
merge_degree  KeyError: Var (45,2)
```

**`use_bw_approx` could not be run under any merge strategy without `populate_bw_factors`.**

### D3 — `get_wmb_message_gradient_factors` re-merged  *(crash under 2 of 4 strategies)*

`nce/inference/message_gradient_factors.py:103`. Same as D1. Raises `KeyError` under subsumption
(`Var (4,2)`) and non-subsumption (`Var (76,2)`); silently re-merges under reduce-NN and
merge_degree. Reached from `nce/sampling/sample_generator.py`.

### D4 — `FastGM.get_wmb_message_gradient` passed `self.config` raw

`nce/inference/graphical_model.py:1555`. Not merely re-merging: `populate_bw_factors` was still
`True`, so it would recurse into backward-factor population. No in-tree caller today; fixed anyway
because it is a loaded gun.

### D5 (bonus, not merge-specific) — `return_factor_list` contract violated

`get_backward_message`'s three early returns ignored `return_factor_list` and handed back a bare
`FastFactor`. Every `return_factor_list=True` caller feeds the result to
`SampleGenerator.sample_tensor_product`, which **iterates** it. Reachable because merging creates
**empty-separator clusters** — measured on `grid10x10.f10` under non-subsumption, cluster 96
(`n_elim=10`, separator `[]`). Also fires without merging on small problems.

---

## 3. Build matrix

`grids/grid10x10.f10`, CPU, build only. 6 merge strategies × 2 population routes × 3 merge bounds
including the degenerate high ones.

| bound | none | subsumption | non_sub | reduce_nn | sub+nonsub | merge_degree |
|---|---|---|---|---|---|---|
| **10** | OK/OK | OK/OK | OK/OK | OK/OK | OK/OK | OK/OK |
| **16** | OK/OK | OK/OK | OK/OK | OK/OK | OK/OK | OK/OK |
| **24** | OK/OK | OK/OK | OK/OK | OK/OK | OK/OK | OK/OK |

(each cell is `wmb / tree_collect`.) **36/36 OK, 0 clusters with a broken upstream.**

Structural degeneracy at the high bounds is real, not nominal — at bound 24, `sub+nonsub` collapses
100 buckets into **7 clusters, max 24 elim vars each**; `non_subsumption` into 17 clusters with
max 21. Artefact: `54-p1.json`.

---

## 4. Correctness — not just population

Three independent tests, increasing in strength. Doc 10's §4.3 log-Z test is deliberately **not**
reused: it only checks that upstream ⊎ downstream integrates to Z, which is insensitive to how mass
is distributed and which doc 10 itself notes the *pre-fix* code also passed.

### 4.1 Pointwise, against ground truth derived independently of the populators

For each cluster, ground truth is built from the **merged bucket tree plus the original factors
only** — `subtree(key)` from `get_senders_receivers`, `upstream = ∪ originals in subtree`,
`downstream = the rest` — and compared **pointwise** against the populators' output, both
eliminated exactly (`bw_ecl = 2²²`) to `separator ∪ elim_vars`.

`grids/grid10x10.f10`, 10 combinations × 2 merge bounds:

| bound | strategy | route | clusters checked | upstream mismatches | downstream mismatches | worst \|Δlog₁₀\| |
|---|---|---|---|---|---|---|
| 10 | subsumption | wmb / tc | 84 / 84 | 0 / 0 | 0 / 0 | 1.5e-4 / 1.8e-4 |
| 10 | non_subsumption | wmb / tc | 38 / 38 | 0 / 0 | 0 / 0 | 1.5e-4 / 1.5e-4 |
| 10 | reduce_nn | wmb / tc | 94 / 94 | 0 / 0 | 0 / 0 | 1.5e-4 / 1.8e-4 |
| 10 | sub+nonsub | wmb / tc | 27 / 27 | 0 / 0 | 0 / 0 | 1.5e-4 / 1.5e-4 |
| 10 | merge_degree | wmb / tc | 96 / 96 | 0 / 0 | 0 / 0 | 1.5e-4 / 1.8e-4 |
| 16 | *(all five)* | wmb / tc | 16–96 | 0 | 0 | ≤1.8e-4 |

**20/20 combinations, 0 mismatches, worst 1.8e-4 log₁₀ = float32 round-off of the exact
elimination.** Same test on a self-contained 5×5 grid (no catalog): 20/20, worst 1.1e-5.
Artefact: `54-p3.json`.

### 4.2 In situ — the backward message as a consumer actually sees it

§4.1 uses build-time bucket state. That is *not* what `compute_message_nn` sees: by the time a
cluster trains, its children's messages have arrived and `get_message_scope()` has grown, while
`approximate_downstream_factors` were computed at build time. So: run a real forward elimination
sweep and at every cluster issue the exact call `bucket.py:447-468` issues, then check

- **A.** the backward message lives on the cluster's *live* separator,
- **B.** at `bw_ecl = 2²²` it equals, pointwise, the exact backward message obtained from the
  **live downstream buckets of that same sweep**,
- **C.** `return_factor_list=True` returns a list.

| problem | strategy | route | clusters | A scope bad | B value bad | C not-a-list | worst \|Δlog₁₀\| |
|---|---|---|---|---|---|---|---|
| grid10x10.f10 | none | wmb / tc | 99 / 99 | 0 | 0 | 0 | 0.0 / 1.8e-4 |
| grid10x10.f10 | subsumption | wmb / tc | 83 / 83 | 0 | 0 | 0 | 7.6e-5 / 1.8e-4 |
| grid10x10.f10 | non_subsumption | wmb / tc | 39 / 39 | 0 | 0 | 0 | 1.1e-4 / 1.5e-4 |
| grid10x10.f10 | reduce_nn | wmb / tc | 93 / 93 | 0 | 0 | 0 | 4.6e-5 / 1.8e-4 |
| grid10x10.f10 | sub+nonsub | wmb / tc | 29 / 29 | 0 | 0 | 0 | 1.1e-4 / 1.5e-4 |
| grid10x10.f10 | merge_degree | wmb / tc | 95 / 95 | 0 | 0 | 0 | 4.6e-5 / 1.8e-4 |
| **dbn/rbm_20** | *(all six)* | wmb / tc | 2–18 | 0 | 0 | 0 | **0.0** |

**24/24 combination-rows clean.** Artefacts: `54-insitu.json`, `54-insitu-rbm20.json`.
(`rbm_20`'s low cluster counts are the `MAX_TGT = 2¹⁸` guard skipping its wide separators, not a
gap in coverage of the strategies. Cache was pointed at `/home/cohenn1/NCE/.model_cache`.)

### 4.3 A side-result worth acting on: `tree_collect` is the more accurate route

The same in-situ harness at the realistic `bw_ecl = 1024` measures the **approximation** error of
each route against exact. This is not a bug, it is the quantity `bw_ecl` trades:

| strategy | route | clusters | \# with error >1e-2 | **worst \|Δlog₁₀\| vs exact** |
|---|---|---|---|---|
| none | wmb / **tree_collect** | 99 | 86 / 86 | 16.11 / **10.40** |
| subsumption | wmb / **tree_collect** | 83 | 82 / 82 | 16.11 / **10.40** |
| non_subsumption | wmb / **tree_collect** | 39 | 27 / 27 | 15.58 / **10.40** |
| reduce_nn | wmb / **tree_collect** | 93 | 83 / 83 | 15.82 / **10.40** |
| sub+nonsub | wmb / **tree_collect** | 29 | 28 / 28 | 16.11 / **10.40** |
| merge_degree | wmb / **tree_collect** | 95 | 84 / 84 | 15.82 / **10.40** |

`tree_collect` is uniformly better and the gap is **5–6 orders of magnitude** on the worst cluster,
because the `wmb` route WMB-eliminates once at population time *and* again at consume time, while
`tree_collect` stores the raw chain and approximates once. On cluster 0: 4.52 vs 0.19.
**Recommendation for the queued experiments: default to `populate_bw_via_tree_collect: true`.**
This is one problem and one `bw_ecl`; it is a strong hint, not a law.

---

## 5. The `use_bw_approx` coupling

Doc 44 §2.2 reported that `fw_bw` could not be run as an isolated arm. **Confirmed, and it was
worse than reported.** `config['use_bw_approx']` gates **three** things at the single site
`bucket.py:414`:

1. computing the backward factors at all (`get_backward_message` → `dataloader.bw_factors`),
2. `DataPreprocessor.use_bw_approx = True` — changes the normalising constant to
   `logsumexp(y+bw) − logsumexp(bw)` (`data_preprocessor.py:108`),
3. `bw_hat` becoming non-`None` in the loss, which adds it to both outputs and targets
   (`losses.py:96`).

Doc 44 named (1) and (2). (3) is separate and is *not* gated by the preprocessor flag — it follows
from `dataloader.bw_factors` being set — so simply not flipping the preprocessor would still have
changed training.

**Separated, in a small additive change:** `FastBucket.get_backward_factor_list()`
(`bucket.py:1428`). Reads the already-populated `approximate_downstream_factors`, returns the
backward factor list, touches no trainer state. Requires `populate_bw_factors: True`; returns
`None` for a cluster `populate_bw_skip_non_nn` skipped.

```python
bw = bucket.get_backward_factor_list()      # use_bw_approx stays False
```

This only works because of D1 and D2 — on `9dbc2d0` this call `KeyError`s or returns a
wrong-scope message under merging, which is the actual reason the `fw_bw` arm was blocked.
**The `fw_bw` arm, IS conditions 5–6, and the residual work are unblocked.**

No new config field was added, so `test_config_docs` is untouched.

---

## 6. Limits — what this does NOT establish

- **No trained end-to-end run.** Every result is structural or tensor-level. No shipped config
  combines merging + `populate_bw_factors` + `use_bw_approx`, and inventing `num_epochs` /
  `num_samples` for one is exactly what the standing instruction forbids. §4.2 is the closest
  substitute: the real consumer call, on live sweep state, for every cluster. **If you want the
  trained run, say which config and it is a short job.**
- **Binary problems only** — `grid10x10.f10`, a synthetic 5×5 grid, `dbn/rbm_20`. Pedigrees
  (mixed domains) untested, per the standing instruction.
- §4.3's route comparison is **one problem, one `bw_ecl`**.
- `merge_non_subsumption` still re-initialises `_scope_at_elim` from the pre-merge
  `message_scopes` (doc 10 §5.3), so `sub+nonsub` makes its *merge choices* on stale scopes.
  Untouched — it is a merge-quality bug, and every population/consumption result above is correct
  for whatever clusters that pass happens to produce.
- `_wmb_eliminate_to_scope` with an empty target scope still trips
  `eliminate_variables(all_but=[])` (doc 10 §5.3). Not reachable on the paths tested; D5 makes the
  empty-separator case safe for `get_backward_message` specifically.

### Hazard classes checked and found CLEAN

Reporting these because a negative result is worth as much as a positive one here.

- **Scope caches read after merging changed the scope.** Checked `_cluster_separator` and
  `message_scopes` against the true separator observed in a simulated forward sweep, 4 strategies ×
  2 routes: **0 under-estimates.** There is a reason: every merge pass absorbs a child into its
  tree *parent*, and since `scope(parent) ⊇ scope(child) \ {child}`, the merged cluster's outgoing
  separator equals the parent's. **Merging along tree edges never widens a separator.** This also
  clears `proposal_scope_for_bucket`'s single-elim-var branch, which doc 44 §5.3 flagged as
  suspicious and left alone — it is fine, and now for a stated reason rather than by luck.
- **Per-level variable recording vs multi-elim-var clusters.** `build_proposal_tree` was doc 44's
  fix; re-checked here and clean. The population copy is asserted unmerged by an existing test.
- **Temporary GMs re-merging at build time.** Instrumented all four merge passes and counted calls
  on non-primary GMs during construction: **0**, for 4 strategies × 2 routes. Doc 10's fixes hold.
  The defects were all at *consume* time, which build-only instrumentation cannot see — that is
  the methodological lesson.

---

## 7. Repro

```bash
cd /tmp/claude-58902/wt-wmbbw            # branch fix/wmb-backward-cluster-tree
PY=/home/cohenn1/NCE/venv/bin/python
N=notebooks/_August-2026/claude_experiments

$PY -m pytest tests/test_wmb_merge_repair.py -q          # 114 passed, 6 skipped
BOUNDS=10,16,24 $PY $N/54_probe.py p1                    # 3: build matrix, 36/36
SKIP_NON_NN=0  $PY $N/54_probe.py p2                     # hazard scan H1/H2/H3
               $PY $N/54_probe2.py                       # consume-time hazards H1b/H4
P3_CATALOG=1 BOUNDS=10,16 $PY $N/54_probe.py p3          # 4.1: pointwise correctness
               $PY $N/54_impact.py                       # 2 D1: numerical impact
BW_ECL=4194304 $PY $N/54_insitu.py                       # 4.2: in situ, exact
BW_ECL=1024 OUT=54-insitu-bwecl1024.json $PY $N/54_insitu.py   # 4.3: route comparison
PROBLEM=dbn/rbm_20 BW_ECL=4194304 OUT=54-insitu-rbm20.json $PY $N/54_insitu.py
```

Pre-fix baseline worktree for the before/after: `/tmp/claude-58902/wt-wmbbw-base` at `9dbc2d0`.
Copying `tests/test_wmb_merge_repair.py` onto it gives **41 failed, 73 passed, 6 skipped**;
on the fix tree **114 passed, 6 skipped**.

| test | fails on `9dbc2d0` |
|---|---|
| `test_backward_factors_available_without_use_bw_approx` | 12/12 |
| `test_backward_message_honours_return_factor_list` | 12/12 |
| `test_get_backward_factors_is_merge_aware` | 8/12 |
| `test_message_gradient_factors_gm_does_not_remerge` | 5/12 |
| `test_backward_message_downstream_gm_does_not_remerge` | 4/12 |

## 8. Gates

- `pytest tests/` — **exactly the 7 pre-existing failures**, 6 in `test_benchmark_configs.py`
  and 1 in `test_config_docs.py`, no others. (Pass count 274 here; it is not a stable quantity —
  GPU-gated tests skip depending on what else is running.)
- `pytest tests/test_determinism_regression.py` — **13 passed, 6 skipped.**
