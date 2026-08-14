# 60 — the frozen code version the paper rerun runs on

**Date:** 2026-08-14 · **Branch:** `frozen/rerun-v1` · **Worktree:** `/tmp/claude-58902/wt-frozen`
· gpu0/gpu3 for anything timed, **cuda:2 never touched**.

> ## THE FROZEN BUILD
>
> ```
> commit 38de3c5    branch frozen/rerun-v1
> ```
>
> **Code is frozen at `38de3c5`.** Everything committed after it is this document
> and its measurement artefacts; `git diff --stat 38de3c5 HEAD -- nce/` is **empty**,
> which is the check to run rather than take my word for it.
>
> Lineage: `9dbc2d0` (five validated fixes + proposal-sampling correctness)
> → `1c45dd3` (`integration/perf-merged`) → three merges → one integration commit.
>
> | commit | what |
> |---|---|
> | `1c45dd3` | base: perf layer (elim-block projection + table-path), already proven bit-identical together |
> | `90821e5` | merge `fix/wmb-backward-cluster-tree` |
> | `a480fc6` | merge `feat/crn-proposal-memo` (contains `feat/crn-default-on`, `exp/memorization-thresholds`) |
> | `41cbd6f` | merge `infra/gpu-scheduler` |
> | `38de3c5` | thermal ballast + the one composition defect found here |
>
> **Composition held, and it was measured, not assumed.**
> - Perf, against a properly-constructed same-CRN baseline: **312,750 / 312,750 replayed
>   real calls bit-equal, max deviation 0.0** — call-for-call the same total as doc 53,
>   now with CRN and the WMB-backward fix live.
> - WMB backward: build matrix **36/36**, pointwise **20/20 combinations, 1,308 clusters,
>   0 mismatches**, worst 1.83e-4 log₁₀ (float32 round-off).
> - CRN: goldens match, and are **verified by perturbation** to still detect a keying
>   defect. Not inferred from a passing run.
> - Suite: **exactly the 7 pre-existing failures**, no new ones.
> - Merge damage: **zero**, established by AST over 158 function bodies, not by reading diffs.
>
> **One real composition defect was found and fixed** (§4). It would have blocked
> **every job in the rerun**. It is exactly the kind of thing that only appears when
> branches meet: neither branch was wrong on its own.
>
> **Ballast: built, tested, and it holds the card outright** — across a 120 s gap an idle
> card falls **83 → 49 °C**, with ballast it stays at **83 °C**, yielding in **0.92–1.17 s**
> when a job arrives. **But it does not close the whole gap.** Measured on the real
> scheduler: the inter-job gap is only ~0.4 s (so there was little there to fill), while
> **8–9 s of GPU idle sits *inside* each job**, before its first GPU work, where ballast is
> forbidden to go. §6 is the honest accounting — read it before trusting any timing from
> this build.

---

## 1. What is in, and what is deliberately out

**In**

| branch | at | what it contributes |
|---|---|---|
| `integration/aug11-fixes` | `9dbc2d0` | base: five validated fixes + proposal-sampling correctness |
| `integration/perf-merged` | `1c45dd3` | elim-block projection (1.505x) + table-path (1.37x) |
| `fix/wmb-backward-cluster-tree` | `43eef99` | WMB backward over the cluster tree; was silently broken under merging |
| `feat/crn-proposal-memo` | `39694c9` | CRN default-on, proposal + memorization paths, collision-free seeds |
| `infra/gpu-scheduler` | `ac30ba9` | scheduler, checkpoint/restart, run provenance |

**Out — Nick's explicit decisions, not mine**

| branch | why out |
|---|---|
| `perf/validated-speedups` | Q21 approval **superseded by Nick on 2026-08-14**: "do not bother applying the three micro-optimizations given their ~0.2% combined end-to-end impact." |
| `perf/shared-encoding` | "very low priority… do not delay consequential work for an optimization that is ~1.09x end-to-end where it applies." |
| `feat/wmb-residual`, `exp/*`, `perf/nn-eval-2b` | experimental, not paper-path. (Note `exp/memorization-thresholds` **is** in, via `feat/crn-proposal-memo`, which merged it deliberately — see doc 57.) |

**Carried in that you should know about:** commit `9b3fe0c` on the CRN lineage
**untracks `.model_cache`** (237 files) and gitignores it. Deliberate there, and it has a
consequence for every run against this build — see §7 and §4.

---

## 2. The merges, and where they actually collided

`fix/wmb-backward-cluster-tree` merged clean into the perf layer: doc 54's claim of zero
file-level overlap holds (`bucket.py`/`graphical_model.py`/`message_gradient_factors.py`/
`backward_message.py` vs `factor.py`/`factor_nn.py`/`sample_generator.py`).

`feat/crn-proposal-memo` **did not**. It touches four of the WMB-backward files and one of
the perf files, because **both branches independently fix the same defect class** — an
unsanitised config handed to a derived `FastGM` — along different axes. Three conflicts,
all "both branches inserted at the same anchor":

| file | HEAD side | CRN side | resolution |
|---|---|---|---|
| `nce/inference/factor.py` | perf's `_SlicePlan` class | `_FACTOR_ID_COUNTER` + `itertools` import | both; independent |
| `nce/inference/message_gradient_factors.py` | disable the four merge passes in the derived GM | drop `proposal_sampling` from the same config | both; neither subsumes the other |
| `nce/utils/backward_message.py` | same | same | both |

Taking both sides is right here, but doc 53's lesson is that "take both sides" is exactly
the resolution that silently produced a bodyless `_elim_table` and a duplicate
`_get_slices` last time. So it was **verified structurally**, not by reading the diff.

### 2.1 AST verification — `60_ast_verify.py`

Parses base (`9dbc2d0`), both parents and the merged tree, and compares **function bodies
by qualified name** across every file touched by more than one merged branch.

| | |
|---|---|
| files checked | **7** |
| function bodies checked | **158** |
| byte-matching a parent exactly | **156** |
| verified line-wise as genuine both-sides combines | **2** |
| duplicate definitions | **0** |
| bodyless functions not present in any source | **0** |
| definitions belonging to **neither** parent | **0** |
| definitions in both parents but lost in the merge | **0** |

The 2 combines are the two config-sanitisation sites, each verified as
*base + parent-1's added lines + parent-2's added lines* with nothing dropped:

```
message_gradient_factors.py::get_wmb_message_gradient_factors   base + 9 from P1 + 4 from P2
backward_message.py::get_backward_message                       base + 26 from P1 + 4 from P2
```

`factor.py`'s conflict is module-level, which the body check does not reach, so it was
checked separately: the merged module's top-level statement list is an **exact multiset
union** of both parents' additions with nothing dropped.

---

## 3. Verification that composition held

### 3.1 Perf — bit-identical against a *same-CRN* baseline

**Constructing the baseline is the whole difficulty and it is worth being explicit.**
A two-worktree A/B against `9dbc2d0` is now **invalid**: CRN default-on deliberately
changes every sampled number, so the two arms would differ for a reason that has nothing
to do with the perf changes. The correct baseline is built **in-process**: run the frozen
build, and for every real call re-execute the **verbatim `9dbc2d0` implementation**
(extracted with `git show`, not re-typed) on the *same inputs in the same process*, and
compare with `torch.equal`. Both sides then see identical inputs, so CRN's effect on
*which* assignments are drawn is common-mode and cancels exactly. That is `53_compose.py`,
re-run unmodified on the frozen build.

| config | card | `FastFactor._get_slices_prepared` | `FactorNN._get_slices` | `FactorNN._eval_elim_block` | max dev | log Z |
|---|---|---|---|---|---|---|
| `dbn/rbm_21` e_max=16 | gpu0 | **292,158 / 292,158** | **4,395 / 4,395** | — | **0.0** | `62.968421936035156` |
| `pedigree19` masked e_max=16 | gpu3 | **14,176 / 14,176** | **727 / 727** | **1,152 / 1,152** | **0.0** | `-58.972801208496094` |
| `grid10x10.f10.wrap` merged e_max=8 | gpu3 | **134 / 134** | **8 / 8** | — | **0.0** | `333.6826477050781` |
| **total** | | **306,468** | **5,130** | **1,152** | **0.0** | |

**312,750 / 312,750 bit-equal.** Every call count reproduces doc 53's **exactly** —
including the 5,130 `FactorNN._get_slices` calls on the sample-generation small path, the
surface neither perf branch's own corpus covered. That the counts are unchanged while the
log Z values all moved is precisely the expected signature: **CRN changes the numbers, not
the control flow**, and the perf rewrite is still bit-exact underneath it.

For comparison, doc 53's pre-CRN log Z values were `62.315547943115234`,
`-59.27334213256836`, `332.1557922363281`. **All three moved. That is intended** (Nick:
"backward compatibility with the old random streams is not important"), and it is why
bit-identity to `9dbc2d0` was *not* the target.

### 3.2 WMB backward — its own evidence survives

Re-run on the frozen build with `54_probe.py`:

| check | result |
|---|---|
| build matrix (6 strategies × 2 routes × bounds 10/16/24) | **36 / 36 OK**, 0 clusters with a broken upstream |
| pointwise vs independent ground truth, `grids/grid10x10.f10`, bounds 10 & 16 | **20 / 20 combinations, 1,308 clusters, 0 upstream mismatches, 0 downstream mismatches** |
| worst deviation | **1.83e-4 log₁₀** — float32 round-off of the exact elimination, matching doc 54 |

Doc 54's 41 regression tests are in the suite and pass (§3.4).

### 3.3 CRN — goldens verified *sensitive*, by perturbation

The pairing, shared-prefix and collision-freedom tests pass in the suite
(`test_common_random_numbers.py`, `test_memorization_crn.py`,
`test_proposal_sampling_path.py`, `test_determinism_guard.py`,
`test_factor_order_lint.py`).

But a golden that matches proves nothing on its own — a golden regenerated against a
broken build matches that broken build perfectly. So `60_golden_perturb.py` establishes
sensitivity by **injecting defects and requiring the golden to catch them**:

| check | result |
|---|---|
| **A** unperturbed vs committed golden | **matches** (`305.3193054199219`, epochs `[10]`) |
| **B** repeated in-process | **identical to A** — the run is deterministic at all |
| **G1** `crn.stream_key ^ 1` (a CRN **keying** defect) | **DETECTED** → `303.9815368652344` |
| **I1** one extra global-RNG draw per NN cluster | **no change** — informational, and a *good* result: it confirms doc 56's collision-free derivation really did remove global RNG state as an input |

G1 is the check that matters: it is the defect class that making CRN default-on actually
risks, and the golden catches it. Note the live hook is `stream_key`, **not** `stream_seed`
— the latter serves only the no-replacement generator, and a perturbation there is a no-op
for the uniform scheme. Perturbing the wrong function would have produced a falsely
reassuring "no change".

**One honest limitation, measured rather than glossed** (`60_golden_numeric.py`). A
one-ULP bump on a single input factor entry does **not** move the golden. That is not a
broken probe — the script asserts the mutation applied (`before != after` on the stored
tensor) — it is washed out. Sweeping the magnitude:

| relative change to one factor entry | 1e-7 | 1e-6 | 1e-4 | 1e-2 |
|---|---|---|---|---|
| golden log Z moves? | no | no | no | **yes** |

So the log Z golden is a **stream/ordering guard, not a fine-grained numerics guard**: it
polices which samples are drawn and in what order, and it will not notice a numeric drift
below ~1e-2 relative on a single factor. Nothing in the rerun depends on it doing more
than that, but do not cite it as evidence of numeric equivalence.

### 3.4 Test suite — gated on the failure list, not the count

```
7 failed, 377 passed, 15 skipped   (6m35s)
```

The failure list is **exactly** the pre-existing 7, unchanged:

```
tests/test_benchmark_configs.py  ... test_default_configs_no_backward_ecl
                                 ... test_no_backward_ecl
                                 ... test_no_num_batches_per_set
                                 ... test_prepare_config_no_warnings
                                 ... test_bw_ecl_zero_config
                                 ... test_bw_ecl_positive_config
tests/test_config_docs.py        ... test_every_schema_field_documented
```

The pass count is not the invariant (GPU-dependent tests skip when cards are busy, and the
CRN branch adds tests); the list is. `tests/test_scheduler_resume.py` passes, so the
bit-identical resume property from doc 55 survives composition.

### 3.5 End to end

Three configs spanning three problem families, all completing with finite log Z and sane
epoch counts (early stopping well inside the 500 cap, so the counts are real, not truncation):

| family | config | log Z | epochs |
|---|---|---|---|
| dbn | `dbn/rbm_21` e_max=16 | `62.968421936035156` | 32 / 41 / 194 |
| pedigree | `pedigree19` masked e_max=16 | `-58.972801208496094` | 71 |
| grids | `grid10x10.f10.wrap` merged e_max=8 | `333.6826477050781` | (exact path) |

Plus two jobs driven through the **actual scheduler** end to end
(`grids/grid10x10.f10`, `reduce_nn` D=4, seeds 42/43): both `done`, rc=0, 60.3 s each,
manifests written with full provenance.

---

## 4. The composition defect that was found — and it would have blocked the rerun

**`nce/scheduler/models.py::cache_root()` ignored `NCE_MODEL_CACHE`.**

It rebuilt the cache path from `catalog_utils.__file__`, while
`catalog_utils.get_catalog()` — the function that *actually loads the model* — resolves
`NCE_MODEL_CACHE` first. Two different answers to "where is the model cache".

On `infra/gpu-scheduler` this was invisible, because `.model_cache` was tracked and
therefore present in every worktree. The CRN lineage **untracks and gitignores it**
(`9b3fe0c`). In the merged build the two resolvers disagree, and since validation runs
*before* dispatch by design, the scheduler marked **every job** `BLOCKED` with
"model missing" — while the loader would have found the model perfectly well via the
override.

```
[blocked] grids_grid10x10.f10__reduce_nn4__s40__...: .../wt-frozen/.model_cache/grids/grid10x10.f10.uai: missing
```

Neither branch is wrong on its own. This is purely an artefact of the two meeting, it is
silent in the sense that it fails *safe* (blocks rather than corrupts), and it would have
stopped the rerun dead on the first sweep. Caught by three dispatch tests in
`tests/test_scheduler_dispatch.py`. Fixed in `38de3c5`: `cache_root()` now resolves
exactly as the loader does.

**No other pair of branches failed to compose.**

---

## 5. Ballast — what was built

`nce/scheduler/ballast.py`, wired into `Dispatcher.sweep()`. When a dispatchable card
carries no job, a `BallastPool` worker runs a saturating fp32 GEMM (doc 59's measured
4 × 8192² workload) on it; the worker is torn down before a job launches there.

**The interaction that would have silently disabled the scheduler.** `gpus.py` treats
"a compute process is attached" as the authoritative busy signal — correctly, since that is
a fact and not a threshold. Ballast attaches one. Wiring it in naively makes every
ballasted card read **BUSY** and the scheduler stops dispatching entirely: the box would sit
100% utilised and 0% productive, and it would look like a scheduler bug rather than a
ballast bug. Hence `ignore_pids` threaded through `query_gpus`/`dispatchable_gpus`, with
the ignored pids' **memory subtracted too** — otherwise the secondary memory signal
re-flags the card and defeats the pid filter.

The three requirements, and how each is met structurally rather than by tuning:

| requirement | mechanism | evidence |
|---|---|---|
| yields instantly for a real job | SIGTERM/SIGINT set a flag checked between GEMM batches, with a per-batch `cuda.synchronize` so the queue cannot run ahead of the flag; `stop()` then blocks on `proc.wait()`, which is what actually frees the CUDA context | measured **0.00 s** yield in the live scheduler run; a real-GPU test asserts the context is gone after `stop()` and that no SIGKILL was needed |
| never touches gpu2 | three independent guards: pool filters `RETIRED_GPU_INDICES`; worker calls `assert_not_retired` on its own `--gpu` *before importing torch*; `ensure()` never widens the caller's free set | `test_pool_refuses_retired`, `test_ensure_never_ballasts_retired`, `test_worker_refuses_retired` (subprocess, asserts non-zero exit) |
| never perturbs a running job | by construction: ballast only ever runs on cards the scheduler knows carry no job, and is stopped **before** dispatch; cards are independent thermal and compute domains | `test_ensure_drops_cards_not_free`, and the dispatch path stops ballast before `_launch` |

`tests/test_ballast.py`: **12 tests, all passing**, including the real-GPU SIGTERM yield.

---

## 6. Ballast — what it actually buys, measured, including what it does not

This section is more negative than I would like, and it is the part worth reading.

### 6.1 It holds the card

`60_ballast_hold.py` on **gpu0**: warm to equilibrium with ballast (doc 59's 360 s
warm-up), then hold a 120 s gap — the length of gap a scheduler produces — with ballast
either left running or stopped.

| arm | equilibrium | min temp during gap | temp at end of gap | mean power in gap | drop |
|---|---|---|---|---|---|
| **no ballast** | 83 °C, 280 W | 49 °C | **49 °C** | 25.3 W | **−34 °C** |
| **ballast** | 83 °C, 279 W | **83 °C** | **83 °C** | 279.2 W | **0 °C** |

**It works, unambiguously.** A card left idle falls **83 → 49 °C in 120 s** — worse than
doc 59's 83 → 61 °C in 60 s, because this gap is twice as long. With ballast the card does
not move at all: minimum temperature during the entire gap is 83 °C, i.e. a job dispatched
at any instant in that window starts from equilibrium rather than from 49 °C.

Yield latency, measured on this run: **0.92 s and 1.17 s** — both well inside the
documented stop timeout, and neither needed SIGKILL.

### 6.2 But the gap it fills is mostly not where the bias comes from

`60_ballast_thermal.py` ran the **real** scheduler over a real 2-job queue, both arms, on
gpu0, sampling telemetry every second. Two findings, both against the premise:

**(a) This scheduler barely produces an inter-job gap.** `_reap()` and dispatch happen in
the *same* `sweep()`, so when work is queued a card is handed its next job within
**~0.4 s** of finishing the last. Ballast never even started during the queued phase — it
engaged only once the queue drained. The "cards cool between jobs" premise assumes a gap
this scheduler does not create while it has work.

**(b) The real idle window is *inside* the job, where ballast is not allowed to go.**
Measured, on both arms:

| | ballast arm | no-ballast arm |
|---|---|---|
| GPU idle after dispatch, before first GPU work | **8.5 s / 9.0 s** | **7.7 s / 9.3 s** |
| fraction of job wall time the GPU is genuinely busy | **43%** | **45%** |

That idle is the runner's CPU-bound setup — process start, torch import, model load,
elimination order — plus teardown. Ballast is stopped *before* the job launches, precisely
so it cannot perturb the job's timings, so **it cannot cover this window**. Covering it
would need the runner to signal "about to touch the GPU, yield now", which is a larger
change than this task, and it trades directly against the no-perturbation requirement.

**(c) For light configs there is no heat to preserve anyway.** `grid10x10.f10` `reduce_nn`
D=4 runs at **62 W, 6% utilisation, 1350 MHz, 41–43 °C**. The two arms' dispatch
temperatures were **41 °C vs 40 °C** — indistinguishable. Doc 59's effect was measured on
a card at 82 °C under a 280 W GEMM; the paper's exposure lives in the *heavy, long* configs
(doc 59 measured NCE at 155–170 W / 46–59%), not in configs like this one.

### 6.3 What this means for the rerun

Ballast is worth keeping — it costs nothing when jobs are queued back to back, and it does
remove cooling across queue-drain gaps, blocked jobs, and the intervals between scheduler
invocations. But **it does not by itself remove the bias doc 59 measured**, because a
material part of that bias is the ~8–9 s of in-job GPU idle before each job's first GPU
work, which scales the same way with job length. Two things follow, and both are decisions
for Nick rather than calls I should make:

1. **Warm-up is still mandatory** (doc 59 item 3, 6 minutes) — ballast does not replace it.
2. Either accept the residual in-job cold start, or have the runner hold ballast through
   its setup phase and yield on a readiness signal just before the first GPU work. The
   second closes the gap properly but puts a ballast process on the same card as a starting
   job, which is exactly what requirement 3 forbids today.

Until one of those is chosen, **do not treat this build's timings as free of the doc-59
bias.** Accuracy numbers are unaffected — they are bit-deterministic and clock-independent.

---

## 7. How to reproduce a run against this build

**Every run must record: commit, config, seeds, hardware.** The scheduler's
`manifest.json` already captures all four per job (`provenance.py` records git commit and
dirty flag, the resolved config, the seed, and `gpu_uuid_in_use` + `gpu_physical_index`).
Use it rather than a lab-notebook line.

```bash
# 0. Get the exact build.
git fetch origin
git worktree add -b rerun-local /path/to/wt frozen/rerun-v1
cd /path/to/wt
git rev-parse HEAD        # must be 38de3c5 (or a descendant that leaves nce/ alone)
git status --porcelain    # must be EMPTY -- provenance records dirty=true otherwise

# 1. Point at a populated model cache. REQUIRED: .model_cache is untracked and
#    gitignored on this lineage, so a fresh worktree has none, and dbn/* cannot
#    be re-downloaded. Never `rm -rf .model_cache && ln -s`.
export NCE_MODEL_CACHE=/home/cohenn1/NCE/.model_cache

# 2. Build a queue. Idempotent -- re-running adds only new cells.
python -m nce.scheduler.enqueue --queue Q.json \
    --problems grids/grid10x10.f10 dbn/rbm_21 \
    --strategies reduce_nn nomerge --bounds 4 8 --seeds 42 43 \
    --base-config base.json

# 3. Look before dispatching.
python -m nce.scheduler.scheduler --queue Q.json --status

# 4. Run. Ballast is ON by default; --no-ballast disables it.
#    TIMING-BEARING runs: gpu0 and gpu3 only.
python -m nce.scheduler.scheduler --queue Q.json --out-dir runs/ --threads 1 --only-gpus 0 3
#    ACCURACY-ONLY runs may also use gpu1 (thermally throttled, ~10% slow, numbers correct):
python -m nce.scheduler.scheduler --queue Q.json --out-dir runs/ --threads 1 --only-gpus 1
```

**Hardware rules, unchanged and non-negotiable**

- **cuda:2 is RETIRED** (silent data corruption). The scheduler refuses it three ways; do
  not route around them. A passing assertion on gpu2 proves nothing.
- **gpu1: accuracy only.** Correct numbers, ~10% slow, sustained. Mark its timings invalid.
- **gpu0 / gpu3: timing-bearing.** Both verified stable for 25 min at the full 280 W cap
  with the thermal bit never set.
- **Warm each card ~6 min before the first timed job** (§6.3), and balance card assignment
  across arms — gpu0 and gpu3 differ by 1.44% at steady state, which would otherwise be
  confounded with the result.
- Pin `--threads 1`. The determinism suite measured a 2-float32-ULP shift between 1 and 4.

**Per-run record.** From `runs/<job>/manifest.json`, the four required fields are
`git.commit` + `git.dirty`, `config` (resolved), `spec.seed`, and
`hardware.gpu_uuid_in_use` / `hardware.gpu_physical_index`. Aggregate `result.json` for
outcomes.

**Do not mix pre- and post-fix numbers.** Nothing measured before 2026-08-14 is comparable
to this build: CRN default-on moved every sampled number, and the WMB-backward fix changed
backward messages on 155 of 263 clusters. Any comparison must be entirely within
`38de3c5`.

---

## 8. Artefacts

All under `notebooks/_August-2026/claude_experiments/`, on `frozen/rerun-v1`:

| file | what |
|---|---|
| `60_ast_verify.py` | AST merge-damage check (§2.1) |
| `53_compose.py` | perf bit-identity replay, re-run unmodified (§3.1) |
| `60-compose-{rbm,ped,grid}.json` | its results |
| `60-wmb-p1.log`, `60-wmb-p3-catalog.log` | WMB build matrix + pointwise (§3.2) |
| `60_golden_perturb.py`, `60-perturb-grid.json` | golden sensitivity by perturbation (§3.3) |
| `60_golden_numeric.py`, `60-numeric.json` | golden numeric-resolution sweep (§3.3) |
| `60_ballast_thermal.py`, `60-thermal-{ballast,noballast}.json` | real-scheduler thermal A/B (§6.2) |
| `60_ballast_hold.py`, `60-ballast-hold.json` | direct ballast hold measurement (§6.1) |
| `_suite2.log` | full test suite (§3.4) |
