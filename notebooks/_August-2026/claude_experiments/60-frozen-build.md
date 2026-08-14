# 60 — the frozen code version the paper rerun runs on

**Date:** 2026-08-14 · **Branch:** `frozen/rerun-v1` · **Worktree:** `/tmp/claude-58902/wt-frozen`
· gpu0/gpu3 for anything timed, **cuda:2 never touched**.

> ## THE FROZEN BUILD
>
> ```
> tag frozen-rerun-v2    branch frozen/rerun-v1
> ```
>
> **`frozen-rerun-v2` supersedes `frozen-rerun-v1` (`38de3c5`).** v1 was never used for any
> run. v2 adds only orchestration: ballast covering in-job setup (§6.2), the `--warmup-s`
> option, per-phase timing in the manifest, and a hard `NCE_MODEL_CACHE` startup assertion.
>
> **v2 changes NOTHING numeric.** `git diff --name-only frozen-rerun-v1 frozen-rerun-v2 -- nce/`
> touches only `nce/scheduler/`; `nce/inference`, `nce/sampling` and `nce/neural_networks` are
> byte-identical. Every log Z in §3 therefore stands unchanged, and results produced under
> either tag are directly comparable. Run that diff rather than take my word for it.
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
> **Ballast: built, tested, and it closes the cold-start gap.** Across a 120 s idle gap a
> card falls **83 → 49 °C**; with ballast it stays at **83 °C**. It also owns each job's
> ~8 s CPU-bound setup window, releasing the card in **0.162 s** immediately before the first
> GPU operation — **proven non-perturbing: +0.219%** against an already-warm-idle reference
> (per-trial CV 2.75%), reproduced independently at +0.211% with a different mechanism.
> The bias being removed is **−5.25%**. With `--warmup-s 360` no job starts cold.
> **§6.3 corrects two numbers from an earlier draft of this document** — the "GPU busy only
> 43–45%" and "inter-job gap ~0.4 s" figures were artefacts of reading the scheduler's
> poll-quantised completion line as a job duration. Read §6 before trusting any timing.

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

### 6.2 Ballast also covers each job's setup window — and that is proven, not asserted

The window that actually matters is **inside the job**: the runner's CPU-bound setup
(process spawn, `import torch`, model validation, catalog load) runs for **~8 s** before the
job touches the GPU, and a card left idle falls from 83 °C to **~72 °C** in that time
(measured decay, §6.1 trace). Ballast now owns that window.

**Why this is not a perturbation.** The timed region begins at the job's *first GPU
operation*. Setup is, by construction, outside every timed region the rerun reports. So
ballast holding the card during setup and getting off before the first GPU op is not
interference — it is the window ballast should own.

**Mechanism.** `Dispatcher._launch` hands the card over **warm**, with ballast still on it,
passing the runner a pause-file and the worker's own paused-marker.
`runner.pause_ballast()` is called in the last CPU-only instant before
`FastGM(...)` — the first GPU work in the process — and **blocks until the worker itself
attests it has stopped**. If that attestation never arrives it **raises and the job fails**,
because a job timed against a card that is also running ballast is silently wrong, which is
worse than a failed job.

**Ballast pauses rather than exits.** An exiting worker costs ~5 s to respawn (process +
`import torch` + CUDA context), and the next job is dispatched within ~0.4 s of the
scheduler reaping the last one, so an exiting worker could never be back in time to cover
the *next* job's setup. A paused worker frees its matrices and queues no kernels, holding
only the bare CUDA context. That is a **weaker** guarantee than a dead process, which is
exactly why it had to be measured.

**The proof** (`60_ballast_handoff.py`, gpu0, CUDA-event timed, ~1,400 trials per arm). The
reference is an *already-warm idle card*; the question is whether the handoff arm matches it.

| arm | temp at start | median | CV | vs reference |
|---|---|---|---|---|
| cold (idle, ambient) | 36 °C | 39.159 ms | 3.38% | **−5.25%** |
| **warm-idle (reference)** | 79 °C | **41.330 ms** | 2.77% | — |
| **handoff** (ballast held setup, then paused) | 80 °C | **41.420 ms** | 2.75% | **+0.219%** |

**+0.219% against a per-trial CV of 2.75% — the handoff is clean.** During the handoff arm's
timed region the ballast process was still alive and its CUDA context still attached (both
`True`, as the pause contract intends) and it still made no measurable difference. Pause
latency: **0.162 s**. The card sat at **83 °C throughout the 9 s setup** and entered the
timed region at 80 °C.

Corroboration from an independent earlier run of the same harness against the older
kill-based handshake: **+0.211%**. Two mechanisms, two runs, same answer.

The **−5.25%** cold row is the bias being removed, and it reproduces doc 59's +5.87%
cold→steady drift. The thermal traces make it visceral: the cold arm's card climbs
**42 → 63 °C across its own timed region** (still warming while being measured), while
warm-idle and handoff are **flat at 81 °C from the first sample to the last**.

### 6.3 Two numbers in an earlier draft of this document were wrong

Both came from trusting the scheduler's own `[done] … after 60.3s` line as a job duration.
It is not one: `_reap()` only runs on the sweep boundary, so a completion is reported
**quantised to the poll interval** (default 30 s). Timing the same job standalone under
`/usr/bin/time` gives **36.39 s**, and the runner's internal accounting sums to 36.13 s —
the 60.3 s figure was ~24 s of scheduler *noticing* latency.

Consequently:

- **"The GPU is genuinely busy only 43–45% of job wall time" was an artefact** of that
  inflated denominator. Against the true 36.4 s wall the GPU-active share is **~76%**. The
  rerun is *not* half setup-bound. **Do not use the scheduler's `after Xs` line as a job
  duration** — use `manifest.timing.t_elim_s` / `total_wall_s`.
- **"The inter-job gap is only ~0.4 s" was also wrong.** 0.4 s is the gap between the
  scheduler *noticing* and dispatching. The card is actually idle from job end to the next
  poll — **~24 s**, far more idle time than the setup window. Fixed: the runner now deletes
  the pause-file as soon as its GPU work is done, so ballast resumes immediately rather than
  up to a poll interval later.

### 6.4 Where a 36.4 s job's time actually goes

Measured per phase from `manifest.timing.phases_s` (grid10x10.f10, reduce_nn D=4, gpu3):

| phase | seed 42 | share | what it is |
|---|---|---|---|
| spawn + interpreter | 0.16 s | 0.4% | `Popen` to first line of `main` |
| pre-timer setup | 8.00 s | 22% | **of which `import torch` + nce imports = 7.59 s**; then model validation, config, manifest |
| model load | 0.01 s | — | page-cached `.uai` |
| ballast pause | 0.38 s | 1% | the handshake |
| FastGM build | 3.35 s | 9% | elimination order + factor upload |
| **elimination (timed region)** | **24.22 s** | **67%** | of which NN training 21.5 s |
| teardown | ~0.3 s | 1% | |

**The dominant per-job overhead is `import torch` at ~7.6 s**, which is per-process and
irreducible without a persistent worker pool. It is ~21% of a 36 s job and would be ~0.5% of
a 1500 s job, so it matters for sizing a sweep of many short jobs and is negligible for long
ones. It is *not* a timing bias in the paper's numbers — it sits outside `t_elim_s` — but it
is real wall-clock cost when planning the sweep.

### 6.5 What this means for the rerun

Ballast now covers all three windows: between jobs, after a job finishes (immediately, not
one poll later), and each job's own setup. The only job that can still start cold is the
**first on each card**, because `sweep()` would otherwise dispatch in the same pass that
starts ballast.

- **The 6-minute warm-up is once per card, not per job.** An earlier draft of this document
  said "still mandatory" without that qualifier and it was read as per-job; doc 59 item 3
  always meant once, before the first timed job. There is no several-hundred-percent
  overhead on short jobs, and no length-correlated bias from warm-up.
- `--warmup-s` (opt-in) closes the remaining first-job case by holding every dispatchable
  card at load before the first dispatch. Doc 59 measured time-to-equilibrium at **330 s
  (gpu0) / 180 s (gpu3)** and recommends 360 s. Off by default because it delays the first
  job, which is a scheduling decision rather than one to impose silently.
- With `--warmup-s 360`, **no job in the sweep starts cold** and the residual thermal bias is
  the +0.219% measured above, i.e. inside noise.
- Accuracy numbers were never affected — they are bit-deterministic and clock-independent.

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
#    --warmup-s 360 holds every card at load before the FIRST dispatch, so that
#    job does not start cold either (doc 59: equilibrium 330 s gpu0 / 180 s gpu3).
#    TIMING-BEARING runs: gpu0 and gpu3 only.
python -m nce.scheduler.scheduler --queue Q.json --out-dir runs/ --threads 1 \
    --only-gpus 0 3 --warmup-s 360
#    ACCURACY-ONLY runs may also use gpu1 (thermally throttled, ~10% slow, numbers
#    correct). Warm-up is pointless here -- do not pay for it:
python -m nce.scheduler.scheduler --queue Q.json --out-dir runs/ --threads 1 --only-gpus 1
```

If `NCE_MODEL_CACHE` is unset or points somewhere without models, `enqueue`, `scheduler`
and `runner` now **all refuse to start**, naming the variable. A sweep can no longer begin
and then block every job on a missing cache.

**Read timings from the manifest, never from the console.** The scheduler's
`[done] … after Xs` line is quantised to the poll interval and overstates job duration (§6.3).
The reportable numbers are `manifest.timing.t_elim_s` (the timed region),
`total_wall_s`, and the per-phase split in `manifest.timing.phases_s`.

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
