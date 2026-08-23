# 63 — Q60: giving the network the WMB estimate as INPUT

> ## Verdict — READ THIS FIRST (endpoint corrected 2026-08-22)
> **The headline below (§2, "input loses to residual on grids by −0.55 dex") rests on an
> endpoint that does not survive scrutiny, and end-to-end log Z does not support it.**
> On the *same four cells*, measuring **|log Z − reference|**:
> - `grid20x20.f10` agrees in direction (input worse, Δ = −1.39) — but the per-seed ranges
>   **overlap**, so it is not resolvable at 3 seeds;
> - `grid10x10.f10.wrap` **reverses**: end-to-end has `input` *better* than residual
>   (Δ = +0.12), where the local metric said it was 0.445 dex worse at p = 1.7e−02;
> - **not one comparison, on any cell, has separated per-seed ranges.**
>
> So the honest current state is **"the four-cell experiment cannot resolve these arms
> end-to-end"**, not "the prediction fails decisively". The `p = 1.6e−09` in §2 comes from
> treating clusters within a run as independent units; they are not (they share an
> elimination and the same trained upstream messages), which inflates significance
> without the end-to-end answer moving. §3's mechanism finding (the net learns a +0.82
> sensitivity on grids, +0.02 on RBMs) is a direct measurement and **stands unaffected**.
>
> **Status: full-benchmark re-run IN FLIGHT** (21 configs, gpu1, launched 2026-08-22).
> Final per-family verdicts land when it completes; see §0.

---

## 0. Full-benchmark extension (in flight)

Extended from 4 cells to the **21 non-grid40 configs** of the reduce-NN benchmark set:
pedigree ×6 (iB=20), grid-small ×6 (iB=10), rbm ×6 (iB=20), rbm ×3 (iB=10). Pedigree was
untested here and is the largest gap. `grid40x40` is **deliberately excluded**: measured
from the live sweep, `grid40x40.f15.wrap` at sub D=10 costs 41,734 s per run, so 8 configs
× 21 runs ≈ 1,948 GPU-hours ≈ 81 days on one card — a separate decision, not something to
slip in.

Each group uses **its own** iB/ecl from the benchmark set (`ecl = 2**iB + 1`); the old
`63_drive.sh` hardcoded `--ib 10 --ecl 1025` and could not express this, so the queue is
now `q60_queue.py` — restartable (it indexes completed runs by `(problem, iB, merge, D,
arm, seed)` read out of each JSON, so relaunching never redoes work) and watchdogged by
`scripts/q60_watchdog.sh`, because this box kills multi-day detached processes.

**Endpoint change, applied to every table below §2.** Primary is now
**end-to-end |log Z − reference|**, per problem, mean over 3 seeds, **with the per-seed
range shown**, and significance judged by whether those ranges separate — plus a sign test
over *cells*, which is the one sign test that is legitimate here since distinct benchmark
problems are independent. Per-cluster local error is retained as **secondary** and
disagreements are flagged. References are `results_for_writeup/problem_overview_table.csv`
`ref_log10Z` where `ref_kind == 'exact'` (17 problems); `rbm_ferro_20` has no exact
reference and is reported reference-free.

> **`pedigree51` reference — RESOLVED, not an open question.** `−73.871` is correct and
> supersedes `−77.27`; Nick confirmed 2026-08-22 ("the higher value one is the correct one"),
> and the correction is already on the record in `docs/25` U10 (exact-solver validated,
> checked against pedigree13 to 4e-6) and `docs/27` §8, which treats `−77.27` as the
> *pre-correction* value. The overview table this analysis reads already carries `−73.871`,
> so **no result here was affected**. The stale `−77.27` in
> `reduce_nn_experiment/benchmark_set.json` has been corrected to match.

Analysis: `q60_e2e_analyze.py --exp 63 /tmp/claude-58902/res63`.

---

Branch `exp/wmb-input-and-fwbw` off `frozen-rerun-v3` (`5d3c8d4`), pushed.
**gpu1 only**, strictly sequential, 44 runs + 4 calibration, zero failures.
gpu0/gpu3 carried the paper rerun and were never touched; gpu2 never used.
**All wall times are gpu1 and INDICATIVE ONLY** (~10% thermal throttle); this is an
accuracy experiment, which is why gpu1 suits it.
Artefacts: `63-results/` (48 run JSONs, `an_s4243.txt`, `an_all.txt`, `probe_*.json`),
`63_run.py`, `63_analyze.py`, `63_verify.py`, `63_probe_weight.py`, `63-PREREG.md`.

---

## 1. What was run

`wmb_input='combined'` appends **one** input column, the cluster's WMB estimate;
`'partitions'` appends **1+k** columns, the estimate plus each mini-bucket partition, so
the net can see the inter-partition dependence WMB's factorisation discards. Target is
the true message in both. Compared against plain NeuroBE (`base`) and residual learning
(`residual` = doc 31's arm 5ε: target `exact − WMB`, residual normaliser, eps-compensated).

4 cells (2 grids, 2 RBMs) × 4 arms × 3 seeds, iB=10, ecl=1025, subsumption merge D=10,
`hidden_sizes='neurobe,3'`, 2000-epoch cap with NeuroBE early stopping — doc 31's protocol,
i.e. the protocol that produced the residual numbers being compared against. CRN is
default-on in this build, so arms sharing a separator draw identical assignments.

**Two correctness gates passed before any result was read.**
1. With the flags off, `grid10x10.f10.wrap` reproduces pristine `frozen-rerun-v3`
   **bit-identically** (logZ and all 11 signed local errors, max diff 0.0). Purely additive.
2. Eval-path features replay training features **exactly**: `undo_normalization(net(x_train))`
   vs `FactorNN.to_exact()` indexed at the training assignments = **0.0** max difference on
   11/11 clusters, both modes. Without this the local errors would be plausible garbage.

Metric: paired per-cluster `|signed_local_error|`, unit `(problem, seed, bucket)`.
`gain(A vs B) = median[ log10|err_B| − log10|err_A| ]`, positive = A more accurate.
**Per family, never pooled** (doc 31 §4: pooling previously hid an entire negative result).

---

## 2. Result — `input` vs `residual` (SUPERSEDED local-error endpoint, seeds 42+43)

> Retained for the record. This is the **secondary** endpoint now; see the
> verdict block and §0 for why, and for the end-to-end numbers that replace it.

| cell | family | n | win/loss | **gain dex** | p_sign | p_wilcoxon |
|---|---|---|---|---|---|---|
| `grid10x10.f10.wrap` | grid | 22 | 5/17 | **−0.445** | 1.7e−02 | 1.3e−03 |
| `grid20x20.f10` | grid | 72 | 11/61 | **−0.553** | 1.6e−09 | 4.3e−10 |
| `dbn/rbm_20` | rbm | 40 | 23/17 | **+0.191** | 4.3e−01 | 1.5e−01 |
| `dbn/rbm_21` | rbm | 42 | 23/19 | **+0.030** | 6.4e−01 | 4.6e−01 |

**GRID median −0.499 dex · RBM median +0.111 dex**

**Verdict against the pre-registered prediction (`input ≥ residual` on grids,
`input > residual` strictly on RBMs):**
- **Grid half: FAILS DECISIVELY.** −0.45 and −0.55 dex, both significant.
- **RBM half: NOT SUPPORTED.** The point estimates are positive but neither cell is
  significant (p = 0.43, 0.64). The pre-registered data do **not** establish the strict
  inequality the prediction required. This is "insufficient evidence", not a win.

### 2.1 Protocol deviation, disclosed
The pre-registered rule was: seeds 42+43 in full, **seed 44 only if the two seeds disagree
in sign in any cell**. All three seeds ran anyway — the queued runs were never killed. The
larger set is therefore supplementary, not primary, and **it does change the RBM
conclusion**, so this matters:

| cell | seeds 42+43 (primary) | all three seeds (supplementary) |
|---|---|---|
| `grid10x10.f10.wrap` | −0.445, p=1.7e−02 | −0.375, p=1.4e−02 |
| `grid20x20.f10` | −0.553, p=1.6e−09 | −0.553, p=2.1e−13 |
| `dbn/rbm_20` | +0.191, **p=0.43 (n.s.)** | +0.248, **p=1.4e−02 (sig.)** |
| `dbn/rbm_21` | +0.030, p=0.64 (n.s.) | +0.067, p=0.13 (n.s.) |

Only with the extra seed does `rbm_20` reach significance. **The pre-registered answer on
the RBM half is "no significant difference"**, and that is the one reported. Reading the
three-seed row as the result would be exactly the quiet criterion-widening the
pre-registration existed to prevent.

## 2.2 Supporting comparisons (seeds 42+43)

| comparison | grid cells | RBM cells |
|---|---|---|
| `input` vs `base` | −0.355 (p=0.05), −0.111 (p=0.41) | +0.009 (p=1.0), −0.068 (p=0.88) |
| `parts` vs `input` | +0.146 (p=0.52), +0.106 (p=0.20) | +0.012 (p=1.0), −0.025 (p=0.88) |
| `residual` vs `base` | +0.259 (p=0.29), **+0.422 (p=2.4e−06)** | **−0.340 (p=1.7e−02)**, −0.138 (p=0.16) |

- **`input` never beats plain baseline anywhere.** It is a wash on RBMs and if anything
  worse on grids.
- **`parts` is null everywhere.** Exposing the separate mini-bucket partitions — the half of
  Q60 that looked most promising — bought nothing measurable. Median k was 2–3 partitions.
- **`residual` vs `base` reproduces doc 31/43** (+0.42 on `grid20x20.f10`, −0.34 on `rbm_20`,
  same signs and comparable magnitudes), which validates the harness.

---

## 3. Why: the mechanism works exactly as hypothesised, and it does not help

Nick's motivation was: *"if the WMB approximation is really bad, the NN will learn to ignore
that piece."* That is **measured and confirmed**, and it is the most interesting number here.

The feature was normalised with the target's own affine map precisely so that a unit
coefficient reproduces residual learning. So measure the coefficient the net actually
learned — `∂ŷ_norm / ∂base_norm`, averaged over the training rows, per cluster
(`63_probe_weight.py`):

| cell | median sensitivity | range | first-layer ‖W[:,wmb]‖ / median ‖W[:,onehot]‖ |
|---|---|---|---|
| `grid10x10.f10.wrap` (ρ_S≈0.91) | **+0.819** | 0.473 … 0.988 | 1.43 |
| `dbn/rbm_20` (ρ_S≈0.52) | **+0.019** | 0.0003 … 0.271 | 1.09 |

Residual learning is exactly `+1.000`. So:

- **On grids the net leans hard on WMB** (0.82, and up to 0.99 on the widest clusters) —
  it finds most of the way to residual learning on its own.
- **On RBMs it discards WMB almost completely** (0.019). Doc 43's rank-collapse story is
  reproduced from the *inside* of the network: where the base only half-orders the message,
  the net learns to ignore it, unprompted.

**And that is precisely why the design cannot win.** It reduces to
`input ≈ max(worse-than-residual, base)`:
- where WMB helps, reaching residual *through two tanh layers* is worse than the residual
  arm's exact additive term outside the network — the 0.82 (not 1.00), with per-row sd ≈0.06,
  is a reconstruction error on a quantity the residual arm gets for free;
- where WMB does not help, the net correctly throws it away and lands back on baseline,
  which is where the RBM "advantage over residual" comes from. It is residual's known loss,
  **not** a gain from the input design — confirmed by `input` vs `base` being null on RBMs.

### 3.1 Correction to my own design note
The implementation comment (and commit `7bf5099`) claims the normalisation makes the
residual hypothesis "**exactly** representable", so that any loss to residual would be
optimisation rather than expressiveness. **That claim is wrong for the architecture actually
used.** `hidden_sizes='neurobe,3'` is a two-hidden-layer **tanh** MLP with **no skip
connection**, so the WMB column reaches the output only through two nonlinearities; an exact
identity map is not representable by such a net. The three arms are *approximately* nested
on the bounded feature domain, not exactly. The measured sensitivity (0.82, not 1.00) is that
gap. So the correct diagnosis is neither "expressiveness" nor "optimisation" but
**parameterisation**: there is no unit-coefficient path from the feature to the output for
the optimiser to find.

---

## 4. The one repair worth trying

Add an explicit **learnable skip from the WMB feature column to the output**, initialised at
1.0: `ŷ_norm = inner(x) + α · base_norm`, `α₀ = 1`. This is not merely a better
initialisation — it creates a path that does not exist today. It makes residual learning
*exactly* representable and the starting point, so the grid win comes by construction, while
`α` is free to decay toward 0 on RBMs, where §3 shows the net already wants ≈0.02. It is the
only variant consistent with everything measured here, and it is ~10 lines. Untested.

**Do not** re-run `partitions` without it: §2.2 shows the extra columns are inert while the
combined column itself cannot be used efficiently.

---

## 5. Honest limits

- **Four cells, two families, all binary, all D=10, iB=10.** Nothing here touches pedigree,
  CSP, non-binary domains, or other i-bounds.
- **The RBM half is underpowered at the pre-registered n.** Two RBM cells at 2 seeds
  (n=40, 42) cannot resolve a +0.03…+0.19 dex effect; the honest statement is "no significant
  difference", not "no difference".
- **The sensitivity probe is one seed on two cells** (`g10w`, `rbm_20`) and measures a mean
  derivative over training rows, not a guarantee about behaviour off the training support.
- **`residual` here is a forward-port**, not doc 31's binary: it reproduces doc 31's signs and
  rough magnitudes, but the cells are not numerically identical to that document's.
- **Wall times are gpu1 and thermally throttled.** For the record and no more: the input arms
  were consistently *cheaper* than baseline (g10w 200s vs 274s) because better output
  conditioning trips early stopping sooner. Cost was never the problem.
- One implementation gotcha, fixed and worth knowing: under `stream_nn_exact` an upstream
  message stays in the bucket as a lazy `FactorNN`, and both `compute_wmb_message` and
  `_get_values` index `.tensor` → `None`. NN factors are now densified for the duration of the
  base computation and restored afterwards.
