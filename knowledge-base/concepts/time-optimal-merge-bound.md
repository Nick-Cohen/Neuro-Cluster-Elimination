---
type: concept
title: The Time-Optimal Merge Bound Is Code-Version-Dependent
status: budding
tags: [this-project, bucket-merging, cost-model, experiment-validity]
created: 2026-08-12
updated: 2026-08-12
---

# The Time-Optimal Merge Bound Is Code-Version-Dependent

There is a wall-clock optimum in the [[merge-bound]] $e_{\max}$: merging more reduces the number of
NN approximations (and hence [[error-accumulation]]) but raises per-cluster cost. The project has
published values for that optimum. **They are properties of a code version, not of the method** —
after one week of performance and correctness fixes the optimum moved, and moved further than the
cost model predicted (doc 17).

## The measurement

Doc 17, branch `integration/aug11-fixes` (`092732d`), gpu0+gpu3, seed 42, reduce-NN arm, 12 runs,
0 failures. Same problems, configs and runner logic as the June reduce-NN study, so it is
like-for-like. Signal is **wall time**; $\log Z$ was deliberately never used as a comparison signal
because at the time it was not reproducible at a fixed seed (see [[bit-exact-reproducibility]]).

| problem | pre-fix optimum (3 seeds) | post-fix measured |
|---|---|---|
| `grid40x40.f15` (sB20) | **6** | **≥16, unbracketed** — 10,247 → 6,426 → 5,107 s at $e$ = 6/12/16, falling monotonically |
| `grid20x20.f5` (sB10) | **10** | **14** — a genuine interior minimum (696 / 486 / **292** / 354 s at 6/10/14/20) |
| `grid10x10.f10.wrap` (sB10) | 20 (plateau) | **no move** — the null control |

grid20 is the one fully bracketed answer: it moved **+4**. grid10 does not move for a *structural*
reason: at $e \ge 16$ the reduce-NN backtracker collapses to the same configuration
($e_\text{eff}$ = 14, 1 NN bucket, sample-count ratio exactly 1.00), so 16 and 20 are the identical
computation. Where the sample count is unchanged, post- matches pre-fix to within **3%** at every
point — which is exactly how a null control should behave.

## Two mechanisms push the optimum up, from opposite ends

This is the part worth remembering, because only one of the two was predicted.

1. **Generation collapsed** — the [[nn-elim-enumeration-redundancy]] fix. SETUP is 8.7–11.9× faster;
   net of the sample-count rise, per-sample gain is ~17–18×, inside doc 09's measured range. On
   grid40 at $e$=16, generation fell 8,498 → **844 s**, from **89% of the run to 17%**.
2. **Training inflated, and inflated *most at low* $e_{\max}$** — the [[num-samples-freeze]] fix.
   Per-bucket sample count rises by a **median 2.26×** over 142 matched buckets, but **3.14× at
   grid40 $e$=6 against 1.82× at $e$=16**. grid40's training loop went 992 → **10,164 s** at $e$=6.
   Nobody predicted this asymmetry.

Both forces push $e_{\max}^{*}$ **up**. The post-fix curves are **training-dominated everywhere
measured**: $T_\text{gen}/T_\text{train}$ at the optimum is 0.20 (grid40) and 0.033 (grid20),
against doc 15's break-even of 0.067 and 0.109.

## What this does and does not invalidate

Doc 17 and doc 25 §M5 are careful here, and the care is the point:

- The published optima are **not wrong**. They are honest wall-clock measurements of the code that
  existed. What is wrong is treating them as advice: *"No published $e_{\max}^{*}$ should be
  compared against the study's observed optima"*, and anything prescriptive ("optimum ≈ 8–12",
  "sweet spot near 12") needs a version-pinning footnote or removal.
- **This is not a perf comparison and must never be quoted as one.** "The integrated branch is 10×
  slower at grid40 $e$=6" is literally true and is *not* a regression — it is doing 3.1× more
  sampling per bucket, on purpose.

## Which prediction was right

- Doc 15 derived ~12.6 for grid40 (~14 in practice); doc 13 predicted a **14.8–18.0** band.
  Measured: grid40 **≥16**, grid20 **14**. Verdict (doc 17 §5): doc 15's number fits grid20 and
  **under-predicts** grid40; **doc 13's band is the better predictor for grid40**. Doc 15's
  *headline* claim — that the published optima describe a binary that no longer exists — is
  confirmed.
- Doc 13's underlying negative result: the corrected marginal $\gamma$ is **19× smaller** than
  published, because ~60% of the published $\gamma$ was intercept ($T_0$ = 4.73 ms on a 7.8 ms
  median $T_\text{gen}$) and the post-fix per-factor NN:table ratio is **8.1×, not 400–800×**. The
  fitted cost model reaches $R^2$ = 0.9960 on 330 rows and is independently confirmed by a
  per-factor pass (coefficients within 1.2× and 4%) and an out-of-sample fixed-structure scan
  (median measured/predicted slope 0.90). Its conclusion: **sample generation is not what caps
  $e_{\max}$.**

## Recorded disagreements about the published numbers

Doc 27's number audit found the published summary statistics do not reconcile, and this bears on
any $e_{\max}$ claim:

- **Median optimal $e_{\max}$.** Published: median **10, IQR [6,12]** over 27 problems. Doc 15
  recomputed **8 over 26**; doc 27 gets **8, IQR [6,10]** over all 27 cells and **10, IQR [8,10]**
  over the 19 non-grid40 problems. The published sentence is internally impossible — it asserts a
  median of 10 over 27 while also stating all eight grid40 problems optimise at 6, which puts those
  eight at sorted positions 2–9, with only 12 of 27 cells at ≥10. **No population reproduces
  IQR [6,12].** Consequence: the model's predicted band 9.7/11.2/12.9 was called "inside the
  predicted band" against an observed 10; against 8 it sits **above** the observation.
- **$\gamma$'s headline value.** 4.35e-8 is the 2026-07-20 idle-GPU rerun; 4.077e-8 is the
  2026-07-15 co-resident run — and **the CSV that shipped is the co-resident one**. Only one of the
  two write-ups can be right about which run the paper quotes. Every downstream conclusion survives
  either way; the scalar differs by 6.6%. Doc 27 also downgrades "γ was measured on NN-free
  clusters" from MEASURED to **unverified**, because the script that claim rests on is not the
  source of the shipped data.

## Honesty note on the runtime estimate

Doc 17 estimated 76 GPU-min for its 12 runs and used **402** (5.3×); makespan 71 min estimated vs
256 actual. Cause: the estimate held the pre-fix training loop fixed, which the sample-count fix
invalidates. Nine of 21 planned runs were cut mid-flight, which is precisely why grid40's optimum
is a **lower bound** rather than a number. Pinning it needs rnn20/rnn24, ~2–4 h.

## Related

- [[merge-bound]] · [[bucket-merging]] · [[error-accumulation]] · [[super-bucket]]
- [[num-samples-freeze]] · [[nn-elim-enumeration-redundancy]] · [[bit-exact-reproducibility]]
- [[terminology-map]] · [[2026-W33]]
