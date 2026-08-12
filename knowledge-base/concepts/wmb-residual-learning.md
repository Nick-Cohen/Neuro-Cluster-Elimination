---
tags: [concept, nce, wmb, neural-networks, training]
status: active
created: 2026-08-11
---

# WMB Residual Learning

Instead of training a neural network factor to predict a cluster's message `m*` directly,
train it to predict the **residual** against a cheap analytic estimate of that message:

```
r = log10(m*) - log10(m_wmb)
```

where `m_wmb` is the forward [[weighted-mini-bucket]] estimate of the same cluster, built with the
**same `iB`/`ecl` that decided NN-vs-exact in the first place**. The cluster then emits
`[*wmb_base_factors, residual_NN]` as a **list**, and `eliminate_variables` routes each message
independently — log-space product is addition, so downstream elimination reconstructs
`m* ≈ base × residual` with no changes to `FactorNN`, `_get_slices`, or `nn_to_FastFactor`.

The premise: `r` has far less dynamic range than `m*` (measured `sd_y/r_std` median ≈ 5, up to 92 on
individual clusters), so the network spends its capacity on the part WMB gets wrong instead of
re-learning the part it already gets right. `r ≤ 0` by the Hölder bound ([[@liu2011holder]]), which
held to float32 round-off under *approximate* upstream context too.

**That premise turns out not to be the mechanism** — see "Why it works" below. Read it before
citing the variance-reduction story.

⚠️ The `sd_y/r_std` numbers themselves are reported inconsistently across documents for the *same*
30 cluster-runs: max 62.0 / median 4.73 (doc 14, whose own per-cluster table nonetheless lists a
cluster at 82.4), ≈90 for that cluster (doc 16), and range [2.73, 92.18] / median 5.04 (doc 18).
Doc 02's original probe on a different problem set measured median ≈ 11, range 2.0–129.

## The two things that must not break

1. **The loss must be evaluated on the reconstruction, not on the residual.**
   `neurobe_weighted_mse` computes importance weights `w = targets · ln_range / sum_ln`. If the
   target were the residual, the weights would key off the *correction* rather than the message
   mass, and `minmax_01` would push the entries with the largest corrections to weight ≈ 0.
   The fix is structural, not a patch: a `WMBResidualNet` wrapper carries the WMB base as an extra
   trailing input column and adds it to the inner net's output, so *every* loss, validation and
   early-stopping site in `Trainer` sees `y_hat = NN_residual + wmb` against the original `y`.
   No duplicate `y`-stats are needed.

2. **The normaliser must be fitted to the residual, not to the message.** This is the whole result.
   With `minmax_01` still fitted to `y`, the residual net has to output
   `(r·ln10 − ln_min_y)/ln_range_y` — wrong in *location* (measured: a constant −3.23 normalised
   units it must learn before it can learn anything) and wrong in *scale* (measured `net_scale`
   median 0.139, i.e. the net was emitting in units 3.5–127x too large). Naively implemented,
   residual learning is **worse than baseline**. Two fixes work:
   - **bias calibration** (9 lines): shift the final layer's bias after the first data load so the
     epoch-0 prediction is unbiased. Fixes location only.
   - **residual-fitted normalisation** (`wmb_residual_norm='residual'`): fit a second
     `DataPreprocessor` to `r`. Fixes both.

## The algebra for residual-fitted normalisation

Write both normalisers as affine maps in natural-log space —
`(off, scale) = (ln_min, ln_range)` for `minmax_01`, `(normalizing_constant, 1)` for
`logspace_mean`:

```
y_norm = (y·ln10 − off_y)/scale_y
r_norm = (r·ln10 − off_r)/scale_r
```

Since `y = r + base`,

```
y_norm = r_norm·(scale_r/scale_y) + (off_r + base·ln10 − off_y)/scale_y
```

so `forward(x) = net_scale · inner(x[:,:-1]) + x[:,-1:]` with `net_scale = scale_r/scale_y` and the
trailing column carrying the affine offset. **The target stays `y_norm`, so the weighting is
bit-identical to baseline's.** The message-fitted case is the special case
`scale_r = scale_y, off_r = off_y` ⇒ `net_scale = 1`, so one code path serves both. The emitted
factor is `FactorNN(inner, dp_residual)`, whose `undo_normalization` recovers exactly `r`.

## What is measured, on one problem

grid10x10.f10, `iB=10 ecl=1025`, no merging, 6 NN clusters × 5 seeds = 30 paired points, primary
metric per-cluster signed local error:

| arm | wins vs baseline | sign test | median \|local err\| |
|---|---|---|---|
| baseline | — | — | 0.084 |
| residual, normaliser fitted to y | 11/30 | p = 0.20 | 0.197 |
| + bias calibration | 19/30 | p = 0.20 | 0.038 |
| + residual-fitted normalisation | **22/30** | **p = 0.016** | **0.026** |

An **epoch-matched baseline control** (early stopping off, 500 epochs — the budget the fitted-norm
arm actually consumes, since its early stopping never fires) is 15/30, p = 1.00 vs baseline, and
loses 25/30 (p = 0.0003) to the fitted-norm arm. So the gain is the parameterisation, not the epoch
budget. The two conditioned arms are **not** separable from each other (17/30, p = 0.58).

Costs: building the WMB base is **0.018 s/run, ~0.01% of wall time** — free. The residual machinery
is **1.08x** wall time at a matched epoch budget.

## Why it works: output conditioning, not variance reduction (doc 20)

The decisive argument is that **arms 2, 3 and 4 share a literally identical residual target**, so
`sd_y/r_std` assigns them the same value and predicts they behave alike. They do not. Changing only
*how the network is asked to emit that target* moves the result from 1.7× worse than baseline to
3.7× better, monotonically with how much of the output normaliser is fixed. A premise about the
target cannot explain a **sign flip on a fixed target**.

Paired by `(cluster, seed)`; gain = `log10(|err|_base / |err|_arm)`; 18 structural clusters, 537
cluster-runs on disk:

| contrast | wins | pooled median gain |
|---|---|---|
| arm 2 (residual, **message** normaliser) vs baseline | **11/30** | **−0.223 dex** |
| arm 3 (+ bias init — location fixed) vs baseline | 79/120 | +0.314 dex |
| arm 4 (**residual** normaliser — location + scale) vs baseline | **109/131** | **+0.569 dex** |
| arm 4 vs epoch-matched control | 64/90 | +0.509 dex |

Doc 20's synthesis, which is the sentence to quote: **"target smallness sets the size of the prize;
output conditioning decides whether you collect it."** Doc 02's premise is
**necessary-but-not-sufficient, not false.**

### Four negative results that keep the story honest

1. **The two candidate mechanisms are one number.** `1/net_scale` and `sd_y/r_std` are collinear at
   Pearson **0.982** (Spearman 0.940) — algebraically, since `net_scale = r_range/y_range` and
   `y_range` is nearly constant. This data **cannot** separate them.
2. **Nothing predicts the per-cluster benefit.** After a max-|ρ| permutation test over 13 candidates
   (20,000 permutations), the best is residual dynamic range at family-wise **p = 0.065**,
   leave-one-cluster-out CV R² ≈ 0.30. The conditioning-specific signature is absent: the normalised
   offset predicts nothing (ρ = −0.06), and within the wrap cell it points the *wrong* way.
3. **Merging does not discriminate them.** Doc 19 hoped it would; merging moved `sd_y/r_std`
   5.15 → 2.30 **and** `1/net_scale` 7.11 → 3.30 — the same direction. Evidence against *both*
   quantities being the driver, not evidence for one over the other. Note also that under merging
   `sd_y/r_std` got **worse** while the benefit got **bigger** (3.2× → 3.7× median).
4. **"Condition without the residual" is a no-op.** `neurobe_mode` sets `minmax_01` fitted to that
   cluster's target; for baseline the target *is* the message, so `y_norm` fills [0,1] exactly and
   `net_scale ≡ 1`, offset ≡ 0, **by construction**. The residual is load-bearing: it *creates* the
   units mismatch that arm 4 then fixes, and it supplies the smaller target.

**Robust, but about absolute error rather than benefit**: cluster size sets the absolute error
(separator size → baseline error ρ = +0.72, fw p = 0.012; `e` → arm-4 error ρ = +0.71, fw p = 0.015)
— and it cancels in the ratio.

⚠️ **Recorded disagreement.** Doc 19 declared doc 02's premise **falsified**. Doc 20 corrects this to
**over-stated**: `sd_y/r_std` is in fact the best-correlated cross-cluster predictor of benefit that
exists on this data (ρ ≈ +0.55), it just fails significance after multiple-comparison correction.

## Epoch policy: the arm was not expensive, it was unfinished (doc 18)

Doc 16's arm-4 numbers were a **lower bound**. Reading the newly-instrumented validation trajectory
(see [[convergence-diagnostic-gap]]) showed the arm never early-stops because it is *genuinely still
improving*: at epoch 500 its validation loss sits a median **2.60×** above its eventual best, and
500→2000 cuts it a further 61%. Ground truth agrees — 22/30, median |local err| 0.0259 → **0.0138**
— with no divergence between validation loss and local error. **19/30 clusters are still improving
at the 2000 cap**, so 2000 is not convergence either.

On `grid10x10.f10.wrap` under merging (11 NN clusters, 55 paired points — the pilot problem
collapses to a *single* NN cluster under merging, whose best attainable two-sided sign-test p is
0.0625): arm 4 beats baseline **48/55, p < 0.0001**, median 0.231 → 0.062, winning on 11/11 clusters.

## Cost, resolved: it was an Adam artefact (docs 22, 26)

Arm 4's 3.86–5.56× cost breached the 2× tolerance and was **~90% extra epochs**, not per-epoch
overhead. Compensating Adam's `eps` for the wrapper's output multiplier brings it to **1.13× with
zero accuracy traded**, and an adversarial control establishes that `eps` explains the *cost* and
none of the *accuracy*. This is important enough to have its own note:
[[adam-eps-and-loss-scale]].

## Untested / not established

Zero-`−inf` problems, float64, non-binary domains, and any second problem family beyond the two
grid cells — `grid20x20.f10` was measured at ≈49 min/run (~8 GPU-hours for a 2-arm × 5-seed
comparison). Doc 19's sweep cost estimate of ≈34 GPU-h was later shown ≈2× high, almost entirely on
`rbm_20`, where the per-cluster cost constant does not transfer across families (1.9× off the grids').

Two further caveats worth carrying:

- ⚠️ **The `log Z` stability benefit did not reproduce.** Docs 14/16 reported across-seed spread
  0.756 → 0.335 → 0.180 (a "4.2×" reduction called "the number worth leaning on"). On the wrap cell
  the spreads are baseline 0.677, bias-init 1.510, arm 4 0.737 — **arm 4 does not narrow it**. Do
  not quote the spread reduction as a general property.
- ⚠️ **Arm 4 vs bias-init depends on the budget.** At a matched 500-epoch cap they are a coin flip
  (17/30, p = 0.58, doc 16). At a matched 2000-epoch cap with each arm on its own patience rule,
  arm 4 wins 25/30, p = 0.0003, median 0.0138 vs 0.0551 (doc 18) — but that is a matched **cap**,
  not matched compute: arm 4 spends 3.6× bias-init's epochs. Both statements are true.
- ⚠️ **"Extra epochs buy baseline nothing" is problem-specific.** True on `grid10x10.f10`
  (15/30, p = 1.000, doc 16); **false on wrap** (38/55, p = 0.0065, doc 18). Doc 18 explicitly
  narrows doc 16's generalisation.

## Where it lives

Branch `feat/wmb-residual` (not merged). Config keys: `wmb_residual`, `wmb_residual_bias_init`,
`wmb_residual_norm ∈ {'message','residual'}`. Touches `bucket.py` (`compute_message_nn`),
`data_loader.py` (`load`), `net.py` (`WMBResidualNet`), `train.py`, `config_schema.py`.
See also the [[weighted-mini-bucket]] gotcha section — building this surfaced two pre-existing
crashes in `compute_wmb_message` and `FactorNN` that affect all WMB work.

Also on the branch: `config['adam_eps']` (`a362290`) and the `Trainer.val_losses` instrumentation
(`2ae8cc1`) — both generally useful and worth landing independently of whether residual learning
ships.

Experiment write-ups (gitignored, on disk), in
`notebooks/_August-2026/claude_experiments/`: `02-wmb-residual-learning.md` (design),
`14-wmb-residual-pilot.md` (build + arms 1–3), `16-wmb-residual-arm4.md` (arm 4 + control),
`18-wmb-residual-next.md` (epoch policy + merging), `19-residual-sweep-readiness.md` (cost +
coverage), `20-residual-mechanism.md` (mechanism), `22-arm5-lr-compensation.md` (arm 5),
`26-eps-control.md` (the control), `31-residual-sweep.md` (sweep design).

## Related

- [[weighted-mini-bucket]] — where the base comes from, and its crash gotchas
- [[adam-eps-and-loss-scale]] — why the arm looked expensive, and the control that saved the result
- [[convergence-diagnostic-gap]] — why the epoch policy was unreadable until 2026-08-11
- [[neural-network-factors]], [[neural-bucket-elimination]], [[loss-functions]]
- [[error-accumulation]] — per-cluster local error is the primary metric here
- [[bit-exact-reproducibility]] — most of these runs predate the determinism fix
- [[2026-W33]]
