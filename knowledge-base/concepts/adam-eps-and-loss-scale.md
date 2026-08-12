---
type: concept
title: Adam's eps Is a Cost Knob, Not an Accuracy Knob
status: budding
tags: [this-project, training, optimization, loss-functions]
created: 2026-08-12
updated: 2026-08-12
---

# Adam's `eps` Is a Cost Knob, Not an Accuracy Knob

When a reparameterisation multiplies a network's output by a constant $s$, the loss scales by
$s^2$ and **Adam is exactly invariant to it except through `eps`**, which is effectively inflated
to $\text{eps}/s^2$. That single algebraic fact explained away the apparent cost of
[[wmb-residual-learning]] — and an adversarial control then showed it explains **none** of the
accuracy. The 2×2 separates cleanly: `eps` is a **cost** knob, the residual parameterisation is an
**accuracy** knob (docs 22, 26).

## The algebra (doc 22)

Derived before any code was written. In the residual wrapper,
$\hat y - y_\text{norm} = s\,(g_\phi - r_\text{norm})$ holds identically on every sample, so
$L(\phi) = s^2 \tilde L(\phi)$ and **every gradient carries $s^2$**, not $s$. Under Adam with
$m_t = s^2 M_t$ and $v_t = s^4 V_t$:

$$\Delta\phi = -\text{lr}\cdot\frac{s^2 \hat M}{s^2\sqrt{\hat V} + \text{eps}}
             = -\text{lr}\cdot\frac{\hat M}{\sqrt{\hat V} + \text{eps}/s^2}$$

($\beta_1, \beta_2$, bias correction and step counter unchanged; `weight_decay=0`,
`grad_clip_norm=None`, `use_amp=False` under `neurobe_mode` — all code-verified.) The exact
compensation is therefore $\text{eps} \to s^2\,\text{eps}$. On the test cell the median $s$ = 0.303
($s^2$ = 0.092, an 11× inflation), worst cluster $s$ = 0.119 (70×), and doc 20's smallest cluster
$s$ = 0.0094 (~$10^4$×).

⚠️ **This corrects doc 20**, which specified the compensation as "set `lr` to `lr/net_scale`
(equivalently, scale Adam's `eps` by `net_scale`)". Doc 22: *wrong twice over — the two are not
equivalent under Adam, and the factor is $s^2$, not $s$.* The `lr/s` variant is a **3.3× learning-rate
increase**, a different intervention entirely: it is cheap (0.41× baseline) but **trades accuracy**
(22/55 against the uncompensated arm, Wilcoxon p = 0.087). Doc 22 also notes doc 20's earlier
across-cluster negative result stands untouched, because swapping $1/s$ for $1/s^2$ is a monotone
map and leaves Spearman literally unchanged.

## What compensating `eps` bought (doc 22)

Measured on `grid10x10.f10.wrap`, `e_max`=10, sB=10, `ecl`=1025, 11 clusters × 5 seeds = 55 paired
points, sequential on gpu1:

| arm | epochs/cluster | cost | accuracy vs the uncompensated arm |
|---|---|---|---|
| baseline | 278.9 | 1.00× | — |
| residual, default `eps` | **977.7** | **3.86×** | — |
| residual, `eps` → $s^2\,$`eps` | **287.1** | **1.13×** | **dead tie**: 29/55, sign p = 0.79, Wilcoxon p = 0.86 |

At the end of training, **92.3%** of the uncompensated arm's coordinates were `eps`-dominated;
compensation drops that to 27%. So the arm's enormous epoch appetite was an optimiser artefact, and
removing it brought the cost from a 3.86× breach of the 2× tolerance to a comfortable 1.13× **with
no accuracy traded**.

## The adversarial control (doc 26)

Doc 22 named its own most dangerous alternative explanation: if `eps` is badly tuned for this loss
scale, maybe *the baseline* is `eps`-starved too and the residual arm's win is really an `eps` win.
Doc 26 ran exactly that control — tune `eps` on the **baseline** itself, no residual, no WMB,
nothing else changed — with both design choices biased *toward* the alternative hypothesis (the
more favourable reused comparator, and best-of-three `eps` values carried forward).

**The control fired**, which is what makes it load-bearing rather than a formality:

- The baseline's default `eps`=1e-8 really is badly tuned: **47.6% of coordinates end training
  `eps`-dominated** and **97.3% sit within a single decade of `eps`**. Doc 22's own smoke-test
  estimate of 12–33% was an underestimate.
- The bite was **predicted in advance** from doc 22's $s^2$ algebra (1.8e-8 predicted vs 1.09e-8
  measured, within 1.7×) — an independent confirmation of the derivation.
- Tuning drives the bite to 0.011 (1e-10) and 0.001 (1e-12, 1e-16).

**And accuracy barely moves.** 1e-10 vs baseline: 34/55, sign p = 0.105, Wilcoxon p = 0.140,
**+0.177 dex**. All three `eps` values agree and none is significant, so nothing hinges on the pick.
Against the residual arm's +0.675 dex, that is a **~26% recovery — and the 26% is itself not
distinguishable from zero**.

**The decisive contrast**: with the `eps` regime held fixed on *both* sides, the residual
parameterisation still wins — 39/55, sign p = 0.0027, Wilcoxon p = 0.0008, +0.368 dex.

### The 2×2

| | default `eps` | tuned `eps` | `eps` main effect |
|---|---|---|---|
| baseline parameterisation | — | 0.75× cost, +0.177 dex (n.s.) | small, not significant |
| residual parameterisation | 3.86× cost | 1.13× cost, +0.044 dex (n.s.) | small, not significant |
| **parameterisation main effect** | **+0.675 dex** | **+0.368 dex** | |

The parameterisation effect is large and significant in **both** `eps` regimes; the `eps` effect is
small and significant in **neither**. Wall times were clean (nothing co-resident) and agree with the
epoch-based costs to **0.1%** — the strongest wall-time/epoch consistency check in the series.

### The per-cluster pattern runs *opposite* to the `eps` story

On the three most `eps`-starved clusters, tuning `eps` on the baseline buys **nothing** (one is
worse) where the residual arm buys 5–10×. Tuning's one big win is the cluster with the **largest**
`net_scale` and the *smallest* bite. Spearman(`net_scale`, residual/base) = **+0.582** — the sign a
scale-driven mechanism needs; Spearman(`net_scale`, tuned/base) = **−0.491** — the wrong sign. If
`eps` were the channel, both would share a sign. Both n=11 and post hoc: *the sign disagreement is
the point, not either p-value*.

## The side finding, actionable on its own

**`eps` = 1e-10 makes the NeuroBE baseline itself 0.75× cheaper at no accuracy cost.** Torch's
default `eps` is simply wrong for this loss scale. Doc 31 promotes a `baseline_epstuned` arm into
the powered sweep for exactly this reason — it is "the first objection a mechanism-aware reviewer
raises".

## Limits, as stated by doc 26 itself

One cell, one problem, one $(sB, ecl)$; n = 55 with seeds as correlated replicates; the sign test
has 80% power only against $p_\text{win} \approx 0.67$ while tuned-`eps` sits at 0.62, so **a real
but modest `eps` benefit of up to roughly +0.2 dex could be hiding here**; the comparison is not
perfectly matched across parameterisations (per-cluster $s^2\text{eps}$ = 9.16e-10 on one side, a
fixed 1e-10 on the other); and all these runs **predate the determinism fix** — see
[[bit-exact-reproducibility]].

## Related

- [[wmb-residual-learning]] · [[loss-functions]] · [[convergence-diagnostic-gap]]
- [[neural-network-factors]] · [[neural-bucket-elimination]] · [[bit-exact-reproducibility]]
- [[2026-W33]]
