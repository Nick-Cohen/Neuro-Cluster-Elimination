# Bucket Merging Improves Neural Bucket Elimination

**Working draft — v0.1 (2026-06-12)**

> Status: skeleton draft assembled from project planning docs, the lab notebook
> (2026-05-31 → 2026-06-04), and the `reduce_nn` / `subsumption_merge` experiment
> results. Sections marked **[TODO]** need author input or additional runs.
> See `NOTES.md` for the list of open items and the provenance of every number.

---

## Abstract

Neural Bucket Elimination (NeuroBE) performs approximate inference on discrete
graphical models by running variable (bucket) elimination and replacing the
messages of high-width buckets — those whose message tables are too large to
represent exactly — with trained neural-network approximations. Because messages
propagate, each approximated bucket injects error that the downstream
elimination compounds, so the accuracy of the final partition-function estimate
degrades with the *number* of neural approximations, not just their individual
quality. We introduce **bucket merging**: before training, adjacent
NN-eligible buckets are greedily combined into larger clusters, subject to a
single tunable cap on the number of variables eliminated per cluster (the
*merge bound*). Merging reduces the count of independently-trained neural
networks — and hence the number of error-injection sites — at the cost of wider
exact sub-computations inside each merged cluster. Across four representative
problem families (grids, pedigrees, restricted Boltzmann machines) at induced-width
budgets iB ∈ {10, 20}, merging reduces partition-function error by roughly
6–7× (up to ~100× on the hardest grid) while *also* reducing wall-clock time,
with a consistent sweet spot at a merge bound of 8–16 eliminated variables.
Beyond that bound the exact cost inside clusters grows as 2^bound and accuracy
flattens or regresses. We also report the systems changes (streaming sample
generation, chunked dense factor materialization, and memory-bounded exact
elimination) required to make wide merged clusters tractable.

---

## 1. Introduction

Probabilistic inference on discrete graphical models — computing the partition
function Z, marginals, or a most-probable explanation — is #P-hard in general.
Bucket (variable) elimination solves it exactly but with cost exponential in the
*induced width* of the elimination order, which is prohibitive on dense or large
models. **Weighted Mini-Bucket (WMB)** elimination caps this cost by splitting
wide buckets into mini-buckets of bounded width (the *i-bound*, `iB`), trading
exactness for a tractable bound.

**Neural Bucket Elimination (NeuroBE)** takes a different tack on the same
bottleneck. Rather than partitioning a wide bucket into independent mini-buckets,
it keeps the bucket whole and *learns* its outgoing message: it draws samples of
the bucket's output configuration, evaluates the exact (log-)message at those
samples, and fits a small neural network to that function. Buckets whose message
table fits under an exact-computation limit (`ecl`) are computed exactly; the
rest are replaced by their learned surrogate. This yields a single estimate of Z
on problems where exact elimination is infeasible.

The weakness of this scheme is **error accumulation**. A learned message is an
approximation; when it is consumed by a downstream bucket, the error propagates
and is folded into every subsequent elimination. A problem that triggers 100+
neural approximations therefore stacks 100+ noisy messages, and the final Z can
be off by several nats even when each individual network fits its samples well.
Empirically (Section 5) the dominant predictor of final error is the **number of
neural-network buckets**, not the per-network training loss.

This paper's contribution is the observation — and the algorithm and systems work
to exploit it — that the number of neural approximations is not fixed by the
problem. Many NN-eligible buckets are adjacent in the bucket tree and can be
**merged** into a single larger cluster that emits one learned (or, once it
collapses under `ecl`, one *exact*) message in place of several. Merging is
governed by one parameter, the **merge bound** (`max_merge_bound`): the maximum
number of variables a merged cluster may eliminate. We show:

1. **Merging is almost free of approximation cost** when it is *subsumption-based*
   (Section 4.1): if a child bucket's scope contains its parent's, absorbing it
   adds no new variables to the cluster.
2. **A small merge bound buys a large accuracy gain.** Reducing the NN count from
   ~30–130 down to ~20–30 cuts Z-error by 6–7× across grids, pedigrees, and RBMs,
   and shortens runtime because fewer networks are trained.
3. **There is a clear sweet spot.** Because the exact work inside a merged cluster
   scales as 2^(eliminated variables), pushing the merge bound past ~16 reintroduces
   the exponential blow-up that NeuroBE exists to avoid — accuracy plateaus or
   worsens and time climbs steeply.

The accuracy improvement required three orthogonal systems fixes to make wide
merged clusters runnable at all (Section 6); without them, merged buckets at iB=20
either hang in sample generation or run out of memory during dense materialization
and exact elimination.

---

## 2. Background

### 2.1 Graphical models and bucket elimination

A discrete graphical model over variables X = {X₁,…,Xₙ} is a product of
non-negative factors f₁,…,f_m, each over a subset (scope) of X. The partition
function is Z = Σ_x ∏_j f_j(x). All factor arithmetic in this work is performed in
log-space: factor product is log-space addition and marginalization is
log-sum-exp.

Bucket elimination fixes an elimination order, assigns each factor to the bucket of
its earliest-eliminated variable, and processes buckets in order. Eliminating a
bucket multiplies its factors, sums out the bucket's variable(s), and sends the
resulting **message** to the bucket of the next-earliest variable in the message's
scope. The cost of a bucket is exponential in its *width* (the size of the message
scope plus the eliminated variables); the max width over the run is the induced
width.

### 2.2 Weighted Mini-Bucket (WMB)

WMB bounds per-bucket cost by partitioning a wide bucket's factors into
**mini-buckets**, each of scope ≤ `iB`, and eliminating each independently with a
weighted (Hölder) bound. This guarantees a bound on Z at cost exponential only in
`iB`, but the bound's tightness degrades as `iB` shrinks below the true width.

### 2.3 NeuroBE: neural message approximation

NeuroBE replaces the *representation* of a wide bucket's message rather than
partitioning the bucket. For a bucket whose exact message table exceeds `ecl`:

1. **Sample** configurations of the message's output scope (NCE supports several
   sampling schemes; the NeuroBE-faithful mode uses importance sampling).
2. **Label** each sample with its exact log-message value (computed by summing out
   the bucket's eliminated variables for that fixed output configuration).
3. **Normalize** the labels to [0,1] by min–max scaling
   `(v − ln_min)/(ln_max − ln_min)`.
4. **Train** a small MLP (ReLU, batch size 256, lr 1e-3, weighted MSE with
   importance weights, patience-2 early stopping on validation loss) to predict the
   normalized label from the configuration.
5. **Denormalize** at query time: `ln_min + nn_out·(ln_max − ln_min)`.

The trained network *is* the bucket's outgoing message: downstream buckets query it
like any other factor. NCE reproduces this pipeline faithfully under a single
`neurobe_mode: true` config flag (project milestone M003), matching NeuroBE's NN
dispatch counts on all 15 working binary-domain benchmark problems.

This pipeline is the NeuroBE method of Agarwal et al. (2022), which builds on Deep
Bucket Elimination (Razeghi et al. 2021); NCE faithfully reproduces it (and matches its
per-bucket NN dispatch counts) via the `neurobe_mode` flag, against the authors' C++
reference implementation (`Clean-NeuroBE/`).

### 2.4 The error-accumulation problem

Each learned message carries approximation error ε_k. Messages compose: bucket k's
output feeds bucket k′ > k, whose own (exact or learned) message now sits on top of
ε_k. To first order the error in log Z accumulates across the chain of approximated
buckets, so the final error grows with the *number* of NN buckets along active
message paths. This motivates reducing that count directly — the goal of merging.

---

## 3. Problem statement

Given a graphical model, an elimination order, an i-bound `iB`, and an
exact-computation limit `ecl`, NeuroBE produces a set B_NN of buckets that will be
neural-approximated (those whose exact message table exceeds `ecl`). We seek a
*clustering* of the bucket tree that:

- **reduces |B_NN|** (fewer error-injection sites), while
- **keeping every cluster's exact internal cost bounded**, so that a merged cluster
  is either (a) collapsed to an exact message because its merged table now fits
  under `ecl`, or (b) still neural-approximated but representing the work of several
  former buckets.

The single control is the **merge bound** `max_merge_bound = D`: no cluster may
eliminate more than D variables. D = 1 recovers plain NeuroBE (no merging);
D → ∞ permits arbitrary merges and reintroduces exponential cost.

---

## 4. Method: bucket merging

NCE implements three merging strategies in
[`nce/inference/graphical_model.py`](../nce/inference/graphical_model.py). All run
*after* the bucket tree and message scopes are computed and *before* training, and
all respect the merge bound D.

### 4.1 Subsumption merging (`merge_join_tree`)

A child bucket whose elimination-time scope **contains** its parent's scope can be
absorbed into the parent without adding any new variable to the cluster — this is
exactly the running-intersection / join-tree condition. Subsumption merges are
therefore *free*: they reduce |B_NN| without enlarging any exact computation. With
no cap (D = ∞), repeated subsumption can collapse a whole branch into one root
super-bucket — which is why a finite merge bound is essential: an unbounded collapse
reintroduces the 2^width blow-up NeuroBE is designed to avoid.

### 4.2 Degree-bounded greedy merging (`merge_by_degree`)

When subsumption alone does not reach the target, NCE greedily merges adjacent
NN-eligible buckets, at each step choosing the pair that **adds the fewest new scope
variables** (a subsumption pair adds zero and is always taken first). Merging stops
for a cluster once it would exceed D eliminated variables. This gives precise
control over the accuracy/cost trade-off via D alone.

### 4.3 NN-count reduction with backtracking (`reduce_nn_merge`)

The accuracy-oriented strategy: for each NN bucket, greedily absorb its ancestor
chain up the bucket tree until the cluster either (a) collapses under `ecl` and
becomes exact, or (b) merges into another NN cluster — directly minimizing |B_NN|.
An optional **backtracking** mode first finds the minimum NN count achievable within
the cap D, then re-runs at the *smallest* cap that still achieves that minimum,
removing the "spare capacity" that greedy over-growth leaves when D is generous
(smallest clusters that still hit the best NN count).

### 4.4 Mechanics

Absorbing a child into a parent appends the child's eliminated variables, prepends
its factors, unions the scopes, and deletes the child from the bucket dictionary.
The elimination loop still walks the full order but skips variables that have been
merged away. A merged cluster's outgoing message is computed exactly when its table
fits under `ecl` (now possible because several former NN buckets fused into one
exact computation) and is otherwise trained as a single network.

---

## 5. Experiments

### 5.1 Setup

- **Problems.** Four families: dense grids (`grid10x10`, `grid20x20`, up to
  `grid40x40`), genetic linkage **pedigrees** (`pedigree7/13/19/34/41/51`), and
  **restricted Boltzmann machines** (`rbm_20/21/22`, including ferromagnetic
  variants). These span the range from sparse-but-deep (pedigrees) to dense
  (grids, RBMs).
- **Budgets.** Induced-width budgets iB ∈ {10, 20}; `ecl = 2^iB + 1`.
- **Sweep.** Merge bound D swept over {1, 2, 4, 8, 12, 16, 20, 24, …}. D = 1 is
  the no-merge NeuroBE baseline.
- **Metric.** Absolute error in log₁₀ Z against the reference partition function;
  number of trained networks (`num_trained`); wall-clock time.
- **Hardware.** 4× NVIDIA TITAN RTX, CUDA; NeuroBE-faithful training settings.

> **[TODO]** State the reference-Z source per problem (exact where available,
> high-`iB` WMB / converged solver otherwise) and the sampling scheme + sample
> count used for the headline runs. Confirm seeds / number of trials per cell.

### 5.2 Headline result — merging helps, with a sweet spot

The cleanest, hand-verified curves from the final experiment (lab notebook
2026-06-03/04). Error is absolute error in log₁₀ Z; "NN" is the number of trained
networks.

| Problem | iB | No-merge (D=1) | Sweet spot | NN reduction | Error improvement |
|---|---|---|---|---|---|
| `grid10x10.f10.wrap` | 10 | err 3.81 | err **0.11** @ D≈8–16, 29 NN | →29 | **6.7×**, faster |
| `grid20x20.f10` | 10 | err 15.84 | err **0.16** @ D≈8–16, 7 NN | →7 | **~100×**, faster |
| `pedigree13` | 20 | err 2.49, 126 NN | err **0.37** @ D=16, 26 NN | 126→26 | **6.7×** |
| `rbm_22` | 10 | err 0.648, 33 NN | err **0.094** @ D=6, 23 NN | 33→23 | **7×**, time halved |

Across all four, no-merge NeuroBE is the *worst* accuracy point on the curve.
The effect is strongest on grids and pedigrees and weaker (but still present) on the
dense RBM, where merging reduces the NN count only modestly yet still improves
accuracy ~7×.

### 5.3 The cost cliff

The static topology sweep (`merge_degree_sweep.csv`) makes the trade-off explicit.
For `pedigree13` at iB=20, the NN count saturates early while the per-cluster exact
cost (sample complexity, ≈ 2^(max eliminated vars)) explodes:

| Merge bound D | NN buckets | max elim. vars / cluster | log₂(total sample complexity) |
|---|---|---|---|
| 1 | 126 | 1 | 11.0 |
| 8 | 29 | 8 | 12.1 |
| 12 | 26 | 12 | 14.6 |
| 16 | 26 | 16 | 17.4 |
| 24 | 26 | 24 | 25.2 |
| ∞ | 26 | 30 | **31.2** |

Past D≈12–16 the NN count is already at its floor (26), so further merging buys *no*
accuracy headroom but multiplies exact cost by ~16× per +4 bound. This is the source
of the sweet spot: **merge just enough to hit the NN-count floor, no further.**

### 5.4 Full sweep (supporting data)

A broader automated sweep over 77 (problem, iB, D) cells
(`reduce_nn_experiment/results/`) confirms the trend and its limits — including the
regime past the sweet spot where exact super-buckets dominate runtime (e.g.
`grid20x20.f10` iB=10 at D=24 reaches 4 NN but takes ~8 h vs. minutes at D≤16, and
`rbm_22` iB=10 at D=24 collapses to 1 NN but takes ~13 h). The aggregated table is
in `NOTES.md`; per-problem accuracy-vs-merge plots are
`reduce_nn_experiment/final_reduce_nn_*.png`.

> **[TODO]** Decide which problems make the main results table vs. the appendix,
> and regenerate publication-quality figures (current PNGs are working plots).

### 5.5 Figures

- `notebooks/May-2026/claude_experiments/merge_degree_experiment/accuracy_and_time_vs_merge.png`
  — accuracy and time vs. merge bound (the core trade-off figure).
- `notebooks/May-2026/claude_experiments/subsumption_merge_analysis/merge_degree_sweep.png`
  — NN-count saturation vs. merge bound across problems.
- `reduce_nn_experiment/final_reduce_nn_{grid10x10f10wrap,grid20x20f10,pedigree13,rbm_22}.png`
  — per-problem accuracy curves.

---

## 6. Systems work to make wide clusters tractable

Merged clusters at iB=20 stress code paths that single buckets never reached. Three
fixes (all verified against unmerged baselines) were required:

1. **Streaming dense materialization** (`factor_nn.py::nn_to_FastFactor`). Densifying
   a learned factor over a ~25-variable scope tried to allocate the whole 2^scope
   input grid (10+ GB). Now streamed in fixed-size chunks (65 536 assignments).
2. **Vectorized sample slicing** (`sample_generator.py::_get_slices`). An inner
   Python loop of 2^(#elim) iterations per assignment chunk made sample generation
   quadratic and effectively hung for clusters eliminating ≥12 variables; replaced
   with NumPy broadcasting.
3. **Memory-bounded exact elimination** (`bucket.py::_compute_message_exact_chunked`).
   A merged *exact* super-bucket's joint can reach 2^30 entries (~4–9 GB). When the
   joint would exceed 2^28 entries, elimination now streams over blocks of the output
   message, never materializing the full joint; verified identical to the dense path
   (max error ~1e-7) including non-binary domains and multiple eliminated variables.

A further throughput optimization makes the sample-generation block size
configurable (`NCE_SAMPLE_BLOCK_LOG2`, `NCE_SAMPLE_ACHUNK`), cutting GPU kernel-launch
overhead (~60% idle at high merge bound) by up to ~32× fewer launches.

---

## 7. Discussion

**Why merging helps.** Merging trades *many* small approximations for *few* larger
exact-or-approximate computations. Each merge either eliminates an error-injection
site outright (the cluster collapses under `ecl` to an exact message) or fuses
several error sources into one network that sees the joint structure directly. The
net effect is fewer, better-conditioned messages on the active paths to the root.

**Why the sweet spot exists.** The merge bound interpolates between two failure
modes: too small (D=1) leaves the full NeuroBE error stack; too large reintroduces
the 2^width exact cost that motivated approximation in the first place. The NN-count
floor is reached well before that cost explodes, so a moderate bound (8–16) captures
essentially all the accuracy gain at modest cost.

**Orthogonality.** Merging is independent of NeuroBE's training choices
(normalization, loss, early stopping). It is a *structural* preprocessing step on the
bucket tree and would compose with any message-approximation backend.

> **[TODO]** Discuss relationship to mini-bucket merging / join-graph clustering in
> the WMB literature (merging is the inverse operation to mini-bucket *splitting*),
> and to anytime/iterative tightening. Note limitations: dense RBMs gain least;
> reference-Z availability bounds which problems we can evaluate.

---

## 8. Related work

**Exact inference.** Variable elimination (Zhang & Poole 1996) and its unifying
organization, bucket elimination (Dechter 1999), compute Z exactly at cost exponential in
the induced width; the equivalent clustering view is junction-/join-tree propagation
(Lauritzen & Spiegelhalter 1988) over a tree decomposition (Kask et al. 2005). Merging is a
re-choice of that tree decomposition — coarser clusters that eliminate several variables at
once — and our subsumption merge is exactly the running-intersection condition of a join tree.

**Bounded approximate inference.** When the width is too large, mini-bucket elimination
(Dechter & Rish 2003) partitions a wide bucket into width-≤*i* mini-buckets; weighted
mini-bucket adds Hölder weights for a tighter Z bound (Liu & Ihler 2011), and iterative
join-graph propagation (Mateescu et al. 2010) trades the bound for iteration on a join-graph.
All are controlled by the same i-bound knob. Bucket *merging* is the **inverse** operation to
mini-bucket *splitting*: where WMB splits a bucket to bound cost, we merge adjacent buckets to
reduce the number of learned messages. AND/OR search (Dechter & Mateescu 2007) attacks the
same tasks by trading elimination's memory for search time.

**Neural and learned surrogates for inference.** NeuroBE (Agarwal et al. 2022), building on
Deep Bucket Elimination (Razeghi et al. 2021), replaces wide bucket *messages* with trained
networks — the scheme this paper extends. A broader literature learns inference computations:
amortized variational inference learns an encoder network (Kingma & Welling 2014; Mnih &
Gregor 2014), and graph-neural-network methods learn message passing on the model graph or
correct belief-propagation messages (Yoon et al. 2019; Garcia Satorras & Welling 2021).

**Positioning.** Bucket merging is a *structural* preprocessing step on the bucket tree: it is
orthogonal to the message-approximation backend (it composes with WMB, NN, decision-tree, or
quantized messages alike) and to NeuroBE's training choices. It complements — rather than
competes with — both the bounded-inference family (a different point on the cost/accuracy
curve, via the i-bound) and the learned-surrogate family (fewer, better-conditioned learned
messages on the active paths to the root).

---

## 9. Conclusion

Bucket merging is a small, single-parameter change to Neural Bucket Elimination that
reduces the number of neural message approximations — and therefore the accumulated
inference error — by 6–7× (up to ~100×) while reducing runtime, across grids,
pedigrees, and RBMs at two width budgets. A merge bound of 8–16 eliminated variables
captures the gain before the exponential exact cost returns. Realizing the benefit at
iB=20 required streaming sample generation, chunked dense materialization, and
memory-bounded exact elimination. Merging is orthogonal to the underlying training
pipeline and should transfer to other message-approximation schemes.

---

## Appendix A — Reproduction

- Merge implementation: [`nce/inference/graphical_model.py`](../nce/inference/graphical_model.py)
  (`merge_join_tree`, `merge_by_degree`, `reduce_nn_merge`).
- Config knobs: `max_merge_bound` (merge bound D), `reduce_nn_backtrack`,
  `neurobe_mode`; see [`docs/config_reference.md`](../docs/config_reference.md).
- Experiment drivers: `notebooks/June-2026/claude_experiments/reduce_nn_experiment/`
  (`run_experiment.py`, `master_sweep.sh`, `results/`).
- Static topology sweep:
  `notebooks/May-2026/claude_experiments/subsumption_merge_analysis/`.

---

## References

> BibTeX for all entries is in [`../knowledge-base/references.bib`](../knowledge-base/references.bib);
> per-source notes (with citation-confidence flags) are in `knowledge-base/literature/`.

- Agarwal, S., Kask, K., Ihler, A., & Dechter, R. (2022). NeuroBE: Escalating Neural Network
  Approximations of Bucket Elimination. *UAI 2022*, PMLR 180, 11–21.
- Dechter, R. (1999). Bucket elimination: A unifying framework for reasoning. *Artificial
  Intelligence*, 113(1–2), 41–85.
- Dechter, R., & Mateescu, R. (2007). AND/OR search spaces for graphical models. *Artificial
  Intelligence*, 171(2–3), 73–106.
- Dechter, R., & Rish, I. (2003). Mini-buckets: A general scheme for bounded inference.
  *Journal of the ACM*, 50(2), 107–153.
- Garcia Satorras, V., & Welling, M. (2021). Neural Enhanced Belief Propagation on Factor
  Graphs. *AISTATS 2021*, PMLR 130, 685–693.
- Kask, K., Dechter, R., Larrosa, J., & Dechter, A. (2005). Unifying tree decompositions for
  reasoning in graphical models. *Artificial Intelligence*, 166(1–2), 165–193.
- Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. *ICLR 2014*.
  arXiv:1312.6114.
- Lauritzen, S. L., & Spiegelhalter, D. J. (1988). Local computations with probabilities on
  graphical structures and their application to expert systems. *J. Royal Statistical Society
  B*, 50(2), 157–224.
- Liu, Q., & Ihler, A. (2011). Bounding the partition function using Hölder's inequality.
  *ICML 2011*, 849–856.
- Mateescu, R., Kask, K., Gogate, V., & Dechter, R. (2010). Join-graph propagation algorithms.
  *Journal of Artificial Intelligence Research*, 37, 279–328.
- Mnih, A., & Gregor, K. (2014). Neural Variational Inference and Learning in Belief Networks.
  *ICML 2014*, PMLR 32, 1791–1799.
- Razeghi, Y., Kask, K., Lu, Y., Baldi, P., Agarwal, S., & Dechter, R. (2021). Deep Bucket
  Elimination. *IJCAI 2021*, 4235–4242.
- Yoon, K., Liao, R., Xiong, Y., Zhang, L., Fetaya, E., Urtasun, R., Zemel, R. S., & Pitkow, X.
  (2019). Inference in Probabilistic Graphical Models by Graph Neural Networks. *Asilomar
  2019*, 868–875. arXiv:1803.07710.
- Zhang, N. L., & Poole, D. (1996). Exploiting causal independence in Bayesian network
  inference. *Journal of Artificial Intelligence Research*, 5, 301–328.
