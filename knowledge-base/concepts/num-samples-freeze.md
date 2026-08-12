---
type: concept
title: The num_samples Freeze
status: budding
tags: [this-project, correctness, sampling, experiment-validity]
created: 2026-08-12
updated: 2026-08-12
---

# The `num_samples` Freeze

NCE's `num_samples` config value may be the string `"nbe,<epsilon>[,<n_min>]"`, meaning
"compute the NeuroBE per-cluster sample count from **this** cluster's width and domain sizes".
The resolution code wrote its answer **back into the run-wide config dict**, so the formula ran
exactly once — for whichever NN cluster happened to be eliminated first — and every later cluster
silently inherited that number. This is the [[sample-generation]] analogue of a caching bug, and
its consequence is that **the training-sample count is confounded with the experimental arm**
(doc 11).

## Mechanism

In `nce/inference/bucket.py`, **both** branches of `compute_message_nn` (memorizer path ~L275,
NN path ~L419) did:

```python
nbe_result = self.get_nbe_num_samples(epsilon)
self.config['num_samples'] = nbe_result['total']   # int, into the SHARED dict
```

`FastBucket.config` **is** `self.gm.config` — one dict for the whole run — and `Trainer.__init__`
takes the same reference. After the first NN cluster the value is an `int`, so the
`isinstance(num_samples_cfg, str)` guard never fires again. The formula itself
(`compute_nbe_num_samples(w, l, epsilon)`, ~L1315) is per-cluster and correct; it simply stopped
being called.

**Measured on `grid10x10.f10`, iB=10, ecl=1025, no merge** (doc 11): all six NN buckets trained on
`m = 6386`; the correct values were 9223 / 7730 / 6386 / 7730 / 6386 / 6386. Distinct `m` across
NN buckets: **1 before, 3 after**. The config ended the run holding the int `6386` instead of the
string `'nbe,0.1'`.

## Why it invalidates comparisons rather than just adding noise

A structure-only pass over 3 problems × 4 merge arms (doc 11 §3, `scope_mismatch = 0` in all 12
cells) found:

- **57–94% of NN clusters were under-sampled in every arm but one.** Worst single cluster got
  **1/8th** of its correct sample count (pedigree7 / sub12, where 34 of 36 clusters were starved).
- Total-sample multipliers $\sum m_\text{correct} / \sum m_\text{frozen}$ span **×0.50 to ×2.93**
  across cells — i.e. the bias runs in **both directions**, which is what kills the
  "all arms were equally under-trained, so the comparison still holds" defence.
- The TL;DR states the confound concretely: at `e_max` 12, reduce-NN used **16,735** samples/net
  while subsumption-only used **6,386** — for the same problem. Any accuracy difference attributed
  to merge strategy partly measures *which arm happened to eliminate a wide cluster first*.

**What is not affected**: everything structural. Merge decisions, NN-vs-exact eligibility, cluster
counts and widths are all computed before any training, so results like "reduce-NN merging cuts
the NN cluster count from 108 to 10" stand unchanged (doc 11 §4).

## Downstream consequences already recorded

- **[[convergence-diagnostic-gap|Epoch counts may measure the wrong thing.]]** Doc 04's
  convergence-speed study found that on grids a *larger* separator predicts *fewer* epochs
  ($\rho = -0.86$) — the opposite of the expected sign. Its stated hypothesis: with every bucket
  training on the same frozen `m`, a wide separator means more parameters fitted to samples from
  an exponentially sparser space, so the net memorises fast and patience fires early. If that is
  right, `epochs_trained` in that study measures **memorisation speed, not message accuracy**, and
  re-running with per-cluster counts would change the interpretation of every epoch number.
- **It moved the time-optimal merge bound.** The fix raises per-cluster `m` by a median 2.26×, and
  more at low `e_max` (3.14× at grid40 `e_max`=6 vs 1.82× at 16), inflating training cost
  asymmetrically — one of the two forces pushing $e_{\max}^{*}$ up in doc 17. See
  [[time-optimal-merge-bound]].

## The fix

Branch `fix/num-samples-per-cluster`, commit `2028abe` (also carried into
`integration/aug11-fixes`; **not** on `perf/nn-eval-fixes`). New `FastBucket` methods next to
`get_nbe_num_samples`:

- `resolve_num_samples()` → dict, parses against *this* bucket's scope, **never assigns to
  `self.config`**;
- `get_num_samples()` → the int.

Nine consumers rewired. The load-bearing one is `Trainer.train`, where `num_samples` drives
`set_size`, `num_sets` and `num_batches_per_set` — and because `set_size` defaults to `None`, in
every NeuroBE config `num_samples` **is** `set_size`. `compute_linear_mse_message` would have
*crashed* (string `// 9`) had it run first. Deliberately **not memoised**: a cache filled before a
bucket's incoming messages arrived would reintroduce a milder version of the same staleness.

⚠️ **Two consumers outside the fix depended on the mutation**: `nce/benchmark/proposal_in_elim.py`
~L138 and `nce/benchmark/training.py` ~L497 both do `config.get('num_samples', 10000)` and need an
int — they worked *only because* of the bug. Doc 11 flagged them; doc 13 repaired them during
integration. Without that repair the proposal-sampling path breaks on the string.

Verification deliberately avoids $\log Z$ (which was not reproducible at the time — see
[[bit-exact-reproducibility]]): assert the config still holds the formula string, assert each NN
bucket's `m` equals the formula for its own scope, assert >1 distinct `m`. PASS, 0 failures.

## Related

- [[sample-generation]] · [[mini-bucket-sampling]] · [[neural-bucket-elimination]]
- [[time-optimal-merge-bound]] · [[bit-exact-reproducibility]] · [[convergence-diagnostic-gap]]
- [[codebase-map]] · [[2026-W33]]
