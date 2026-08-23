# 64 — memorization selection: `fw_true` vs `fw_bw`

> ## Verdict — READ THIS FIRST
> **The pre-registered primary endpoint is wrong for this experiment, and it misled.**
> On the local-error metric, `fw_bw` beat `fw_true` by **+8.388 dex, 50/0, p = 1.8e−15**
> with median |error| of *exactly zero* on two cells. End-to-end |log Z − reference| says
> something quite different on the same three cells:
>
> | cell | end-to-end `fw_bw` | end-to-end `fw_true` | Δ | verdict |
> |---|---|---|---|---|
> | `grid10x10.f10.wrap` | 0.1724 | **0.0750** | −0.097 | overlap — `fw_true` nominally better |
> | `grid20x20.f10` | 0.2171 | **0.1913** | −0.026 | overlap — `fw_true` nominally better |
> | `dbn/rbm_20` | **0.0008** | 0.0764 | +0.076 | **SEPARATE — `fw_bw` genuinely better** |
>
> **Only `rbm_20` is a real separation.** On both grids `fw_true` is nominally better and
> the per-seed ranges overlap. The pre-registered metric got the sign backwards on 2 of 3
> cells while reporting p = 1.8e−15.
>
> **Why the metric fails, structurally:** memorized entries are exact *by construction*, so
> a cluster's local error collapses toward zero on exactly the entries the arm memorized —
> whether or not the final answer improves. `fw_bw` memorizes higher-*contribution* entries,
> so it zeroes the local metric harder while, on grid20, making log Z worse. The
> `p = 1.8e−15` additionally treats clusters within a run as independent; they are not.
>
> **What does hold on all three cells:** memorization of either kind beats no memorization
> end-to-end (`fw_true` vs `base` and `fw_bw` vs `base` favour memorization 3/3). The open
> question is only *which selection rule*, and that is 1-for-3 in favour of `fw_bw`.
>
> **Status: full-benchmark re-run IN FLIGHT** (21 configs, gpu1, launched 2026-08-22).

---

## 0. What is being tested

Memorization currently picks the top-K separator entries by **forward** message value.
What actually matters for log Z is an entry's **contribution**, forward × backward. `fw_bw`
ranks by that instead. Arms: `base` (no memorization) | `fw_true` (forward, the status quo)
| `fw_bw` (forward × approximate backward).

The approximate backward factors are built from the existing cluster structure via
`FastBucket.get_backward_factor_list()` **without** enabling `use_bw_approx` and without
changing the NN training target or the preprocessor. Every run records `preproc_audit` so
that is *checked, not asserted* — `64_analyze.py` leads with that constraint audit, because
a silently-degraded `fw_bw` arm is indistinguishable from a null result.

Operating point is doc 50's recommendation: memorize 10% of entries, 2:1 sample pool,
`bw_ecl=1024`, 500 epochs, subsumption merge D=10.

## 1. Full-benchmark extension (in flight)

Extended from 3 cells to the **21 non-grid40 configs** of the reduce-NN benchmark set:
pedigree ×6 (iB=20), grid-small ×6 (iB=10), rbm ×6 (iB=20), rbm ×3 (iB=10). Pedigree was
untested and is the largest gap. `grid40x40` is **deliberately excluded**: measured from the
live sweep, `grid40x40.f15.wrap` at sub D=10 costs 41,734 s per run — 8 configs × 21 runs
≈ 1,948 GPU-hours ≈ 81 days on one card.

Each group uses **its own** iB/ecl (`ecl = 2**iB + 1`); the old `64_drive.sh` hardcoded
`--ib 10 --ecl 1025` and could not express that, so the queue is now `q60_queue.py`
(restartable: completed runs are indexed by `(problem, iB, merge, D, arm, seed)` read out
of each JSON) with `scripts/q60_watchdog.sh` restarting it, since this box kills multi-day
detached processes.

**Endpoint, as corrected:** primary is **end-to-end |log Z − reference|**, per problem, mean
over 3 seeds, **with the per-seed range printed**, significance judged by whether those
ranges separate, plus a sign test over *cells* (independent problems — the one legitimate
sign test here). Per-cluster local error is kept as **secondary** with disagreements
flagged. References: `results_for_writeup/problem_overview_table.csv` `ref_log10Z` where
`ref_kind == 'exact'` (17 problems). `rbm_ferro_20` has no exact reference and is reported
reference-free.

> **`pedigree51` reference — RESOLVED, not an open question.** `−73.871` is correct and
> supersedes `−77.27`; Nick confirmed 2026-08-22 ("the higher value one is the correct one"),
> and the correction is already on the record in `docs/25` U10 (exact-solver validated,
> checked against pedigree13 to 4e-6) and `docs/27` §8, which treats `−77.27` as the
> *pre-correction* value. The overview table this analysis reads already carries `−73.871`,
> so **no result here was affected**. The stale `−77.27` in
> `reduce_nn_experiment/benchmark_set.json` has been corrected to match.

## 2. Constraints held

- **gpu1 only.** gpu0/gpu3 carry the paper rerun whose timings are a published result;
  gpu2 is retired for silent data corruption.
- Wall times are gpu1 and **indicative only** (~10% thermal throttle). This is an accuracy
  experiment, which is why gpu1 suits it.
- 3 seeds (42, 43, 44). No config value overridden — `device`, `num_epochs`, `ecl`, `iB`,
  `hidden_sizes`, `loss_fn` are all as specified.

Artefacts: `64-results/` (run JSONs), `64_run.py`, `64_analyze.py` (constraint audit +
secondary local metric), `q60_queue.py`, `q60_e2e_analyze.py` (primary end-to-end).

Analysis: `q60_e2e_analyze.py --exp 64 /tmp/claude-58902/res64`.
