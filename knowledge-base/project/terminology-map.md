---
type: project
title: Terminology Map (NCE ↔ literature)
status: growing
tags: [this-project, terminology, writing]
created: 2026-06-12
updated: 2026-08-12
---

# Terminology Map (NCE ↔ literature)

How NCE's code/config terms map to standard graphical-models vocabulary. **Use the
standard term in papers**; flag project-specific terms when they appear. See the
[[glossary]] for definitions and [[codebase-map]] for where each lives.

> ⚠️ **Read the 2026-08 rename block below before writing any prose.** The code names and the
> writing names diverged in August 2026 and the code was *not* renamed. Writing straight from
> the config keys now produces wrong prose.

| NCE term | Standard term | Notes |
|---|---|---|
| `iB` | **s-bound (sB)** in NCE prose since 2026-08; **i-bound** only where it really partitions | See the rename block. The exact-vs-learned gate compares `iB` against `FastBucket.get_width()`, which is the **separator** size, not a cluster width ([[codebase-map]]). The same value *is* a true i-bound when passed to `compute_wmb_message` ([[iB-parameter]], [[mini-bucket-elimination]], [[@dechter2003minibuckets]]). |
| `ecl` | (no standard term) | "Exact-computation limit" = max **table size** (entries) for exact computation. Project-specific; can override `iB` when partitioning. Closest standard idea: a table-size cap. |
| `EC` | table size / state-space size | Product of domain sizes of a bucket's scope. |
| induced width | **induced width / treewidth** | Standard ([[induced-width]]). NCE sometimes says "width". |
| `max_merge_bound` | (project-specific) | **Merge bound**, written **`e_max`** in prose since 2026-08 (was `D`). Max **number of variables** eliminated per merged cluster. Was `max_cluster_size`. NOT "merge degree". [[merge-bound]]. |
| (no config key) | — | **`e_max*`** = the *predicted cost-optimal* merge bound from the cost model (was `D*`). Distinct from the *observed* time-optimal value; the two disagree, see [[time-optimal-merge-bound]]. |
| `merge_degree` (arg of `merge_by_degree`) | — | A **different, older** knob from `max_merge_bound`. Do not call either one "merge degree" in prose. |
| "super-bucket" | **merged cluster / super-cluster** | Loose term in the literature; prefer "merged cluster" unless quoting. [[super-bucket]], [[@kask2005unifying]]. |
| `merge_join_tree` | subsumption / running-intersection merge | [[running-intersection-property]], [[join-tree]]. |
| `fdb` (loss suffix) | "forward diff barrier" | Project-specific: stop-gradient (detached) on the normalizer. [[loss-functions]]. |
| `neurobe_mode` | NeuroBE reproduction | Faithful [[neural-bucket-elimination]] defaults ([[@agarwal2022neurobe]]). |
| message scope | **separator** (junction-tree term) / message scope | Variables remaining after elimination. |
| `wtminfill_order` | **weighted min-fill** ordering | Standard heuristic ([[elimination-ordering]]). |
| log-space (NCE) | log base **10** | NCE uses $\log_{10}$; `pyGMs` and most papers use natural log. Convert with $\ln 10$. [[log-space-convention]]. |

## The 2026-08 rename: `D` → `e_max`, `iB` → s-bound

`D*` → `e_max*` was swept across **12 files / 90 lines, zero occurrences left**, along with bare
merge-bound `D` forms (`D = 10`, `D ∈ {…}`, `median D = 10`). The sweep is recorded in
`notebooks/June-2026/claude_experiments/reduce_nn_experiment/.claude/ai-ops/state/recovery.md`
(lines 11–26); the twelve files are `hardware_constants.md`, August docs 00-TLDR / 03 / 05 / 07 /
08 / 08a, the July write-up (`RESULTS.md`, `README.md`, `hyperparameters_and_cost.md`) and June's
`REPLY_TO_PAPER_AGENT.md` / `EXPERIMENT_HANDOFF.md`. **CSV field names (`r['D']`) were deliberately
left alone** — data columns, not notation; renaming them breaks the readers.

**No `nce/` code was renamed** — the config keys are still `max_merge_bound`, `iB`, `ecl`. This is
deliberate, not an oversight: renaming config keys "would break over 2,000 configuration files"
(doc 08 Q32), and the June record says to leave it "until the branch work settles rather than churn
identifiers under a running change". Nick's ruling, inline in doc 02 (decision **D9**):

> "Yeah, `e_max` and s-bound are better. For wmb, `iB` is typically the term used. I think `ecl`
> can stay put in the code for now."

So there are now two vocabularies and you must translate in one direction:

| write this | for this code name | never write |
|---|---|---|
| `e_max` (merge bound) | `max_merge_bound` | `D`, "merge degree", "cluster size" |
| `e_max*` (predicted optimum) | — (derived) | `D*` |
| **s-bound**, `sB` | `iB` | "i-bound", when you mean the gate |
| `ecl` (exact-computation limit) | `ecl` | — |

Benchmark cells are named `(problem, s-bound)` — e.g. `rbm_21_iB20` is "rbm 21 @ sB 20"
(doc 27). The benchmark is **27 (problem, s-bound) cells over 4 families**, and the s-bound
changes the *structure*, so `rbm_21` at sB 10 and at sB 20 are correctly not the same cell
(doc 27).

**The rename is incomplete in the artifacts and that is a live trap** (doc 25 §M6). `e_max` is
dominant in the July package (48 occurrences over 5 files) but `D` survives in figure captions,
CSV column names (`merge_counts.csv` has a `D` column), file names (`nn_vs_D_table.csv`) and
*exclusively* in both June `.tex` theory notes; the arm names themselves are `rnn12` / `sub16`.
For the s-bound, `local_error_tables.md` headings say `_sB10` while the CSV behind them says
`_iB10`, and all 27 problem keys are `<problem>_iB10` / `_iB20`. Doc 25's recommendation, which
this note adopts: **sweep prose, captions and headers; leave CSV column names and config keys
alone.** Expect to read `iB` in data and write `sB` in text.

⚠️ This entry supersedes the pre-2026-08 mapping `iB → i-bound / "use the standard term in
papers"`, which doc 25 flagged as being in direct tension with Nick's D9 ruling above.

### `e_max` bounds a COUNT, not a state-space product

This is the single most repeated confusion of the week (docs 01, 02, 27; lab 2026-08-10 and
2026-08-10 rev 2). The merge passes test

```python
merged_size = len(set(cur_bucket.elim_vars + parent_bucket.elim_vars))
if max_merge_bound is not None and merged_size > int(max_merge_bound): break
```

— `graphical_model.py` ~L613 (`merge_join_tree`) and ~L678 (`merge_non_subsumption`), with the
docstring ~L578 saying so outright. It is a **count of eliminated variables**, roughly 2..24. It
does **not** cap $\prod_{v \in \text{elim}} |D_v|$.

- "elimination size 1025" is **not** a valid `e_max`. 1025 is an **`ecl`** value, the one
  paired with sB = 10 (doc 01).
- On binary problems the two readings coincide at `e_max = 10` ($2^{10} = 1024 \le 1025$).
  They diverge on mixed-domain problems: largest legal `e` under the state-product reading is
  10 for grids/RBMs, **6** for pedigree13 ($k_{\max}$=3), **5** for pedigree7 (4), **4** for
  pedigree51/41 (5) — and no single count expresses the rule (docs 01, 02).
- **Recorded disagreement — Nick's intent vs the code.** Nick resolved the intended semantics as
  the *state product* ($\prod |D_v| \le 1025$; lab 2026-08-10 rev 2, doc 02 §2.4, decision D11),
  while the code implements — and every August experiment ran with — the *count*. The fix was
  scoped twice (~6 lines plus a `max_merge_states` config field, doc 02; ~15 lines reusing the
  `states_prod` helper at `graphical_model.py` ~L878, doc 01) and **applied neither time**. When
  quoting an `e_max`, say which reading it is under.
- **Recorded disagreement — how much it matters.** Doc 25 §S4 calls it severe: worst case is four
  orders of magnitude ($5^{10}$ vs $2^{10}$), so a pooled "median optimal `e_max` over 27 problems"
  pools incomparable units across families. Doc 27 §5 audits it and concludes the opposite —
  $k_{\max}$ is the wrong statistic, the pedigrees are **effectively binary on average**
  ($\bar k$ = 1.87–2.23 vs exactly 2.00 for grids and RBMs), so re-expressing each observed optimum
  as $e^{*}\log_2 \bar k$ moves the pooled median over 27 cells from 8.0 to **8.00**, and only to
  9.29 on a deliberately pessimistic $k_{\max}$ basis — inside the ±2 sweep resolution. Doc 27's
  verdict: "real in principle and quantitatively negligible here. The headline does not move."

## Citation gotchas (get these right when writing)

- **NeuroBE** = **UAI 2022** (PMLR 180, pp. 11–21), title "Escalating *Neural Network*
  Approximations *of* Bucket Elimination", authors Agarwal, Kask, Ihler, Dechter — **not
  AAAI 2022**, not "NN…to". ([[@agarwal2022neurobe]])
- **Deep Bucket Elimination** = IJCAI 2021; six authors; **Marinescu and Ihler are not
  authors**. ([[@razeghi2021deep]])
- **Mini-buckets JACM 2003** is **Dechter & Rish** (Dechter first). ([[@dechter2003minibuckets]])
- **Bucket elimination** title splits: 1999 AI journal = "…for reasoning"; 1996 UAI / 1998
  chapter = "…for probabilistic inference". ([[@dechter1999bucket]])

## Related

- [[glossary]] · [[codebase-map]] · [[nce-method-overview]] · [[merge-bound]] · [[2026-W33]]
