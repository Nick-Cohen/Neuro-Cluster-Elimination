---
type: concept
title: Backward-Factor Population Under Merging
status: budding
tags: [this-project, correctness, wmb, backward-messages, bucket-merging]
created: 2026-08-12
updated: 2026-08-12
---

# Backward-Factor Population Under Merging

[[backward-messages|Backward-factor population]] builds each cluster's *downstream context* by
running a second, WMB-based sweep over a copy of the graphical model. It worked for single-variable
buckets and was **broken for merged clusters** — crashing under two of the four merge strategies
and, where it did not crash, silently substituting a partially-marginalised surrogate for the
cluster's true upstream. This blocked all proposal sampling under [[bucket-merging]] (docs 01, 02;
repaired in doc 10).

## Four defects

Found by doc 01 §0a and doc 02 §0.3, all build-only on `grid10x10.f10`:

1. **The population copy re-merges (crash).** `_create_population_copy` sets
   `pop_config['use_join_tree_merge'] = False` — **1 of the 4 merge passes**. `use_reduce_nn_merge`,
   `use_non_subsumption_merge` and `merge_degree` are untouched and `FastGM.__init__` dispatches all
   four independently. The copy therefore merges, and `get_senders_receivers` raw-indexes
   `self.buckets[var]` and raises. Measured: no-merge 100/100 OK; subsumption-only OK; non-subsumption
   → `KeyError: Var (0,2)`; sub+non-sub → the same. It kills **both** population routes.
   (`_wmb_eliminate_to_scope` had the identical omission.)
2. **`get_senders_receivers` was not merge-aware.** It walked the full `elim_order` — but absorbed
   buckets are removed from `self.buckets` while *staying* in `elim_order` — and it discarded only
   the key variable from the outgoing scope, where `FastBucket.get_message_scope` already correctly
   discards **all** of `bucket.elim_vars`.
3. **Merged clusters got a single-variable surrogate for their upstream.** Where tree-collect does
   run, it assigns `orig_bucket.approximate_upstream_factors = list(copy_bucket.factors)`. But
   `copy_bucket` eliminates *one* variable while the cluster eliminates up to `e_max`, and the
   absorbed children's variables have **already been WMB-eliminated out** by the time of the
   snapshot. The `.get()` guard prevents the crash; it does not make the content right. This is the
   difference between merge-*tolerant* and merge-*correct*.
4. **The proposal scope blew up.** `build_proposal_for_bucket` branches on cluster size: single-var
   buckets read the cached `message_scopes`, but merged clusters union every label in
   `upstream + downstream + bucket.factors`. Measured at `e_max`=12: cluster 60 (3 elim vars,
   separator 1) derived a scope of **14**; cluster 37 → **16**; cluster 31 → **17** — while every
   single-var bucket stayed at 2–3 despite carrying 37–54 downstream factors.

## How wrong was the surrogate, numerically?

This is the number doc 02 flagged as unmeasured and doc 10 §4.4 supplied. Comparing each cluster's
forward message at a realistic `bw_ecl = 1024` against the same at `bw_ecl = 2²²`:

**Cluster 60 was off by 2.699 $\log_{10}$ — a factor of ~500 — pre-fix, and 0.0 post-fix.**
Cluster 37: 0.0 → 7.6e-06.

Doc 10 is scrupulous that its *other* test does not show this: the $\log Z$ decomposition check
passes on pre-fix code too on the one row that runs, because when the WMB steps are exact,
summing a subtree out early or late gives the same answer. §4.3 proves the new decomposition is
complete and non-overlapping; **§4.4 is the evidence that the old upstream was numerically wrong.**

## The repair

Branch `fix/wmb-under-merging` (worktree `/tmp/wmb-fix`), commits `171d554` (baseline snapshot) +
`93f6a06` (the repair). Not merged, not pushed.

- Defects 1: all four passes disabled in *both* `_create_population_copy` and
  `_wmb_eliminate_to_scope`.
- Defect 2: `get_senders_receivers` skips vars absent from `self.buckets` and discards all
  `elim_vars`.
- Defect 3: **a cluster's upstream cannot be patched at the point of use — it has to be accumulated
  across the whole copy sweep.** Five new helpers record each variable's copy-bucket factor list at
  the moment it is processed, plus the ids of the messages that bucket *produced*; per cluster,
  union the members' snapshots and drop cluster-internal messages, reconstructing exactly
  `(all cluster originals) + (forward messages entering from outside)`.
- Defect 4: the cluster branch now just calls `bucket.get_message_scope()`, factored out as
  `proposal_scope_for_bucket` so the scope can be asserted without building a tree.
- Plus: tree-collect now honours `populate_bw_skip_non_nn` (only the WMB route implemented it) —
  on grid20x20.f10 under reduce-NN that skips **292 of 308** clusters.

**Results.** Build matrix **2/12 → 12/12** across 6 merge strategies × 2 population routes;
reduce-NN — the acceptance criterion, and a strategy doc 01 never tested — works on grid10x10.f10
and grid20x20.f10. Cluster upstream missing elim vars **5/5 → 0/5**. Proposal scope 14/16/17
against a true separator of 1/1/0 → **exactly the separator**, beating doc 01's own regression
target. With `bw_ecl = 2²²` the upstream+downstream decomposition reproduces exact
$\log_{10} Z = 303.085957$ to **7.2e-05** for every strategy and route. New suite
`tests/test_wmb_merge_repair.py` (186 lines, 5 tests × 12 parametrisations, 3.3 s): 48 errors /
9 failures before, 54 passed / 6 skipped after, with the same 7 pre-existing unrelated failures on
both trees.

Doc 13 later ran the first **trained** elimination under reduce-NN merging with
`populate_bw_factors` on both routes; both complete, with bucket 9 consuming 23 pre-computed WMB
backward factors.

## The blocker underneath the blocker

Doc 10 §1 found something worth remembering independently: **no branch in the repository contained
`merge_join_tree`, `reduce_nn_merge`, `merge_non_subsumption`, `merge_by_degree`, the tree-collect
populator, `_wmb_eliminate_to_scope`, `proposal_sampler.py`, or any of the five no-replacement
samplers.** They existed only as uncommitted working-tree files (781 uncommitted insertions in
`graphical_model.py` alone). Branching-and-patching was impossible; every August branch therefore
starts with a "check in the pre-existing working state" commit. This is the same "no single source
of truth" blocker first raised in [[2026-W24]], still unresolved two months later.

## Estimate vs actual, recorded honestly

Doc 01 estimated ~80–120 lines across 4 sites. Doc 10 measured **~340 source lines + 186 test
lines** — roughly 3×, because of the accumulation point above. Doc 10 also corrects doc 01 twice:
**two of doc 01's four defect-3 sites were non-issues** once the copy is guaranteed unmerged
(`has_elim_var`, and a logging-only `bucket_var` argument), and **doc 01 mislabelled defect 2** —
it implies tree-collect's *downstream* is also a partially-marginalised surrogate; only its
**upstream** was wrong.

## Scope

Not done: no proposal *tree* was built (only its scope asserted), pedigrees untouched, all CPU.
Note also that "WMB is merge-safe" and "WMB is safe under approximate upstream context" are
**different claims** — the first was verified here, the second was false; see the gotcha section of
[[weighted-mini-bucket]].

## Related

- [[backward-messages]] · [[bucket-merging]] · [[weighted-mini-bucket]] · [[super-bucket]]
- [[importance-sampling]] · [[mini-bucket-sampling]] · [[codebase-map]] · [[2026-W33]]
