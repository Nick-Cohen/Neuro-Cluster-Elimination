---
type: project
title: Codebase Map (concept → code)
status: growing
tags: [this-project, code, navigation]
created: 2026-06-12
updated: 2026-08-12
---

# Codebase Map (concept → code)

Where each concept lives in `nce/`. Line numbers are approximate (core table re-verified
2026-08-12 on branch `perf/nn-eval-fixes`); re-grep if they drift. Fuller traces:
`.claude/ai-ops/state/codebase-trace.md`, `.planning/codebase/`.

⚠️ **Much of the August-2026 work lives on unmerged branches.** Where a symbol below exists
only on a branch, the branch is named. See [[2026-W33]].

| Concept | File | Symbol / notes |
|---|---|---|
| [[graphical-model-structure]] | `nce/inference/graphical_model.py` | `FastGM`; `eliminate_variables()`, bucket dispatch |
| [[bucket-structure]] | `nce/inference/bucket.py` | `FastBucket`; `compute_message_exact()` |
| [[factor-operations]] | `nce/inference/factor.py` | `FastFactor` (log10); `__mul__`, `eliminate()` |
| [[neural-network-factors]] | `nce/inference/factor_nn.py` | `FactorNN`; `nn_to_FastFactor` (streamed materialization) |
| [[elimination-ordering]] | `nce/inference/elimination_order.py` | `wtminfill_order()` (weighted min-fill) |
| [[weighted-mini-bucket]] / [[mini-bucket-elimination]] | `nce/inference/bucket.py` | `compute_wmb_message()` ~L997, `FastBucket._create_mini_buckets()` ~L1228; static twin `FastGM._create_mini_buckets()` `graphical_model.py` ~L1632 |
| [[bucket-merging]] — **the four merge passes** | `nce/inference/graphical_model.py` | `merge_join_tree` ~L573, `merge_non_subsumption` ~L650, `merge_by_degree` ~L712, `reduce_nn_merge` ~L791. All four read `max_merge_bound` (= `e_max`, [[merge-bound]]). |
| [[super-bucket]] | `nce/inference/bucket.py` | `FastBucket` with `len(elim_vars) > 1` |
| memory-bounded exact elim | `nce/inference/bucket.py` | `_compute_message_exact_chunked` ~L88 (`block_limit=2**26`) |
| [[sample-generation]] / [[importance-sampling]] | `nce/sampling/sample_generator.py` | `SampleGenerator`; the per-factor slice call is `sample_tensor_product` ~L251 |
| **`_get_slices`, both paths** | `nce/inference/factor.py` ~L207 (table) · `nce/inference/factor_nn.py` ~L151 (NN) | Same contract, two implementations. The table path always restricted enumeration to the elim vars actually in the factor's scope; the NN path did not — the [[nn-elim-enumeration-redundancy]] bug. Block variants: `_eval_elim_block` `factor.py` ~L281 / `factor_nn.py` ~L241. `FactorNN` has **no** `_get_values` on `perf/nn-eval-fixes` (inherits `factor.py` ~L340, which indexes a `None` tensor) — fixed only on `feat/wmb-residual`. |
| [[loss-functions]] | `nce/neural_networks/losses.py` | `logspace_mse_fdb`, `linspace_mse_fdb`, `neurobe_weighted_mse`, `unnormalized_kl`, … |
| NN training | `nce/neural_networks/train.py` | `Trainer.train()`; NeuroBE patience block ~L859. `Trainer.val_losses` is **write-only on `perf/nn-eval-fixes`** — nothing ever appended to it (doc 18); the append landed as `2ae8cc1` on `feat/wmb-residual` and is re-applied on `fix/determinism`. |
| **per-cluster sample count** | `nce/inference/bucket.py` | `'nbe,<eps>[,<n_min>]'` parsed at ~L265 (sample path) and ~L409 (`compute_message_nn`); formula in `compute_nbe_num_samples` ~L1315 / `get_nbe_num_samples` ~L1339. Both sites do `self.config['num_samples'] = nbe_result['total']` — the [[num-samples-freeze]] bug. Fix (`bucket.get_num_samples()`) is on `fix/num-samples-per-cluster` / `integration/aug11-fixes`, not on `perf/nn-eval-fixes`. |
| **backward-factor population** | `nce/inference/graphical_model.py` | `_create_population_copy` ~L1809 (disables only `use_join_tree_merge` — 1 of the 4 merge passes), `get_senders_receivers` ~L1087, `populate_backward_factors_wmb` ~L1948, tree-collect populator. All four repaired on `fix/wmb-under-merging` — see [[backward-factor-population-under-merging]]. |
| **determinism-critical sites** | see [[bit-exact-reproducibility]] | `FastGM._create_buckets_from_factors` `graphical_model.py` ~L220 (was `set(factors)`); `PyGMsWMBBackward` `nce/utils/pygms_wmb_interface.py` ~L648 (was `set(A) | set(B)` over identity-hashed pyGMs nodes); NeuroBE patience comparison `train.py` ~L880 (+ new `neurobe_es_min_delta`). Fixed on `fix/determinism` (`d5cca0e`). |
| dead code | `nce/neural_networks/net.py` | `BitVectorLookup` ~L193 — its only construction site `FastBucket.compute_one_to_one_nn` has **zero callers** in the repo (doc 28). `Memorizer` ~L137 is all-or-nothing with an O(N) Python-loop forward (doc 01). |
| config schema | `nce/config_schema.py` | nested 6-section config; `max_merge_bound`, `neurobe_mode`, `iB`, `ecl` |
| pyGMs interop | `nce/utils/pygms_conversion.py`, `pygms_wmb_interface.py` | log10 ↔ ln conversion (`LN10`) |
| benchmarks | `nce/benchmark_problems/` | `nbe_sanity_check`, `neurobe_binary`, `small_problems` |

## Decision logic (exact vs learned) — `FastGM.process_bucket`, `graphical_model.py` ~L427

```python
if (exact or (bucket_width <= self.iB and bucket_ec <= self.ecl)) or bucket_msg_complexity < self.complexity_limit:
    output_message = bucket.compute_message_exact()          # ~L439
else:
    if self.is_populating_backward_factors:
        output_messages = bucket.compute_wmb_message(self.iB)  # ~L452
    elif self.config.get('approximation_method') == 'nn':
        output_message = bucket.compute_message_nn()           # ~L457
```

**Read the gate carefully — `bucket_width` is not the bucket's width.**
`FastBucket.get_width()` (`bucket.py` ~L1356) returns `len(self.get_message_scope())`, i.e. the
size of the **separator**, and `get_ec()` (~L1382) returns the separator's state-space product.
So the `iB`/`ecl` gate bounds the *outgoing message*, not the cluster. This is why prose about
these experiments says **s-bound (sB)** rather than i-bound — see [[terminology-map]]. The same
`self.iB` value *is* used as a genuine i-bound when it is passed to `compute_wmb_message(self.iB)`
and reaches `_create_mini_buckets`, so one config key plays two roles.

`compute_message_nn`'s only live caller is the `else` branch above (verified, lab 2026-08-10 rev 2),
so "sample only for NN-approximated messages" is free.

## Tests

`tests/` on `perf/nn-eval-fixes`: `test_inference`, `test_nn_training`, `test_regression`,
`test_robustness`, `test_config_schema`, `test_neurobe_mode`, `test_benchmark_configs`,
`test_config_docs`, `test_scaled_checkpoints`, plus `conftest.py` and `PATTERN.md`.
7 failures in `test_benchmark_configs` / `test_config_docs` are **pre-existing config-bookkeeping
failures**, confirmed unrelated to the August work (docs 13, 30).

Branch-only suites:

- `tests/test_wmb_merge_repair.py` — branch `fix/wmb-under-merging`, 186 lines, build-only CPU
  matrix over 6 merge strategies × 2 population routes (doc 10).
- `tests/test_determinism_regression.py` + `tests/goldens/determinism_goldens.json` + marker/flag
  plumbing in `tests/conftest.py` — branch `fix/determinism` (`f212869`), the
  [[bit-exact-reproducibility]] suite: 13 CPU tests in 23.8 s, 19 CUDA tests (`--gpu`) in ~8.8 min,
  9 pinned goldens with provenance (doc 30). No `nce/` file changed by it.

## Related

- [[nce-method-overview]] · [[terminology-map]] · [[three-layer-architecture]] · [[2026-W33]]
