# S02: ECL Tuning & Comparison Experiments — Research

**Date:** 2026-03-12

## Summary

S02 has two deliverables: (1) per-problem ecl values that match NeuroBE's NN counts for all 15 binary-domain problems (R036), and (2) a combined comparison table from running those problems through NCE neurobe_mode (R038).

The ecl formula is confirmed: **`ecl = 2^width_problem - 1`** where `width_problem` comes directly from the NeuroBE results CSV. This follows from the dispatch condition analysis — NCE triggers NN when `message_size > ecl`, NeuroBE triggers when `_Width > width_problem`, and for binary domains `message_size = 2^(nce_width)` where `nce_width = _Width - 1`. Setting `ecl = 2^wp - 1` makes these equivalent.

**Critical blocker discovered:** `FastGM._load_from_uai()` cannot currently load the 15 binary-domain problems from the model catalog. The `.vo` file format skips the root variable (by design — `skip_first=True` in pyGMs `GraphModel.__init__`), producing an elimination order with n-1 variables. Factors whose only variable is the root can't be placed in any bucket, raising `ValueError`. This affects ALL catalog models loaded via `.vo` files, not just neurobe_mode. The fix is straightforward — either prefer `model.order` (which has all n vars from `.ord.elim` format) over the `.vo` file, or add root-var bucket handling in `_create_buckets_from_factors`. This must be fixed in T01 before any experiments can run.

## Recommendation

**Three-task structure:**

- **T01: Fix root-variable loading + build neurobe config module.** Fix `_load_from_uai` to handle root variable (prefer `model.order` when available, or absorb root-only factors as constants). Then build a `neurobe_binary_problems` config builder in `nce/benchmark_problems/` that generates neurobe_mode configs with per-problem ecl values computed from the NeuroBE results CSV. Verify NN counts match for all 15 problems using `get_large_message_buckets()` — no GPU needed, just bucket structure.

- **T02: Run experiments and collect results.** Run all 15 problems through NCE neurobe_mode on CUDA. This is the GPU-intensive task. Each problem trains 1–4 NNs with 500 epochs max (patience-based early stopping will cut most shorter). Based on NeuroBE runtimes (0.01–0.59 hrs per problem, ~1.5 hrs total), NCE should take similar order of magnitude. Run in background, collect results.

- **T03: Build comparison table and verify.** Parse NCE results alongside NeuroBE CSV into a combined comparison table (CSV + printed). Columns: Problem, NCE_log_Z, NeuroBE_log_Z, NCE_NNs, NeuroBE_NNs, NCE_time, NeuroBE_time. Verify NN counts match. Assess result plausibility.

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| Per-problem ecl values | NeuroBE results CSV `width_problem` column | `ecl = 2^wp - 1` is deterministic — no trial-and-error needed |
| NN count verification | `FastGM.get_large_message_buckets(iB=25, ecl=ecl)` | Returns bucket list without running inference — instant verification |
| neurobe_mode config | `NEUROBE_DEFAULTS` in config_schema.py (16 keys) | S01 built the full preset; only ecl and num_samples vary per problem |
| Benchmark set pattern | `BenchmarkSet` class in `nbe_sanity_check.py` | Proven pattern: problems list + configs dict, used by small_problems too |
| Model loading from catalog | `get_catalog()` → `catalog[key]` → `model.file` | All 15 binary-domain models already cached in `.model_cache/` |
| NeuroBE hidden dim sizing | `hidden_sizes='neurobe,3'` mode in bucket.py | S01 implemented `scope_size * b` computation — matches NeuroBE's `nArgs * var_dim` |
| NeuroBE sample count | `num_samples='nbe,0.1'` mode | Existing pseudo-dimension formula in `FastBucket.compute_nbe_num_samples()` |

## Existing Code and Patterns

- `Clean-NeuroBE/results/binary_min_nn/binary_domain_results.csv` — Ground truth. 16 rows (15 working + or_chain_10 failed). Columns: Problem, Type, nVars, Evidence, iB, width_problem, MaxWidth, NNs, Avg_Test_MSE, Log_Z, Runtime_hrs. All use iB=25.

- `nce/benchmark_problems/small_problems.py` — Has all 15 binary-domain models in `_MODEL_KEYS` (plus 9 non-binary models). `_AUTO_ECL` values do NOT match NeuroBE dispatch — 9 of 15 differ. The new neurobe config builder should NOT reuse these ecl values.

- `nce/benchmark_problems/nbe_sanity_check.py` — Pattern to follow for the new benchmark module. Key structure: `_MODEL_KEYS` list, per-model maps for varying params, `_build_*_configs()` function, module-level `BenchmarkSet` instance.

- `nce/inference/graphical_model.py` lines 22-35 — `FastGM.__init__` extracts `model.order` (tuple of 100 ints from `.ord.elim`) but then `_load_from_uai` calls `uai_to_GM(order_file=uai+'.vo')` which overrides it with the 99-var `.vo` order. This is the root-variable bug.

- `nce/inference/graphical_model.py` line 400 — Dispatch condition: `if (exact or (bucket_width <= self.iB and bucket_ec <= self.ecl)) or bucket_msg_complexity < self.complexity_limit`. With iB=25 and all problems having MaxWidth ≤ 24, `bucket_width <= iB` is always true. Dispatch depends entirely on `bucket_ec > ecl`.

- `nce/inference/graphical_model.py` line 675 — `get_large_message_buckets(iB, ecl)` pre-computes which buckets would get NN treatment without running inference. Uses `message_size > ecl` condition. Useful for NN count verification in T01.

- `nce/inference/bucket.py` lines 1123-1131 — `get_message_scope()` returns scope minus eliminated var. `get_width()` = len of message scope. `get_ec()` = `get_message_size()` = product of domain sizes for message scope vars.

- `nce/config_schema.py` lines 146-163 — `NEUROBE_DEFAULTS` dict with 16 keys. Includes `iB=25`, `hidden_sizes='neurobe,3'`, `batch_size=256`, `lr=0.001`, `num_epochs=500`, etc. The ecl must be overridden per problem.

## Constraints

- **All 15 problems are binary domain (domain size 2).** Message sizes are powers of 2. `ecl = 2^wp - 1` is exact.

- **iB=25 for all problems.** NeuroBE used iB=25 universally. NEUROBE_DEFAULTS already sets this. No per-problem iB variation needed.

- **MaxWidth ≤ 24 for all 15 problems.** With iB=25, the `bucket_width > iB` condition never triggers. NN dispatch is purely ecl-based.

- **or_chain_10.fg is excluded.** Failed in NeuroBE (marked with `0*` NNs, `—` for Log_Z). Not in the catalog. 15 problems, not 16.

- **`model.order` vs `.vo` file.** `model.order` from catalog returns all n vars (from `.ord.elim` format). `.vo` file skips root. `_load_from_uai` currently prefers `.vo`. Must fix to use `model.order` when passed as `elim_order`.

- **GPU availability.** 4× NVIDIA TITAN RTX. Default device='cuda'. Can run problems in parallel on different GPUs if runtime is a concern, but NeuroBE total was ~1.5 hrs — serial is fine.

- **`num_samples='nbe,0.1'` uses epsilon=0.1.** NeuroBE binary_min_nn used epsilon=0.1. The pseudo-dimension formula gives per-bucket sample counts. S01's NEUROBE_DEFAULTS does NOT set num_samples — must be set per problem or globally in the neurobe config builder.

## Common Pitfalls

- **Root variable bucket placement.** `_create_buckets_from_factors` raises ValueError when a factor's only variable isn't in the elimination order. The `.vo` file excludes the root. Fix: when `elim_order` is passed to `FastGM.__init__`, prefer it over `.vo` file. Or: in `_create_buckets_from_factors`, absorb root-only factors as constants (add their log-value to a running Z accumulator).

- **`_load_from_uai` ignores passed `elim_order` when `.vo` file exists.** `uai_to_GM(order_file=ord_file, elim_order=elim_order)` — if `ord_file` exists, it's used and `elim_order` is ignored. Fix: don't pass `order_file` when `elim_order` is already provided.

- **ecl vs auto_ecl confusion.** `small_problems.py` has `_AUTO_ECL` values that don't match NeuroBE dispatch for 9/15 problems. The neurobe config builder must compute its own ecl values from the NeuroBE CSV, not reuse `_AUTO_ECL`.

- **`get_large_message_buckets` uses full scope width for iB check.** `num_vars = len(scope)` includes the eliminated var. `process_bucket` uses `get_width()` which excludes it. For iB=25 with MaxWidth≤24, this discrepancy doesn't matter. But document it.

- **num_samples not in NEUROBE_DEFAULTS.** The neurobe_mode preset doesn't set `num_samples`. Each problem needs a value. Use `'nbe,0.1'` (NeuroBE's epsilon=0.1 pseudo-dimension formula) for all problems.

- **Evidence handling.** 7 of 15 problems have evidence (BN_1 through BN_11 with varying evidence counts). Evidence files exist in `.model_cache/`. `_load_from_uai` auto-detects `.evid` files. But if `model.evidence` is also passed, evidence could be applied twice. Check: when fixing the root-var bug, ensure evidence is applied exactly once.

## Open Risks

- **Root-variable fix may change bucket structure.** Adding the root variable to the elimination order adds one more bucket. This bucket will always be exact (trivially small). But it changes `num_trained` counting and `get_large_message_buckets` output. Verify NN counts are unaffected — root var factors should be small enough to stay under ecl.

- **Numerical divergence between NCE and NeuroBE.** Even with matched algorithms: different RNG seeds, PyTorch vs libtorch numerical differences, Python vs C++ floating-point ordering. Results should be "comparable" (same order of magnitude error), not "identical." The comparison table should show both values and let the user judge.

- **Sample generation differences.** NeuroBE uses 80/20 train/val split of pseudo-dimension-based count. NCE's `sampling_scheme='all'` with `val_set` handling may differ. This is a known difference from S01 research — accepted as-is for this comparison.

- **Runtime estimation uncertainty.** NeuroBE total runtime was ~1.5 hrs. NCE may differ due to: Python overhead, different sample counts, early stopping behavior differences. Estimate 1–3 hrs total. Run in background without timeout.

- **Degenerate bucket epsilon guard.** S01's DataPreprocessor handles `ln_max == ln_min` with an epsilon guard, but this is untested in real inference (S01 Forward Intelligence). May surface on problems with constant-value buckets.

## Candidate Requirements (Advisory)

None — R036 and R038 are well-scoped. The root-variable fix is a prerequisite bug fix, not a new requirement.

## Skills Discovered

| Technology | Skill | Status |
|------------|-------|--------|
| PyTorch | — | none relevant — domain-specific inference work |
| Python scientific computing | — | none relevant |

No relevant skills for this slice. The work is benchmark configuration, a bug fix, and experiment execution.

## ECL Values (Computed)

Derived from NeuroBE results CSV using `ecl = 2^width_problem - 1`:

| Problem | width_problem | ecl | NeuroBE NNs |
|---------|--------------|-----|-------------|
| BN_1 | 19 | 524287 | 2 |
| BN_2 | 21 | 2097151 | 3 |
| BN_3 | 15 | 32767 | 1 |
| BN_5 | 15 | 32767 | 1 |
| BN_7 | 18 | 262143 | 1 |
| BN_8 | 23 | 8388607 | 4 |
| BN_9 | 23 | 8388607 | 1 |
| BN_10 | 15 | 32767 | 2 |
| BN_11 | 18 | 262143 | 1 |
| grid10x10.f5.wrap | 21 | 2097151 | 1 |
| smokers_20 | 19 | 524287 | 1 |
| 10_14_s.binary | 15 | 32767 | 3 |
| 10_16_s.binary | 16 | 65535 | 2 |
| 11_17_s.binary | 19 | 524287 | 1 |
| 11_4_s.binary | 18 | 262143 | 1 |

Note: 6 of 15 match existing `_AUTO_ECL` in small_problems.py. The other 9 have smaller auto_ecl values (more aggressive NN dispatch).

## Catalog Key Mapping

| Catalog Key | NeuroBE Name | ModelFile |
|-------------|-------------|-----------|
| bn/BN_1 | BN_1 | BN_1.uai |
| bn/BN_2 | BN_2 | BN_2.uai |
| bn/BN_3 | BN_3 | BN_3.uai |
| bn/BN_5 | BN_5 | BN_5.uai |
| bn/BN_7 | BN_7 | BN_7.uai |
| bn/BN_8 | BN_8 | BN_8.uai |
| bn/BN_9 | BN_9 | BN_9.uai |
| bn/BN_10 | BN_10 | BN_10.uai |
| bn/BN_11 | BN_11 | BN_11.uai |
| grids/grid10x10.f5.wrap | grid10x10.f5.wrap | grid10x10.f5.wrap.uai |
| alchemy/smokers_20 | smokers_20 | smokers_20.uai |
| segmentation/10_14_s.binary | 10_14_s.binary | 10_14_s.binary.uai |
| segmentation/10_16_s.binary | 10_16_s.binary | 10_16_s.binary.uai |
| segmentation/11_17_s.binary | 11_17_s.binary | 11_17_s.binary.uai |
| segmentation/11_4_s.binary | 11_4_s.binary | 11_4_s.binary.uai |

## Sources

- NeuroBE results CSV (`Clean-NeuroBE/results/binary_min_nn/binary_domain_results.csv`) — ground truth ecl values, NN counts, log_Z
- NCE `_load_from_uai` source (graphical_model.py:146-173) — root-variable bug analysis
- pyGMs `GraphModel.__init__` source — `.vo` file `skip_first=True` behavior confirmed
- NCE `get_large_message_buckets` source (graphical_model.py:675-720) — dispatch condition analysis
- NCE `process_bucket` source (graphical_model.py:390-450) — runtime dispatch condition
- S01 summary forward intelligence — NEUROBE_DEFAULTS keys, fragile points
