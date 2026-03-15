# S02: ECL Tuning & Comparison Experiments

**Goal:** All 15 binary-domain problems run through NCE neurobe_mode with matched NN counts, producing a combined comparison table against NeuroBE C++ ground truth.
**Demo:** A printed/CSV comparison table showing Problem, NCE_log_Z, NeuroBE_log_Z, NCE_NNs, NeuroBE_NNs, NCE_time, NeuroBE_time for all 15 problems. NN counts match NeuroBE for every problem.

## Must-Haves

- Root-variable loading bug fixed — all 15 binary-domain models load from catalog without error
- Per-problem ecl values computed from NeuroBE CSV using `ecl = 2^width_problem - 1` (D031)
- NN counts match NeuroBE for all 15 problems (verified by `get_large_message_buckets` before GPU runs)
- neurobe benchmark config builder module following BenchmarkSet pattern
- All 15 problems run through NCE neurobe_mode inference on CUDA
- Combined comparison CSV/table with both NCE and NeuroBE results
- `pytest tests/` passes (134+ tests green)

## Proof Level

- This slice proves: integration (real inference runs on real models with real GPU training)
- Real runtime required: yes (GPU inference on 15 problems)
- Human/UAT required: yes (comparison table reviewed for result plausibility)

## Verification

- `source venv/bin/activate && python -c "from nce.benchmark_problems import neurobe_binary; print(len(neurobe_binary.problems))"` → 15
- `source venv/bin/activate && python scripts/verify_nn_counts.py` → all 15 problems show matching NN counts (0 mismatches)
- `source venv/bin/activate && python -m pytest tests/ -v` → 134+ passed, 0 failed
- `ls notebooks/March-2025/neurobe_comparison_results.csv` → file exists with 15 data rows
- Combined comparison table printed to stdout with all 15 problems

## Observability / Diagnostics

- Runtime signals: Per-problem print during experiment run (problem name, NNs trained, log_Z, elapsed time)
- Inspection surfaces: `scripts/verify_nn_counts.py` — standalone NN count verification without GPU; comparison CSV for post-hoc analysis
- Failure visibility: If a problem fails during inference, the experiment script catches the exception, logs the problem name and error, and continues to next problem
- Redaction constraints: none

## Integration Closure

- Upstream surfaces consumed: S01's `neurobe_mode` config preset (NEUROBE_DEFAULTS in config_schema.py), DataPreprocessor minmax_01 mode, neurobe_weighted_mse loss, patience-based early stopping, neurobe hidden sizes, Net ReLU activation
- New wiring introduced in this slice: neurobe_binary benchmark module (config builder + BenchmarkSet), fixed root-variable model loading in `_load_from_uai`, experiment runner script, comparison table builder
- What remains before the milestone is truly usable end-to-end: nothing — this is the final slice of M003

## Tasks

- [ ] **T01: Fix root-variable loading and build neurobe benchmark config module** `est:45m`
  - Why: Root-variable bug in `_load_from_uai` blocks all 15 models from loading. The neurobe benchmark config module (with per-problem ecl values from NeuroBE CSV) is the foundation for all experiments. NN count verification (R036) must pass before burning GPU hours.
  - Files: `nce/inference/graphical_model.py`, `nce/benchmark_problems/neurobe_binary.py`, `nce/benchmark_problems/__init__.py`, `scripts/verify_nn_counts.py`
  - Do: (1) Fix `_load_from_uai` to prefer `elim_order` param over `.vo` file when elim_order is provided — don't pass `order_file` to `uai_to_GM` when we already have an elim_order. (2) Build `neurobe_binary.py` following nbe_sanity_check pattern: 15 model keys, per-problem ecl values from NeuroBE CSV (`ecl = 2^wp - 1`), `num_samples='nbe,0.1'`, `neurobe_mode=True`, iB=25. (3) Export from `__init__.py`. (4) Write `scripts/verify_nn_counts.py` that loads all 15 models, calls `get_large_message_buckets(iB=25, ecl=ecl)`, and asserts NN count matches NeuroBE for every problem.
  - Verify: `python scripts/verify_nn_counts.py` → 15/15 match, 0 mismatches. `pytest tests/ -v` → 134+ passed.
  - Done when: All 15 models load successfully, NN counts match NeuroBE for all 15, verification script passes, existing tests still green.

- [ ] **T02: Run neurobe_mode experiments on all 15 problems** `est:2h`
  - Why: This is the core R038 execution — running the actual inference experiments that produce the NCE results for comparison.
  - Files: `scripts/run_neurobe_experiments.py`, `notebooks/March-2025/neurobe_comparison_results.csv`
  - Do: (1) Write `scripts/run_neurobe_experiments.py` that iterates all 15 neurobe_binary problems, runs `FastGM` inference with neurobe_mode configs on CUDA, captures log_Z, num_trained, elapsed time per problem. Catches exceptions per-problem and logs failures without stopping. Writes results to CSV. (2) Run in background on GPU. (3) Collect results CSV when complete.
  - Verify: CSV exists with 15 rows. No problems failed. Each row has log_Z, NNs, and time values.
  - Done when: All 15 problems have completed inference and results are saved to CSV.

- [ ] **T03: Build combined comparison table and verify results** `est:30m`
  - Why: The deliverable is the side-by-side comparison (R038). This task parses NCE results alongside NeuroBE CSV into the final comparison format and verifies NN counts match.
  - Files: `scripts/build_comparison_table.py`, `notebooks/March-2025/neurobe_comparison_table.csv`
  - Do: (1) Write `scripts/build_comparison_table.py` that reads NCE results CSV and NeuroBE `binary_domain_results.csv`, joins on problem name, produces combined table with columns: Problem, NCE_log_Z, NeuroBE_log_Z, NCE_NNs, NeuroBE_NNs, NCE_time_hrs, NeuroBE_time_hrs. Prints formatted table to stdout. Saves as CSV. (2) Verify NN counts match for all 15 problems (assert NCE_NNs == NeuroBE_NNs). (3) Run existing test suite to confirm nothing broke.
  - Verify: `python scripts/build_comparison_table.py` prints table with all 15 problems. NN counts match. `pytest tests/` → 134+ passed.
  - Done when: Combined comparison table CSV exists, NN counts verified matching, table printed for human review.

## Files Likely Touched

- `nce/inference/graphical_model.py` (root-variable fix in `_load_from_uai`)
- `nce/benchmark_problems/neurobe_binary.py` (new — config builder module)
- `nce/benchmark_problems/__init__.py` (export new module)
- `scripts/verify_nn_counts.py` (new — NN count verification)
- `scripts/run_neurobe_experiments.py` (new — experiment runner)
- `scripts/build_comparison_table.py` (new — comparison table builder)
- `notebooks/March-2025/neurobe_comparison_results.csv` (new — raw NCE results)
- `notebooks/March-2025/neurobe_comparison_table.csv` (new — combined comparison)
