# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-21)

**Core value:** Running a tweaked experiment should be as simple as editing a config file and executing one command
**Current focus:** Phase 4 - Plotting Integration

## Current Position

Phase: 4 of 4 (Plotting Integration)
Plan: 2 of 2 in current phase
Status: Milestone complete
Last activity: 2026-03-10 - Completed quick task 21: Add no_exact_bw graphs for 9 pattern-1 problems

Progress: [##########] 100%

## Performance Metrics

**Velocity:**
- Total plans completed: 7
- Average duration: 3.3 min
- Total execution time: 24 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-config-entry-point | 2 | 4 min | 2 min |
| 02-execution-core | 2 | 8 min | 4 min |
| 03-output-organization | 1 | 5 min | 5 min |
| 04-plotting-integration | 2 | 4 min | 2 min |

**Recent Trend:**
- Last 5 plans: 02-02 (5 min), 03-01 (5 min), 04-01 (3 min), 04-02 (1 min)
- Trend: stable

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- yaml.safe_load() for security (01-01)
- Full data batch defaults: sampling_scheme='all', val_set='all' (01-01)
- Path resolution: absolute as-is, relative to config_dir, tilde expanded (01-01)
- validate_config() returns new dict (never mutate input) (01-01)
- Error messages to stderr with exit code 1 (01-02)
- Example configs in examples/ directory (01-02)
- run_id extracted from experiment dict, not CLI arg (02-01)
- torch imported inside main() to respect CUDA_VISIBLE_DEVICES (02-01)
- Output structure: arch_X/bw_ecl_Y/run_Z/ for unique experiment dirs (02-02)
- Wave-based execution: one experiment per GPU at a time (02-02)
- Per-GPU log files for debugging (gpu_N.log) (02-02)
- Folder name format: YYYY-MM-DD_HHMM_problem_loss (03-01)
- Skip aggregation for single-run experiments (03-01)
- duration_std only computed when num_runs > 1 (03-01)
- Symlog linthresh=1.0 default for local error plots (04-01)
- Confidence bands use alpha=0.3 for fill_between (04-01)
- Averaged plots only generated when num_runs > 1 (04-01)
- Lazy import inside try block for fault isolation (04-02)
- Plot failures don't fail experiments (nice-to-have pattern) (04-02)
- Benchmark config pairing: dual export (dict by key + ordered list) for flexible usage (quick-3)
- BenchmarkSet class consolidates problems + configs into single importable object (quick-4)
- Config dicts fully populated with all 42 fields from reference get_config() template (quick-4)
- NeuroBE formula computes 48997 not 48999 for w=20,l=3,eps=0.1 - floating-point difference from doc table (quick-5)
- 'nbe,<value>' config string pattern for deferred computation based on bucket properties (quick-5)
- Config updates: loss_fn='weighted_mse', skip_early_stopping=False, use_bw_approx=False for NeuroBE defaults (quick-5)
- Constructor auto-calls dope_factors() when config['dope_factors']=True, no manual call needed (quick-7)
- matching_var() converts int label to Var before eliminate_variables(up_to=var) (quick-7)
- get_log_partition_function() is the correct inference API, not run() (quick-7)
- ecl=2^22 benchmark config results in num_trained=0 for ALL 5 sanity check models (quick-8)
- set_size must be clamped to num_samples when NBE adaptive sampling gives fewer samples than set_size (quick-8)
- FactorNN.tensor is None (lazy representation); access labels/is_nn instead of tensor.shape (quick-8)
- catalog Model uses model.num_vars not model.X (quick-8)
- grid10x10 pre-elimination bucket width=4 (sparse original factors); actual induced width during elimination is 10-21 (quick-10, corrected quick-11)
- custom_hidden_sizes callback is called during compute_message_nn() when bucket has full induced-width scope; use this for any per-bucket width inspection (quick-11)
- grid10x10 induced width is 10-21; 33 NN-eligible buckets with ecl=512, iB=10 (quick-11)
- FastBucket.epochs_trained stores actual epochs run (t.losses[-1][0]+1); FastBucket.trained_hidden_sizes stores resolved hidden sizes, both available after compute_message_nn() (quick-12)
- 500-epoch NBE training on nbe_sanity_check problems exceeds 10min/problem even with 18+ CPU cores; single NN bucket training can take 30+ min CPU (quick-12)
- pyGMs catalog model.file expects files in subdirs (bn/, objdetect/) but cache root has flat files; need symlinks for offline access (quick-15)
- fastgm.buckets is empty after get_log_partition_function() -- buckets deleted post-elimination; use fastgm.per_bucket_training_log (now populated by bucket.py during compute_message_nn) (quick-18)
- grid10x10.f10.uai available offline at /home/cohenn1/UAI/Submissions/IBIA-PR-V2/test-results/1200/; copy to .model_cache/grids/; generate .ord via pyGMs.eliminationOrder('minfill') (quick-18)

### Pending Todos

None.

### Blockers/Concerns

None.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 001 | Run minimal experiment on grid10x10.f5.wrap with pyGMs catalog loading | 2026-02-23 | 0a296bc | [001-run-minimal-experiment-on-grid10x10-f5-w](./quick/001-run-minimal-experiment-on-grid10x10-f5-w/) |
| 002 | Fix backward ECL bug in _get_values() for partial overlap | 2026-02-23 | 2bd098c | [002-fix-backward-ecl-bug-in-sample-generator](./quick/002-fix-backward-ecl-bug-in-sample-generator/) |
| 003 | Test plotting functionality and document usage | 2026-02-23 | - | [003-test-plotting-functionality-and-document](./quick/003-test-plotting-functionality-and-document/) |
| 005 | Fix WMB in-place tensor modification corrupting shared factors | 2026-02-23 | d3a2a5e | [005-debug-bw-sensitivity-nan-bug](./quick/005-debug-bw-sensitivity-nan-bug/) |
| 006 | Write usage guide for running probs12_4 experiments | 2026-02-28 | 57acf5a | [6-write-usage-guide-for-running-12-4-exper](./quick/6-write-usage-guide-for-running-12-4-exper/) |
| 007 | Disable nbe_early_stopping, launch grid10x10 + probs12_4 benchmarks | 2026-02-28 | 8f4f839 | [7-disable-nbe-early-stopping-run-grid10x10](./quick/7-disable-nbe-early-stopping-run-grid10x10/) |
| 008 | Fix quantization model: rewrite QuantizationSolver with recursive binary splitting | 2026-03-03 | 41072c8 | [1-fix-the-quantization-model-for-fast-exec](./quick/1-fix-the-quantization-model-for-fast-exec/) |
| 009 | Create benchmark_problems module with neuro_be_sanity_check set | 2026-03-03 | a7ff31a | [2-create-benchmark-problems-module-with-ne](./quick/2-create-benchmark-problems-module-with-ne/) |
| 010 | Add optional neuroBE config sets to benchmark_problems module | 2026-03-03 | 6a95ebf | [3-add-optional-nn-config-sets-to-benchmark](./quick/3-add-optional-nn-config-sets-to-benchmark/) |
| 011 | Restructure benchmark_problems: BenchmarkSet class, rename to nbe_sanity_check | 2026-03-03 | 8fb968a | [4-restructure-benchmark-problems-rename-to](./quick/4-restructure-benchmark-problems-rename-to/) |
| 012 | Implement NeuroBE num_samples function, update configs, add grid10x10 | 2026-03-04 | 9412ad8 | [5-implement-nbe-num-samples-function-updat](./quick/5-implement-nbe-num-samples-function-updat/) |
| 013 | Create comprehensive NBE algorithm evaluation plan | 2026-03-05 | - | [6-create-comprehensive-nbe-algorithm-evalu](./quick/6-create-comprehensive-nbe-algorithm-evalu/) |
| 014 | Fix NBE configs, update eval plan, create pre-smoke scripts | 2026-03-05 | 5722360 | [7-fix-nbe-configs-update-eval-plan-create-](./quick/7-fix-nbe-configs-update-eval-plan-create-/) |
| 015 | Execute NBE evaluation plan (Phases 0a-1), fix 4 config bugs | 2026-03-05 | ea208ef | [8-execute-the-nbe-evaluation-plan-from-doc](./quick/8-execute-the-nbe-evaluation-plan-from-doc/) |
| 016 | Change ecl from 2^iB to 2^(iB-1) matching NeuroBE internals | 2026-03-05 | 59d5dc2 | [9-change-ecl-from-2-ib-to-2-ib-1-in-nbe-co](./quick/9-change-ecl-from-2-ib-to-2-ib-1-in-nbe-co/) |
| 015 | Execute NBE evaluation plan Phases 0a-3, fix set_size/model.X bugs | 2026-03-05 | fb93b9f | [8-execute-the-nbe-evaluation-plan-from-doc](./quick/8-execute-the-nbe-evaluation-plan-from-doc/) |
| 017 | Log per-bucket hidden sizes to file for grid10x10.f5.wrap with NBE config | 2026-03-05 | c3fe0a3 | [10-log-per-bucket-hidden-sizes-to-file-for-](./quick/10-log-per-bucket-hidden-sizes-to-file-for-/) |
| 018 | Fix hidden sizes script: capture bucket widths during elimination (induced width 10-21, 33 NN buckets) | 2026-03-05 | 1e1e8b7 | [11-fix-hidden-sizes-script-capture-bucket-w](./quick/11-fix-hidden-sizes-script-capture-bucket-w/) |
| 019 | Design benchmark experiment: WMSE vs UKL across 24 small_problems (5 configs, 120 experiments) | 2026-03-05 | 539d67a | [13-design-benchmark-experiment-with-wmse-an](./quick/13-design-benchmark-experiment-with-wmse-an/) |
| 020 | Run full NBE experiment (500 epochs) on all 5 problems, log epochs-to-early-stopping per bucket | 2026-03-06 | cca2941 | [12-run-full-nbe-experiment-on-all-5-problem](./quick/12-run-full-nbe-experiment-on-all-5-problem/) |
| 021 | Apply assumption-prevention rules to CLAUDE.md, gsd-executor.md, gsd-planner.md | 2026-03-06 | 9ee0d85 | [14-apply-assumption-prevention-rules-to-cla](./quick/14-apply-assumption-prevention-rules-to-cla/) |
| 022 | Run WMSE vs UKL benchmark: 120 experiments (5 configs x 24 problems x 5000 epochs) across 4 GPUs | 2026-03-06 | 0f2bd32 | [15-run-wmse-vs-ukl-benchmark-experiment-5-c](./quick/15-run-wmse-vs-ukl-benchmark-experiment-5-c/) |
| 023 | Codify NBE retrospective lessons in CLAUDE.md (Config Fidelity, Pre-Flight, Zombie, Algorithm Literacy) | 2026-03-07 | 310cbf3 | [16-follow-fixes-in-retrospective-nbe-experi](./quick/16-follow-fixes-in-retrospective-nbe-experi/) |
| 024 | Write comprehensive experiment execution guide (docs/experiment_execution_guide.md) | 2026-03-09 | 93dde3a | [17-write-experiment-execution-instructions-](./quick/17-write-experiment-execution-instructions-/) |
| 025 | Create grid10x10.f10 UKL experiment workflow: runner, analysis, CSV/pickle outputs; fix per-bucket data collection bug | 2026-03-09 | 8cc2401 | [18-create-experiment-workflow-run-grid10x10](./quick/18-create-experiment-workflow-run-grid10x10/) |
| 026 | Document pyGMs catalog hang root cause and model cache setup guide (docs/model_cache_setup.md) | 2026-03-09 | e736006 | [19-fix-playground-py-hanging-on-fastgm-crea](./quick/19-fix-playground-py-hanging-on-fastgm-crea/) |
| 027 | Create visualize_updated.py: per-problem abs error bar charts (linear+symlog) and 96-row summary CSV for WMSE vs UKL benchmark | 2026-03-10 | 47754d9 | [20-update-benchmark-wmse-ukl-graphs-with-pe](./quick/20-update-benchmark-wmse-ukl-graphs-with-pe/) |
| 028 | Add no_exact_bw graphs for 9 pattern-1 problems (4 configs, ukl_bw30 OOM) | 2026-03-10 | bcee5a1 | [21-add-per-problem-absolute-error-graphs-fo](./quick/21-add-per-problem-absolute-error-graphs-fo/) |
| 029 | Create grouped bar chart comparing paper WMB/NeuroBE errors vs our WMB for 4 overlapping problems | 2026-03-10 | 5453c6f | [22-create-chart-comparing-sanity-check-neur](./quick/22-create-chart-comparing-sanity-check-neur/) |

## Session Continuity

Last session: 2026-03-10
Stopped at: Completed quick task 22 - Created compare_paper_vs_sanity_check.py and PNG chart comparing paper vs our WMB results
Resume file: None

---
*State initialized: 2026-02-21*
