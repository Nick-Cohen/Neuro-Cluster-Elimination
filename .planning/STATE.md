# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-21)

**Core value:** Running a tweaked experiment should be as simple as editing a config file and executing one command
**Current focus:** Phase 4 - Plotting Integration

## Current Position

Phase: 4 of 4 (Plotting Integration)
Plan: 2 of 2 in current phase
Status: Milestone complete
Last activity: 2026-03-04 - Completed quick task 012: implement NeuroBE num_samples function

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

## Session Continuity

Last session: 2026-03-04 00:52
Stopped at: Completed quick task 5 (implement NeuroBE num_samples function)
Resume file: None

---
*State initialized: 2026-02-21*
