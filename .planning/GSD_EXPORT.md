# GSD Planning Export — NCE Project

## Export Metadata

- **Export date:** 2026-03-12
- **GSD version:** 1.0
- **Project name:** NCE — Experiment Runner Framework
- **Export purpose:** Preserve all GSD 1.0 planning state for migration to GSD 2.0. This file is self-contained and captures every planning artifact needed to reconstruct the project's full planning context in a new GSD version.
- **Exported by:** quick task 24

---

## 1. Project Definition

Full contents of `.planning/PROJECT.md`:

```markdown
# Experiment Runner Framework

## What This Is

A streamlined experiment runner for backward message approximation experiments in probabilistic graphical models. Provides YAML-based configuration, automatic multi-GPU execution, auto-generated plots, and multi-run averaging — making it trivial to rerun experiments with small tweaks like changing problem instances or epoch counts.

## Core Value

Running a tweaked experiment should be as simple as editing a config file and executing one command — no hunting through code, no risk of misconfiguring loops or hyperparameters.

## Requirements

### Validated

- ✓ Neural network approximation of bucket messages — (`nce/neural_networks/`)
- ✓ Backward message computation for UKL loss — (`nce/utils/backward_message.py`)
- ✓ Multi-architecture support (linear, hidden layers)
- ✓ Error tracking during training — (`track_errors` in FastGM)
- ✓ Multi-GPU experiment distribution — v1.0 Phase 2
- ✓ JSON result output with per-bucket metrics — v1.0 Phase 3
- ✓ Plotting of results (local error, UKL loss, summaries) — v1.0 Phase 4
- ✓ YAML config-based experiment definition — v1.0 Phase 1
- ✓ Single command to run experiment — v1.0 Phase 1
- ✓ Auto-plotting after experiment completes — v1.0 Phase 4
- ✓ Multi-run support — v1.0 Phase 2
- ✓ Per-run output folders with results and plots — v1.0 Phase 3
- ✓ Averaged plots across runs — v1.0 Phase 4
- ✓ Symlog y-axis for local error plots — v1.0 Phase 4
- ✓ Timestamped experiment folders — v1.0 Phase 3

### Active

- [ ] Config restructure: flat dict → nested sections (nn, backward, sampling, training, etc.)
- [ ] Remove dead config items from old/dead code paths
- [ ] Clean up config field names for clarity
- [ ] Full config documentation guide (every field explained)
- [ ] Code comments enforcing config-doc sync
- [ ] FastGM picklable with full state (per-bucket training logs, NN weights, etc.)
- [ ] Comprehensive logging to configurable log file
- [ ] Standalone plotting functions that operate on FastGM objects
- [ ] Per-NN learning curve visualization
- [ ] Verify config restructure doesn't break existing experiments

### Out of Scope

- Interactive experiment monitoring/dashboard — adds complexity, not needed for research workflow
- Cloud/cluster job submission — local multi-GPU is sufficient
- Automatic hyperparameter search — manual control is preferred
- Real-time plot updates during training — post-hoc plotting is fine

## Current Milestone: v1.1 Config & Visualization

**Goal:** Make configs clean, documented, and well-organized; add rich state capture and built-in plotting that works directly on FastGM objects.

**Target features:**
- Config restructure: nested sections, dead field removal, name cleanup
- Config documentation: comprehensive guide + code-level sync enforcement
- FastGM pickling with full training state preserved
- Standalone plotting functions (learning curves, local errors, etc.)
- Comprehensive logging to configurable file paths
- Verification that restructure doesn't break existing experiments

## Context

**Research context:**
- Investigating whether improved backward message accuracy yields better partition function estimates
- Neural networks need long training times (30k+ epochs)
- Need to run same experiment multiple times for statistical validity

**Config pain points (v1.1 motivation):**
- 42-field flat config dict with dead items from old code paths
- NN config mixed with decision tree, backward message, sampling config
- Hard to read, edit, and understand what fields mean
- No documentation guide for config fields

**Visualization pain points (v1.1 motivation):**
- FastGM state lost after inference (buckets deleted post-elimination)
- No way to inspect per-NN learning curves after the fact
- Plotting requires separate scripts, not integrated with objects

## Constraints

- **Tech stack**: Python 3.11, PyTorch 2.0.1+cu117, existing NCE infrastructure — must integrate with current codebase
- **GPUs**: 0, 1, 2, 3 available for parallel execution
- **Compatibility**: Must work with existing UAI problem files and pyGMs catalog

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| YAML config over CLI flags | Config files are self-documenting, versionable, easier to tweak | — Pending |
| Adapt existing scripts vs rewrite | 765-line plotting script works, just needs integration | — Pending |
| Per-run folders + averaged folder | Need individual runs for debugging, means for publication | — Pending |

---
*Last updated: 2026-03-10 after milestone v1.1 start*
```

---

## 2. Requirements

Full contents of `.planning/REQUIREMENTS.md`:

```markdown
# Requirements: Experiment Runner Framework

**Defined:** 2026-02-21
**Core Value:** Running a tweaked experiment should be as simple as editing a config file and executing one command

## v1.0 Requirements (Complete)

### Configuration

- [x] **CFG-01**: User can define experiment via YAML config file
- [x] **CFG-02**: Config supports: problem, epochs, bw_ecl values, loss function, architectures, num_runs, gpus
- [x] **CFG-03**: Config file is saved with experiment results for reproducibility

### Execution

- [x] **EXC-01**: Single command runs entire experiment (`python run_experiment.py config.yaml`)
- [x] **EXC-02**: Experiments distribute across specified GPUs (0,1,2,3)
- [x] **EXC-03**: Multiple runs execute when num_runs > 1

### Output Organization

- [x] **OUT-01**: Each experiment gets timestamped folder
- [x] **OUT-02**: Each run gets separate subfolder with results and plots
- [x] **OUT-03**: Averaged results and plots go in `averaged/` subfolder
- [x] **OUT-04**: Summary JSON captures overall metrics

### Plotting

- [x] **PLT-01**: Plots auto-generate after experiment completes
- [x] **PLT-02**: Per-bucket local error plots with symlog y-axis
- [x] **PLT-03**: Per-bucket UKL loss plots
- [x] **PLT-04**: Summary plots (bar chart, heatmap, combined views)
- [x] **PLT-05**: Mean plots across runs when num_runs > 1

## v1.1 Requirements

Requirements for Config & Visualization milestone. Each maps to roadmap phases.

### Config Cleanup

- [ ] **CFG2-01**: Dead config fields from removed/unused code paths are identified and removed
- [ ] **CFG2-02**: Config restructured from flat dict to nested sections (nn, backward, sampling, training, inference, output)
- [ ] **CFG2-03**: Config field names cleaned up for clarity and consistency
- [ ] **CFG2-04**: Comprehensive config documentation guide written (every field explained with type, default, and purpose)
- [ ] **CFG2-05**: Code comments added at config definition sites enforcing doc-sync
- [ ] **CFG2-06**: Config validation updated to match new nested structure with clear error messages

### Visualization

- [ ] **VIZ-01**: FastGM object is picklable with full training state (per-bucket training logs, NN weights, loss histories)
- [ ] **VIZ-02**: Comprehensive logging system writes all training details to a configurable log file path
- [ ] **VIZ-03**: Standalone plotting functions accept FastGM objects (e.g. `plot_learning_curves(fastgm)`)
- [ ] **VIZ-04**: Per-NN learning curve visualization (loss over epochs for individual bucket NNs)
- [ ] **VIZ-05**: Comparison plotting functions accept multiple FastGM objects/logs and plot side-by-side comparisons

### Verification

- [ ] **VER-01**: Existing experiments produce identical results with restructured config (regression test)

## Future Requirements

### Enhanced Configuration

- **CFG-04**: Config inheritance (base config + overrides)
- **CFG-05**: Named experiment presets

### Monitoring

- **MON-01**: Progress output during long-running experiments
- **MON-02**: Notification when experiment completes

### Analysis

- **ANL-01**: Confidence intervals on mean plots
- **ANL-02**: Statistical significance tests between configurations

## Out of Scope

| Feature | Reason |
|---------|--------|
| Interactive dashboard | Adds complexity, post-hoc analysis is sufficient |
| Cloud/cluster submission | Local multi-GPU is sufficient for current needs |
| Hyperparameter search | Manual control preferred for research |
| Real-time plot updates | Post-hoc plotting is fine |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CFG-01 | Phase 1 (v1.0) | Complete |
| CFG-02 | Phase 1 (v1.0) | Complete |
| CFG-03 | Phase 2 (v1.0) | Complete |
| EXC-01 | Phase 1 (v1.0) | Complete |
| EXC-02 | Phase 2 (v1.0) | Complete |
| EXC-03 | Phase 2 (v1.0) | Complete |
| OUT-01 | Phase 3 (v1.0) | Complete |
| OUT-02 | Phase 3 (v1.0) | Complete |
| OUT-03 | Phase 3 (v1.0) | Complete |
| OUT-04 | Phase 3 (v1.0) | Complete |
| PLT-01 | Phase 4 (v1.0) | Complete |
| PLT-02 | Phase 4 (v1.0) | Complete |
| PLT-03 | Phase 4 (v1.0) | Complete |
| PLT-04 | Phase 4 (v1.0) | Complete |
| PLT-05 | Phase 4 (v1.0) | Complete |
| CFG2-01 | Phase 5 (v1.1) | Pending |
| CFG2-02 | Phase 5 (v1.1) | Pending |
| CFG2-03 | Phase 5 (v1.1) | Pending |
| CFG2-04 | Phase 6 (v1.1) | Pending |
| CFG2-05 | Phase 6 (v1.1) | Pending |
| CFG2-06 | Phase 5 (v1.1) | Pending |
| VIZ-01 | Phase 7 (v1.1) | Pending |
| VIZ-02 | Phase 7 (v1.1) | Pending |
| VIZ-03 | Phase 8 (v1.1) | Pending |
| VIZ-04 | Phase 8 (v1.1) | Pending |
| VIZ-05 | Phase 8 (v1.1) | Pending |
| VER-01 | Phase 9 (v1.1) | Pending |

**Coverage:**
- v1.0 requirements: 15 total (all complete)
- v1.1 requirements: 12 total
- Mapped to phases: 12 (100%)
- Unmapped: 0

---
*Requirements defined: 2026-02-21*
*Last updated: 2026-03-10 after v1.1 roadmap creation (phases 5-9)*
```

---

## 3. Roadmap

Full contents of `.planning/ROADMAP.md`:

```markdown
# Roadmap: Experiment Runner Framework

## Overview

Transform the existing hardcoded experiment pipeline into a YAML-configurable framework. Phase 1 establishes config parsing and entry point. Phase 2 adapts the existing multi-GPU runner for configurable execution with multi-run support. Phase 3 builds output organization (timestamped folders, per-run subfolders). Phase 4 integrates the existing 765-line plotting script with auto-generation and multi-run averaging.

Milestone v1.1 (phases 5-9) makes configs clean and documented, adds rich state capture to FastGM, and delivers standalone plotting functions that operate directly on FastGM objects.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3, 4): v1.0 milestone work (complete)
- Integer phases (5, 6, 7, 8, 9): v1.1 milestone work
- Decimal phases (e.g., 2.1): Urgent insertions if needed

### v1.0 Phases (Complete)

- [x] **Phase 1: Config + Entry Point** - YAML config parsing and single-command execution
- [x] **Phase 2: Execution Core** - Multi-GPU distribution and multi-run support
- [x] **Phase 3: Output Organization** - Timestamped folders, per-run structure, summary JSON
- [x] **Phase 4: Plotting Integration** - Auto-plotting, symlog axes, mean plots across runs

### v1.1 Phases

- [ ] **Phase 5: Config Restructure** - Flat dict replaced with nested sections, dead fields removed, names cleaned, validation updated
- [ ] **Phase 6: Config Documentation** - Comprehensive field guide written, code-level doc-sync comments added
- [ ] **Phase 7: FastGM State & Logging** - FastGM picklable with full training state, comprehensive log file output
- [ ] **Phase 8: Plotting Functions** - Standalone plot functions accept FastGM objects, per-NN learning curves
- [ ] **Phase 9: Verification** - Existing experiments produce identical results after config restructure

## Phase Details

### Phase 1: Config + Entry Point
**Goal**: User can define and run experiments via YAML config with a single command
**Depends on**: Nothing (first phase)
**Requirements**: CFG-01, CFG-02, EXC-01
**Success Criteria** (what must be TRUE):
  1. User can write a YAML config specifying problem, epochs, bw_ecl values, loss function, architectures, num_runs, and gpus
  2. User can run `python run_experiment.py config.yaml` and see experiment begin
  3. Config validation rejects invalid/missing required fields with clear error messages
**Plans**: 2 plans

Plans:
- [x] 01-01-PLAN.md — Config loading and validation module
- [x] 01-02-PLAN.md — Entry point script and example config

### Phase 2: Execution Core
**Goal**: Experiments distribute across GPUs and support multiple runs
**Depends on**: Phase 1
**Requirements**: EXC-02, EXC-03, CFG-03
**Success Criteria** (what must be TRUE):
  1. Experiments distribute across specified GPUs (e.g., gpus: [0,1,2,3] runs 4 parallel processes)
  2. When num_runs > 1, each run executes independently with different random seeds
  3. Config file is copied to experiment output folder for reproducibility
**Plans**: 2 plans

Plans:
- [x] 02-01-PLAN.md — Worker process for single experiment execution
- [x] 02-02-PLAN.md — GPU distribution runner and entry point integration

### Phase 3: Output Organization
**Goal**: Results organized in timestamped folders with per-run structure
**Depends on**: Phase 2
**Requirements**: OUT-01, OUT-02, OUT-03, OUT-04
**Success Criteria** (what must be TRUE):
  1. Each experiment creates timestamped folder (e.g., `2026-02-21_14-30_grid10x10_ukl/`)
  2. Each run has separate subfolder with results (`run_1/`, `run_2/`, etc.)
  3. Averaged results folder exists when num_runs > 1 (`averaged/`)
  4. Summary JSON in experiment root captures overall metrics (success/fail counts, timing)
**Plans**: 1 plan

Plans:
- [x] 03-01-PLAN.md — Descriptive folder naming, summary JSON, and result aggregation

### Phase 4: Plotting Integration
**Goal**: Plots auto-generate after experiment with symlog axes and multi-run averaging
**Depends on**: Phase 3
**Requirements**: PLT-01, PLT-02, PLT-03, PLT-04, PLT-05
**Success Criteria** (what must be TRUE):
  1. Plots generate automatically after experiment completes (no separate command needed)
  2. Local error plots use symlog y-axis showing 0, 10^1, 10^2, etc.
  3. UKL loss plots and summary plots (bar chart, heatmap) generate for each run
  4. When num_runs > 1, mean plots appear in `averaged/` folder
**Plans**: 2 plans

Plans:
- [x] 04-01-PLAN.md — Core plotting module with symlog and multi-run averaging
- [x] 04-02-PLAN.md — Integration into experiment runner

### Phase 5: Config Restructure
**Goal**: Config is organized into readable nested sections with dead fields removed and names clarified
**Depends on**: Phase 4
**Requirements**: CFG2-01, CFG2-02, CFG2-03, CFG2-06
**Success Criteria** (what must be TRUE):
  1. User can write a config with nested sections (nn, backward, sampling, training, inference, output) instead of a flat 42-field dict
  2. All dead config fields (from removed/unused code paths) are absent from new configs and the codebase raises an error if they appear
  3. Field names are consistent and self-describing (no abbreviations that require looking up)
  4. Config validation catches missing required fields and unexpected keys with messages that name the offending field and its section
**Plans**: 2 plans

Plans:
- [ ] 05-01-PLAN.md -- Config schema module with nested structure, validation, and flatten functions; FastGM integration
- [ ] 05-02-PLAN.md -- Nested config builders for benchmark sets; experiment_config.py integration

### Phase 6: Config Documentation
**Goal**: Every config field is documented in a written guide and enforced at the definition site in code
**Depends on**: Phase 5
**Requirements**: CFG2-04, CFG2-05
**Success Criteria** (what must be TRUE):
  1. A documentation guide exists listing every config field with its type, default value, and purpose in plain language
  2. Every field definition in the codebase has a comment linking to or reproducing its documentation entry
  3. Adding a new config field without updating the guide is detectable (e.g., code comment or assertion marking doc-sync requirement)
**Plans**: TBD

### Phase 7: FastGM State & Logging
**Goal**: FastGM preserves full training state after inference and writes comprehensive logs to a configurable file
**Depends on**: Phase 4
**Requirements**: VIZ-01, VIZ-02
**Success Criteria** (what must be TRUE):
  1. A FastGM object can be pickled to disk and unpickled, with all per-bucket training logs and NN weights intact
  2. Unpickling a FastGM and inspecting it yields the same per-bucket loss histories and trained weight counts as immediately after inference
  3. When a log file path is set in config, all training events (epoch, loss, bucket id) are written line-by-line to that file during inference
**Plans**: TBD

### Phase 8: Plotting Functions
**Goal**: Standalone functions generate learning curves, local error plots, and cross-experiment comparisons directly from FastGM objects
**Depends on**: Phase 7
**Requirements**: VIZ-03, VIZ-04, VIZ-05
**Success Criteria** (what must be TRUE):
  1. User can call `plot_learning_curves(fastgm)` after inference (or after unpickling) and receive per-NN loss-over-epoch plots without writing any extra code
  2. Each bucket's NN produces a distinct learning curve subplot showing its individual training trajectory
  3. Plotting functions work on a pickled-then-unpickled FastGM the same as on a live one
  4. Comparison functions accept multiple FastGM objects/logs and plot side-by-side comparisons (e.g. `compare_experiments([fastgm1, fastgm2], labels=[...])`)
**Plans**: TBD

### Phase 9: Verification
**Goal**: Config restructure is confirmed safe — existing experiments reproduce the same results with the new config format
**Depends on**: Phase 5, Phase 6
**Requirements**: VER-01
**Success Criteria** (what must be TRUE):
  1. A regression test script translates an existing v1.0-style flat config to the new nested format and runs inference on a reference problem
  2. The partition function estimate produced by the restructured config matches the estimate from the original config within numerical tolerance
  3. The test script can be run in one command and reports pass/fail clearly
**Plans**: TBD

## Progress

**Execution Order:**
v1.0: 1 -> 2 -> 3 -> 4 (complete)
v1.1: 5 -> 6 (config track), 7 -> 8 (viz track), then 9 (both tracks must complete first)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Config + Entry Point | 2/2 | Complete | 2026-02-21 |
| 2. Execution Core | 2/2 | Complete | 2026-02-21 |
| 3. Output Organization | 1/1 | Complete | 2026-02-21 |
| 4. Plotting Integration | 2/2 | Complete | 2026-02-22 |
| 5. Config Restructure | 0/2 | Planned | - |
| 6. Config Documentation | 0/? | Not started | - |
| 7. FastGM State & Logging | 0/? | Not started | - |
| 8. Plotting Functions | 0/? | Not started | - |
| 9. Verification | 0/? | Not started | - |

---
*Roadmap created: 2026-02-21*
*v1.1 phases added: 2026-03-10*
```

---

## 4. Project State

Full contents of `.planning/STATE.md`:

```markdown
# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-10)

**Core value:** Running a tweaked experiment should be as simple as editing a config file and executing one command
**Current focus:** Milestone v1.1 — Config & Visualization

## Current Position

Phase: Phase 5 (Config Restructure) — Not started
Plan: —
Status: Roadmap defined, ready to plan Phase 5
Last activity: 2026-03-10 — v1.1 roadmap created (phases 5-9)

Progress: [          ] 0% (v1.1)

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
| 030 | Create .uai.vo (SDBE format) and .uai.evid files for all 24 small_problems benchmark models | 2026-03-10 | 133a16b | [23-create-uai-uai-vo-and-uai-evid-files-for](./quick/23-create-uai-uai-vo-and-uai-evid-files-for/) |

## Session Continuity

Last session: 2026-03-10
Stopped at: v1.1 roadmap created — phases 5-9 defined, ready to plan Phase 5
Resume file: None

---
*State initialized: 2026-02-21*
*Updated: 2026-03-10 after v1.1 roadmap creation*
```

---

## 5. Planning Config

Full contents of `.planning/config.json`:

```json
{
  "workflow": {
    "research": false
  }
}
```

---

## 6. Phase Plans (Active)

Phase plans for phases 1-4 are complete and their PLAN.md files are no longer on disk (only SUMMARYs remain). Phase 5 is planned and has two plan files on disk.

### Phase 5: Config Restructure

#### Plan 05-01: Config schema module with nested structure, validation, and flatten functions; FastGM integration

```markdown
---
phase: 05-config-restructure
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - nce/config_schema.py
  - nce/inference/graphical_model.py
autonomous: true
requirements: [CFG2-01, CFG2-02, CFG2-03, CFG2-06]

must_haves:
  truths:
    - "Config dict supports nested sections (inference, nn, training, sampling, backward, output) as input format"
    - "All dead config fields (num_epochs2, loss_fn2, complexity_limit, exact, memorizer) raise errors if present"
    - "Field names are cleaned up (lr -> learning_rate, ecl -> exact_computation_limit, iB -> i_bound, fdb -> forward_diff_barrier)"
    - "A flatten_config() function converts nested config to the flat dict used internally by all consumer code"
    - "Config validation catches missing required fields and unexpected keys with section-specific error messages"
  artifacts:
    - path: "nce/config_schema.py"
      provides: "Nested config schema definition, flatten/validate functions, dead field detection"
      min_lines: 150
    - path: "nce/inference/graphical_model.py"
      provides: "FastGM.__init__ calls flatten_config() on input, internal code unchanged"
  key_links:
    - from: "nce/config_schema.py"
      to: "nce/inference/graphical_model.py"
      via: "FastGM.__init__ imports and calls flatten_config()"
      pattern: "from nce.config_schema import"
---

<objective>
Create the nested config schema module and integrate it as the entry point in FastGM.

Purpose: Replace the flat 42-field config dict with a nested, validated structure. The key insight is that ALL internal code continues to use the flat dict -- we add a translation layer at the entry point (FastGM.__init__) that accepts nested configs and flattens them for internal use. This means we can restructure configs WITHOUT touching any consumer code in bucket.py, train.py, etc.

Output: `nce/config_schema.py` module with schema definition, validation, flattening, and dead-field detection. Updated `FastGM.__init__` that auto-flattens nested configs.
</objective>

[Full plan body describes SECTIONS schema with 6 sections: inference, nn, training, sampling, backward, output.
Field name mappings: lr->learning_rate, ecl->exact_computation_limit, iB->i_bound, fdb->forward_diff_barrier, etc.
Two tasks: (1) Create config_schema.py, (2) Integrate into FastGM.__init__]
```

#### Plan 05-02: Nested config builders for benchmark sets; experiment_config.py integration

```markdown
---
phase: 05-config-restructure
plan: 02
type: execute
wave: 2
depends_on: [05-01]
files_modified:
  - nce/benchmark_problems/nbe_sanity_check.py
  - nce/benchmark_problems/small_problems.py
  - nce/benchmark_problems/__init__.py
  - notebooks/_1-2026/experiment_config.py
autonomous: true
requirements: [CFG2-02, CFG2-03, CFG2-06]

must_haves:
  truths:
    - "Benchmark configs offer a nested-format builder alongside the existing flat-format builder"
    - "experiment_config.py validate_config integrates with config_schema validation for nested configs"
    - "Users can write benchmark configs using new readable field names (learning_rate, i_bound, etc.)"
  artifacts:
    - path: "nce/benchmark_problems/nbe_sanity_check.py"
      provides: "Nested config builder alongside existing flat builder"
      min_lines: 100
    - path: "nce/benchmark_problems/small_problems.py"
      provides: "Nested config builder alongside existing flat builder"
      min_lines: 100
    - path: "notebooks/_1-2026/experiment_config.py"
      provides: "Updated validate_config with config_schema integration"
---

<objective>
Add nested config format to benchmark problem sets and integrate config_schema validation into the experiment runner.

Purpose: Users should be able to write configs using the new readable nested format. Benchmark configs serve as templates and documentation -- offering them in nested format demonstrates the new structure. The experiment runner's validate_config should leverage config_schema for nn_config validation.

Output: Updated benchmark configs with nested builders, updated experiment_config.py with config_schema integration.
</objective>

[Full plan body describes _build_nbe_configs_nested() and _build_default_configs_nested() functions,
validate_nn_section() in experiment_config.py, and backward compatibility requirements.]
```

---

## 7. Quick Task History

The complete Quick Tasks table is preserved verbatim in Section 4 (Project State) above.

### Quick Task Directory Manifest

All directories in `.planning/quick/` as of export date, excluding task 24 (this task):

| Directory | Has PLAN | Has SUMMARY | Notes |
|-----------|----------|-------------|-------|
| `1-fix-the-quantization-model-for-fast-exec` | yes | yes | Quick #008 |
| `2-create-benchmark-problems-module-with-ne` | yes | yes | Quick #009 |
| `3-add-optional-nn-config-sets-to-benchmark` | yes | yes | Quick #010 |
| `4-restructure-benchmark-problems-rename-to` | yes | yes | Quick #011 |
| `5-implement-nbe-num-samples-function-updat` | yes | yes | Quick #012 |
| `6-create-comprehensive-nbe-algorithm-evalu` | yes | yes | Quick #013 |
| `7-fix-nbe-configs-update-eval-plan-create-` | yes | yes | Quick #014 |
| `8-execute-the-nbe-evaluation-plan-from-doc` | yes | yes | Quick #015 |
| `9-change-ecl-from-2-ib-to-2-ib-1-in-nbe-co` | yes | yes | Quick #016 |
| `10-log-per-bucket-hidden-sizes-to-file-for-` | yes | yes | Quick #017 |
| `11-fix-hidden-sizes-script-capture-bucket-w` | yes | yes | Quick #018 |
| `12-run-full-nbe-experiment-on-all-5-problem` | yes | yes | Quick #020 |
| `13-design-benchmark-experiment-with-wmse-an` | yes | yes | Quick #019 |
| `14-apply-assumption-prevention-rules-to-cla` | no | no | Quick #021 — no artifacts on disk |
| `15-run-wmse-vs-ukl-benchmark-experiment-5-c` | yes | yes | Quick #022 |
| `16-follow-fixes-in-retrospective-nbe-experi` | yes | yes | Quick #023 |
| `17-write-experiment-execution-instructions-` | yes | yes | Quick #024 |
| `18-create-experiment-workflow-run-grid10x10` | yes | yes | Quick #025 |
| `19-fix-playground-py-hanging-on-fastgm-crea` | yes | yes | Quick #026 |
| `20-update-benchmark-wmse-ukl-graphs-with-pe` | yes | yes | Quick #027 |
| `21-add-per-problem-absolute-error-graphs-fo` | yes | yes | Quick #028 |
| `22-create-chart-comparing-sanity-check-neur` | yes | yes | Quick #029 |
| `23-create-uai-uai-vo-and-uai-evid-files-for` | yes | yes | Quick #030 |

**Note:** Quick tasks 001-007 (original tasks from 2026-02-23 to 2026-02-28) no longer have directories on disk. Their state is only preserved in STATE.md's Quick Tasks table above.

---

## 8. Key Decisions Registry

Decisions extracted from STATE.md Accumulated Context, grouped by category.

### Config & Validation Decisions

| Decision | Source |
|----------|--------|
| yaml.safe_load() for security (not yaml.load) | 01-01 |
| validate_config() returns new dict — never mutate input | 01-01 |
| Error messages to stderr with exit code 1 | 01-02 |
| Example configs in examples/ directory | 01-02 |
| Path resolution: absolute as-is, relative to config_dir, tilde expanded | 01-01 |
| Full data batch defaults: sampling_scheme='all', val_set='all' | 01-01 |
| Config dicts fully populated with all 42 fields from reference get_config() template | quick-4 |
| 'nbe,<value>' config string pattern for deferred computation based on bucket properties | quick-5 |
| Config updates: loss_fn='weighted_mse', skip_early_stopping=False, use_bw_approx=False for NeuroBE defaults | quick-5 |
| ecl=2^(iB-1) matching NeuroBE internals (not 2^iB) | quick-16 |

### Architecture & Data Structure Decisions

| Decision | Source |
|----------|--------|
| BenchmarkSet class consolidates problems + configs into single importable object | quick-4 |
| Benchmark config pairing: dual export (dict by key + ordered list) for flexible usage | quick-3 |
| Constructor auto-calls dope_factors() when config['dope_factors']=True, no manual call needed | quick-7 |
| matching_var() converts int label to Var before eliminate_variables(up_to=var) | quick-7 |
| get_log_partition_function() is the correct inference API, not run() | quick-7 |
| FactorNN.tensor is None (lazy representation); access labels/is_nn instead of tensor.shape | quick-8 |
| fastgm.buckets is empty after get_log_partition_function() — use fastgm.per_bucket_training_log | quick-18 |
| FastBucket.epochs_trained stores actual epochs run; FastBucket.trained_hidden_sizes available after compute_message_nn() | quick-12 |

### Algorithm Decisions

| Decision | Source |
|----------|--------|
| grid10x10 pre-elimination bucket width=4 (sparse); actual induced width during elimination is 10-21 | quick-10, quick-11 |
| grid10x10 induced width is 10-21; 33 NN-eligible buckets with ecl=512, iB=10 | quick-11 |
| custom_hidden_sizes callback called during compute_message_nn() for per-bucket width inspection | quick-11 |
| set_size must be clamped to num_samples when NBE adaptive sampling gives fewer samples | quick-8 |
| catalog Model uses model.num_vars not model.X | quick-8 |
| NeuroBE formula computes 48997 not 48999 for w=20,l=3,eps=0.1 — floating-point difference from doc table | quick-5 |

### Experiment & Runtime Decisions

| Decision | Source |
|----------|--------|
| run_id extracted from experiment dict, not CLI arg | 02-01 |
| torch imported inside main() to respect CUDA_VISIBLE_DEVICES | 02-01 |
| Output structure: arch_X/bw_ecl_Y/run_Z/ for unique experiment dirs | 02-02 |
| Wave-based execution: one experiment per GPU at a time | 02-02 |
| Per-GPU log files for debugging (gpu_N.log) | 02-02 |
| Folder name format: YYYY-MM-DD_HHMM_problem_loss | 03-01 |
| Skip aggregation for single-run experiments | 03-01 |
| duration_std only computed when num_runs > 1 | 03-01 |
| Averaged plots only generated when num_runs > 1 | 04-01 |
| Plot failures don't fail experiments (nice-to-have pattern) | 04-02 |
| Lazy import inside try block for fault isolation | 04-02 |
| 500-epoch NBE training exceeds 10min/problem even with 18+ CPU cores | quick-12 |

### Plotting Decisions

| Decision | Source |
|----------|--------|
| Symlog linthresh=1.0 default for local error plots | 04-01 |
| Confidence bands use alpha=0.3 for fill_between | 04-01 |

### Model Cache / File System Decisions

| Decision | Source |
|----------|--------|
| pyGMs catalog model.file expects files in subdirs (bn/, objdetect/) but cache root has flat files; need symlinks for offline access | quick-15 |
| grid10x10.f10.uai available offline at /home/cohenn1/UAI/Submissions/IBIA-PR-V2/test-results/1200/ | quick-18 |
| ecl=2^22 benchmark config results in num_trained=0 for ALL 5 sanity check models | quick-8 |

---

## 9. Migration Notes

### What Is Complete (v1.0)

Phases 1-4 are **complete**. Their PLAN.md files are **no longer on disk** — only SUMMARY.md files exist for completed phases. The SUMMARY files are located at:
- `.planning/phases/01-config-entry-point/01-01-SUMMARY.md`
- `.planning/phases/01-config-entry-point/01-02-SUMMARY.md`
- `.planning/phases/02-execution-core/02-01-SUMMARY.md`
- `.planning/phases/02-execution-core/02-02-SUMMARY.md`
- `.planning/phases/03-output-organization/03-01-SUMMARY.md`
- `.planning/phases/04-plotting-integration/04-01-SUMMARY.md`
- `.planning/phases/04-plotting-integration/04-02-SUMMARY.md`

### What Is Planned But Not Started (v1.1)

**Phase 5: Config Restructure** — PLAN files exist on disk at:
- `.planning/phases/05-config-restructure/05-01-PLAN.md`
- `.planning/phases/05-config-restructure/05-02-PLAN.md`

These are ready to execute. Phase 5 is the **next planned execution**.

**Phases 6-9** — No PLAN files. Plan counts are TBD (marked as "?" in roadmap). Plans need to be generated before execution.

### Requirements Status

- **v1.0 requirements (CFG-01 through PLT-05):** All 15 complete
- **v1.1 requirements:** All 12 pending (CFG2-01 through VER-01)

### Quick Tasks Without Directories on Disk

The following quick tasks from STATE.md have no corresponding directories in `.planning/quick/`:
- Quick #001 (001-run-minimal-experiment-on-grid10x10-f5-w)
- Quick #002 (002-fix-backward-ecl-bug-in-sample-generator)
- Quick #003 (003-test-plotting-functionality-and-document)
- Quick #005 (005-debug-bw-sensitivity-nan-bug)
- Quick #006 (6-write-usage-guide-for-running-12-4-exper)
- Quick #007 (7-disable-nbe-early-stopping-run-grid10x10)

These early tasks were completed before the quick task filing system was fully established. Their outcomes are captured in the Decisions registry above.

### Quick Task #14 Missing Files

Directory `14-apply-assumption-prevention-rules-to-cla` exists but has no PLAN.md or SUMMARY.md on disk. The task (Apply assumption-prevention rules to CLAUDE.md, gsd-executor.md, gsd-planner.md) was completed on 2026-03-06 with commit 9ee0d85.

### Files/Artifacts That May Need Recreation in GSD 2.0

1. **Phase plan files for phases 1-4**: PLAN.md files were deleted after phase completion. SUMMARY.md files remain. GSD 2.0 should be able to reconstruct context from SUMMARYs.
2. **Quick task directories 001-007**: No directories exist. STATE.md is the only record.
3. **config.json**: Currently minimal `{"workflow": {"research": false}}`. GSD 2.0 may need a different config structure.

### Project Ready-State for GSD 2.0 Resume

The project is in a clean, ready-to-resume state:
- **No blockers or pending todos**
- **Next action**: Execute Phase 5, Plan 01 (`05-01-PLAN.md` already written and ready)
- **Current milestone**: v1.1 Config & Visualization, 0% complete
- **v1.0 milestone**: 100% complete (all 7 plans executed, all 15 requirements met)

---

*GSD Export created: 2026-03-12*
*Covers: GSD 1.0 planning state as of 2026-03-12*
*Total quick tasks documented: 30 (001-030)*
*Total phase plans documented: 9 plans across phases 1-5*
