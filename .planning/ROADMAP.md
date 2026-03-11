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
**Plans**: TBD

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
| 5. Config Restructure | 0/? | Not started | - |
| 6. Config Documentation | 0/? | Not started | - |
| 7. FastGM State & Logging | 0/? | Not started | - |
| 8. Plotting Functions | 0/? | Not started | - |
| 9. Verification | 0/? | Not started | - |

---
*Roadmap created: 2026-02-21*
*v1.1 phases added: 2026-03-10*
