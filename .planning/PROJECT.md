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
