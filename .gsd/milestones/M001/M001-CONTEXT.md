# M001: Config & Visualization — Context

**Gathered:** 2026-03-12
**Status:** Ready for planning

## Project Description

NCE is a Python package for neural network-based approximate inference on probabilistic graphical models. v1.0 delivered a working experiment runner framework. v1.1 (this milestone) restructures configs for readability, adds state preservation and visualization, and verifies nothing breaks.

## Why This Milestone

The 42-field flat config dict is the primary user interface for experiments, and it's a mess — dead fields from removed code paths, cryptic abbreviations (ecl, iB, fdb, lr), no documentation. Researchers waste time guessing what fields mean and which ones are still active.

FastGM state is lost after inference because buckets are deleted during elimination. There's no way to inspect per-NN learning curves, compare experiments, or debug training issues after the fact without re-running.

## User-Visible Outcome

### When this milestone is complete, the user can:

- Write experiment configs using readable nested sections instead of a flat 42-field dict
- Continue using old flat configs without any changes (backward compat)
- Pickle a FastGM after inference and inspect per-bucket training metadata later
- Optionally save full NN weights and convert outputs back to original scale
- Call `plot_learning_curves(fastgm)` on a saved FastGM and get per-NN loss plots
- Compare two experiments side-by-side with a single function call
- Find structured training logs in a configurable log file
- Read a documentation guide explaining every config field
- Run a one-command regression test proving the restructure didn't change inference results

### Entry point / environment

- Entry point: Python API (`FastGM.__init__`, plotting functions, pickle load/save)
- Environment: local dev with CUDA GPUs
- Live dependencies involved: none (PyTorch, matplotlib, pickle — all local)

## Completion Class

- Contract complete means: config_schema validates and flattens correctly, pickle round-trips preserve metadata, plotting functions produce figures
- Integration complete means: FastGM.__init__ accepts both nested and flat configs and produces identical inference results
- Operational complete means: none (no services or daemons)

## Final Integrated Acceptance

To call this milestone complete, we must prove:

- A nested config produces the same partition function estimate as the equivalent flat config on a reference problem
- A FastGM can be pickled after inference, unpickled in a fresh session, and its training metadata inspected and plotted
- All existing benchmark sets work with both config formats

## Risks and Unknowns

- **Pickle compatibility with PyTorch objects** — NN state dicts, data preprocessors, and pyGMs Var objects may have pickle edge cases. Retire in S04.
- **Config field inventory completeness** — the actual set of live fields may differ from what the old plans documented. Must be audited from code, not assumed from prior docs.
- **Backward compat edge cases** — some flat configs may use fields that overlap with nested section names. Detection logic must be robust.

## Existing Codebase / Prior Art

- `nce/inference/graphical_model.py` (1425 lines) — FastGM class, reads config via `self.config['key']` and `self.config.get('key', default)` throughout
- `nce/inference/bucket.py` (1219 lines) — FastBucket, reads config similarly, stores training metadata before deletion
- `nce/neural_networks/train.py` (1537 lines) — Trainer class, stores `self.losses` and `self.val_losses` as list of (epoch, value) tuples
- `nce/inference/factor_nn.py` — FactorNN, stores `self.net`, `self.data_processor`, `self.losses`
- `nce/benchmark_problems/nbe_sanity_check.py` — BenchmarkSet with `_build_nbe_configs()` producing flat 42-field dicts
- `nce/benchmark_problems/small_problems.py` — Similar pattern, 24 models with flat configs
- `notebooks/_1-2026/experiment_config.py` (154 lines) — YAML config loading and validation for experiment runner
- `nce/utils/plots.py` (178 lines) — Existing plotting utilities (factor comparison, not learning curves)
- `.planning/phases/05-config-restructure/` — Prior detailed plans for config schema (05-01, 05-02) — useful reference but written for old GSD system

> See `.gsd/DECISIONS.md` for all architectural and pattern decisions — it is an append-only register; read it during planning, append to it during execution.

## Relevant Requirements

- R001–R006: Config restructure (S01, S02)
- R007–R008: Config documentation (S03)
- R009–R012: State preservation (S04)
- R013–R015: Visualization (S05)
- R016: Logging (S06)
- R017: Regression verification (S07)

## Scope

### In Scope

- Nested config schema with 6 sections (inference, nn, training, sampling, backward, output)
- Flat→nested auto-detection and backward compat
- Dead field identification and removal (error on use)
- Field name cleanup with old-name aliases
- Config validation with section-specific error messages
- Config documentation guide
- FastGM pickle with training metadata (default) and optional NN weights
- Undo-normalization function accessible from saved state
- New `nce/visualization/` module with learning curves and comparison plots
- Structured logging to configurable log file
- Regression test script

### Out of Scope / Non-Goals

- Changing any internal inference logic (bucket elimination, factor operations, training loops)
- Config inheritance or presets (deferred to R025/R026)
- Interactive dashboards or real-time plotting
- Modifying the experiment runner's execution logic (multi-GPU, wave-based, etc.)

## Technical Constraints

- Python 3.11, PyTorch 2.0.1+cu117, pyGMs
- All internal code must continue using flat config dict — no consumer code changes
- GPUs 0, 1, 2, 3 available for testing
- Must work with existing UAI problem files and pyGMs catalog

## Integration Points

- `FastGM.__init__` — single integration point for config_schema; calls `prepare_config()` (or similar) and gets back a flat dict
- `BenchmarkSet` pattern — benchmark configs must offer nested builders alongside existing flat builders
- `experiment_config.py` — YAML validation should leverage config_schema for nn_config validation

## Open Questions

- Exact list of dead config fields — must be audited from code during S01 execution, not assumed from prior plans
- Whether `per_bucket_training_log` captures enough metadata or needs extension for full learning curve data (currently only label, epochs_trained, hidden_sizes — loss curves live in FactorNN.losses which gets consumed)
