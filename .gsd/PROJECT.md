# Project

## What This Is

NCE is a Python package implementing neural network-based inference for graphical models. It combines variable elimination with neural network approximations to perform approximate inference on probabilistic graphical models (PGMs), particularly using Weighted Mini-Bucket elimination (WMB). An experiment runner framework (v1.0) provides YAML-based configuration, automatic multi-GPU execution, auto-generated plots, and multi-run averaging.

## Core Value

Running a tweaked experiment should be as simple as editing a config file and executing one command — no hunting through code, no risk of misconfiguring loops or hyperparameters.

## Current State

- **v1.0 complete:** YAML config parsing, multi-GPU distribution, timestamped output folders, auto-plotting with symlog axes and multi-run averaging — all 15 v1.0 requirements validated.
- **M001 complete:** Config restructured into 6 nested sections with validation, backward compat for flat configs, state preservation with loss curves, visualization module, structured JSONL logging, and regression test — all 17 requirements validated.
- **M002 complete:** pytest-based test suite covering exact inference correctness, NN training, convergence, and loss function robustness — 125 tests pass in ~30s, all 24 requirements validated.
- **30 quick tasks completed:** Bug fixes, benchmark experiments (WMSE vs UKL across 24 problems), algorithm evaluation, plotting tools, model cache setup.
- **Codebase:** Three-layer architecture (inference, neural_networks, sampling) in `nce/`. Experiment runner in `notebooks/_1-2026/`. Benchmark problem sets in `nce/benchmark_problems/`. Test suite in `tests/`.

## Architecture / Key Patterns

- **Three-layer design:** `nce/inference/` (graphical model ops), `nce/neural_networks/` (training/loss), `nce/sampling/` (data generation)
- **Log-space operations:** All factor operations in log-space (`__mul__` = addition, `eliminate()` = log-sum-exp)
- **Config entry point:** `prepare_config()` in `nce/config_schema.py` auto-detects flat/nested input, validates, resolves aliases, and returns a flat dict. `FastGM.__init__` calls this as its single config integration point.
- **Config formats:** Nested (recommended, 6 sections: inference, nn, training, sampling, backward, output) and flat (legacy, auto-detected). Both produce identical flat dicts internally.
- **BenchmarkSet pattern:** `BenchmarkSet(problems, configs)` — problems list + configs dict of config lists. Both `nbe` (flat) and `nbe_nested` config builders available.
- **State preservation:** `nce/state/` module — `save_state(fastgm, path)` extracts per-bucket training logs (loss curves, epochs, hidden sizes) and optional NN weights. `load_state(path)` returns a plain dict for inspection.
- **Visualization:** `nce/visualization/` module — `plot_learning_curves()` and `compare_experiments()` accept live FastGM or loaded state dicts.
- **Training logging:** `nce/training_logger.py` — JSONL events via `nce.training` logger namespace (isolated from root logger suppression).
- **Device management:** Factors and buckets must be on consistent devices (CUDA/CPU)
- **Bucket lifecycle:** Buckets created during init, consumed during elimination, deleted after — training metadata captured into `per_bucket_training_log` before deletion
- **Test suite:** `tests/` with pytest — hand-built factor fixtures (binary chain, ternary chain, star graph) with known partition functions, complete NN training config fixture, parametrized loss function robustness tests. `tests/PATTERN.md` documents how to add regression tests.

## Capability Contract

See `.gsd/REQUIREMENTS.md` for the explicit capability contract, requirement status, and coverage mapping.

## Milestone Sequence

- [x] M001: Config & Visualization — Clean configs, state preservation, standalone plotting, regression verification
- [x] M002: Test Suite — Correctness, functional, convergence, and robustness tests with extensible failure-mode pattern
- [ ] M003: NeuroBE Reproduction Mode — Faithful reproduction of NeuroBE algorithm in NCE for direct comparison on 15 binary-domain problems
