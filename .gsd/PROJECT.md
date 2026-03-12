# Project

## What This Is

NCE is a Python package implementing neural network-based inference for graphical models. It combines variable elimination with neural network approximations to perform approximate inference on probabilistic graphical models (PGMs), particularly using Weighted Mini-Bucket elimination (WMB). An experiment runner framework (v1.0) provides YAML-based configuration, automatic multi-GPU execution, auto-generated plots, and multi-run averaging.

## Core Value

Running a tweaked experiment should be as simple as editing a config file and executing one command — no hunting through code, no risk of misconfiguring loops or hyperparameters.

## Current State

- **v1.0 complete:** YAML config parsing, multi-GPU distribution, timestamped output folders, auto-plotting with symlog axes and multi-run averaging — all 15 v1.0 requirements validated.
- **30 quick tasks completed:** Bug fixes, benchmark experiments (WMSE vs UKL across 24 problems), algorithm evaluation, plotting tools, model cache setup.
- **Codebase:** Three-layer architecture (inference, neural_networks, sampling) in `nce/`. Experiment runner in `notebooks/_1-2026/`. Benchmark problem sets in `nce/benchmark_problems/`.
- **Config pain point:** 42-field flat config dict with dead items, cryptic abbreviations, no documentation.
- **Visualization pain point:** FastGM state lost after inference (buckets deleted), no way to inspect per-NN learning curves after the fact.
- **No formal test suite.** Testing done through ad-hoc notebooks and scripts.

## Architecture / Key Patterns

- **Three-layer design:** `nce/inference/` (graphical model ops), `nce/neural_networks/` (training/loss), `nce/sampling/` (data generation)
- **Log-space operations:** All factor operations in log-space (`__mul__` = addition, `eliminate()` = log-sum-exp)
- **Config as flat dict:** `nn_config` passed to `FastGM.__init__`, copied to `self.config`, fields read via `self.config['key']` or `self.config.get('key', default)` throughout
- **BenchmarkSet pattern:** `BenchmarkSet(problems, configs)` — problems list + configs dict of config lists
- **Device management:** Factors and buckets must be on consistent devices (CUDA/CPU)
- **Bucket lifecycle:** Buckets created during init, consumed during elimination, deleted after — training metadata must be captured before deletion

## Capability Contract

See `.gsd/REQUIREMENTS.md` for the explicit capability contract, requirement status, and coverage mapping.

## Milestone Sequence

- [ ] M001: Config & Visualization — Clean configs, state preservation, standalone plotting, regression verification
- [ ] M002: Test Suite — Correctness, functional, convergence, and robustness tests with extensible failure-mode pattern
