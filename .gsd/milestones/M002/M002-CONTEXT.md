# M002: Test Suite — Context

**Gathered:** 2026-03-12
**Status:** Waiting for M001 completion

## Project Description

NCE is a Python package for neural network-based approximate inference on probabilistic graphical models. After M001 delivers config restructure and visualization, M002 builds a formal test suite covering correctness, functional, convergence, and robustness testing.

## Why This Milestone

No formal test suite exists. Testing is done through ad-hoc notebooks and one-off scripts in `notebooks/`. This makes it easy for regressions to slip in unnoticed, and common failure modes (infinity outputs, all-zero targets, domain size edge cases) aren't systematically checked.

## User-Visible Outcome

### When this milestone is complete, the user can:

- Run `pytest tests/` and see all core functionality validated
- Add a new failure-mode regression test by following an established pattern
- Trust that exact inference, single-bucket training, multi-domain variables, convergence, and edge-case robustness are all covered
- Catch regressions before they reach experiment runs

### Entry point / environment

- Entry point: `pytest tests/` from project root
- Environment: local dev with CUDA GPUs
- Live dependencies involved: none

## Completion Class

- Contract complete means: pytest suite passes, covers all R018–R024 requirements
- Integration complete means: tests exercise real inference and training paths, not mocks
- Operational complete means: none

## Final Integrated Acceptance

To call this milestone complete, we must prove:

- `pytest tests/` passes with all tests green
- At least one test per requirement R018–R024 exists and exercises real code paths
- Adding a new failure-mode test requires only creating a new test function following the established pattern

## Risks and Unknowns

- **Test runtime** — real inference and training tests may be slow. Need to balance coverage with speed (small problems, few epochs).
- **GPU dependency** — some tests require CUDA. Need skip markers for CPU-only environments.
- **M001 dependency** — test suite should cover both old flat and new nested config paths from M001.

## Existing Codebase / Prior Art

- `notebooks/2025-07/test_new_loss_NN.ipynb` — ad-hoc loss function testing
- `notebooks/Older/test_loss_fns.py` — informal loss testing scripts
- No pytest infrastructure, no conftest.py, no test directory

> See `.gsd/DECISIONS.md` for all architectural and pattern decisions.

## Relevant Requirements

- R018: Exact inference correctness test
- R019: Single bucket training functional test
- R020: Domain size ≥3 variable handling test
- R021: Convergence test (loss decreases over ~50 epochs)
- R022: No-infinity output robustness test
- R023: All-neg-inf / all-zero target handling test
- R024: Extensible failure-mode regression pattern

## Scope

### In Scope

- pytest-based test suite in `tests/` directory
- Correctness, functional, convergence, and robustness tests
- Extensible pattern for adding failure-mode regression tests
- GPU skip markers for CPU-only environments

### Out of Scope / Non-Goals

- CI/CD integration (local-only for now)
- Performance/benchmark tests
- UI or plotting tests (visual verification is manual)

## Technical Constraints

- Must work with M001's config restructure (test both flat and nested configs)
- Tests should run in reasonable time (< 5 minutes total)
- Small reference problems for fast execution

## Open Questions

- Exact test problem selection — need small problems where exact partition function is known/computable quickly
- Whether to test config_schema.py validation separately or only through integration
