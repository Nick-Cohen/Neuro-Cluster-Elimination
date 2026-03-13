---
estimated_steps: 5
estimated_files: 1
---

# T01: Build shared fixtures and hand-built test problems

**Slice:** S01 — Inference & Training Test Suite
**Milestone:** M002

## Description

Create the shared fixture infrastructure that all subsequent test tasks depend on. This includes a complete `nn_training_config` fixture (derived from the existing `reference_flat_config` with CPU/small-epoch overrides), and three hand-built factor problem fixtures with analytically known partition function values: a binary chain (Z=2.0), a ternary chain (Z=4.5), and a star graph (produces wide enough buckets for NN training).

## Steps

1. Read existing `tests/conftest.py` to understand current fixture patterns and `reference_flat_config` field inventory.
2. Build `nn_training_config` fixture: copy all fields from `reference_flat_config`, override `device='cpu'`, `num_epochs=50`, `ecl=4`, `iB=2`, `hidden_sizes=[8,8]`, `sampling_scheme='all'`, `num_samples=256`, `set_size=None`, `dope_factors=False`, `seed=42`. Remove dead fields (`backward_ecl`, `num_batches_per_set`). Run `prepare_config()` on it to verify it's accepted without error.
3. Build `binary_chain_factors` fixture: two variables X0, X1 (domain 2), one pairwise factor with tensor `log10([[0.5, 0.5], [0.5, 0.5]])` (uniform). Analytic Z = 2.0, so `expected_log10_z = math.log10(2.0)`. Return dict with `factors`, `elim_order`, `expected_log10_z`.
4. Build `ternary_chain_factors` fixture: two variables X0, X1 (domain 3), one pairwise factor with known non-uniform values. Analytic Z = sum of all entries. Return dict with `factors`, `elim_order`, `expected_log10_z = math.log10(4.5)`.
5. Build `star_graph_factors` fixture: hub variable X0 (domain 2) connected to X1, X2, X3 (domain 2) via pairwise factors, plus a 3-way factor on (X0, X1, X2) to ensure bucket 0 has message scope ≥3 variables (message_size ≥ 8, exceeding ecl=4). Compute analytic Z by enumerating all 16 assignments. Return dict with `factors`, `elim_order`, `expected_log10_z`.

## Must-Haves

- [ ] `nn_training_config` fixture includes all fields needed by Trainer (no KeyError on any config access)
- [ ] `binary_chain_factors` fixture returns factors in log10 space with correct `expected_log10_z`
- [ ] `ternary_chain_factors` fixture returns domain-3 factors with correct `expected_log10_z`
- [ ] `star_graph_factors` fixture produces a bucket with message_size > ecl=4 (triggers NN path)
- [ ] All 110 existing tests still collected and passing

## Verification

- `source venv/bin/activate && python -m pytest tests/ --co -q | tail -1` — still shows 110 tests collected
- `source venv/bin/activate && python -c "from tests.conftest import *"` — no import errors
- Quick smoke test: instantiate each fixture's factors, build a FastGM, verify no crash

## Observability Impact

- Signals added/changed: None (fixtures are passive data providers)
- How a future agent inspects this: Read conftest.py to see all available fixtures and their docstrings
- Failure state exposed: pytest fixture errors will name the specific fixture that failed to construct

## Inputs

- `tests/conftest.py` — existing fixture patterns and `reference_flat_config` (42 fields)
- `nce/inference/factor.py` — `FastFactor` constructor API
- S01-RESEARCH.md — confirmed Z values (binary=2.0, ternary=4.5), star graph topology, config field requirements

## Expected Output

- `tests/conftest.py` — extended with 4 new fixtures: `nn_training_config`, `binary_chain_factors`, `ternary_chain_factors`, `star_graph_factors`
