---
id: T01
parent: S01
milestone: M002
provides:
  - nn_training_config fixture (40-field validated config for CPU NN training)
  - binary_chain_factors fixture (Z=2.0, domain 2)
  - ternary_chain_factors fixture (Z=4.5, domain 3)
  - star_graph_factors fixture (message_size=8 > ecl=4, triggers NN path)
key_files:
  - tests/conftest.py
key_decisions:
  - nn_training_config returns prepare_config() output (validated dict) rather than raw dict, catching config errors at fixture construction time
  - Star graph uses 3 pairwise + 1 three-way factor to ensure bucket 0 message scope spans 3 variables (size 8), reliably exceeding ecl=4
patterns_established:
  - Problem fixtures return dicts with 'factors', 'elim_order', 'expected_log10_z' keys — standardized interface for all inference/training tests
observability_surfaces:
  - pytest fixture errors name the specific fixture that failed to construct
duration: ~15min
verification_result: passed
completed_at: 2026-03-12
blocker_discovered: false
---

# T01: Build shared fixtures and hand-built test problems

**Added 4 fixtures to conftest.py: nn_training_config, binary_chain_factors, ternary_chain_factors, star_graph_factors — all validated against analytic Z values and prepare_config.**

## What Happened

Extended `tests/conftest.py` with four new fixtures:

1. **nn_training_config**: 40-field flat config derived from `reference_flat_config` with CPU/small-epoch overrides (device=cpu, num_epochs=50, ecl=4, iB=2, hidden_sizes=[8,8], sampling_scheme=all, dope_factors=False). Dead fields removed. Returned value is the output of `prepare_config()` so any config validation error surfaces at fixture construction time.

2. **binary_chain_factors**: X0-X1 (domain 2), uniform pairwise factor. Z=2.0, verified via FastGM exact elimination.

3. **ternary_chain_factors**: X0-X1 (domain 3), uniform pairwise factor. Z=4.5, verified via FastGM exact elimination.

4. **star_graph_factors**: Hub X0 connected to X1,X2,X3 via 3 pairwise factors + 1 three-way factor f012(X0,X1,X2). Eliminating X0 first produces message scope {X1,X2,X3} with size 8, exceeding ecl=4. Z=1.052 computed by full enumeration of 16 assignments.

## Verification

- `python -m pytest tests/ --co -q | tail -1` → 110 tests collected (unchanged)
- `python -c "from tests.conftest import *"` → no import errors
- `prepare_config(nn_training_config)` → accepted without error (40 fields)
- Smoke test: built FastGM from each fixture's factors, ran exact elimination, verified log10(Z) within 1e-5 of analytic values
- Star graph with ecl=4: `get_large_message_buckets()` returns bucket [0] — NN path triggered
- `python -m pytest tests/ -q` → 110 passed

## Diagnostics

Fixtures are passive data providers. Inspect by reading `tests/conftest.py` docstrings. Each problem fixture's docstring documents the topology, factor values, and analytic Z derivation.

## Deviations

None.

## Known Issues

None.

## Files Created/Modified

- `tests/conftest.py` — Added imports (math, torch, FastFactor, prepare_config) and 4 new fixtures: nn_training_config, binary_chain_factors, ternary_chain_factors, star_graph_factors
