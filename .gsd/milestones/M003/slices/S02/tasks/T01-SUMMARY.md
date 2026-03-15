---
id: T01
parent: S02
milestone: M003
provides:
  - Fixed _load_from_uai to prefer elim_order over .vo file (root-variable bug)
  - neurobe_binary benchmark config module with 15 problems and per-problem ecl values
  - NN count verification script (scripts/verify_nn_counts.py)
key_files:
  - nce/inference/graphical_model.py
  - nce/benchmark_problems/neurobe_binary.py
  - nce/benchmark_problems/__init__.py
  - scripts/verify_nn_counts.py
key_decisions:
  - neurobe_binary configs use neurobe_mode=True with minimal per-problem overrides (ecl, num_samples, val_set, dope_factors, device, seed) — NEUROBE_DEFAULTS fills the rest via prepare_config
  - NEUROBE_NN_COUNTS dict exported from neurobe_binary module for reuse in verification and future experiment scripts
patterns_established:
  - neurobe benchmark modules use neurobe_mode=True for config defaults rather than duplicating NEUROBE_DEFAULTS keys
observability_surfaces:
  - scripts/verify_nn_counts.py — standalone NN count verification without GPU, exits 0/1
  - NEUROBE_NN_COUNTS dict in neurobe_binary.py — ground truth for programmatic verification
duration: 15m
verification_result: passed
completed_at: 2026-03-12
blocker_discovered: false
---

# T01: Fixed root-variable loading and built neurobe_binary benchmark config module

**Fixed `_load_from_uai` to prefer `elim_order` over `.vo` file, built `neurobe_binary` BenchmarkSet with 15 problems and per-problem ecl values, all 15 NN counts match NeuroBE.**

## What Happened

The `.vo` file format skips the root variable (n-1 vars), causing `_create_buckets_from_factors` to raise `ValueError` for root-only factors. `uai_to_GM` prioritizes `order_file` over `elim_order` when both are provided. Fix: when `elim_order` is not None, call `uai_to_GM` without `order_file` so it uses the full n-var order from `model.order`.

Built `nce/benchmark_problems/neurobe_binary.py` following the `nbe_sanity_check.py` pattern. 15 binary-domain models with per-problem `ecl = 2^width_problem - 1` values from NeuroBE CSV. Configs use `neurobe_mode=True` so `NEUROBE_DEFAULTS` fills training params. Exported `NEUROBE_NN_COUNTS` dict for programmatic verification.

Wrote `scripts/verify_nn_counts.py` — loads all 15 models on CPU, calls `get_large_message_buckets(iB=25, ecl=ecl)`, compares against NeuroBE ground truth. All 15 match.

## Verification

- `python scripts/verify_nn_counts.py` → all 15 MATCH, exit code 0
- `python -c "from nce.benchmark_problems import neurobe_binary; print(len(neurobe_binary.problems))"` → 15
- `pytest tests/ -v` → 134 passed, 0 failed

### Slice-level checks status (T01 is intermediate):
- [x] neurobe_binary import → 15 problems
- [x] verify_nn_counts.py → all 15 match
- [x] pytest → 134 passed
- [ ] neurobe_comparison_results.csv — T02
- [ ] Combined comparison table — T03

## Diagnostics

- Run `python scripts/verify_nn_counts.py` for instant NN count verification (no GPU needed)
- Mismatch lines show problem name, expected count, actual count for debugging
- `NEUROBE_NN_COUNTS` dict in `neurobe_binary.py` can be imported for programmatic checks

## Deviations

None.

## Known Issues

None.

## Files Created/Modified

- `nce/inference/graphical_model.py` — Fixed `_load_from_uai` to not pass `order_file` when `elim_order` is provided
- `nce/benchmark_problems/neurobe_binary.py` — New: 15 binary-domain models with neurobe_mode configs and per-problem ecl values
- `nce/benchmark_problems/__init__.py` — Added `neurobe_binary` export
- `scripts/verify_nn_counts.py` — New: standalone NN count verification script
