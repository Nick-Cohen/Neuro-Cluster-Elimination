---
estimated_steps: 5
estimated_files: 5
---

# T01: Fix root-variable loading and build neurobe benchmark config module

**Slice:** S02 — ECL Tuning & Comparison Experiments
**Milestone:** M003

## Description

Two things block experiments: (1) `_load_from_uai` can't load 15 binary-domain models because `.vo` files skip the root variable, and (2) no neurobe benchmark config module exists with correct per-problem ecl values. This task fixes the loading bug (prefer `elim_order` over `.vo` file when provided), builds the neurobe_binary config module with ecl values from NeuroBE CSV, and writes a verification script that proves NN counts match NeuroBE for all 15 problems (R036).

## Steps

1. **Fix root-variable loading in `_load_from_uai`** — When `elim_order` is provided (from `model.order`), don't pass `order_file` to `uai_to_GM`. This makes `uai_to_GM` use the `elim_order` path (all n vars) instead of the `.vo` path (n-1 vars). Keep the `.vo` fallback when `elim_order` is None. Test: load `bn/BN_1` from catalog via `FastGM(model=catalog['bn/BN_1'], ...)` and verify no ValueError.

2. **Handle evidence-only application** — When `model.evidence` is provided AND `.evid` file exists, evidence could be applied twice. Check: `_load_from_uai` currently applies evidence from `.evid` file OR from `evid` param, never both (the `elif` handles this). Verify this is still correct after the fix. Also verify that when `elim_order` is passed, the `evid` param still flows through correctly.

3. **Build `nce/benchmark_problems/neurobe_binary.py`** — Follow `nbe_sanity_check.py` pattern: `_MODEL_KEYS` list (15 keys from catalog mapping table), `_NEUROBE_ECL` dict with per-problem ecl values computed as `2^width_problem - 1` from NeuroBE CSV, `_build_neurobe_configs()` that creates configs with `neurobe_mode=True`, per-problem `ecl`, `num_samples='nbe,0.1'`, `val_set=True`, `dope_factors=True`, `device='cuda'`, `seed=42`. Module-level `neurobe_binary = BenchmarkSet(...)` instance.

4. **Export from `__init__.py`** — Add `from .neurobe_binary import neurobe_binary` to `nce/benchmark_problems/__init__.py`.

5. **Write `scripts/verify_nn_counts.py`** — Standalone script that loads each of the 15 models via `FastGM(model=model, nn_config=config, device='cpu')`, calls `get_large_message_buckets(iB=25, ecl=config['ecl'])`, compares count to NeuroBE's NN count. Prints per-problem results. Exits with code 0 if all match, code 1 if any mismatch. Expected NeuroBE NN counts from CSV: BN_1=2, BN_2=3, BN_3=1, BN_5=1, BN_7=1, BN_8=4, BN_9=1, BN_10=2, BN_11=1, grid10x10.f5.wrap=1, smokers_20=1, 10_14_s.binary=3, 10_16_s.binary=2, 11_17_s.binary=1, 11_4_s.binary=1.

## Must-Haves

- [ ] `_load_from_uai` prefers `elim_order` over `.vo` file when both are available
- [ ] All 15 binary-domain models load from catalog without ValueError
- [ ] `neurobe_binary.py` exports a BenchmarkSet with 15 problems and neurobe_mode configs
- [ ] Per-problem ecl values use `2^width_problem - 1` formula (D031)
- [ ] `num_samples='nbe,0.1'` set for all 15 configs
- [ ] `verify_nn_counts.py` passes — all 15 NN counts match NeuroBE
- [ ] `pytest tests/` still passes (134+ tests green)

## Verification

- `source venv/bin/activate && python scripts/verify_nn_counts.py` → prints 15 lines, all "MATCH", exits 0
- `source venv/bin/activate && python -c "from nce.benchmark_problems import neurobe_binary; print(len(neurobe_binary.problems))"` → 15
- `source venv/bin/activate && python -m pytest tests/ -v` → 134+ passed

## Observability Impact

- Signals added/changed: `verify_nn_counts.py` prints per-problem comparison (problem, expected NNs, actual NNs, MATCH/MISMATCH)
- How a future agent inspects this: Run `python scripts/verify_nn_counts.py` for instant NN count verification without GPU
- Failure state exposed: Mismatch lines show which problems have wrong NN counts and by how much

## Inputs

- `Clean-NeuroBE/results/binary_min_nn/binary_domain_results.csv` — ground truth ecl values (width_problem column) and NN counts
- `nce/benchmark_problems/nbe_sanity_check.py` — BenchmarkSet pattern to follow
- `nce/config_schema.py` — NEUROBE_DEFAULTS dict (16 keys from S01)
- S02-RESEARCH.md catalog key mapping table — maps NeuroBE problem names to catalog keys
- S02-RESEARCH.md ECL values table — pre-computed ecl = 2^wp - 1 for all 15 problems

## Expected Output

- `nce/inference/graphical_model.py` — `_load_from_uai` fixed to prefer elim_order over .vo file
- `nce/benchmark_problems/neurobe_binary.py` — new module with 15 neurobe_mode configs
- `nce/benchmark_problems/__init__.py` — exports neurobe_binary
- `scripts/verify_nn_counts.py` — standalone NN count verification script (exits 0 on success)
