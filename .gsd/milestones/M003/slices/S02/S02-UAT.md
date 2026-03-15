# S02: ECL Tuning & Comparison Experiments — UAT

**Milestone:** M003
**Written:** 2026-03-15

## UAT Type

- UAT mode: mixed (artifact-driven + live-runtime)
- Why this mode is sufficient: NN count matching is artifact-verified (scripts/verify_nn_counts.py), but the comparison table requires live GPU inference results. The human reviews the comparison table for log_Z plausibility — automated checks verify NN count parity but cannot judge whether log_Z values are "reasonable."

## Preconditions

- Virtual environment activated: `source venv/bin/activate`
- GPU available: `nvidia-smi` shows free GPU memory
- Experiment completed: `notebooks/March-2025/neurobe_comparison_results.csv` exists with 15 success rows (experiment running as of 2026-03-15 16:45 PDT)

## Smoke Test

```bash
python -c "from nce.benchmark_problems import neurobe_binary; print(len(neurobe_binary.problems))"
# Expected: 15
```

## Test Cases

### 1. NN count parity (R036)

1. `python scripts/verify_nn_counts.py`
2. **Expected:** All 15 problems show MATCH. Exit code 0. Output ends with "All 15 problems MATCH."

### 2. Library test suite (regression)

1. `python -m pytest tests/ -v`
2. **Expected:** 134 passed, 0 failed.

### 3. Combined comparison table (R038)

1. Wait for experiment to complete (check: `ps aux | grep run_neurobe | grep -v grep` returns empty)
2. `python scripts/build_comparison_table.py`
3. **Expected:** Formatted table with 15 rows. All rows show MATCH for NN counts. NCE_log_Z values are finite numbers. Exit code 0.

### 4. Log_Z plausibility review (human)

1. Examine comparison table output from test case 3
2. **Expected:** NCE log_Z values are in the same order of magnitude as NeuroBE log_Z values. No wildly divergent results (>10x difference flagged by script).

### 5. Benchmark module importability

1. `python -c "from nce.benchmark_problems import neurobe_binary; print(neurobe_binary.configs['neurobe'][0]['neurobe_mode'])"`
2. **Expected:** `True`

## Edge Cases

### All 15 models load without error

1. `python -c "from nce.benchmark_problems.neurobe_binary import neurobe_binary; [print(f'{i}: ok') for i, _ in enumerate(neurobe_binary.problems)]"`
2. **Expected:** Prints `0: ok` through `14: ok` — no ValueError from root-variable loading.

### Experiment handles per-problem failures gracefully

1. If any problem fails during the experiment run, check `notebooks/March-2025/neurobe_comparison_results.csv`
2. **Expected:** Failed problems have Status=failed with Error description. Successful problems still have valid results. Script continues past failures.

## Failure Signals

- `verify_nn_counts.py` shows any MISMATCH → ecl formula or benchmark config is wrong
- Comparison table shows all NCE_log_Z empty → experiment failed (check CSV Status/Error columns)
- pytest shows failures → regression introduced by S02 code changes
- `build_comparison_table.py` exits 1 → NN count mismatch or failed problems in NCE results

## Requirements Proved By This UAT

- R036 (Matched NN counts via ecl tuning) — test case 1 proves all 15 NN counts match NeuroBE
- R038 (Combined NeuroBE comparison results table) — test cases 3+4 prove the comparison table exists with both NCE and NeuroBE data, and log_Z values are plausible

## Not Proven By This UAT

- Whether NCE neurobe_mode produces numerically identical results to NeuroBE C++ — expected to differ due to implementation differences (Python vs C++, PyTorch vs custom NN, floating point paths). The comparison shows the same algorithm structure produces similar results, not identical ones.
- Long-term reproducibility of results across different GPU hardware or CUDA versions
- Performance comparison validity — NeuroBE times are all 0.0000 hrs in the CSV, making time comparison uninformative

## Notes for Tester

- The experiment (15-problem CUDA inference) was launched at 2026-03-15 16:07 PDT and is still running as of artifact creation. Check `ps aux | grep run_neurobe` — if still running, wait for completion before running test case 3.
- BN_1 (ecl=524287) and BN_8 (4 NNs, ecl=8388607) are the slowest problems. Estimated total runtime: 30-90 minutes.
- If the experiment needs re-running: `python scripts/run_neurobe_experiments.py` (overwrites CSV)
- The comparison table flags any NCE log_Z diverging >10% from NeuroBE with a WARNING line. Some divergence is expected — these are approximate inference methods with different implementations.
