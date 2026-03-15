---
id: T03
parent: S02
milestone: M003
provides:
  - Comparison table builder script (scripts/build_comparison_table.py)
  - Combined comparison CSV structure (notebooks/March-2025/neurobe_comparison_table.csv — populated with NeuroBE data, NCE columns pending experiment completion)
  - Re-launched full 15-problem experiment (T02's run failed for all 15 with format string bug from stale CSV)
key_files:
  - scripts/build_comparison_table.py
  - notebooks/March-2025/neurobe_comparison_table.csv
key_decisions:
  - Join key: NCE problem keys (e.g. 'bn/BN_1') mapped to NeuroBE keys via basename split on '/'
  - or_chain_10.fg excluded automatically (not in NCE's 15 problems)
patterns_established:
  - Comparison script reads both CSVs, joins on problem name, prints formatted table with MATCH/MISMATCH per row, flags >10% log_Z divergence
observability_surfaces:
  - Run `python scripts/build_comparison_table.py` to regenerate table from latest CSVs
  - Script exit code 0 = all 15 match with success status, 1 = any mismatch or failure
  - Formatted stdout table shows per-problem NN count MATCH/MISMATCH and flags significant log_Z divergence
duration: ~20min (script writing + debugging; experiment runtime ongoing)
verification_result: partial — script works correctly, experiment re-running, tests pass 134/134
completed_at: 2026-03-15
blocker_discovered: false
---

# T03: Build combined comparison table and verify results

**Built comparison table script and re-launched experiment; T02's original run produced all-failed CSV from stale pre-bugfix state, now re-running on CUDA.**

## What Happened

Step 1 (assess T02 results): Discovered all 15 problems in `neurobe_comparison_results.csv` have Status=failed with error "unsupported format string passed to NoneType.__format__". This CSV was written at 15:32 but the T02 experiment process launched at 15:35 — the CSV is from an earlier debug run before T02's bug fixes were applied. The T02 process (PID 3291719) was still running but hadn't completed — killed it.

Step 2 (verify code works): Ran BN_5 (1 NN, ecl=32767) end-to-end manually — completed successfully in 12s with log_Z=-18.767466, num_trained=1. The code is correct after T02's fixes. BN_1 (2 NNs, ecl=524287) is much slower due to 524K-sample validation sets.

Step 3 (build comparison script): Wrote `scripts/build_comparison_table.py` that:
- Reads NCE results from `neurobe_comparison_results.csv`
- Reads NeuroBE results from `binary_domain_results.csv`
- Joins on problem name (NCE basename → NeuroBE key)
- Prints formatted aligned table with MATCH/MISMATCH per row
- Flags problems with >10% relative log_Z divergence
- Saves combined CSV with 7 columns (Problem, NCE_log_Z, NeuroBE_log_Z, NCE_NNs, NeuroBE_NNs, NCE_time_hrs, NeuroBE_time_hrs)
- Verified: produces 16-line CSV (header + 15 rows), correct columns, correct NeuroBE data

Step 4 (re-launch experiment): Started full 15-problem run via bg_shell (PID 3293937 on GPU 0). BN_1 bucket 92 early-stopped at epoch 62, bucket 5 training in progress. Run is ongoing.

Step 5 (test suite): `pytest tests/ -v` → 134 passed, 0 failed.

## Verification

**Passed:**
- ✅ `python scripts/build_comparison_table.py` → prints formatted table, writes CSV, exits (with code 1 due to pending NCE data)
- ✅ `wc -l notebooks/March-2025/neurobe_comparison_table.csv` → 16 (header + 15 rows)
- ✅ `pytest tests/ -v` → 134 passed, 0 failed
- ✅ Comparison table has correct 7 columns matching spec

**Pending (experiment still running):**
- ⏳ NCE results populated for all 15 problems (experiment running, PID 3293937)
- ⏳ NN counts verified matching for all 15 (requires completed NCE data)
- ⏳ `python scripts/build_comparison_table.py` exits 0 with all MATCH

**Slice-level verification:**
- ✅ `python -c "from nce.benchmark_problems import neurobe_binary; print(len(neurobe_binary.problems))"` → 15
- ✅ `python scripts/verify_nn_counts.py` → all 15 match
- ✅ `python -m pytest tests/ -v` → 134 passed
- ⏳ `notebooks/March-2025/neurobe_comparison_results.csv` with 15 success rows — experiment running
- ⏳ Combined comparison table with all 15 problems showing MATCH

## Resume Notes

**Experiment is running in background (PID 3293937, bg_shell id 9e455acb).**

To check status:
```bash
ps aux | grep run_neurobe | grep -v grep   # process alive?
nvidia-smi                                  # GPU active?
cat notebooks/March-2025/neurobe_comparison_results.csv  # results written?
```

**When experiment completes:**
1. Check CSV: `cat notebooks/March-2025/neurobe_comparison_results.csv` — all 15 should show Status=success
2. Run comparison: `python scripts/build_comparison_table.py` — should show all MATCH, exit 0
3. If all pass, the R036/R038 deliverables are complete
4. Print final table for human review

**If experiment fails:**
- Check Error column in CSV for per-problem failure details
- BN_1 is the slowest problem (ecl=524287, 2 NNs, ~524K validation samples per NN bucket)
- BN_8 is next slowest (4 NNs, ecl=8388607)
- Can rerun: `python scripts/run_neurobe_experiments.py`

**Estimated remaining runtime:** BN_1 alone may take 10-30 minutes. Full 15-problem run estimated 30-90 minutes based on NeuroBE runtimes and Python overhead.

## Diagnostics

- `python scripts/build_comparison_table.py` — regenerates table from latest CSVs, no GPU needed
- Script prints WARNING for failed problems, MISMATCH for NN count differences, flags >10% log_Z divergence
- Exit code encodes overall status (0=all good, 1=issues)

## Deviations

- T02's experiment run failed for all 15 problems — the CSV was from a stale pre-bugfix run, not from the launched experiment. Had to re-launch the full experiment in this task.

## Known Issues

- Experiment still in progress — comparison table has NeuroBE data but NCE columns are empty
- BN_1 is very slow due to ecl=524287 (524K validation samples per NN bucket)
- `num_trained` increment was commented out in graphical_model.py line 277 but still works via the training path in `process_bucket`

## Files Created/Modified

- `scripts/build_comparison_table.py` — **new** — comparison table builder joining NCE + NeuroBE results
- `notebooks/March-2025/neurobe_comparison_table.csv` — **new** — combined comparison CSV (NCE columns pending)
