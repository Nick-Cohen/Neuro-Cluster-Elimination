---
estimated_steps: 4
estimated_files: 3
---

# T03: Build combined comparison table and verify results

**Slice:** S02 — ECL Tuning & Comparison Experiments
**Milestone:** M003

## Description

Parse NCE experiment results alongside NeuroBE C++ results into the final combined comparison table (R038 deliverable). Verify NN counts match for all 15 problems (R036 final proof). The table is the milestone's primary output — it enables direct assessment of NCE neurobe_mode reproduction quality.

## Steps

1. **Write `scripts/build_comparison_table.py`** — Script that:
   - Reads NCE results from `notebooks/March-2025/neurobe_comparison_results.csv`
   - Reads NeuroBE results from `Clean-NeuroBE/results/binary_min_nn/binary_domain_results.csv`
   - Joins on problem name (NCE Problem column matches NeuroBE Problem column)
   - Produces combined table with columns: Problem, NCE_log_Z, NeuroBE_log_Z, NCE_NNs, NeuroBE_NNs, NCE_time_hrs, NeuroBE_time_hrs
   - Asserts NCE_NNs == NeuroBE_NNs for all 15 problems (prints PASS/FAIL per row)
   - Prints formatted table to stdout (aligned columns for readability)
   - Saves combined table to `notebooks/March-2025/neurobe_comparison_table.csv`

2. **Run the comparison script** — Execute and review output. Verify all 15 problems present, NN counts match, log_Z values are in reasonable range (same order of magnitude as NeuroBE, not identical due to RNG/numerical differences).

3. **Run full test suite** — `pytest tests/ -v` to confirm nothing broke during the slice. Should be 134+ tests passing.

4. **Print final summary** — Display the comparison table for human review. Note any problems where NCE log_Z diverges significantly from NeuroBE (> 10% relative difference) as items for the user to investigate.

## Must-Haves

- [ ] Combined comparison table CSV has all 15 problems
- [ ] NN counts match NeuroBE for all 15 problems (R036 verified)
- [ ] Table includes Problem, NCE_log_Z, NeuroBE_log_Z, NCE_NNs, NeuroBE_NNs, NCE_time_hrs, NeuroBE_time_hrs
- [ ] Formatted table printed to stdout for human review
- [ ] `pytest tests/` passes (134+ tests green)

## Verification

- `python scripts/build_comparison_table.py` → prints table, all NN counts show MATCH, exits 0
- `wc -l notebooks/March-2025/neurobe_comparison_table.csv` → 16 (header + 15 rows)
- `pytest tests/ -v` → 134+ passed, 0 failed

## Observability Impact

- Signals added/changed: None (this task is reporting, not runtime)
- How a future agent inspects this: Read `notebooks/March-2025/neurobe_comparison_table.csv` for combined results; run `python scripts/build_comparison_table.py` to regenerate
- Failure state exposed: NN count mismatches printed with expected vs actual values

## Inputs

- `notebooks/March-2025/neurobe_comparison_results.csv` — T02's raw NCE results
- `Clean-NeuroBE/results/binary_min_nn/binary_domain_results.csv` — NeuroBE ground truth
- S02-RESEARCH.md ECL values table — expected NN counts for cross-reference

## Expected Output

- `scripts/build_comparison_table.py` — comparison table builder script
- `notebooks/March-2025/neurobe_comparison_table.csv` — combined comparison table (the R038 deliverable)
