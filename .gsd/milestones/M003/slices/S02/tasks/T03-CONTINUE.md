# T03 Resume — Build combined comparison table and verify results

## Status: Partially Complete

### What's Done
- ✅ `scripts/build_comparison_table.py` written and tested (Step 1)
- ✅ `pytest tests/ -v` → 134 passed (Step 3)
- ✅ Comparison table CSV structure verified (16 lines, correct columns)

### What's Pending
- ⏳ 15-problem experiment running on CUDA (PID 3293937, bg_shell id 9e455acb)
- ⏳ Step 2: Run comparison script with completed NCE results, verify all 15 MATCH
- ⏳ Step 4: Print final summary table for human review
- ⏳ Must-haves: NN counts match verification, NCE data in comparison table

### Resume Steps
1. Check if experiment completed: `ps aux | grep run_neurobe | grep -v grep`
2. If not running, check results: `cat notebooks/March-2025/neurobe_comparison_results.csv`
3. If all 15 show Status=success: `python scripts/build_comparison_table.py` → should exit 0 with all MATCH
4. If experiment failed: diagnose from Error column, fix and re-run
5. Print final table for human review (Step 4)
6. Update T03-SUMMARY.md with final verification results
7. Mark T03 `[x]` in S02-PLAN.md
8. Run slice-level verification checks
9. Update STATE.md

### Key Context
- T02's CSV was stale (pre-bugfix). Code works correctly — verified BN_5 end-to-end (12s, log_Z=-18.767466).
- BN_1 is slowest (ecl=524287, 2 NNs). Estimated total runtime: 30-90 minutes.
- The comparison script handles failed problems gracefully (shows WARNING + MISMATCH).
