---
phase: quick
plan: 11
subsystem: benchmark-problems
tags: [nbe, hidden-sizes, elimination, bucket-width, induced-width]
dependency-graph:
  requires: [quick-10]
  provides: [grid10x10-hidden-sizes-output]
  affects: []
tech-stack:
  added: []
  patterns: [custom_hidden_sizes-callback-hook]
key-files:
  created: []
  modified:
    - notebooks/March-2026/claude_experiments/nbe_log_hidden_sizes.py
    - notebooks/March-2026/claude_experiments/nbe_eval_results/grid10x10_hidden_sizes.txt
decisions:
  - Use custom_hidden_sizes callback (called during elimination) instead of pre-elimination bucket scan
  - callback returns computed hidden_sizes so training proceeds normally at num_epochs=1
metrics:
  duration: 5 min
  completed: 2026-03-05
  tasks_completed: 1
  files_modified: 2
---

# Quick Task 11: Fix hidden-sizes script - capture bucket widths during elimination Summary

## One-liner

Rewrote nbe_log_hidden_sizes.py to use the custom_hidden_sizes callback hook (called inside compute_message_nn during actual elimination) instead of pre-elimination bucket scan, yielding true induced widths 10-21 for 33 NN-eligible buckets on grid10x10.f5.wrap.

## What Was Done

### Task 1: Rewrite nbe_log_hidden_sizes.py

**Root cause of bug:** The previous script (quick-10) iterated over `fastgm.buckets` and called `bucket.get_width()` BEFORE elimination. At that point, each bucket only contains its originally assigned factors (max scope width 4 for grid10x10). All widths <= iB=10, so no NN-eligible buckets were found.

**Fix:** Used the `config['custom_hidden_sizes']` callback mechanism in bucket.py (lines 200-203). This callback is called inside `compute_message_nn()` for each NN-eligible bucket DURING elimination. At call time, the bucket has already received all messages from earlier-eliminated buckets, so `get_width()` and `get_message_size()` return the true induced-width scope.

**Implementation:**
1. Defined `capture_and_compute_hidden_sizes(bucket)` callback that:
   - Calls `bucket.get_message_size()` and `bucket.get_width()` at elimination time
   - Computes `h = b * ceil(log2(message_size))` using same NBE formula
   - Appends `{label, width, ec, msg_size, h, hidden_sizes}` to a list
   - Returns `[h, h]` so NN training proceeds normally
2. Set `config['custom_hidden_sizes'] = capture_and_compute_hidden_sizes`
3. Set `num_epochs=1` to minimize training overhead
4. Called `get_log_partition_function()` to trigger full elimination
5. Wrote output table to `nbe_eval_results/grid10x10_hidden_sizes.txt`

**Result:** 33 NN-eligible buckets captured with widths ranging 10-21 (induced width), confirming the fix works. The previous script produced an empty table.

## Key Output

```
NN-eligible buckets captured during elimination (bucket widths reflect induced width):

Bucket     Width    EC             Msg Size       h=b*ceil(log2(msg))    hidden_sizes
------------------------------------------------------------------------------------------
65         10       1024.0         1024.0         10                     [10, 10]
...
18         21       2097152.0      2097152.0      21                     [21, 21]
...

Total NN-eligible buckets: 33
Width range: 10 - 21
h range: 10 - 21
```

## Deviations from Plan

None - plan executed exactly as written. Note: both the script and output file are in the gitignored `notebooks/` directory, so no git commit was made for the task itself. The planning metadata commit documents completion.

## Self-Check: PASSED

- Script exists and runs: CONFIRMED
- Output file exists: CONFIRMED at notebooks/March-2026/claude_experiments/nbe_eval_results/grid10x10_hidden_sizes.txt
- Bucket widths > 4: CONFIRMED (range 10-21, matching induced width, not pre-elimination max of 4)
- All 33 NN-eligible buckets captured: CONFIRMED
