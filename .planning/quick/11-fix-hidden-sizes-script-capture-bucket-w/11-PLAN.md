# Quick Task 11: Fix hidden sizes script — capture bucket widths during elimination

## Goal

The previous script (quick-10) checked bucket widths BEFORE elimination. At that point, buckets only contain original factors and have small widths (max 4). During actual elimination, messages from earlier buckets are added to later buckets, growing their scope up to the induced width (21 for grid10x10). Fix the script to capture hidden sizes during actual elimination.

## Root Cause

`FastGM.__init__` places original factors into buckets. Before elimination, each bucket only contains its assigned factors. `get_width()` returns message scope size from those factors alone. The true bucket width at elimination time is larger because it includes received messages from earlier buckets.

## Task 1: Rewrite nbe_log_hidden_sizes.py to capture info during elimination

**files:** `notebooks/March-2026/claude_experiments/nbe_log_hidden_sizes.py`
**action:** Rewrite the script to use the `custom_hidden_sizes` config callback (bucket.py:200-203) as a hook. This function is called during `compute_message_nn()` for each NN-eligible bucket DURING elimination, when the bucket has its final scope.

The callback:
1. Computes hidden sizes using the same NBE formula (b * ceil(log2(message_size)))
2. Stores bucket label, width, ec, msg_size, and computed hidden_sizes in a list
3. Returns the hidden_sizes for actual training

Set `num_epochs=1` to minimize training time. Run `get_log_partition_function()` to trigger elimination.

Output: table of all NN-eligible buckets with their widths and hidden sizes, written to `nbe_eval_results/grid10x10_hidden_sizes.txt`.

**verify:** Script runs, output file has rows with bucket widths > 4 (up to ~21)
**done:** Output file shows per-bucket hidden sizes for all NN-eligible buckets
