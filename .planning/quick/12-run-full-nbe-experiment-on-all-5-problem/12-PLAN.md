# Quick Task 12: Run full NBE experiment, log epochs-to-early-stopping

## Goal

Run the full NBE experiment on all 5 nbe_sanity_check problems with full epochs (500). For each problem, record (bucket_idx, hidden_sizes, epochs_used) for every NN-trained bucket. Save results to file.

## Task 1: Add epochs_trained tracking to bucket.py

**files:** `nce/inference/bucket.py`
**action:** After `t.train()` (line 321), store the actual epochs used on the bucket:
```python
self.epochs_trained = t.losses[-1][0] + 1 if t.losses else 0
```
Also store hidden_sizes on the bucket for later access:
```python
self.trained_hidden_sizes = hidden_sizes
```
These lines go right after line 321 (after t.train(), before the loss_fn2 check).

**verify:** No import changes, minimal 2-line addition
**done:** Buckets store epochs_trained and trained_hidden_sizes after NN training

## Task 2: Write and run the full experiment script

**files:** `notebooks/March-2026/claude_experiments/nbe_full_experiment.py`
**action:** Create script that:
1. Iterates all 5 nbe_sanity_check problems with their NBE configs (full num_epochs=500)
2. Runs `get_log_partition_function()` on each
3. After elimination, iterates `fastgm.buckets` to find buckets with `epochs_trained` attribute
4. Collects (bucket_idx, hidden_sizes, epochs_used) tuples
5. Writes per-problem output to `nbe_eval_results/nbe_full_epochs.txt`
6. Pings Discord on errors

**verify:** Script runs on all 5 problems, output file has epochs data
**done:** Output file shows per-bucket epochs for all problems
