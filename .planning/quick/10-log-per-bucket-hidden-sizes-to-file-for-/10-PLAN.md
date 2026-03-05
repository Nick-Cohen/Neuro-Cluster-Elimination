---
phase: quick-10
plan: 1
type: execute
wave: 1
depends_on: []
files_modified:
  - notebooks/March-2026/claude_experiments/nbe_log_hidden_sizes.py
  - notebooks/March-2026/claude_experiments/nbe_eval_results/grid10x10_hidden_sizes.txt
autonomous: true
requirements: [QUICK-10]
must_haves:
  truths:
    - "Per-bucket hidden sizes for grid10x10.f5.wrap are logged to a file"
    - "Script shows bucket label, width, message size, h computation, and final hidden_sizes for each NN-eligible bucket"
  artifacts:
    - path: "notebooks/March-2026/claude_experiments/nbe_log_hidden_sizes.py"
      provides: "Script to run grid10x10 with NBE config and log hidden sizes"
    - path: "notebooks/March-2026/claude_experiments/nbe_eval_results/grid10x10_hidden_sizes.txt"
      provides: "Output file with per-bucket hidden size data"
  key_links: []
---

<objective>
Log per-bucket hidden sizes to file for grid10x10.f5.wrap with NBE config.

Purpose: Understand what hidden layer sizes the NBE formula assigns to each NN-eligible bucket in grid10x10, to verify the adaptive sizing logic.
Output: A text file listing each NN-eligible bucket's width, message size, computed h value, and final hidden_sizes.
</objective>

<execution_context>
@/home/cohenn1/.claude/get-shit-done/workflows/execute-plan.md
@/home/cohenn1/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@docs/task_log_hidden_sizes.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create and run hidden sizes logging script</name>
  <files>
    notebooks/March-2026/claude_experiments/nbe_log_hidden_sizes.py
    notebooks/March-2026/claude_experiments/nbe_eval_results/grid10x10_hidden_sizes.txt
  </files>
  <action>
Create the script at notebooks/March-2026/claude_experiments/nbe_log_hidden_sizes.py with the EXACT content from docs/task_log_hidden_sizes.md (the script in the "Script to Write" section). Do not modify the script logic.

Then run it:
```bash
/home/cohenn1/NCE/venv/bin/python notebooks/March-2026/claude_experiments/nbe_log_hidden_sizes.py
```

The script will:
1. Load grid10x10.f5.wrap (problem index 4) from nbe_sanity_check benchmark set
2. Create FastGM with NBE config (num_epochs=1, device=cpu)
3. Iterate over buckets in elimination order, skipping exact-eligible ones
4. For each NN-eligible bucket, compute h = b * ceil(log2(message_size)) and log it
5. Print results and write to nbe_eval_results/grid10x10_hidden_sizes.txt
  </action>
  <verify>
    <automated>test -f /home/cohenn1/NCE/notebooks/March-2026/claude_experiments/nbe_eval_results/grid10x10_hidden_sizes.txt && head -10 /home/cohenn1/NCE/notebooks/March-2026/claude_experiments/nbe_eval_results/grid10x10_hidden_sizes.txt</automated>
  </verify>
  <done>grid10x10_hidden_sizes.txt exists and contains per-bucket hidden size data with columns for bucket label, width, message size, h computation, and hidden_sizes</done>
</task>

</tasks>

<verification>
- Output file exists at notebooks/March-2026/claude_experiments/nbe_eval_results/grid10x10_hidden_sizes.txt
- File contains header lines with model name, config parameters
- File contains rows for each NN-eligible bucket with hidden size info
</verification>

<success_criteria>
- Script runs without errors
- Output file is populated with per-bucket hidden size data for grid10x10.f5.wrap
</success_criteria>

<output>
After completion, create `.planning/quick/10-log-per-bucket-hidden-sizes-to-file-for-/10-SUMMARY.md`
</output>
