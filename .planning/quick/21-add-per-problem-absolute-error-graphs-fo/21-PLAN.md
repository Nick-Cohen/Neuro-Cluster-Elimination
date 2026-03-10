---
phase: quick-21
plan: 1
type: execute
wave: 1
depends_on: []
files_modified:
  - notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/visualize_updated.py
autonomous: true
---

<objective>
Add per-problem absolute error bar charts (linear + symlog) for the 9 pattern-1 problems
(4 configs completed, only ukl_bw30 failed) to a no_exact_bw/ subfolder.
</objective>

<tasks>
<task type="auto">
  <name>Task 1: Add pattern-1 graph generation to visualize_updated.py</name>
  <files>notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/visualize_updated.py</files>
  <action>
  Add a section between the clean-problem plots and the summary CSV that:
  1. Identifies pattern-1 problems (all 4 non-bw30 configs completed, not in clean set)
  2. Generates linear + symlog bar charts using only CONFIGS_NO_BW30
  3. Saves to results/updated_graphs_and_table/no_exact_bw/
  </action>
  <done>18 PNG files (9 problems x 2 scales) in no_exact_bw/ subfolder</done>
</task>
</tasks>
