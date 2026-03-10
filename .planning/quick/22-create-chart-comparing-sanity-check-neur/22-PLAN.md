---
phase: quick-22
plan: 1
type: execute
wave: 1
depends_on: []
files_modified:
  - notebooks/3-2026/claude_experiments/compare_paper_vs_sanity_check.py
autonomous: true
requirements: [QUICK-22]

must_haves:
  truths:
    - "Chart shows side-by-side comparison of paper WMB error, paper NeuroBE errors, and our WMB error for 4 matching problems"
    - "Errors are computed using refZ from the paper CSV as ground truth"
    - "Chart is saved as PNG in the claude_experiments directory"
  artifacts:
    - path: "notebooks/3-2026/claude_experiments/compare_paper_vs_sanity_check.py"
      provides: "Comparison chart script"
      min_lines: 60
    - path: "notebooks/3-2026/claude_experiments/compare_paper_vs_sanity_check.png"
      provides: "Output chart image"
  key_links:
    - from: "compare_paper_vs_sanity_check.py"
      to: "NCE-Data/NeuroBE_paper_results.csv"
      via: "pandas read_csv"
      pattern: "pd\\.read_csv"
    - from: "compare_paper_vs_sanity_check.py"
      to: "nbe_eval_results/nbe_full_epochs.txt"
      via: "hardcoded dict (parsed results)"
      pattern: "nbe_full_epochs"
---

<objective>
Create a Python script that produces a grouped bar chart comparing NeuroBE paper results with our sanity check WMB results for the 4 overlapping problems.

Purpose: Visually assess how our WMB implementation compares to the published WMB and NeuroBE baselines from the paper.
Output: compare_paper_vs_sanity_check.py script and .png chart
</objective>

<execution_context>
@/home/cohenn1/.claude/get-shit-done/workflows/execute-plan.md
@/home/cohenn1/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@NCE-Data/NeuroBE_paper_results.csv
@notebooks/March-2026/claude_experiments/nbe_eval_results/nbe_full_epochs.txt
@notebooks/3-2026/claude_experiments/plot_exact_vs_nbe_logz.py (style reference)

<interfaces>
<!-- Paper CSV columns (relevant subset): -->
<!-- name, refZ, WMB_error, NeuroBE_avg_error, NeuroBE_min_error -->

<!-- nbe_full_epochs.txt results (ecl=2^(iB-1), 0 NN buckets = WMB with that ecl): -->
<!-- pedigree13: logZ = -31.764139 -->
<!-- grid40x40.f10: logZ = 5349.348633 -->
<!-- grid20x20.f10: logZ = 1197.311035 -->
<!-- rbm_20: logZ = 57.987244 -->

<!-- Phase 3 results (ecl=2^22, essentially exact WMB): -->
<!-- pedigree13: logZ = -23.4234 -->
<!-- grid40x40.f10: ERROR (network timeout) -->
<!-- grid20x20.f10: logZ = 1332.5142 -->
<!-- rbm_20: logZ = 58.5306 -->

<!-- refZ from paper CSV (ground truth for error computation): -->
<!-- pedigree13: -31.18 -->
<!-- grid4040f10 (grid40x40.f10): 5490 -->
<!-- grid2020f10 (grid20x20.f10): 1311.98 -->
<!-- rbm20 (rbm_20): 58.53 -->

<!-- Name mapping (paper name -> our name): -->
<!-- pedigree13 -> pedigree13 -->
<!-- grid4040f10 -> grid40x40.f10 -->
<!-- grid2020f10 -> grid20x20.f10 -->
<!-- rbm20 -> rbm_20 -->
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create comparison chart script and generate PNG</name>
  <files>notebooks/3-2026/claude_experiments/compare_paper_vs_sanity_check.py</files>
  <action>
Create a Python script that:

1. Reads the paper CSV from `/home/cohenn1/NCE/NCE-Data/NeuroBE_paper_results.csv` using pandas.

2. Defines our sanity check results as a hardcoded dict (from nbe_full_epochs.txt):
   - pedigree13: logZ = -31.764139
   - grid40x40.f10: logZ = 5349.348633
   - grid20x20.f10: logZ = 1197.311035
   - rbm_20: logZ = 57.987244

   Also include Phase 3 exact WMB results for comparison:
   - pedigree13: logZ = -23.4234
   - grid20x20.f10: logZ = 1332.5142
   - rbm_20: logZ = 58.5306
   - grid40x40.f10: None (errored)

3. Matches the 4 overlapping problems using name mapping:
   - paper "pedigree13" (row id=1, section="i-bound=20") -> our "pedigree13"
   - paper "grid4040f10" (row id=1, section="i-bound=20") -> our "grid40x40.f10"
   - paper "grid2020f10" (row id=4, section="i-bound=10") -> our "grid20x20.f10"
   - paper "rbm20" (row id=1, section="i-bound=20") -> our "rbm_20"

4. Computes errors using refZ from the paper CSV as ground truth:
   - our_wmb_error = |our_logZ - refZ| (from nbe_full_epochs.txt)
   - our_exact_wmb_error = |phase3_logZ - refZ| (from phase 3, where available)

5. Creates a figure with two subplots:

   **Left subplot: Grouped bar chart** showing for each of the 4 problems:
   - Paper WMB error (from WMB_error column)
   - Paper NeuroBE avg error (from NeuroBE_avg_error column)
   - Paper NeuroBE min error (from NeuroBE_min_error column)
   - Our WMB error (ecl=2^(iB-1), computed as |our_logZ - refZ|)
   - Our exact WMB error (ecl=2^22, computed as |phase3_logZ - refZ|, skip grid40x40 which errored)

   Use distinct colors for each bar group. Add value labels on bars (fontsize=7).
   Use log scale on y-axis since errors span orders of magnitude (0.0007 to 215).
   Title: "Paper vs Our Results: Absolute Error in log Z"

   **Right subplot: Table or text summary** showing the raw numbers for each problem:
   problem, refZ, paper_wmb_err, paper_nbe_avg, paper_nbe_min, our_wmb_err, our_exact_wmb_err

6. Save to `/home/cohenn1/NCE/notebooks/3-2026/claude_experiments/compare_paper_vs_sanity_check.png` at dpi=150.

7. Print a summary table to stdout with all values.

Style notes (follow existing conventions from plot_exact_vs_nbe_logz.py):
- Use matplotlib, numpy, pandas
- edgecolor='black', linewidth=0.5 on bars
- grid(axis='y', alpha=0.3)
- rotation=25 on x labels, ha='right'
- tight_layout() + bbox_inches='tight'
- figsize around (18, 7) to accommodate 5 bar groups per problem

Important: The nbe_full_epochs.txt results have 0 NN-trained buckets. This means they are pure WMB results (no neural network approximation was used). The ecl was set to 2^(iB-1) which is the NeuroBE default, but since all bucket widths fell below this threshold, no NN training occurred. Label these bars accordingly as "Our WMB (ecl=2^(iB-1))" and "Our WMB (ecl=2^22)" respectively.
  </action>
  <verify>
    Run the script: `/home/cohenn1/NCE/venv/bin/python /home/cohenn1/NCE/notebooks/3-2026/claude_experiments/compare_paper_vs_sanity_check.py`
    - Script exits with code 0
    - PNG file exists at notebooks/3-2026/claude_experiments/compare_paper_vs_sanity_check.png
    - Summary table is printed to stdout showing all 4 problems with error values
  </verify>
  <done>
    - PNG chart saved showing grouped bars for all 4 matching problems
    - Each problem shows paper WMB error, paper NeuroBE avg/min errors, our WMB error(s)
    - Errors computed using refZ from paper CSV as ground truth
    - Summary table printed to stdout
  </done>
</task>

</tasks>

<verification>
- Script runs without errors
- PNG file is generated and non-empty
- All 4 matching problems are represented in the chart
- Error values match manual computation (e.g., pedigree13 our WMB error = |-31.764139 - (-31.18)| = 0.584139)
</verification>

<success_criteria>
- Chart clearly shows how our WMB results compare to published paper baselines
- All 5 error sources (paper WMB, paper NeuroBE avg, paper NeuroBE min, our WMB ecl=2^(iB-1), our WMB ecl=2^22) are distinguishable
- refZ from paper CSV is used as ground truth for all error computations
</success_criteria>

<output>
After completion, create `.planning/quick/22-create-chart-comparing-sanity-check-neur/22-SUMMARY.md`
</output>
