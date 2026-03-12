---
phase: quick-20
plan: 1
type: execute
wave: 1
depends_on: []
files_modified:
  - notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/visualize_updated.py
autonomous: true
requirements: [VIZ-01, VIZ-02, VIZ-03]

must_haves:
  truths:
    - "Per-problem absolute error bar charts (linear + log scale) saved as PNG files"
    - "Summary CSV with all required columns exists and has correct data"
    - "Config analysis answer about track_errors documented in script output"
  artifacts:
    - path: "notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/visualize_updated.py"
      provides: "Visualization script producing updated graphs and CSV"
      min_lines: 150
    - path: "notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/results/updated_graphs_and_table/"
      provides: "Output directory with per-problem PNGs and summary.csv"
  key_links:
    - from: "visualize_updated.py"
      to: "results/{config_name}/*.json"
      via: "json.load for each result file"
      pattern: "json\\.load"
    - from: "visualize_updated.py"
      to: "nce.benchmark_problems.small_problems"
      via: "import for ground truth PR, width, num_vars"
      pattern: "from nce\\.benchmark_problems"
---

<objective>
Create a visualization script that reads existing benchmark WMSE vs UKL result JSONs and produces:
(1) Per-problem grouped bar charts of absolute error (linear + log scale), (2) a summary CSV table,
and (3) answers whether the experiments used track_errors config toggle.

Purpose: Enable analysis of the WMSE vs UKL benchmark results with per-problem granularity,
custom color scheme/labels, and a machine-readable summary table.

Output: visualize_updated.py script + PNG plots + summary.csv in results/updated_graphs_and_table/
</objective>

<execution_context>
@/home/cohenn1/.claude/get-shit-done/workflows/execute-plan.md
@/home/cohenn1/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/visualize_results.py
@nce/benchmark_problems/small_problems.py

<interfaces>
<!-- Result JSON structure (from results/{config_name}/{modelfile}.json): -->
```json
{
  "config_name": "wmse_bw0",
  "modelfile": "10_14_s.binary.uai",
  "loss_fn": "weighted_logspace_mse",
  "bw_ecl": 0,
  "ecl": 32767,
  "num_epochs": 5000,
  "hidden_sizes": [3, 3],
  "log_z_estimate": -35.279,
  "num_buckets_trained": 3,
  "duration_seconds": 317.759,
  "status": "completed"
}
```

<!-- pyGMs Model attributes available: -->
```python
model.modelfile  # e.g. "smokers_20.uai"
model.PR         # ground truth log Z
model.width      # treewidth / induced width
model.num_vars   # number of variables
```

<!-- small_problems module: -->
```python
from nce.benchmark_problems.small_problems import small_problems, _AUTO_ECL
# small_problems.problems: list of 24 Model objects
# _AUTO_ECL: dict mapping modelfile -> ecl value
```

<!-- Config name to label/color mapping (user-specified): -->
```
wmse_bw0   -> 'WMSE (no bw)'        -> Red
ukl_bw0    -> 'UKL (no bw)'         -> Orange
ukl_bw8    -> 'UKL (bw ib=3)'       -> Yellow
ukl_bw_ecl -> 'UKL (bw ib=fw ib)'   -> Green
ukl_bw30   -> 'UKL (bw=exact)'      -> Blue
Ground Truth -> Black dashed line at y=0
```

<!-- 12 "clean" problems have all 5 configs complete; 21/24 have 4 configs; ukl_bw30 only has 12 -->
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create visualize_updated.py with per-problem absolute error plots and summary CSV</name>
  <files>notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/visualize_updated.py</files>
  <action>
Create a standalone Python script at notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/visualize_updated.py that:

**Setup:**
- sys.path.insert(0, '/home/cohenn1/NCE')
- Import from nce.benchmark_problems.small_problems: small_problems, _AUTO_ECL
- RESULTS_DIR = Path(__file__).parent / 'results'
- OUTPUT_DIR = RESULTS_DIR / 'updated_graphs_and_table'
- Create OUTPUT_DIR if not exists
- CONFIGS list: ['wmse_bw0', 'ukl_bw0', 'ukl_bw8', 'ukl_bw_ecl', 'ukl_bw30']

**Config label and color mappings (exact user specification):**
```python
CONFIG_LABELS = {
    'wmse_bw0': 'WMSE (no bw)',
    'ukl_bw0': 'UKL (no bw)',
    'ukl_bw8': 'UKL (bw ib=3)',
    'ukl_bw_ecl': 'UKL (bw ib=fw ib)',
    'ukl_bw30': 'UKL (bw=exact)',
}
CONFIG_COLORS = {
    'wmse_bw0': 'red',
    'ukl_bw0': 'orange',
    'ukl_bw8': 'gold',       # Yellow-ish that's visible
    'ukl_bw_ecl': 'green',
    'ukl_bw30': 'blue',
}
```

**Data loading:**
- Load all completed result JSONs from results/{config}/*.json (same pattern as visualize_results.py)
- Build ground truth dict: {modelfile: model.PR for model in small_problems.problems}
- Build width dict: {modelfile: model.width for model in small_problems.problems}
- Build num_vars dict: {modelfile: model.num_vars for model in small_problems.problems}
- Identify the 12 clean problems (all 5 configs completed) - sorted by modelfile

**Per-problem absolute error plots (2 plots per clean problem):**

For each of the 12 clean problems, create TWO figures:

1. **Linear scale plot:**
   - Grouped bar chart with 5 bars (one per config) + ground truth line
   - Y-axis = absolute error = |log_z_estimate - model.PR|
   - X-axis labels = CONFIG_LABELS values
   - Colors per CONFIG_COLORS
   - Add a black dashed horizontal line at y=0 labeled 'Ground Truth'
   - Title: "{problem_name}, iB_2 = {round(math.log2(ecl))}, num_trained={N}, num_epochs=5000\nNeuroBE loss and varied bw iB UKL Comparison"
     where problem_name = modelfile without .uai, ecl from _AUTO_ECL, N from result's num_buckets_trained (use wmse_bw0's value since all configs share same ecl)
   - figsize=(10, 6), tight_layout
   - Save as: OUTPUT_DIR / f"{problem_name}_abs_error_linear.png" (problem_name = modelfile.replace('.uai',''))

2. **Log scale plot:**
   - Same as linear but y-axis uses matplotlib's symlog scale with linthresh=1e-3
   - Use ax.set_yscale('symlog', linthresh=1e-3)
   - Y tick labels should show values like 0, 0.001, 0.01, 0.1, 1, 10, 100 etc.
   - Save as: OUTPUT_DIR / f"{problem_name}_abs_error_log.png"

For BOTH plots: use bar chart (not grouped across problems -- each problem is its own figure with 5 bars side by side). Use 150 dpi. Add grid on y-axis with alpha=0.3.

**Summary CSV:**

Build a DataFrame with one row per completed experiment (all configs, all problems -- not just clean 12). Columns:
- problem_name: modelfile without .uai extension
- width: from model.width
- num_vars: from model.num_vars
- log_Z_ground_truth: model.PR
- log_Z_hat: result JSON log_z_estimate
- err: log_Z_hat - log_Z_ground_truth
- abs_err: |err|
- num_trained: result JSON num_buckets_trained
- time: result JSON duration_seconds
- num_samples: 100000 (hardcoded -- all experiments use sampling_scheme='all' with num_samples=100000)
- architecture: str(result JSON hidden_sizes) -- e.g. "[3, 3]"
- loss_fn: result JSON loss_fn
- config_name: result JSON config_name (extra column for filtering)

Sort by problem_name then config_name. Save to OUTPUT_DIR / 'summary.csv' with index=False.

**Config analysis (track_errors answer):**

At the end of the script, print:
```
CONFIG ANALYSIS: track_errors
The default small_problems config has track_errors=False.
The benchmark's build_experiment_config() does not change track_errors.
Therefore, NO experiments in this benchmark used error tracking.
```

**Close all figures** with plt.close('all') after saving each to avoid memory issues with 24 plots.
  </action>
  <verify>
    <automated>cd /home/cohenn1/NCE && venv/bin/python notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/visualize_updated.py && ls -la notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/results/updated_graphs_and_table/*.png | wc -l && ls -la notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/results/updated_graphs_and_table/summary.csv && head -3 notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/results/updated_graphs_and_table/summary.csv</automated>
  </verify>
  <done>
    - 24 PNG files exist in updated_graphs_and_table/ (12 problems x 2 scales)
    - summary.csv exists with correct columns and one row per completed experiment
    - Script runs without errors and prints track_errors analysis
    - Colors match specification: Red/Orange/Yellow/Green/Blue
    - Labels match specification exactly
    - Title format includes iB_2, num_trained, num_epochs
  </done>
</task>

</tasks>

<verification>
- Run the script: `cd /home/cohenn1/NCE && venv/bin/python notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/visualize_updated.py`
- Verify 24 PNG files produced (12 clean problems x 2 scale types)
- Verify summary.csv has correct columns and row count (should be ~96 rows: 21*4 + 12*1 for the 5 configs with varying completion)
- Spot check a PNG to confirm colors and labels match spec
</verification>

<success_criteria>
- visualize_updated.py runs end-to-end without errors
- 24 PNG files saved to results/updated_graphs_and_table/
- summary.csv saved with all required columns, one row per completed experiment
- Config analysis about track_errors printed to stdout
</success_criteria>

<output>
After completion, create `.planning/quick/20-update-benchmark-wmse-ukl-graphs-with-pe/20-SUMMARY.md`
</output>
