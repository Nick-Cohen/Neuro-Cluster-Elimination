"""
Visualization script for WMSE vs UKL benchmark results.

Produces:
  1. Per-problem grouped bar charts (linear + log scale) for the 12 "clean"
     problems (all 5 configs completed) saved as PNG files.
  2. A summary CSV table covering all completed experiments (all configs,
     all problems).
  3. A printed answer about whether track_errors was used in the experiments.

Output location: results/updated_graphs_and_table/
"""

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/cohenn1/NCE')
from nce.benchmark_problems.small_problems import small_problems, _AUTO_ECL

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent / 'results'
OUTPUT_DIR = RESULTS_DIR / 'updated_graphs_and_table'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = ['wmse_bw0', 'ukl_bw0', 'ukl_bw8', 'ukl_bw_ecl', 'ukl_bw30']

# ---------------------------------------------------------------------------
# Config label and color mappings (exact user specification)
# ---------------------------------------------------------------------------
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
    'ukl_bw8': 'gold',        # Yellow-ish that's visible
    'ukl_bw_ecl': 'green',
    'ukl_bw30': 'blue',
}

# ---------------------------------------------------------------------------
# Ground truth and model metadata from small_problems
# ---------------------------------------------------------------------------
GROUND_TRUTH = {m.modelfile: m.PR for m in small_problems.problems}
MODEL_WIDTH = {m.modelfile: m.width for m in small_problems.problems}
MODEL_NUM_VARS = {m.modelfile: m.num_vars for m in small_problems.problems}

# ---------------------------------------------------------------------------
# Load all completed results
# ---------------------------------------------------------------------------
rows = []
for cfg in CONFIGS:
    cfg_dir = RESULTS_DIR / cfg
    if not cfg_dir.exists():
        print(f"WARNING: Config directory not found: {cfg_dir}")
        continue
    for f in cfg_dir.glob('*.json'):
        with open(f) as fh:
            d = json.load(fh)
        if d.get('status') == 'completed':
            rows.append(d)

df_all = pd.DataFrame(rows)
print(f"Loaded {len(df_all)} completed results")
print(f"Configs present: {sorted(df_all['config_name'].unique())}")
print(f"Problems present: {df_all['modelfile'].nunique()}")

# ---------------------------------------------------------------------------
# Identify the 12 clean problems (all 5 configs completed)
# ---------------------------------------------------------------------------
problems_per_config = df_all.groupby('config_name')['modelfile'].apply(set)
clean_problems = sorted(set.intersection(*problems_per_config.values))
print(f"\n{len(clean_problems)} problems with all 5 configs complete:")
for p in clean_problems:
    print(f"  {p}")

df_clean = df_all[df_all['modelfile'].isin(clean_problems)].copy()

# Pivot for log_z estimates on clean problems
pivot_logz = df_clean.pivot(index='modelfile', columns='config_name', values='log_z_estimate')
pivot_logz = pivot_logz[CONFIGS]

# ---------------------------------------------------------------------------
# Per-problem absolute error plots (2 plots per clean problem)
# ---------------------------------------------------------------------------
print(f"\nGenerating per-problem plots for {len(clean_problems)} clean problems ...")

for problem in clean_problems:
    problem_name = problem.replace('.uai', '')
    ecl = _AUTO_ECL.get(problem, 0)
    iB_2 = round(math.log2(ecl)) if ecl > 0 else 0

    # Get num_trained from wmse_bw0 (all configs share same ecl)
    wmse_row = df_clean[(df_clean['modelfile'] == problem) & (df_clean['config_name'] == 'wmse_bw0')]
    N = int(wmse_row['num_buckets_trained'].iloc[0]) if len(wmse_row) > 0 else '?'

    gt = GROUND_TRUTH.get(problem, None)
    if gt is None:
        print(f"  WARNING: No ground truth for {problem}, skipping")
        continue

    # Compute absolute errors per config
    abs_errors = []
    bar_labels = []
    bar_colors = []
    for cfg in CONFIGS:
        if problem in pivot_logz.index and cfg in pivot_logz.columns:
            logz_hat = pivot_logz.loc[problem, cfg]
            abs_err = abs(logz_hat - gt)
        else:
            abs_err = 0.0
        abs_errors.append(abs_err)
        bar_labels.append(CONFIG_LABELS[cfg])
        bar_colors.append(CONFIG_COLORS[cfg])

    x = np.arange(len(CONFIGS))
    title = (
        f"{problem_name}, iB_2 = {iB_2}, num_trained={N}, num_epochs=5000\n"
        f"NeuroBE loss and varied bw iB UKL Comparison"
    )

    # --- Linear scale plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, abs_errors, color=bar_colors, alpha=0.85)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.2, label='Ground Truth')
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=10)
    ax.set_ylabel('Absolute Error |log Z estimate - log Z true|')
    ax.set_xlabel('Configuration')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    out_path = OUTPUT_DIR / f"{problem_name}_abs_error_linear.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close('all')

    # --- Log (symlog) scale plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, abs_errors, color=bar_colors, alpha=0.85)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.2, label='Ground Truth')
    ax.set_yscale('symlog', linthresh=1e-3)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=10)
    ax.set_ylabel('Absolute Error |log Z estimate - log Z true| (symlog scale)')
    ax.set_xlabel('Configuration')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    out_path = OUTPUT_DIR / f"{problem_name}_abs_error_log.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close('all')

    print(f"  Saved plots for {problem_name}")

print(f"\nDone generating plots. Total PNG files:")
png_count = len(list(OUTPUT_DIR.glob('*.png')))
print(f"  {png_count} PNG files in {OUTPUT_DIR}")

# ---------------------------------------------------------------------------
# Pattern-1 problems: only ukl_bw30 failed, 4 other configs completed
# Generate per-problem plots in no_exact_bw/ subfolder
# ---------------------------------------------------------------------------
NO_BW_DIR = OUTPUT_DIR / 'no_exact_bw'
NO_BW_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS_NO_BW30 = ['wmse_bw0', 'ukl_bw0', 'ukl_bw8', 'ukl_bw_ecl']
all_modelfiles = [m.modelfile for m in small_problems.problems]

# Find problems with exactly the 4 non-bw30 configs completed but bw30 missing/failed
pattern1_problems = []
for mf in sorted(all_modelfiles):
    if mf in clean_problems:
        continue  # already graphed
    completed_cfgs = set(
        df_all[(df_all['modelfile'] == mf)]['config_name'].values
    )
    has_all_4 = all(c in completed_cfgs for c in CONFIGS_NO_BW30)
    if has_all_4:
        pattern1_problems.append(mf)

print(f"\n{len(pattern1_problems)} pattern-1 problems (4 configs, no exact bw):")
for p in pattern1_problems:
    print(f"  {p}")

df_p1 = df_all[df_all['modelfile'].isin(pattern1_problems)].copy()
pivot_p1 = df_p1.pivot(index='modelfile', columns='config_name', values='log_z_estimate')

for problem in pattern1_problems:
    problem_name = problem.replace('.uai', '')
    ecl = _AUTO_ECL.get(problem, 0)
    iB_2 = round(math.log2(ecl)) if ecl > 0 else 0

    wmse_row = df_p1[(df_p1['modelfile'] == problem) & (df_p1['config_name'] == 'wmse_bw0')]
    N = int(wmse_row['num_buckets_trained'].iloc[0]) if len(wmse_row) > 0 else '?'

    gt = GROUND_TRUTH.get(problem, None)
    if gt is None:
        continue

    abs_errors = []
    bar_labels = []
    bar_colors = []
    for cfg in CONFIGS_NO_BW30:
        if problem in pivot_p1.index and cfg in pivot_p1.columns:
            logz_hat = pivot_p1.loc[problem, cfg]
            abs_err = abs(logz_hat - gt)
        else:
            abs_err = 0.0
        abs_errors.append(abs_err)
        bar_labels.append(CONFIG_LABELS[cfg])
        bar_colors.append(CONFIG_COLORS[cfg])

    x = np.arange(len(CONFIGS_NO_BW30))
    title = (
        f"{problem_name}, iB_2 = {iB_2}, num_trained={N}, num_epochs=5000\n"
        f"NeuroBE loss and varied bw iB UKL Comparison (no exact bw)"
    )

    # --- Linear scale plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, abs_errors, color=bar_colors, alpha=0.85)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.2, label='Ground Truth')
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=10)
    ax.set_ylabel('Absolute Error |log Z estimate - log Z true|')
    ax.set_xlabel('Configuration')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(NO_BW_DIR / f"{problem_name}_abs_error_linear.png"), dpi=150)
    plt.close('all')

    # --- Log (symlog) scale plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, abs_errors, color=bar_colors, alpha=0.85)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.2, label='Ground Truth')
    ax.set_yscale('symlog', linthresh=1e-3)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=10)
    ax.set_ylabel('Absolute Error |log Z estimate - log Z true| (symlog scale)')
    ax.set_xlabel('Configuration')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(NO_BW_DIR / f"{problem_name}_abs_error_log.png"), dpi=150)
    plt.close('all')

    print(f"  Saved plots for {problem_name}")

p1_png_count = len(list(NO_BW_DIR.glob('*.png')))
print(f"\n  {p1_png_count} PNG files in {NO_BW_DIR}")

# ---------------------------------------------------------------------------
# Summary CSV (all completed experiments, not just clean 12)
# ---------------------------------------------------------------------------
print("\nBuilding summary CSV ...")

summary_rows = []
for _, row in df_all.iterrows():
    modelfile = row['modelfile']
    logz_hat = row['log_z_estimate']
    gt = GROUND_TRUTH.get(modelfile, None)
    err = (logz_hat - gt) if gt is not None else None
    abs_err = abs(err) if err is not None else None
    summary_rows.append({
        'problem_name': modelfile.replace('.uai', ''),
        'width': MODEL_WIDTH.get(modelfile, None),
        'num_vars': MODEL_NUM_VARS.get(modelfile, None),
        'log_Z_ground_truth': gt,
        'log_Z_hat': logz_hat,
        'err': err,
        'abs_err': abs_err,
        'num_trained': row.get('num_buckets_trained', None),
        'time': row.get('duration_seconds', None),
        'num_samples': 100000,  # all experiments use sampling_scheme='all', num_samples=100000
        'architecture': str(row.get('hidden_sizes', [])),
        'loss_fn': row.get('loss_fn', None),
        'config_name': row.get('config_name', None),
    })

df_summary = pd.DataFrame(summary_rows)
df_summary = df_summary.sort_values(['problem_name', 'config_name']).reset_index(drop=True)

csv_path = OUTPUT_DIR / 'summary.csv'
df_summary.to_csv(str(csv_path), index=False)
print(f"Saved summary CSV: {csv_path}")
print(f"  Rows: {len(df_summary)}")
print(f"  Columns: {list(df_summary.columns)}")
print(f"\nFirst 3 rows:")
print(df_summary.head(3).to_string())

# ---------------------------------------------------------------------------
# Config analysis: track_errors
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("CONFIG ANALYSIS: track_errors")
print("=" * 60)
print("The default small_problems config has track_errors=False.")
print("The benchmark's build_experiment_config() does not change track_errors.")
print("Therefore, NO experiments in this benchmark used error tracking.")
