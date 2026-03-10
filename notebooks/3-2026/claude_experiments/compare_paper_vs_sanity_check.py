"""
Grouped bar chart comparing NeuroBE paper results vs our WMB sanity check results
for the 4 overlapping problems.

Paper data: NCE-Data/NeuroBE_paper_results.csv
Our results:
  - nbe_full_epochs.txt (ecl=2^(iB-1), 0 NN buckets = pure WMB)
  - Phase 3 (ecl=2^22, essentially exact WMB)

Name mapping (paper name -> our name):
  pedigree13  (i-bound=20, id=1)  -> pedigree13
  grid4040f10 (i-bound=20, id=1)  -> grid40x40.f10
  grid2020f10 (i-bound=10, id=4)  -> grid20x20.f10
  rbm20       (i-bound=20, id=1)  -> rbm_20

Error computation: |our_logZ - refZ| where refZ comes from the paper CSV.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ---------------------------------------------------------------------------
# Our WMB results (from nbe_full_epochs.txt, ecl=2^(iB-1), 0 NN buckets)
# These are pure WMB results - no neural network approximation was used.
# ---------------------------------------------------------------------------
our_wmb_logz = {
    'pedigree13':    -31.764139,
    'grid40x40.f10':  5349.348633,
    'grid20x20.f10':  1197.311035,
    'rbm_20':           57.987244,
}

# Phase 3 results (ecl=2^22, essentially exact WMB)
# grid40x40.f10 errored (network timeout), so it is excluded.
our_exact_wmb_logz = {
    'pedigree13':    -23.4234,
    'grid40x40.f10':  None,   # errored
    'grid20x20.f10':  1332.5142,
    'rbm_20':           58.5306,
}

# ---------------------------------------------------------------------------
# Name mapping: paper CSV name -> our name -> (section, id) to select row
# ---------------------------------------------------------------------------
MATCHING = [
    {'paper_name': 'pedigree13',  'our_name': 'pedigree13',    'section': 'i-bound=20', 'id': 1},
    {'paper_name': 'grid4040f10', 'our_name': 'grid40x40.f10', 'section': 'i-bound=20', 'id': 1},
    {'paper_name': 'grid2020f10', 'our_name': 'grid20x20.f10', 'section': 'i-bound=10', 'id': 4},
    {'paper_name': 'rbm20',       'our_name': 'rbm_20',        'section': 'i-bound=20', 'id': 1},
]

# ---------------------------------------------------------------------------
# Load paper CSV
# ---------------------------------------------------------------------------
csv_path = '/home/cohenn1/NCE/NCE-Data/NeuroBE_paper_results.csv'
df = pd.read_csv(csv_path)

# ---------------------------------------------------------------------------
# Extract matching rows and compute errors
# ---------------------------------------------------------------------------
records = []
for m in MATCHING:
    row = df[(df['name'] == m['paper_name']) &
             (df['section'] == m['section']) &
             (df['id'] == m['id'])].iloc[0]

    ref_z         = float(row['refZ'])
    paper_wmb_err = float(row['WMB_error'])
    paper_nbe_avg = float(row['NeuroBE_avg_error'])
    paper_nbe_min = float(row['NeuroBE_min_error'])

    our_logz  = our_wmb_logz[m['our_name']]
    our_err   = abs(our_logz - ref_z) if our_logz is not None else None

    exact_logz = our_exact_wmb_logz[m['our_name']]
    exact_err  = abs(exact_logz - ref_z) if exact_logz is not None else None

    records.append({
        'our_name':      m['our_name'],
        'paper_name':    m['paper_name'],
        'ref_z':         ref_z,
        'paper_wmb_err': paper_wmb_err,
        'paper_nbe_avg': paper_nbe_avg,
        'paper_nbe_min': paper_nbe_min,
        'our_wmb_logz':  our_logz,
        'our_wmb_err':   our_err,
        'our_exact_logz': exact_logz,
        'our_exact_err': exact_err,
    })

# ---------------------------------------------------------------------------
# Print summary table to stdout
# ---------------------------------------------------------------------------
header = (f"{'Problem':<20} {'refZ':>10} {'PaperWMB':>10} "
          f"{'PaperNBE avg':>13} {'PaperNBE min':>13} "
          f"{'OurWMB err':>12} {'OurExact err':>13}")
sep = '-' * len(header)
print()
print("=" * len(header))
print("Summary: Absolute Error in log Z (|computed - refZ|)")
print("=" * len(header))
print(header)
print(sep)
for r in records:
    our_e  = f"{r['our_wmb_err']:.4f}"   if r['our_wmb_err']  is not None else 'N/A'
    ex_e   = f"{r['our_exact_err']:.4f}" if r['our_exact_err'] is not None else 'N/A (errored)'
    print(f"{r['our_name']:<20} {r['ref_z']:>10.3f} {r['paper_wmb_err']:>10.4f} "
          f"{r['paper_nbe_avg']:>13.4f} {r['paper_nbe_min']:>13.4f} "
          f"{our_e:>12} {ex_e:>13}")
print(sep)
print()

# ---------------------------------------------------------------------------
# Manual verification spot-check
# pedigree13: |(-31.764139) - (-31.18)| = 0.584139
# ---------------------------------------------------------------------------
expected_pedigree13 = abs(-31.764139 - (-31.18))
actual_pedigree13   = records[0]['our_wmb_err']
assert abs(expected_pedigree13 - actual_pedigree13) < 1e-5, \
    f"Spot-check failed: expected {expected_pedigree13}, got {actual_pedigree13}"
print(f"Spot-check passed: pedigree13 our WMB error = {actual_pedigree13:.6f}")
print()

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
problems_display = [r['our_name'] for r in records]
n_problems = len(records)

# 5 bar groups per problem
bar_labels = [
    'Paper WMB',
    'Paper NeuroBE avg',
    'Paper NeuroBE min',
    'Our WMB (ecl=2^(iB-1))',
    'Our WMB (ecl=2^22)',
]
bar_colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336']
n_bars = len(bar_labels)

fig, (ax_chart, ax_table) = plt.subplots(
    1, 2,
    figsize=(18, 7),
    gridspec_kw={'width_ratios': [2, 1]}
)

x = np.arange(n_problems)
total_width = 0.75
width = total_width / n_bars
offsets = np.linspace(-(total_width - width) / 2, (total_width - width) / 2, n_bars)

def get_bar_values(r, idx):
    vals = [
        r['paper_wmb_err'],
        r['paper_nbe_avg'],
        r['paper_nbe_min'],
        r['our_wmb_err'],
        r['our_exact_err'],
    ]
    return vals[idx]

bars_all = []
for i, (label, color, offset) in enumerate(zip(bar_labels, bar_colors, offsets)):
    values = []
    for r in records:
        v = get_bar_values(r, i)
        values.append(v if v is not None else np.nan)
    bars = ax_chart.bar(
        x + offset, values, width,
        label=label,
        color=color,
        edgecolor='black',
        linewidth=0.5,
    )
    bars_all.append(bars)

    # Value labels on bars
    for bar, val in zip(bars, values):
        if np.isnan(val):
            # Mark missing bars
            ax_chart.text(
                bar.get_x() + bar.get_width() / 2,
                ax_chart.get_ylim()[0] * 1.05 if ax_chart.get_yscale() == 'log' else 0.05,
                'err',
                ha='center', va='bottom', fontsize=6, color='gray',
                rotation=90
            )
            continue
        height = bar.get_height()
        ax_chart.annotate(
            f'{val:.3f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 2),
            textcoords='offset points',
            ha='center', va='bottom',
            fontsize=6,
            rotation=90
        )

ax_chart.set_yscale('log')
ax_chart.set_xlabel('Problem', fontsize=11)
ax_chart.set_ylabel('Absolute Error in log Z (log scale)', fontsize=11)
ax_chart.set_title('Paper vs Our Results: Absolute Error in log Z', fontsize=12, fontweight='bold')
ax_chart.set_xticks(x)
ax_chart.set_xticklabels(problems_display, rotation=25, ha='right', fontsize=9)
ax_chart.legend(loc='upper left', fontsize=8)
ax_chart.grid(axis='y', alpha=0.3)

# ---------------------------------------------------------------------------
# Right subplot: summary table
# ---------------------------------------------------------------------------
ax_table.axis('off')

col_labels = ['Problem', 'refZ', 'PaperWMB', 'PaperNBE avg', 'PaperNBE min',
              'OurWMB\n(ecl=2^(iB-1))', 'OurWMB\n(ecl=2^22)']

table_data = []
for r in records:
    our_e   = f"{r['our_wmb_err']:.3f}"   if r['our_wmb_err']  is not None else 'N/A'
    ex_e    = f"{r['our_exact_err']:.3f}" if r['our_exact_err'] is not None else 'err'
    table_data.append([
        r['our_name'],
        f"{r['ref_z']:.2f}",
        f"{r['paper_wmb_err']:.3f}",
        f"{r['paper_nbe_avg']:.3f}",
        f"{r['paper_nbe_min']:.3f}",
        our_e,
        ex_e,
    ])

tbl = ax_table.table(
    cellText=table_data,
    colLabels=col_labels,
    loc='center',
    cellLoc='center',
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.1, 1.8)

ax_table.set_title('Raw Values', fontsize=11, fontweight='bold', pad=10)

plt.tight_layout(rect=[0, 0, 1, 1])

out_path = '/home/cohenn1/NCE/notebooks/3-2026/claude_experiments/compare_paper_vs_sanity_check.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved chart: {out_path}")
plt.close()
