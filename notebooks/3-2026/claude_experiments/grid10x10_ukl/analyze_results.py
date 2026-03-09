#!/usr/bin/env python3
"""Analysis script: load pickle and produce graphs and tables.

Loads the experiment pickle produced by run_grid10x10_ukl.py and generates:
  1. Summary table (printed to console): all CSV columns formatted nicely
  2. Bar chart: log_Z_star vs log_Z_hat side-by-side (results/logz_comparison.png)
  3. Per-bucket epochs histogram (results/bucket_epochs_histogram.png)
  4. Per-bucket hidden sizes table (printed to console)
  5. Regenerated CSV from pickle (proves round-trip completeness)

Usage:
    # Default (looks for results/experiment_results.pkl):
    /home/cohenn1/NCE/venv/bin/python analyze_results.py

    # Custom pickle path:
    /home/cohenn1/NCE/venv/bin/python analyze_results.py --pickle-path /path/to/pkl

Notes for future Claude Code agents:
    - This script intentionally does NOT depend on nce imports -- it works
      from the pickle alone, proving the pickle is self-contained.
    - All plots are saved to the results/ directory alongside the pickle.
    - The regenerated CSV should match the original exactly (same values).
"""

import argparse
import csv
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, '/home/cohenn1/NCE')

SCRIPT_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = SCRIPT_DIR / 'results'


def load_pickle(pickle_path):
    """Load experiment results pickle.

    Args:
        pickle_path: Path to the pickle file.

    Returns:
        Results dict with keys: config, model_name, model_width, model_num_vars,
        model_PR, log_z_hat, log_Z_star, err, abs_err, num_trained,
        duration_seconds, per_bucket_data, timestamp.
    """
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)


def print_summary_table(results):
    """Print a formatted summary table of all CSV columns."""
    print("=" * 70)
    print("EXPERIMENT SUMMARY TABLE")
    print("=" * 70)
    fields = [
        ('problem_name',  results['model_name']),
        ('width',         results['model_width']),
        ('num_vars',      results['model_num_vars']),
        ('log_Z_star',    f"{results['log_Z_star']:.6f}"),
        ('log_Z_hat',     f"{results['log_z_hat']:.6f}" if results['log_z_hat'] is not None else 'N/A'),
        ('err',           f"{results['err']:.6f}" if results['err'] is not None else 'N/A'),
        ('abs_err',       f"{results['abs_err']:.6f}" if results['abs_err'] is not None else 'N/A'),
        ('num_trained',   results['num_trained']),
        ('time_seconds',  f"{results['duration_seconds']:.1f}"),
        ('num_samples',   results['config']['num_samples']),
        ('architecture',  str(results['config']['hidden_sizes'])),
        ('loss_fn',       results['config']['loss_fn']),
    ]
    col_w = max(len(k) for k, _ in fields) + 2
    for key, val in fields:
        print(f"  {key:<{col_w}}: {val}")
    print()


def print_bucket_table(results):
    """Print a table of per-bucket hidden sizes and epochs trained."""
    per_bucket = results.get('per_bucket_data', [])
    if not per_bucket:
        print("No per-bucket data found in pickle.")
        return

    print("=" * 70)
    print(f"PER-BUCKET TRAINING DATA ({len(per_bucket)} NN buckets)")
    print("=" * 70)
    print(f"  {'Bucket Label':>12s}  {'Epochs Trained':>14s}  {'Hidden Sizes'}")
    print("  " + "-" * 60)
    for bd in sorted(per_bucket, key=lambda x: x['label']):
        print(f"  {bd['label']:>12d}  {bd['epochs_trained']:>14d}  {bd['hidden_sizes']}")
    print()

    # Stats
    epochs_list = [bd['epochs_trained'] for bd in per_bucket]
    if epochs_list:
        print(f"  Epochs summary:")
        print(f"    min    = {min(epochs_list)}")
        print(f"    max    = {max(epochs_list)}")
        print(f"    mean   = {sum(epochs_list)/len(epochs_list):.1f}")
        print(f"    median = {sorted(epochs_list)[len(epochs_list)//2]}")
    print()


def plot_logz_comparison(results, output_path):
    """Bar chart: log_Z_star vs log_Z_hat side by side with value labels.

    Args:
        results: Experiment results dict.
        output_path: Path to save the PNG.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    log_z_star = results['log_Z_star']
    log_z_hat = results['log_z_hat']
    model_name = results['model_name']
    err = results['err']

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        ['log_Z_star\n(exact)', 'log_Z_hat\n(NCE estimate)'],
        [log_z_star, log_z_hat],
        color=['steelblue', 'darkorange'],
        width=0.5,
        edgecolor='black',
        linewidth=0.8,
    )

    # Value labels on bars
    for bar, val in zip(bars, [log_z_star, log_z_hat]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold',
        )

    err_str = f'{err:+.4f}' if err is not None else 'N/A'
    ax.set_title(
        f'log Z comparison: {model_name}\nerr (hat - star) = {err_str} (log10)',
        fontsize=12, pad=12,
    )
    ax.set_ylabel('log Z (log10 space)', fontsize=11)
    ax.set_ylim(min(log_z_star, log_z_hat) - 1, max(log_z_star, log_z_hat) + 2)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {output_path}")


def plot_bucket_epochs_histogram(results, output_path):
    """Histogram of epochs_trained values across all NN buckets.

    Args:
        results: Experiment results dict.
        output_path: Path to save the PNG.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    per_bucket = results.get('per_bucket_data', [])
    if not per_bucket:
        print("No per-bucket data; skipping histogram.")
        return

    epochs_list = [bd['epochs_trained'] for bd in per_bucket]
    model_name = results['model_name']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(epochs_list, bins=20, color='steelblue', edgecolor='black', linewidth=0.7)
    ax.set_xlabel('Epochs Trained', fontsize=11)
    ax.set_ylabel('Number of NN Buckets', fontsize=11)
    ax.set_title(
        f'Per-bucket epochs trained: {model_name}\n'
        f'({len(per_bucket)} NN buckets, max={max(epochs_list)}, '
        f'mean={sum(epochs_list)/len(epochs_list):.1f})',
        fontsize=12, pad=12,
    )
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {output_path}")


def regenerate_csv(results, csv_path):
    """Regenerate the CSV from the pickle (round-trip proof).

    Args:
        results: Experiment results dict.
        csv_path: Path to write the regenerated CSV.
    """
    fieldnames = [
        'problem_name', 'width', 'num_vars',
        'log_Z_star', 'log_Z_hat', 'err', 'abs_err',
        'num_trained', 'time_seconds', 'num_samples',
        'architecture', 'loss_fn',
    ]
    row = {
        'problem_name': results['model_name'],
        'width': results['model_width'],
        'num_vars': results['model_num_vars'],
        'log_Z_star': results['log_Z_star'],
        'log_Z_hat': results['log_z_hat'],
        'err': results['err'],
        'abs_err': results['abs_err'],
        'num_trained': results['num_trained'],
        'time_seconds': results['duration_seconds'],
        'num_samples': results['config']['num_samples'],
        'architecture': str(results['config']['hidden_sizes']),
        'loss_fn': results['config']['loss_fn'],
    }
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    print(f"CSV regenerated from pickle: {csv_path}")
    print("(Regenerated CSV proves pickle contains all needed data)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze grid10x10.f10 UKL experiment results from pickle.'
    )
    parser.add_argument(
        '--pickle-path', type=str,
        default=str(RESULTS_DIR / 'experiment_results.pkl'),
        help='Path to the experiment pickle file (default: results/experiment_results.pkl)',
    )
    args = parser.parse_args()

    pickle_path = Path(args.pickle_path)
    if not pickle_path.exists():
        print(f"ERROR: Pickle not found: {pickle_path}", file=sys.stderr)
        print("       Run run_grid10x10_ukl.py first to generate results.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading pickle: {pickle_path}")
    results = load_pickle(pickle_path)
    print(f"Timestamp: {results.get('timestamp', 'unknown')}")
    print()

    # 1. Summary table
    print_summary_table(results)

    # 2. Log Z bar chart
    logz_plot_path = RESULTS_DIR / 'logz_comparison.png'
    try:
        plot_logz_comparison(results, logz_plot_path)
    except Exception as e:
        print(f"Warning: log Z plot failed: {e}")

    # 3. Per-bucket epochs histogram
    hist_plot_path = RESULTS_DIR / 'bucket_epochs_histogram.png'
    try:
        plot_bucket_epochs_histogram(results, hist_plot_path)
    except Exception as e:
        print(f"Warning: histogram plot failed: {e}")

    print()

    # 4. Per-bucket hidden sizes table
    print_bucket_table(results)

    # 5. Regenerate CSV from pickle (round-trip test)
    regen_csv_path = RESULTS_DIR / 'grid10x10_f10_ukl_results_from_pickle.csv'
    regenerate_csv(results, regen_csv_path)

    print("Analysis complete.")
    print(f"All outputs in: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
