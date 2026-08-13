#!/usr/bin/env python3
"""Combine benchmark experiment results for comparison.

Usage:
    python combine_experiments.py <path_to_yaml>

Example YAML:
    experiment_name: ukl_vs_nbe
    paths: /path/to/run1, /path/to/run2
    names: UKL, NeuroBE
"""
import argparse
import sys
import os
os.chdir('/home/cohenn1/NCE')
sys.path.insert(0, '/home/cohenn1/NCE')

import json
import math
import shutil
import time
import yaml
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(experiment_dir):
    """Load all per-bucket metrics.json files from an experiment directory.

    Returns dict of bucket_id -> metrics dict.
    """
    metrics = {}
    exp_path = Path(experiment_dir)
    for metrics_file in exp_path.rglob('metrics.json'):
        with open(metrics_file) as f:
            data = json.load(f)
        # Derive bucket_id from the directory name
        bucket_id = metrics_file.parent.name
        metrics[bucket_id] = data
    return metrics


def is_valid_result(metrics):
    """Check if a result completed without errors (no NaN values)."""
    error_tracking = metrics.get('error_tracking', [])
    if not error_tracking:
        return False
    for entry in error_tracking:
        for val in entry:
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                return False
    final_loss = metrics.get('final_loss')
    if final_loss is not None and (math.isnan(final_loss) or math.isinf(final_loss)):
        return False
    return True


def copy_plots(src_dir, dst_dir, name, bucket_id):
    """Copy loss and local_error plots with prefixed names."""
    src = Path(src_dir)
    # Find the bucket's output dir (may be nested: bucket_id/bucket_id/)
    candidates = list(src.rglob(f'{bucket_id}/loss.png'))
    if not candidates:
        return

    bucket_src = candidates[0].parent

    for plot_name in ['loss.png', 'local_error.png', 'approximation.png']:
        src_file = bucket_src / plot_name
        if src_file.exists():
            base = plot_name.replace('.png', '')
            dst_file = Path(dst_dir) / f'{name}_{base}.png'
            shutil.copy2(src_file, dst_file)


def plot_combined_local_errors(all_series, output_path, title,
                               time_limit_minutes=60, early_stopping=False):
    """Plot combined local error curves with time-scaled x-axis.

    Args:
        all_series: list of (name, error_tracking_data) where error_tracking_data
                    is list of [epoch, loss, log_z_err, abs_log_z_err]
        output_path: where to save the PNG
        title: plot title
        time_limit_minutes: what the final epoch maps to in minutes
            (ignored when early_stopping=True)
        early_stopping: if True, use the global max epoch across all series
            as the x-axis limit instead of stretching each series to
            time_limit_minutes. Series that stopped early will end shorter.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if early_stopping:
        # Find global max epoch across all series
        global_max_epoch = max(
            (entry[0] for _, data in all_series for entry in data if data),
            default=1,
        )

    for name, data in all_series:
        if not data:
            continue
        epochs = [entry[0] for entry in data]
        abs_errors = [entry[3] for entry in data]

        max_epoch = epochs[-1]
        if max_epoch == 0:
            continue

        if early_stopping:
            # Scale by global max so all series share the same x-axis scale
            times = [e * (time_limit_minutes / global_max_epoch) for e in epochs]
        else:
            # Stretch each series to fill time_limit_minutes
            times = [e * (time_limit_minutes / max_epoch) for e in epochs]

        ax.semilogy(times, abs_errors, linewidth=1.2, marker='o', markersize=4, label=name)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Time (minutes)", fontsize=10)
    ax.set_ylabel("|log Z err| (log scale)", fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Combine benchmark experiment results for comparison.')
    parser.add_argument('config', help='Path to YAML config file')
    parser.add_argument('--time', type=float, default=60,
                        help='Time limit in minutes for local error plot x-axis (default: 60)')
    parser.add_argument('--early-stopping', action='store_true',
                        help='Use global max epoch as x-axis limit instead of '
                             'stretching each series to --time')
    args = parser.parse_args()

    config_path = args.config
    time_limit_minutes = args.time

    with open(config_path) as f:
        config = yaml.safe_load(f)

    early_stopping = args.early_stopping or config.get('early_stopping', False)

    experiment_name = config['experiment_name']
    paths = [p.strip() for p in config['paths'].split(',')]

    if 'names' in config:
        names = [n.strip() for n in config['names'].split(',')]
        if len(paths) != len(names):
            print(f"Error: {len(paths)} paths but {len(names)} names")
            sys.exit(1)
    else:
        # Auto-derive names from experiment_name in each run's copied config
        names = []
        for p in paths:
            run_config_files = list(Path(p).glob('*.yaml'))
            derived_name = None
            for rc in run_config_files:
                with open(rc) as rcf:
                    run_config = yaml.safe_load(rcf)
                if run_config and 'experiment_name' in run_config:
                    derived_name = run_config['experiment_name']
                    break
            if derived_name is None:
                # Fallback: use the folder name's suffix after timestamp
                folder = Path(p).name
                parts = folder.split('_', 2)  # e.g. 20260325_222939_myexp
                derived_name = parts[2] if len(parts) > 2 else folder
            names.append(derived_name)
        print(f"Auto-derived names from configs: {names}")

    # Create output directory
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'/home/cohenn1/NCE/data/hard_buckets/benchmark_results/{timestamp}_combined_{experiment_name}')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Combining {len(paths)} experiments: {names}")
    print(f"Output: {output_dir}")

    # Load all metrics
    all_metrics = {}
    for name, path in zip(names, paths):
        all_metrics[name] = load_metrics(path)
        print(f"  {name}: {len(all_metrics[name])} buckets found at {path}")

    # Find all bucket IDs across experiments
    all_bucket_ids = set()
    for m in all_metrics.values():
        all_bucket_ids.update(m.keys())
    all_bucket_ids = sorted(all_bucket_ids)

    failed_buckets = {}  # bucket_id -> list of (name, reason)

    for bucket_id in all_bucket_ids:
        bucket_dir = output_dir / bucket_id
        bucket_dir.mkdir(parents=True, exist_ok=True)

        combined_series = []

        for name, path in zip(names, paths):
            metrics = all_metrics[name].get(bucket_id)

            if metrics is None:
                failed_buckets.setdefault(bucket_id, []).append((name, 'missing'))
                continue

            if not is_valid_result(metrics):
                failed_buckets.setdefault(bucket_id, []).append((name, 'invalid (NaN/Inf)'))
                continue

            # Copy individual plots
            copy_plots(path, bucket_dir, name, bucket_id)

            # Collect error tracking for combined plot
            error_tracking = metrics.get('error_tracking', [])
            if error_tracking:
                combined_series.append((name, error_tracking))

        # Build width suffix from first available metrics for this bucket
        width_suffix = ''
        for name in names:
            m = all_metrics[name].get(bucket_id)
            if m and 'bucket_metadata' in m:
                bm = m['bucket_metadata']
                sw = bm.get('scope_width')
                ms = bm.get('message_size')
                if sw is not None and ms is not None and ms > 0:
                    width_suffix = f", Width {sw} ({math.log2(ms):.1f})"
                break

        # Plot combined local errors for this bucket
        if len(combined_series) >= 1:
            plot_combined_local_errors(
                combined_series,
                str(bucket_dir / 'combined_local_error.png'),
                title=f"Local Error Comparison — {bucket_id}{width_suffix}",
                time_limit_minutes=time_limit_minutes,
                early_stopping=early_stopping,
            )

    # Create local_errors folder with copies of all combined_local_error plots
    local_errors_dir = output_dir / 'local_errors'
    local_errors_dir.mkdir(parents=True, exist_ok=True)
    for bucket_id in all_bucket_ids:
        src_file = output_dir / bucket_id / 'combined_local_error.png'
        if src_file.exists():
            dst_file = local_errors_dir / f'combined_local_errors_{bucket_id}.png'
            shutil.copy2(src_file, dst_file)

    # Write summary of failures
    summary = {
        'experiment_name': experiment_name,
        'names': names,
        'paths': paths,
        'num_buckets': len(all_bucket_ids),
        'failed': failed_buckets,
    }
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults: {output_dir}")
    if failed_buckets:
        print(f"\nFailed/skipped buckets:")
        for bid, failures in failed_buckets.items():
            for name, reason in failures:
                print(f"  {bid}: {name} — {reason}")
    else:
        print("All buckets valid across all experiments.")


if __name__ == '__main__':
    main()
