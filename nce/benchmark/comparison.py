"""Comparison chart generation for bucket benchmark history.

Reads history.jsonl, filters runs by duration, finds historical best per
bucket, and generates a grouped bar chart comparing current run vs best.
"""

import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def load_history(history_path: str) -> List[dict]:
    """Load all run records from history.jsonl.
    
    Args:
        history_path: Path to history.jsonl file
    
    Returns:
        List of run record dicts, one per line
    """
    if not os.path.exists(history_path):
        return []
    
    records = []
    with open(history_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[ComparisonChart] WARNING: Skipping invalid JSON line: {e}")
    
    return records


def find_current_run(records: List[dict], run_id: str) -> Optional[dict]:
    """Find the run record matching run_id.
    
    Args:
        records: List of all run records
        run_id: Target run_id
    
    Returns:
        Run record dict or None if not found
    """
    for record in records:
        if record.get('run_id') == run_id:
            return record
    return None


def filter_by_duration(records: List[dict], max_duration: int, 
                       exclude_run_id: str) -> List[dict]:
    """Filter records to those with time_limit_per_bucket <= max_duration.
    
    Excludes the current run (to get historical comparison only).
    
    Args:
        records: List of all run records
        max_duration: Maximum time_limit_per_bucket value
        exclude_run_id: Run ID to exclude (the current run)
    
    Returns:
        Filtered list of run records
    """
    filtered = []
    for record in records:
        if record.get('run_id') == exclude_run_id:
            continue
        
        duration = record.get('time_limit_per_bucket', 0)
        if duration <= max_duration:
            filtered.append(record)
    
    return filtered


def extract_best_errors(records: List[dict]) -> Dict[str, float]:
    """Extract best final_local_error per bucket across all records.
    
    Args:
        records: List of run records
    
    Returns:
        Dict mapping bucket_id -> best final_local_error (lowest value)
    """
    best_errors = {}
    
    for record in records:
        for bucket in record.get('buckets', []):
            bucket_id = bucket.get('bucket_id')
            error = bucket.get('final_local_error')
            
            if bucket_id is None or error is None:
                continue
            
            # Lower error is better
            if bucket_id not in best_errors or error < best_errors[bucket_id]:
                best_errors[bucket_id] = error
    
    return best_errors


def plot_comparison(history_path: str, current_run_id: str, output_path: str) -> str:
    """Generate comparison chart: current run vs historical best per bucket.
    
    Reads history.jsonl, filters runs by duration ≤ current run's duration,
    finds historical best final_local_error per bucket, and plots a grouped
    bar chart (current vs best) with semilogy scale.
    
    Args:
        history_path: Path to history.jsonl
        current_run_id: Run ID for the current run
        output_path: Path for output PNG
    
    Returns:
        str: Path to generated PNG file
    
    Raises:
        ValueError: If current run not found in history
    """
    # Load history
    records = load_history(history_path)
    
    if not records:
        # No history — chart shows only current run
        print(f"[ComparisonChart] No history found, generating current-run-only chart")
        current_run = {'run_id': current_run_id, 'buckets': []}
    else:
        # Find current run
        current_run = find_current_run(records, current_run_id)
        if current_run is None:
            raise ValueError(
                f"Current run_id '{current_run_id}' not found in history. "
                f"Available: {[r.get('run_id') for r in records]}"
            )
    
    # Extract current run's bucket errors
    current_buckets = {}
    for bucket in current_run.get('buckets', []):
        bucket_id = bucket.get('bucket_id')
        error = bucket.get('final_local_error')
        if bucket_id and error is not None:
            current_buckets[bucket_id] = error
    
    if not current_buckets:
        print(f"[ComparisonChart] WARNING: Current run has no bucket results")
        # Create empty chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"No results for run {current_run_id}",
                ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=100)
        plt.close(fig)
        return output_path
    
    # Filter historical runs by duration
    current_duration = current_run.get('time_limit_per_bucket', 0)
    historical_runs = filter_by_duration(records, current_duration, current_run_id)
    
    # Extract best errors from historical runs
    best_errors = extract_best_errors(historical_runs)
    
    # Build chart data
    bucket_ids = sorted(current_buckets.keys())
    x_positions = np.arange(len(bucket_ids))
    width = 0.35
    
    current_values = [current_buckets[bid] for bid in bucket_ids]
    best_values = [best_errors.get(bid, None) for bid in bucket_ids]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(10, len(bucket_ids) * 0.5), 6))
    
    # Plot bars
    ax.bar(x_positions - width/2, current_values, width, 
           label=f'Current ({current_run_id})', alpha=0.8)
    
    # Only plot historical bars if we have historical data
    if any(v is not None for v in best_values):
        # Replace None with NaN for plotting (will be skipped)
        best_values_plot = [v if v is not None else np.nan for v in best_values]
        ax.bar(x_positions + width/2, best_values_plot, width,
               label=f'Historical Best (≤{current_duration}s)', alpha=0.8)
    else:
        # No historical comparison available
        ax.text(0.5, 0.95, "No historical comparison available", 
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Semilogy scale (errors span orders of magnitude)
    ax.set_yscale('log')
    
    # Labels and title
    ax.set_xlabel('Bucket ID', fontsize=11)
    ax.set_ylabel('Final Local Error (|log Z error|)', fontsize=11)
    ax.set_title(f'Bucket Benchmark Comparison\n'
                 f'Current: {current_run_id} vs Historical Best (duration ≤ {current_duration}s)',
                 fontsize=12)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(bucket_ids, rotation=45, ha='right', fontsize=8)
    ax.legend(loc='best')
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close(fig)
    
    print(f"[ComparisonChart] Saved to {output_path}")
    return output_path
