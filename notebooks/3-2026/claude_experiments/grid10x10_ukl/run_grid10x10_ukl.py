#!/usr/bin/env python3
"""Experiment runner: grid10x10.f10 with UKL loss (unnormalized_kl).

This script runs a single NCE inference experiment on grid10x10.f10 using
unnormalized KL divergence loss. It produces:
  - CSV with all metrics (results/grid10x10_f10_ukl_results.csv)
  - Pickle with full experiment state (results/experiment_results.pkl)
  - Console summary with per-bucket training info

Usage:
    /home/cohenn1/NCE/venv/bin/python run_grid10x10_ukl.py

Configuration (hardcoded per quick-18 plan):
    loss_fn='unnormalized_kl', ecl=1024, bw_ecl=0, num_epochs=500,
    iB=10, device='cuda', hidden_sizes='nbe,1', num_samples='nbe,0.35'

Model:
    grids/grid10x10.f10 (width=12, num_vars=100, PR=303.085957 in log10)

Pre-flight:
    - Checks CUDA availability; exits with error if CUDA not found.
    - Prints GPU name and memory info.
    - Pings Discord when complete.

Notes for future Claude Code agents:
    - torch is imported INSIDE main() to respect CUDA_VISIBLE_DEVICES.
    - fastgm.buckets is a dict (var -> FastBucket), not a list.
    - FastBucket.epochs_trained and trained_hidden_sizes are set AFTER
      compute_message_nn() runs -- not available on pre-elimination buckets.
    - model.PR is log10 exact log Z; log_z_hat from get_log_partition_function()
      is also in log10 space.
    - err = log_Z_hat - log_Z_star (positive = overestimate, negative = underestimate)
"""

import os
import sys
import csv
import json
import pickle
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

# NCE must be on path
sys.path.insert(0, '/home/cohenn1/NCE')

# Results directory (relative to this script)
SCRIPT_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = SCRIPT_DIR / 'results'


def build_config():
    """Build the full 42-field UKL config for grid10x10.f10.

    All fields are populated following the nbe_sanity_check config template
    from nce/benchmark_problems/nbe_sanity_check.py (_build_nbe_configs).

    Changes from NeuroBE defaults:
        - loss_fn='unnormalized_kl' (not 'weighted_logspace_mse')
        - bw_ecl=0 (no backward information)
        - backward_ecl=0 (no backward information)
        - ecl=2**10=1024 (iB=10 -> ecl=2^10)
        - populate_bw_factors=False (bw_ecl=0)
        - use_bw_approx=False (bw_ecl=0)
    """
    return {
        'device': 'cuda',
        'hidden_sizes': 'nbe,1',
        'optimizer': 'adam',
        'lr': 0.001,
        'lr_decay': 1.0,
        'momentum': 0.9,
        'inverse_time_decay_constant': 100,
        'patience': 20,
        'min_lr': 1e-8,
        'num_epochs': 500,
        'num_epochs2': 0,
        'nbe_early_stopping': False,
        'nbe_warmup_epochs': 0,
        'skip_early_stopping': False,
        'sampling_scheme': 'uniform',
        'batch_size': 256,
        'set_size': None,
        'num_samples': 'nbe,0.35',
        'num_batches_per_set': 1,
        'loss_fn': 'unnormalized_kl',
        'traced_losses': [],
        'val_set': True,
        'fdb': False,
        'use_bw_approx': False,
        'populate_bw_factors': False,
        'ecl': 2**10,           # 1024
        'iB': 10,
        'approximation_method': 'nn',
        'bw_ecl': 0,
        'backward_ecl': 0,
        'backward_iB': 10,
        'use_linspace_bias': False,
        'use_memorizer': False,
        'display_intermediate': False,
        'track_errors': False,
        'plot_messages': False,
        'debug': False,
        'lower_dim': False,
        'dope_factors': True,
        'gather_message_stats': False,
        'stratify_samples': False,
        'seed': 42,
    }


def main():
    """Run the grid10x10.f10 UKL experiment end to end."""
    import torch
    import numpy as np
    import random
    from nce.benchmark_problems.catalog_utils import get_catalog
    from nce.inference.graphical_model import FastGM

    # -----------------------------------------------------------------------
    # Config
    # -----------------------------------------------------------------------
    config = build_config()

    print("=" * 70)
    print("grid10x10.f10 UKL Experiment")
    print("=" * 70)
    print(f"Config:")
    for k, v in sorted(config.items()):
        print(f"  {k:35s}: {v}")
    print()

    # -----------------------------------------------------------------------
    # Pre-flight: CUDA check
    # -----------------------------------------------------------------------
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This experiment requires a GPU.", file=sys.stderr)
        print("       Do not downgrade to CPU without explicit user approval.", file=sys.stderr)
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    gpu_free = (torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated(0)) / 1024**3
    print(f"GPU: {gpu_name}")
    print(f"  Total memory: {gpu_total:.1f} GB")
    print(f"  Free memory:  {gpu_free:.1f} GB")
    print()

    # -----------------------------------------------------------------------
    # Set seeds
    # -----------------------------------------------------------------------
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**31))
    random.seed(seed)

    # -----------------------------------------------------------------------
    # Load model from catalog
    # -----------------------------------------------------------------------
    print("Loading model from catalog...")
    catalog = get_catalog()
    model = catalog['grids/grid10x10.f10']

    model_name = model.modelfile        # e.g. 'grid10x10.f10.uai'
    model_pr = model.PR                 # exact log Z (log10 space)
    model_num_vars = model.num_vars     # 100
    model_width = model.width           # 12

    print(f"Model: {model_name}")
    print(f"  num_vars = {model_num_vars}")
    print(f"  width    = {model_width}")
    print(f"  PR (log10 exact log Z) = {model_pr}")
    print()

    # -----------------------------------------------------------------------
    # Run inference
    # -----------------------------------------------------------------------
    print("Running FastGM inference...")
    print(f"  loss_fn  = {config['loss_fn']}")
    print(f"  ecl      = {config['ecl']}")
    print(f"  iB       = {config['iB']}")
    print(f"  epochs   = {config['num_epochs']}")
    print(f"  bw_ecl   = {config['bw_ecl']}")
    print()

    t_start = time.time()
    fastgm = FastGM(model=model, nn_config=config, device='cuda')
    log_z_hat = fastgm.get_log_partition_function()
    t_end = time.time()

    duration_seconds = t_end - t_start
    log_z_hat_float = float(log_z_hat) if log_z_hat is not None else None

    print(f"Inference complete in {duration_seconds:.1f}s")
    print(f"  log_Z_hat (log10) = {log_z_hat_float}")
    print(f"  log_Z_star (log10) = {model_pr}")
    print()

    # -----------------------------------------------------------------------
    # Compute metrics
    # -----------------------------------------------------------------------
    log_Z_star = model_pr       # exact log Z (log10)
    log_Z_hat = log_z_hat_float # estimated log Z (log10)
    err = log_Z_hat - log_Z_star if log_Z_hat is not None else None
    abs_err = abs(err) if err is not None else None
    num_trained = getattr(fastgm, 'num_trained', 0)

    print("Metrics:")
    print(f"  err (log_Z_hat - log_Z_star) = {err:.6f}" if err is not None else "  err = None")
    print(f"  abs_err = {abs_err:.6f}" if abs_err is not None else "  abs_err = None")
    print(f"  num_trained (NN buckets) = {num_trained}")
    print()

    # -----------------------------------------------------------------------
    # Collect per-bucket data (available after inference)
    # -----------------------------------------------------------------------
    per_bucket_data = []
    for var, bucket in fastgm.buckets.items():
        if hasattr(bucket, 'epochs_trained'):
            per_bucket_data.append({
                'label': getattr(bucket, 'label', var),
                'epochs_trained': bucket.epochs_trained,
                'hidden_sizes': bucket.trained_hidden_sizes,
            })

    print(f"Per-bucket training data ({len(per_bucket_data)} NN buckets):")
    for bd in per_bucket_data:
        print(f"  bucket {bd['label']:3d}: epochs={bd['epochs_trained']:4d}, "
              f"hidden_sizes={bd['hidden_sizes']}")
    print()

    # -----------------------------------------------------------------------
    # Output directory
    # -----------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Write CSV
    # -----------------------------------------------------------------------
    csv_path = RESULTS_DIR / 'grid10x10_f10_ukl_results.csv'
    fieldnames = [
        'problem_name', 'width', 'num_vars',
        'log_Z_star', 'log_Z_hat', 'err', 'abs_err',
        'num_trained', 'time_seconds', 'num_samples',
        'architecture', 'loss_fn',
    ]
    row = {
        'problem_name': model_name,
        'width': model_width,
        'num_vars': model_num_vars,
        'log_Z_star': log_Z_star,
        'log_Z_hat': log_Z_hat,
        'err': err,
        'abs_err': abs_err,
        'num_trained': num_trained,
        'time_seconds': duration_seconds,
        'num_samples': config['num_samples'],
        'architecture': str(config['hidden_sizes']),
        'loss_fn': config['loss_fn'],
    }
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    print(f"CSV written: {csv_path}")

    # -----------------------------------------------------------------------
    # Pickle all results
    # -----------------------------------------------------------------------
    timestamp = datetime.now(timezone.utc).isoformat()
    results_dict = {
        'config': config,
        'model_name': model_name,
        'model_width': model_width,
        'model_num_vars': model_num_vars,
        'model_PR': model_pr,
        'log_z_hat': log_Z_hat,
        'log_Z_star': log_Z_star,
        'err': err,
        'abs_err': abs_err,
        'num_trained': num_trained,
        'duration_seconds': duration_seconds,
        'per_bucket_data': per_bucket_data,
        'timestamp': timestamp,
    }
    pkl_path = RESULTS_DIR / 'experiment_results.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(results_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Pickle written: {pkl_path}")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Model:        {model_name}")
    print(f"Width:        {model_width}")
    print(f"Num vars:     {model_num_vars}")
    print(f"Loss fn:      {config['loss_fn']}")
    print(f"Epochs:       {config['num_epochs']}")
    print(f"ecl:          {config['ecl']}")
    print(f"bw_ecl:       {config['bw_ecl']}")
    print(f"log_Z_star:   {log_Z_star:.6f} (log10)")
    print(f"log_Z_hat:    {log_Z_hat:.6f} (log10)")
    print(f"err:          {err:.6f} (log10)" if err is not None else "err: None")
    print(f"abs_err:      {abs_err:.6f} (log10)" if abs_err is not None else "abs_err: None")
    print(f"num_trained:  {num_trained} NN buckets")
    print(f"Duration:     {duration_seconds:.1f}s")
    print(f"Timestamp:    {timestamp}")
    print()
    print(f"Output files:")
    print(f"  {csv_path}")
    print(f"  {pkl_path}")

    # -----------------------------------------------------------------------
    # Discord ping
    # -----------------------------------------------------------------------
    try:
        ping_msg = (
            f"grid10x10.f10 UKL experiment COMPLETE: "
            f"err={err:.4f} (log10), {num_trained} NN buckets, {duration_seconds:.0f}s. "
            f"Results in notebooks/3-2026/claude_experiments/grid10x10_ukl/results/"
        )
        subprocess.run(
            [os.path.expanduser('~/.claude/ai-ops/scripts/ping_nick.sh'), ping_msg],
            check=False,
        )
    except Exception:
        pass  # Non-critical

    return 0


if __name__ == '__main__':
    sys.exit(main())
