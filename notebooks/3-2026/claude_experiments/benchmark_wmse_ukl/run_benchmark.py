#!/usr/bin/env python3
"""WMSE vs UKL Benchmark: 5 configurations x 24 small_problems x 5000 epochs.

Runs 120 experiments across multiple GPUs to compare weighted logspace MSE
(NeuroBE's loss) against unnormalized KL with varying backward information levels.

Usage:
    # List all 120 jobs without running:
    python run_benchmark.py --dry-run

    # Run full benchmark across 4 GPUs:
    python run_benchmark.py --mode orchestrator --gpus 0,1,2,3

    # Run a single job (spawned by orchestrator):
    python run_benchmark.py --mode worker --job-id 42
"""

import argparse
import copy
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Add NCE to path for imports
sys.path.insert(0, '/home/cohenn1/NCE')

from nce.benchmark_problems.small_problems import small_problems, _AUTO_ECL


# ---------------------------------------------------------------------------
# Configuration builder (copied from verify_config_correctness.py -- verified)
# ---------------------------------------------------------------------------

def build_experiment_config(base_cfg, loss_fn, bw_ecl, num_epochs=5000):
    """Build an experiment config from base small_problems config.

    Args:
        base_cfg: Base config dict from small_problems.configs['default']
        loss_fn: Loss function name string
        bw_ecl: Backward ECL value (0 for no backward info)
        num_epochs: Number of training epochs

    Returns:
        Complete nn_config dict ready for FastGM
    """
    cfg = copy.deepcopy(base_cfg)
    cfg['loss_fn'] = loss_fn
    cfg['num_epochs'] = num_epochs
    cfg['skip_early_stopping'] = True
    cfg['nbe_early_stopping'] = False
    cfg['batch_size'] = 10000000  # Very large to ensure single batch

    cfg['bw_ecl'] = bw_ecl
    cfg['backward_ecl'] = bw_ecl
    cfg['populate_bw_factors'] = bw_ecl > 0
    cfg['use_bw_approx'] = bw_ecl > 0

    return cfg


# ---------------------------------------------------------------------------
# The 5 experiment configurations
# ---------------------------------------------------------------------------

CONFIGS = [
    {'name': 'wmse_bw0', 'loss_fn': 'weighted_logspace_mse', 'bw_ecl': 0},
    {'name': 'ukl_bw0', 'loss_fn': 'unnormalized_kl', 'bw_ecl': 0},
    {'name': 'ukl_bw8', 'loss_fn': 'unnormalized_kl', 'bw_ecl': 8},
    {'name': 'ukl_bw_ecl', 'loss_fn': 'unnormalized_kl', 'bw_ecl': 'auto_ecl'},
    {'name': 'ukl_bw30', 'loss_fn': 'unnormalized_kl', 'bw_ecl': 2**30},
]


# ---------------------------------------------------------------------------
# Job construction
# ---------------------------------------------------------------------------

def build_all_jobs():
    """Build the list of 120 experiment jobs.

    Returns:
        List of job dicts (without the 'model' key -- models are loaded in worker).
        Each dict has: job_id, config_name, problem_idx, modelfile, nn_config,
        loss_fn, bw_ecl, ecl.
    """
    base_configs = small_problems.configs['default']
    problems = small_problems.problems

    jobs = []
    for config in CONFIGS:
        for i, (model, base_cfg) in enumerate(zip(problems, base_configs)):
            bw_ecl = config['bw_ecl']
            if bw_ecl == 'auto_ecl':
                bw_ecl = base_cfg['ecl']  # per-problem auto_ecl

            nn_config = build_experiment_config(base_cfg, config['loss_fn'], bw_ecl)

            jobs.append({
                'job_id': len(jobs),
                'config_name': config['name'],
                'problem_idx': i,
                'modelfile': model.modelfile,
                'nn_config': nn_config,
                'loss_fn': config['loss_fn'],
                'bw_ecl': bw_ecl,
                'ecl': base_cfg['ecl'],
            })

    return jobs


# ---------------------------------------------------------------------------
# Worker: run a single experiment
# ---------------------------------------------------------------------------

def run_single_experiment(job, results_dir):
    """Run one experiment: create FastGM, train, collect results.

    Args:
        job: Job dict with nn_config, problem_idx, config_name, modelfile, etc.
        results_dir: Path to results root directory.

    Returns:
        Result dict with log_z, duration, status, etc.
    """
    import torch
    import numpy as np
    import random
    from nce.inference.graphical_model import FastGM

    # Set seeds
    seed = job['nn_config']['seed']
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**31))
    random.seed(seed)

    start_time = datetime.now()
    result = {
        'job_id': job['job_id'],
        'config_name': job['config_name'],
        'modelfile': job['modelfile'],
        'problem_idx': job['problem_idx'],
        'loss_fn': job['loss_fn'],
        'bw_ecl': job['bw_ecl'],
        'ecl': job['ecl'],
        'num_epochs': job['nn_config']['num_epochs'],
        'hidden_sizes': job['nn_config']['hidden_sizes'],
        'seed': seed,
        'start_time': start_time.isoformat(),
        'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'error': None,
        'traceback': None,
    }

    try:
        # Load the model for this problem
        model = small_problems.problems[job['problem_idx']]

        # Create FastGM and run inference
        fastgm = FastGM(model=model, nn_config=job['nn_config'])
        log_z = fastgm.get_log_partition_function()

        end_time = datetime.now()
        result['log_z_estimate'] = float(log_z) if log_z is not None else None
        result['num_buckets_trained'] = getattr(fastgm, 'num_trained', 0)
        result['duration_seconds'] = (end_time - start_time).total_seconds()
        result['end_time'] = end_time.isoformat()
        result['status'] = 'completed'

    except Exception as e:
        end_time = datetime.now()
        result['log_z_estimate'] = None
        result['num_buckets_trained'] = 0
        result['duration_seconds'] = (end_time - start_time).total_seconds()
        result['end_time'] = end_time.isoformat()
        result['status'] = 'failed'
        result['error'] = str(e)
        result['traceback'] = traceback.format_exc()

    # Save result JSON
    config_dir = Path(results_dir) / job['config_name']
    config_dir.mkdir(parents=True, exist_ok=True)
    result_path = config_dir / f"{job['modelfile']}.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(results_dir):
    """Configure logging to both console and file.

    Args:
        results_dir: Path to results directory for log file.
    """
    log_path = Path(results_dir) / 'benchmark.log'
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger('benchmark')
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(console_fmt)

    # File handler
    file_handler = logging.FileHandler(str(log_path))
    file_handler.setLevel(logging.INFO)
    file_fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Orchestrator mode
# ---------------------------------------------------------------------------

def run_orchestrator(args):
    """Distribute all 120 jobs across GPUs in waves."""
    import torch

    results_dir = args.results_dir
    gpus = [int(g) for g in args.gpus.split(',')]
    num_gpus = len(gpus)

    logger = setup_logging(results_dir)

    # Build all jobs
    jobs = build_all_jobs()
    total_jobs = len(jobs)

    # Pre-run validation
    assert len(small_problems.problems) == 24, \
        f"Expected 24 problems, got {len(small_problems.problems)}"
    assert total_jobs == 120, \
        f"Expected 120 jobs (5 configs x 24 problems), got {total_jobs}"

    gpu_count = torch.cuda.device_count()
    logger.info(f"GPU count: {gpu_count}")
    for i in range(gpu_count):
        logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Config summary
    logger.info("=" * 70)
    logger.info("WMSE vs UKL Benchmark")
    logger.info("=" * 70)
    logger.info(f"Total experiments: {total_jobs}")
    logger.info(f"Configurations: {len(CONFIGS)}")
    logger.info(f"Problems: {len(small_problems.problems)}")
    logger.info(f"GPUs to use: {gpus}")
    logger.info(f"Waves: {(total_jobs + num_gpus - 1) // num_gpus}")
    logger.info("")

    for cfg in CONFIGS:
        bw_str = str(cfg['bw_ecl']) if cfg['bw_ecl'] != 'auto_ecl' else 'auto_ecl (per-problem)'
        logger.info(f"  {cfg['name']:15s} | loss_fn={cfg['loss_fn']:30s} | bw_ecl={bw_str}")
    logger.info("")

    # Sanity check: count NN-eligible buckets for smallest problem (BN_3, ecl=16383)
    from nce.inference.graphical_model import FastGM
    sanity_model = small_problems.problems[1]  # BN_3
    sanity_cfg = copy.deepcopy(small_problems.configs['default'][1])
    sanity_cfg['device'] = 'cpu'
    sanity_cfg['num_epochs'] = 0
    sanity_gm = FastGM(model=sanity_model, nn_config=sanity_cfg, device='cpu')
    nn_buckets = sanity_gm.get_large_message_buckets(iB=100, ecl=16383)
    logger.info(f"Sanity check: BN_3 (ecl=16383) has {len(nn_buckets)} NN-eligible buckets")
    logger.info("")

    logger.info(f"Starting {total_jobs} experiments across {num_gpus} GPUs.")
    logger.info("Estimated ~4-5 min/experiment, ~8-10 hours total (sequential), ~2-3 hours with 4 GPUs.")
    logger.info("")

    # Track results
    all_results = []
    completed = 0
    failed = 0
    failed_jobs = []
    overall_start = time.time()

    # Script path for subprocess spawning
    script_path = str(Path(__file__).resolve())

    # Process in waves
    job_idx = 0
    wave_num = 0

    while job_idx < total_jobs:
        wave_num += 1
        wave_start = time.time()
        active_procs = []

        # Spawn one job per GPU
        for gpu_id in gpus:
            if job_idx >= total_jobs:
                break

            job = jobs[job_idx]
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

            cmd = [
                sys.executable, script_path,
                '--mode', 'worker',
                '--job-id', str(job['job_id']),
                '--results-dir', str(results_dir),
            ]

            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            active_procs.append((proc, job, gpu_id))
            logger.info(f"[{job_idx + 1}/{total_jobs}] Spawned: {job['config_name']:15s} | "
                        f"{job['modelfile']:45s} | GPU {gpu_id}")
            job_idx += 1

        # Wait for all processes in this wave (NO timeout)
        for proc, job, gpu_id in active_procs:
            stdout, stderr = proc.communicate()

            duration_str = "?"
            status = "?"

            # Try to read the result JSON
            result_path = Path(results_dir) / job['config_name'] / f"{job['modelfile']}.json"
            if result_path.exists():
                try:
                    with open(result_path, 'r') as f:
                        result = json.load(f)
                    duration_str = f"{result.get('duration_seconds', 0):.1f}s"
                    status = result.get('status', 'unknown')
                    all_results.append(result)
                except (json.JSONDecodeError, IOError):
                    status = 'result_read_error'

            if proc.returncode == 0 and status == 'completed':
                completed += 1
                logger.info(f"  Done: {job['config_name']:15s} | {job['modelfile']:45s} | "
                            f"GPU {gpu_id} | {duration_str} | {status}")
            else:
                failed += 1
                failed_jobs.append({
                    'job_id': job['job_id'],
                    'config_name': job['config_name'],
                    'modelfile': job['modelfile'],
                    'returncode': proc.returncode,
                    'stderr': stderr[-500:] if stderr else '',
                })
                logger.info(f"  FAIL: {job['config_name']:15s} | {job['modelfile']:45s} | "
                            f"GPU {gpu_id} | rc={proc.returncode}")
                if stderr:
                    logger.info(f"        stderr: {stderr[-200:]}")

        wave_duration = time.time() - wave_start
        logger.info(f"Wave {wave_num} completed in {wave_duration:.1f}s "
                     f"({completed + failed}/{total_jobs} done, {failed} failed)")
        logger.info("")

    # Overall summary
    overall_duration = time.time() - overall_start
    logger.info("=" * 70)
    logger.info("BENCHMARK COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total: {total_jobs} experiments")
    logger.info(f"Completed: {completed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total duration: {overall_duration:.1f}s ({overall_duration / 3600:.1f}h)")
    logger.info("")

    # Per-config summary
    per_config = {}
    for result in all_results:
        cn = result.get('config_name', 'unknown')
        if cn not in per_config:
            per_config[cn] = {'completed': 0, 'failed': 0, 'durations': []}
        if result.get('status') == 'completed':
            per_config[cn]['completed'] += 1
            per_config[cn]['durations'].append(result.get('duration_seconds', 0))
        else:
            per_config[cn]['failed'] += 1

    for cn, stats in per_config.items():
        mean_dur = sum(stats['durations']) / len(stats['durations']) if stats['durations'] else 0
        logger.info(f"  {cn:15s}: completed={stats['completed']}, failed={stats['failed']}, "
                     f"mean_duration={mean_dur:.1f}s")

    if failed_jobs:
        logger.info("")
        logger.info("Failed experiments:")
        for fj in failed_jobs:
            logger.info(f"  job_id={fj['job_id']}, {fj['config_name']} | {fj['modelfile']}")

    # Write aggregate summary.json
    summary = {
        'total_experiments': total_jobs,
        'completed': completed,
        'failed': failed,
        'total_duration_seconds': overall_duration,
        'per_config_summary': {},
        'failed_experiments': failed_jobs,
    }
    for cn, stats in per_config.items():
        mean_dur = sum(stats['durations']) / len(stats['durations']) if stats['durations'] else 0
        summary['per_config_summary'][cn] = {
            'completed': stats['completed'],
            'failed': stats['failed'],
            'mean_duration': mean_dur,
        }

    summary_path = Path(results_dir) / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nSummary written to: {summary_path}")

    # Discord ping when finished
    try:
        subprocess.run(
            [os.path.expanduser('~/.claude/ai-ops/scripts/ping_nick.sh'),
             f"WMSE vs UKL benchmark COMPLETE: {completed}/{total_jobs} experiments done, "
             f"{failed} failed, {overall_duration / 3600:.1f}h total. "
             f"Results in notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/results/"],
            check=False,
        )
    except Exception:
        pass  # Non-critical

    return 0 if failed == 0 else 1


# ---------------------------------------------------------------------------
# Worker mode
# ---------------------------------------------------------------------------

def run_worker(args):
    """Run a single experiment job identified by --job-id."""
    # Rebuild full job list to get this specific job
    jobs = build_all_jobs()
    job_id = args.job_id

    if job_id < 0 or job_id >= len(jobs):
        print(f"Error: job-id {job_id} out of range [0, {len(jobs) - 1}]", file=sys.stderr)
        return 1

    job = jobs[job_id]
    results_dir = args.results_dir

    print(f"Worker: job_id={job_id}, config={job['config_name']}, "
          f"model={job['modelfile']}, bw_ecl={job['bw_ecl']}")

    result = run_single_experiment(job, results_dir)

    print(f"Worker done: status={result['status']}, "
          f"log_z={result.get('log_z_estimate')}, "
          f"duration={result.get('duration_seconds', 0):.1f}s")

    return 0 if result['status'] == 'completed' else 1


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def run_dry_run(args):
    """Print all 120 jobs without executing."""
    jobs = build_all_jobs()
    gpus = [int(g) for g in args.gpus.split(',')]

    print(f"Total problems: {len(small_problems.problems)}")
    print(f"Total configurations: {len(CONFIGS)}")
    print(f"Total jobs: {len(jobs)}")
    print(f"GPUs: {gpus}")
    print(f"Waves: {(len(jobs) + len(gpus) - 1) // len(gpus)}")
    print()

    # Config summary table
    print(f"{'Config':<15s} | {'Loss Function':<30s} | {'bw_ecl':<15s}")
    print("-" * 65)
    for cfg in CONFIGS:
        bw_str = str(cfg['bw_ecl']) if cfg['bw_ecl'] != 'auto_ecl' else 'auto_ecl'
        print(f"{cfg['name']:<15s} | {cfg['loss_fn']:<30s} | {bw_str:<15s}")
    print()

    # Job list
    print(f"{'Job':>4s} | {'Config':<15s} | {'Modelfile':<45s} | {'bw_ecl':>12s} | {'ecl':>12s} | {'GPU':>3s}")
    print("-" * 105)
    for job in jobs:
        gpu = gpus[job['job_id'] % len(gpus)]
        print(f"{job['job_id']:4d} | {job['config_name']:<15s} | {job['modelfile']:<45s} | "
              f"{job['bw_ecl']:12d} | {job['ecl']:12d} | {gpu:3d}")

    # Pre-run validation checks
    print()
    print("Pre-run validation:")
    print(f"  Problems: {len(small_problems.problems)} (expected 24) {'PASS' if len(small_problems.problems) == 24 else 'FAIL'}")
    print(f"  Jobs: {len(jobs)} (expected 120) {'PASS' if len(jobs) == 120 else 'FAIL'}")

    import torch
    gpu_count = torch.cuda.device_count()
    print(f"  GPUs available: {gpu_count} {'PASS' if gpu_count >= 1 else 'FAIL'}")
    for i in range(gpu_count):
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='WMSE vs UKL Benchmark: 5 configs x 24 problems x 5000 epochs'
    )
    parser.add_argument('--mode', choices=['orchestrator', 'worker'],
                        default='orchestrator',
                        help='Run mode (default: orchestrator)')
    parser.add_argument('--job-id', type=int, default=None,
                        help='Job ID (required for worker mode)')
    parser.add_argument('--results-dir', type=str,
                        default=str(Path(__file__).parent / 'results'),
                        help='Path to results directory')
    parser.add_argument('--gpus', type=str, default='0,1,2,3',
                        help='Comma-separated GPU IDs (default: 0,1,2,3)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print job list and exit without running')

    args = parser.parse_args()

    if args.dry_run:
        return run_dry_run(args)

    if args.mode == 'worker':
        if args.job_id is None:
            parser.error('--job-id is required for worker mode')
        return run_worker(args)

    return run_orchestrator(args)


if __name__ == '__main__':
    sys.exit(main())
