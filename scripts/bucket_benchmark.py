#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Multi-GPU bucket benchmark coordinator.

Orchestrates parallel training of hard buckets across multiple GPUs.
Spawns worker subprocesses (max 1 per GPU) in round-robin fashion,
collects results, and prints a summary.

This is T01 — history tracking and comparison charts come in T02.

Usage:
    python scripts/bucket_benchmark.py config.yaml fast --gpus 0,1,2,3
    python scripts/bucket_benchmark.py config.yaml slow --gpus 0

Mode mapping:
    fast → 60 second time limit per bucket
    slow → 3600 second (1 hour) time limit per bucket

Note: Time limits are checked at epoch boundaries — actual runtime may
      exceed the limit by one epoch duration.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _json_default(obj):
    """JSON serializer for types not natively serializable."""
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
    except ImportError:
        pass
    
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    
    return str(obj)


def generate_run_id():
    """Generate unique run_id from timestamp."""
    return datetime.now(timezone.utc).strftime('run_%Y%m%d_%H%M%S')


def compute_config_hash(config):
    """Compute MD5 hash of sorted config keys for reproducibility tracking."""
    config_str = json.dumps(
        sorted(config.items(), key=lambda x: str(x[0])),
        default=str,
    )
    return hashlib.md5(config_str.encode()).hexdigest()


def append_to_history(run_record, history_path):
    """Append a single run record to history.jsonl.
    
    Creates directory and file if they don't exist. Writes one JSON line
    per run with immediate flush.
    
    Args:
        run_record: dict with run metadata
        history_path: str path to history.jsonl file
    """
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    
    with open(history_path, 'a') as f:
        json.dump(run_record, f, default=_json_default)
        f.write('\n')
        f.flush()
    
    print(f"[BenchmarkCoordinator] Run record appended to {history_path}")


def parse_worker_results(tmp_dir, bucket_list):
    """Parse all worker JSONL result files from temp directory.
    
    Args:
        tmp_dir: str path to temp directory with worker results
        bucket_list: list of bucket dicts (for validation)
    
    Returns:
        list of parsed bucket result dicts
    """
    bucket_results = []
    
    for bucket_entry in bucket_list:
        bucket_id = bucket_entry['bucket_id']
        result_path = os.path.join(tmp_dir, f'{bucket_id}.json')
        
        if os.path.exists(result_path):
            try:
                with open(result_path) as f:
                    # Read JSONL (single line)
                    line = f.read().strip()
                    if line:
                        bucket_result = json.loads(line)
                        bucket_results.append(bucket_result)
            except Exception as e:
                print(f"[BenchmarkCoordinator] WARNING: Could not parse {result_path}: {e}",
                      file=sys.stderr)
    
    return bucket_results


def build_run_record(run_id, config, mode, time_limit, bucket_results):
    """Build complete run record for history tracking.
    
    Args:
        run_id: str unique run identifier
        config: dict validated config
        mode: str 'fast' or 'slow'
        time_limit: int seconds per bucket
        bucket_results: list of parsed bucket result dicts from workers
    
    Returns:
        dict with complete run metadata
    """
    return {
        'run_id': run_id,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'config_hash': compute_config_hash(config),
        'config': config,
        'mode': mode,
        'time_limit_per_bucket': time_limit,
        'buckets': bucket_results,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-GPU bucket benchmark coordinator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/bucket_benchmark.py config.yaml fast --gpus 0,1,2,3
  python scripts/bucket_benchmark.py config.yaml slow --gpus 0

Mode details:
  fast: 60 second time limit per bucket (quick iteration)
  slow: 3600 second (1 hour) time limit per bucket (convergence testing)

Time limit enforcement:
  - Checked at epoch boundaries (not mid-epoch)
  - Actual runtime may exceed limit by one epoch duration
  - Workers print checkpoint messages at configured intervals
        """
    )
    parser.add_argument('config_path',
                        help='Path to YAML config file')
    parser.add_argument('mode', choices=['fast', 'slow'],
                        help='Training mode (fast=60s, slow=3600s per bucket)')
    parser.add_argument('--gpus', default='0,1,2,3',
                        help='Comma-separated GPU IDs (default: 0,1,2,3)')
    parser.add_argument('--output-dir', default='data/benchmark_output',
                        help='Output directory for per-bucket results '
                             '(default: data/benchmark_output)')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config and bucket list loading
# ---------------------------------------------------------------------------

def load_config(config_path):
    """Load and validate config via prepare_config().

    Args:
        config_path: Path to YAML config file

    Returns:
        dict: Validated config dict

    Raises:
        SystemExit: If config file not found or validation fails
    """
    # Defer imports to runtime
    from nce.config_schema import prepare_config
    
    if not os.path.isfile(config_path):
        print(f"[BenchmarkCoordinator] ERROR: Config file not found: {config_path}",
              file=sys.stderr)
        sys.exit(1)

    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"[BenchmarkCoordinator] ERROR: Failed to parse YAML config: {e}",
              file=sys.stderr)
        sys.exit(1)

    try:
        config = prepare_config(config, strict=False)
    except Exception as e:
        print(f"[BenchmarkCoordinator] ERROR: Config validation failed: {e}",
              file=sys.stderr)
        sys.exit(1)

    return config


def load_bucket_list(bucket_list_path):
    """Load the bucket list manifest from S01.

    Args:
        bucket_list_path: Path to bucket_list.json

    Returns:
        list: List of bucket dicts with keys:
              - bucket_id: str
              - file: str (relative path to .pt file)
              - problem_key: str
              - bucket_label: int
              - selection_error: float

    Raises:
        SystemExit: If manifest not found or invalid
    """
    if not os.path.isfile(bucket_list_path):
        print(f"[BenchmarkCoordinator] ERROR: Bucket list not found: {bucket_list_path}",
              file=sys.stderr)
        print(f"[BenchmarkCoordinator] Run S01's select_hard_buckets.py first to generate it.",
              file=sys.stderr)
        sys.exit(1)

    try:
        with open(bucket_list_path) as f:
            bucket_list = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[BenchmarkCoordinator] ERROR: Invalid JSON in bucket list: {e}",
              file=sys.stderr)
        sys.exit(1)

    if not isinstance(bucket_list, list):
        print(f"[BenchmarkCoordinator] ERROR: Expected bucket_list to be a list, "
              f"got {type(bucket_list).__name__}",
              file=sys.stderr)
        sys.exit(1)

    return bucket_list


# ---------------------------------------------------------------------------
# Worker spawning and orchestration
# ---------------------------------------------------------------------------

def spawn_worker(bucket_entry, gpu_id, config_json, output_dir,
                 time_limit, tmp_dir):
    """Spawn a single worker subprocess on the specified GPU.

    Args:
        bucket_entry: Dict with 'bucket_id', 'file', etc.
        gpu_id: int GPU index
        config_json: str JSON-serialized config
        output_dir: str output directory
        time_limit: int seconds
        tmp_dir: str temp directory for result files

    Returns:
        Tuple of (bucket_id, gpu_id, proc, result_path)
    """
    bucket_id = bucket_entry['bucket_id']
    pt_filename = bucket_entry['file']
    pt_path = os.path.join('data', 'hard_buckets', pt_filename)

    result_path = os.path.join(tmp_dir, f'{bucket_id}.json')

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    cmd = [
        sys.executable,
        'scripts/bucket_benchmark_worker.py',
        '--bucket-id', bucket_id,
        '--pt-path', pt_path,
        '--config-json', config_json,
        '--output-dir', output_dir,
        '--time-limit', str(time_limit),
        '--result-path', result_path,
    ]

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    return (bucket_id, gpu_id, proc, result_path)


def run_worker_pool(bucket_list, gpus, config, output_dir, time_limit):
    """Orchestrate multi-GPU worker pool with round-robin scheduling.

    Args:
        bucket_list: List of bucket dicts from S01
        gpus: List of int GPU IDs
        config: Validated config dict
        output_dir: str output directory
        time_limit: int seconds per bucket

    Returns:
        tuple: (results dict (bucket_id -> result dict), tmp_dir path)
    """
    # Serialize config once for all workers
    config_json = json.dumps(config)

    # Create temp dir for worker result files
    tmp_dir = tempfile.mkdtemp(prefix='bucket_benchmark_')
    print(f"[BenchmarkCoordinator] Worker temp dir: {tmp_dir}")

    # Build round-robin GPU queues
    gpu_queues = {g: deque() for g in gpus}
    for i, bucket_entry in enumerate(bucket_list):
        gpu_id = gpus[i % len(gpus)]
        gpu_queues[gpu_id].append(bucket_entry)

    # Track active workers and results
    active = {}  # gpu_id -> (bucket_id, gpu_id, proc, result_path)
    results = {}  # bucket_id -> result dict
    failed = []   # list of bucket_id strings

    num_buckets = len(bucket_list)
    completed_count = 0

    # Start one worker per GPU
    print(f"\n[BenchmarkCoordinator] Starting worker pool: "
          f"{num_buckets} buckets across {len(gpus)} GPUs")
    for gpu_id in gpus:
        if gpu_queues[gpu_id]:
            bucket_entry = gpu_queues[gpu_id].popleft()
            w = spawn_worker(bucket_entry, gpu_id, config_json, output_dir,
                            time_limit, tmp_dir)
            active[gpu_id] = w
            print(f"[BenchmarkCoordinator] Spawned {w[0]} on GPU {gpu_id} (PID {w[2].pid})")

    print(f"[BenchmarkCoordinator] Running (max {len(gpus)} concurrent workers)...\n")

    # Poll for completed workers
    start_time = time.time()
    while active:
        for gpu_id in list(active.keys()):
            bucket_id, gid, proc, result_path = active[gpu_id]
            retcode = proc.poll()

            if retcode is not None:
                # Worker finished
                stdout_text = proc.stdout.read()
                stderr_text = proc.stderr.read()

                # Print worker output
                if stdout_text.strip():
                    print(stdout_text.strip())
                if stderr_text.strip():
                    print(stderr_text.strip(), file=sys.stderr)

                completed_count += 1

                if retcode == 0:
                    # Success — load result
                    try:
                        with open(result_path) as f:
                            result = json.load(f)
                        results[bucket_id] = result
                        final_err = result.get('final_local_error', 'N/A')
                        print(f"[BenchmarkCoordinator] ✓ {bucket_id} complete "
                              f"({completed_count}/{num_buckets}) — "
                              f"final_local_error={final_err}\n")
                    except Exception as e:
                        print(f"[BenchmarkCoordinator] ✗ {bucket_id} completed "
                              f"but result file unreadable: {e}",
                              file=sys.stderr)
                        failed.append(bucket_id)
                else:
                    # Failure
                    print(f"[BenchmarkCoordinator] ✗ {bucket_id} FAILED "
                          f"(exit {retcode}, GPU {gid}) ({completed_count}/{num_buckets})\n",
                          file=sys.stderr)
                    failed.append(bucket_id)

                # Spawn next bucket on this GPU if any remain
                if gpu_queues[gpu_id]:
                    next_entry = gpu_queues[gpu_id].popleft()
                    w = spawn_worker(next_entry, gpu_id, config_json, output_dir,
                                    time_limit, tmp_dir)
                    active[gpu_id] = w
                    print(f"[BenchmarkCoordinator] Spawned {w[0]} on GPU {gpu_id} (PID {w[2].pid})\n")
                else:
                    del active[gpu_id]

        if active:
            time.sleep(5)

    total_time = time.time() - start_time

    print(f"\n[BenchmarkCoordinator] Worker pool complete: "
          f"{len(results)}/{num_buckets} succeeded, {len(failed)} failed, "
          f"total time {total_time:.1f}s")

    if failed:
        print(f"[BenchmarkCoordinator] Failed buckets: {failed}",
              file=sys.stderr)

    return results, tmp_dir


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(results, time_limit, total_time):
    """Print final summary of benchmark run.

    Args:
        results: dict of bucket_id -> result dict
        time_limit: int seconds per bucket
        total_time: float total wall time
    """
    print(f"\n{'='*70}")
    print(f"Benchmark Summary")
    print(f"{'='*70}")
    print(f"  Total buckets:     {len(results)}")
    print(f"  Time limit/bucket: {time_limit}s")
    print(f"  Total wall time:   {total_time:.1f}s")
    print()

    if not results:
        print("  No successful results.")
        return

    print(f"  Per-bucket final local errors:")
    for bucket_id, result in sorted(results.items()):
        final_err = result.get('final_local_error', 'N/A')
        epochs = result.get('epochs_completed', 0)
        wall_time = result.get('wall_time', 0)
        print(f"    {bucket_id:50s}  {final_err:10}  "
              f"(epochs={epochs}, time={wall_time:.1f}s)")

    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Map mode to time limit
    time_limits = {'fast': 60, 'slow': 3600}
    time_limit = time_limits[args.mode]

    # Parse GPU list
    try:
        gpus = [int(x.strip()) for x in args.gpus.split(',')]
    except ValueError:
        print(f"[BenchmarkCoordinator] ERROR: Invalid --gpus format: {args.gpus}",
              file=sys.stderr)
        print(f"[BenchmarkCoordinator] Expected comma-separated integers, e.g. '0,1,2,3'",
              file=sys.stderr)
        sys.exit(1)

    # Generate unique run ID
    run_id = generate_run_id()

    print(f"\n{'='*70}")
    print(f"Bucket Benchmark Coordinator")
    print(f"{'='*70}")
    print(f"  Run ID:         {run_id}")
    print(f"  Config:         {args.config_path}")
    print(f"  Mode:           {args.mode} ({time_limit}s per bucket)")
    print(f"  GPUs:           {gpus}")
    print(f"  Output dir:     {args.output_dir}")
    print(f"{'='*70}\n")

    # Load config
    config = load_config(args.config_path)
    print(f"[BenchmarkCoordinator] Config loaded and validated")
    print(f"[BenchmarkCoordinator] Config hash: {compute_config_hash(config)}")

    # Load bucket list
    bucket_list_path = os.path.join('data', 'hard_buckets', 'bucket_list.json')
    bucket_list = load_bucket_list(bucket_list_path)
    print(f"[BenchmarkCoordinator] Loaded {len(bucket_list)} buckets from manifest\n")

    # Run worker pool
    start_time = time.time()
    results, tmp_dir = run_worker_pool(bucket_list, gpus, config, args.output_dir, time_limit)
    total_time = time.time() - start_time

    # Print summary
    print_summary(results, time_limit, total_time)

    # Parse worker JSONL results
    print(f"\n[BenchmarkCoordinator] Collecting worker results from {tmp_dir}...")
    bucket_results = parse_worker_results(tmp_dir, bucket_list)
    print(f"[BenchmarkCoordinator] Collected {len(bucket_results)} bucket results")

    # Build and append run record to history
    run_record = build_run_record(run_id, config, args.mode, time_limit, bucket_results)
    history_path = os.path.join('data', 'hard_buckets', 'history.jsonl')
    append_to_history(run_record, history_path)

    # Generate comparison chart (T02 step 5 — will implement in next step)
    try:
        from nce.benchmark.comparison import plot_comparison
        comparison_path = os.path.join('data', 'hard_buckets', f'comparison_{run_id}.png')
        plot_comparison(history_path, run_id, comparison_path)
        print(f"[BenchmarkCoordinator] Comparison chart: {comparison_path}")
    except Exception as e:
        print(f"[BenchmarkCoordinator] WARNING: Comparison chart generation failed: {e}",
              file=sys.stderr)

    # Exit with non-zero if any buckets failed
    num_failed = len(bucket_list) - len(results)
    if num_failed > 0:
        print(f"\n[BenchmarkCoordinator] Exiting with code 1 ({num_failed} buckets failed)",
              file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\n[BenchmarkCoordinator] All buckets completed successfully")
        sys.exit(0)


if __name__ == '__main__':
    main()
