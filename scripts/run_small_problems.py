#!/usr/bin/env python3
"""Run all 24 small_problems and collect log_Z estimates.

Takes a YAML config for shared training parameters. Per-problem auto_ecl
values are automatically applied from small_problems._AUTO_ECL.

Usage:
    python scripts/run_small_problems.py config.yaml
    python scripts/run_small_problems.py config.yaml --output results.json
    python scripts/run_small_problems.py config.yaml --output results.csv
    python scripts/run_small_problems.py config.yaml --gpus 0,1,2,3
    python scripts/run_small_problems.py config.yaml --ecl 15

The YAML config should contain training parameters (num_epochs, loss_fn,
hidden_sizes, etc.) but NOT ecl — that's set per-problem automatically
unless overridden with --ecl.

Example config.yaml:
    inference:
      device: cuda
      i_bound: 100
      approximation_method: nn
    nn:
      hidden_sizes: [3, 3]
    training:
      num_epochs: 10000
      loss_fn: unnormalized_kl
      batch_size: 100000
      learning_rate: 0.01
      skip_early_stopping: true
    sampling:
      sampling_scheme: all
      num_samples: 100000
"""
import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all small_problems and collect log_Z estimates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('config_path', help='Path to YAML config file')
    parser.add_argument('--output', '-o', default=None,
                        help='Output path (.json or .csv). Default: prints to stdout')
    parser.add_argument('--problems', default=None,
                        help='Comma-separated problem indices (0-23) to run. '
                             'Default: all 24')
    parser.add_argument('--gpus', default=None,
                        help='Comma-separated GPU IDs for parallel execution '
                             '(e.g. 0,1,2,3). Default: single GPU, sequential')
    parser.add_argument('--ecl', type=int, default=None,
                        help='Override auto_ecl with this fixed ecl value for all problems')
    return parser.parse_args()


def run_sequential(indices, base_config, ecl_override=None):
    """Run problems sequentially on a single device."""
    from nce.benchmark_problems.small_problems import small_problems, _AUTO_ECL
    from nce.config_schema import prepare_config
    from nce.inference.graphical_model import FastGM

    problems = small_problems.problems
    results = []

    for i in indices:
        model = problems[i]
        ecl = ecl_override if ecl_override is not None else _AUTO_ECL[model.modelfile]

        config = copy.deepcopy(base_config)
        # Inject ecl and default iB=100 (so only ecl controls NN dispatch)
        if any(k in config for k in ('inference', 'nn', 'training', 'sampling', 'backward', 'output')):
            config.setdefault('inference', {})
            config['inference']['ecl'] = ecl
            config['inference'].setdefault('i_bound', 100)
        else:
            config['ecl'] = ecl
            config.setdefault('iB', 100)

        config = prepare_config(config, strict=False)
        device = config.get('device', 'cuda')

        print(f"\n[{len(results)+1}/{len(indices)}] {model.modelfile}  (ecl={ecl})")

        start = time.time()
        try:
            fastgm = FastGM(model=model, nn_config=config, device=device)
            log_Z = fastgm.get_log_partition_function()
            duration = time.time() - start
            print(f"  log_Z = {log_Z:.6f}  ({duration:.1f}s)")
            results.append({
                'problem': model.modelfile,
                'index': i,
                'log_Z': float(log_Z),
                'ecl': ecl,
                'duration_seconds': round(duration, 2),
                'status': 'ok',
            })
        except Exception as e:
            duration = time.time() - start
            print(f"  FAILED: {e}  ({duration:.1f}s)")
            results.append({
                'problem': model.modelfile,
                'index': i,
                'log_Z': None,
                'ecl': ecl,
                'duration_seconds': round(duration, 2),
                'status': f'error: {e}',
            })

    return results


def run_parallel(indices, base_config, gpus, ecl_override=None):
    """Run problems in parallel across multiple GPUs."""
    from nce.benchmark_problems.small_problems import small_problems, _AUTO_ECL

    problems = small_problems.problems
    config_json = json.dumps(base_config)
    tmp_dir = tempfile.mkdtemp(prefix='run_small_problems_')

    def _get_ecl(modelfile):
        return ecl_override if ecl_override is not None else _AUTO_ECL[modelfile]

    def _launch_worker(idx, gpu_id):
        result_path = os.path.join(tmp_dir, f'{idx}.json')
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        cmd = [sys.executable, 'scripts/run_small_problems_worker.py',
               '--problem-index', str(idx),
               '--config-json', config_json,
               '--result-path', result_path]
        if ecl_override is not None:
            cmd += ['--ecl-override', str(ecl_override)]
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        modelfile = problems[idx].modelfile
        ecl = _get_ecl(modelfile)
        print(f"  Started {modelfile} (ecl={ecl}) on GPU {gpu_id} [PID {proc.pid}]")
        return proc, result_path

    # Round-robin assignment
    gpu_queues = {g: deque() for g in gpus}
    for i, idx in enumerate(indices):
        gpu_id = gpus[i % len(gpus)]
        gpu_queues[gpu_id].append(idx)

    active = {}   # gpu_id -> (idx, proc, result_path)
    results = {}  # idx -> result dict
    completed = 0
    total = len(indices)

    print(f"\nParallel mode: {total} problems across GPUs {gpus}")
    print(f"Worker temp dir: {tmp_dir}")
    print(f"{'='*70}")

    # Launch initial workers
    for gpu_id in gpus:
        if gpu_queues[gpu_id]:
            idx = gpu_queues[gpu_id].popleft()
            proc, result_path = _launch_worker(idx, gpu_id)
            active[gpu_id] = (idx, proc, result_path)

    # Poll for completion
    while active:
        for gpu_id in list(active.keys()):
            idx, proc, result_path = active[gpu_id]
            retcode = proc.poll()
            if retcode is None:
                continue

            # Worker finished
            stdout = proc.stdout.read()
            stderr = proc.stderr.read()
            if stdout.strip():
                print(stdout.strip())
            if stderr.strip():
                # Only print actual errors, skip routine warnings
                for line in stderr.strip().split('\n'):
                    if 'error' in line.lower() or 'exception' in line.lower() or 'traceback' in line.lower():
                        print(f"  STDERR: {line}", file=sys.stderr)

            completed += 1
            modelfile = problems[idx].modelfile

            if os.path.exists(result_path):
                with open(result_path) as f:
                    result = json.load(f)
                results[idx] = result
                log_z_str = f"{result['log_Z']:.6f}" if result['log_Z'] is not None else 'FAILED'
                print(f"  [{completed}/{total}] {modelfile}: "
                      f"log_Z={log_z_str}  ({result['duration_seconds']:.1f}s) "
                      f"[GPU {gpu_id}]")
            else:
                print(f"  [{completed}/{total}] {modelfile}: FAILED (no result file) "
                      f"[GPU {gpu_id}]")
                results[idx] = {
                    'problem': modelfile,
                    'index': idx,
                    'log_Z': None,
                    'ecl': _get_ecl(modelfile),
                    'duration_seconds': 0,
                    'status': 'error: worker produced no result',
                }

            # Launch next on this GPU
            if gpu_queues[gpu_id]:
                next_idx = gpu_queues[gpu_id].popleft()
                proc, result_path = _launch_worker(next_idx, gpu_id)
                active[gpu_id] = (next_idx, proc, result_path)
            else:
                del active[gpu_id]

        if active:
            time.sleep(2)

    # Return in original index order
    return [results[idx] for idx in indices]


def print_summary(results):
    """Print summary table."""
    print(f"\n{'='*70}")
    print(f"{'Problem':<45} {'log_Z':>12}  {'Time':>8}")
    print(f"{'-'*45} {'-'*12}  {'-'*8}")
    for r in results:
        log_z_str = f"{r['log_Z']:.6f}" if r['log_Z'] is not None else 'FAILED'
        print(f"{r['problem']:<45} {log_z_str:>12}  {r['duration_seconds']:>7.1f}s")


def write_output(results, output_path):
    """Write results to JSON or CSV."""
    out_path = Path(output_path)
    if out_path.suffix == '.csv':
        import csv
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'problem', 'index', 'log_Z', 'ecl', 'duration_seconds', 'status'])
            writer.writeheader()
            writer.writerows(results)
    else:
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


def main():
    args = parse_args()

    with open(args.config_path) as f:
        base_config = yaml.safe_load(f)

    # Determine problem indices
    if args.problems:
        indices = [int(x.strip()) for x in args.problems.split(',')]
    else:
        # Need to know how many problems there are
        from nce.benchmark_problems.small_problems import small_problems
        indices = list(range(len(small_problems.problems)))

    print(f"Running {len(indices)} problems with config: {args.config_path}")
    print(f"{'='*70}")

    total_start = time.time()

    if args.gpus:
        gpus = [int(x.strip()) for x in args.gpus.split(',')]
        results = run_parallel(indices, base_config, gpus, ecl_override=args.ecl)
    else:
        results = run_sequential(indices, base_config, ecl_override=args.ecl)

    total_time = time.time() - total_start

    print_summary(results)
    print(f"\nTotal wall time: {total_time:.1f}s")

    if args.output:
        write_output(results, args.output)
    else:
        print(f"Use --output results.json or --output results.csv to save")


if __name__ == '__main__':
    main()
