#!/usr/bin/env python
"""Hard bucket selection pipeline: Phase 1 (parallel training) + Phase 2 (precomputation).

Phase 1: Distributes 24 small_problems across GPUs via subprocess workers.
          Each worker trains with error_tracking=True and writes per-bucket error data.
          Coordinator merges results and identifies hard buckets (abs_log_Z_err > threshold).

Phase 2: For each hard bucket, runs exact upstream elimination and caches
          factor tensors + exact forward/backward messages to .pt files.

Usage:
    python scripts/select_hard_buckets.py --top-n 0.1 --gpus 0,1,2,3
    python scripts/select_hard_buckets.py --skip-phase1 --top-n 0.05  # reuse Phase 1 results
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path


NUM_PROBLEMS = 24  # small_problems has exactly 24 models


# --- Phase 1: Selection ------------------------------------------------------

def _spawn_worker(problem_index, gpu_id, tmp_dir):
    """Spawn a single worker subprocess on the specified GPU. Returns (problem_index, gpu_id, proc, output_path)."""
    output_path = os.path.join(tmp_dir, f'problem_{problem_index}.json')

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    cmd = [
        sys.executable,
        'scripts/select_hard_buckets_worker.py',
        '--problem-index', str(problem_index),
        '--output-path', output_path,
        '--top-n', str(threshold),
    ]

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return (problem_index, gpu_id, proc, output_path)


def run_phase1(gpus, output_dir):
    """Spawn workers across GPUs (max 1 per GPU), merge results, identify hard buckets."""
    print(f"\n{'='*60}")
    print(f"Phase 1: Selection run ({NUM_PROBLEMS} problems across {len(gpus)} GPUs)")
    print(f"  Running at most 1 worker per GPU to avoid OOM")
    print(f"{'='*60}\n")

    # Create temp dir for individual worker output files
    tmp_dir = tempfile.mkdtemp(prefix='hard_bucket_selection_')
    print(f"Worker output dir: {tmp_dir}")

    # Build round-robin GPU assignment for all problems
    problem_gpu = [(i, gpus[i % len(gpus)]) for i in range(NUM_PROBLEMS)]

    # Group problems by GPU to run sequentially within each GPU
    from collections import deque
    gpu_queues = {g: deque() for g in gpus}
    for i, g in problem_gpu:
        gpu_queues[g].append(i)

    # Track all finished workers for merging
    finished = {}  # problem_index -> output_path
    failed = []
    active = {}  # gpu_id -> (problem_index, gpu_id, proc, output_path)

    # Start one worker per GPU
    for g in gpus:
        if gpu_queues[g]:
            pidx = gpu_queues[g].popleft()
            w = _spawn_worker(pidx, g, tmp_dir)
            active[g] = w
            print(f"  Spawned problem {pidx} on GPU {g} (PID {w[2].pid})")

    print(f"\nRunning {NUM_PROBLEMS} problems (max {len(gpus)} concurrent)...")

    while active:
        for g in list(active.keys()):
            pidx, gpu_id, proc, output_path = active[g]
            retcode = proc.poll()
            if retcode is not None:
                # Worker finished
                stdout_text = proc.stdout.read()
                stderr_text = proc.stderr.read()
                if stdout_text.strip():
                    print(f"  {stdout_text.strip()}")
                if retcode != 0:
                    failed.append(pidx)
                    print(f"  [Problem {pidx}] FAILED (exit {retcode}) on GPU {gpu_id}")
                    if stderr_text.strip():
                        lines = stderr_text.strip().split('\n')
                        for line in lines[-10:]:
                            print(f"    stderr: {line}")
                finished[pidx] = output_path

                # Launch next problem for this GPU if any remain
                if gpu_queues[g]:
                    next_pidx = gpu_queues[g].popleft()
                    w = _spawn_worker(next_pidx, g, tmp_dir)
                    active[g] = w
                    print(f"  Spawned problem {next_pidx} on GPU {g} (PID {w[2].pid})")
                else:
                    del active[g]

        if active:
            time.sleep(5)

    total_done = len(finished)
    print(f"\nPhase 1 complete: {total_done - len(failed)}/{total_done} succeeded, {len(failed)} failed")

    if failed:
        print(f"  Failed problem indices: {sorted(failed)}")

    # Merge worker results in problem order
    all_results = []
    for i in range(NUM_PROBLEMS):
        output_path = os.path.join(tmp_dir, f'problem_{i}.json')
        if os.path.exists(output_path):
            with open(output_path) as f:
                result = json.load(f)
            all_results.append(result)
        else:
            print(f"  WARNING: No output file for problem {i}")
            all_results.append({'problem_index': i, 'error': 'No output file', 'buckets': []})

    # Save merged results
    selection_results_path = os.path.join(output_dir, 'selection_results.json')
    with open(selection_results_path, 'w') as f:
        json.dump({
            'selection_strategy': f'top-{top_n}',
            'selection_date': datetime.now().isoformat(),
            'num_problems': NUM_PROBLEMS,
            'gpus': gpus,
            'results': all_results,
        }, f, indent=2)
    print(f"\nSaved full results to {selection_results_path}")

    return all_results


def identify_hard_buckets(all_results, top_n):
    """Select top-N hardest buckets by abs_log_Z_err across all problems."""
    all_buckets = []
    total_nn_buckets = 0

    # Collect all buckets with their errors
    for result in all_results:
        if 'error' in result and 'buckets' not in result:
            continue  # Skip failed problems
        problem_key = result.get('problem_key', result.get('model_file', 'unknown'))
        auto_ecl = result.get('auto_ecl', 0)
        model_file = result.get('model_file', '')

        for bucket in result.get('buckets', []):
            total_nn_buckets += 1
            final_err = bucket.get('final_abs_log_Z_err')
            if final_err is not None:
                all_buckets.append({
                    'problem_index': result['problem_index'],
                    'problem_key': problem_key,
                    'model_file': model_file,
                    'bucket_label': bucket['label'],
                    'selection_error': final_err,
                    'num_epochs': bucket['num_epochs'],
                    'auto_ecl': auto_ecl,
                })

    # Sort by error (descending) and take top N
    all_buckets.sort(key=lambda x: x['selection_error'], reverse=True)
    hard_buckets = all_buckets[:top_n]

    print(f"\n{'='*60}")
    print(f"Hard bucket identification (top-{top_n} by abs_log_Z_err)")
    print(f"{'='*60}")
    print(f"  Total NN buckets trained: {total_nn_buckets}")
    print(f"  Hard buckets selected: {len(hard_buckets)}")
    print(f"\nTop {len(hard_buckets)} hardest buckets:")
    for i, hb in enumerate(hard_buckets, 1):
        print(f"  {i:2d}. {hb['problem_key']:30s} bucket {hb['bucket_label']:4d}: "
              f"abs_log_Z_err={hb['selection_error']:.6f}")
    print()

    return hard_buckets, total_nn_buckets


# --- Phase 2: Precomputation -------------------------------------------------

def run_phase2(hard_buckets, output_dir):
    """For each hard bucket, compute exact messages and cache to .pt files."""
    import copy
    import torch
    from nce.benchmark_problems.small_problems import small_problems
    from nce.config_schema import prepare_config
    from nce.inference.graphical_model import FastGM
    from nce.utils.backward_message import get_backward_message

    print(f"\n{'='*60}")
    print(f"Phase 2: Precomputation ({len(hard_buckets)} hard buckets)")
    print(f"{'='*60}\n")

    cached_files = []

    for i, hb in enumerate(hard_buckets):
        problem_idx = hb['problem_index']
        bucket_label = hb['bucket_label']
        problem_key = hb['problem_key']
        auto_ecl = hb['auto_ecl']

        try:
            model = small_problems.problems[problem_idx]
            config = copy.deepcopy(small_problems.configs['default'][problem_idx])

            # Use exact elimination for upstream messages
            config = prepare_config(config)

            fastgm = FastGM(model=model, nn_config=config, device='cuda')

            # Eliminate up to (but not including) the target bucket, using exact computation
            target_var = fastgm.matching_var(bucket_label)
            if target_var is None:
                raise ValueError(f"No matching var for bucket_label={bucket_label} in problem {problem_key}")

            fastgm.eliminate_variables(up_to=target_var, exact=True)

            # Get the target bucket with upstream messages propagated
            bucket = fastgm.buckets[target_var]

            # Compute exact forward message
            exact_fw = bucket.compute_message_exact()

            # Compute exact backward message
            exact_bw, _ = get_backward_message(
                fastgm, bucket_label,
                iB=100, backward_ecl=2**30,
                return_factor_list=False,
            )

            # Extract factor tensors (raw tensors, not FastFactor objects)
            factors_data = []
            for f in bucket.factors:
                factors_data.append({
                    'tensor': f.tensor.detach().cpu(),
                    'labels': list(f.labels),
                })

            # Build scope and domain sizes from message scope
            scope = list(bucket.get_message_scope())
            domain_sizes = [fastgm.matching_var(v).states for v in scope]

            # Extract elimination variables
            elim_vars_data = []
            for v_label in bucket.elim_vars:
                var = fastgm.matching_var(v_label)
                elim_vars_data.append({
                    'label': int(v_label),
                    'states': int(var.states) if var else 0,
                })

            save_dict = {
                'factors': factors_data,
                'exact_fw': {
                    'tensor': exact_fw.tensor.detach().cpu(),
                    'labels': list(exact_fw.labels),
                },
                'exact_bw': {
                    'tensor': exact_bw.tensor.detach().cpu(),
                    'labels': list(exact_bw.labels),
                },
                'bucket_label': int(bucket_label),
                'scope': scope,
                'domain_sizes': domain_sizes,
                'elim_vars': elim_vars_data,
                'problem_key': problem_key,
                'model_file': hb['model_file'],
                'auto_ecl': auto_ecl,
                'selection_error': hb['selection_error'],
                'selection_epochs': hb['num_epochs'],
            }

            # Sanitize problem key for filename
            safe_key = problem_key.replace('/', '_').replace('.', '_')
            filename = f"{safe_key}__bucket_{bucket_label}.pt"
            filepath = os.path.join(output_dir, filename)

            torch.save(save_dict, filepath)
            cached_files.append({
                'file': filename,
                'problem_key': problem_key,
                'bucket_label': bucket_label,
                'selection_error': hb['selection_error'],
                'auto_ecl': auto_ecl,
            })

            fw_shape = list(exact_fw.tensor.shape)
            bw_shape = list(exact_bw.tensor.shape)
            print(f"  [{i+1}/{len(hard_buckets)}] {problem_key} bucket {bucket_label} "
                  f"— fw shape {fw_shape}, bw shape {bw_shape}")

            # Free GPU memory between problems
            del fastgm, bucket, exact_fw, exact_bw
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  [{i+1}/{len(hard_buckets)}] {problem_key} bucket {bucket_label} — FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nPhase 2 complete: {len(cached_files)}/{len(hard_buckets)} buckets cached")
    return cached_files


# --- Manifest generation -----------------------------------------------------

def write_manifest(cached_files, hard_buckets, output_dir, top_n, total_nn_buckets):
    """Write bucket_list.json manifest."""
    manifest_buckets = []
    for cf in cached_files:
        # Build bucket ID: sanitized_problem_key__bucket_label
        safe_key = cf['problem_key'].replace('/', '_').replace('.', '_')
        bucket_id = f"{safe_key}__{cf['bucket_label']}"

        manifest_buckets.append({
            'id': bucket_id,
            'problem_key': cf['problem_key'],
            'bucket_label': cf['bucket_label'],
            'selection_error': cf['selection_error'],
            'auto_ecl': cf['auto_ecl'],
            'file': cf['file'],
        })

    manifest = {
        'selection_strategy': f'top-{top_n}',
        'selection_date': datetime.now().isoformat(),
        'num_problems': NUM_PROBLEMS,
        'total_nn_buckets': total_nn_buckets,
        'buckets': manifest_buckets,
    }

    manifest_path = os.path.join(output_dir, 'bucket_list.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to {manifest_path}")
    print(f"  {len(manifest_buckets)} hard buckets catalogued")
    return manifest_path


# --- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Hard bucket selection pipeline: identify hard buckets and precompute cached data"
    )
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of hardest buckets to select for hard bucket selection (default: 0.1)')
    parser.add_argument('--gpus', type=str, default='0,1,2,3',
                        help='Comma-separated GPU IDs for Phase 1 workers (default: 0,1,2,3)')
    parser.add_argument('--output-dir', type=str, default='data/hard_buckets',
                        help='Output directory for cached .pt files and manifests')
    parser.add_argument('--skip-phase1', action='store_true',
                        help='Skip Phase 1 and reuse existing selection_results.json')
    args = parser.parse_args()

    gpus = [int(g.strip()) for g in args.gpus.split(',')]
    output_dir = args.output_dir

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # -- Phase 1 --
    if args.skip_phase1:
        selection_results_path = os.path.join(output_dir, 'selection_results.json')
        if not os.path.exists(selection_results_path):
            print(f"ERROR: --skip-phase1 specified but {selection_results_path} not found")
            sys.exit(1)
        print(f"Skipping Phase 1, loading results from {selection_results_path}")
        with open(selection_results_path) as f:
            data = json.load(f)
        all_results = data['results']
    else:
        all_results = run_phase1(gpus, output_dir)

    # -- Identify hard buckets --
    hard_buckets, total_nn_buckets = identify_hard_buckets(all_results, args.top_n)

    if not hard_buckets:
        print("\nNo hard buckets found. Consider lowering --top-n.")
        # Still write an empty manifest
        write_manifest([], hard_buckets, output_dir, args.top_n, total_nn_buckets)
        return

    # -- Phase 2 --
    cached_files = run_phase2(hard_buckets, output_dir)

    # -- Manifest --
    write_manifest(cached_files, hard_buckets, output_dir, args.top_n, total_nn_buckets)

    print(f"\n{'='*60}")
    print(f"Pipeline complete")
    print(f"{'='*60}")
    print(f"  Output dir: {output_dir}")
    print(f"  Hard buckets: {len(cached_files)}")
    print(f"  Threshold: {args.threshold}")


if __name__ == '__main__':
    main()
