#!/usr/bin/env python
"""End-to-end verification for the multi-GPU bucket benchmark CLI.

Tests the full M004 pipeline:
  1. Finds real .pt files from S01 or generates synthetic 2-bucket data
  2. Runs bucket_benchmark.py twice with different configs
  3. Validates history.jsonl has 2 entries with correct schema
  4. Validates comparison chart generated and shows both runs
  5. Validates per-bucket output folders with loss.png, error.png, metrics.json

Exit 0 on success, exit 1 on failure with diagnostics.

Usage:
    python scripts/verify_bucket_benchmark.py
"""
import argparse
import copy
import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

# GPU guard for automatic execution on deepreasoning if needed
try:
    from scripts.gpu_guard import ensure_gpu_server
    ensure_gpu_server()
except ImportError:
    # gpu_guard not available — assume environment is correct
    pass


# ---------------------------------------------------------------------------
# Helpers for finding/generating .pt files
# ---------------------------------------------------------------------------

def _find_pt_files():
    """Look for existing .pt files from S01's hard bucket pipeline.

    Checks data/hard_buckets/bucket_list.json manifest and returns
    a list of 1-2 bucket entries, or None if not found.

    Returns:
        list[dict] or None: List of bucket entries with keys:
            - bucket_id: str
            - file: str (relative filename)
            - problem_key: str
            - bucket_label: int
            - selection_error: float
    """
    base_dir = os.path.join('data', 'hard_buckets')
    manifest_path = os.path.join(base_dir, 'bucket_list.json')

    if not os.path.isfile(manifest_path):
        return None

    try:
        with open(manifest_path) as f:
            bucket_list = json.load(f)

        if not isinstance(bucket_list, list) or len(bucket_list) == 0:
            return None

        # Return first 2 entries
        return bucket_list[:2]

    except Exception:
        return None


def _generate_synthetic_pt(base_dir, device='cuda'):
    """Generate synthetic .pt files for 2 buckets from smokers_20.

    Creates bucket_list.json manifest with 2 entries and writes
    corresponding .pt files.

    Args:
        base_dir: str path to data/hard_buckets
        device: str 'cuda' or 'cpu'

    Returns:
        list[dict]: List of 2 bucket entries matching S01 schema
    """
    from nce.benchmark_problems.small_problems import small_problems
    from nce.config_schema import prepare_config
    from nce.inference.graphical_model import FastGM
    from nce.utils.backward_message import get_backward_message

    print("[Verify] Generating synthetic .pt files from smokers_20...")

    model = small_problems.problems[0]  # smokers_20
    config = copy.deepcopy(small_problems.configs['default'][0])
    config['device'] = device
    config = prepare_config(config, strict=False)

    fastgm = FastGM(model=model, nn_config=config, device=device)

    # Find NN-eligible buckets
    large_buckets = fastgm.get_large_message_buckets(ecl=config.get('ecl'))
    if len(large_buckets) < 2:
        raise RuntimeError(
            f"Need at least 2 NN-eligible buckets, found {len(large_buckets)} "
            f"in smokers_20 with ecl={config.get('ecl')}"
        )

    # Use first 2 NN-eligible buckets in elimination order
    target_labels = []
    for var in fastgm.elim_order:
        if var.label in large_buckets:
            target_labels.append(var.label)
            if len(target_labels) == 2:
                break

    if len(target_labels) < 2:
        target_labels = large_buckets[:2]

    print(f"[Verify] Target bucket labels: {target_labels}")

    bucket_entries = []

    for i, target_label in enumerate(target_labels):
        # Reset fastgm for each bucket
        fastgm = FastGM(model=model, nn_config=config, device=device)

        target_var = fastgm.matching_var(target_label)
        fastgm.eliminate_variables(up_to=target_var, exact=True)
        bucket = fastgm.buckets[target_var]

        # Exact forward message
        exact_fw = bucket.compute_message_exact()

        # Exact backward message
        exact_bw, _ = get_backward_message(
            fastgm, target_label,
            iB=100, backward_ecl=2**30,
            return_factor_list=False,
        )

        # Extract factor data
        factors_data = []
        for f in bucket.factors:
            factors_data.append({
                'tensor': f.tensor.detach().cpu(),
                'labels': list(f.labels),
            })

        scope = list(bucket.get_message_scope())
        domain_sizes = [fastgm.matching_var(v).states for v in scope]

        elim_vars_data = []
        for v_label in bucket.elim_vars:
            var = fastgm.matching_var(v_label)
            elim_vars_data.append({
                'label': int(v_label),
                'states': int(var.states) if var else 0,
            })

        # Build sanitized bucket_id (must match training.py logic)
        # training.py uses safe_key = problem_key.replace('/', '_').replace('.', '_')
        # So we need to store problem_key WITH the .uai extension
        problem_key = model.modelfile  # Keep full 'smokers_20.uai'
        safe_key = problem_key.replace('/', '_').replace('.', '_')
        bucket_id = f"{safe_key}__bucket_{target_label}"

        # File naming uses simplified version (no .uai in filename)
        problem_key_simple = os.path.splitext(os.path.basename(problem_key))[0]
        filename = f"{problem_key_simple}_bucket_{target_label}.pt"
        pt_path = os.path.join(base_dir, filename)

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
            'bucket_label': int(target_label),
            'scope': scope,
            'domain_sizes': domain_sizes,
            'elim_vars': elim_vars_data,
            'problem_key': model.modelfile,
            'auto_ecl': config.get('ecl', 262143),
        }

        import torch
        torch.save(save_dict, pt_path)
        print(f"[Verify] Saved {pt_path}")

        # Build manifest entry (matches S01 schema)
        # bucket_id should be consistent with training.py transformation
        bucket_entries.append({
            'bucket_id': bucket_id,  # e.g., 'smokers_20_uai__bucket_400'
            'file': filename,
            'problem_key': problem_key,  # Keep full 'smokers_20.uai'
            'bucket_label': int(target_label),
            'selection_error': 0.0,  # Synthetic — no real selection error
        })

        # Free GPU memory
        del fastgm, bucket, exact_fw, exact_bw
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Write bucket_list.json
    manifest_path = os.path.join(base_dir, 'bucket_list.json')
    with open(manifest_path, 'w') as f:
        json.dump(bucket_entries, f, indent=2)
    print(f"[Verify] Wrote {manifest_path}")

    return bucket_entries


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def _create_configs(temp_dir):
    """Create two different nn_configs as YAML files.

    Config A: UKL loss + [3, 3] hidden
    Config B: UKL loss + [5, 5] hidden

    Args:
        temp_dir: str path to temp directory

    Returns:
        tuple: (config_a_path, config_b_path)
    """
    from nce.benchmark_problems.small_problems import small_problems

    # Start from full default config (has all Trainer fields)
    base_config = copy.deepcopy(small_problems.configs['default'][0])

    # Config A: [3, 3] hidden
    config_a = copy.deepcopy(base_config)
    config_a.update({
        'loss_fn': 'unnormalized_kl',
        'hidden_sizes': [3, 3],
        'lr': 0.01,
        'num_epochs': 100000,
        'sampling_scheme': 'all',
        'batch_size': 100000,
        'set_size': 100000,
        'num_samples': 100000,
        'seed': 42,
        'device': 'cuda',
    })

    # Config B: [5, 5] hidden
    config_b = copy.deepcopy(base_config)
    config_b.update({
        'loss_fn': 'unnormalized_kl',
        'hidden_sizes': [5, 5],
        'lr': 0.01,
        'num_epochs': 100000,
        'sampling_scheme': 'all',
        'batch_size': 100000,
        'set_size': 100000,
        'num_samples': 100000,
        'seed': 42,
        'device': 'cuda',
    })

    config_a_path = os.path.join(temp_dir, 'config_a.yaml')
    config_b_path = os.path.join(temp_dir, 'config_b.yaml')

    import yaml
    with open(config_a_path, 'w') as f:
        yaml.dump(config_a, f)
    with open(config_b_path, 'w') as f:
        yaml.dump(config_b, f)

    print(f"[Verify] Created config A: {config_a_path}")
    print(f"[Verify] Created config B: {config_b_path}")

    return config_a_path, config_b_path


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------

def _run_benchmark(config_path, mode, gpus, output_dir):
    """Run bucket_benchmark.py with subprocess.run().

    Args:
        config_path: str path to YAML config
        mode: str 'fast' or 'slow'
        gpus: str comma-separated GPU IDs
        output_dir: str output directory

    Returns:
        subprocess.CompletedProcess result
    """
    cmd = [
        sys.executable,
        'scripts/bucket_benchmark.py',
        config_path,
        mode,
        '--gpus', gpus,
        '--output-dir', output_dir,
    ]

    print(f"[Verify] Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Print output for diagnostics
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    return result


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

def _run_checks(history_path, comparison_chart_pattern, output_dir, bucket_entries):
    """Run all validation checks on benchmark outputs.

    Returns:
        tuple: (passed: list[str], failed: list[tuple[str, str]])
    """
    passed = []
    failed = []

    # -----------------------------------------------------------------------
    # Check 1: history.jsonl exists and is readable
    # -----------------------------------------------------------------------
    if not os.path.isfile(history_path):
        failed.append(("history.jsonl exists",
                       f"Not found: {history_path}"))
        return passed, failed  # Can't proceed without history file
    else:
        passed.append(f"history.jsonl exists: {history_path}")

    # -----------------------------------------------------------------------
    # Check 2: history.jsonl has exactly 2 entries
    # -----------------------------------------------------------------------
    try:
        with open(history_path) as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        failed.append(("history.jsonl readable",
                       f"Could not read file: {e}"))
        return passed, failed

    if len(lines) != 2:
        failed.append(("history.jsonl has 2 entries",
                       f"Found {len(lines)} entries"))
        return passed, failed
    else:
        passed.append(f"history.jsonl has 2 entries")

    # -----------------------------------------------------------------------
    # Check 3: Both entries parseable as JSON with required keys
    # -----------------------------------------------------------------------
    required_keys = [
        'run_id', 'timestamp', 'config_hash', 'config',
        'mode', 'time_limit_per_bucket', 'buckets',
    ]
    per_bucket_required = [
        'bucket_id', 'pt_path', 'epochs_completed', 'wall_time',
        'final_loss', 'error_tracking_data', 'final_local_error',
    ]

    history_records = []
    for i, line in enumerate(lines, 1):
        try:
            record = json.loads(line)
            history_records.append(record)
        except json.JSONDecodeError as e:
            failed.append((f"entry {i} parseable JSON",
                           f"JSON decode error: {e}"))
            return passed, failed

        # Check top-level keys
        missing = [k for k in required_keys if k not in record]
        if missing:
            failed.append((f"entry {i} required keys",
                           f"Missing keys: {missing}"))
        else:
            passed.append(f"entry {i} has all {len(required_keys)} required keys")

        # Check per-bucket keys
        buckets = record.get('buckets', [])
        if not isinstance(buckets, list) or len(buckets) == 0:
            failed.append((f"entry {i} buckets list non-empty",
                           f"buckets is {type(buckets).__name__} with length {len(buckets)}"))
        else:
            passed.append(f"entry {i} buckets list has {len(buckets)} entries")

            for j, bucket_result in enumerate(buckets, 1):
                missing_bucket = [k for k in per_bucket_required if k not in bucket_result]
                if missing_bucket:
                    failed.append((f"entry {i} bucket {j} required keys",
                                   f"Missing keys: {missing_bucket}"))

    if not failed:
        passed.append(f"all per-bucket records have required keys")

    # -----------------------------------------------------------------------
    # Check 4: config_hash is 32-char hex
    # -----------------------------------------------------------------------
    hex_pattern = re.compile(r'^[0-9a-f]{32}$')
    for i, record in enumerate(history_records, 1):
        config_hash = record.get('config_hash', '')
        if hex_pattern.match(config_hash):
            passed.append(f"entry {i} config_hash valid: {config_hash}")
        else:
            failed.append((f"entry {i} config_hash format",
                           f"Expected 32-char hex, got: {config_hash}"))

    # -----------------------------------------------------------------------
    # Check 5: timestamp is valid ISO 8601
    # -----------------------------------------------------------------------
    for i, record in enumerate(history_records, 1):
        timestamp = record.get('timestamp', '')
        try:
            datetime.fromisoformat(timestamp)
            passed.append(f"entry {i} timestamp valid: {timestamp}")
        except ValueError:
            failed.append((f"entry {i} timestamp ISO 8601",
                           f"Invalid format: {timestamp}"))

    # -----------------------------------------------------------------------
    # Check 6: error_tracking_data is dict with epoch keys
    # -----------------------------------------------------------------------
    for i, record in enumerate(history_records, 1):
        buckets = record.get('buckets', [])
        for j, bucket_result in enumerate(buckets, 1):
            etd = bucket_result.get('error_tracking_data', {})
            if not isinstance(etd, dict):
                failed.append((f"entry {i} bucket {j} error_tracking_data type",
                               f"Expected dict, got {type(etd).__name__}"))
            elif len(etd) == 0:
                failed.append((f"entry {i} bucket {j} error_tracking_data non-empty",
                               f"Dict is empty"))
            else:
                # Check that keys are epoch numbers (integers as strings)
                try:
                    epochs = [int(k) for k in etd.keys()]
                    passed.append(f"entry {i} bucket {j} error_tracking_data has {len(epochs)} epochs")
                except ValueError as e:
                    failed.append((f"entry {i} bucket {j} error_tracking_data keys",
                                   f"Expected integer epoch keys: {e}"))

    # -----------------------------------------------------------------------
    # Check 7: Per-bucket output folders exist
    # -----------------------------------------------------------------------
    for i, record in enumerate(history_records, 1):
        buckets = record.get('buckets', [])
        for j, bucket_result in enumerate(buckets, 1):
            bucket_id = bucket_result.get('bucket_id', 'UNKNOWN')
            bucket_output_dir = os.path.join(output_dir, bucket_id)
            if os.path.isdir(bucket_output_dir):
                passed.append(f"entry {i} bucket {j} output dir exists: {bucket_output_dir}")
            else:
                failed.append((f"entry {i} bucket {j} output dir exists",
                               f"Not found: {bucket_output_dir}"))

    # -----------------------------------------------------------------------
    # Check 8: loss.png, local_error.png, metrics.json exist and valid
    # -----------------------------------------------------------------------
    for i, record in enumerate(history_records, 1):
        buckets = record.get('buckets', [])
        for j, bucket_result in enumerate(buckets, 1):
            bucket_id = bucket_result.get('bucket_id', 'UNKNOWN')
            bucket_output_dir = os.path.join(output_dir, bucket_id)

            if not os.path.isdir(bucket_output_dir):
                continue  # Already failed in check 7

            # Check PNG files
            for png_name in ['loss.png', 'local_error.png']:
                png_path = os.path.join(bucket_output_dir, png_name)
                if os.path.isfile(png_path):
                    size = os.path.getsize(png_path)
                    if size > 0:
                        passed.append(f"entry {i} bucket {j} {png_name} exists ({size} bytes)")
                    else:
                        failed.append((f"entry {i} bucket {j} {png_name} non-zero",
                                       f"File is 0 bytes"))
                else:
                    failed.append((f"entry {i} bucket {j} {png_name} exists",
                                   f"Not found: {png_path}"))

            # Check metrics.json
            metrics_path = os.path.join(bucket_output_dir, 'metrics.json')
            if os.path.isfile(metrics_path):
                try:
                    with open(metrics_path) as f:
                        metrics = json.load(f)

                    metrics_required = [
                        'epochs_completed', 'final_loss', 'final_local_error',
                        'error_tracking', 'losses', 'wall_time', 'config_hash',
                        'bucket_metadata', 'timestamp',
                    ]
                    missing_metrics = [k for k in metrics_required if k not in metrics]
                    if not missing_metrics:
                        passed.append(f"entry {i} bucket {j} metrics.json valid")
                    else:
                        failed.append((f"entry {i} bucket {j} metrics.json keys",
                                       f"Missing keys: {missing_metrics}"))
                except json.JSONDecodeError as e:
                    failed.append((f"entry {i} bucket {j} metrics.json parseable",
                                   f"JSON parse error: {e}"))
            else:
                failed.append((f"entry {i} bucket {j} metrics.json exists",
                               f"Not found: {metrics_path}"))

    # -----------------------------------------------------------------------
    # Check 9: Comparison chart PNGs exist with non-zero size
    # -----------------------------------------------------------------------
    # Pattern: data/hard_buckets/comparison_run_*.png
    comparison_dir = os.path.dirname(history_path)
    comparison_charts = []
    if os.path.isdir(comparison_dir):
        for fname in os.listdir(comparison_dir):
            if fname.startswith('comparison_run_') and fname.endswith('.png'):
                comparison_charts.append(os.path.join(comparison_dir, fname))

    if len(comparison_charts) >= 2:
        passed.append(f"found {len(comparison_charts)} comparison chart(s)")
        for chart_path in comparison_charts:
            size = os.path.getsize(chart_path)
            if size > 0:
                passed.append(f"comparison chart exists: {chart_path} ({size} bytes)")
            else:
                failed.append((f"comparison chart non-zero: {chart_path}",
                               f"File is 0 bytes"))
    else:
        failed.append(("comparison charts exist",
                       f"Expected at least 2, found {len(comparison_charts)}"))

    return passed, failed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end verification of bucket benchmark CLI"
    )
    parser.add_argument('--device', default='cpu',
                        help='Device for training (default: cpu for verification, '
                             'use cuda for production)')
    parser.add_argument('--output-dir', default='/tmp/benchmark_verify_output',
                        help='Output directory (default: /tmp/benchmark_verify_output)')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"[Verify] === Bucket Benchmark End-to-End Verification ===")
    print(f"[Verify] Device: {args.device}")
    print(f"[Verify] Output dir: {args.output_dir}")
    print(f"{'='*70}\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Find or generate .pt files
    # -----------------------------------------------------------------------
    print(f"[Verify] === Step 1: Find or generate .pt files ===")
    bucket_entries = _find_pt_files()

    if bucket_entries is not None:
        print(f"[Verify] Found {len(bucket_entries)} existing bucket(s) from S01")
        pt_source = "real"
    else:
        print(f"[Verify] No existing buckets found — generating synthetic...")
        base_dir = os.path.join('data', 'hard_buckets')
        os.makedirs(base_dir, exist_ok=True)
        bucket_entries = _generate_synthetic_pt(base_dir, args.device)
        pt_source = "synthetic"

    print(f"[Verify] Using {len(bucket_entries)} bucket(s) (source: {pt_source})")
    for entry in bucket_entries:
        print(f"[Verify]   - {entry['bucket_id']}: {entry['file']}")
    print()

    # -----------------------------------------------------------------------
    # Step 2: Create two different configs
    # -----------------------------------------------------------------------
    print(f"[Verify] === Step 2: Create configs ===")
    temp_dir = tempfile.mkdtemp(prefix='verify_bucket_benchmark_')
    print(f"[Verify] Temp dir: {temp_dir}")
    config_a_path, config_b_path = _create_configs(temp_dir)
    print()

    # -----------------------------------------------------------------------
    # Step 3: First benchmark run
    # -----------------------------------------------------------------------
    print(f"[Verify] === Step 3: First benchmark run (config A) ===")
    result_a = _run_benchmark(
        config_a_path,
        mode='fast',
        gpus='0',
        output_dir=args.output_dir,
    )

    if result_a.returncode != 0:
        print(f"[Verify] FAIL: First benchmark run exited with code {result_a.returncode}",
              file=sys.stderr)
        sys.exit(1)

    print(f"[Verify] First benchmark run completed successfully\n")

    # -----------------------------------------------------------------------
    # Step 4: Second benchmark run
    # -----------------------------------------------------------------------
    print(f"[Verify] === Step 4: Second benchmark run (config B) ===")
    result_b = _run_benchmark(
        config_b_path,
        mode='fast',
        gpus='0',
        output_dir=args.output_dir,
    )

    if result_b.returncode != 0:
        print(f"[Verify] FAIL: Second benchmark run exited with code {result_b.returncode}",
              file=sys.stderr)
        sys.exit(1)

    print(f"[Verify] Second benchmark run completed successfully\n")

    # -----------------------------------------------------------------------
    # Step 5: Run validation checks
    # -----------------------------------------------------------------------
    print(f"[Verify] === Step 5: Validation checks ===")
    history_path = os.path.join('data', 'hard_buckets', 'history.jsonl')
    comparison_chart_pattern = os.path.join('data', 'hard_buckets', 'comparison_*.png')

    passed, failed = _run_checks(
        history_path,
        comparison_chart_pattern,
        args.output_dir,
        bucket_entries,
    )

    # -----------------------------------------------------------------------
    # Step 6: Print summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"[Verify] === Validation Summary ===")
    print(f"{'='*70}")

    for p in passed:
        print(f"  ✓ PASS: {p}")

    if failed:
        print()
        for name, diag in failed:
            print(f"  ✗ FAIL: {name} — {diag}")

    print()
    print(f"[Verify] Total checks: {len(passed)} passed, {len(failed)} failed")
    print(f"{'='*70}\n")

    if failed:
        print(f"[Verify] VERIFICATION FAILED — {len(failed)} check(s) failed")
        sys.exit(1)
    else:
        print(f"[Verify] VERIFICATION PASSED — all {len(passed)} checks passed")
        sys.exit(0)


if __name__ == '__main__':
    main()
