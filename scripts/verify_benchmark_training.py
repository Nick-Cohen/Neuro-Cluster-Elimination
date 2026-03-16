#!/usr/bin/env python
"""End-to-end verification for the single-bucket benchmark training harness.

Exercises the full train_single_bucket() pipeline:
  1. Loads a .pt file (real from S01 or synthetic fallback)
  2. Runs training with a time limit
  3. Validates output folder contains correct plots and metrics

Exit 0 on success, exit 1 on failure with diagnostics.

Usage:
    python scripts/verify_benchmark_training.py --time-limit 30
    python scripts/verify_benchmark_training.py --device cpu --time-limit 15
"""
import argparse
import copy
import json
import os
import sys
import traceback

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_pt_file():
    """Look for an existing .pt file from S01's hard bucket pipeline.

    Checks:
      1. data/hard_buckets/bucket_list.json manifest → first listed .pt
      2. Any .pt file in data/hard_buckets/
    Returns path or None.
    """
    base_dir = os.path.join('data', 'hard_buckets')

    # Check manifest first
    manifest_path = os.path.join(base_dir, 'bucket_list.json')
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            # Expect a list of dicts with 'file' key, or a list of filenames
            for entry in manifest:
                fname = entry['file'] if isinstance(entry, dict) else str(entry)
                pt_path = os.path.join(base_dir, fname)
                if os.path.isfile(pt_path):
                    return pt_path
        except Exception:
            pass  # Fall through to directory scan

    # Direct directory scan
    if os.path.isdir(base_dir):
        for fname in sorted(os.listdir(base_dir)):
            if fname.endswith('.pt'):
                return os.path.join(base_dir, fname)

    return None


def _generate_synthetic_pt(output_path, device):
    """Generate a synthetic .pt file matching S01's schema.

    Uses smokers_20 (small_problems index 0) — the smallest/fastest
    problem. Runs exact elimination to find the first NN-eligible bucket,
    computes exact forward/backward messages, and saves in the .pt schema
    expected by train_single_bucket().

    Args:
        output_path: Where to write the .pt file.
        device: Device for computation ('cuda' or 'cpu').

    Returns:
        Path to the saved .pt file.
    """
    from nce.benchmark_problems.small_problems import small_problems
    from nce.config_schema import prepare_config
    from nce.inference.graphical_model import FastGM
    from nce.utils.backward_message import get_backward_message

    print("[Verify] Generating synthetic .pt from smokers_20...")

    model = small_problems.problems[0]
    config = copy.deepcopy(small_problems.configs['default'][0])
    config['device'] = device
    config = prepare_config(config, strict=False)

    fastgm = FastGM(model=model, nn_config=config, device=device)

    # Find the first NN-eligible bucket
    large_buckets = fastgm.get_large_message_buckets(ecl=config.get('ecl'))
    if not large_buckets:
        raise RuntimeError(
            "No NN-eligible buckets found in smokers_20 with "
            f"ecl={config.get('ecl')}. Cannot generate synthetic .pt."
        )

    # Use first NN-eligible bucket in elimination order
    target_label = None
    for var in fastgm.elim_order:
        if var.label in large_buckets:
            target_label = var.label
            break
    if target_label is None:
        target_label = large_buckets[0]

    print(f"[Verify] Target bucket label: {target_label}")

    # Eliminate up to the target bucket
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

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(save_dict, output_path)
    print(f"[Verify] Synthetic .pt saved to {output_path}")

    # Free GPU memory
    del fastgm, bucket, exact_fw, exact_bw
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_path


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

def _run_checks(result, output_dir):
    """Run all validation checks on the training result.

    Returns (passed: list[str], failed: list[tuple[str, str]]).
    Each failure is (check_name, diagnostic_message).
    """
    passed = []
    failed = []

    bucket_id = result.get('bucket_id', 'UNKNOWN')
    bucket_output_dir = os.path.join(output_dir, bucket_id)

    # --- Result dict checks ---

    # 1. epochs_completed > 0
    ec = result.get('epochs_completed', 0)
    if ec > 0:
        passed.append(f"epochs_completed={ec}")
    else:
        failed.append(("epochs_completed > 0",
                        f"Got {ec}"))

    # 2. error_tracking_data non-empty
    etd = result.get('error_tracking_data', [])
    if len(etd) > 0:
        passed.append(f"error_tracking_data has {len(etd)} entries")
    else:
        failed.append(("error_tracking_data non-empty",
                        "Got empty list"))

    # 3. wall_time > 0
    wt = result.get('wall_time', 0)
    if wt > 0:
        passed.append(f"wall_time={wt:.2f}s")
    else:
        failed.append(("wall_time > 0", f"Got {wt}"))

    # 4. losses non-empty
    losses = result.get('losses', [])
    if len(losses) > 0:
        passed.append(f"losses has {len(losses)} entries")
    else:
        failed.append(("losses non-empty", "Got empty list"))

    # --- Output folder checks ---

    # 5. Output folder exists
    if os.path.isdir(bucket_output_dir):
        passed.append(f"output dir exists: {bucket_output_dir}")
    else:
        failed.append(("output dir exists",
                        f"Not found: {bucket_output_dir}"))
        # Can't check files if dir doesn't exist
        return passed, failed

    # 6-7. PNG files exist and have non-zero size
    for png_name in ['loss.png', 'local_error.png']:
        png_path = os.path.join(bucket_output_dir, png_name)
        if os.path.isfile(png_path):
            size = os.path.getsize(png_path)
            if size > 0:
                passed.append(f"{png_name} exists ({size} bytes)")
            else:
                failed.append((f"{png_name} non-zero",
                                f"File exists but is 0 bytes"))
        else:
            failed.append((f"{png_name} exists",
                            f"Not found: {png_path}"))

    # 8. metrics.json exists, is parseable, and has required keys
    metrics_path = os.path.join(bucket_output_dir, 'metrics.json')
    if os.path.isfile(metrics_path):
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)

            required_keys = [
                'epochs_completed', 'final_loss', 'final_local_error',
                'error_tracking', 'losses', 'wall_time', 'config_hash',
                'bucket_metadata', 'timestamp',
            ]
            missing = [k for k in required_keys if k not in metrics]
            if not missing:
                passed.append(f"metrics.json valid with all {len(required_keys)} required keys")
            else:
                failed.append(("metrics.json keys",
                                f"Missing keys: {missing}"))
        except json.JSONDecodeError as e:
            failed.append(("metrics.json parseable",
                            f"JSON parse error: {e}"))
    else:
        failed.append(("metrics.json exists",
                        f"Not found: {metrics_path}"))

    return passed, failed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end verification of benchmark training harness")
    parser.add_argument('--device', default='cuda',
                        help='Device for training (default: cuda)')
    parser.add_argument('--time-limit', type=int, default=30,
                        help='Training time limit in seconds (default: 30)')
    parser.add_argument('--output-dir', default='/tmp/benchmark_training_verify',
                        help='Output directory (default: /tmp/benchmark_training_verify)')
    args = parser.parse_args()

    print(f"[Verify] === Benchmark Training Verification ===")
    print(f"[Verify] Device: {args.device}")
    print(f"[Verify] Time limit: {args.time_limit}s")
    print(f"[Verify] Output dir: {args.output_dir}")
    print()

    # -----------------------------------------------------------------------
    # Step 1: Find or generate .pt file
    # -----------------------------------------------------------------------
    pt_path = _find_pt_file()
    if pt_path is not None:
        print(f"[Verify] Found existing .pt file: {pt_path}")
        pt_source = "real"
    else:
        print("[Verify] No existing .pt files found — generating synthetic...")
        synthetic_dir = os.path.join(args.output_dir, '_synthetic')
        synthetic_path = os.path.join(synthetic_dir, 'synthetic_bucket.pt')
        pt_path = _generate_synthetic_pt(synthetic_path, args.device)
        pt_source = "synthetic"

    print(f"[Verify] Using .pt file: {pt_path} (source: {pt_source})")
    print()

    # -----------------------------------------------------------------------
    # Step 2: Build config and run training
    # -----------------------------------------------------------------------
    # Start from the full default config (has all required Trainer fields)
    # and override the fields relevant to verification.
    from nce.benchmark_problems.small_problems import small_problems as _sp
    nn_config = copy.deepcopy(_sp.configs['default'][0])
    nn_config.update({
        'loss_fn': 'unnormalized_kl',
        'hidden_sizes': [3, 3],
        'lr': 0.01,
        'num_epochs': 100000,
        'sampling_scheme': 'all',
        'batch_size': 100000,
        'set_size': 100000,
        'num_samples': 100000,
        'seed': 42,
        'device': args.device,
    })

    try:
        from nce.benchmark.training import train_single_bucket

        result = train_single_bucket(
            bucket_pt_path=pt_path,
            nn_config=nn_config,
            time_limit_seconds=args.time_limit,
            output_dir=args.output_dir,
            device=args.device,
        )
    except Exception as e:
        print(f"\n[Verify] FAIL: train_single_bucket() raised an exception:")
        traceback.print_exc()
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 3: Validate results
    # -----------------------------------------------------------------------
    print()
    print(f"[Verify] === Validation Checks ===")
    passed, failed = _run_checks(result, args.output_dir)

    # -----------------------------------------------------------------------
    # Step 4: Print summary
    # -----------------------------------------------------------------------
    print()
    for p in passed:
        print(f"  PASS: {p}")
    for name, diag in failed:
        print(f"  FAIL: {name} — {diag}")

    print()
    bucket_id = result.get('bucket_id', '?')
    final_loss = result.get('final_loss')
    final_error = result.get('final_local_error')
    wall_time = result.get('wall_time', 0)
    epochs = result.get('epochs_completed', 0)

    print(f"[Verify] Summary:")
    print(f"  Bucket ID:     {bucket_id}")
    print(f"  PT source:     {pt_source}")
    print(f"  Epochs:        {epochs}")
    print(f"  Final loss:    {final_loss}")
    print(f"  Final |err|:   {final_error}")
    print(f"  Wall time:     {wall_time:.2f}s")
    print(f"  Checks:        {len(passed)} passed, {len(failed)} failed")
    print()

    if failed:
        print(f"[Verify] FAIL — {len(failed)} check(s) failed")
        sys.exit(1)
    else:
        print(f"[Verify] PASS — all {len(passed)} checks passed")
        sys.exit(0)


if __name__ == '__main__':
    main()
