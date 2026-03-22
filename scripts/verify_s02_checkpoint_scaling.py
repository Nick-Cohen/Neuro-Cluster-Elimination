#!/usr/bin/env python
"""Integration verification for S02 checkpoint scaling.

Validates that small-batch training produces ≥5 checkpoints in error_tracking_data
when batch_size << message_size, proving the checkpoint scaling formula works end-to-end.

Usage:
    python scripts/verify_s02_checkpoint_scaling.py
    python scripts/verify_s02_checkpoint_scaling.py --device cpu
"""
import argparse
import copy
import json
import os
import sys
import traceback

import torch


def _find_large_bucket():
    """Find a .pt file with message_size ≥ 10000.

    Returns: (pt_path, message_size) or (None, 0) if not found.
    """
    base_dir = os.path.join('data', 'hard_buckets')

    # Try manifest first
    manifest_path = os.path.join(base_dir, 'bucket_list.json')
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            for entry in manifest:
                fname = entry['file'] if isinstance(entry, dict) else str(entry)
                pt_path = os.path.join(base_dir, fname)
                if os.path.isfile(pt_path):
                    # Load and check message_size
                    data = torch.load(pt_path, map_location='cpu', weights_only=False)
                    domain_sizes = data.get('domain_sizes', [])
                    message_size = 1
                    for ds in domain_sizes:
                        message_size *= ds
                    if message_size >= 10000:
                        return pt_path, message_size
        except Exception as e:
            print(f"[Verify] Warning: Failed to parse manifest: {e}")

    # Direct directory scan
    if os.path.isdir(base_dir):
        for fname in sorted(os.listdir(base_dir)):
            if fname.endswith('.pt'):
                pt_path = os.path.join(base_dir, fname)
                try:
                    data = torch.load(pt_path, map_location='cpu', weights_only=False)
                    domain_sizes = data.get('domain_sizes', [])
                    message_size = 1
                    for ds in domain_sizes:
                        message_size *= ds
                    if message_size >= 10000:
                        return pt_path, message_size
                except Exception as e:
                    print(f"[Verify] Warning: Failed to load {fname}: {e}")
                    continue

    return None, 0


def _generate_synthetic_large_bucket(output_path, device):
    """Generate a synthetic .pt file with message_size ≥ 10000.

    Uses the largest available problem from small_problems to maximize
    the chance of finding a large bucket. Falls back to a problem with
    variables of domain size 100, which should produce message_size ≥ 10000.

    Args:
        output_path: Where to write the .pt file.
        device: Device for computation ('cuda' or 'cpu').

    Returns:
        (pt_path, message_size) tuple.
    """
    from nce.benchmark_problems.small_problems import small_problems
    from nce.config_schema import prepare_config
    from nce.inference.graphical_model import FastGM

    print("[Verify] Generating synthetic bucket with message_size ≥ 10000...")

    # Try each problem in small_problems, starting from the end (largest)
    for idx in reversed(range(len(small_problems.problems))):
        model = small_problems.problems[idx]
        config = copy.deepcopy(small_problems.configs['default'][idx])
        config['device'] = device
        config['iB'] = 10  # Allow large messages
        config = prepare_config(config, strict=False)

        print(f"[Verify] Trying problem {idx}: {model.modelfile}")

        fastgm = FastGM(model=model, nn_config=config, device=device)

        # Find first bucket with message_size ≥ 10000
        for label, bucket in fastgm.buckets.items():
            # Compute message_size (product of domain sizes)
            scope = bucket.scope
            domain_sizes = [fastgm.variables[v].states for v in scope]
            message_size = 1
            for ds in domain_sizes:
                message_size *= ds

            if message_size >= 10000:
                print(f"[Verify] Found bucket with message_size={message_size}")

                # Compute exact messages
                exact_fw = bucket.compute_message_exact(forward=True)
                exact_bw = bucket.compute_message_exact(forward=False)

                # Build save dict
                factors_data = []
                for f in bucket.factors:
                    factors_data.append({
                        'tensor': f.tensor.detach().cpu(),
                        'labels': list(f.labels),
                    })

                elim_vars_data = list(bucket.elim_vars)

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
                    'bucket_label': int(label),
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

                return output_path, message_size

        # Problem didn't have large enough bucket, try next
        del fastgm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # If we get here, no problem had a large bucket
    raise RuntimeError("Could not find or generate a bucket with message_size ≥ 10000")


def _run_validation_checks(result, batch_size, message_size):
    """Run validation checks on the training result.

    Returns: (passed: list[str], failed: list[tuple[str, str]]).
    Each entry is (check_name, diagnostic).
    """
    passed = []
    failed = []

    error_tracking_data = result.get('error_tracking_data', [])
    epochs_completed = result.get('epochs_completed', 0)

    # Extract epoch numbers from error_tracking_data
    if error_tracking_data:
        epochs = [e[0] for e in error_tracking_data]
    else:
        epochs = []

    # Check 1: ≥5 checkpoints (MAIN GOAL)
    check_name = "len(error_tracking_data) ≥ 5"
    if len(error_tracking_data) >= 5:
        passed.append((check_name, f"Got {len(error_tracking_data)} checkpoints"))
    else:
        failed.append((check_name,
                       f"Expected ≥5, got {len(error_tracking_data)}. "
                       f"Scaling may not be working or time limit hit too early."))

    # Check 2: Epoch 0 present (baseline)
    check_name = "epoch 0 in error_tracking_data"
    if 0 in epochs:
        passed.append((check_name, "Epoch 0 (baseline) present"))
    else:
        failed.append((check_name,
                       f"Epoch 0 missing. Epochs: {epochs}"))

    # Check 3: Epochs are sorted
    check_name = "epochs sorted"
    if epochs == sorted(epochs):
        passed.append((check_name, "Epochs in ascending order"))
    else:
        failed.append((check_name,
                       f"Unsorted epochs: {epochs} vs {sorted(epochs)}"))

    # Check 4: Epochs are unique
    check_name = "epochs unique"
    if len(epochs) == len(set(epochs)):
        passed.append((check_name, "No duplicate epochs"))
    else:
        failed.append((check_name,
                       f"Duplicates found: {len(epochs)} total, {len(set(epochs))} unique"))

    # Check 5: All epochs <= epochs_completed
    check_name = "all epochs ≤ epochs_completed"
    invalid_epochs = [e for e in epochs if e > epochs_completed]
    if not invalid_epochs:
        passed.append((check_name, f"All epochs ≤ {epochs_completed}"))
    else:
        failed.append((check_name,
                       f"Found epochs > epochs_completed ({epochs_completed}): {invalid_epochs}"))

    # Compute diagnostic info
    from nce.neural_networks.train import get_error_tracking_epochs, get_scaled_error_tracking_epochs
    num_epochs = 500  # From config below
    base_schedule = get_error_tracking_epochs(num_epochs)
    scaled_schedule = get_scaled_error_tracking_epochs(num_epochs, batch_size, message_size)
    scaling_factor = batch_size / message_size

    return passed, failed, {
        'scaling_factor': scaling_factor,
        'base_schedule': base_schedule,
        'scaled_schedule': scaled_schedule,
        'actual_epochs': epochs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Integration verification for S02 checkpoint scaling"
    )
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--time-limit', type=int, default=60,
                        help='Time limit in seconds for training')
    parser.add_argument('--output-dir', type=str, default='output/verify_s02',
                        help='Output directory for training artifacts')
    args = parser.parse_args()

    print(f"[Verify] === S02 Checkpoint Scaling Integration Test ===")
    print(f"[Verify] Device: {args.device}")
    print(f"[Verify] Time limit: {args.time_limit}s")
    print()

    # -----------------------------------------------------------------------
    # Step 1: Find or generate bucket with message_size ≥ 10000
    # -----------------------------------------------------------------------
    pt_path, message_size = _find_large_bucket()

    if pt_path is not None:
        print(f"[Verify] Found existing bucket: {pt_path}")
        print(f"[Verify] message_size: {message_size}")
        pt_source = "real"
    else:
        print("[Verify] No existing large buckets found — generating synthetic...")
        synthetic_dir = os.path.join(args.output_dir, '_synthetic')
        synthetic_path = os.path.join(synthetic_dir, 'large_bucket.pt')
        try:
            pt_path, message_size = _generate_synthetic_large_bucket(synthetic_path, args.device)
            pt_source = "synthetic"
        except Exception as e:
            print(f"\n[Verify] FAIL: Could not generate synthetic bucket: {e}")
            traceback.print_exc()
            sys.exit(1)

    print(f"[Verify] Using bucket: {pt_path} (source: {pt_source})")
    print(f"[Verify] message_size: {message_size}")
    print()

    # -----------------------------------------------------------------------
    # Step 2: Build small-batch config and run training
    # -----------------------------------------------------------------------
    from nce.benchmark_problems.small_problems import small_problems as _sp
    nn_config = copy.deepcopy(_sp.configs['default'][0])

    # Small-batch configuration with adaptive batch_size
    # Target scaling_factor around 0.01-0.02 to produce 4-5 checkpoints from base schedule
    target_scaling_factor = 0.015
    batch_size = max(100, int(target_scaling_factor * message_size))
    
    nn_config.update({
        'batch_size': batch_size,
        'num_epochs': 500,
        'device': args.device,
        'loss_fn': 'unnormalized_kl',
        'hidden_sizes': [3, 3],
        'lr': 0.01,
        'sampling_scheme': 'all',
        'set_size': 100000,
        'num_samples': 100000,
        'seed': 42,
    })

    actual_scaling_factor = batch_size / message_size
    print(f"[Verify] Config: batch_size={batch_size}, num_epochs=500")
    print(f"[Verify] Scaling factor: {actual_scaling_factor:.6f} (target: {target_scaling_factor})")
    print()

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

    passed, failed, diagnostics = _run_validation_checks(result, batch_size, message_size)

    # Print per-check results
    print()
    for check_name, detail in passed:
        print(f"  ✅ PASS: {check_name}")
        print(f"           {detail}")

    for check_name, detail in failed:
        print(f"  ❌ FAIL: {check_name}")
        print(f"           {detail}")

    # -----------------------------------------------------------------------
    # Step 4: Print diagnostics
    # -----------------------------------------------------------------------
    print()
    print(f"[Verify] === Diagnostics ===")
    print(f"  Scaling factor:     {diagnostics['scaling_factor']:.6f}")
    print(f"  Base schedule:      {diagnostics['base_schedule']}")
    print(f"  Scaled schedule:    {diagnostics['scaled_schedule']}")
    print(f"  Actual epochs:      {diagnostics['actual_epochs']}")
    print(f"  Epochs completed:   {result.get('epochs_completed', 0)}")
    print(f"  Wall time:          {result.get('wall_time', 0):.2f}s")
    print()

    # -----------------------------------------------------------------------
    # Step 5: Final summary
    # -----------------------------------------------------------------------
    num_checkpoints = len(result.get('error_tracking_data', []))
    print(f"[Verify] === Summary ===")
    print(f"  Bucket source:      {pt_source}")
    print(f"  message_size:       {message_size}")
    print(f"  batch_size:         {batch_size}")
    print(f"  Checkpoints:        {num_checkpoints}")
    print(f"  Checks:             {len(passed)} passed, {len(failed)} failed")
    print()

    if failed:
        print(f"[Verify] FAIL — {len(failed)} check(s) failed")
        sys.exit(1)
    else:
        print(f"[Verify] PASS — all {len(passed)} checks passed")
        print(f"[Verify] Checkpoint scaling working correctly: {num_checkpoints} checkpoints produced")
        sys.exit(0)


if __name__ == '__main__':
    main()
