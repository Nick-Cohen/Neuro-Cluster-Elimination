"""Verify float64 precision mode works end-to-end.

Runs train_single_bucket on the same bucket in float32 and float64 modes,
then compares results and checks that caching works correctly.
"""

import os
import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, '/home/cohenn1/NCE')

from nce.benchmark.training import train_single_bucket

# Pick a small bucket for fast testing
BUCKET_PT = '/home/cohenn1/NCE/data/hard_buckets/or_chain_10_fg_uai__bucket_60.pt'
OUTPUT_BASE = '/home/cohenn1/NCE/data/benchmark_output/float64_verify'
DEVICE = 'cuda'
TIME_LIMIT = 30  # 30 seconds is plenty for verification

BASE_CONFIG = {
    'num_epochs': 200,
    'loss_fn': 'logspace_mse_fdb',
    'num_samples': 10000,
    'batch_size': 100000,
    'hidden_sizes': [],
    'lr': 0.001,
    'seed': 42,
    'use_bw_approx': False,
    'use_amp': True,
    'sampling_scheme': 'all',
}


def run_test(use_float64, label):
    """Run train_single_bucket with given float64 setting."""
    config = dict(BASE_CONFIG)
    config['use_float64'] = use_float64
    if use_float64:
        config['use_amp'] = False  # AMP incompatible with float64

    output_dir = os.path.join(OUTPUT_BASE, label)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running: {label} (use_float64={use_float64})")
    print(f"{'='*60}")

    result = train_single_bucket(
        bucket_pt_path=BUCKET_PT,
        nn_config=config,
        time_limit_seconds=TIME_LIMIT,
        output_dir=output_dir,
        device=DEVICE,
    )
    return result


def check_dtypes_in_pipeline(use_float64):
    """Verify tensor dtypes throughout the pipeline."""
    from nce.benchmark.training import _load_bucket_data
    from nce.utils.dtype_utils import get_dtype

    config = {'use_float64': use_float64}
    dtype = get_dtype(config)
    expected = torch.float64 if use_float64 else torch.float32

    print(f"\n--- Dtype check (use_float64={use_float64}) ---")
    print(f"  get_dtype() returns: {dtype}")
    assert dtype == expected, f"Expected {expected}, got {dtype}"

    bucket_data = _load_bucket_data(BUCKET_PT, DEVICE, dtype=dtype)

    # Check factor dtypes
    for i, f in enumerate(bucket_data['factors']):
        assert f.tensor.dtype == expected, \
            f"Factor {i} dtype {f.tensor.dtype} != {expected}"
    print(f"  All {len(bucket_data['factors'])} factors: {expected} OK")

    # Check exact message dtypes
    assert bucket_data['exact_fw'].tensor.dtype == expected
    assert bucket_data['exact_bw'].tensor.dtype == expected
    print(f"  exact_fw/exact_bw: {expected} OK")

    return True


def check_f64_cache():
    """Check that .f64.pt cache file was created."""
    pt_path = Path(BUCKET_PT)
    f64_path = pt_path.with_suffix('.f64.pt')
    exists = f64_path.exists()
    print(f"\n--- Float64 cache check ---")
    print(f"  {f64_path}: {'EXISTS' if exists else 'MISSING'}")
    if exists:
        size_mb = f64_path.stat().st_size / (1024 * 1024)
        print(f"  Size: {size_mb:.2f} MB")
    return exists


def check_cache_reuse():
    """Verify second float64 load uses cache (no conversion message)."""
    from nce.benchmark.training import _load_bucket_data
    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        _load_bucket_data(BUCKET_PT, DEVICE, dtype=torch.float64)
    output = f.getvalue()

    uses_cache = "Loading cached float64 data" in output
    print(f"\n--- Cache reuse check ---")
    print(f"  Second load uses cache: {uses_cache}")
    if not uses_cache:
        print(f"  Output was: {output}")
    return uses_cache


def compare_results(r32, r64):
    """Compare float32 vs float64 results."""
    print(f"\n{'='*60}")
    print("COMPARISON: float32 vs float64")
    print(f"{'='*60}")

    print(f"  Epochs completed: f32={r32['epochs_completed']}, f64={r64['epochs_completed']}")
    print(f"  Final loss:       f32={r32['final_loss']:.8e}, f64={r64['final_loss']:.8e}")
    print(f"  Final error:      f32={r32['final_local_error']:.8e}, f64={r64['final_local_error']:.8e}")
    print(f"  Wall time:        f32={r32['wall_time']:.2f}s, f64={r64['wall_time']:.2f}s")

    if r32['error_tracking_data'] and r64['error_tracking_data']:
        print(f"\n  Error tracking comparison (epoch, f32_err, f64_err, diff):")
        # Match by epoch
        f32_by_epoch = {t[0]: t for t in r32['error_tracking_data']}
        f64_by_epoch = {t[0]: t for t in r64['error_tracking_data']}
        common_epochs = sorted(set(f32_by_epoch.keys()) & set(f64_by_epoch.keys()))
        for epoch in common_epochs:
            e32 = f32_by_epoch[epoch][3]  # abs_log_z_err
            e64 = f64_by_epoch[epoch][3]
            diff = e64 - e32
            print(f"    epoch {epoch:>5d}: f32={e32:.8e}, f64={e64:.8e}, diff={diff:+.8e}")


def main():
    print("Float64 Precision Mode Verification")
    print("="*60)

    # Step 1: Dtype checks
    print("\n[Step 1] Verifying dtype resolution...")
    check_dtypes_in_pipeline(use_float64=False)
    check_dtypes_in_pipeline(use_float64=True)

    # Step 2: Run float32 baseline
    print("\n[Step 2] Running float32 baseline...")
    r32 = run_test(use_float64=False, label='float32')

    # Step 3: Run float64
    print("\n[Step 3] Running float64...")
    r64 = run_test(use_float64=True, label='float64')

    # Step 4: Check .f64.pt cache was created
    print("\n[Step 4] Checking float64 cache...")
    cache_ok = check_f64_cache()

    # Step 5: Verify cache reuse on second load
    print("\n[Step 5] Verifying cache reuse...")
    reuse_ok = check_cache_reuse()

    # Step 6: Compare results
    compare_results(r32, r64)

    # Summary
    print(f"\n{'='*60}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Float32 training:  {'PASS' if r32['epochs_completed'] > 0 else 'FAIL'}")
    print(f"  Float64 training:  {'PASS' if r64['epochs_completed'] > 0 else 'FAIL'}")
    print(f"  Float64 cache:     {'PASS' if cache_ok else 'FAIL'}")
    print(f"  Cache reuse:       {'PASS' if reuse_ok else 'FAIL'}")

    all_pass = (r32['epochs_completed'] > 0 and r64['epochs_completed'] > 0
                and cache_ok and reuse_ok)
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
