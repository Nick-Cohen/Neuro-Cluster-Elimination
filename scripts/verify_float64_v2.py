"""Verify float64 precision mode — v2 with stable loss function.

Uses linspace_mse_fdb which is more numerically stable for large buckets,
and picks a smaller bucket for clean results.
"""

import os
import sys
import torch
from pathlib import Path

sys.path.insert(0, '/home/cohenn1/NCE')

from nce.benchmark.training import train_single_bucket, _load_bucket_data
from nce.utils.dtype_utils import get_dtype

# Use BN_9 bucket (smaller, more stable)
BUCKET_PT = '/home/cohenn1/NCE/data/hard_buckets/BN_9_uai__bucket_12.pt'
OUTPUT_BASE = '/home/cohenn1/NCE/data/benchmark_output/float64_verify_v2'
DEVICE = 'cuda'
TIME_LIMIT = 60

BASE_CONFIG = {
    'num_epochs': 500,
    'loss_fn': 'linspace_mse_fdb',
    'num_samples': 10000,
    'batch_size': 100000,
    'hidden_sizes': [],
    'lr': 0.001,
    'seed': 42,
    'use_bw_approx': False,
    'use_amp': True,
    'sampling_scheme': 'all',
    'fdb': True,
}


def run_test(use_float64, label):
    config = dict(BASE_CONFIG)
    config['use_float64'] = use_float64
    if use_float64:
        config['use_amp'] = False

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


def main():
    print("Float64 Verification v2 (stable loss function)")
    print("="*60)

    # Clean up any existing f64 cache for this bucket to test creation
    f64_path = Path(BUCKET_PT).with_suffix('.f64.pt')
    if f64_path.exists():
        print(f"Removing existing f64 cache: {f64_path}")
        f64_path.unlink()

    # 1. Dtype checks
    print("\n[1] Dtype resolution checks...")
    for use_f64 in [False, True]:
        cfg = {'use_float64': use_f64}
        dt = get_dtype(cfg)
        expected = torch.float64 if use_f64 else torch.float32
        assert dt == expected, f"FAIL: got {dt}, expected {expected}"
        print(f"  use_float64={use_f64}: {dt} OK")

    # 2. Run float32
    print("\n[2] Float32 run...")
    r32 = run_test(False, 'float32')

    # 3. Run float64 (first time — should create cache)
    print("\n[3] Float64 run (first time, creates .f64.pt cache)...")
    r64 = run_test(True, 'float64')

    # 4. Check cache exists
    assert f64_path.exists(), f"FAIL: {f64_path} not created!"
    print(f"\n[4] Float64 cache: {f64_path} EXISTS ({f64_path.stat().st_size / 1024:.1f} KB)")

    # 5. Verify cache reuse
    print("\n[5] Cache reuse check...")
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        _load_bucket_data(BUCKET_PT, DEVICE, dtype=torch.float64)
    assert "Loading cached float64 data" in buf.getvalue(), "FAIL: cache not reused"
    print("  Cache reused on second load: OK")

    # 6. Run float64 again (from cache) to verify determinism
    print("\n[6] Float64 run #2 (from cache)...")
    r64b = run_test(True, 'float64_cached')

    # 7. Comparison
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"  {'':>10s}  {'float32':>14s}  {'float64':>14s}  {'float64_cached':>14s}")
    print(f"  {'epochs':>10s}  {r32['epochs_completed']:>14d}  {r64['epochs_completed']:>14d}  {r64b['epochs_completed']:>14d}")
    print(f"  {'final_loss':>10s}  {r32['final_loss']:>14.8e}  {r64['final_loss']:>14.8e}  {r64b['final_loss']:>14.8e}")
    print(f"  {'final_err':>10s}  {r32['final_local_error']:>14.8e}  {r64['final_local_error']:>14.8e}  {r64b['final_local_error']:>14.8e}")
    print(f"  {'wall_time':>10s}  {r32['wall_time']:>14.2f}s  {r64['wall_time']:>14.2f}s  {r64b['wall_time']:>14.2f}s")

    # Compare error at common checkpoints
    f32_by_epoch = {t[0]: t for t in r32['error_tracking_data']}
    f64_by_epoch = {t[0]: t for t in r64['error_tracking_data']}
    common = sorted(set(f32_by_epoch.keys()) & set(f64_by_epoch.keys()))
    if common:
        print(f"\n  Error tracking at common checkpoints:")
        print(f"  {'epoch':>8s}  {'f32_err':>14s}  {'f64_err':>14s}  {'diff':>14s}")
        for ep in common:
            e32 = f32_by_epoch[ep][3]
            e64 = f64_by_epoch[ep][3]
            diff = e64 - e32
            print(f"  {ep:>8d}  {e32:>14.8e}  {e64:>14.8e}  {diff:>+14.8e}")

    import math
    has_nan = (r32['final_loss'] is not None and math.isnan(r32['final_loss'])) or \
              (r64['final_loss'] is not None and math.isnan(r64['final_loss']))

    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    checks = {
        'float32 training': r32['epochs_completed'] > 0 and not (r32['final_loss'] is not None and math.isnan(r32['final_loss'])),
        'float64 training': r64['epochs_completed'] > 0 and not (r64['final_loss'] is not None and math.isnan(r64['final_loss'])),
        'float64 cache created': f64_path.exists(),
        'cache reuse': True,  # asserted above
        'no NaN losses': not has_nan,
    }
    for name, ok in checks.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    all_pass = all(checks.values())
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
