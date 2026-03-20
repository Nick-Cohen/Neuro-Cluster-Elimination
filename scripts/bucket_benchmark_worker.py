#!/usr/bin/env python
"""Worker subprocess for bucket_benchmark.py.

Wraps train_single_bucket() with JSON-based argument parsing and result
serialization. Designed to be spawned with CUDA_VISIBLE_DEVICES isolation.

Exit 0 on success, exit 1 on failure. Writes result dict to --result-path.

Usage:
    python scripts/bucket_benchmark_worker.py \\
        --bucket-id "or_chain_10__bucket_5" \\
        --pt-path "data/hard_buckets/or_chain_10_bucket_5.pt" \\
        --config-json '{"loss_fn": "unnormalized_kl", ...}' \\
        --output-dir "data/benchmark_output" \\
        --time-limit 60 \\
        --result-path "/tmp/result.json"
"""
import argparse
import json
import sys
import traceback


def _json_default(obj):
    """JSON serializer for types not natively serializable.
    
    Handles torch tensors and sets. Matches pattern from S02's training.py.
    """
    # Import torch only when needed
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
    except ImportError:
        pass
    
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    
    return str(obj)


def main():
    parser = argparse.ArgumentParser(
        description="Bucket benchmark training worker subprocess"
    )
    parser.add_argument('--bucket-id', required=True,
                        help='Sanitized bucket identifier (for logging)')
    parser.add_argument('--pt-path', required=True,
                        help='Path to precomputed .pt file from S01')
    parser.add_argument('--config-json', required=True,
                        help='Training config as JSON string')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for plots and metrics')
    parser.add_argument('--time-limit', type=int, required=True,
                        help='Training time limit in seconds')
    parser.add_argument('--result-path', required=True,
                        help='Path to write result JSON')
    args = parser.parse_args()

    # Import training function AFTER arg parsing (so --help works without deps)
    from nce.benchmark.training import train_single_bucket

    # -----------------------------------------------------------------------
    # Parse config JSON
    # -----------------------------------------------------------------------
    try:
        config = json.loads(args.config_json)
    except json.JSONDecodeError as e:
        print(f"[BenchmarkWorker] FAIL: Invalid config JSON: {e}",
              file=sys.stderr)
        sys.exit(1)

    device = config.get('device', 'cuda')

    # -----------------------------------------------------------------------
    # Log start
    # -----------------------------------------------------------------------
    print(f"[BenchmarkWorker] Starting: bucket_id={args.bucket_id}")
    print(f"[BenchmarkWorker]   pt_path={args.pt_path}")
    print(f"[BenchmarkWorker]   time_limit={args.time_limit}s")
    print(f"[BenchmarkWorker]   device={device}")
    print(f"[BenchmarkWorker]   config keys: {sorted(config.keys())}")
    sys.stdout.flush()

    # -----------------------------------------------------------------------
    # Call train_single_bucket
    # -----------------------------------------------------------------------
    try:
        result = train_single_bucket(
            bucket_pt_path=args.pt_path,
            nn_config=config,
            time_limit_seconds=args.time_limit,
            output_dir=args.output_dir,
            device=device,
        )
    except Exception as e:
        print(f"[BenchmarkWorker] FAIL: train_single_bucket() raised exception:",
              file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Write result to temp file as JSONL
    # -----------------------------------------------------------------------
    try:
        # Extract required fields for history tracking
        # Convert error_tracking_data tuples to dict with epochs as keys
        error_tracking_dict = {}
        for epoch, loss, log_z_err, abs_log_z_err in result.get('error_tracking_data', []):
            error_tracking_dict[int(epoch)] = {
                'loss': _json_default(loss),
                'log_z_err': _json_default(log_z_err),
                'abs_log_z_err': _json_default(abs_log_z_err),
            }
        
        jsonl_record = {
            'bucket_id': args.bucket_id,
            'pt_path': args.pt_path,
            'epochs_completed': result.get('epochs_completed', 0),
            'wall_time': result.get('wall_time', 0),
            'final_loss': _json_default(result.get('final_loss')),
            'error_tracking_data': error_tracking_dict,
            'final_local_error': _json_default(result.get('final_local_error')),
        }
        
        # Write as single JSONL line (no newlines within the JSON)
        with open(args.result_path, 'w') as f:
            json.dump(jsonl_record, f, default=_json_default)
            f.write('\n')  # JSONL requires trailing newline
    except Exception as e:
        print(f"[BenchmarkWorker] FAIL: Could not write result JSON: {e}",
              file=sys.stderr)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Log completion
    # -----------------------------------------------------------------------
    print(f"[BenchmarkWorker] Complete: bucket_id={args.bucket_id}, "
          f"epochs={result.get('epochs_completed', 0)}, "
          f"wall_time={result.get('wall_time', 0):.1f}s, "
          f"final_local_error={result.get('final_local_error')}")
    sys.stdout.flush()
    sys.exit(0)


if __name__ == '__main__':
    main()
