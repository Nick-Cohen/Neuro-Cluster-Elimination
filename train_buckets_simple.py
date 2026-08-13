#!/usr/bin/env python3
"""
Direct single-bucket training script (no multi-GPU orchestration, no subprocesses).
Runs training sequentially on the hard buckets.

Usage:
    python train_buckets_simple.py [--fast|--medium|--slow] [--gpu 0] [--config path/to/config.yaml]
"""
import sys
import os
os.chdir('/home/cohenn1/NCE')
sys.path.insert(0, '/home/cohenn1/NCE')

import argparse
import json
import time
import torch
from pathlib import Path

from nce.benchmark.training import train_single_bucket

def main():
    parser = argparse.ArgumentParser(description='Train hard buckets sequentially')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--gpu', type=int, default=None, help='GPU number (sets CUDA_VISIBLE_DEVICES)')
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML (default: benchmark_config.yaml)')
    parser.add_argument('--bucket', type=str, default=None,
                        help="Filter to specific bucket(s). Formats: "
                             "'29_wcsp_uai__bucket_54' (exact bucket_id), "
                             "'29_wcsp,54' (problem substring + bucket label), "
                             "'BN_8' (all buckets matching problem substring)")
    parser.add_argument('--add_noise', action='store_true',
                        help='Add Gaussian noise to exact forward message (variance = fw_var / 10)')
    parser.add_argument('--multiply_messages', type=float, nargs=2, default=None,
                        metavar=('FW_MULT', 'BW_MULT'),
                        help='Multiply forward message by FW_MULT and backward message by BW_MULT '
                             '(e.g. --multiply_messages 2 4)')
    speed = parser.add_mutually_exclusive_group()
    speed.add_argument('--fast', action='store_true', help='Fast mode: 60s per bucket (default)')
    speed.add_argument('--medium', action='store_true', help='Medium mode: 1200s (20m) per bucket')
    speed.add_argument('--slow', action='store_true', help='Slow mode: 3600s per bucket')
    args = parser.parse_args()

    # Set GPU before any CUDA init
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Load config
    config_path = Path(args.config) if args.config else Path('/home/cohenn1/NCE/benchmark_config.yaml')
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override device
    config['device'] = args.device

    # Time limit
    if args.slow:
        time_limit = 3600
    elif args.medium:
        time_limit = 1200
    else:
        time_limit = 60
    
    # Load bucket list
    bucket_list_path = Path('/home/cohenn1/NCE/data/hard_buckets/bucket_list.json')
    with open(bucket_list_path) as f:
        bucket_list = json.load(f)

    # Filter buckets if --bucket specified
    if args.bucket:
        filt = args.bucket
        if ',' in filt:
            # Format: 'problem_substr,bucket_label'
            problem_part, label_part = filt.split(',', 1)
            label = int(label_part.strip())
            bucket_list = [b for b in bucket_list
                           if problem_part.strip() in b['problem_key'] and b['bucket_label'] == label]
        else:
            # Could be exact bucket_id or a problem substring
            exact = [b for b in bucket_list if b['bucket_id'] == filt]
            if exact:
                bucket_list = exact
            else:
                bucket_list = [b for b in bucket_list if filt in b['bucket_id']]
        if not bucket_list:
            print(f"No buckets matched filter '{args.bucket}'")
            return

    # Noise seed (one per run, shared across all buckets for reproducibility)
    import random
    noise_seed = random.randint(0, 2**31 - 1) if args.add_noise else None
    if noise_seed is not None:
        print(f"Noise enabled: seed={noise_seed}")

    # Output directory
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    experiment_name = config.get('experiment_name')
    folder_name = f'{timestamp}_{experiment_name}' if experiment_name else timestamp
    output_dir = Path(f'/home/cohenn1/NCE/data/hard_buckets/benchmark_results/{folder_name}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy config file to output directory
    import shutil
    shutil.copy2(config_path, output_dir / config_path.name)

    print(f"=" * 60)
    print(f"Single-Bucket Training")
    print(f"=" * 60)
    print(f"Config: {config_path}")
    print(f"Device: {config['device']}")
    print(f"Loss function: {config['loss_fn']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Time limit: {time_limit}s per bucket")
    print(f"Buckets: {len(bucket_list)}")
    print(f"Output: {output_dir}")
    print(f"=" * 60)
    
    results = []
    count = 0
    
    for idx, bucket_info in enumerate(bucket_list):
        # count += 1
        # if count <= 7:
        #     continue
        bucket_id = bucket_info['bucket_id']
        pt_path = Path('/home/cohenn1/NCE/data/hard_buckets') / bucket_info['file']
        
        print(f"\n[{idx+1}/{len(bucket_list)}] Training {bucket_id}...")
        print(f"  .pt file: {pt_path}")
        
        bucket_output_dir = output_dir / bucket_id
        bucket_output_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        try:
        # if True:
            result = train_single_bucket(
                bucket_pt_path=str(pt_path),
                nn_config=config,
                output_dir=str(bucket_output_dir),
                time_limit_seconds=time_limit,
                device='cuda',
                noise_seed=noise_seed,
                multiply_messages=args.multiply_messages,
            )
            
            wall_time = time.time() - start_time
            
            print(f"  ✓ Completed: {result['epochs_completed']} epochs in {wall_time:.1f}s")
            print(f"    Final loss: {result['final_loss']:.6f}")
            print(f"    Final local error: {result['final_local_error']:.6f}")
            
            results.append({
                'bucket_id': bucket_id,
                'success': True,
                'epochs_completed': result['epochs_completed'],
                'wall_time': wall_time,
                'final_loss': result['final_loss'],
                'final_local_error': result['final_local_error']
            })
            
        except Exception as e:
        # else:
            wall_time = time.time() - start_time
            print(f"  ✗ FAILED after {wall_time:.1f}s: {e}")
            results.append({
                'bucket_id': bucket_id,
                'success': False,
                'error': str(e),
                'wall_time': wall_time
            })
            return
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"Summary")
    print(f"=" * 60)
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    print(f"Successful: {successful}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    
    if successful > 0:
        print(f"\nResults are in: {output_dir}")
        print(f"Each bucket has:")
        print(f"  - loss.png (loss curve)")
        print(f"  - local_error.png (error curve)")
        print(f"  - metrics.json (all metrics)")
    
    # Save summary
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'config': config,
            'time_limit': time_limit,
            'noise_seed': noise_seed,
            'multiply_messages': args.multiply_messages,
            'num_buckets': len(bucket_list),
            'successful': successful,
            'failed': failed,
            'results': results
        }, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")

if __name__ == '__main__':
    main()
