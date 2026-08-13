#!/usr/bin/env python3
"""
Single script to run the bucket benchmark.
Usage: python run_benchmark.py [--gpus 0,1,2,3] [--fast]
"""
import sys
import os
import subprocess
import argparse

# Ensure we're in the NCE directory
os.chdir('/home/cohenn1/NCE')
sys.path.insert(0, '/home/cohenn1/NCE')

def main():
    parser = argparse.ArgumentParser(description='Run single-bucket benchmark')
    parser.add_argument('--gpus', type=str, default='0', help='Comma-separated GPU IDs (default: 0)')
    parser.add_argument('--fast', action='store_true', help='Use fast mode (60s per bucket, default)')
    parser.add_argument('--slow', action='store_true', help='Use slow mode (3600s per bucket)')
    args = parser.parse_args()
    
    # Determine mode
    mode = 'slow' if args.slow else 'fast'
    
    config_path = '/home/cohenn1/NCE/benchmark_config.yaml'
    
    cmd = [
        sys.executable,  # Use current Python interpreter
        'scripts/bucket_benchmark.py',
        config_path,
        mode,
        '--gpus', args.gpus
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Config: {config_path}")
    print(f"Mode: {mode} ({60 if mode == 'fast' else 3600}s per bucket)")
    print(f"GPUs: {args.gpus}")
    print(f"Output: data/hard_buckets/benchmark_results/")
    print("=" * 60)
    
    # Run the benchmark
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == '__main__':
    main()
