#!/usr/bin/env python
"""Precompute approximate backward messages for hard buckets at specified bw_iB2 levels.

For each bucket in bucket_list.json, reconstructs the live FastGM, computes the
backward message via WMB at the given bw_iB2, and saves it as a .pt file.

Usage:
    python scripts/precompute_bw_messages.py --bw-ib2 10
    python scripts/precompute_bw_messages.py --bw-ib2 5 10 15
"""
import argparse
import copy
import json
import os
import sys
import time

os.chdir('/home/cohenn1/NCE')
sys.path.insert(0, '/home/cohenn1/NCE')

import torch
from pathlib import Path

from nce.benchmark_problems.small_problems import small_problems
from nce.config_schema import prepare_config
from nce.inference.graphical_model import FastGM
from nce.utils.backward_message import get_backward_message


HARD_BUCKETS_DIR = Path('/home/cohenn1/NCE/data/hard_buckets')
BW_CACHE_DIR = HARD_BUCKETS_DIR / 'bw_cache'


def find_problem_idx(problem_key):
    for idx, model in enumerate(small_problems.problems):
        if model.modelfile == problem_key:
            return idx
    raise ValueError(f"Problem key '{problem_key}' not found")


def reconstruct_fastgm_and_bucket(problem_idx, bucket_label, bw_iB, bw_ecl, device):
    """Reconstruct the FastGM up to the target bucket, then compute bw message."""
    model = small_problems.problems[problem_idx]
    # Minimal config — just need exact elimination to reconstruct upstream state
    config = prepare_config({
        'iB': 100,
        'ecl': 2**30,
        'device': device,
        'approximation_method': 'nn',
        'loss_fn': 'unnormalized_kl',
        'hidden_sizes': [],
        'num_epochs': 0,
        'num_samples': 1,
    }, strict=False)

    fastgm = FastGM(model=model, nn_config=config, device=device)
    target_var = fastgm.matching_var(bucket_label)
    if target_var is None:
        raise ValueError(f"No matching var for bucket_label={bucket_label}")

    # Eliminate exactly up to (not including) the target bucket
    fastgm.eliminate_variables(up_to=target_var, exact=True)

    # Compute backward message at the specified approximation level
    bw_msg, _ = get_backward_message(
        fastgm,
        bucket_label,
        backward_factors=None,
        iB=bw_iB,
        backward_ecl=bw_ecl,
        approximation_method='wmb',
        return_factor_list=False,
    )

    return bw_msg


def main():
    parser = argparse.ArgumentParser(description='Precompute backward messages for hard buckets')
    parser.add_argument('--bw-ib2', type=int, nargs='+', required=True,
                        help='bw_iB2 values to precompute (e.g. 5 10 15)')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    bucket_list_path = HARD_BUCKETS_DIR / 'bucket_list.json'
    with open(bucket_list_path) as f:
        bucket_list = json.load(f)

    BW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for bw_ib2 in args.bw_ib2:
        bw_iB = bw_ib2
        bw_ecl = (2 ** bw_ib2) - 1

        level_dir = BW_CACHE_DIR / f'bw_ib2_{bw_ib2}'
        level_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Precomputing bw messages: bw_iB2={bw_ib2} (bw_iB={bw_iB}, bw_ecl={bw_ecl})")
        print(f"Output: {level_dir}")
        print(f"{'='*60}")

        for idx, bucket_info in enumerate(bucket_list):
            bucket_id = bucket_info['bucket_id']
            bucket_label = bucket_info['bucket_label']
            pt_path = HARD_BUCKETS_DIR / bucket_info['file']

            out_path = level_dir / f'{bucket_id}.pt'
            if out_path.exists():
                print(f"  [{idx+1}/{len(bucket_list)}] {bucket_id} — already cached, skipping")
                continue

            # Read the correct problem_key from the .pt file (bucket_list.json has sanitized names)
            pt_data = torch.load(pt_path, map_location='cpu', weights_only=False)
            problem_key = pt_data['problem_key']
            del pt_data

            print(f"  [{idx+1}/{len(bucket_list)}] {bucket_id}...", end=' ', flush=True)
            t0 = time.time()

            problem_idx = find_problem_idx(problem_key)
            bw_msg = reconstruct_fastgm_and_bucket(
                problem_idx, bucket_label, bw_iB, bw_ecl, args.device
            )

            # Save as .pt with tensor + labels (same format as exact_bw in the bucket .pt files)
            torch.save({
                'tensor': bw_msg.tensor.detach().cpu(),
                'labels': list(bw_msg.labels),
                'bw_ib2': bw_ib2,
                'bw_iB': bw_iB,
                'bw_ecl': bw_ecl,
                'bucket_id': bucket_id,
                'problem_key': problem_key,
                'bucket_label': bucket_label,
            }, out_path)

            elapsed = time.time() - t0
            print(f"done ({elapsed:.1f}s), shape={list(bw_msg.tensor.shape)}")

        print(f"\nAll buckets cached for bw_iB2={bw_ib2}")

    # Write index of available levels
    available = sorted(int(d.name.split('_')[-1]) for d in BW_CACHE_DIR.iterdir() if d.is_dir())
    index_path = BW_CACHE_DIR / 'index.json'
    with open(index_path, 'w') as f:
        json.dump({'available_bw_ib2': available}, f, indent=2)
    print(f"\nIndex written: {index_path}")
    print(f"Available bw_iB2 levels: {available}")


if __name__ == '__main__':
    main()