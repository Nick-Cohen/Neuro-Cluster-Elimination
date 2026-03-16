#!/usr/bin/env python
"""Verify cached hard bucket data: check .pt file schema, tensor shapes, finiteness, manifest consistency.

Usage:
    python scripts/verify_hard_buckets.py                          # default dir
    python scripts/verify_hard_buckets.py --dir data/hard_buckets  # explicit dir
"""
import argparse
import json
import os
import sys

import torch


REQUIRED_TOP_KEYS = {
    'factors', 'exact_fw', 'exact_bw', 'bucket_label', 'scope',
    'domain_sizes', 'elim_vars', 'problem_key', 'auto_ecl', 'selection_error',
}

MESSAGE_SUB_KEYS = {'tensor', 'labels'}


def verify_pt_file(filepath):
    """Verify a single .pt file. Returns (pass: bool, issues: list[str])."""
    issues = []
    filename = os.path.basename(filepath)

    try:
        data = torch.load(filepath, map_location='cpu', weights_only=False)
    except Exception as e:
        return False, [f"Failed to load: {e}"]

    # Check top-level keys
    missing_keys = REQUIRED_TOP_KEYS - set(data.keys())
    if missing_keys:
        issues.append(f"Missing keys: {missing_keys}")

    # Check exact_fw structure
    if 'exact_fw' in data:
        if not isinstance(data['exact_fw'], dict):
            issues.append(f"exact_fw is not a dict (got {type(data['exact_fw']).__name__})")
        else:
            missing_sub = MESSAGE_SUB_KEYS - set(data['exact_fw'].keys())
            if missing_sub:
                issues.append(f"exact_fw missing sub-keys: {missing_sub}")
            elif not isinstance(data['exact_fw']['tensor'], torch.Tensor):
                issues.append(f"exact_fw.tensor is not a Tensor")
            elif not torch.isfinite(data['exact_fw']['tensor']).all():
                issues.append(f"exact_fw.tensor contains NaN/Inf")

    # Check exact_bw structure
    if 'exact_bw' in data:
        if not isinstance(data['exact_bw'], dict):
            issues.append(f"exact_bw is not a dict (got {type(data['exact_bw']).__name__})")
        else:
            missing_sub = MESSAGE_SUB_KEYS - set(data['exact_bw'].keys())
            if missing_sub:
                issues.append(f"exact_bw missing sub-keys: {missing_sub}")
            elif not isinstance(data['exact_bw']['tensor'], torch.Tensor):
                issues.append(f"exact_bw.tensor is not a Tensor")
            elif not torch.isfinite(data['exact_bw']['tensor']).all():
                issues.append(f"exact_bw.tensor contains NaN/Inf")

    # Check factors list
    if 'factors' in data:
        if not isinstance(data['factors'], list):
            issues.append(f"factors is not a list")
        else:
            for j, fdata in enumerate(data['factors']):
                if not isinstance(fdata, dict):
                    issues.append(f"factors[{j}] is not a dict")
                elif 'tensor' not in fdata or 'labels' not in fdata:
                    issues.append(f"factors[{j}] missing tensor/labels")
                elif not isinstance(fdata['tensor'], torch.Tensor):
                    issues.append(f"factors[{j}].tensor is not a Tensor")
                elif not torch.isfinite(fdata['tensor']).all():
                    issues.append(f"factors[{j}].tensor contains NaN/Inf")

    # Check tensor shape consistency: product of domain_sizes should match fw tensor numel
    if 'exact_fw' in data and 'domain_sizes' in data and 'scope' in data:
        fw = data['exact_fw']
        if isinstance(fw, dict) and 'tensor' in fw and isinstance(fw['tensor'], torch.Tensor):
            expected_numel = 1
            for ds in data['domain_sizes']:
                expected_numel *= ds
            actual_numel = fw['tensor'].numel()
            if expected_numel != actual_numel:
                issues.append(
                    f"Shape mismatch: domain_sizes product={expected_numel}, "
                    f"fw tensor numel={actual_numel}"
                )

    passed = len(issues) == 0
    return passed, issues


def main():
    parser = argparse.ArgumentParser(description="Verify cached hard bucket .pt files")
    parser.add_argument('--dir', type=str, default='data/hard_buckets',
                        help='Directory containing .pt files and bucket_list.json')
    args = parser.parse_args()

    data_dir = args.dir

    if not os.path.isdir(data_dir):
        print(f"FAIL: Directory {data_dir} does not exist")
        sys.exit(1)

    # Find .pt files
    pt_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])

    # Check manifest
    manifest_path = os.path.join(data_dir, 'bucket_list.json')
    manifest_exists = os.path.exists(manifest_path)
    manifest_data = None
    manifest_issues = []

    if manifest_exists:
        try:
            with open(manifest_path) as f:
                manifest_data = json.load(f)
        except Exception as e:
            manifest_issues.append(f"Failed to parse bucket_list.json: {e}")

    if manifest_data is not None:
        # Check required manifest keys
        required_manifest_keys = {'threshold', 'selection_date', 'num_problems',
                                  'total_nn_buckets', 'buckets'}
        missing = required_manifest_keys - set(manifest_data.keys())
        if missing:
            manifest_issues.append(f"Manifest missing keys: {missing}")

        # Check manifest bucket entries
        if 'buckets' in manifest_data:
            manifest_files = set()
            for entry in manifest_data['buckets']:
                required_entry_keys = {'id', 'problem_key', 'bucket_label',
                                       'selection_error', 'auto_ecl', 'file'}
                missing_ek = required_entry_keys - set(entry.keys())
                if missing_ek:
                    manifest_issues.append(f"Bucket entry missing keys: {missing_ek}")
                if 'file' in entry:
                    manifest_files.add(entry['file'])

            # Cross-check: manifest files vs actual .pt files
            actual_files = set(pt_files)
            in_manifest_not_disk = manifest_files - actual_files
            on_disk_not_manifest = actual_files - manifest_files
            if in_manifest_not_disk:
                manifest_issues.append(f"In manifest but not on disk: {in_manifest_not_disk}")
            if on_disk_not_manifest:
                manifest_issues.append(f"On disk but not in manifest: {on_disk_not_manifest}")
    elif not manifest_exists:
        manifest_issues.append("bucket_list.json not found")

    # Verify each .pt file
    print(f"\nVerifying {len(pt_files)} .pt files in {data_dir}/\n")

    total_pass = 0
    total_fail = 0

    for pt_file in pt_files:
        filepath = os.path.join(data_dir, pt_file)
        passed, issues = verify_pt_file(filepath)
        if passed:
            total_pass += 1
            print(f"  PASS  {pt_file}")
        else:
            total_fail += 1
            print(f"  FAIL  {pt_file}")
            for issue in issues:
                print(f"        - {issue}")

    # Report manifest status
    if manifest_issues:
        print(f"\n  Manifest issues:")
        for issue in manifest_issues:
            print(f"    - {issue}")
        total_fail += 1
    elif manifest_data is not None:
        print(f"\n  PASS  bucket_list.json ({len(manifest_data.get('buckets', []))} entries)")
        total_pass += 1

    # Summary
    print(f"\n{'='*60}")
    if total_fail == 0 and total_pass > 0:
        print(f"PASS — {total_pass} checks passed, 0 failed")
        sys.exit(0)
    elif total_fail == 0 and total_pass == 0:
        print(f"WARN — No .pt files found in {data_dir}/")
        sys.exit(1)
    else:
        print(f"FAIL — {total_pass} passed, {total_fail} failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
