#!/usr/bin/env python
"""Verify that NCE NN bucket counts match NeuroBE for all 15 binary-domain problems.

Loads each model via FastGM with neurobe_mode config, computes which buckets
would receive NN treatment using get_large_message_buckets(iB=25, ecl=...),
and compares the count against NeuroBE's ground truth.

No GPU required — only builds bucket structure, does not train.

Usage:
    python scripts/verify_nn_counts.py

Exit codes:
    0 — all 15 problems match
    1 — one or more mismatches
"""
import sys

from nce.benchmark_problems.neurobe_binary import (
    _MODEL_KEYS,
    _NEUROBE_ECL,
    NEUROBE_NN_COUNTS,
    neurobe_binary,
)
from nce.inference.graphical_model import FastGM


def main():
    problems = neurobe_binary.problems
    configs = neurobe_binary.configs['neurobe']

    mismatches = 0
    print(f"{'Problem':<30} {'Expected':>8} {'Actual':>8} {'Status'}")
    print("-" * 60)

    for key, model, config in zip(_MODEL_KEYS, problems, configs):
        expected = NEUROBE_NN_COUNTS[key]
        ecl = _NEUROBE_ECL[key]

        # Load model on CPU — only need bucket structure, not training
        fastgm = FastGM(model=model, nn_config=config, device='cpu')
        nn_buckets = fastgm.get_large_message_buckets(iB=25, ecl=ecl)
        actual = len(nn_buckets)

        status = "MATCH" if actual == expected else "MISMATCH"
        if actual != expected:
            mismatches += 1
        print(f"{key:<30} {expected:>8} {actual:>8} {status}")

    print("-" * 60)
    if mismatches == 0:
        print(f"All {len(_MODEL_KEYS)} problems MATCH.")
    else:
        print(f"{mismatches} of {len(_MODEL_KEYS)} problems MISMATCH.")

    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
