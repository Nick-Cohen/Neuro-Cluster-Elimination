#!/usr/bin/env python
"""Run NCE neurobe_mode inference on all 15 binary-domain problems.

Iterates through the neurobe_binary benchmark set, runs full variable
elimination with NN approximation on CUDA, and writes results to CSV.

Per-problem exceptions are caught and logged — one failure does not
kill the run.

Usage:
    python scripts/run_neurobe_experiments.py

Output:
    notebooks/March-2025/neurobe_comparison_results.csv
"""
import csv
import os
import sys
import time
import traceback

import torch

from nce.benchmark_problems.neurobe_binary import (
    _MODEL_KEYS,
    NEUROBE_NN_COUNTS,
    neurobe_binary,
)
from nce.inference.graphical_model import FastGM


OUTPUT_DIR = os.path.join("notebooks", "March-2025")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "neurobe_comparison_results.csv")

CSV_COLUMNS = [
    "Problem",
    "NCE_log_Z",
    "NCE_NNs",
    "NCE_time_hrs",
    "Status",
    "Error",
]


def run_single_problem(key, model, config):
    """Run inference on a single problem. Returns a result dict."""
    t0 = time.time()

    fastgm = FastGM(model=model, nn_config=config, device=config["device"])
    fastgm.eliminate_variables(all=True)

    elapsed = time.time() - t0
    log_z = fastgm.log_partition_function
    num_trained = fastgm.num_trained

    return {
        "Problem": key,
        "NCE_log_Z": f"{log_z:.6f}",
        "NCE_NNs": num_trained,
        "NCE_time_hrs": f"{elapsed / 3600:.4f}",
        "Status": "success",
        "Error": "",
    }


def main():
    problems = neurobe_binary.problems
    configs = neurobe_binary.configs["neurobe"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    total_start = time.time()

    print(f"{'='*70}")
    print(f"NCE neurobe_mode experiments — 15 binary-domain problems")
    print(f"Device: cuda | Output: {OUTPUT_CSV}")
    print(f"{'='*70}")
    print(flush=True)

    for i, (key, model, config) in enumerate(zip(_MODEL_KEYS, problems, configs)):
        expected_nns = NEUROBE_NN_COUNTS[key]
        print(f"\n[{i+1}/15] {key}  (expected NNs: {expected_nns})")
        print("-" * 50, flush=True)

        try:
            result = run_single_problem(key, model, config)
            print(
                f"  ✓ log_Z={result['NCE_log_Z']}  "
                f"NNs={result['NCE_NNs']}  "
                f"time={result['NCE_time_hrs']} hrs",
                flush=True,
            )
            if int(result["NCE_NNs"]) != expected_nns:
                print(
                    f"  ⚠ NN count mismatch: got {result['NCE_NNs']}, "
                    f"expected {expected_nns}",
                    flush=True,
                )
        except Exception as exc:
            elapsed = time.time() - total_start
            tb = traceback.format_exc()
            print(f"  ✗ FAILED: {exc}", flush=True)
            print(tb, flush=True)
            result = {
                "Problem": key,
                "NCE_log_Z": "",
                "NCE_NNs": "",
                "NCE_time_hrs": "",
                "Status": "failed",
                "Error": str(exc).replace("\n", " ")[:200],
            }
        finally:
            # Free GPU memory between problems
            torch.cuda.empty_cache()

        results.append(result)

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(results)

    total_elapsed = time.time() - total_start
    succeeded = sum(1 for r in results if r["Status"] == "success")

    print(f"\n{'='*70}")
    print(f"DONE: {succeeded}/15 succeeded  |  Total time: {total_elapsed/3600:.2f} hrs")
    print(f"Results written to: {OUTPUT_CSV}")
    print(f"{'='*70}", flush=True)

    return 0 if succeeded == 15 else 1


if __name__ == "__main__":
    sys.exit(main())
