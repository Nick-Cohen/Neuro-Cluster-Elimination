#!/usr/bin/env python
"""Build combined NCE vs NeuroBE comparison table for binary-domain problems.

Reads NCE experiment results and NeuroBE C++ results, joins on problem name,
verifies NN counts match, and produces a formatted comparison table.

Usage:
    python scripts/build_comparison_table.py

Inputs:
    notebooks/March-2025/neurobe_comparison_results.csv  (NCE results from T02)
    Clean-NeuroBE/results/binary_min_nn/binary_domain_results.csv  (NeuroBE ground truth)

Output:
    notebooks/March-2025/neurobe_comparison_table.csv  (combined comparison table)
"""
import csv
import os
import sys


NCE_CSV = os.path.join("notebooks", "March-2025", "neurobe_comparison_results.csv")
NEUROBE_CSV = os.path.join(
    "Clean-NeuroBE", "results", "binary_min_nn", "binary_domain_results.csv"
)
OUTPUT_CSV = os.path.join("notebooks", "March-2025", "neurobe_comparison_table.csv")

OUTPUT_COLUMNS = [
    "Problem",
    "NCE_log_Z",
    "NeuroBE_log_Z",
    "NCE_NNs",
    "NeuroBE_NNs",
    "NCE_time_hrs",
    "NeuroBE_time_hrs",
]


def nce_key_to_neurobe_key(nce_key: str) -> str:
    """Convert NCE problem key (e.g. 'bn/BN_1') to NeuroBE key ('BN_1')."""
    return nce_key.split("/")[-1]


def load_nce_results(path: str) -> dict:
    """Load NCE results CSV into dict keyed by NeuroBE-style problem name."""
    results = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            neurobe_key = nce_key_to_neurobe_key(row["Problem"])
            results[neurobe_key] = row
    return results


def load_neurobe_results(path: str) -> dict:
    """Load NeuroBE results CSV into dict keyed by problem name."""
    results = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results[row["Problem"]] = row
    return results


def main():
    # --- Load data ---
    if not os.path.exists(NCE_CSV):
        print(f"ERROR: NCE results not found: {NCE_CSV}")
        print("Run scripts/run_neurobe_experiments.py first.")
        return 1

    if not os.path.exists(NEUROBE_CSV):
        print(f"ERROR: NeuroBE results not found: {NEUROBE_CSV}")
        return 1

    nce = load_nce_results(NCE_CSV)
    neurobe = load_neurobe_results(NEUROBE_CSV)

    # --- Check for failed NCE problems ---
    failed = [k for k, v in nce.items() if v.get("Status") == "failed"]
    if failed:
        print(f"WARNING: {len(failed)} NCE problems failed: {', '.join(failed)}")
        print("Re-run scripts/run_neurobe_experiments.py to fix.\n")

    # --- Join on problem name (NCE problems only — skip or_chain which NCE doesn't have) ---
    combined = []
    all_match = True
    missing = []

    for nce_key, nce_row in nce.items():
        if nce_key not in neurobe:
            missing.append(nce_key)
            continue

        nb_row = neurobe[nce_key]

        # Parse NN counts
        nce_nns_str = nce_row.get("NCE_NNs", "")
        neurobe_nns_raw = nb_row.get("NNs", "")
        # NeuroBE uses "0*" for or_chain — strip asterisk
        neurobe_nns_str = neurobe_nns_raw.rstrip("*")

        try:
            nce_nns = int(nce_nns_str) if nce_nns_str else None
        except ValueError:
            nce_nns = None
        try:
            neurobe_nns = int(neurobe_nns_str) if neurobe_nns_str else None
        except ValueError:
            neurobe_nns = None

        # NN count match check
        if nce_nns is not None and neurobe_nns is not None:
            match = nce_nns == neurobe_nns
        else:
            match = False  # Can't verify if data is missing

        if not match:
            all_match = False

        # Parse log_Z values for divergence check
        nce_log_z_str = nce_row.get("NCE_log_Z", "")
        neurobe_log_z_str = nb_row.get("Log_Z", "")
        # NeuroBE uses "—" for missing values
        if neurobe_log_z_str in ("—", ""):
            neurobe_log_z_str = ""

        combined.append(
            {
                "Problem": nce_key,
                "NCE_log_Z": nce_log_z_str,
                "NeuroBE_log_Z": neurobe_log_z_str,
                "NCE_NNs": nce_nns_str,
                "NeuroBE_NNs": neurobe_nns_raw,
                "NCE_time_hrs": nce_row.get("NCE_time_hrs", ""),
                "NeuroBE_time_hrs": nb_row.get("Runtime_hrs", ""),
                "_match": match,
                "_nce_log_z": float(nce_log_z_str) if nce_log_z_str else None,
                "_neurobe_log_z": float(neurobe_log_z_str) if neurobe_log_z_str else None,
            }
        )

    if missing:
        print(f"WARNING: {len(missing)} NCE problems not found in NeuroBE: {missing}")

    # --- Print formatted table ---
    print()
    print("=" * 100)
    print("NCE vs NeuroBE Comparison — Binary Domain Problems")
    print("=" * 100)

    header = (
        f"{'Problem':<22} {'NCE_log_Z':>12} {'NeuroBE_log_Z':>13} "
        f"{'NCE_NNs':>8} {'NeuroBE_NNs':>11} "
        f"{'NCE_hrs':>9} {'NeuroBE_hrs':>11}  {'NNs'}"
    )
    print(header)
    print("-" * 100)

    nn_matches = 0
    nn_mismatches = 0
    divergent = []

    for row in combined:
        match_str = "MATCH" if row["_match"] else "MISMATCH"
        if row["_match"]:
            nn_matches += 1
        else:
            nn_mismatches += 1

        line = (
            f"{row['Problem']:<22} {row['NCE_log_Z']:>12} {row['NeuroBE_log_Z']:>13} "
            f"{row['NCE_NNs']:>8} {row['NeuroBE_NNs']:>11} "
            f"{row['NCE_time_hrs']:>9} {row['NeuroBE_time_hrs']:>11}  {match_str}"
        )
        print(line)

        # Check for significant log_Z divergence (> 10% relative difference)
        nce_z = row["_nce_log_z"]
        nb_z = row["_neurobe_log_z"]
        if nce_z is not None and nb_z is not None and nb_z != 0:
            rel_diff = abs(nce_z - nb_z) / abs(nb_z)
            if rel_diff > 0.10:
                divergent.append(
                    (row["Problem"], nce_z, nb_z, rel_diff)
                )

    print("-" * 100)
    print(f"NN count verification: {nn_matches} MATCH, {nn_mismatches} MISMATCH out of {len(combined)} problems")

    if all_match and nn_matches == 15:
        print("✓ R036 VERIFIED: All 15 problems have matching NN counts")
    elif nn_mismatches > 0:
        print(f"✗ R036 FAILED: {nn_mismatches} NN count mismatches")
    else:
        print(f"⚠ R036 PARTIAL: Only {nn_matches}/15 problems verified (some data missing)")

    if divergent:
        print(f"\n⚠ Significant log_Z divergence (>10% relative) in {len(divergent)} problems:")
        for name, nce_z, nb_z, rd in divergent:
            print(f"  {name}: NCE={nce_z:.4f} vs NeuroBE={nb_z:.4f} (rel_diff={rd:.1%})")

    if failed:
        print(f"\n⚠ {len(failed)} problems have Status=failed — results incomplete")

    print("=" * 100)

    # --- Write combined CSV ---
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in combined:
            writer.writerow({k: row[k] for k in OUTPUT_COLUMNS})

    print(f"\nCombined table saved to: {OUTPUT_CSV}")
    row_count = len(combined)
    print(f"Rows: {row_count} (expected 15)")

    # Exit code: 0 if all 15 match, 1 otherwise
    if all_match and nn_matches == 15 and not failed:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
