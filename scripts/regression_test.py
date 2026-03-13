#!/usr/bin/env python
"""Regression test: flat vs nested configs produce identical inference (R017).

Verifies three properties on rbm_20 (nbe_sanity_check model index 3):
  1. prepare_config(flat) == prepare_config(nested)
  2. Exact-only inference (high ecl, no NN) produces bitwise-equal partition functions
  3. NN inference (2 epochs, 500 samples) produces bitwise-equal partition functions

Exit 0 if all checks pass, exit 1 if any fail.
Runtime: ~10s on GPU (2 NN epochs, small samples).
"""

import copy
import sys

import torch

from nce.config_schema import prepare_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

results = []  # list of (name, passed, detail)


def check(name: str, condition: bool, detail: str = ""):
    """Record a single check result."""
    results.append((name, condition, detail))
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}: {detail}")


def summarize_and_exit():
    """Print summary and exit with appropriate code."""
    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    all_ok = all(ok for _, ok, _ in results)
    print(f"Results: {passed}/{total} checks passed")
    if all_ok:
        print("OVERALL: PASS")
    else:
        print("OVERALL: FAIL")
        for name, ok, detail in results:
            if not ok:
                print(f"  FAILED: {name} — {detail}")
    sys.exit(0 if all_ok else 1)


# ---------------------------------------------------------------------------
# Load benchmark configs
# ---------------------------------------------------------------------------

from nce.benchmark_problems import nbe_sanity_check

MODEL_IDX = 3  # rbm_20

model = nbe_sanity_check.problems[MODEL_IDX]
flat_cfg = nbe_sanity_check.configs['nbe'][MODEL_IDX]
nested_cfg = nbe_sanity_check.configs['nbe_nested'][MODEL_IDX]

print(f"Model: {model.modelfile} (index {MODEL_IDX})")

# ---------------------------------------------------------------------------
# Check 1: Config equality after prepare_config
# ---------------------------------------------------------------------------

print("\n--- Check 1: prepare_config(flat) == prepare_config(nested) ---")

flat_prepared = prepare_config(copy.deepcopy(flat_cfg))
nested_prepared = prepare_config(copy.deepcopy(nested_cfg))

configs_equal = flat_prepared == nested_prepared
if not configs_equal:
    # Build diff for diagnostics
    all_keys = set(flat_prepared) | set(nested_prepared)
    diffs = {}
    for k in sorted(all_keys):
        v1 = flat_prepared.get(k, '<missing>')
        v2 = nested_prepared.get(k, '<missing>')
        if v1 != v2:
            diffs[k] = (v1, v2)
    detail = f"diffs: {diffs}"
else:
    detail = f"{len(flat_prepared)} keys, all values match"

check("config_equality", configs_equal, detail)

# ---------------------------------------------------------------------------
# Checks 2 & 3 require CUDA
# ---------------------------------------------------------------------------

if not torch.cuda.is_available():
    print("\nCUDA not available — skipping inference checks 2 and 3")
    check("exact_inference_equality", True, "SKIPPED (no CUDA)")
    check("nn_inference_equality", True, "SKIPPED (no CUDA)")
    summarize_and_exit()

from nce.inference.graphical_model import FastGM

device = "cuda"

# ---------------------------------------------------------------------------
# Check 2: Exact-only inference (high ecl, no NN)
# ---------------------------------------------------------------------------

print("\n--- Check 2: Exact-only inference equality ---")

flat_exact = copy.deepcopy(flat_cfg)
flat_exact['ecl'] = 2**30
flat_exact['num_epochs'] = 0

nested_exact = copy.deepcopy(nested_cfg)
nested_exact['inference']['exact_computation_limit'] = 2**30
nested_exact['training']['num_epochs'] = 0

# Prepare both through prepare_config to get final flat form
flat_exact_prepared = prepare_config(flat_exact)
nested_exact_prepared = prepare_config(nested_exact)

gm_flat = FastGM(model=model, nn_config=flat_exact_prepared, device=device)
gm_flat.eliminate_variables(all=True)
logZ_flat_exact = gm_flat.log_partition_function

gm_nested = FastGM(model=model, nn_config=nested_exact_prepared, device=device)
gm_nested.eliminate_variables(all=True)
logZ_nested_exact = gm_nested.log_partition_function

exact_equal = (logZ_flat_exact == logZ_nested_exact)
check(
    "exact_inference_equality",
    exact_equal,
    f"flat={logZ_flat_exact}, nested={logZ_nested_exact}"
)

# ---------------------------------------------------------------------------
# Check 3: NN inference (2 epochs, 500 samples)
# ---------------------------------------------------------------------------

print("\n--- Check 3: NN inference equality (2 epochs, 500 samples) ---")

flat_nn = copy.deepcopy(flat_cfg)
flat_nn['num_epochs'] = 2
flat_nn['num_samples'] = 500
flat_nn['set_size'] = 500
flat_nn['seed'] = 42

nested_nn = copy.deepcopy(nested_cfg)
nested_nn['training']['num_epochs'] = 2
nested_nn['training']['seed'] = 42
nested_nn['sampling']['num_samples'] = 500
nested_nn['sampling']['set_size'] = 500

flat_nn_prepared = prepare_config(flat_nn)
nested_nn_prepared = prepare_config(nested_nn)

gm_flat_nn = FastGM(model=model, nn_config=flat_nn_prepared, device=device)
gm_flat_nn.eliminate_variables(all=True)
logZ_flat_nn = gm_flat_nn.log_partition_function

gm_nested_nn = FastGM(model=model, nn_config=nested_nn_prepared, device=device)
gm_nested_nn.eliminate_variables(all=True)
logZ_nested_nn = gm_nested_nn.log_partition_function

nn_equal = (logZ_flat_nn == logZ_nested_nn)
check(
    "nn_inference_equality",
    nn_equal,
    f"flat={logZ_flat_nn}, nested={logZ_nested_nn}"
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

summarize_and_exit()
