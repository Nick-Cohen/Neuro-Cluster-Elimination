#!/usr/bin/env python
"""End-to-end verification of S04: FastGM State Preservation.

Exercises the full pipeline:
  1. Load a small benchmark problem (rbm_20), run NN inference
  2. Verify per_bucket_training_log has loss curves
  3. Save/load metadata-only state and assert round-trip correctness
  4. Save/load weights-included state and assert round-trip correctness
  5. Test undo_normalization with a saved normalizing constant

Exit 0 if all checks pass, exit 1 if any fail.
Runtime: ~50 seconds (two inference passes, 3 epochs each, small samples).
"""

import math
import os
import sys
from collections import OrderedDict

import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

results = []  # list of (group_name, passed: bool, detail: str)


def check(group: str, condition: bool, detail: str = ""):
    """Record a single assertion result."""
    results.append((group, condition, detail))
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {group}: {detail}")


def summarize_and_exit():
    """Print summary, clean up temp files, exit."""
    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    all_ok = all(ok for _, ok, _ in results)
    print(f"Results: {passed}/{total} checks passed")
    if all_ok:
        print("OVERALL: PASS")
    else:
        print("OVERALL: FAIL")
        for group, ok, detail in results:
            if not ok:
                print(f"  FAILED: {group} — {detail}")
    # Clean up temp files
    for p in ["/tmp/test_s04_meta.pkl", "/tmp/test_s04_weights.pkl"]:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
    sys.exit(0 if all_ok else 1)


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Minimal config that forces NN training on a handful of buckets.
# rbm_20 has 40 binary vars; with iB=20 and ecl=2^19, 20 buckets use NN.
# 3 epochs, 2048 samples → each inference pass finishes in ~25s.
_BASE_CONFIG = {
    "device": device,
    "hidden_sizes": [3, 3],
    "optimizer": "adam",
    "lr": 0.01,
    "lr_decay": 1,
    "momentum": 0.9,
    "inverse_time_decay_constant": 10,
    "patience": 1,
    "min_lr": 1e-8,
    "num_epochs": 3,
    "num_epochs2": 0,
    "nbe_early_stopping": False,
    "skip_early_stopping": True,
    "sampling_scheme": "uniform",
    "batch_size": 512,
    "set_size": 2048,
    "num_samples": 2048,
    "loss_fn": "unnormalized_kl",
    "traced_losses": [],
    "val_set": True,
    "fdb": False,
    "use_bw_approx": False,
    "populate_bw_factors": False,
    "ecl": 2**19,  # 524288 — 20 of 40 buckets exceed this and use NN
    "iB": 20,
    "approximation_method": "nn",
    "bw_ecl": 0,
    "backward_iB": 20,
    "use_linspace_bias": False,
    "use_memorizer": False,
    "display_intermediate": False,
    "track_errors": False,
    "plot_messages": False,
    "debug": False,
    "lower_dim": False,
    "dope_factors": True,
    "gather_message_stats": False,
    "stratify_samples": False,
    "seed": 42,
}

# ---------------------------------------------------------------------------
# Step 1: Load benchmark problem and run NN inference (metadata-only)
# ---------------------------------------------------------------------------

print("\n--- Step 1: Load problem & run NN inference (metadata-only) ---")

from nce.benchmark_problems import nbe_sanity_check
from nce.inference.graphical_model import FastGM
from nce.state import save_state, load_state, undo_normalization

# rbm_20 is index 3 in nbe_sanity_check: 40 binary vars, reliable .vo file
model = nbe_sanity_check.problems[3]

config_meta = _BASE_CONFIG.copy()
config_meta["save_nn_weights"] = False

print(f"Problem: {model.modelfile}")
print(f"Config: ecl={config_meta['ecl']}, num_epochs=3, hidden_sizes=[3,3], device={device}")

fastgm = FastGM(model=model, nn_config=config_meta, device=device)
fastgm.eliminate_variables(all=True)

log = fastgm.per_bucket_training_log

check("training_log_nonempty",
      len(log) > 0,
      f"per_bucket_training_log has {len(log)} entries")

nn_buckets_with_losses = [e for e in log if len(e.get("losses", [])) > 0]
check("losses_present",
      len(nn_buckets_with_losses) > 0,
      f"{len(nn_buckets_with_losses)}/{len(log)} entries have non-empty losses")

# Spot-check first entry with losses
if nn_buckets_with_losses:
    entry0 = nn_buckets_with_losses[0]
    check("loss_tuple_format",
          isinstance(entry0["losses"][0], (tuple, list)) and len(entry0["losses"][0]) == 2,
          f"losses[0] = {entry0['losses'][0]}")
    check("epochs_trained_positive",
          entry0.get("epochs_trained", 0) > 0,
          f"epochs_trained = {entry0.get('epochs_trained')}")
    check("hidden_sizes_is_list",
          isinstance(entry0.get("hidden_sizes"), list),
          f"hidden_sizes = {entry0.get('hidden_sizes')}")
    check("val_losses_is_list",
          isinstance(entry0.get("val_losses"), list),
          f"val_losses type ok, len={len(entry0.get('val_losses', []))}")
else:
    check("loss_tuple_format", False, "No entries with losses to inspect")
    check("epochs_trained_positive", False, "Skipped — no entries")
    check("hidden_sizes_is_list", False, "Skipped — no entries")
    check("val_losses_is_list", False, "Skipped — no entries")

# ---------------------------------------------------------------------------
# Step 2: Metadata-only save/load round-trip
# ---------------------------------------------------------------------------

print("\n--- Step 2: Metadata-only save/load round-trip ---")

meta_path = "/tmp/test_s04_meta.pkl"
save_state(fastgm, meta_path, save_weights=False)
state_meta = load_state(meta_path)

check("meta_has_training_log",
      "per_bucket_training_log" in state_meta,
      "per_bucket_training_log key present")

check("meta_log_nonempty",
      len(state_meta.get("per_bucket_training_log", [])) > 0,
      f"loaded log has {len(state_meta.get('per_bucket_training_log', []))} entries")

check("meta_has_config",
      "config" in state_meta,
      "config key present")

check("meta_has_logZ",
      "logZ" in state_meta,
      f"logZ = {state_meta.get('logZ')}")

check("meta_has_elim_order",
      "elim_order" in state_meta and state_meta["elim_order"] is not None,
      f"elim_order length = {len(state_meta.get('elim_order', []))}")

check("meta_has_num_trained",
      "num_trained" in state_meta and state_meta["num_trained"] > 0,
      f"num_trained = {state_meta.get('num_trained')}")

check("meta_has_bucket_complexities",
      "bucket_complexities" in state_meta,
      f"bucket_complexities length = {len(state_meta.get('bucket_complexities', []))}")

# Check that NN weights are stripped in metadata-only mode
meta_entries_with_weights = [
    e for e in state_meta["per_bucket_training_log"]
    if "nn_state_dict" in e
]
check("meta_no_weights",
      len(meta_entries_with_weights) == 0,
      f"{len(meta_entries_with_weights)} entries have nn_state_dict (expected 0)")

# Verify loss curves survived round-trip
meta_entries_with_losses = [
    e for e in state_meta["per_bucket_training_log"]
    if len(e.get("losses", [])) > 0
]
check("meta_losses_roundtrip",
      len(meta_entries_with_losses) == len(nn_buckets_with_losses),
      f"loaded losses count ({len(meta_entries_with_losses)}) matches original ({len(nn_buckets_with_losses)})")

if meta_entries_with_losses:
    me = meta_entries_with_losses[0]
    check("meta_entry_losses_format",
          isinstance(me["losses"][0], (tuple, list)) and len(me["losses"][0]) == 2,
          f"loaded losses[0] = {me['losses'][0]}")
    check("meta_entry_epochs_trained",
          me.get("epochs_trained", 0) > 0,
          f"epochs_trained = {me.get('epochs_trained')}")
    check("meta_entry_hidden_sizes",
          isinstance(me.get("hidden_sizes"), list),
          f"hidden_sizes = {me.get('hidden_sizes')}")

# ---------------------------------------------------------------------------
# Step 3: Weights-included save/load round-trip
# ---------------------------------------------------------------------------

print("\n--- Step 3: Weights-included save/load round-trip ---")

# Re-run inference with save_nn_weights=True so weights are captured
config_weights = _BASE_CONFIG.copy()
config_weights["save_nn_weights"] = True

fastgm_w = FastGM(model=model, nn_config=config_weights, device=device)
fastgm_w.eliminate_variables(all=True)

weights_path = "/tmp/test_s04_weights.pkl"
save_state(fastgm_w, weights_path, save_weights=True)
state_weights = load_state(weights_path)

wlog = state_weights["per_bucket_training_log"]
entries_with_weights = [e for e in wlog if "nn_state_dict" in e]

check("weights_captured",
      len(entries_with_weights) > 0,
      f"{len(entries_with_weights)}/{len(wlog)} entries have nn_state_dict")

if entries_with_weights:
    we = entries_with_weights[0]
    check("weights_is_dict",
          isinstance(we["nn_state_dict"], (dict, OrderedDict)),
          f"nn_state_dict type = {type(we['nn_state_dict']).__name__}")

    check("weights_nonempty",
          len(we["nn_state_dict"]) > 0,
          f"nn_state_dict has {len(we['nn_state_dict'])} keys")

    check("normalizing_constant_present",
          "normalizing_constant" in we,
          "normalizing_constant key present")

    nc = we.get("normalizing_constant")
    check("normalizing_constant_finite",
          nc is not None and math.isfinite(nc),
          f"normalizing_constant = {nc}")
else:
    check("weights_is_dict", False, "No weight entries to inspect")
    check("weights_nonempty", False, "Skipped")
    check("normalizing_constant_present", False, "Skipped")
    check("normalizing_constant_finite", False, "Skipped")

# ---------------------------------------------------------------------------
# Step 4: undo_normalization
# ---------------------------------------------------------------------------

print("\n--- Step 4: undo_normalization ---")

if entries_with_weights:
    nc = entries_with_weights[0]["normalizing_constant"]
    dummy = torch.tensor([0.0, 1.0, -1.0])
    result = undo_normalization(dummy, nc)

    check("undo_norm_finite",
          torch.all(torch.isfinite(result)).item(),
          f"result = {result.tolist()}")

    check("undo_norm_different",
          not torch.allclose(result, dummy) or nc == 0.0,
          f"result differs from input (nc={nc})")

    check("undo_norm_shape",
          result.shape == dummy.shape,
          f"shape {result.shape} == {dummy.shape}")
else:
    check("undo_norm_finite", False, "No weight entries — skipped")
    check("undo_norm_different", False, "Skipped")
    check("undo_norm_shape", False, "Skipped")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

summarize_and_exit()
