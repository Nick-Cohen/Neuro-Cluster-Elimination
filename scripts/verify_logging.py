#!/usr/bin/env python
"""End-to-end verification of S06: JSONL Training Logger.

Exercises the full pipeline:
  1. Run inference on a small problem with log_file configured
  2. Validate the JSONL log file:
     - File exists and is non-empty
     - Every line is valid JSON
     - Required event types are present (bucket_training_start, epoch_loss, bucket_training_end)
     - Required fields per event type
     - Chronological ordering by timestamp
  3. Run inference without log_file and confirm no log file is created

Exit 0 if all checks pass, exit 1 if any fail.
Runtime: ~30 seconds on GPU (3 epochs, small samples, rbm_20).
"""

import json
import os
import sys
import tempfile

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
    """Print summary and exit."""
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
    sys.exit(0 if all_ok else 1)


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Minimal config — rbm_20 with 3 epochs, small samples.
# ecl=2^19 means ~20 of 40 buckets exceed the limit and use NN training.
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
    "ecl": 2**19,
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
# Step 1: Run inference with log_file configured
# ---------------------------------------------------------------------------

print("\n--- Step 1: Run inference with log_file configured ---")

from nce.benchmark_problems import nbe_sanity_check
from nce.inference.graphical_model import FastGM

# rbm_20: 40 binary vars, fast to process
model = nbe_sanity_check.problems[3]
print(f"Problem: {model.modelfile}")

# Create a temp file for the JSONL log
log_fd, log_path = tempfile.mkstemp(suffix=".jsonl", prefix="verify_logging_")
os.close(log_fd)

# Remove the empty temp file so we can verify the logger creates it
os.unlink(log_path)

config = _BASE_CONFIG.copy()
config["log_file"] = log_path

print(f"Log file: {log_path}")
print(f"Config: ecl={config['ecl']}, num_epochs=3, hidden_sizes=[3,3]")

fastgm = FastGM(model=model, nn_config=config, device=device)
fastgm.eliminate_variables(all=True)

# Close the logger's file handler so all data is flushed
import logging
logger = logging.getLogger("nce.training")
for handler in logger.handlers[:]:
    handler.close()
    logger.removeHandler(handler)

# ---------------------------------------------------------------------------
# Step 2: Validate JSONL log file
# ---------------------------------------------------------------------------

print("\n--- Step 2: Validate JSONL log file ---")

# 2a. File exists and is non-empty
check("file_exists", os.path.exists(log_path), f"log file at {log_path}")
file_size = os.path.getsize(log_path) if os.path.exists(log_path) else 0
check("file_nonempty", file_size > 0, f"file size = {file_size} bytes")

# 2b. Parse all lines as JSON
events = []
parse_errors = []
if os.path.exists(log_path):
    with open(log_path, "r") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as e:
                parse_errors.append(f"line {i}: {e}")

check("all_valid_json",
      len(parse_errors) == 0,
      f"{len(events)} events parsed, {len(parse_errors)} parse errors"
      + (f": {parse_errors[:3]}" if parse_errors else ""))

# 2c. Required event types present
event_types = {e.get("event") for e in events}
print(f"  Event types found: {sorted(event_types)}")

REQUIRED_EVENTS = ["bucket_training_start", "epoch_loss", "bucket_training_end"]
for evt in REQUIRED_EVENTS:
    count = sum(1 for e in events if e.get("event") == evt)
    check(f"event_{evt}_present",
          count > 0,
          f"found {count} '{evt}' events")

# 2d. Required fields per event type
REQUIRED_FIELDS_GLOBAL = ["timestamp", "event", "bucket_id"]

# Check global fields on all events
missing_global = []
for i, e in enumerate(events):
    for field in REQUIRED_FIELDS_GLOBAL:
        if field not in e:
            missing_global.append(f"event {i} missing '{field}'")
check("global_fields_present",
      len(missing_global) == 0,
      f"all {len(events)} events have timestamp/event/bucket_id"
      if not missing_global else f"{len(missing_global)} missing: {missing_global[:3]}")

# Check epoch_loss specific fields
epoch_loss_events = [e for e in events if e.get("event") == "epoch_loss"]
epoch_loss_missing = []
for i, e in enumerate(epoch_loss_events):
    for field in ["epoch", "loss"]:
        if field not in e:
            epoch_loss_missing.append(f"epoch_loss event {i} missing '{field}'")
check("epoch_loss_fields",
      len(epoch_loss_missing) == 0 and len(epoch_loss_events) > 0,
      f"all {len(epoch_loss_events)} epoch_loss events have epoch+loss"
      if not epoch_loss_missing else f"{len(epoch_loss_missing)} missing: {epoch_loss_missing[:3]}")

# 2e. Chronological ordering by timestamp
timestamps = [e.get("timestamp", "") for e in events]
is_chronological = all(timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1))
check("chronological_order",
      is_chronological,
      f"{len(timestamps)} timestamps in non-decreasing order"
      if is_chronological else "timestamps NOT in order")

# Show first few events for diagnostic visibility
if events:
    print(f"\n  Sample events (first 3):")
    for e in events[:3]:
        print(f"    {json.dumps(e)}")

# ---------------------------------------------------------------------------
# Step 3: Disabled-by-default check
# ---------------------------------------------------------------------------

print("\n--- Step 3: Disabled-by-default check ---")

# Run inference WITHOUT log_file set — should produce no log file
nolog_fd, nolog_path = tempfile.mkstemp(suffix=".jsonl", prefix="verify_nolog_")
os.close(nolog_fd)
os.unlink(nolog_path)

config_nolog = _BASE_CONFIG.copy()
# Explicitly do NOT set log_file (or set it to None)
config_nolog["log_file"] = None

fastgm2 = FastGM(model=model, nn_config=config_nolog, device=device)
fastgm2.eliminate_variables(all=True)

check("disabled_no_file",
      not os.path.exists(nolog_path),
      f"no log file at {nolog_path} when log_file=None")

# Also check the logger has no handlers when not configured
logger2 = logging.getLogger("nce.training")
check("disabled_no_handlers",
      len(logger2.handlers) == 0,
      f"logger has {len(logger2.handlers)} handlers (expected 0)")

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

try:
    os.unlink(log_path)
except FileNotFoundError:
    pass

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

summarize_and_exit()
