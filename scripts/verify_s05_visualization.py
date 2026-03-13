#!/usr/bin/env python
"""End-to-end verification of S05: Visualization Module.

Exercises both ``plot_learning_curves()`` and ``compare_experiments()``
on real inference data from rbm_20.  Tests all three input modes
(FastGM, state dict, file path) plus edge cases.

Exit 0 if all checks pass, exit 1 if any fail.
Runtime: ~30 seconds (one inference pass, 3 epochs, small samples).
"""

import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# ---------------------------------------------------------------------------
# Helpers (same pattern as S04)
# ---------------------------------------------------------------------------

results = []  # list of (group_name, passed: bool, detail: str)
_temp_files = []  # paths to clean up at exit


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
    for p in _temp_files:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
    sys.exit(0 if all_ok else 1)


# ---------------------------------------------------------------------------
# Shared config (matches S04 verification)
# ---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

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
    "save_nn_weights": False,
}

# ---------------------------------------------------------------------------
# Setup: Run inference, save state, load state
# ---------------------------------------------------------------------------

print("\n--- Setup: Run inference on rbm_20 ---")

from nce.benchmark_problems import nbe_sanity_check
from nce.inference.graphical_model import FastGM
from nce.state import save_state, load_state
from nce.visualization import plot_learning_curves, compare_experiments

model = nbe_sanity_check.problems[3]  # rbm_20
config = _BASE_CONFIG.copy()

print(f"Problem: {model.modelfile}")
fastgm = FastGM(model=model, nn_config=config, device=device)
fastgm.eliminate_variables(all=True)

# Save state to disk for file-path tests
state_path = tempfile.mktemp(suffix=".pkl", prefix="test_s05_state_")
_temp_files.append(state_path)
save_state(fastgm, state_path, save_weights=False)

# Load state dict for dict tests
state_dict = load_state(state_path)

# Count entries with non-empty losses (expected subplot count)
training_log = fastgm.per_bucket_training_log
entries_with_losses = [e for e in training_log if e.get("losses")]
n_plottable = len(entries_with_losses)
print(f"Training log: {len(training_log)} entries, {n_plottable} with losses")


# ===================================================================
# Group 1: plot_learning_curves
# ===================================================================

print("\n--- plot_learning_curves checks ---")

# 1. FastGM → returns Figure
fig1 = plot_learning_curves(fastgm)
check("plc_fastgm_returns_figure",
      isinstance(fig1, plt.Figure),
      f"type = {type(fig1).__name__}")

# 2. Subplot count matches entries with losses
visible_axes = [ax for ax in fig1.axes if ax.get_visible()]
check("plc_subplot_count",
      len(visible_axes) == n_plottable,
      f"visible axes = {len(visible_axes)}, expected {n_plottable}")

# 3. Each subplot has at least one line (train loss)
all_have_lines = all(len(ax.get_lines()) >= 1 for ax in visible_axes)
check("plc_subplots_have_lines",
      all_have_lines,
      f"all {len(visible_axes)} visible axes have ≥1 line")

plt.close(fig1)

# 4. save_path produces a file on disk
save_png_path = tempfile.mktemp(suffix=".png", prefix="test_s05_plot_")
_temp_files.append(save_png_path)
fig_save = plot_learning_curves(fastgm, save_path=save_png_path)
check("plc_save_path_creates_file",
      os.path.isfile(save_png_path) and os.path.getsize(save_png_path) > 0,
      f"file exists, size = {os.path.getsize(save_png_path) if os.path.isfile(save_png_path) else 0}")
plt.close(fig_save)

# 5. State dict → returns Figure
fig_dict = plot_learning_curves(state_dict)
check("plc_state_dict_returns_figure",
      isinstance(fig_dict, plt.Figure),
      "accepted state dict")
plt.close(fig_dict)

# 6. File path → returns Figure
fig_path = plot_learning_curves(state_path)
check("plc_file_path_returns_figure",
      isinstance(fig_path, plt.Figure),
      "accepted file path str")
plt.close(fig_path)

# 7. Empty training log → Figure returned, no crash
empty_state = {"per_bucket_training_log": []}
fig_empty = plot_learning_curves(empty_state)
check("plc_empty_log_no_crash",
      isinstance(fig_empty, plt.Figure),
      "returned Figure for empty log")
empty_visible = [ax for ax in fig_empty.axes if ax.get_visible()]
check("plc_empty_log_axes_count",
      len(empty_visible) <= 1,
      f"empty log: {len(empty_visible)} visible axes (expected ≤1)")
plt.close(fig_empty)

# 8. Entry with empty losses list → skipped (no subplot for it)
sparse_state = {"per_bucket_training_log": [
    {"label": 1, "losses": [(1, 0.5), (2, 0.3)], "val_losses": [], "epochs_trained": 2, "hidden_sizes": [3]},
    {"label": 2, "losses": [], "val_losses": [], "epochs_trained": 0, "hidden_sizes": [3]},
    {"label": 3, "losses": [(1, 0.8)], "val_losses": [], "epochs_trained": 1, "hidden_sizes": [3]},
]}
fig_sparse = plot_learning_curves(sparse_state)
sparse_visible = [ax for ax in fig_sparse.axes if ax.get_visible()]
check("plc_empty_losses_skipped",
      len(sparse_visible) == 2,
      f"2 entries with data → {len(sparse_visible)} visible subplots (expected 2)")
plt.close(fig_sparse)

# 9. max_subplots=2 caps subplot count
fig_capped = plot_learning_curves(fastgm, max_subplots=2)
capped_visible = [ax for ax in fig_capped.axes if ax.get_visible()]
check("plc_max_subplots_cap",
      len(capped_visible) <= 2,
      f"max_subplots=2 → {len(capped_visible)} visible axes")
plt.close(fig_capped)


# ===================================================================
# Group 2: compare_experiments
# ===================================================================

print("\n--- compare_experiments checks ---")

# 1. Two state dicts → returns Figure
fig_cmp = compare_experiments([state_dict, state_dict], labels=["run_a", "run_b"])
check("ce_returns_figure",
      isinstance(fig_cmp, plt.Figure),
      f"type = {type(fig_cmp).__name__}")

# 2. Has at least 2 axes (summary + at least one overlay)
cmp_visible = [ax for ax in fig_cmp.axes if ax.get_visible()]
check("ce_has_multiple_axes",
      len(cmp_visible) >= 2,
      f"{len(cmp_visible)} visible axes (expected ≥2)")

# 3. Summary axis title contains "Summary" or "Final Loss"
summary_ax = cmp_visible[0]
summary_title = summary_ax.get_title()
check("ce_summary_title",
      "Summary" in summary_title or "Final Loss" in summary_title,
      f"summary title: {summary_title!r}")

plt.close(fig_cmp)

# 4. File paths accepted
fig_paths = compare_experiments([state_path, state_path], labels=["path_a", "path_b"])
check("ce_file_paths_accepted",
      isinstance(fig_paths, plt.Figure),
      "accepted list of file paths")
plt.close(fig_paths)

# 5. Labels appear in legend
fig_labeled = compare_experiments([state_dict, state_dict], labels=["Alpha", "Beta"])
labeled_visible = [ax for ax in fig_labeled.axes if ax.get_visible()]
# Check that at least one axis has a legend with the experiment names
found_label = False
for ax in labeled_visible:
    legend = ax.get_legend()
    if legend is not None:
        legend_texts = [t.get_text() for t in legend.get_texts()]
        if "Alpha" in legend_texts or "Beta" in legend_texts:
            found_label = True
            break
check("ce_labels_in_legend",
      found_label,
      "experiment labels found in axis legend")
plt.close(fig_labeled)

# 6. save_path produces a file on disk
save_cmp_path = tempfile.mktemp(suffix=".png", prefix="test_s05_compare_")
_temp_files.append(save_cmp_path)
fig_cmp_save = compare_experiments(
    [state_dict, state_dict], labels=["s1", "s2"], save_path=save_cmp_path
)
check("ce_save_path_creates_file",
      os.path.isfile(save_cmp_path) and os.path.getsize(save_cmp_path) > 0,
      f"file exists, size = {os.path.getsize(save_cmp_path) if os.path.isfile(save_cmp_path) else 0}")
plt.close(fig_cmp_save)

# 7. Single source (degenerate) → no crash
fig_single = compare_experiments([state_dict], labels=["only"])
check("ce_single_source_no_crash",
      isinstance(fig_single, plt.Figure),
      "single source accepted without crash")
plt.close(fig_single)

# 8. Two sources with no overlapping labels → Figure with summary only
no_overlap_a = {"per_bucket_training_log": [
    {"label": 100, "losses": [(1, 0.5)], "val_losses": [], "epochs_trained": 1, "hidden_sizes": [3]},
]}
no_overlap_b = {"per_bucket_training_log": [
    {"label": 200, "losses": [(1, 0.8)], "val_losses": [], "epochs_trained": 1, "hidden_sizes": [3]},
]}
fig_no_overlap = compare_experiments([no_overlap_a, no_overlap_b], labels=["X", "Y"])
no_overlap_visible = [ax for ax in fig_no_overlap.axes if ax.get_visible()]
check("ce_no_overlap_summary_only",
      len(no_overlap_visible) == 1,
      f"no shared labels → {len(no_overlap_visible)} visible axes (expected 1 = summary only)")
plt.close(fig_no_overlap)


# ===================================================================
# Summary
# ===================================================================

summarize_and_exit()
