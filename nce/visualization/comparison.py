"""Cross-experiment training comparison visualization.

Compare training results across multiple inference runs by overlaying
learning curves and summarizing final losses per bucket.

Usage::

    from nce.visualization import compare_experiments

    fig = compare_experiments(
        [fastgm_a, state_dict_b, "run_c.pkl"],
        labels=["baseline", "new_loss", "tuned_lr"],
    )

    import matplotlib.pyplot as plt
    plt.close(fig)
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nce.visualization.learning_curves import _extract_training_log


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare_experiments(
    sources: List[Any],
    labels: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    max_subplots: int = 20,
) -> plt.Figure:
    """Compare training across multiple experiments.

    Produces a figure with two panel types:

    1. **Summary bar chart** (first subplot): grouped bars showing the
       final loss per bucket for each experiment.
    2. **Per-bucket overlays** (remaining subplots): one subplot per
       shared bucket label (present in ≥2 experiments), with overlaid
       learning curves and a legend identifying experiments.

    Args:
        sources: List of sources — each can be a live FastGM, a state
            dict (from ``load_state()``), or a file path (str/Path).
        labels: Optional display labels for each experiment.  If *None*,
            auto-generated as ``"Experiment 1"``, ``"Experiment 2"``, etc.
        save_path: If provided, the figure is saved to this path.
        max_subplots: Maximum number of per-bucket overlay subplots
            (does not count the summary panel).

    Returns:
        A ``matplotlib.figure.Figure``.  The caller owns the figure.

    Raises:
        ValueError: If *sources* is empty.
    """
    if not sources:
        raise ValueError(
            "compare_experiments: sources list is empty — "
            "provide at least one FastGM, state dict, or file path."
        )

    # --- Normalize sources ---
    n_exp = len(sources)
    if labels is None:
        labels = [f"Experiment {i + 1}" for i in range(n_exp)]
    if len(labels) != n_exp:
        raise ValueError(
            f"compare_experiments: len(labels)={len(labels)} does not match "
            f"len(sources)={n_exp}."
        )

    # Build per-experiment bucket index: label → entry
    # Also collect all bucket labels in insertion order.
    all_bucket_labels: List[str] = []  # ordered, unique
    seen_labels: Set[str] = set()
    experiment_maps: List[Dict[str, Dict]] = []

    for src in sources:
        log = _extract_training_log(src)
        bmap: Dict[str, Dict] = {}
        for entry in log:
            blabel = str(entry.get("label", "?"))
            bmap[blabel] = entry
            if blabel not in seen_labels:
                all_bucket_labels.append(blabel)
                seen_labels.add(blabel)
        experiment_maps.append(bmap)

    # --- Identify shared bucket labels (in ≥2 experiments) ---
    shared_labels = [
        bl for bl in all_bucket_labels
        if sum(1 for emap in experiment_maps if bl in emap) >= 2
    ]

    # Cap overlay subplots
    total_shared = len(shared_labels)
    if total_shared > max_subplots:
        warnings.warn(
            f"compare_experiments: {total_shared} shared bucket labels exceed "
            f"max_subplots={max_subplots}; showing first {max_subplots}.",
            stacklevel=2,
        )
        shared_labels = shared_labels[:max_subplots]

    n_overlays = len(shared_labels)
    n_total_subplots = 1 + n_overlays  # summary + overlays

    # --- Layout ---
    ncols = min(4, n_total_subplots)
    nrows = math.ceil(n_total_subplots / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 4 * nrows),
        squeeze=False,
    )

    # Color cycle for experiments
    cmap = plt.cm.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(n_exp)]

    # --- Panel 1: Summary bar chart ---
    ax_summary = axes[0][0]
    _draw_summary_bars(ax_summary, all_bucket_labels, experiment_maps, labels, colors)

    if not shared_labels:
        ax_summary.set_title("Final Loss per Bucket (no shared buckets)", fontsize=10)
    else:
        ax_summary.set_title("Final Loss per Bucket — Summary", fontsize=10)

    # --- Per-bucket overlay panels ---
    for overlay_idx, blabel in enumerate(shared_labels):
        flat_idx = 1 + overlay_idx  # skip summary panel
        row, col = divmod(flat_idx, ncols)
        ax = axes[row][col]

        for exp_idx, (emap, exp_label) in enumerate(zip(experiment_maps, labels)):
            entry = emap.get(blabel)
            if entry is None:
                continue
            losses = entry.get("losses", [])
            if not losses:
                continue
            epochs = [t[0] for t in losses]
            values = [t[1] for t in losses]
            ax.plot(
                epochs, values,
                label=exp_label,
                color=colors[exp_idx],
                linewidth=1.2,
            )

        ax.set_title(f"Bucket {blabel}", fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Loss", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)

    # --- Hide unused subplot slots ---
    for flat_idx in range(n_total_subplots, nrows * ncols):
        row, col = divmod(flat_idx, ncols)
        axes[row][col].set_visible(False)

    # --- Overall title ---
    suptitle = f"Experiment Comparison — {n_exp} experiments"
    if n_overlays > 0:
        suptitle += f", {n_overlays} shared buckets"
    elif len(all_bucket_labels) > 0:
        suptitle += " (no overlapping bucket labels)"
    fig.suptitle(suptitle, fontsize=12, y=1.01)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_final_loss(entry: Dict) -> Optional[float]:
    """Return the last loss value from an entry, or None if unavailable."""
    losses = entry.get("losses", [])
    if not losses:
        return None
    # Each element is (epoch, loss_value)
    return losses[-1][1]


def _draw_summary_bars(
    ax: plt.Axes,
    bucket_labels: List[str],
    experiment_maps: List[Dict[str, Dict]],
    exp_labels: List[str],
    colors: List,
) -> None:
    """Draw a grouped bar chart of final loss per bucket on *ax*."""
    n_exp = len(experiment_maps)
    n_buckets = len(bucket_labels)

    if n_buckets == 0:
        ax.text(
            0.5, 0.5,
            "No buckets with training data.",
            ha="center", va="center", fontsize=11, transform=ax.transAxes,
        )
        ax.set_axis_off()
        return

    x = np.arange(n_buckets, dtype=float)
    total_width = 0.8
    bar_width = total_width / max(n_exp, 1)

    for exp_idx, (emap, exp_label) in enumerate(zip(experiment_maps, exp_labels)):
        final_losses = []
        for bl in bucket_labels:
            entry = emap.get(bl)
            if entry is not None:
                val = _get_final_loss(entry)
                final_losses.append(val if val is not None else 0.0)
            else:
                final_losses.append(0.0)

        offset = (exp_idx - n_exp / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            final_losses,
            width=bar_width * 0.9,
            label=exp_label,
            color=colors[exp_idx],
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"B{bl}" for bl in bucket_labels],
        fontsize=7,
        rotation=45 if n_buckets > 6 else 0,
        ha="right" if n_buckets > 6 else "center",
    )
    ax.set_ylabel("Final Loss", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
