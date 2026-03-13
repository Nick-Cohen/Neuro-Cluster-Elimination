"""Per-bucket learning curve visualization.

Plot training (and optional validation) loss over epochs for each
NN-trained bucket in an inference run.

Usage::

    from nce.visualization import plot_learning_curves

    fig = plot_learning_curves(fastgm)          # live FastGM
    fig = plot_learning_curves(state_dict)       # dict from load_state()
    fig = plot_learning_curves("run.pkl")        # file path

    # Caller owns the figure — close when done:
    import matplotlib.pyplot as plt
    plt.close(fig)
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless use
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helper: normalize any supported source to a training log list
# ---------------------------------------------------------------------------

def _extract_training_log(source: Any) -> List[Dict]:
    """Extract ``per_bucket_training_log`` from *source*.

    Accepted source types:

    * **FastGM** — object with a ``per_bucket_training_log`` attribute.
    * **dict** — state dict (as returned by ``load_state()``) containing
      a ``per_bucket_training_log`` key.
    * **str | Path** — filesystem path; loaded via ``nce.state.load_state()``.

    Returns:
        List of training-log entry dicts.  Each entry has at minimum
        ``label``, ``losses``, ``val_losses``, ``epochs_trained``, and
        ``hidden_sizes`` keys.

    Raises:
        TypeError: If *source* is none of the above types.  The error
            message names the unrecognized type for easy debugging.
    """
    # 1. Live FastGM — has the attribute directly
    if hasattr(source, "per_bucket_training_log"):
        return source.per_bucket_training_log

    # 2. State dict
    if isinstance(source, dict):
        if "per_bucket_training_log" in source:
            return source["per_bucket_training_log"]
        raise TypeError(
            f"_extract_training_log: dict source has no 'per_bucket_training_log' key "
            f"(keys present: {list(source.keys())})"
        )

    # 3. File path
    if isinstance(source, (str, Path)):
        from nce.state.state import load_state
        state = load_state(source)
        return state["per_bucket_training_log"]

    raise TypeError(
        f"_extract_training_log: unrecognized source type {type(source).__name__!r}. "
        f"Expected a FastGM instance, a state dict, or a file path (str/Path)."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_learning_curves(
    source: Any,
    save_path: Optional[Union[str, Path]] = None,
    max_subplots: int = 20,
    bucket_labels: Optional[Sequence[int]] = None,
) -> plt.Figure:
    """Create a multi-subplot figure of per-bucket learning curves.

    One subplot is created for each NN-trained bucket that has a non-empty
    ``losses`` list.  Training loss is always plotted; validation loss is
    overlaid when ``val_losses`` is non-empty.

    Args:
        source: A live FastGM, a state dict (from ``load_state()``), or
            a file path (str/Path) to a saved state pickle.
        save_path: If provided, the figure is saved to this path via
            ``fig.savefig(..., bbox_inches='tight')``.
        max_subplots: Maximum number of subplots to render.  If the
            training log has more plottable entries, only the first
            *max_subplots* are shown and a warning is emitted.
        bucket_labels: If provided, only entries whose ``label`` is in
            this sequence are plotted.

    Returns:
        A ``matplotlib.figure.Figure``.  **The caller owns the figure and
        should call** ``plt.close(fig)`` **when done** to free memory.

    Raises:
        TypeError: If *source* is an unrecognized type (propagated from
            ``_extract_training_log``).
    """
    training_log = _extract_training_log(source)

    # --- Filter to entries with actual loss data ---
    entries = [e for e in training_log if e.get("losses")]

    # --- Optionally filter by bucket label ---
    if bucket_labels is not None:
        label_set = set(bucket_labels)
        entries = [e for e in entries if e.get("label") in label_set]

    # --- Cap at max_subplots ---
    total = len(entries)
    if total > max_subplots:
        warnings.warn(
            f"plot_learning_curves: {total} plottable entries exceed "
            f"max_subplots={max_subplots}; showing first {max_subplots}.",
            stacklevel=2,
        )
        entries = entries[:max_subplots]

    n_entries = len(entries)

    # --- Edge case: nothing to plot ---
    if n_entries == 0:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        ax.text(
            0.5, 0.5,
            "No NN-trained buckets with loss data to display.",
            ha="center", va="center", fontsize=12, transform=ax.transAxes,
        )
        ax.set_axis_off()
        if save_path is not None:
            fig.savefig(str(save_path), bbox_inches="tight")
        return fig

    # --- Grid layout ---
    ncols = min(4, n_entries)
    nrows = math.ceil(n_entries / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 3.5 * nrows),
        squeeze=False,
    )

    for idx, entry in enumerate(entries):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        # Unpack (epoch, loss_value) tuples
        epochs_train = [t[0] for t in entry["losses"]]
        losses_train = [t[1] for t in entry["losses"]]
        ax.plot(epochs_train, losses_train, label="train", linewidth=1.2)

        val_losses = entry.get("val_losses", [])
        if val_losses:
            epochs_val = [t[0] for t in val_losses]
            losses_val = [t[1] for t in val_losses]
            ax.plot(epochs_val, losses_val, label="val", linewidth=1.2, linestyle="--")
            ax.legend(fontsize=8)

        # Title with hidden_sizes annotation
        hs = entry.get("hidden_sizes", "?")
        ax.set_title(f"Bucket {entry['label']}  (h={hs})", fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Loss", fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for idx in range(n_entries, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # Overall title
    title = f"Learning Curves — {n_entries} NN buckets"
    if total > max_subplots:
        title += f" (showing {max_subplots} of {total})"
    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), bbox_inches="tight")

    return fig
