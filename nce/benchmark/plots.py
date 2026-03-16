"""Benchmark training plot generation.

Two single-purpose plotting functions for the benchmark training harness:
loss curves and local error curves. Both use the Agg backend and produce
standalone PNG files suitable for batch/headless execution.

Follows the matplotlib pattern from nce/visualization/learning_curves.py:
  - matplotlib.use("Agg") before pyplot import
  - fig.savefig(path, bbox_inches="tight")
  - explicit plt.close(fig) for memory cleanup
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss_curve(
    losses: List[Tuple[int, float]],
    output_path: str,
    title: Optional[str] = None,
) -> str:
    """Plot training loss over epochs on a semilogy scale.

    Args:
        losses: List of (epoch, loss_value) tuples.
        output_path: File path for the output PNG.
        title: Plot title. Defaults to "Training Loss".

    Returns:
        The output_path that was written (for chaining).
    """
    if title is None:
        title = "Training Loss"

    epochs = [t[0] for t in losses]
    values = [t[1] for t in losses]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(epochs, values, linewidth=1.2)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss (log scale)", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_local_error_curve(
    error_tracking_data: List[Tuple[int, float, float, float]],
    output_path: str,
    title: Optional[str] = None,
) -> str:
    """Plot absolute log-Z error over epochs on a semilogy scale.

    Args:
        error_tracking_data: List of (epoch, loss, log_z_err, abs_log_z_err)
            tuples as produced by train_single_bucket().
        output_path: File path for the output PNG.
        title: Plot title. Defaults to "Local Error (|log Z err|)".

    Returns:
        The output_path that was written (for chaining).
    """
    if title is None:
        title = "Local Error (|log Z err|)"

    epochs = [t[0] for t in error_tracking_data]
    abs_errors = [t[3] for t in error_tracking_data]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(epochs, abs_errors, linewidth=1.2, marker='o', markersize=4)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("|log Z err| (log scale)", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path
