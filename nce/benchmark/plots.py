"""Benchmark training plot generation.

Plotting functions for the benchmark training harness:
loss curves, local error curves, and factor comparison plots.
All use the Agg backend and produce standalone PNG files suitable
for batch/headless execution.

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
import numpy as np
import torch


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


def plot_top_assignments(
    exact_fw: "FastFactor",
    approx_fw: "FastFactor",
    exact_bw: "FastFactor",
    output_path: str,
    title_prefix: str = "Top Assignments",
    max_assignments: int = 100,
    coverage_threshold: float = 0.999,
) -> str:
    """Plot exact vs approximate forward message at the most important assignments.

    Importance is determined by the product exact_fw * exact_bw (unnormalized
    log-probabilities).  We select the smallest set of assignments whose
    cumulative probability mass reaches *coverage_threshold*, capped at
    *max_assignments*.

    Args:
        exact_fw: Exact forward message (FastFactor, log10 values).
        approx_fw: Approximate forward message (FastFactor, log10 values).
        exact_bw: Exact backward message (FastFactor, log10 values).
        output_path: File path for the output PNG.
        title_prefix: Prefix for the plot title (bucket info).
        max_assignments: Hard cap on the number of assignments to show.
        coverage_threshold: Cumulative probability fraction to cover (0-1).

    Returns:
        The output_path that was written.
    """
    # 1. Compute product factor (log10 space — FastFactor.__mul__ adds in log-space)
    product = exact_fw * exact_bw
    product_flat = product.tensor.detach().flatten()

    # 2. Sort descending
    sorted_vals, sorted_indices = torch.sort(product_flat, descending=True)

    # 3. Find how many assignments cover the threshold.
    #    Values are in log10 — convert to natural log for logsumexp.
    ln10 = torch.log(torch.tensor(10.0, device=sorted_vals.device))
    sorted_ln = sorted_vals * ln10
    total_ln = torch.logsumexp(sorted_ln, dim=0)  # log(Z) in natural log

    # Cumulative logsumexp
    cumulative_ln = torch.logcumsumexp(sorted_ln, dim=0)
    # Fraction of probability covered at each rank
    cum_fraction = torch.exp(cumulative_ln - total_ln)

    # First index where coverage >= threshold (or all if never reached)
    above = (cum_fraction >= coverage_threshold).nonzero(as_tuple=True)[0]
    if len(above) > 0:
        n_for_coverage = above[0].item() + 1  # +1 for 1-indexed count
    else:
        n_for_coverage = len(sorted_vals)

    n = min(max_assignments, n_for_coverage)
    actual_coverage = cum_fraction[n - 1].item()

    # 4. Extract exact and approx values at the selected indices
    exact_flat = exact_fw.tensor.detach().flatten()
    approx_flat = approx_fw.tensor.detach().flatten()

    top_indices = sorted_indices[:n]
    exact_top = exact_flat[top_indices].cpu().numpy()
    approx_top = approx_flat[top_indices].cpu().numpy()

    # 5. Plot
    ranks = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ranks, exact_top, marker='_', markersize=6, markeredgewidth=1.5,
            linestyle='None', label="Exact", color="#1f77b4")
    ax.plot(ranks, approx_top, marker='_', markersize=6, markeredgewidth=1.5,
            linestyle='None', label="Approx", color="#ff7f0e")
    ax.set_xlabel("Assignment rank (1 = most important)", fontsize=10)
    ax.set_ylabel("Log10 unnormalized probability", fontsize=10)

    coverage_pct = actual_coverage * 100
    title = f"{title_prefix} — top {n} assignments ({coverage_pct:.1f}% probability)"
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path
