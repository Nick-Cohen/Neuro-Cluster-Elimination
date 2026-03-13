"""Visualization tools for NCE inference results.

Provides plotting functions that accept live FastGM objects, state dicts
(from ``load_state()``), or file paths — returning matplotlib Figures
without calling ``plt.show()``.
"""

from nce.visualization.learning_curves import plot_learning_curves

__all__ = ["plot_learning_curves"]
