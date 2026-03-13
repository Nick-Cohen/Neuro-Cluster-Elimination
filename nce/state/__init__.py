"""State preservation for FastGM inference results.

Provides save/load of post-inference state (loss curves, configs, optional NN weights)
and utility functions for working with saved state.
"""

from nce.state.state import save_state, load_state, undo_normalization

__all__ = ["save_state", "load_state", "undo_normalization"]
