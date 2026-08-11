"""Dtype resolution utility for float32/float64 precision mode."""

import torch


def get_dtype(config):
    """Return torch.float64 if use_float64 is enabled, else torch.float32."""
    if config.get('use_float64', False):
        return torch.float64
    return torch.float32
