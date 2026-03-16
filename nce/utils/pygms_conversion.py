"""
Utilities for converting between pyGMs and NCE factor representations.
"""

import numpy as np
import torch
from typing import List, Optional

# Conversion constant
LN10 = np.log(10)


def pygms_factor_to_fastfactor(pf, device: str = 'cpu'):
    """
    Convert a pyGMs Factor to an NCE FastFactor.

    pyGMs factors are in natural log space, FastFactors are in log10 space.

    Args:
        pf: pyGMs Factor object (in natural log space)
        device: Device for the output tensor ('cpu' or 'cuda')

    Returns:
        FastFactor in log10 space
    """
    from ..inference.factor import FastFactor

    # Get variable labels
    labels = [v.label for v in pf.vars]

    # Convert from natural log to log10
    table_ln = pf.t
    table_log10 = table_ln / LN10

    # Create tensor with appropriate shape
    if labels:
        shape = tuple(v.states for v in pf.vars)
        tensor = torch.tensor(table_log10.reshape(shape), dtype=torch.float32, device=device)
    else:
        # Scalar factor
        tensor = torch.tensor(float(table_log10.flatten()[0]), dtype=torch.float32, device=device)

    return FastFactor(tensor, labels)


def fastfactor_to_pygms(ff, X: Optional[List] = None):
    """
    Convert an NCE FastFactor to a pyGMs Factor.

    FastFactors are in log10 space, pyGMs factors are in natural log space.

    Args:
        ff: NCE FastFactor object (in log10 space)
        X: List of pyGMs Var objects. If None, creates new variables.

    Returns:
        pyGMs Factor in natural log space
    """
    import sys
    if '/home/cohenn1/SDBE/PyGMs' not in sys.path:
        sys.path.insert(0, '/home/cohenn1/SDBE/PyGMs')
    import pyGMs as gm

    # Get variable indices from labels
    var_indices = [v if isinstance(v, int) else v.label for v in ff.labels]

    # Create pyGMs variables if not provided
    if X is None:
        # Infer states from tensor shape
        if ff.labels:
            states = list(ff.tensor.shape)
            X = [gm.Var(i, states[j]) for j, i in enumerate(var_indices)]
        else:
            X = []

    # Convert tensor from log10 to natural log
    table_log10 = ff.tensor.cpu().numpy()
    table_ln = table_log10 * LN10

    if ff.labels:
        vars_list = [X[i] if isinstance(X, list) and i < len(X) else X[i] for i in range(len(var_indices))]
        # Handle case where X is a dict or list indexed by variable label
        if isinstance(X, dict):
            vars_list = [X[i] for i in var_indices]
        elif isinstance(X, list) and len(X) > max(var_indices):
            vars_list = [X[i] for i in var_indices]
        pf = gm.Factor(vars_list, table_ln.flatten())
    else:
        # Scalar factor
        pf = gm.Factor([], [float(table_ln.flatten()[0])])

    return pf


def pygms_factors_to_fastfactors(factors, device: str = 'cpu'):
    """
    Convert a list of pyGMs Factors to NCE FastFactors.

    Args:
        factors: List of pyGMs Factor objects
        device: Device for the output tensors

    Returns:
        List of FastFactors
    """
    return [pygms_factor_to_fastfactor(f, device) for f in factors]
