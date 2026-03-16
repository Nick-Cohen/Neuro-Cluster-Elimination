"""
Utilities for loading UAI graphical model files.

Migrated from pyGMs.neuro to keep pyGMs as a stock dependency.
"""
import os
from pyGMs.filetypes import readUai
from pyGMs.graphmodel import GraphModel


def uai_to_GM(uai_file, order_file=None, evid_file=None, elim_order=None):
    """Load a UAI file and return a pyGMs GraphModel with optional elimination order."""
    factors = readUai(uai_file)
    if order_file is not None and os.path.exists(order_file):
        return GraphModel(factorList=factors, elim_order_file=order_file)
    elif elim_order is not None:
        return GraphModel(factorList=factors, elim_order=elim_order)
    else:
        return GraphModel(factorList=factors)


def get_order(order_file):
    """Read an elimination order from a .vo file."""
    order = []
    skip_first = True
    with open(order_file) as f:
        for line in f:
            if line[0].isdigit():
                if skip_first:
                    skip_first = False
                    continue
                order.append(int(line))
    return order
