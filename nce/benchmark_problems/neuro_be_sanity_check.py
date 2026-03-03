"""neuro_be_sanity_check benchmark set.

Contains 4 models for quick sanity-checking of neural bucket elimination:
- pedigree13: large pedigree network (1077 vars, width 32)
- grid40x40.f10: large grid model (1600 vars, width 54)
- grid20x20.f10: medium grid model (400 vars)
- rbm_20: restricted Boltzmann machine (40 vars, width 20)
"""
from .catalog_utils import get_catalog


def _load_benchmark_set():
    """Load the neuro_be_sanity_check benchmark models from the catalogue."""
    catalog = get_catalog()
    models = [
        catalog['pedigree/pedigree13'],
        catalog['grids/grid40x40.f10'],
        catalog['grids/grid20x20.f10'],
        catalog['dbn/rbm_20'],
    ]
    return models


# Load on import so users can do: benchmark_problems.neuro_be_sanity_check
neuro_be_sanity_check = _load_benchmark_set()
