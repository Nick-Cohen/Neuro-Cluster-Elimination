"""neuro_be_sanity_check benchmark set.

Contains 4 models for quick sanity-checking of neural bucket elimination:
- pedigree13: large pedigree network (1077 vars, width 32)
- grid40x40.f10: large grid model (1600 vars, width 54)
- grid20x20.f10: medium grid model (400 vars)
- rbm_20: restricted Boltzmann machine (40 vars, width 20)

Each model has a paired neuroBE config dict available via
``neuro_be_sanity_check_configs`` (keyed by catalogue key) or
``neuro_be_sanity_check_configs_list`` (same order as the models list).
"""
from .catalog_utils import get_catalog

# Catalogue keys used by this benchmark set (single source of truth).
_MODEL_KEYS = [
    'pedigree/pedigree13',
    'grids/grid40x40.f10',
    'grids/grid20x20.f10',
    'dbn/rbm_20',
]

# Per-model neuroBE hidden_sizes multiplier.
_HIDDEN_SIZES_MAP = {
    'pedigree/pedigree13': 'nbe,3',
    'grids/grid40x40.f10': 'nbe,1',
    'grids/grid20x20.f10': 'nbe,1',
    'dbn/rbm_20': 'nbe,3',
}


def _load_benchmark_set():
    """Load the neuro_be_sanity_check benchmark models from the catalogue."""
    catalog = get_catalog()
    return [catalog[key] for key in _MODEL_KEYS]


def _build_configs():
    """Build neuroBE config dicts for each model in the benchmark set.

    Returns a dict keyed by catalogue key.  Each value is an nn_config dict
    containing batch_size, hidden_sizes, and num_samples settings appropriate
    for neuroBE experiments on that model.
    """
    configs = {}
    for key in _MODEL_KEYS:
        configs[key] = {
            'batch_size': 256,
            'hidden_sizes': _HIDDEN_SIZES_MAP[key],
            'num_samples': 'nbe',
        }
    return configs


# Load on import so users can do: benchmark_problems.neuro_be_sanity_check
neuro_be_sanity_check = _load_benchmark_set()

# Config dicts keyed by catalogue key.
neuro_be_sanity_check_configs = _build_configs()

# Config dicts as a list, same order as neuro_be_sanity_check models list.
neuro_be_sanity_check_configs_list = [neuro_be_sanity_check_configs[key] for key in _MODEL_KEYS]
