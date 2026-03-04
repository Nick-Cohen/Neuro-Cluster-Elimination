"""nbe_sanity_check benchmark set.

Contains 4 models for quick sanity-checking of neural bucket elimination:
- pedigree13: large pedigree network (1077 vars, width 32)
- grid40x40.f10: large grid model (1600 vars, width 54)
- grid20x20.f10: medium grid model (400 vars)
- rbm_20: restricted Boltzmann machine (40 vars, width 20)

Usage:
    from nce.benchmark_problems import nbe_sanity_check

    for model, config in zip(nbe_sanity_check.problems, nbe_sanity_check.configs['nbe']):
        print(model.modelfile, config)
"""
from .catalog_utils import get_catalog


class BenchmarkSet:
    """A named collection of benchmark problems with associated config sets."""
    def __init__(self, problems, configs):
        self.problems = problems   # list of Model objects
        self.configs = configs     # dict: config_name -> list of config dicts (same order as problems)


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
    """Load the nbe_sanity_check benchmark models from the catalogue."""
    catalog = get_catalog()
    return [catalog[key] for key in _MODEL_KEYS]


def _build_nbe_configs():
    """Build fully-populated neuroBE config dicts for each model.

    Returns a list of config dicts (one per model, same order as _MODEL_KEYS).
    Each dict contains all fields needed to run a neuroBE experiment.
    """
    configs = []
    for key in _MODEL_KEYS:
        configs.append({
            'device': 'cuda',
            'hidden_sizes': _HIDDEN_SIZES_MAP[key],
            'optimizer': 'adam',
            'lr': 0.01,
            'lr_decay': 1.0,
            'momentum': 0.9,
            'inverse_time_decay_constant': 100,
            'patience': 20,
            'min_lr': 1e-8,
            'num_epochs': 10000,
            'num_epochs2': 0,
            'nbe_early_stopping': False,
            'nbe_warmup_epochs': 0,
            'skip_early_stopping': True,
            'sampling_scheme': 'uniform',
            'batch_size': 256,
            'set_size': 50000,
            'num_samples': 'nbe',
            'num_batches_per_set': 1,
            'loss_fn': 'unnormalized_kl',
            'traced_losses': [],
            'val_set': True,
            'fdb': False,
            'use_bw_approx': True,
            'populate_bw_factors': False,
            'ecl': 2**22,
            'iB': 10,
            'approximation_method': 'wmb',
            'bw_ecl': None,
            'backward_ecl': 2**22,
            'backward_iB': 10,
            'use_linspace_bias': False,
            'use_memorizer': False,
            'display_intermediate': False,
            'track_errors': False,
            'plot_messages': False,
            'debug': False,
            'lower_dim': False,
            'dope_factors': False,
            'gather_message_stats': False,
            'stratify_samples': False,
            'seed': 42,
        })
    return configs


# Module-level instance: single import gives access to both problems and configs.
nbe_sanity_check = BenchmarkSet(
    problems=_load_benchmark_set(),
    configs={'nbe': _build_nbe_configs()},
)
