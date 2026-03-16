"""nbe_sanity_check benchmark set.

Contains 5 models for quick sanity-checking of neural bucket elimination:
- pedigree13: large pedigree network (1077 vars, width 32)
- grid40x40.f10: large grid model (1600 vars, width 54)
- grid20x20.f10: medium grid model (400 vars)
- rbm_20: restricted Boltzmann machine (40 vars, width 20)
- grid10x10.f5.wrap: small grid model (100 vars)

NeuroBE hyperparameters (from NeuroBE Config.h):
- num_epochs=500
- lr=0.001
- loss_fn='weighted_logspace_mse'
- dope_factors=True (replaces -inf with finite values for stable training)

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
    'grids/grid10x10.f5.wrap',
]

# Per-model neuroBE hidden_sizes multiplier.
_HIDDEN_SIZES_MAP = {
    'pedigree/pedigree13': 'nbe,3',
    'grids/grid40x40.f10': 'nbe,1',
    'grids/grid20x20.f10': 'nbe,1',
    'dbn/rbm_20': 'nbe,3',
    'grids/grid10x10.f5.wrap': 'nbe,1',
}

# Per-model neuroBE num_samples epsilon values.
_NUM_SAMPLES_MAP = {
    'pedigree/pedigree13': 'nbe,0.1',
    'grids/grid40x40.f10': 'nbe,0.35',
    'grids/grid20x20.f10': 'nbe,0.35',
    'dbn/rbm_20': 'nbe,0.1',
    'grids/grid10x10.f5.wrap': 'nbe,0.35',
}

# Per-model i-bound values.
_IB_MAP = {
    'pedigree/pedigree13': 20,
    'grids/grid40x40.f10': 20,
    'grids/grid20x20.f10': 10,
    'dbn/rbm_20': 20,
    'grids/grid10x10.f5.wrap': 10,
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
            'lr': 0.001,
            'lr_decay': 1.0,
            'momentum': 0.9,
            'inverse_time_decay_constant': 100,
            'patience': 20,
            'min_lr': 1e-8,
            'num_epochs': 500,
            'num_epochs2': 0,
            'nbe_early_stopping': False,
            'nbe_warmup_epochs': 0,
            'skip_early_stopping': False,
            'sampling_scheme': 'uniform',
            'batch_size': 256,
            'set_size': None,
            'num_samples': _NUM_SAMPLES_MAP[key],
            'loss_fn': 'weighted_logspace_mse',
            'traced_losses': [],
            'val_set': True,
            'fdb': False,
            'use_bw_approx': False,
            'populate_bw_factors': False,
            'ecl': 2**(_IB_MAP[key] - 1),
            'iB': _IB_MAP[key],
            'approximation_method': 'nn',
            'bw_ecl': None,
            'backward_iB': _IB_MAP[key],
            'use_linspace_bias': False,
            'use_memorizer': False,
            'display_intermediate': False,
            'track_errors': False,
            'plot_messages': False,
            'debug': False,
            'lower_dim': False,
            'dope_factors': True,
            'gather_message_stats': False,
            'stratify_samples': False,
            'seed': 42,
        })
    return configs


def _build_nbe_nested_configs():
    """Build nested-format neuroBE config dicts for each model.

    Returns a list of 5 nested config dicts (one per model, same order as
    _MODEL_KEYS). Uses the 6-section schema from config_schema.NESTED_SECTIONS
    with readable alias names where available.

    These configs produce identical output to _build_nbe_configs() when passed
    through prepare_config().
    """
    configs = []
    for key in _MODEL_KEYS:
        configs.append({
            'inference': {
                'device': 'cuda',
                'exact_computation_limit': 2**(_IB_MAP[key] - 1),
                'i_bound': _IB_MAP[key],
                'approximation_method': 'nn',
                'dope_factors': True,
            },
            'nn': {
                'hidden_sizes': _HIDDEN_SIZES_MAP[key],
                'use_linspace_bias': False,
                'use_memorizer': False,
            },
            'training': {
                'num_epochs': 500,
                'num_epochs_phase2': 0,
                'loss_fn': 'weighted_logspace_mse',
                'optimizer': 'adam',
                'learning_rate': 0.001,
                'learning_rate_decay': 1.0,
                'momentum': 0.9,
                'batch_size': 256,
                'patience': 20,
                'min_learning_rate': 1e-8,
                'seed': 42,
                'skip_early_stopping': False,
                'nbe_early_stopping': False,
                'nbe_warmup_epochs': 0,
                'inverse_time_decay_constant': 100,
            },
            'sampling': {
                'sampling_scheme': 'uniform',
                'num_samples': _NUM_SAMPLES_MAP[key],
                'set_size': None,
                'val_set': True,
                'stratify_samples': False,
                'lower_dim': False,
            },
            'backward': {
                'use_backward_approximation': False,
                'populate_backward_factors': False,
                'backward_ecl': None,
                'backward_i_bound': _IB_MAP[key],
                'forward_diff_barrier': False,
            },
            'output': {
                'display_intermediate': False,
                'track_errors': False,
                'plot_messages': False,
                'debug': False,
                'gather_message_stats': False,
                'traced_losses': [],
            },
        })
    return configs


# Module-level instance: single import gives access to both problems and configs.
nbe_sanity_check = BenchmarkSet(
    problems=_load_benchmark_set(),
    configs={
        'nbe': _build_nbe_configs(),
        'nbe_nested': _build_nbe_nested_configs(),
    },
)
