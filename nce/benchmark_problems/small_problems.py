"""small_problems benchmark set.

Contains 24 unique models from the probs12_4 iB10 and iB15 problem sets.
Duplicates across sets have been removed. Configs use the template from
full_algorithm_12_4_bwecl.py with ecl values set to auto_ecl from
problem_ecl_values.csv.

Config sets:
- 'default': nn model (hidden_sizes=[3,3]) with auto_ecl values

Usage:
    from nce.benchmark_problems import small_problems

    for model, config in zip(small_problems.problems, small_problems.configs['default']):
        print(model.modelfile, config['ecl'])

    # Bulk-update bw_ecl for all problems:
    set_bw_ecl(small_problems, 'default', 1024)
"""
from .catalog_utils import get_catalog
from .nbe_sanity_check import BenchmarkSet


# Unique models from iB10 + iB15 sets (duplicates removed).
# Format: (catalogue_key, source_iB)
_MODELS = [
    # iB10-only problems
    ('alchemy/smokers_20', 10),
    ('bn/BN_3', 10),
    ('bn/BN_5', 10),
    ('bn/BN_7', 10),
    ('bn/BN_10', 10),
    ('bn/BN_11', 10),
    ('segmentation/10_14_s.binary', 10),
    ('segmentation/10_16_s.binary', 10),
    ('segmentation/11_4_s.binary', 10),
    # Problems in both iB10 and iB15 (listed once, using iB10)
    ('bn/BN_1', 10),
    ('promedas/or_chain_10.fg', 10),
    ('segmentation/11_17_s.binary', 10),
    ('objdetect/deer_rescaled_0034.K15.F1.5.model', 10),
    ('objdetect/deer_rescaled_0294.K10.F1.75.model', 10),
    ('grids/grid10x10.f5.wrap', 10),
    # iB15-only problems
    ('bn/BN_2', 15),
    ('bn/BN_8', 15),
    ('bn/BN_9', 15),
    ('objdetect/deer_rescaled_0034.K10.F2.model', 15),
    ('objdetect/deer_rescaled_0034.K15.F1.75.model', 15),
    ('objdetect/deer_rescaled_0034.K20.F1.25.model', 15),
    ('objdetect/deer_rescaled_0034.K20.F1.5.model', 15),
    ('csp/29.wcsp', 15),
    ('csp/404.wcsp', 15),
]

_MODEL_KEYS = [key for key, _ in _MODELS]
_SOURCE_IB = {key: ib for key, ib in _MODELS}

# auto_ecl values from problem_ecl_values.csv (keyed by modelfile).
# For problems in both iB10/iB15, values are identical; we use whichever appears.
_AUTO_ECL = {
    'smokers_20.uai': 262143,
    'BN_1.uai': 524287,
    'BN_2.uai': 2097151,
    'BN_3.uai': 16383,
    'BN_5.uai': 16383,
    'BN_7.uai': 131071,
    'BN_8.uai': 8388607,
    'BN_9.uai': 4194303,
    'BN_10.uai': 32767,
    'BN_11.uai': 131071,
    'or_chain_10.fg.uai': 262143,
    '10_14_s.binary.uai': 32767,
    '10_16_s.binary.uai': 65535,
    '11_17_s.binary.uai': 262143,
    '11_4_s.binary.uai': 131071,
    'deer_rescaled_0034.K15.F1.5.model.uai': 1048575,
    'deer_rescaled_0294.K10.F1.75.model.uai': 1771560,
    'grid10x10.f5.wrap.uai': 1048575,
    'deer_rescaled_0034.K10.F2.model.uai': 19487170,
    'deer_rescaled_0034.K15.F1.75.model.uai': 16777215,
    'deer_rescaled_0034.K20.F1.25.model.uai': 4084100,
    'deer_rescaled_0034.K20.F1.5.model.uai': 4084100,
    '29.wcsp.uai': 2097151,
    '404.wcsp.uai': 4194303,
}


def _load_benchmark_set():
    """Load the small_problems benchmark models from the catalogue."""
    catalog = get_catalog()
    return [catalog[key] for key in _MODEL_KEYS]


def _get_auto_ecl(modelfile):
    """Look up auto_ecl for a model by its modelfile name."""
    return _AUTO_ECL[modelfile]


def _build_default_configs():
    """Build config dicts using the full_algorithm_12_4_bwecl.py template.

    Uses nn model type (hidden_sizes=[3,3]) with per-problem auto_ecl values.
    bw_ecl defaults to 0 (no backward info). Use set_bw_ecl() to change.
    """
    catalog = get_catalog()
    configs = []
    for key in _MODEL_KEYS:
        model = catalog[key]
        auto_ecl = _get_auto_ecl(model.modelfile)
        configs.append({
            'device': 'cuda',
            'hidden_sizes': [3, 3],
            'optimizer': 'adam',
            'lr': 0.01,
            'lr_decay': 1,
            'momentum': 0.9,
            'inverse_time_decay_constant': 10,
            'patience': 1,
            'min_lr': 1e-8,
            'num_epochs': 10000,
            'num_epochs2': 0,
            'nbe_early_stopping': False,
            'skip_early_stopping': True,
            'sampling_scheme': 'all',
            'batch_size': 100000,
            'set_size': 100000,
            'num_samples': 100000,
            'loss_fn': 'unnormalized_kl',
            'traced_losses': [],
            'val_set': 'all',
            'fdb': False,
            'use_bw_approx': True,
            'populate_bw_factors': False,
            'ecl': auto_ecl,
            'iB': 100,
            'approximation_method': 'nn',
            'bw_ecl': 0,
            'backward_iB': 100,
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


def _build_default_nested_configs():
    """Build nested-format config dicts for each model.

    Returns a list of 24 nested config dicts (one per model, same order as
    _MODEL_KEYS). Uses the 6-section schema from config_schema.NESTED_SECTIONS
    with readable alias names where available.

    These configs produce identical output to _build_default_configs() when
    passed through prepare_config().
    """
    catalog = get_catalog()
    configs = []
    for key in _MODEL_KEYS:
        model = catalog[key]
        auto_ecl = _get_auto_ecl(model.modelfile)
        configs.append({
            'inference': {
                'device': 'cuda',
                'exact_computation_limit': auto_ecl,
                'i_bound': 100,
                'approximation_method': 'nn',
                'dope_factors': False,
            },
            'nn': {
                'hidden_sizes': [3, 3],
                'use_linspace_bias': False,
                'use_memorizer': False,
            },
            'training': {
                'num_epochs': 10000,
                'num_epochs_phase2': 0,
                'loss_fn': 'unnormalized_kl',
                'optimizer': 'adam',
                'learning_rate': 0.01,
                'learning_rate_decay': 1,
                'momentum': 0.9,
                'batch_size': 100000,
                'patience': 1,
                'min_learning_rate': 1e-8,
                'seed': 42,
                'skip_early_stopping': True,
                'nbe_early_stopping': False,
                'inverse_time_decay_constant': 10,
            },
            'sampling': {
                'sampling_scheme': 'all',
                'num_samples': 100000,
                'set_size': 100000,
                'val_set': 'all',
                'stratify_samples': False,
                'lower_dim': False,
            },
            'backward': {
                'use_backward_approximation': True,
                'populate_backward_factors': False,
                'backward_ecl': 0,
                'backward_i_bound': 100,
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


def set_bw_ecl(benchmark_set, config_name, value):
    """Set bw_ecl for all problems in a config set.

    Also updates populate_bw_factors to stay consistent.

    Args:
        benchmark_set: BenchmarkSet instance (e.g. small_problems)
        config_name: Config set name (e.g. 'default')
        value: New bw_ecl value. Use 0 for no backward info,
               or a power of 2 (e.g. 1024, 2**20, 2**30) for backward info.
    """
    for cfg in benchmark_set.configs[config_name]:
        cfg['bw_ecl'] = value
        cfg['populate_bw_factors'] = value > 0


# Module-level instance
small_problems = BenchmarkSet(
    problems=_load_benchmark_set(),
    configs={
        'default': _build_default_configs(),
        'default_nested': _build_default_nested_configs(),
    },
)
