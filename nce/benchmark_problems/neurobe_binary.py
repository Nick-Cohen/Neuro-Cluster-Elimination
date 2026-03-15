"""neurobe_binary benchmark set.

Contains 15 binary-domain models for NeuroBE comparison experiments.
Each model uses neurobe_mode with per-problem ecl values computed as
``ecl = 2^width_problem - 1`` from the NeuroBE results CSV, ensuring
the NN dispatch condition matches NeuroBE's C++ implementation.

All models use iB=25, hidden_sizes='neurobe,3', num_samples='nbe,0.1',
and the full NEUROBE_DEFAULTS preset via neurobe_mode=True.

Expected NeuroBE NN counts (from binary_domain_results.csv):
    BN_1=2, BN_2=3, BN_3=1, BN_5=1, BN_7=1, BN_8=4, BN_9=1,
    BN_10=2, BN_11=1, grid10x10.f5.wrap=1, smokers_20=1,
    10_14_s.binary=3, 10_16_s.binary=2, 11_17_s.binary=1, 11_4_s.binary=1

Usage:
    from nce.benchmark_problems import neurobe_binary

    for model, config in zip(neurobe_binary.problems, neurobe_binary.configs['neurobe']):
        print(model.modelfile, config['ecl'])
"""
from .catalog_utils import get_catalog
from .nbe_sanity_check import BenchmarkSet


# Catalogue keys — 15 binary-domain models (single source of truth).
_MODEL_KEYS = [
    'bn/BN_1',
    'bn/BN_2',
    'bn/BN_3',
    'bn/BN_5',
    'bn/BN_7',
    'bn/BN_8',
    'bn/BN_9',
    'bn/BN_10',
    'bn/BN_11',
    'grids/grid10x10.f5.wrap',
    'alchemy/smokers_20',
    'segmentation/10_14_s.binary',
    'segmentation/10_16_s.binary',
    'segmentation/11_17_s.binary',
    'segmentation/11_4_s.binary',
]

# Per-problem ecl values: ecl = 2^width_problem - 1
# width_problem sourced from Clean-NeuroBE/results/binary_min_nn/binary_domain_results.csv
_NEUROBE_ECL = {
    'bn/BN_1':                      2**19 - 1,   # wp=19, ecl=524287
    'bn/BN_2':                      2**21 - 1,   # wp=21, ecl=2097151
    'bn/BN_3':                      2**15 - 1,   # wp=15, ecl=32767
    'bn/BN_5':                      2**15 - 1,   # wp=15, ecl=32767
    'bn/BN_7':                      2**18 - 1,   # wp=18, ecl=262143
    'bn/BN_8':                      2**23 - 1,   # wp=23, ecl=8388607
    'bn/BN_9':                      2**23 - 1,   # wp=23, ecl=8388607
    'bn/BN_10':                     2**15 - 1,   # wp=15, ecl=32767
    'bn/BN_11':                     2**18 - 1,   # wp=18, ecl=262143
    'grids/grid10x10.f5.wrap':      2**21 - 1,   # wp=21, ecl=2097151
    'alchemy/smokers_20':           2**19 - 1,   # wp=19, ecl=524287
    'segmentation/10_14_s.binary':  2**15 - 1,   # wp=15, ecl=32767
    'segmentation/10_16_s.binary':  2**16 - 1,   # wp=16, ecl=65535
    'segmentation/11_17_s.binary':  2**19 - 1,   # wp=19, ecl=524287
    'segmentation/11_4_s.binary':   2**18 - 1,   # wp=18, ecl=262143
}

# Expected NeuroBE NN counts (for verification).
NEUROBE_NN_COUNTS = {
    'bn/BN_1':                      2,
    'bn/BN_2':                      3,
    'bn/BN_3':                      1,
    'bn/BN_5':                      1,
    'bn/BN_7':                      1,
    'bn/BN_8':                      4,
    'bn/BN_9':                      1,
    'bn/BN_10':                     2,
    'bn/BN_11':                     1,
    'grids/grid10x10.f5.wrap':      1,
    'alchemy/smokers_20':           1,
    'segmentation/10_14_s.binary':  3,
    'segmentation/10_16_s.binary':  2,
    'segmentation/11_17_s.binary':  1,
    'segmentation/11_4_s.binary':   1,
}


def _load_benchmark_set():
    """Load the 15 binary-domain benchmark models from the catalogue."""
    catalog = get_catalog()
    return [catalog[key] for key in _MODEL_KEYS]


def _build_neurobe_configs():
    """Build neurobe_mode config dicts for each of the 15 binary-domain models.

    Returns a list of config dicts (one per model, same order as _MODEL_KEYS).
    Each dict uses neurobe_mode=True so NEUROBE_DEFAULTS are applied by
    prepare_config(), with per-problem ecl overrides from NeuroBE CSV.
    """
    configs = []
    for key in _MODEL_KEYS:
        configs.append({
            'neurobe_mode': True,
            'ecl': _NEUROBE_ECL[key],
            'num_samples': 'nbe,0.1',
            'val_set': True,
            'dope_factors': True,
            'device': 'cuda',
            'seed': 42,
        })
    return configs


# Module-level instance: single import gives access to both problems and configs.
neurobe_binary = BenchmarkSet(
    problems=_load_benchmark_set(),
    configs={
        'neurobe': _build_neurobe_configs(),
    },
)
