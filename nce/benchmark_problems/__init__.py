"""Benchmark problem sets for NCE experiments.

Usage:
    from nce.benchmark_problems import neuro_be_sanity_check, neuro_be_sanity_check_configs

    for model in neuro_be_sanity_check:
        config = neuro_be_sanity_check_configs.get(model.modelfile)
        print(model.modelfile, config)

    # Or use the list form for easy zipping:
    from nce.benchmark_problems import neuro_be_sanity_check_configs_list

    for model, config in zip(neuro_be_sanity_check, neuro_be_sanity_check_configs_list):
        print(model.modelfile, config)
"""
from .neuro_be_sanity_check import neuro_be_sanity_check
from .neuro_be_sanity_check import neuro_be_sanity_check_configs
from .neuro_be_sanity_check import neuro_be_sanity_check_configs_list
from .catalog_utils import get_catalog
