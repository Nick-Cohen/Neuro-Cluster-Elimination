"""Benchmark problem sets for NCE experiments.

Usage:
    from nce.benchmark_problems import neuro_be_sanity_check

    for model in neuro_be_sanity_check:
        print(model.modelfile, model.num_vars)
"""
from .neuro_be_sanity_check import neuro_be_sanity_check
from .catalog_utils import get_catalog
