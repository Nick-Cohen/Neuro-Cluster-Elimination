"""Benchmark problem sets for NCE experiments.

Usage:
    from nce.benchmark_problems import nbe_sanity_check

    for model, config in zip(nbe_sanity_check.problems, nbe_sanity_check.configs['nbe']):
        print(model.modelfile, config)
"""
from .nbe_sanity_check import nbe_sanity_check
from .nbe_sanity_check import BenchmarkSet
from .small_problems import small_problems, set_bw_ecl
from .catalog_utils import get_catalog
