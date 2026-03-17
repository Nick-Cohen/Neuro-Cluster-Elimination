"""Benchmark problem sets for NCE experiments.

Requires additional dependencies (requests) for pyGMs model catalog access.
Install with: pip install requests

Usage:
    from nce.benchmark_problems import nbe_sanity_check

    for model, config in zip(nbe_sanity_check.problems, nbe_sanity_check.configs['nbe']):
        print(model.modelfile, config)
"""
try:
    from .nbe_sanity_check import nbe_sanity_check
    from .nbe_sanity_check import BenchmarkSet
    from .small_problems import small_problems, set_bw_ecl
    from .neurobe_binary import neurobe_binary
    from .catalog_utils import get_catalog
except ImportError as e:
    import warnings
    warnings.warn(
        f"benchmark_problems not fully available: {e}. "
        "Install missing dependencies (e.g. 'pip install requests') to use benchmark problem sets."
    )
    nbe_sanity_check = None
    BenchmarkSet = None
    small_problems = None
    set_bw_ecl = None
    neurobe_binary = None
    get_catalog = None
