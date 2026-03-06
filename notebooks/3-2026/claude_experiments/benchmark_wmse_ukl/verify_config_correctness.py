#!/usr/bin/env python3
"""Verify that the 5 experiment configurations produce correct nn_config dicts.

For each configuration, builds the config dict and checks key fields match
the experiment design specification.

Run: /home/cohenn1/NCE/venv/bin/python verify_config_correctness.py
"""
import sys
sys.path.insert(0, '/home/cohenn1/NCE')

import copy
from nce.benchmark_problems.small_problems import small_problems, _AUTO_ECL


def build_experiment_config(base_cfg, loss_fn, bw_ecl, num_epochs=5000):
    """Build an experiment config from base small_problems config.

    Args:
        base_cfg: Base config dict from small_problems.configs['default']
        loss_fn: Loss function name string
        bw_ecl: Backward ECL value (0 for no backward info)
        num_epochs: Number of training epochs

    Returns:
        Complete nn_config dict ready for FastGM
    """
    cfg = copy.deepcopy(base_cfg)
    cfg['loss_fn'] = loss_fn
    cfg['num_epochs'] = num_epochs
    cfg['skip_early_stopping'] = True
    cfg['nbe_early_stopping'] = False
    cfg['batch_size'] = 10000000  # Very large to ensure single batch

    cfg['bw_ecl'] = bw_ecl
    cfg['backward_ecl'] = bw_ecl
    cfg['populate_bw_factors'] = bw_ecl > 0
    cfg['use_bw_approx'] = bw_ecl > 0

    return cfg


def verify_config(config_name, cfg, expected):
    """Verify a config dict against expected values.

    Args:
        config_name: Human-readable config name
        cfg: The config dict to verify
        expected: Dict of field -> expected_value

    Returns:
        True if all checks pass, False otherwise
    """
    passed = True
    for field, expected_val in expected.items():
        actual_val = cfg.get(field)
        if actual_val != expected_val:
            print(f"  FAIL: {field} = {actual_val!r}, expected {expected_val!r}")
            passed = False
    return passed


def main():
    base_configs = small_problems.configs['default']
    problems = small_problems.problems
    all_passed = True
    total_checks = 0

    # Use BN_1 (index 9) as representative problem
    bn1_idx = 9  # bn/BN_1 is at index 9 in _MODELS
    base_cfg = base_configs[bn1_idx]
    bn1_ecl = base_cfg['ecl']  # Should be 524287

    print(f"Representative problem: bn/BN_1 (index {bn1_idx}, ecl={bn1_ecl})")
    print(f"Total problems: {len(problems)}")
    print()

    # Common expected values (all configs)
    common_expected = {
        'num_epochs': 5000,
        'skip_early_stopping': True,
        'nbe_early_stopping': False,
        'sampling_scheme': 'all',
        'batch_size': 10000000,
        'hidden_sizes': [3, 3],
        'iB': 100,
        'seed': 42,
        'dope_factors': False,
    }

    # ---- Configuration 1: WMSE (no backward info) ----
    print("--- Config 1: WMSE (no backward info) ---")
    cfg1 = build_experiment_config(base_cfg, 'weighted_logspace_mse', 0)
    config1_expected = {
        **common_expected,
        'loss_fn': 'weighted_logspace_mse',
        'bw_ecl': 0,
        'use_bw_approx': False,
        'populate_bw_factors': False,
        'backward_ecl': 0,
        'ecl': bn1_ecl,
    }
    ok = verify_config('Config 1', cfg1, config1_expected)
    total_checks += len(config1_expected)
    if ok:
        print(f"  PASS: All {len(config1_expected)} fields correct")
    else:
        all_passed = False
    print()

    # ---- Configuration 2: UKL (no backward info) ----
    print("--- Config 2: UKL (no backward info) ---")
    cfg2 = build_experiment_config(base_cfg, 'unnormalized_kl', 0)
    config2_expected = {
        **common_expected,
        'loss_fn': 'unnormalized_kl',
        'bw_ecl': 0,
        'use_bw_approx': False,
        'populate_bw_factors': False,
        'backward_ecl': 0,
    }
    ok = verify_config('Config 2', cfg2, config2_expected)
    total_checks += len(config2_expected)
    if ok:
        print(f"  PASS: All {len(config2_expected)} fields correct")
    else:
        all_passed = False
    print()

    # ---- Configuration 3: UKL + bw_ecl = 8 ----
    print("--- Config 3: UKL + bw_ecl = 8 ---")
    cfg3 = build_experiment_config(base_cfg, 'unnormalized_kl', 8)
    config3_expected = {
        **common_expected,
        'loss_fn': 'unnormalized_kl',
        'bw_ecl': 8,
        'use_bw_approx': True,
        'populate_bw_factors': True,
        'backward_ecl': 8,
    }
    ok = verify_config('Config 3', cfg3, config3_expected)
    total_checks += len(config3_expected)
    if ok:
        print(f"  PASS: All {len(config3_expected)} fields correct")
    else:
        all_passed = False
    print()

    # ---- Configuration 4: UKL + bw_ecl = auto_ecl (per-problem) ----
    print("--- Config 4: UKL + bw_ecl = auto_ecl (per-problem) ---")
    # Check a few representative problems
    test_problems = [
        (9, 'BN_1.uai', 524287),
        (1, 'BN_3.uai', 16383),
        (14, 'grid10x10.f5.wrap.uai', 1048575),
    ]
    config4_ok = True
    for idx, modelfile, expected_ecl in test_problems:
        cfg4 = build_experiment_config(base_configs[idx], 'unnormalized_kl', base_configs[idx]['ecl'])
        config4_expected = {
            'loss_fn': 'unnormalized_kl',
            'bw_ecl': expected_ecl,
            'use_bw_approx': True,
            'populate_bw_factors': True,
            'backward_ecl': expected_ecl,
            'ecl': expected_ecl,
        }
        ok = verify_config(f'Config 4 ({modelfile})', cfg4, config4_expected)
        total_checks += len(config4_expected)
        if ok:
            print(f"  PASS: {modelfile} bw_ecl={expected_ecl}")
        else:
            config4_ok = False
            all_passed = False

    # Verify that all 24 problems have their auto_ecl correctly mapped
    for i, (model, cfg) in enumerate(zip(problems, base_configs)):
        expected_ecl = _AUTO_ECL[model.modelfile]
        if cfg['ecl'] != expected_ecl:
            print(f"  FAIL: Problem {i} ({model.modelfile}) ecl={cfg['ecl']}, expected {expected_ecl}")
            config4_ok = False
            all_passed = False
    total_checks += 24

    if config4_ok:
        print(f"  PASS: All per-problem ecl values match _AUTO_ECL (24 problems verified)")
    print()

    # ---- Configuration 5: UKL + bw_ecl = 2^30 ----
    print("--- Config 5: UKL + bw_ecl = 2^30 ---")
    cfg5 = build_experiment_config(base_cfg, 'unnormalized_kl', 2**30)
    config5_expected = {
        **common_expected,
        'loss_fn': 'unnormalized_kl',
        'bw_ecl': 1073741824,
        'use_bw_approx': True,
        'populate_bw_factors': True,
        'backward_ecl': 1073741824,
    }
    ok = verify_config('Config 5', cfg5, config5_expected)
    total_checks += len(config5_expected)
    if ok:
        print(f"  PASS: All {len(config5_expected)} fields correct")
    else:
        all_passed = False
    print()

    # ---- Summary ----
    print("=" * 60)
    if all_passed:
        print(f"PASS: All {total_checks} checks passed across 5 configurations")
    else:
        print(f"FAIL: Some checks failed (see above)")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
