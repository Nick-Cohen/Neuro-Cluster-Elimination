"""Shared fixtures for config schema tests and inference/training tests.

Reference configs are built from the S01-RESEARCH field inventory (55 live fields
across 6 sections) and the nbe_sanity_check benchmark template (42 fields).

Hand-built problem fixtures provide analytically known partition functions for
exact inference tests, NN training tests, and robustness tests.
"""
import math
import os

import pytest
import torch

from nce.config_schema import prepare_config
from nce.inference.factor import FastFactor


# ---------------------------------------------------------------------------
# Tier selection for the bit-exact regression suite (test_determinism_regression).
#
#   default          -> CPU tier only (seconds; safe for every change)
#   --gpu            -> also the CUDA tier (minutes; for changes touching numerics)
#   --det-algos      -> also the torch.use_deterministic_algorithms(True) probe
#                       (2.5x slower again; requires CUBLAS_WORKSPACE_CONFIG)
#
# Env equivalents: NCE_TEST_GPU=1, NCE_TEST_DET_ALGOS=1.
# ---------------------------------------------------------------------------
def pytest_addoption(parser):
    parser.addoption('--gpu', action='store_true', default=False,
                     help='run the CUDA tier of the determinism regression suite')
    parser.addoption('--det-algos', action='store_true', default=False,
                     help='run the torch.use_deterministic_algorithms(True) probe '
                          '(implies --gpu; needs CUBLAS_WORKSPACE_CONFIG)')


def pytest_configure(config):
    config.addinivalue_line('markers', 'gpu: needs a CUDA device; opt in with --gpu')
    config.addinivalue_line('markers', 'det_algos: enables torch deterministic '
                                       'algorithms in a subprocess; opt in with --det-algos')


def _opt(config, flag, env):
    return bool(config.getoption(flag)) or os.environ.get(env) == '1'


def pytest_collection_modifyitems(config, items):
    want_det = _opt(config, '--det-algos', 'NCE_TEST_DET_ALGOS')
    want_gpu = _opt(config, '--gpu', 'NCE_TEST_GPU') or want_det
    skip_gpu = pytest.mark.skip(reason='CUDA tier not selected (pass --gpu or NCE_TEST_GPU=1)')
    skip_det = pytest.mark.skip(reason='det-algos probe not selected '
                                       '(pass --det-algos or NCE_TEST_DET_ALGOS=1)')
    for item in items:
        if 'det_algos' in item.keywords and not want_det:
            item.add_marker(skip_det)
        elif 'gpu' in item.keywords and not want_gpu:
            item.add_marker(skip_gpu)


# ---------------------------------------------------------------------------
# Reference flat config — mirrors nbe_sanity_check's pedigree13 config exactly
# (all 42 fields that appear in the benchmark, INCLUDING the 2 dead fields).
# ---------------------------------------------------------------------------
@pytest.fixture
def reference_flat_config():
    """Complete flat config dict matching the nbe_sanity_check template (42 fields).

    This is what existing code produces today. Includes dead fields
    (backward_ecl, num_batches_per_set) that the benchmark still carries.
    """
    return {
        # inference
        'device': 'cuda',
        'ecl': 2**19,  # 2^(iB-1) for iB=20
        'iB': 20,
        'approximation_method': 'nn',
        'dope_factors': True,
        # nn
        'hidden_sizes': 'nbe,3',
        'use_linspace_bias': False,
        'use_memorizer': False,
        # training
        'num_epochs': 500,
        'num_epochs2': 0,
        'loss_fn': 'weighted_logspace_mse',
        'optimizer': 'adam',
        'lr': 0.001,
        'lr_decay': 1.0,
        'momentum': 0.9,
        'batch_size': 256,
        'patience': 20,
        'min_lr': 1e-8,
        'seed': 42,
        'skip_early_stopping': False,
        'nbe_early_stopping': False,
        'nbe_warmup_epochs': 0,
        'inverse_time_decay_constant': 100,
        # sampling
        'sampling_scheme': 'uniform',
        'num_samples': 'nbe,0.1',
        'set_size': None,
        'val_set': True,
        'stratify_samples': False,
        'lower_dim': False,
        # backward
        'use_bw_approx': False,
        'populate_bw_factors': False,
        'bw_ecl': None,
        'backward_iB': 20,
        'fdb': False,
        # output
        'display_intermediate': False,
        'track_errors': False,
        'plot_messages': False,
        'debug': False,
        'gather_message_stats': False,
        'traced_losses': [],
        # dead fields (present in benchmark, never read from config)
        'backward_ecl': None,
        'num_batches_per_set': 1,
    }


# ---------------------------------------------------------------------------
# Equivalent nested config — same values as reference_flat_config, organized
# into the 6 nested sections defined in S01-RESEARCH.
# Does NOT include dead fields (nested configs should be clean).
# ---------------------------------------------------------------------------
@pytest.fixture
def equivalent_nested_config():
    """Same values as reference_flat_config expressed in nested section format.

    Uses the readable alias names from D007 where they differ from internal names.
    """
    return {
        'inference': {
            'device': 'cuda',
            'exact_computation_limit': 2**19,
            'i_bound': 20,
            'approximation_method': 'nn',
            'dope_factors': True,
        },
        'nn': {
            'hidden_sizes': 'nbe,3',
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
            'num_samples': 'nbe,0.1',
            'set_size': None,
            'val_set': True,
            'stratify_samples': False,
            'lower_dim': False,
        },
        'backward': {
            'use_backward_approximation': False,
            'populate_backward_factors': False,
            'backward_ecl': None,
            'backward_i_bound': 20,
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
    }


# ---------------------------------------------------------------------------
# Minimal configs — bare minimum for nn mode
# ---------------------------------------------------------------------------
@pytest.fixture
def minimal_flat_config():
    """Bare minimum flat config for approximation_method='nn'."""
    return {
        'approximation_method': 'nn',
        'loss_fn': 'logspace_mse_fdb',
        'num_epochs': 100,
        'num_samples': 1000,
        'iB': 10,
        'ecl': 512,
        'device': 'cpu',
    }


@pytest.fixture
def minimal_nested_config():
    """Bare minimum nested config for approximation_method='nn'."""
    return {
        'inference': {
            'approximation_method': 'nn',
            'i_bound': 10,
            'exact_computation_limit': 512,
            'device': 'cpu',
        },
        'training': {
            'loss_fn': 'logspace_mse_fdb',
            'num_epochs': 100,
        },
        'sampling': {
            'num_samples': 1000,
        },
    }


# ===========================================================================
# NN training config — complete config suitable for CPU-based NN training
# on small hand-built problems.
# ===========================================================================
@pytest.fixture
def nn_training_config():
    """Complete flat config for CPU-based NN training on small problems.

    Derived from reference_flat_config with overrides for fast CPU training:
    device='cpu', num_epochs=50, ecl=4, iB=2, hidden_sizes=[8,8],
    sampling_scheme='all', num_samples=256, dope_factors=False, seed=42.

    Dead fields (backward_ecl, num_batches_per_set) are removed so
    prepare_config(strict=True) accepts it.

    All fields that Trainer, SampleGenerator, and DataPreprocessor access
    via bare dict indexing are included.
    """
    config = {
        # inference
        'device': 'cpu',
        'ecl': 4,
        'iB': 2,
        'approximation_method': 'nn',
        'dope_factors': False,
        # nn
        'hidden_sizes': [8, 8],
        'use_linspace_bias': False,
        'use_memorizer': False,
        # training
        'num_epochs': 50,
        'num_epochs2': 0,
        'loss_fn': 'logspace_mse_fdb',
        'optimizer': 'adam',
        'lr': 0.001,
        'lr_decay': 1.0,
        'momentum': 0.9,
        'batch_size': 256,
        'patience': 20,
        'min_lr': 1e-8,
        'seed': 42,
        'skip_early_stopping': False,
        'nbe_early_stopping': False,
        'nbe_warmup_epochs': 0,
        'inverse_time_decay_constant': 100,
        # sampling
        'sampling_scheme': 'all',
        'num_samples': 256,
        'set_size': None,
        'val_set': True,
        'stratify_samples': False,
        'lower_dim': False,
        # backward
        'use_bw_approx': False,
        'populate_bw_factors': False,
        'bw_ecl': None,
        'backward_iB': 20,
        'fdb': False,
        # output
        'display_intermediate': False,
        'track_errors': False,
        'plot_messages': False,
        'debug': False,
        'gather_message_stats': False,
        'traced_losses': [],
    }
    # Validate: prepare_config should accept this without error
    prepared = prepare_config(config)
    # Return the prepared (validated) config
    return prepared


# ===========================================================================
# NeuroBE training config — raw dict for neurobe-mode NN training.
# NOT passed through prepare_config because neurobe-specific fields
# (normalization_mode, neurobe_early_stopping, etc.) aren't registered yet.
# Built directly to match NEUROBE_DEFAULTS from S01-RESEARCH.
# ===========================================================================
@pytest.fixture
def neurobe_training_config():
    """Raw config dict for NeuroBE-mode CPU training on small problems.

    Based on nn_training_config with neurobe-mode overrides:
    normalization_mode='minmax_01', loss_fn='neurobe_weighted_mse',
    neurobe_early_stopping=True, neurobe_stop_iter=2, activation='relu',
    use_amp=False, hidden_sizes='neurobe,3', sampling_scheme='all',
    lower_dim=True, skip_early_stopping=True, nbe_early_stopping=False.

    Not validated through prepare_config since neurobe fields don't exist
    in the schema yet — will be validated once T04 adds them.
    """
    return {
        # inference
        'device': 'cpu',
        'ecl': 4,
        'iB': 2,
        'approximation_method': 'nn',
        'dope_factors': False,
        # nn — neurobe overrides
        'hidden_sizes': 'neurobe,3',
        'activation': 'relu',
        'use_linspace_bias': False,
        'use_memorizer': False,
        # training — neurobe overrides
        'num_epochs': 50,
        'num_epochs2': 0,
        'loss_fn': 'neurobe_weighted_mse',
        'optimizer': 'adam',
        'lr': 0.001,
        'lr_decay': 1.0,
        'momentum': 0.9,
        'batch_size': 256,
        'patience': 20,
        'min_lr': 1e-8,
        'seed': 42,
        'skip_early_stopping': True,
        'nbe_early_stopping': False,
        'nbe_warmup_epochs': 0,
        'inverse_time_decay_constant': 100,
        'use_amp': False,
        # neurobe-specific early stopping
        'neurobe_early_stopping': True,
        'neurobe_stop_iter': 2,
        # normalization
        'normalization_mode': 'minmax_01',
        # sampling
        'sampling_scheme': 'all',
        'num_samples': 256,
        'set_size': None,
        'val_set': True,
        'stratify_samples': False,
        'lower_dim': True,
        # backward — disabled for neurobe
        'use_bw_approx': False,
        'populate_bw_factors': False,
        'bw_ecl': None,
        'backward_iB': 20,
        'fdb': False,
        # output
        'display_intermediate': False,
        'track_errors': False,
        'plot_messages': False,
        'debug': False,
        'gather_message_stats': False,
        'traced_losses': [],
    }


# ===========================================================================
# Hand-built factor problem fixtures
# ===========================================================================

@pytest.fixture
def binary_chain_factors():
    """Two binary variables with a uniform pairwise factor.

    Topology: X0 — X1 (both domain 2)
    Factor: f(X0, X1) = [[0.5, 0.5], [0.5, 0.5]] (uniform)
    Analytic Z = sum of all entries = 4 * 0.5 = 2.0

    Returns dict with:
        factors: list of FastFactor in log10 space
        elim_order: elimination order [0, 1]
        expected_log10_z: math.log10(2.0)
    """
    probs = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    factor = FastFactor(torch.log10(probs), [0, 1])
    return {
        'factors': [factor],
        'elim_order': [0, 1],
        'expected_log10_z': math.log10(2.0),
    }


@pytest.fixture
def ternary_chain_factors():
    """Two ternary variables with a uniform pairwise factor.

    Topology: X0 — X1 (both domain 3)
    Factor: f(X0, X1) = all entries 0.5 (3x3 uniform)
    Analytic Z = 9 * 0.5 = 4.5

    Returns dict with:
        factors: list of FastFactor in log10 space
        elim_order: elimination order [0, 1]
        expected_log10_z: math.log10(4.5)
    """
    probs = torch.tensor([[0.5, 0.5, 0.5],
                          [0.5, 0.5, 0.5],
                          [0.5, 0.5, 0.5]])
    factor = FastFactor(torch.log10(probs), [0, 1])
    return {
        'factors': [factor],
        'elim_order': [0, 1],
        'expected_log10_z': math.log10(4.5),
    }


@pytest.fixture
def star_graph_factors():
    """Star graph with hub X0 connected to X1, X2, X3 (all domain 2).

    Topology:
        X0 -- X1  (pairwise f01)
        X0 -- X2  (pairwise f02)
        X0 -- X3  (pairwise f03)
        (X0, X1, X2)  (3-way factor f012)

    The 3-way factor ensures that when X0 is eliminated first,
    the resulting message has scope {X1, X2, X3} with message_size = 8,
    exceeding ecl=4 and triggering the NN training path.

    Analytic Z is computed by enumerating all 2^4 = 16 assignments:
        Z = sum over (x0,x1,x2,x3) of f01*f02*f03*f012

    Returns dict with:
        factors: list of FastFactor in log10 space
        elim_order: [0, 1, 2, 3]
        expected_log10_z: math.log10(Z)
    """
    import itertools

    # Pairwise factors (linear-space probabilities)
    f01_probs = torch.tensor([[0.8, 0.2], [0.2, 0.8]])
    f02_probs = torch.tensor([[0.6, 0.4], [0.4, 0.6]])
    f03_probs = torch.tensor([[0.7, 0.3], [0.3, 0.7]])

    # 3-way factor on (X0, X1, X2): shape [2, 2, 2]
    f012_probs = torch.tensor([
        [[0.9, 0.1], [0.3, 0.7]],   # X0=0
        [[0.2, 0.8], [0.6, 0.4]],   # X0=1
    ])

    # Compute analytic Z by enumerating all 16 assignments
    Z = 0.0
    for x0, x1, x2, x3 in itertools.product(range(2), repeat=4):
        val = (f01_probs[x0, x1] * f02_probs[x0, x2]
               * f03_probs[x0, x3] * f012_probs[x0, x1, x2])
        Z += val.item()

    # Build FastFactor list in log10 space
    factors = [
        FastFactor(torch.log10(f01_probs), [0, 1]),
        FastFactor(torch.log10(f02_probs), [0, 2]),
        FastFactor(torch.log10(f03_probs), [0, 3]),
        FastFactor(torch.log10(f012_probs), [0, 1, 2]),
    ]

    return {
        'factors': factors,
        'elim_order': [0, 1, 2, 3],
        'expected_log10_z': math.log10(Z),
    }
