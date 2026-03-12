"""Shared fixtures for config schema tests.

Reference configs are built from the S01-RESEARCH field inventory (55 live fields
across 6 sections) and the nbe_sanity_check benchmark template (42 fields).
"""
import pytest


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
