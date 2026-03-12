"""Acceptance tests for nce.config_schema (S01: Config Schema & Flat Translation).

Each test function encodes a requirement from S01's acceptance criteria.
Tests import from nce.config_schema which T02 will create — until then,
all tests fail with ImportError, confirming the harness works and the
stopping condition is clear.

Requirements covered:
  R001 — Nested→flat translation with correct internal keys
  R002 — Dead fields raise ValueError with descriptive message
  R003 — Field aliases resolve to internal names
  R004 — Unknown fields / missing required fields raise ValueError
  R005 — Flat config passthrough / auto-detection
  R006 — prepare_config returns plain mutable dict
"""
import pytest

from nce.config_schema import prepare_config


# ===================================================================
# R001: Nested config → flat translation
# ===================================================================


class TestNestedToFlatTranslation:
    """R001: Nested config correctly translates to internal flat keys."""

    def test_nested_to_flat_translation(self, equivalent_nested_config):
        """Nested config flattens to correct internal key names."""
        result = prepare_config(equivalent_nested_config)

        # Check that readable names were translated to internal names
        assert 'ecl' in result, "exact_computation_limit should map to 'ecl'"
        assert 'iB' in result, "i_bound should map to 'iB'"
        assert 'lr' in result, "learning_rate should map to 'lr'"
        assert 'fdb' in result, "forward_diff_barrier should map to 'fdb'"
        assert 'bw_ecl' in result, "backward_ecl should map to 'bw_ecl'"
        assert 'backward_iB' in result, "backward_i_bound should map to 'backward_iB'"
        assert 'use_bw_approx' in result, "use_backward_approximation should map to 'use_bw_approx'"
        assert 'populate_bw_factors' in result, "populate_backward_factors should map to 'populate_bw_factors'"
        assert 'min_lr' in result, "min_learning_rate should map to 'min_lr'"
        assert 'lr_decay' in result, "learning_rate_decay should map to 'lr_decay'"
        assert 'num_epochs2' in result, "num_epochs_phase2 should map to 'num_epochs2'"

        # Readable/nested names should NOT appear in the flat output
        assert 'exact_computation_limit' not in result
        assert 'i_bound' not in result
        assert 'learning_rate' not in result
        assert 'forward_diff_barrier' not in result
        assert 'use_backward_approximation' not in result
        assert 'min_learning_rate' not in result

        # Section keys should not appear in the flat output
        for section in ('inference', 'nn', 'training', 'sampling', 'backward', 'output'):
            assert section not in result, f"Section key '{section}' should not be in flat output"

    def test_nested_to_flat_values_preserved(self, equivalent_nested_config):
        """Nested config values are correctly propagated to flat output."""
        result = prepare_config(equivalent_nested_config)

        assert result['ecl'] == 2**19
        assert result['iB'] == 20
        assert result['lr'] == 0.001
        assert result['num_epochs'] == 500
        assert result['loss_fn'] == 'weighted_logspace_mse'
        assert result['hidden_sizes'] == 'nbe,3'
        assert result['device'] == 'cuda'
        assert result['fdb'] is False
        assert result['bw_ecl'] is None
        assert result['backward_iB'] == 20

    def test_nested_and_flat_produce_same_result(
        self, reference_flat_config, equivalent_nested_config
    ):
        """R001 + R005: Equivalent nested and flat configs produce identical flat output.

        The flat config has dead fields stripped; the nested config never had them.
        After prepare_config, both should produce the same dict.
        """
        flat_result = prepare_config(reference_flat_config)
        nested_result = prepare_config(equivalent_nested_config)

        # Both results should have the same keys and values
        # (flat result will have dead fields stripped, matching nested result)
        assert flat_result == nested_result

    def test_polymorphic_types(self, minimal_nested_config):
        """R001: Polymorphic field types are accepted without error.

        hidden_sizes: list, str, 'bias_only'
        batch_size: int, 'all'
        num_samples: int, str
        val_set: bool, 'all', None
        """
        # hidden_sizes as list
        config = {**minimal_nested_config}
        config['nn'] = {'hidden_sizes': [64, 64]}
        result = prepare_config(config)
        assert result['hidden_sizes'] == [64, 64]

        # hidden_sizes as string
        config['nn'] = {'hidden_sizes': 'nbe,3'}
        result = prepare_config(config)
        assert result['hidden_sizes'] == 'nbe,3'

        # hidden_sizes as 'bias_only'
        config['nn'] = {'hidden_sizes': 'bias_only'}
        result = prepare_config(config)
        assert result['hidden_sizes'] == 'bias_only'

        # batch_size as int
        config_flat = {
            'approximation_method': 'nn',
            'loss_fn': 'logspace_mse_fdb',
            'num_epochs': 10,
            'num_samples': 100,
            'batch_size': 128,
        }
        result = prepare_config(config_flat)
        assert result['batch_size'] == 128

        # batch_size as 'all'
        config_flat['batch_size'] = 'all'
        result = prepare_config(config_flat)
        assert result['batch_size'] == 'all'

        # num_samples as int
        config_flat['num_samples'] = 5000
        result = prepare_config(config_flat)
        assert result['num_samples'] == 5000

        # num_samples as string
        config_flat['num_samples'] = 'nbe,0.35'
        result = prepare_config(config_flat)
        assert result['num_samples'] == 'nbe,0.35'

        # val_set as bool, 'all', None
        for val in [True, False, 'all', None]:
            config_flat['val_set'] = val
            result = prepare_config(config_flat)
            assert result['val_set'] == val


# ===================================================================
# R002: Dead fields raise ValueError
# ===================================================================


class TestDeadFields:
    """R002: Dead fields produce clear errors."""

    def test_dead_field_error_backward_ecl(self):
        """backward_ecl (as a config key) raises ValueError."""
        config = {
            'approximation_method': 'nn',
            'loss_fn': 'logspace_mse_fdb',
            'num_epochs': 10,
            'num_samples': 100,
            'backward_ecl': 512,  # dead field — code reads bw_ecl
        }
        with pytest.raises(ValueError, match='backward_ecl'):
            prepare_config(config)

    def test_dead_field_error_num_batches_per_set(self):
        """num_batches_per_set raises ValueError."""
        config = {
            'approximation_method': 'nn',
            'loss_fn': 'logspace_mse_fdb',
            'num_epochs': 10,
            'num_samples': 100,
            'num_batches_per_set': 2,  # dead field — computed locally
        }
        with pytest.raises(ValueError, match='num_batches_per_set'):
            prepare_config(config)

    def test_dead_field_error_message_suggests_alternative(self):
        """Error message names the dead field and suggests the correct alternative."""
        config = {
            'approximation_method': 'nn',
            'loss_fn': 'logspace_mse_fdb',
            'num_epochs': 10,
            'num_samples': 100,
            'backward_ecl': 512,
        }
        with pytest.raises(ValueError) as exc_info:
            prepare_config(config)

        msg = str(exc_info.value)
        assert 'backward_ecl' in msg, "Error should name the dead field"
        assert 'bw_ecl' in msg, "Error should suggest the correct alternative"

    def test_dead_field_in_nested_config_errors(self):
        """Dead field inside a nested section also raises ValueError."""
        config = {
            'inference': {
                'approximation_method': 'nn',
            },
            'training': {
                'loss_fn': 'logspace_mse_fdb',
                'num_epochs': 10,
                'num_batches_per_set': 2,  # dead field in nested section
            },
            'sampling': {
                'num_samples': 100,
            },
        }
        with pytest.raises(ValueError, match='num_batches_per_set'):
            prepare_config(config)


# ===================================================================
# R003: Field aliases resolve to internal names
# ===================================================================


class TestAliasResolution:
    """R003: Readable alias names resolve to internal (old) names."""

    def test_alias_resolution_flat(self):
        """Aliases in flat config resolve to internal names."""
        config = {
            'approximation_method': 'nn',
            'learning_rate': 0.01,          # alias for lr
            'exact_computation_limit': 256,  # alias for ecl
            'i_bound': 10,                   # alias for iB
            'forward_diff_barrier': True,    # alias for fdb
            'loss_fn': 'logspace_mse_fdb',
            'num_epochs': 10,
            'num_samples': 100,
        }
        result = prepare_config(config)

        assert result['lr'] == 0.01
        assert result['ecl'] == 256
        assert result['iB'] == 10
        assert result['fdb'] is True

        # Alias names should NOT remain in the output
        assert 'learning_rate' not in result
        assert 'exact_computation_limit' not in result
        assert 'i_bound' not in result
        assert 'forward_diff_barrier' not in result

    def test_alias_resolution_nested(self):
        """Aliases in nested config sections resolve to internal names."""
        config = {
            'inference': {
                'exact_computation_limit': 256,
                'i_bound': 10,
            },
            'training': {
                'learning_rate': 0.01,
                'min_learning_rate': 1e-6,
                'learning_rate_decay': 0.99,
                'loss_fn': 'logspace_mse_fdb',
                'num_epochs': 10,
                'num_epochs_phase2': 50,
            },
            'backward': {
                'forward_diff_barrier': True,
                'use_backward_approximation': True,
                'backward_i_bound': 15,
            },
            'sampling': {
                'num_samples': 100,
            },
        }
        result = prepare_config(config)

        assert result['lr'] == 0.01
        assert result['min_lr'] == 1e-6
        assert result['lr_decay'] == 0.99
        assert result['ecl'] == 256
        assert result['iB'] == 10
        assert result['fdb'] is True
        assert result['use_bw_approx'] is True
        assert result['backward_iB'] == 15
        assert result['num_epochs2'] == 50
        assert result['populate_bw_factors'] not in result or 'populate_bw_factors' in result

    def test_internal_name_also_accepted(self):
        """Internal names (lr, ecl, iB, fdb) work directly in flat config."""
        config = {
            'lr': 0.01,
            'ecl': 256,
            'iB': 10,
            'fdb': True,
            'loss_fn': 'logspace_mse_fdb',
            'num_epochs': 10,
            'num_samples': 100,
        }
        result = prepare_config(config)

        assert result['lr'] == 0.01
        assert result['ecl'] == 256
        assert result['iB'] == 10
        assert result['fdb'] is True


# ===================================================================
# R004: Unknown fields / missing required fields raise ValueError
# ===================================================================


class TestValidationErrors:
    """R004: Invalid configs produce clear, section-aware errors."""

    def test_unknown_nested_field_error(self):
        """Unknown field in a nested section raises ValueError naming the section."""
        config = {
            'training': {
                'loss_fn': 'logspace_mse_fdb',
                'num_epochs': 10,
                'totally_fake_field': 42,  # unknown
            },
            'sampling': {
                'num_samples': 100,
            },
        }
        with pytest.raises(ValueError) as exc_info:
            prepare_config(config)

        msg = str(exc_info.value)
        assert 'totally_fake_field' in msg, "Error should name the unknown field"
        assert 'training' in msg, "Error should name the section"

    def test_unknown_section_error(self):
        """Unknown top-level section in nested config raises ValueError."""
        config = {
            'inference': {
                'device': 'cpu',
            },
            'quantum_computing': {  # not a valid section
                'qubits': 42,
            },
        }
        with pytest.raises(ValueError, match='quantum_computing'):
            prepare_config(config)

    def test_required_fields_nn_missing_loss_fn(self):
        """Missing loss_fn when approximation_method='nn' raises ValueError."""
        config = {
            'approximation_method': 'nn',
            'num_epochs': 10,
            'num_samples': 100,
            # loss_fn missing
        }
        with pytest.raises(ValueError, match='loss_fn'):
            prepare_config(config)

    def test_required_fields_nn_missing_num_epochs(self):
        """Missing num_epochs when approximation_method='nn' raises ValueError."""
        config = {
            'approximation_method': 'nn',
            'loss_fn': 'logspace_mse_fdb',
            'num_samples': 100,
            # num_epochs missing
        }
        with pytest.raises(ValueError, match='num_epochs'):
            prepare_config(config)


# ===================================================================
# R005: Flat config passthrough and auto-detection
# ===================================================================


class TestFlatPassthrough:
    """R005: Flat configs pass through unchanged; detection is correct."""

    def test_flat_passthrough(self, reference_flat_config):
        """Flat config passes through with all live fields preserved.

        Dead fields (backward_ecl, num_batches_per_set) are stripped,
        but all other fields remain with their original values.
        """
        result = prepare_config(reference_flat_config)

        # All live fields should be present with original values
        assert result['ecl'] == reference_flat_config['ecl']
        assert result['iB'] == reference_flat_config['iB']
        assert result['lr'] == reference_flat_config['lr']
        assert result['loss_fn'] == reference_flat_config['loss_fn']
        assert result['hidden_sizes'] == reference_flat_config['hidden_sizes']
        assert result['num_epochs'] == reference_flat_config['num_epochs']
        assert result['device'] == reference_flat_config['device']
        assert result['sampling_scheme'] == reference_flat_config['sampling_scheme']
        assert result['fdb'] == reference_flat_config['fdb']

    def test_auto_detect_flat(self, reference_flat_config):
        """Flat config is detected correctly — no section nesting attempted."""
        # Remove dead fields to avoid ValueError
        config = {k: v for k, v in reference_flat_config.items()
                  if k not in ('backward_ecl', 'num_batches_per_set')}
        result = prepare_config(config)

        # Should still have flat keys, not section dicts
        assert isinstance(result.get('ecl'), int)
        assert not isinstance(result.get('ecl'), dict)

    def test_auto_detect_nested(self, equivalent_nested_config):
        """Nested config is detected correctly — sections are flattened."""
        result = prepare_config(equivalent_nested_config)

        # Section keys should be gone, replaced by flat internal keys
        assert 'inference' not in result
        assert 'training' not in result
        assert 'ecl' in result
        assert 'lr' in result


# ===================================================================
# R006: prepare_config returns a plain mutable dict
# ===================================================================


class TestReturnType:
    """R006: Return value semantics."""

    def test_prepare_config_returns_dict(self, minimal_flat_config):
        """Return value is a plain dict instance, mutable."""
        result = prepare_config(minimal_flat_config)

        assert isinstance(result, dict)
        assert type(result) is dict  # not a subclass or wrapper

        # Mutable — can add and modify keys (required by consumer code)
        result['new_key'] = 'new_value'
        assert result['new_key'] == 'new_value'
        result['device'] = 'cuda'
        assert result['device'] == 'cuda'

    def test_prepare_config_with_none(self):
        """prepare_config(None) returns empty dict or minimal defaults."""
        result = prepare_config(None)

        assert isinstance(result, dict)
        assert type(result) is dict

    def test_prepare_config_with_empty_dict(self):
        """prepare_config({}) returns a dict without erroring."""
        result = prepare_config({})

        assert isinstance(result, dict)
        assert type(result) is dict


# ===================================================================
# R005: Benchmark config passthrough (integration-flavored)
# ===================================================================


class TestBenchmarkPassthrough:
    """R005: Real benchmark configs pass through prepare_config correctly."""

    def test_benchmark_config_passthrough(self):
        """Import actual nbe_sanity_check config, run through prepare_config,
        verify all expected live fields present with correct values.

        Note: benchmark configs contain dead fields (backward_ecl,
        num_batches_per_set). Per D011, flat configs with dead fields
        get a warning + strip (not error), preserving backward compat.
        """
        # Import the benchmark config builder to get a real config dict
        from nce.benchmark_problems.nbe_sanity_check import _build_nbe_configs

        configs = _build_nbe_configs()
        config = configs[0]  # pedigree13

        # This should NOT raise — flat configs with dead fields warn+strip
        result = prepare_config(config)

        # Core fields should be preserved exactly
        assert result['device'] == 'cuda'
        assert result['iB'] == 20
        assert result['ecl'] == 2**19
        assert result['hidden_sizes'] == 'nbe,3'
        assert result['num_epochs'] == 500
        assert result['lr'] == 0.001
        assert result['loss_fn'] == 'weighted_logspace_mse'
        assert result['approximation_method'] == 'nn'
        assert result['num_samples'] == 'nbe,0.1'
        assert result['batch_size'] == 256
        assert result['seed'] == 42

        # Dead fields should be stripped
        assert 'backward_ecl' not in result
        assert 'num_batches_per_set' not in result
