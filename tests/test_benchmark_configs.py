"""Tests for benchmark config cleanliness and helper functions (S02).

Verifies that benchmark config builders produce clean configs without dead
fields, that helper functions like set_bw_ecl() write only live fields,
and that nested config builders round-trip to identical flat output.

Test classes:
  TestCleanBenchmarkConfigs — dead fields absent from builder output
  TestSetBwEcl — set_bw_ecl() writes bw_ecl + populate_bw_factors, NOT backward_ecl
  TestBenchmarkConfigValidation — all flat configs pass prepare_config() cleanly
  TestNestedBuilderRoundTrip — prepare_config(nested) == prepare_config(flat) per model
  TestNestedBuilderValidation — nested configs pass prepare_config(strict=True)
  TestWorkerConfigClean — worker build_nn_config() output has no dead fields
"""
import warnings
from pathlib import Path

import pytest

from nce.config_schema import prepare_config


# ===================================================================
# Dead fields must be absent from benchmark config builders
# ===================================================================


class TestCleanBenchmarkConfigs:
    """Assert backward_ecl and num_batches_per_set are absent from all
    benchmark config builder output."""

    def test_nbe_configs_no_backward_ecl(self):
        """backward_ecl absent from all 5 nbe_sanity_check configs."""
        try:
            from nce.benchmark_problems.nbe_sanity_check import _build_nbe_configs
        except (ValueError, TypeError, OSError) as exc:
            pytest.skip(f"Benchmark config builder not available: {exc}")

        configs = _build_nbe_configs()
        assert len(configs) == 5, f"Expected 5 configs, got {len(configs)}"
        for i, cfg in enumerate(configs):
            assert 'backward_ecl' not in cfg, (
                f"nbe config {i} still contains 'backward_ecl'"
            )

    def test_nbe_configs_no_num_batches_per_set(self):
        """num_batches_per_set absent from all 5 nbe_sanity_check configs."""
        try:
            from nce.benchmark_problems.nbe_sanity_check import _build_nbe_configs
        except (ValueError, TypeError, OSError) as exc:
            pytest.skip(f"Benchmark config builder not available: {exc}")

        configs = _build_nbe_configs()
        for i, cfg in enumerate(configs):
            assert 'num_batches_per_set' not in cfg, (
                f"nbe config {i} still contains 'num_batches_per_set'"
            )

    def test_default_configs_no_backward_ecl(self):
        """backward_ecl absent from all 24 small_problems default configs."""
        try:
            from nce.benchmark_problems.small_problems import _build_default_configs
        except (ValueError, TypeError, OSError) as exc:
            pytest.skip(f"Benchmark config builder not available: {exc}")

        configs = _build_default_configs()
        assert len(configs) == 24, f"Expected 24 configs, got {len(configs)}"
        for i, cfg in enumerate(configs):
            assert 'backward_ecl' not in cfg, (
                f"default config {i} still contains 'backward_ecl'"
            )

    def test_default_configs_no_num_batches_per_set(self):
        """num_batches_per_set absent from all 24 small_problems default configs."""
        try:
            from nce.benchmark_problems.small_problems import _build_default_configs
        except (ValueError, TypeError, OSError) as exc:
            pytest.skip(f"Benchmark config builder not available: {exc}")

        configs = _build_default_configs()
        for i, cfg in enumerate(configs):
            assert 'num_batches_per_set' not in cfg, (
                f"default config {i} still contains 'num_batches_per_set'"
            )


# ===================================================================
# set_bw_ecl must not write backward_ecl
# ===================================================================


class TestSetBwEcl:
    """set_bw_ecl() writes bw_ecl and populate_bw_factors but NOT backward_ecl."""

    def _make_benchmark_set(self):
        """Create a minimal BenchmarkSet for testing set_bw_ecl."""
        from nce.benchmark_problems.nbe_sanity_check import BenchmarkSet
        configs = [
            {'bw_ecl': 0, 'populate_bw_factors': False},
            {'bw_ecl': 0, 'populate_bw_factors': False},
        ]
        return BenchmarkSet(problems=[], configs={'test': configs})

    def test_set_bw_ecl_writes_bw_ecl(self):
        """set_bw_ecl sets bw_ecl to the given value."""
        from nce.benchmark_problems.small_problems import set_bw_ecl
        bs = self._make_benchmark_set()
        set_bw_ecl(bs, 'test', 1024)
        for cfg in bs.configs['test']:
            assert cfg['bw_ecl'] == 1024

    def test_set_bw_ecl_writes_populate_bw_factors(self):
        """set_bw_ecl sets populate_bw_factors=True when value > 0."""
        from nce.benchmark_problems.small_problems import set_bw_ecl
        bs = self._make_benchmark_set()
        set_bw_ecl(bs, 'test', 1024)
        for cfg in bs.configs['test']:
            assert cfg['populate_bw_factors'] is True

    def test_set_bw_ecl_zero_disables_populate(self):
        """set_bw_ecl with value=0 sets populate_bw_factors=False."""
        from nce.benchmark_problems.small_problems import set_bw_ecl
        bs = self._make_benchmark_set()
        set_bw_ecl(bs, 'test', 0)
        for cfg in bs.configs['test']:
            assert cfg['bw_ecl'] == 0
            assert cfg['populate_bw_factors'] is False

    def test_set_bw_ecl_does_not_write_backward_ecl(self):
        """set_bw_ecl must NOT write a backward_ecl key."""
        from nce.benchmark_problems.small_problems import set_bw_ecl
        bs = self._make_benchmark_set()
        set_bw_ecl(bs, 'test', 1024)
        for cfg in bs.configs['test']:
            assert 'backward_ecl' not in cfg, (
                "set_bw_ecl should not write 'backward_ecl'"
            )


# ===================================================================
# All benchmark configs pass prepare_config() without warnings
# ===================================================================


class TestBenchmarkConfigValidation:
    """Each flat benchmark config passes prepare_config() with no warnings."""

    def test_nbe_configs_no_warnings(self):
        """All 5 nbe_sanity_check configs pass prepare_config() silently."""
        try:
            from nce.benchmark_problems.nbe_sanity_check import _build_nbe_configs
        except (ValueError, TypeError, OSError) as exc:
            pytest.skip(f"Benchmark config builder not available: {exc}")

        configs = _build_nbe_configs()
        for i, cfg in enumerate(configs):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = prepare_config(cfg)

            dead_field_warnings = [
                x for x in w if 'dead' in str(x.message).lower()
                or 'backward_ecl' in str(x.message)
                or 'num_batches_per_set' in str(x.message)
            ]
            assert len(dead_field_warnings) == 0, (
                f"nbe config {i} emitted dead-field warnings: "
                f"{[str(x.message) for x in dead_field_warnings]}"
            )

            # Verify the result is a valid dict with expected keys
            assert isinstance(result, dict)
            assert 'ecl' in result
            assert 'iB' in result

    def test_default_configs_no_warnings(self):
        """All 24 small_problems default configs pass prepare_config() silently."""
        try:
            from nce.benchmark_problems.small_problems import _build_default_configs
        except (ValueError, TypeError, OSError) as exc:
            pytest.skip(f"Benchmark config builder not available: {exc}")

        configs = _build_default_configs()
        for i, cfg in enumerate(configs):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = prepare_config(cfg)

            dead_field_warnings = [
                x for x in w if 'dead' in str(x.message).lower()
                or 'backward_ecl' in str(x.message)
                or 'num_batches_per_set' in str(x.message)
            ]
            assert len(dead_field_warnings) == 0, (
                f"default config {i} emitted dead-field warnings: "
                f"{[str(x.message) for x in dead_field_warnings]}"
            )

            assert isinstance(result, dict)
            assert 'ecl' in result

    def test_nbe_configs_strict_mode(self):
        """All nbe configs pass prepare_config(strict=True) without error."""
        try:
            from nce.benchmark_problems.nbe_sanity_check import _build_nbe_configs
        except (ValueError, TypeError, OSError) as exc:
            pytest.skip(f"Benchmark config builder not available: {exc}")

        configs = _build_nbe_configs()
        for i, cfg in enumerate(configs):
            # Should not raise ValueError
            result = prepare_config(cfg, strict=True)
            assert isinstance(result, dict), f"Config {i}: expected dict"

    def test_default_configs_strict_mode(self):
        """All default configs pass prepare_config(strict=True) without error."""
        try:
            from nce.benchmark_problems.small_problems import _build_default_configs
        except (ValueError, TypeError, OSError) as exc:
            pytest.skip(f"Benchmark config builder not available: {exc}")

        configs = _build_default_configs()
        for i, cfg in enumerate(configs):
            result = prepare_config(cfg, strict=True)
            assert isinstance(result, dict), f"Config {i}: expected dict"


# ===================================================================
# Nested builder round-trip equality: prepare_config(nested) == prepare_config(flat)
# ===================================================================


class TestNestedBuilderRoundTrip:
    """prepare_config(nested[i]) == prepare_config(flat[i]) for every model."""

    @pytest.mark.parametrize("model_idx", range(5), ids=[f"nbe-{i}" for i in range(5)])
    def test_nbe_round_trip(self, model_idx):
        """Nested nbe config round-trips to same flat output as flat nbe config."""
        try:
            from nce.benchmark_problems.nbe_sanity_check import (
                _build_nbe_configs,
                _build_nbe_nested_configs,
            )
        except (ValueError, TypeError, OSError) as exc:
            pytest.skip(f"Benchmark config builder not available: {exc}")

        flat_configs = _build_nbe_configs()
        nested_configs = _build_nbe_nested_configs()

        flat_result = prepare_config(flat_configs[model_idx])
        nested_result = prepare_config(nested_configs[model_idx])

        assert flat_result == nested_result, (
            f"nbe model {model_idx}: nested config round-trip mismatch.\n"
            f"  Keys only in flat:   {set(flat_result) - set(nested_result)}\n"
            f"  Keys only in nested: {set(nested_result) - set(flat_result)}\n"
            f"  Value diffs: {_diff_dicts(flat_result, nested_result)}"
        )

    @pytest.mark.parametrize("model_idx", range(24), ids=[f"default-{i}" for i in range(24)])
    def test_default_round_trip(self, model_idx):
        """Nested default config round-trips to same flat output as flat default config."""
        try:
            from nce.benchmark_problems.small_problems import (
                _build_default_configs,
                _build_default_nested_configs,
            )
        except (ValueError, TypeError, OSError) as exc:
            pytest.skip(f"Benchmark config builder not available: {exc}")

        flat_configs = _build_default_configs()
        nested_configs = _build_default_nested_configs()

        flat_result = prepare_config(flat_configs[model_idx])
        nested_result = prepare_config(nested_configs[model_idx])

        assert flat_result == nested_result, (
            f"default model {model_idx}: nested config round-trip mismatch.\n"
            f"  Keys only in flat:   {set(flat_result) - set(nested_result)}\n"
            f"  Keys only in nested: {set(nested_result) - set(flat_result)}\n"
            f"  Value diffs: {_diff_dicts(flat_result, nested_result)}"
        )


# ===================================================================
# Nested builder validation: strict mode passes for all nested configs
# ===================================================================


class TestNestedBuilderValidation:
    """Each nested config passes prepare_config(strict=True) without error."""

    @pytest.mark.parametrize("model_idx", range(5), ids=[f"nbe-{i}" for i in range(5)])
    def test_nbe_nested_strict(self, model_idx):
        """Nested nbe config passes strict validation."""
        try:
            from nce.benchmark_problems.nbe_sanity_check import _build_nbe_nested_configs
        except (ValueError, TypeError, OSError) as exc:
            pytest.skip(f"Benchmark config builder not available: {exc}")

        configs = _build_nbe_nested_configs()
        result = prepare_config(configs[model_idx], strict=True)
        assert isinstance(result, dict)
        assert 'ecl' in result
        assert 'iB' in result

    @pytest.mark.parametrize("model_idx", range(24), ids=[f"default-{i}" for i in range(24)])
    def test_default_nested_strict(self, model_idx):
        """Nested default config passes strict validation."""
        try:
            from nce.benchmark_problems.small_problems import _build_default_nested_configs
        except (ValueError, TypeError, OSError) as exc:
            pytest.skip(f"Benchmark config builder not available: {exc}")

        configs = _build_default_nested_configs()
        result = prepare_config(configs[model_idx], strict=True)
        assert isinstance(result, dict)
        assert 'ecl' in result


# ===================================================================
# Worker's build_nn_config() output is clean (no dead fields)
# ===================================================================


class TestWorkerConfigClean:
    """build_nn_config() must not emit dead fields or trigger warnings."""

    @staticmethod
    def _import_build_nn_config():
        """Import build_nn_config from the notebooks worker module."""
        import importlib
        import sys as _sys
        worker_dir = str(Path(__file__).resolve().parent.parent / 'notebooks' / '_1-2026')
        if worker_dir not in _sys.path:
            _sys.path.insert(0, worker_dir)
        # Force reimport in case it was cached from a prior test session
        if 'worker' in _sys.modules:
            del _sys.modules['worker']
        mod = importlib.import_module('worker')
        return mod.build_nn_config

    @staticmethod
    def _sample_config():
        """Return a minimal experiment config dict for build_nn_config()."""
        return {
            'epochs': 100,
            'sampling_scheme': 'all',
            'batch_size': 1000,
            'num_samples': 1000,
            'set_size': 1000,
            'loss': 'unnormalized_kl',
            'val_set': 'all',
            'ecl': 1024,
        }

    def test_no_backward_ecl(self):
        """backward_ecl must not appear in build_nn_config() output."""
        build_nn_config = self._import_build_nn_config()
        nn_config = build_nn_config(self._sample_config(), [128, 128], bw_ecl=64, seed=42)
        assert 'backward_ecl' not in nn_config

    def test_no_num_batches_per_set(self):
        """num_batches_per_set must not appear in build_nn_config() output."""
        build_nn_config = self._import_build_nn_config()
        nn_config = build_nn_config(self._sample_config(), [128, 128], bw_ecl=64, seed=42)
        assert 'num_batches_per_set' not in nn_config

    def test_prepare_config_no_warnings(self):
        """build_nn_config() output passes prepare_config() without dead-field warnings."""
        build_nn_config = self._import_build_nn_config()
        nn_config = build_nn_config(self._sample_config(), [], bw_ecl=0, seed=1)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prepare_config(nn_config)

        dead_field_warnings = [
            x for x in w if 'dead' in str(x.message).lower()
            or 'backward_ecl' in str(x.message)
            or 'num_batches_per_set' in str(x.message)
        ]
        assert len(dead_field_warnings) == 0, (
            f"build_nn_config() triggered dead-field warnings: "
            f"{[str(x.message) for x in dead_field_warnings]}"
        )

    def test_bw_ecl_zero_config(self):
        """build_nn_config() with bw_ecl=0 produces valid config."""
        build_nn_config = self._import_build_nn_config()
        nn_config = build_nn_config(self._sample_config(), [64], bw_ecl=0, seed=7)
        assert nn_config['use_bw_approx'] is False
        assert nn_config['populate_bw_factors'] is False
        assert nn_config['bw_ecl'] is None

    def test_bw_ecl_positive_config(self):
        """build_nn_config() with bw_ecl>0 sets backward fields correctly."""
        build_nn_config = self._import_build_nn_config()
        nn_config = build_nn_config(self._sample_config(), [64], bw_ecl=256, seed=7)
        assert nn_config['use_bw_approx'] is True
        assert nn_config['populate_bw_factors'] is True
        assert nn_config['bw_ecl'] == 256


def _diff_dicts(d1, d2):
    """Return a dict of keys where values differ between d1 and d2."""
    diffs = {}
    all_keys = set(d1) | set(d2)
    for k in sorted(all_keys):
        v1 = d1.get(k, '<missing>')
        v2 = d2.get(k, '<missing>')
        if v1 != v2:
            diffs[k] = (v1, v2)
    return diffs
