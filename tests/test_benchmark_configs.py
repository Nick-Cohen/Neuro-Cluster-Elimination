"""Tests for benchmark config cleanliness and helper functions (S02-T01).

Verifies that benchmark config builders produce clean configs without dead
fields, and that helper functions like set_bw_ecl() write only live fields.

Test classes:
  TestCleanBenchmarkConfigs — dead fields absent from builder output
  TestSetBwEcl — set_bw_ecl() writes bw_ecl + populate_bw_factors, NOT backward_ecl
  TestBenchmarkConfigValidation — all flat configs pass prepare_config() cleanly
"""
import warnings

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
