"""Exact inference correctness tests on hand-built problems.

Covers:
- R018: Exact inference produces known-correct partition function values
- R020: Domain ≥ 3 variables work through the exact inference path

All tests force the exact computation path by setting ecl=2**30,
ensuring no NN approximation is used.
"""
import math

import pytest

from nce.config_schema import prepare_config
from nce.inference.graphical_model import FastGM


def _make_exact_config(base_config):
    """Override a base config to force fully-exact inference.

    Sets ecl=2**30 so every bucket falls under the exact computation
    limit, and dope_factors=False to avoid perturbing factor values.
    Returns a fresh prepare_config()-validated dict.
    """
    overrides = {
        **base_config,
        'ecl': 2**30,
        'iB': 100,
        'dope_factors': False,
    }
    return prepare_config(overrides)


class TestExactInference:
    """Verify exact variable elimination produces analytically known Z values."""

    def test_binary_chain_exact_z(self, binary_chain_factors, nn_training_config):
        """Binary chain: two domain-2 vars, uniform pairwise factor → Z=2.0 (R018)."""
        config = _make_exact_config(nn_training_config)
        fixture = binary_chain_factors

        gm = FastGM(
            factors=fixture['factors'],
            elim_order=fixture['elim_order'],
            nn_config=config,
            device='cpu',
        )
        gm.eliminate_variables(all=True)

        expected = fixture['expected_log10_z']
        actual = gm.log_partition_function
        assert abs(actual - expected) < 1e-5, (
            f"Binary chain log10(Z): got {actual:.8f}, expected {expected:.8f} "
            f"(diff={abs(actual - expected):.2e})"
        )

    def test_ternary_chain_exact_z(self, ternary_chain_factors, nn_training_config):
        """Ternary chain: two domain-3 vars, uniform pairwise factor → Z=4.5 (R020)."""
        config = _make_exact_config(nn_training_config)
        fixture = ternary_chain_factors

        gm = FastGM(
            factors=fixture['factors'],
            elim_order=fixture['elim_order'],
            nn_config=config,
            device='cpu',
        )
        gm.eliminate_variables(all=True)

        expected = fixture['expected_log10_z']
        actual = gm.log_partition_function
        assert abs(actual - expected) < 1e-5, (
            f"Ternary chain log10(Z): got {actual:.8f}, expected {expected:.8f} "
            f"(diff={abs(actual - expected):.2e})"
        )

    def test_star_graph_exact_z(self, star_graph_factors, nn_training_config):
        """Star graph: hub + 3 leaves with 3-way factor → exact Z matches analytic (R018)."""
        config = _make_exact_config(nn_training_config)
        fixture = star_graph_factors

        gm = FastGM(
            factors=fixture['factors'],
            elim_order=fixture['elim_order'],
            nn_config=config,
            device='cpu',
        )
        gm.eliminate_variables(all=True)

        expected = fixture['expected_log10_z']
        actual = gm.log_partition_function
        assert abs(actual - expected) < 1e-5, (
            f"Star graph log10(Z): got {actual:.8f}, expected {expected:.8f} "
            f"(diff={abs(actual - expected):.2e})"
        )
