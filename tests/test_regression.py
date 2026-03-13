"""Regression tests: flat vs nested configs produce identical inference (R017).

Verifies on rbm_20 (nbe_sanity_check model index 3):
  - prepare_config(flat) == prepare_config(nested)
  - Exact-only inference produces bitwise-equal partition functions
  - NN inference (2 epochs, 500 samples) produces bitwise-equal partition functions

Test classes:
  TestConfigRegression — config equality and inference equality checks
"""

import copy

import pytest
import torch

from nce.config_schema import prepare_config


MODEL_IDX = 3  # rbm_20


def _load_configs():
    """Load flat and nested configs for rbm_20. Returns (model, flat_cfg, nested_cfg)."""
    from nce.benchmark_problems import nbe_sanity_check
    return (
        nbe_sanity_check.problems[MODEL_IDX],
        nbe_sanity_check.configs['nbe'][MODEL_IDX],
        nbe_sanity_check.configs['nbe_nested'][MODEL_IDX],
    )


class TestConfigRegression:
    """Flat and nested configs produce identical results through the full pipeline."""

    def test_config_equality(self):
        """prepare_config(flat) == prepare_config(nested) for rbm_20."""
        _, flat_cfg, nested_cfg = _load_configs()

        flat_prepared = prepare_config(copy.deepcopy(flat_cfg))
        nested_prepared = prepare_config(copy.deepcopy(nested_cfg))

        # Build diff for diagnostic message
        all_keys = set(flat_prepared) | set(nested_prepared)
        diffs = {}
        for k in sorted(all_keys):
            v1 = flat_prepared.get(k, '<missing>')
            v2 = nested_prepared.get(k, '<missing>')
            if v1 != v2:
                diffs[k] = (v1, v2)

        assert flat_prepared == nested_prepared, (
            f"prepare_config mismatch for rbm_20 (index {MODEL_IDX}).\n"
            f"  Differing keys: {diffs}"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_exact_inference_equality(self):
        """Exact-only inference (high ecl, no NN) produces identical partition functions."""
        from nce.inference.graphical_model import FastGM

        model, flat_cfg, nested_cfg = _load_configs()
        device = "cuda"

        # Flat config: override ecl high, no NN training
        flat_exact = copy.deepcopy(flat_cfg)
        flat_exact['ecl'] = 2**30
        flat_exact['num_epochs'] = 0
        flat_exact_prepared = prepare_config(flat_exact)

        # Nested config: same overrides in nested structure
        nested_exact = copy.deepcopy(nested_cfg)
        nested_exact['inference']['exact_computation_limit'] = 2**30
        nested_exact['training']['num_epochs'] = 0
        nested_exact_prepared = prepare_config(nested_exact)

        gm_flat = FastGM(model=model, nn_config=flat_exact_prepared, device=device)
        gm_flat.eliminate_variables(all=True)
        logZ_flat = gm_flat.log_partition_function

        gm_nested = FastGM(model=model, nn_config=nested_exact_prepared, device=device)
        gm_nested.eliminate_variables(all=True)
        logZ_nested = gm_nested.log_partition_function

        assert logZ_flat == logZ_nested, (
            f"Exact inference partition function mismatch for rbm_20.\n"
            f"  flat:   {logZ_flat}\n"
            f"  nested: {logZ_nested}"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_nn_inference_equality(self):
        """NN inference (2 epochs, 500 samples) produces identical partition functions."""
        from nce.inference.graphical_model import FastGM

        model, flat_cfg, nested_cfg = _load_configs()
        device = "cuda"

        # Flat config: 2 epochs, 500 samples, fixed seed
        flat_nn = copy.deepcopy(flat_cfg)
        flat_nn['num_epochs'] = 2
        flat_nn['num_samples'] = 500
        flat_nn['set_size'] = 500
        flat_nn['seed'] = 42
        flat_nn_prepared = prepare_config(flat_nn)

        # Nested config: same overrides in nested structure
        nested_nn = copy.deepcopy(nested_cfg)
        nested_nn['training']['num_epochs'] = 2
        nested_nn['training']['seed'] = 42
        nested_nn['sampling']['num_samples'] = 500
        nested_nn['sampling']['set_size'] = 500
        nested_nn_prepared = prepare_config(nested_nn)

        gm_flat = FastGM(model=model, nn_config=flat_nn_prepared, device=device)
        gm_flat.eliminate_variables(all=True)
        logZ_flat = gm_flat.log_partition_function

        gm_nested = FastGM(model=model, nn_config=nested_nn_prepared, device=device)
        gm_nested.eliminate_variables(all=True)
        logZ_nested = gm_nested.log_partition_function

        assert logZ_flat == logZ_nested, (
            f"NN inference partition function mismatch for rbm_20.\n"
            f"  flat:   {logZ_flat}\n"
            f"  nested: {logZ_nested}"
        )
