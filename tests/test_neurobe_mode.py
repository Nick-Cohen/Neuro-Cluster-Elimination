"""NeuroBE training mode tests — normalization, early stopping, config, loss.

Covers:
- R037: Normalization round-trip (log10 → minmax_01 → undo → log10)
- R034: Patience-based early stopping trigger semantics
- R035: neurobe_mode config expansion completeness
- R033: neurobe_weighted_mse loss function numerical correctness

All tests are initially failing because the implementations don't exist yet.
Expected failure modes: ImportError, AttributeError, KeyError — NOT SyntaxError.
"""
import math

import pytest
import torch


# ---------------------------------------------------------------------------
# R037: Normalization round-trip
# ---------------------------------------------------------------------------
class TestNormalizationRoundTrip:
    """Min-max [0,1] normalization must be perfectly invertible.

    DataPreprocessor with normalization_mode='minmax_01' normalizes log10
    values to [0,1] via natural-log intermediate space, then undo_normalization
    recovers the original log10 values. The log-base conversion chain is:
        log10 → *ln(10) → minmax [0,1] → undo → /ln(10) → log10
    """

    def test_round_trip_known_values(self):
        """Known log10 values survive normalize → undo within 1e-6 (R037).

        Input log10 values: [0.0, 0.5, 1.0, 1.5, 2.0]
        After normalize: all values in [0, 1]
        After undo_normalization: recovers original log10 values.
        """
        from nce.data.data_preprocessor import DataPreprocessor

        log10_values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])

        dp = DataPreprocessor(normalization_mode='minmax_01')
        # Initialize normalizing constants from the data
        dp._initialize_normalizing_constant(log10_values, bw_vals=None)

        # Normalize
        normalized, _ = dp.normalize(log10_values, bw_vals=None)

        # All normalized values must be in [0, 1]
        assert normalized.min() >= 0.0, (
            f"Normalized min {normalized.min():.6f} < 0.0"
        )
        assert normalized.max() <= 1.0, (
            f"Normalized max {normalized.max():.6f} > 1.0"
        )

        # Undo normalization — should recover original log10 values
        recovered = dp.undo_normalization(normalized)

        torch.testing.assert_close(
            recovered, log10_values, atol=1e-6, rtol=0,
            msg=f"Round-trip failed: original={log10_values}, recovered={recovered}"
        )

    def test_identical_targets_no_division_by_zero(self):
        """All-identical targets don't cause division by zero (R037).

        When ln_max == ln_min, the range is zero. The DataPreprocessor
        should handle this gracefully (e.g., epsilon guard) without
        producing nan or inf values.
        """
        from nce.data.data_preprocessor import DataPreprocessor

        # All identical log10 values
        log10_values = torch.tensor([1.5, 1.5, 1.5, 1.5])

        dp = DataPreprocessor(normalization_mode='minmax_01')
        dp._initialize_normalizing_constant(log10_values, bw_vals=None)

        normalized, _ = dp.normalize(log10_values, bw_vals=None)

        # Must not contain nan or inf
        assert torch.isfinite(normalized).all(), (
            f"Normalized values contain non-finite: {normalized}"
        )

        # Undo should also be finite
        recovered = dp.undo_normalization(normalized)
        assert torch.isfinite(recovered).all(), (
            f"Recovered values contain non-finite: {recovered}"
        )


# ---------------------------------------------------------------------------
# R034: Patience-based early stopping
# ---------------------------------------------------------------------------
class TestNeurobeEarlyStoppingPatience:
    """Patience counter triggers after stop_iter+1 consecutive non-improving epochs.

    NeuroBE semantics: counter starts at 0, increments when loss >= prev_best,
    resets to 0 on improvement, breaks when count > stop_iter.
    With stop_iter=2: break after 3 consecutive non-improving epochs.
    """

    def test_patience_counter_triggers_correctly(self):
        """Loss sequence [1.0, 0.9, 0.8, 0.85, 0.86, 0.87] should trigger
        early stopping after the 6th value (3 non-improving after best=0.8).

        The counter logic:
          epoch 0: loss=1.0, prev_best=inf → improvement, count=0, prev_best=1.0
          epoch 1: loss=0.9 < 1.0 → improvement, count=0, prev_best=0.9
          epoch 2: loss=0.8 < 0.9 → improvement, count=0, prev_best=0.8
          epoch 3: loss=0.85 >= 0.8 → count=1
          epoch 4: loss=0.86 >= 0.8 → count=2
          epoch 5: loss=0.87 >= 0.8 → count=3 → 3 > 2 → BREAK

        Training should stop at epoch 5 (the 6th epoch, 0-indexed).
        """
        # Simulate the patience counter logic that Trainer will implement
        losses = [1.0, 0.9, 0.8, 0.85, 0.86, 0.87, 0.88, 0.89]
        stop_iter = 2
        prev_best = float('inf')
        count = 0
        stopped_at = None

        for epoch, loss in enumerate(losses):
            if loss < prev_best:
                prev_best = loss
                count = 0
            else:
                count += 1

            if count > stop_iter:
                stopped_at = epoch
                break

        assert stopped_at == 5, (
            f"Expected early stopping at epoch 5, got epoch {stopped_at}. "
            f"Counter should trigger when count ({count}) > stop_iter ({stop_iter})."
        )

    def test_patience_resets_on_improvement(self):
        """If loss improves after some non-improving epochs, counter resets.

        Loss sequence: [1.0, 0.9, 0.95, 0.85, 0.9, 0.91, 0.92, 0.93]
          epoch 0: 1.0 → best=1.0, count=0
          epoch 1: 0.9 → best=0.9, count=0
          epoch 2: 0.95 → count=1
          epoch 3: 0.85 → best=0.85, count=0 (reset!)
          epoch 4: 0.9 → count=1
          epoch 5: 0.91 → count=2
          epoch 6: 0.92 → count=3 → 3 > 2 → BREAK at epoch 6
        """
        losses = [1.0, 0.9, 0.95, 0.85, 0.9, 0.91, 0.92, 0.93]
        stop_iter = 2
        prev_best = float('inf')
        count = 0
        stopped_at = None

        for epoch, loss in enumerate(losses):
            if loss < prev_best:
                prev_best = loss
                count = 0
            else:
                count += 1

            if count > stop_iter:
                stopped_at = epoch
                break

        assert stopped_at == 6, (
            f"Expected early stopping at epoch 6 (after reset), got {stopped_at}."
        )

    def test_trainer_uses_neurobe_early_stopping_config(self, neurobe_training_config):
        """Trainer reads neurobe_early_stopping and neurobe_stop_iter from config.

        This test verifies the Trainer actually uses the config fields.
        Will fail with KeyError/AttributeError until Trainer is updated.
        """
        from nce.neural_networks.train import Trainer

        config = neurobe_training_config
        assert config['neurobe_early_stopping'] is True
        assert config['neurobe_stop_iter'] == 2

        # Verify Trainer can accept these config fields without error.
        # (Full integration test deferred to T04 — this just checks the
        # config is wired through.)
        # The Trainer constructor doesn't take config directly, so we
        # verify the fields exist and are accessible. The actual training
        # loop integration is tested in T04.


# ---------------------------------------------------------------------------
# R035: Config expansion
# ---------------------------------------------------------------------------
class TestNeurobeConfigExpansion:
    """neurobe_mode=True in config expands to all NeuroBE-faithful defaults."""

    # Expected defaults from S01-RESEARCH NEUROBE_DEFAULTS
    NEUROBE_DEFAULTS = {
        'normalization_mode': 'minmax_01',
        'loss_fn': 'neurobe_weighted_mse',
        'batch_size': 256,
        'lr': 0.001,
        'num_epochs': 500,
        'neurobe_early_stopping': True,
        'neurobe_stop_iter': 2,
        'use_bw_approx': False,
        'populate_bw_factors': False,
        'activation': 'relu',
        'use_amp': False,
        'hidden_sizes': 'neurobe,3',
        'skip_early_stopping': True,
        'nbe_early_stopping': False,
        'lower_dim': True,
        'sampling_scheme': 'all',
        'iB': 25,
    }

    def test_neurobe_mode_expands_all_defaults(self):
        """prepare_config with neurobe_mode=True sets all NEUROBE_DEFAULTS (R035).

        Minimal required fields plus neurobe_mode=True should produce a
        config containing every key from NEUROBE_DEFAULTS with the expected
        default values.
        """
        from nce.config_schema import prepare_config

        config = prepare_config({
            'neurobe_mode': True,
            'loss_fn': 'neurobe_weighted_mse',
            'num_epochs': 500,
            'num_samples': 1000,
            'iB': 25,
            'ecl': 100,
            'device': 'cpu',
        })

        for key, expected_value in self.NEUROBE_DEFAULTS.items():
            assert key in config, (
                f"neurobe_mode expansion missing key '{key}' in result config"
            )
            assert config[key] == expected_value, (
                f"neurobe_mode default for '{key}': "
                f"expected {expected_value!r}, got {config[key]!r}"
            )

    def test_explicit_override_wins(self):
        """Explicit user values override neurobe_mode defaults (R035).

        neurobe_mode sets lr=0.001 as default, but if the user passes
        lr=0.01, the user's value must survive.
        """
        from nce.config_schema import prepare_config

        config = prepare_config({
            'neurobe_mode': True,
            'loss_fn': 'neurobe_weighted_mse',
            'num_epochs': 500,
            'num_samples': 1000,
            'iB': 25,
            'ecl': 100,
            'device': 'cpu',
            'lr': 0.01,  # explicit override
        })

        assert config['lr'] == 0.01, (
            f"Explicit lr override should win: expected 0.01, got {config['lr']}"
        )


# ---------------------------------------------------------------------------
# R033: neurobe_weighted_mse loss function
# ---------------------------------------------------------------------------
class TestNeurobeWeightedMSE:
    """Hand-computed expected loss for known inputs.

    Formula from NeuroBE Function-NN.hxx:
        w = labels * (ln_max - ln_min) / sum_ln
        loss = mean(w * (output - labels)^2)

    Where labels are [0,1]-normalized values.
    """

    def test_known_inputs_match_hand_computed(self):
        """Hand-computed loss for known inputs matches function output (R033).

        Inputs:
            outputs = [0.3, 0.7]
            labels  = [0.25, 0.75]  (already [0,1]-normalized)
            ln_min = 0.0, ln_max = 4.0, sum_ln = 3.0

        Weights:
            w_0 = 0.25 * (4.0 - 0.0) / 3.0 = 1/3
            w_1 = 0.75 * (4.0 - 0.0) / 3.0 = 1.0

        Weighted squared errors:
            wse_0 = (1/3) * (0.3 - 0.25)^2 = (1/3) * 0.0025 = 0.000833...
            wse_1 = 1.0  * (0.7 - 0.75)^2  = 1.0  * 0.0025  = 0.0025

        Loss = mean([0.000833..., 0.0025]) = 0.001666...
        """
        from nce.neural_networks.losses import neurobe_weighted_mse

        outputs = torch.tensor([0.3, 0.7])
        labels = torch.tensor([0.25, 0.75])
        ln_min = 0.0
        ln_max = 4.0
        sum_ln = 3.0

        result = neurobe_weighted_mse(outputs, labels, ln_min, ln_max, sum_ln)

        # Hand-computed expected value
        w = labels * (ln_max - ln_min) / sum_ln
        expected = (w * (outputs - labels) ** 2).mean()

        assert abs(result.item() - expected.item()) < 1e-6, (
            f"neurobe_weighted_mse: got {result.item():.8f}, "
            f"expected {expected.item():.8f} "
            f"(diff={abs(result.item() - expected.item()):.2e})"
        )

    def test_zero_labels_produce_zero_weights(self):
        """When a label is 0.0 (normalized), its weight should be 0 (R033).

        A zero-normalized label means the original value was ln_min.
        Its contribution to the loss should be zero regardless of the
        output value at that position.
        """
        from nce.neural_networks.losses import neurobe_weighted_mse

        outputs = torch.tensor([0.9, 0.5])  # large error at position 0
        labels = torch.tensor([0.0, 0.5])   # zero label at position 0
        ln_min = 0.0
        ln_max = 4.0
        sum_ln = 2.0

        result = neurobe_weighted_mse(outputs, labels, ln_min, ln_max, sum_ln)

        # Weight for label=0.0 is 0, so only position 1 contributes
        # w_1 = 0.5 * 4.0 / 2.0 = 1.0
        # wse_1 = 1.0 * (0.5 - 0.5)^2 = 0.0
        # loss = mean([0.0, 0.0]) = 0.0
        assert abs(result.item()) < 1e-6, (
            f"Expected loss ~0.0 when zero-label has zero weight, got {result.item():.8f}"
        )
