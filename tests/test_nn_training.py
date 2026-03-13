"""Single-bucket NN training and convergence tests.

Covers:
- R019: NN training completes without error on CPU; at least one bucket trains via NN
- R021: Loss measurably decreases over 50 epochs (convergence)

Uses the star graph fixture which produces a multi-variable message exceeding
ecl=4, triggering the NN training path. All tests use seed=42 for reproducibility.
"""
import copy

import pytest
import torch

from nce.config_schema import prepare_config
from nce.inference.graphical_model import FastGM


def _run_star_graph_nn(star_graph_factors, nn_training_config):
    """Build and run NN inference on the star graph problem.

    Returns the FastGM instance after full variable elimination.
    """
    fixture = star_graph_factors
    gm = FastGM(
        factors=fixture['factors'],
        elim_order=fixture['elim_order'],
        nn_config=nn_training_config,
        device='cpu',
    )
    gm.eliminate_variables(all=True)
    return gm


class TestNNTrainingCompletes:
    """Verify NN training runs to completion on the star graph problem (R019)."""

    def test_nn_training_completes(self, star_graph_factors, nn_training_config):
        """NN training on star graph completes without exception (R019).

        With ecl=4, the star graph's bucket 0 has message_size=8 which
        exceeds the exact computation limit, forcing the NN path.
        Asserts that per_bucket_training_log is non-empty (at least one
        bucket was trained via NN) and that each entry contains loss data.
        """
        gm = _run_star_graph_nn(star_graph_factors, nn_training_config)

        assert len(gm.per_bucket_training_log) > 0, (
            "per_bucket_training_log is empty — no bucket was trained via NN. "
            "Check that ecl=4 is small enough to trigger NN path on the star graph."
        )

        # Every NN-trained bucket should have recorded loss data
        for entry in gm.per_bucket_training_log:
            assert len(entry['losses']) > 0, (
                f"Bucket '{entry['label']}' has empty losses list despite NN training."
            )

    def test_training_log_format(self, star_graph_factors, nn_training_config):
        """Training log entries have the expected structure (R019).

        Guards against format regressions that would break downstream
        visualization and analysis tools.
        """
        gm = _run_star_graph_nn(star_graph_factors, nn_training_config)

        assert len(gm.per_bucket_training_log) > 0, (
            "per_bucket_training_log is empty — cannot verify format."
        )

        required_keys = {'label', 'epochs_trained', 'hidden_sizes', 'losses', 'val_losses'}

        for entry in gm.per_bucket_training_log:
            missing = required_keys - set(entry.keys())
            assert not missing, (
                f"Training log entry for bucket '{entry.get('label', '?')}' "
                f"is missing keys: {missing}. Present keys: {set(entry.keys())}"
            )

            # Structural checks on field types
            assert isinstance(entry['label'], (str, int)), (
                f"label should be str or int, got {type(entry['label'])}"
            )
            assert isinstance(entry['epochs_trained'], (int, float)), (
                f"epochs_trained should be numeric, got {type(entry['epochs_trained'])}"
            )
            assert isinstance(entry['hidden_sizes'], (list, str)), (
                f"hidden_sizes should be list or str, got {type(entry['hidden_sizes'])}"
            )
            assert isinstance(entry['losses'], list), (
                f"losses should be list, got {type(entry['losses'])}"
            )
            assert isinstance(entry['val_losses'], list), (
                f"val_losses should be list, got {type(entry['val_losses'])}"
            )

            # Each loss entry is a (epoch, loss_value) tuple
            if entry['losses']:
                first_loss = entry['losses'][0]
                assert len(first_loss) == 2, (
                    f"Loss entry should be (epoch, loss_value) tuple, "
                    f"got length {len(first_loss)}: {first_loss}"
                )


class TestConvergence:
    """Verify NN training loss decreases over epochs (R021)."""

    def test_convergence_loss_decreases(self, star_graph_factors, nn_training_config):
        """Loss decreases: final avg < 0.9 × initial avg for ≥1 NN bucket (R021).

        Compares the average loss over the first 5 epochs to the average
        over the last 5 epochs. At least one NN-trained bucket must show
        a 10%+ decrease. Uses seed=42 for reproducibility.

        Research baseline: star graph loss goes from ~5.46 to ~2.64 over
        50 epochs (seed 42, logspace_mse_fdb loss).
        """
        gm = _run_star_graph_nn(star_graph_factors, nn_training_config)

        assert len(gm.per_bucket_training_log) > 0, (
            "per_bucket_training_log is empty — no NN-trained buckets to check."
        )

        any_converged = False
        diagnostics = []

        for entry in gm.per_bucket_training_log:
            losses = entry['losses']  # list of (epoch, loss_value) tuples
            if len(losses) < 10:
                diagnostics.append(
                    f"  Bucket '{entry['label']}': only {len(losses)} epochs, "
                    f"need ≥10 for convergence check"
                )
                continue

            # Extract loss values (second element of each tuple)
            loss_values = [lv for _, lv in losses]
            initial_avg = sum(loss_values[:5]) / 5
            final_avg = sum(loss_values[-5:]) / 5

            converged = final_avg < 0.9 * initial_avg
            if converged:
                any_converged = True

            diagnostics.append(
                f"  Bucket '{entry['label']}': "
                f"initial_avg={initial_avg:.4f}, final_avg={final_avg:.4f}, "
                f"ratio={final_avg / initial_avg:.4f}, "
                f"converged={'YES' if converged else 'NO'}"
            )

        diag_str = "\n".join(diagnostics)
        assert any_converged, (
            f"No NN-trained bucket showed ≥10% loss decrease.\n"
            f"Loss trajectory per bucket:\n{diag_str}"
        )
