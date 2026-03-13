"""Loss function robustness edge-case tests.

Covers:
- R022: Loss functions don't crash on inf inputs
- R023: Loss functions don't crash on all-neg-inf or all-zero targets

These tests call loss functions directly with real tensors — no Trainer,
no mocking. The requirement is that no unhandled exception is raised.
The result may be inf or nan; that's documented but not asserted against.
"""
import math

import pytest
import torch

from nce.neural_networks.losses import (
    from_logspace_mse,
    linspace_mse_fdb,
    logspace_mse_fdb,
)

# Loss functions that accept the standard (outputs, targets, bw_hat=None) signature
# and don't require additional mandatory arguments like sigma_f/sigma_g.
STANDARD_LOSS_FNS = [
    pytest.param(logspace_mse_fdb, id="logspace_mse_fdb"),
    pytest.param(linspace_mse_fdb, id="linspace_mse_fdb"),
    pytest.param(from_logspace_mse, id="from_logspace_mse"),
]


class TestInfInputNoCrash:
    """R022: Loss functions survive inf in the outputs tensor without crashing."""

    @pytest.mark.parametrize("loss_fn", STANDARD_LOSS_FNS)
    def test_loss_fn_inf_input_no_crash(self, loss_fn):
        """Calling a loss function with inf in outputs must not raise.

        The result may be inf or nan — the contract is no unhandled exception.
        """
        outputs = torch.tensor([float("inf"), 0.0, -1.0])
        targets = torch.tensor([1.0, 0.5, -0.5])

        result = loss_fn(outputs, targets)

        # Document what the function actually returns (no assertion on finiteness)
        is_finite = torch.isfinite(result).item() if isinstance(result, torch.Tensor) else math.isfinite(result)
        outcome = "finite" if is_finite else ("nan" if (isinstance(result, torch.Tensor) and torch.isnan(result).item()) else "inf")
        # Parametrized test name already identifies the function; this just
        # documents actual behavior for anyone reading test output with -v -s.
        print(f"  {loss_fn.__name__} with inf input → {outcome} ({result})")


class TestNegInfTargetsNoCrash:
    """R023: Loss functions survive all-neg-inf targets without crashing."""

    @pytest.mark.parametrize("loss_fn", STANDARD_LOSS_FNS)
    def test_loss_fn_neg_inf_targets_no_crash(self, loss_fn):
        """All-neg-inf targets (log-space zero probability) must not raise."""
        outputs = torch.tensor([1.0, 0.5, -0.5])
        targets = torch.tensor([float("-inf"), float("-inf"), float("-inf")])

        result = loss_fn(outputs, targets)

        is_finite = torch.isfinite(result).item() if isinstance(result, torch.Tensor) else math.isfinite(result)
        outcome = "finite" if is_finite else ("nan" if (isinstance(result, torch.Tensor) and torch.isnan(result).item()) else "inf")
        print(f"  {loss_fn.__name__} with all -inf targets → {outcome} ({result})")

    @pytest.mark.parametrize("loss_fn", STANDARD_LOSS_FNS)
    def test_loss_fn_zero_targets_no_crash(self, loss_fn):
        """All-zero targets (equal probability in log-space) must not raise."""
        outputs = torch.tensor([1.0, 0.5, -0.5])
        targets = torch.tensor([0.0, 0.0, 0.0])

        result = loss_fn(outputs, targets)

        is_finite = torch.isfinite(result).item() if isinstance(result, torch.Tensor) else math.isfinite(result)
        assert is_finite, (
            f"{loss_fn.__name__} returned non-finite result ({result}) on all-zero targets — "
            f"this is unexpected since all-zero targets are a valid uniform distribution."
        )
        print(f"  {loss_fn.__name__} with all-zero targets → finite ({result})")
