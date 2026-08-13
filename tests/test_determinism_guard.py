"""The `torch.use_deterministic_algorithms(..., warn_only=True)` guard.

Requirement (Q51, option A): "use the deterministic-algorithms guard in warning
mode". Implementation and the argument for warning mode over strict mode:
`nce/utils/determinism.py`.

Every test here restores the process-global flag on the way out. It is genuinely
global: leaving it on would slow the rest of the session by ~2.5x and could
change which kernels other tests run.
"""
import os

import pytest
import torch

from nce.config_schema import prepare_config
from nce.inference.graphical_model import FastGM
from nce.utils import determinism


@pytest.fixture(autouse=True)
def _restore_global_flag():
    was_on = torch.are_deterministic_algorithms_enabled()
    yield
    determinism.disable_determinism_guard()
    if was_on:                              # pragma: no cover - not expected
        torch.use_deterministic_algorithms(True, warn_only=True)


def test_guard_enables_warning_mode_not_strict_mode():
    assert not torch.are_deterministic_algorithms_enabled()
    assert determinism.enable_determinism_guard(verbose=False)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.is_deterministic_algorithms_warn_only_enabled(), (
        'the guard turned on STRICT mode. On CUDA that raises on the first '
        'F.linear unless CUBLAS_WORKSPACE_CONFIG is set before process start '
        '(doc 28 section 4.3), so a run would die before reaching anything '
        'interesting.')
    assert determinism.is_enabled()


def test_guard_is_idempotent():
    determinism.enable_determinism_guard(verbose=False)
    determinism.enable_determinism_guard(verbose=False)
    assert torch.are_deterministic_algorithms_enabled()


def test_guard_is_off_by_default():
    """It costs ~2.5x wall time; the paper sweep must not pay that by accident."""
    assert not determinism.guard_requested(None)
    assert not determinism.guard_requested({'device': 'cuda'})
    assert not torch.are_deterministic_algorithms_enabled()


def test_guard_requested_from_config_and_env(monkeypatch):
    assert determinism.guard_requested({'deterministic_guard': True})
    monkeypatch.setenv('NCE_DETERMINISM_GUARD', '1')
    assert determinism.guard_requested({})
    monkeypatch.setenv('NCE_DETERMINISM_GUARD', '0')
    assert not determinism.guard_requested({})


def test_strict_mode_refuses_without_cublas_workspace_config(monkeypatch):
    """Fail loudly rather than blow up mid-run inside a linear layer."""
    if not torch.cuda.is_available():
        pytest.skip('the CUBLAS precondition only applies on CUDA')
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    with pytest.raises(RuntimeError, match='CUBLAS_WORKSPACE_CONFIG'):
        determinism.enable_determinism_guard(warn_only=False, verbose=False)


def test_fastgm_turns_the_guard_on_when_the_config_asks(binary_chain_factors,
                                                        nn_training_config):
    """The wiring, not just the helper.

    A guard nothing calls is not a guard. This builds a real FastGM with
    `deterministic_guard=True` and asserts the process flag flipped.
    """
    cfg = prepare_config({**nn_training_config, 'ecl': 2 ** 30, 'iB': 100,
                          'dope_factors': False, 'deterministic_guard': True})
    assert not torch.are_deterministic_algorithms_enabled()
    FastGM(factors=binary_chain_factors['factors'],
           elim_order=binary_chain_factors['elim_order'],
           nn_config=cfg, device='cpu')
    assert torch.are_deterministic_algorithms_enabled(), (
        'FastGM did not enable the determinism guard despite '
        "config['deterministic_guard'] = True -- the call in FastGM.__init__ "
        'was removed or moved after an early return.')
    assert torch.is_deterministic_algorithms_warn_only_enabled()


def test_fastgm_leaves_the_guard_off_by_default(binary_chain_factors,
                                                nn_training_config):
    cfg = prepare_config({**nn_training_config, 'ecl': 2 ** 30, 'iB': 100,
                          'dope_factors': False})
    FastGM(factors=binary_chain_factors['factors'],
           elim_order=binary_chain_factors['elim_order'],
           nn_config=cfg, device='cpu')
    assert not torch.are_deterministic_algorithms_enabled()


@pytest.mark.gpu
def test_nondeterministic_cuda_op_warns_instead_of_raising():
    """Warning mode must actually WARN, not raise, and not silently pass.

    Probe choice was MEASURED on torch 2.0.1+cu117: of `bincount`, `put_`,
    `kthvalue`, `index_add_`, `scatter_add_`, `index_put_(accumulate=True)` and
    `median` on CUDA, only the first three actually warn. `index_add_` is
    documented as nondeterministic but is silent on this build, so picking it
    "because the docs list it" would have produced a test that fails for the
    wrong reason. Without a probe like this, `warn_only=True` could be silently
    degrading to "no checks at all" and nothing would notice.
    """
    if not torch.cuda.is_available():
        pytest.skip('no CUDA device visible')
    determinism.enable_determinism_guard(verbose=False)
    with determinism.collect_nondeterministic_warnings() as seen:
        torch.bincount(torch.tensor([0, 1, 1, 2], device='cuda'))
        torch.cuda.synchronize()
    assert seen.messages, (
        'torch.bincount on CUDA produced no nondeterminism warning under '
        'use_deterministic_algorithms(True, warn_only=True). Either torch gained '
        'a deterministic implementation (fine -- pick another probe from the '
        'measured list in this docstring) or the guard is not in force.')
