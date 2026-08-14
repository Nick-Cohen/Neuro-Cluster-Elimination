"""The torch deterministic-algorithms guard, in WARNING mode.

WHAT IT DOES
------------
`torch.use_deterministic_algorithms(True, warn_only=True)` makes torch prefer a
deterministic kernel wherever one exists, and emit a `UserWarning` (instead of
raising) at the operations that have no deterministic implementation. It is a
detector for a class of nondeterminism the goldens in
`tests/test_determinism_regression.py` cannot see by construction: a kernel
whose output depends on scheduling would make the goldens flap, but nothing in
the codebase would say WHERE.

WHY WARNING MODE AND NOT STRICT MODE
------------------------------------
Strict mode (`warn_only=False`) raises. On this box, with torch 2.0.1+cu117, it
raises on the very first `F.linear` unless `CUBLAS_WORKSPACE_CONFIG` is set in
the environment BEFORE the process starts (measured in doc 28 section 4.3;
doc 21 had recorded it as raising nothing, which was wrong). A guard that dies
at the first linear layer reports nothing about the rest of the pipeline. Warn
mode reaches the end of the run and reports every offending op.

WHY IT IS OPT-IN AND NOT ON BY DEFAULT
--------------------------------------
Kernel SELECTION is the same in warn mode as in strict mode -- only the
reaction to a missing deterministic kernel differs -- so the cost is the same:
doc 28 measured 2.52x wall time for one CUDA case. Turning that on by default
would multiply the paper sweep's runtime for a diagnostic almost every run does
not need.

HOW TO TURN IT ON
-----------------
    NCE_DETERMINISM_GUARD=1 python your_run.py       # environment
    config['deterministic_guard'] = True             # or per-run config

`FastGM.__init__` calls `maybe_enable_from_config` once per model build; the
call is idempotent within a process.
"""
import os
import warnings

import torch

_ENV_VAR = 'NCE_DETERMINISM_GUARD'
_ENABLED = False


def guard_requested(config=None):
    """True when the env var or the config asks for the guard."""
    if os.environ.get(_ENV_VAR) == '1':
        return True
    if config is not None and config.get('deterministic_guard', False):
        return True
    return False


def is_enabled():
    return _ENABLED


def enable_determinism_guard(warn_only=True, verbose=True):
    """Turn the guard on. Idempotent; returns True if it is now on.

    `warn_only=False` is accepted but will raise on CUDA without
    CUBLAS_WORKSPACE_CONFIG -- see the module docstring. It exists so the
    det-algos probe in the regression suite can share this entry point.
    """
    global _ENABLED
    if _ENABLED:
        return True
    if not warn_only and torch.cuda.is_available() and \
            os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
        raise RuntimeError(
            'enable_determinism_guard(warn_only=False) on CUDA requires '
            'CUBLAS_WORKSPACE_CONFIG=:4096:8 in the environment BEFORE the '
            'process starts; torch reads it at cuBLAS handle creation. Use '
            'warn_only=True (the default) if you cannot set it.')
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    _ENABLED = True
    if verbose:
        print('[determinism] torch.use_deterministic_algorithms(True, warn_only=%s) '
              'is ON. Nondeterministic ops will emit UserWarning; expect ~2.5x '
              'slower (doc 28).' % warn_only)
    return True


def disable_determinism_guard():
    """Only for tests -- restores the process-global default."""
    global _ENABLED
    torch.use_deterministic_algorithms(False)
    _ENABLED = False


def maybe_enable_from_config(config, verbose=True):
    """Entry point used by FastGM. Returns True if the guard is on afterwards."""
    if guard_requested(config):
        return enable_determinism_guard(warn_only=True, verbose=verbose)
    return _ENABLED


class collect_nondeterministic_warnings:
    """Context manager collecting the warnings the guard raises.

    Usage:
        with collect_nondeterministic_warnings() as seen:
            ...
        print(seen.messages)     # one string per nondeterministic op hit
    """

    def __enter__(self):
        self.messages = []
        self._ctx = warnings.catch_warnings(record=True)
        self._log = self._ctx.__enter__()
        warnings.simplefilter('always')
        return self

    def __exit__(self, *exc):
        self.messages = [str(w.message) for w in self._log
                         if 'does not have a deterministic implementation' in str(w.message)]
        return self._ctx.__exit__(*exc)
