"""Model-cache validation. Run BEFORE dispatching a job, never after.

WHY THIS IS NOT PARANOIA
------------------------
`.model_cache` is unreliable in worktrees:

  * Its root is derived from `catalog_utils.__file__`, not the cwd, so every
    worktree has a *separate* cache and "it works in the main checkout" tells
    you nothing about a worktree.
  * It is only partially tracked in git (237 files committed, 64 present only
    as untracked files in the main checkout), so a fresh worktree is missing
    models that appear to exist.
  * A miss re-downloads SILENTLY, except `dbn/*`, whose host is unreachable and
    hard-fails. `pyGMs`' downloader calls `requests.get()` with NO TIMEOUT, so a
    miss can hang a job indefinitely rather than failing it.
  * At least one cached `.uai` was found to be an **HTML error page** that had
    been saved under the model's filename. It is the right size to look
    plausible and it parses as "a file that exists".

So validation checks that the file exists, is readable, and actually looks like
a UAI model -- not merely that `os.path.exists` is true. A job whose model fails
validation is marked BLOCKED and never dispatched, because the alternative is a
worker that hangs on a no-timeout HTTP GET while holding a GPU lease.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

# First token of a UAI file must be one of these network types.
_UAI_TYPES = ('MARKOV', 'BAYES', 'WCSP', 'MPE')

# An HTML error page saved as a .uai starts with one of these once stripped.
_HTML_MARKERS = ('<!doctype', '<html', '<head', '<?xml', '<!DOCTYPE')


def cache_root() -> str:
    """The cache directory THIS interpreter will use (worktree-dependent).

    MUST agree with `catalog_utils.get_catalog`, which is what actually loads
    the model. It resolves `NCE_MODEL_CACHE` first and only then falls back to
    the path derived from its own `__file__`. Recomputing just the fallback
    here made validation disagree with loading the moment `.model_cache` became
    untracked (it is gitignored on the CRN lineage), so every job in a fresh
    worktree was BLOCKED as "model missing" while the loader would have found
    it perfectly well via the override.
    """
    env = os.environ.get('NCE_MODEL_CACHE')
    if env:
        # Resolve WITHOUT importing catalog_utils. Importing it pulls in
        # nce.benchmark_problems.__init__, which eagerly builds benchmark sets
        # and therefore needs a working cache -- so validating a broken cache
        # would itself crash on the broken cache. Chicken and egg; the env var
        # is authoritative anyway, exactly as in catalog_utils.get_catalog.
        return env
    from nce.benchmark_problems import catalog_utils
    return catalog_utils._DEFAULT_CACHE


def assert_cache_configured(root: str = None) -> str:
    """Fail NOW, naming the variable, if the model cache is not usable.

    `.model_cache` is untracked and gitignored on this lineage, so a fresh
    worktree has an empty one and `dbn/*` cannot be re-downloaded. Without this
    check a sweep starts happily and every job fails ~8 s later with a per-job
    "model missing" -- which, launched unattended overnight, means waking up to
    a queue of BLOCKED jobs and no runs. The whole point is to make forgetting
    `NCE_MODEL_CACHE` impossible rather than merely documented.

    Returns the resolved root so callers can log it.
    """
    root = root or cache_root()
    env_set = bool(os.environ.get('NCE_MODEL_CACHE'))
    hint = ('NCE_MODEL_CACHE is NOT set, so the cache resolved to the path '
            'derived from the package location.'
            if not env_set else
            'NCE_MODEL_CACHE is set to %r.' % os.environ.get('NCE_MODEL_CACHE'))

    if not os.path.isdir(root):
        raise RuntimeError(
            'Model cache directory does not exist: %s\n  %s\n'
            '  Set NCE_MODEL_CACHE to a populated cache, e.g.\n'
            '      export NCE_MODEL_CACHE=/home/cohenn1/NCE/.model_cache\n'
            '  Note dbn/* cannot be re-downloaded, so never delete or '
            'symlink over a populated cache.' % (root, hint))

    n_uai = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        n_uai += sum(1 for f in filenames if f.endswith('.uai'))
        if n_uai:
            break
    if n_uai == 0:
        raise RuntimeError(
            'Model cache at %s contains no .uai files.\n  %s\n'
            '  Set NCE_MODEL_CACHE to a populated cache, e.g.\n'
            '      export NCE_MODEL_CACHE=/home/cohenn1/NCE/.model_cache\n'
            '  Note dbn/* cannot be re-downloaded.' % (root, hint))
    return root


def model_paths(problem_key: str, root: str = None) -> Dict[str, str]:
    """Expected local paths for a problem key, WITHOUT touching the network.

    Deliberately built by string manipulation rather than by asking pyGMs,
    because `Model.file` / `.order` / `.evidence` are lazy properties whose
    getter is the download trigger.
    """
    root = root or cache_root()
    category, _, name = problem_key.rpartition('/')
    base = os.path.join(root, category, name)
    return {
        'uai': base + '.uai',
        'order': base + '.uai.ord',
        'evidence': base + '.uai.evid',
    }


def _looks_like_uai(path: str) -> Tuple[bool, str]:
    """Read the head of the file and decide whether it is really a UAI model."""
    try:
        with open(path, 'rb') as fh:
            head = fh.read(512)
    except OSError as e:
        return False, 'unreadable: %s' % e
    if not head.strip():
        return False, 'file is empty'
    try:
        text = head.decode('utf-8', errors='replace')
    except Exception as e:  # pragma: no cover
        return False, 'undecodable: %s' % e
    stripped = text.lstrip()
    low = stripped.lower()
    for marker in _HTML_MARKERS:
        if low.startswith(marker.lower()):
            return False, ('file is an HTML/XML page, not a model -- this is a '
                           'saved download error, delete it and re-fetch')
    first = stripped.split()[0] if stripped.split() else ''
    if first.upper() not in _UAI_TYPES:
        return False, ('first token is %r, expected one of %s'
                       % (first[:32], '/'.join(_UAI_TYPES)))
    return True, 'ok'


def validate(problem_key: str, root: str = None,
             require_order: bool = True) -> Dict[str, Any]:
    """Full pre-dispatch validation for one problem.

    `require_order` (.uai.ord): REQUIRED by default. FastGM falls back to
    computing an elimination order itself when the file is absent, and a rerun
    that silently uses a different elimination order than the original is
    exactly the divergence this effort exists to prevent.

    `.uai.evid` is NOT required: it is genuinely absent for many models in the
    real cache (e.g. grids/grid10x10.f10 has none in either checkout), so
    demanding it would block valid jobs. Its presence is reported, not enforced.
    """
    paths = model_paths(problem_key, root)
    problems: List[str] = []
    ok_uai, why = (False, 'missing')
    if os.path.isfile(paths['uai']):
        ok_uai, why = _looks_like_uai(paths['uai'])
        if not ok_uai:
            problems.append('%s: %s' % (paths['uai'], why))
    else:
        problems.append('%s: missing (a dispatch would trigger a silent '
                        're-download, or hang -- pyGMs uses requests.get with '
                        'no timeout)' % paths['uai'])

    if require_order and not os.path.isfile(paths['order']):
        problems.append('%s: missing elimination-order sidecar -- FastGM would '
                        'silently compute its own order, which may differ from '
                        'the original run' % paths['order'])

    if problem_key.startswith('dbn/') and problems:
        problems.append('NOTE: dbn/* cannot be re-downloaded -- its host is '
                        'unreachable and hard-fails. This must be fixed by '
                        'copying the file in, not by rerunning.')

    return {
        'problem_key': problem_key,
        'paths': paths,
        'uai_valid': ok_uai,
        'uai_check': why,
        'size_bytes': (os.path.getsize(paths['uai'])
                       if os.path.isfile(paths['uai']) else None),
        'has_evidence': os.path.isfile(paths['evidence']),
        'ok': not problems,
        'problems': problems,
    }


def validate_many(problem_keys: List[str], root: str = None) -> Dict[str, Any]:
    results = {k: validate(k, root) for k in sorted(set(problem_keys))}
    return {
        'cache_root': root or cache_root(),
        'ok': [k for k, v in sorted(results.items()) if v['ok']],
        'bad': {k: v['problems'] for k, v in sorted(results.items())
                if not v['ok']},
        'results': results,
    }
