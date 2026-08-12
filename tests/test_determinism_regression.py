"""Bit-exact regression suite for the variable-elimination pipeline.

WHAT THIS PINS AND WHY IT IS POSSIBLE AT ALL
--------------------------------------------
Until 2026-08-12 no change to `nce/` could be verified by measurement: two
identical runs of the same config returned different answers. Root cause
(notebooks/_August-2026/claude_experiments/21-determinism.md):
`FastGM._create_buckets_from_factors` called `set(factors)` on `FastFactor`,
which defines neither `__hash__` nor `__eq__`, so bucket factor order followed
memory addresses. That is the association order of a log-space factor product,
so every message value moved by ~3-4 float32 ULP -- and 275+ SGD epochs amplify
that into a genuinely different model (doc 21 section 2.2). A second instance
lived in `pygms_wmb_interface.get_backward_message`. Both are fixed on
`fix/determinism` @ `d5cca0e`.

This file turns that into a guard. Every golden below was MEASURED by running
the case; none was copied from a document. Provenance (branch, commit, device,
GPU model, torch build, date, thread count) is stored per case in
`tests/goldens/determinism_goldens.json` and reprinted in every failure message,
because a golden is only meaningful relative to a code state.

TIERS
-----
CPU tier (default, no flags)   ~35 s wall, single-threaded, no GPU.
                               Run it on every change.
CUDA tier (`--gpu`)            ~7 min wall on one TITAN RTX.
                               Run it on changes that touch the numerics:
                               factors, buckets, sampling, losses, training.
det-algos probe (`--det-algos`)  ~1 min extra. Re-runs one CUDA case in a
                               subprocess under
                               `torch.use_deterministic_algorithms(True)`.

    python -m pytest tests/test_determinism_regression.py -v
    CUDA_VISIBLE_DEVICES=3 python -m pytest tests/test_determinism_regression.py -v --gpu
    CUDA_VISIBLE_DEVICES=3 python -m pytest tests/test_determinism_regression.py -v --det-algos

Regenerate goldens after an intended numerical change (writes provenance too):

    python tests/test_determinism_regression.py --regen cpu
    CUDA_VISIBLE_DEVICES=3 python tests/test_determinism_regression.py --regen cuda

CUBLAS_WORKSPACE_CONFIG -- READ BEFORE TOUCHING THE det-algos PROBE
-------------------------------------------------------------------
`torch.use_deterministic_algorithms(True)` DOES raise on CUDA on this box --
on the very first `F.linear` -- unless `CUBLAS_WORKSPACE_CONFIG` is set
(doc 28 section 4.3; doc 21 recorded it as raising nothing, which was wrong).
A guard that does not set it dies before reaching any interesting kernel and so
tests nothing. `require_cublas_workspace_config()` below therefore raises a
loud, explicit error instead of letting the probe silently pass or fail for the
wrong reason, and the subprocess is launched with the variable set. MEASURED
cost of the deterministic path: 2.52x wall time (doc 28), so it is opt-in.

WHAT THIS SUITE DOES **NOT** PROTECT AGAINST
--------------------------------------------
Stated conservatively; each item is untested here, not known-good.
  * grid20x20 / grid40x40; the `sub*`, `nomerge` and `merge_degree` arms;
    the decision-tree path; multi-set (`s > 0`) training; `use_memorizer`.
  * Multi-threaded CPU. MEASURED here: the pedigree CPU case returns
    -10.624578475952148 at 1 thread and -10.624576568603516 at 4 and at 8
    threads (2 float32 ULP apart) -- reproducibly so, but a *different* value.
    The CPU goldens are therefore valid only at `torch.set_num_threads(1)`, and
    `test_cpu_tier_is_single_threaded` fails loudly if that is not in force.
  * The float `scatter_add_` sampler paths (`no_replacement_sampler_v2.py:121`,
    `no_replacement_sampler_multilevel_lean.py:256`). No case here reaches them.
  * Cross-GPU reproducibility. Every CUDA golden was measured on one TITAN RTX;
    the CUDA tests SKIP (they do not pass) on a different GPU model.
  * Concurrency: everything here is fixed-seed, single-process, one run at a time.
  * Absolute correctness. These are equality-to-a-recorded-value tests. A change
    that makes the pipeline uniformly wrong in a reproducible way passes.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from nce.config_schema import prepare_config
from nce.inference.bucket import FastBucket
from nce.inference.graphical_model import FastGM

GOLDENS_PATH = Path(__file__).parent / 'goldens' / 'determinism_goldens.json'


# ---------------------------------------------------------------------------
# Case definitions. `config` is passed verbatim to prepare_config after
# popping problem_key, so a case is exactly reproducible from this file.
# ---------------------------------------------------------------------------
_CPU_BASE = dict(neurobe_mode=True, ecl=1025, sampling_scheme='uniform',
                 stream_nn_exact=True, device='cpu', seed=42,
                 use_reduce_nn_merge=True, reduce_nn_backtrack=True)

CASES = {
    # ---------------- CPU tier -------------------------------------------
    'cpu_grid_plain': dict(
        tier='cpu',
        covers='plain binary grid; reduce-NN merge (D=4, backtracking); doped; '
               '1 NN cluster; nbe sample budget; unchunked exact path',
        config=dict(_CPU_BASE, problem_key='grids/grid10x10.f10', iB=10,
                    max_merge_bound=4, dope_factors=True, num_samples='nbe,0.1',
                    num_epochs=10)),
    'cpu_grid_merged': dict(
        tier='cpu',
        covers='wrapped grid (induced width 21 vs 12); reduce-NN merge D=8 -> '
               'genuinely merged clusters; 3 NN clusters',
        config=dict(_CPU_BASE, problem_key='grids/grid10x10.f10.wrap', iB=10,
                    max_merge_bound=8, dope_factors=True, num_samples=2048,
                    num_epochs=5)),
    'cpu_pedigree_plain': dict(
        tier='cpu',
        covers='real pedigree (mixed domain sizes k in {2,3,4}, 334 vars); '
               '10 NN clusters -> 10 independent early-stopping decisions',
        config=dict(_CPU_BASE, problem_key='pedigree/pedigree1', iB=8,
                    max_merge_bound=6, dope_factors=True, num_samples=2048,
                    num_epochs=5)),
    'cpu_pedigree_masked': dict(
        tier='cpu',
        covers='masked_net arm (two-head trunk, BCE mask loss, boolean-mask '
               'indexing backward, no AMP) on undoped factors so -inf targets '
               'are exercised; mixed domains; 5 NN clusters',
        config=dict(_CPU_BASE, problem_key='pedigree/pedigree1', iB=10,
                    max_merge_bound=8, dope_factors=False, masked_net=True,
                    num_samples=2048, num_epochs=5)),
    'cpu_local_error': dict(
        tier='cpu',
        covers='compute_local_error=True -> FastGM.local_errors is POPULATED, '
               'which also exercises get_backward_message (the second site of '
               'the identity-hash bug) and an exact WMB backward at iB=100',
        config=dict(_CPU_BASE, problem_key='grids/grid10x10.f10', iB=10,
                    max_merge_bound=4, dope_factors=True, num_samples='nbe,0.1',
                    num_epochs=10, compute_local_error=True)),

    # ---------------- CUDA tier ------------------------------------------
    # Verbatim configs from the completed reduce-NN study
    # (notebooks/June-2026/claude_experiments/reduce_nn_experiment/configs/),
    # so a failure here is directly comparable to the study's own numbers.
    'cuda_grid_plain': dict(
        tier='cuda',
        covers='study cell grid10x10f10_iB10_rnn4_s42; plain grid, 1 NN cluster, '
               'full 500-epoch cap so early stopping actually binds',
        config=dict(problem_key='grids/grid10x10.f10', neurobe_mode=True, iB=10,
                    ecl=1025, num_samples='nbe,0.1', sampling_scheme='uniform',
                    stream_nn_exact=True, dope_factors=True, device='cuda',
                    seed=42, use_reduce_nn_merge=True, max_merge_bound=4,
                    reduce_nn_backtrack=True)),
    'cuda_grid_merged': dict(
        tier='cuda',
        covers='study cell grid10x10f10wrap_iB10_rnn8_s42; wrapped grid, merged '
               'clusters, 3 NN clusters',
        config=dict(problem_key='grids/grid10x10.f10.wrap', neurobe_mode=True,
                    iB=10, ecl=1025, num_samples='nbe,0.1',
                    sampling_scheme='uniform', stream_nn_exact=True,
                    dope_factors=True, device='cuda', seed=42,
                    use_reduce_nn_merge=True, max_merge_bound=8,
                    reduce_nn_backtrack=True)),
    'cuda_rbm': dict(
        tier='cuda',
        covers='study cell rbm_21_iB20_rnn10_s42; RBM family, iB=20, 12 NN '
               'clusters -- the cell doc 23 measured as most artefact-sensitive',
        config=dict(problem_key='dbn/rbm_21', neurobe_mode=True, iB=20,
                    ecl=1048577, num_samples='nbe,0.1', sampling_scheme='uniform',
                    stream_nn_exact=True, dope_factors=True, device='cuda',
                    seed=42, use_reduce_nn_merge=True, max_merge_bound=10,
                    reduce_nn_backtrack=True)),
    'cuda_pedigree_masked': dict(
        tier='cuda',
        covers='study cell pedigree19_iB20_rnn16_masked_s42; pedigree (max '
               'domain 5, induced width 27), masked_net, undoped, iB=20',
        config=dict(problem_key='pedigree/pedigree19', neurobe_mode=True, iB=20,
                    ecl=1048577, num_samples='nbe,0.1', sampling_scheme='uniform',
                    stream_nn_exact=True, dope_factors=False, masked_net=True,
                    device='cuda', seed=42, use_reduce_nn_merge=True,
                    max_merge_bound=16, reduce_nn_backtrack=True)),
}

CPU_CASES = [k for k, v in CASES.items() if v['tier'] == 'cpu']
CUDA_CASES = [k for k, v in CASES.items() if v['tier'] == 'cuda']

# Case re-run inside the tier to prove bit-reproducibility independently of the
# goldens (so a legitimately-updated golden cannot hide a new ordering bug).
CPU_REPLICATE_CASE = 'cpu_grid_plain'
CUDA_REPLICATE_CASE = 'cuda_grid_plain'
DET_ALGOS_CASE = 'cuda_grid_plain'

# Cases whose per-bucket factor ORDER is digested at build time. This is the
# cheapest possible reproducer of the doc-21 bug: build-only, no training.
ORDER_CASES = ['cpu_grid_plain', 'cpu_pedigree_plain', 'cpu_pedigree_masked']


# ---------------------------------------------------------------------------
# Environment guards
# ---------------------------------------------------------------------------
def require_cublas_workspace_config():
    """Raise unless the cuBLAS workspace is pinned.

    MEASURED (doc 28 section 4.3): on this box + torch 2.0.1+cu117,
    `torch.use_deterministic_algorithms(True)` raises RuntimeError on the first
    `F.linear` when CUBLAS_WORKSPACE_CONFIG is unset. A determinism guard that
    does not set it therefore dies before touching anything it meant to test.
    """
    val = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if val not in (':4096:8', ':16:8'):
        raise RuntimeError(
            'CUBLAS_WORKSPACE_CONFIG is %r. torch.use_deterministic_algorithms(True) '
            'raises on the first F.linear on CUDA without it, so this probe would '
            'fail for the wrong reason and prove nothing. Set '
            'CUBLAS_WORKSPACE_CONFIG=:4096:8 in the environment BEFORE the process '
            'starts (torch reads it at cuBLAS handle creation).' % (val,))


@pytest.fixture(scope='session', autouse=True)
def _pin_cpu_threads():
    """CPU goldens are only valid single-threaded -- see the module docstring."""
    torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _load_model(problem_key):
    from nce.benchmark_problems.catalog_utils import get_catalog
    return get_catalog()[problem_key]


def run_case(name, chunk_limit=None, chunk_block=None):
    """Build and fully eliminate one case. Returns the observable outcome dict.

    `chunk_limit` / `chunk_block` lower FastBucket._EXACT_JOINT_NUMEL_LIMIT and
    the chunk size so the memory-bounded exact path is forced on a small
    problem; the surrounding code is otherwise untouched.
    """
    cfg = dict(CASES[name]['config'])
    problem_key = cfg.pop('problem_key')
    full = prepare_config(cfg, strict=False)

    orig_limit = FastBucket._EXACT_JOINT_NUMEL_LIMIT
    orig_chunked = FastBucket._compute_message_exact_chunked
    chunk_calls = []
    if chunk_limit is not None:
        FastBucket._EXACT_JOINT_NUMEL_LIMIT = chunk_limit

        def _counting(self, scope, elim_labels, block_limit=chunk_block):
            chunk_calls.append(self.label)
            return orig_chunked(self, scope, elim_labels, block_limit=chunk_block)

        FastBucket._compute_message_exact_chunked = _counting
    try:
        t0 = time.time()
        gm = FastGM(model=_load_model(problem_key), nn_config=full,
                    device=full['device'])
        gm.eliminate_variables(all=True)
        wall = time.time() - t0
    finally:
        FastBucket._EXACT_JOINT_NUMEL_LIMIT = orig_limit
        FastBucket._compute_message_exact_chunked = orig_chunked

    epochs = [d.get('epochs_trained')
              for d in getattr(gm, 'per_bucket_training_log', [])
              if d.get('epochs_trained') is not None]
    local_errors = [repr(r.get('signed_local_error'))
                    for r in getattr(gm, 'local_errors', [])]
    return {
        'log_z_repr': repr(float(gm.log_partition_function)),
        'epochs': epochs,
        'n_nn_clusters': len(epochs),
        'local_errors': local_errors,
        'n_chunked_calls': len(chunk_calls),
        'wall_s': round(wall, 2),
    }


def build_case(name):
    """Build the FastGM only -- no elimination, no training."""
    cfg = dict(CASES[name]['config'])
    problem_key = cfg.pop('problem_key')
    full = prepare_config(cfg, strict=False)
    return FastGM(model=_load_model(problem_key), nn_config=full,
                  device=full['device'])


def bucket_factor_digest(gm, order_sensitive=True):
    """Digest every bucket's factor list.

    order_sensitive=True is the quantity the doc-21 bug moved; the
    order-insensitive digest is the control that stayed constant throughout
    (doc 21 section 1.3), and proves no factor changed bucket.
    """
    outer = hashlib.sha256()
    for key in sorted(gm.buckets, key=lambda v: getattr(v, 'label', v)):
        bucket = gm.buckets[key]
        parts = []
        for f in bucket.factors:
            inner = hashlib.sha256(repr(list(f.labels)).encode())
            tensor = getattr(f, 'tensor', None)
            if tensor is None:
                inner.update(b'<nn>' + repr(type(f).__name__).encode())
            else:
                inner.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            parts.append(inner.hexdigest())
        if not order_sensitive:
            parts = sorted(parts)
        outer.update(str(getattr(key, 'label', key)).encode())
        outer.update(''.join(parts).encode())
    return outer.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Golden storage + diagnosis
# ---------------------------------------------------------------------------
def _load_goldens():
    if not GOLDENS_PATH.exists():
        pytest.fail('missing golden file %s -- regenerate with '
                    '`python tests/test_determinism_regression.py --regen cpu`'
                    % GOLDENS_PATH)
    with open(GOLDENS_PATH) as fh:
        return json.load(fh)


def _gpu_name():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(0)


def _float32_ulps(a, b):
    """Signed float32 ULP distance; None when either side is not finite."""
    import numpy as np
    fa, fb = np.float32(a), np.float32(b)
    if not (np.isfinite(fa) and np.isfinite(fb)):
        return None
    ia = np.array([fa]).view(np.int32)[0].astype(np.int64)
    ib = np.array([fb]).view(np.int32)[0].astype(np.int64)
    flip = lambda i: (np.int64(-2 ** 31) - i) if i < 0 else i  # noqa: E731
    return int(flip(ib) - flip(ia))


def _provenance_line(prov):
    return ('branch %s @ %s | %s | torch %s | threads %s | %s | measured %s'
            % (prov.get('branch'), prov.get('commit'), prov.get('device_name'),
               prov.get('torch'), prov.get('threads'), prov.get('host'),
               prov.get('date')))


def _report(case, got, gold, prov):
    """Human-diagnosable mismatch report. A bare float assertion is useless."""
    lines = ['', 'GOLDEN MISMATCH: %s' % case,
             '  covers: %s' % CASES[case]['covers']]

    g_lz, a_lz = gold['log_z_repr'], got['log_z_repr']
    if g_lz != a_lz:
        gv, av = float(g_lz), float(a_lz)
        delta = av - gv
        rel = abs(delta) / max(abs(gv), 1e-30)
        ulps = _float32_ulps(gv, av)
        lines += ['  log Z',
                  '    golden  %s' % g_lz,
                  '    actual  %s' % a_lz,
                  '    delta   %+.6e   relative %.3e   float32 ULP %s'
                  % (delta, rel, 'n/a' if ulps is None else ulps),
                  '    reading: <~10 ULP is a re-association of the same sum '
                  '(a numerics-neutral refactor moved an accumulation order);',
                  '             a large delta is a behaviour change.']
    else:
        lines.append('  log Z   MATCHES (%s)' % g_lz)

    ge, ae = gold['epochs'], got['epochs']
    if ge != ae:
        lines.append('  per-bucket epochs  (%d golden vs %d actual clusters)'
                     % (len(ge), len(ae)))
        if len(ge) != len(ae):
            lines += ['    CLUSTER COUNT CHANGED -- the merge/partition decision '
                      'moved, not just the arithmetic.',
                      '    golden  %r' % (ge,), '    actual  %r' % (ae,)]
        else:
            diffs = [(i, g, a) for i, (g, a) in enumerate(zip(ge, ae)) if g != a]
            lines.append('    %d of %d buckets differ; first at index %d '
                         '(golden %d -> actual %d)'
                         % (len(diffs), len(ge), diffs[0][0], diffs[0][1], diffs[0][2]))
            lines.append('    all diffs: %s'
                         % ', '.join('[%d] %d->%d' % d for d in diffs[:12]))
            lines.append('    reading: early stopping is a knife edge past epoch '
                         '~100 (doc 21 section 2.2), so a moved stopping epoch is '
                         'the most sensitive detector of an upstream bit change.')
    else:
        lines.append('  per-bucket epochs   MATCH (%r)' % (ae,))

    gl, al = gold.get('local_errors', []), got.get('local_errors', [])
    if gl != al:
        lines += ['  local_errors', '    golden  %r' % (gl,), '    actual  %r' % (al,)]
    elif gl:
        lines.append('  local_errors        MATCH (%d records)' % len(gl))

    lines += ['  golden provenance: %s' % _provenance_line(prov),
              '  this run:          %s' % _provenance_line(_current_provenance(
                  CASES[case]['config'].get('device', 'cpu'))),
              '  If the change was intended, re-measure with:',
              '    python tests/test_determinism_regression.py --regen %s'
              % CASES[case]['tier'],
              '  and record in the commit message WHAT moved and by how much.', '']
    return '\n'.join(lines)


def _git(args):
    try:
        return subprocess.run(['git'] + args, cwd=str(Path(__file__).resolve().parents[1]),
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return 'unknown'


def _current_provenance(device):
    return {
        'branch': _git(['rev-parse', '--abbrev-ref', 'HEAD']),
        'commit': _git(['rev-parse', '--short', 'HEAD']),
        'dirty': bool(_git(['status', '--porcelain', '--', 'nce'])),
        'device': device,
        'device_name': (_gpu_name() if device == 'cuda' else 'cpu'),
        'torch': torch.__version__,
        'threads': torch.get_num_threads(),
        'host': os.uname().nodename,
        'python': sys.version.split()[0],
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
    }


# ---------------------------------------------------------------------------
# CPU tier
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def goldens():
    return _load_goldens()


@pytest.fixture(scope='module')
def cpu_results():
    """Every CPU case, run once. MEASURED: results are independent of the order
    the cases run in within a process (checked forwards and reversed), so a
    module-scope cache is safe."""
    return {name: run_case(name) for name in CPU_CASES}


def test_cpu_tier_is_single_threaded():
    """CPU goldens are 1-thread values -- fail loudly rather than pass by luck.

    MEASURED 2026-08-12: cpu_pedigree_plain gives -10.624578475952148 at 1
    thread and -10.624576568603516 at 4 and at 8 threads. Both are stable
    run-to-run at their own thread count; they are not the same number.
    """
    assert torch.get_num_threads() == 1, (
        'torch.get_num_threads() == %d. The CPU goldens in %s were measured at 1 '
        'thread and differ by 2 float32 ULP at higher thread counts; comparing '
        'against them multi-threaded tests nothing. The session fixture '
        '_pin_cpu_threads should have set this.'
        % (torch.get_num_threads(), GOLDENS_PATH.name))


@pytest.mark.parametrize('case', ORDER_CASES)
def test_bucket_factor_order_is_stable_across_builds(case, goldens):
    """The doc-21 bug, reduced to its cheapest reproducer (build only).

    Pre-fix, two builds *inside one process* produced different per-bucket
    factor orders (doc 21 section 1.3: order digests a40f6368 / 1fd7a571 in one
    process) while the order-INSENSITIVE digest was constant. Any future
    `set()`/`dict` over an identity-hashed NCE object reintroduces exactly that,
    and this test costs under a second.
    """
    gm_a, gm_b = build_case(case), build_case(case)
    ord_a, ord_b = bucket_factor_digest(gm_a), bucket_factor_digest(gm_b)
    set_a = bucket_factor_digest(gm_a, order_sensitive=False)
    set_b = bucket_factor_digest(gm_b, order_sensitive=False)

    assert set_a == set_b, (
        '%s: the SET of factors per bucket differs between two builds in one '
        'process (%s vs %s). That is worse than the ordering bug -- factors are '
        'landing in different buckets.' % (case, set_a, set_b))
    assert ord_a == ord_b, (
        '%s: bucket factor ORDER differs between two builds in one process '
        '(%s vs %s) while the factor set is identical (%s). This is the '
        'signature of iterating an identity-hashed container -- see '
        'notebooks/_August-2026/claude_experiments/21-determinism.md section 1.3. '
        'Look for a new set()/dict over FastFactor or a pyGMs Node.'
        % (case, ord_a, ord_b, set_a))

    gold = goldens['cases'][case]['golden']
    assert ord_a == gold['bucket_order_digest'], (
        '%s: bucket factor order is self-consistent but no longer matches the '
        'recorded order (golden %s, actual %s). Deterministic, but a DIFFERENT '
        'association order for the factor product -- every downstream value '
        'will have moved by a few ULP. Expect the log Z goldens to fail too.\n'
        '  golden provenance: %s'
        % (case, gold['bucket_order_digest'], ord_a,
           _provenance_line(goldens['cases'][case]['provenance'])))


@pytest.mark.parametrize('case', CPU_CASES)
def test_cpu_golden(case, cpu_results, goldens):
    """log Z and the per-bucket epoch vector must equal the recorded values."""
    got, entry = cpu_results[case], goldens['cases'][case]
    gold, prov = entry['golden'], entry['provenance']
    ok = (got['log_z_repr'] == gold['log_z_repr']
          and got['epochs'] == gold['epochs']
          and got['local_errors'] == gold.get('local_errors', []))
    assert ok, _report(case, got, gold, prov)


def test_cpu_case_is_bit_reproducible_in_process(cpu_results):
    """Re-run one case and demand bit-equality with its first run.

    Independent of the goldens on purpose: if someone re-measures the goldens
    after a real change, this still fails the moment run-to-run reproducibility
    itself regresses.
    """
    first = cpu_results[CPU_REPLICATE_CASE]
    second = run_case(CPU_REPLICATE_CASE)
    assert (first['log_z_repr'], first['epochs']) == (second['log_z_repr'], second['epochs']), (
        'Two runs of %s in ONE process disagree -- run-to-run determinism itself '
        'has regressed, independently of any golden value.\n'
        '  run 1: log Z %s epochs %r\n  run 2: log Z %s epochs %r'
        % (CPU_REPLICATE_CASE, first['log_z_repr'], first['epochs'],
           second['log_z_repr'], second['epochs']))


def test_local_errors_are_actually_populated(cpu_results):
    """Guard against a vacuous check.

    Doc 28 section 6.2: its local_errors equality check passed vacuously because
    the field was empty in every run (`compute_local_error` defaults to False).
    `cpu_local_error` sets the flag, so this asserts the field is non-empty
    before test_cpu_golden compares it.
    """
    got = cpu_results['cpu_local_error']
    assert got['local_errors'], (
        'cpu_local_error produced no local_errors records, so the equality '
        'check on that field in test_cpu_golden would pass vacuously. Either '
        "compute_local_error stopped populating FastGM.local_errors, or the "
        'case stopped training any NN cluster (n_nn_clusters=%d).'
        % got['n_nn_clusters'])


def test_chunked_exact_path_matches_unchunked(goldens):
    """Force `_compute_message_exact_chunked` and demand an identical answer.

    The chunked path only triggers above `_EXACT_JOINT_NUMEL_LIMIT` (2**28), far
    beyond anything a fast test can reach, so the limit and the block size are
    lowered to force it on a small problem. That exercises the real blocking
    loop (including the stream_nn_exact slice of NN factors), just at a smaller
    scale. MEASURED: 16 chunked calls, bit-identical log Z.
    """
    case = 'cpu_grid_plain'
    plain = run_case(case)
    chunked = run_case(case, chunk_limit=2 ** 8, chunk_block=2 ** 6)
    assert chunked['n_chunked_calls'] > 0, (
        'lowering _EXACT_JOINT_NUMEL_LIMIT did not route any message through '
        '_compute_message_exact_chunked -- the test is no longer covering that '
        'path (gate moved, or the method was renamed).')
    assert chunked['log_z_repr'] == plain['log_z_repr'] and \
        chunked['epochs'] == plain['epochs'], (
        'chunked exact path disagrees with the unchunked one on %s '
        '(%d chunked calls):\n  unchunked log Z %s epochs %r\n'
        '  chunked   log Z %s epochs %r\n  delta %+.6e (%s float32 ULP)\n'
        'The chunked path documents itself as producing an identical result.'
        % (case, chunked['n_chunked_calls'], plain['log_z_repr'], plain['epochs'],
           chunked['log_z_repr'], chunked['epochs'],
           float(chunked['log_z_repr']) - float(plain['log_z_repr']),
           _float32_ulps(float(plain['log_z_repr']), float(chunked['log_z_repr']))))
    gold = goldens['cases'][case]['golden']
    assert chunked['log_z_repr'] == gold['log_z_repr'], (
        'chunked and unchunked agree with each other but not with the golden; '
        'see the test_cpu_golden failure for the diagnosis.')


def test_det_algos_guard_rejects_missing_cublas_workspace_config(monkeypatch):
    """The CUBLAS requirement must be impossible to omit silently.

    Doc 21 recorded `use_deterministic_algorithms(True)` as raising nothing on
    CUDA; doc 28 section 4.3 measured that it raises on the first F.linear
    unless CUBLAS_WORKSPACE_CONFIG is set. This asserts the guard fires (loud
    error) rather than letting a probe run that would prove nothing.
    """
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    with pytest.raises(RuntimeError, match='CUBLAS_WORKSPACE_CONFIG'):
        require_cublas_workspace_config()
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    require_cublas_workspace_config()


# ---------------------------------------------------------------------------
# CUDA tier
# ---------------------------------------------------------------------------
def _require_matching_gpu(entry):
    if not torch.cuda.is_available():
        pytest.skip('no CUDA device visible')
    want = entry['provenance'].get('device_name')
    have = _gpu_name()
    if want != have:
        pytest.skip(
            'golden was measured on %r, this device is %r. Cross-GPU bit-identity '
            'is NOT tested (doc 28 section 5) -- skipping rather than reporting a '
            'failure that may only mean "different silicon". Re-measure with '
            '--regen cuda on this GPU if you want a golden for it.' % (want, have))


@pytest.fixture(scope='module')
def cuda_results(request):
    if not (request.config.getoption('--gpu') or request.config.getoption('--det-algos')
            or os.environ.get('NCE_TEST_GPU') == '1'
            or os.environ.get('NCE_TEST_DET_ALGOS') == '1'):
        pytest.skip('CUDA tier not selected')
    if not torch.cuda.is_available():
        pytest.skip('no CUDA device visible')
    return {name: run_case(name) for name in CUDA_CASES}


@pytest.mark.gpu
@pytest.mark.parametrize('case', CUDA_CASES)
def test_cuda_golden(case, cuda_results, goldens):
    entry = goldens['cases'][case]
    _require_matching_gpu(entry)
    got = cuda_results[case]
    gold, prov = entry['golden'], entry['provenance']
    ok = (got['log_z_repr'] == gold['log_z_repr']
          and got['epochs'] == gold['epochs'])
    assert ok, _report(case, got, gold, prov)


@pytest.mark.gpu
def test_cuda_case_is_bit_reproducible(cuda_results, goldens):
    """Same role as the CPU replicate test, on the GPU arithmetic."""
    _require_matching_gpu(goldens['cases'][CUDA_REPLICATE_CASE])
    first = cuda_results[CUDA_REPLICATE_CASE]
    second = run_case(CUDA_REPLICATE_CASE)
    assert (first['log_z_repr'], first['epochs']) == (second['log_z_repr'], second['epochs']), (
        'Two CUDA runs of %s in ONE process disagree -- run-to-run determinism '
        'on GPU has regressed.\n  run 1: log Z %s epochs %r\n'
        '  run 2: log Z %s epochs %r'
        % (CUDA_REPLICATE_CASE, first['log_z_repr'], first['epochs'],
           second['log_z_repr'], second['epochs']))


@pytest.mark.gpu
@pytest.mark.det_algos
def test_cuda_golden_under_deterministic_algorithms(goldens):
    """Re-run one CUDA case with torch's deterministic kernels forced.

    Run in a SUBPROCESS because `use_deterministic_algorithms` is process-global
    and MEASURED 2.52x slower (doc 28 section 4.3), and because
    CUBLAS_WORKSPACE_CONFIG must be present before cuBLAS initialises.

    If the ordinary run were quietly using a nondeterministic kernel, forcing
    the deterministic path would generally change the answer. It does not.
    """
    entry = goldens['cases'][DET_ALGOS_CASE]
    _require_matching_gpu(entry)
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', NCE_DET_ALGOS='1')
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                           '--run-case', DET_ALGOS_CASE],
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0, (
        'deterministic-algorithms subprocess failed (rc=%d).\n'
        'A RuntimeError naming CuBLAS/CUBLAS_WORKSPACE_CONFIG means the env var '
        'did not reach the child -- that is a harness bug, not a kernel finding '
        '(doc 28 section 4.3).\n--- stderr tail ---\n%s'
        % (proc.returncode, proc.stderr[-3000:]))
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    gold = entry['golden']
    assert got['log_z_repr'] == gold['log_z_repr'] and got['epochs'] == gold['epochs'], (
        _report(DET_ALGOS_CASE, got, gold, entry['provenance'])
        + '\n  NOTE: this run had torch.use_deterministic_algorithms(True). A '
          'mismatch HERE but not in test_cuda_golden means the ordinary path is '
          'using a nondeterministic kernel whose result differs from the '
          'deterministic one.')


# ---------------------------------------------------------------------------
# CLI: golden regeneration and the det-algos child process
# ---------------------------------------------------------------------------
def _regen(tier):
    torch.set_num_threads(1)
    names = [n for n, v in CASES.items() if v['tier'] == tier]
    if tier == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('--regen cuda needs a visible CUDA device')
    data = {'schema_version': 1,
            'note': ('Every value here was MEASURED by running the case. '
                     'Provenance is per case because these numbers are only '
                     'meaningful relative to a code state.'),
            'cases': {}}
    if GOLDENS_PATH.exists():
        with open(GOLDENS_PATH) as fh:
            data = json.load(fh)
    data.setdefault('cases', {})
    for name in names:
        print('[regen] %s ...' % name, flush=True)
        res = run_case(name)
        golden = {'log_z_repr': res['log_z_repr'], 'epochs': res['epochs'],
                  'n_nn_clusters': res['n_nn_clusters'],
                  'local_errors': res['local_errors']}
        if name in ORDER_CASES:
            golden['bucket_order_digest'] = bucket_factor_digest(build_case(name))
        data['cases'][name] = {
            'tier': tier,
            'covers': CASES[name]['covers'],
            'config': CASES[name]['config'],
            'golden': golden,
            'measured_wall_s': res['wall_s'],
            'provenance': _current_provenance(CASES[name]['config'].get('device', 'cpu')),
        }
        print('        log Z %s  epochs %r  (%.1f s)'
              % (res['log_z_repr'], res['epochs'], res['wall_s']), flush=True)
    GOLDENS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GOLDENS_PATH, 'w') as fh:
        json.dump(data, fh, indent=2)
        fh.write('\n')
    print('[regen] wrote %s' % GOLDENS_PATH)


def _main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--regen', choices=['cpu', 'cuda'],
                    help='re-measure the goldens for one tier')
    ap.add_argument('--run-case', help='run one case and print its result as JSON')
    args = ap.parse_args()
    if args.regen:
        _regen(args.regen)
    elif args.run_case:
        if os.environ.get('NCE_DET_ALGOS') == '1':
            require_cublas_workspace_config()
            torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        print(json.dumps(run_case(args.run_case)))
    else:
        ap.error('nothing to do: pass --regen or --run-case')


if __name__ == '__main__':
    _main()
