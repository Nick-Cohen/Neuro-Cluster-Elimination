"""Structured per-cluster sample-generation tracing ("gamma v2").

WHY
---
The original gamma measurement (``gamma_fit_points.csv``) recorded only
``T_gen, m, r, e, k_max, w_scope`` per cluster and fitted a single scalar

    gamma = T_gen / (m * r * k_max ** e)   ->   4.35e-8 s/op

That instrumentation cannot answer the question the cost model actually needs,
because it conflates three different things:

  1. table-factor arithmetic and neural-network evaluation are pooled into one
     ``r``.  Doc 03 measures a single NN factor at 400-800x a table factor, so a
     cluster's cost is dominated by ``n_nn``, a field the old CSV does not have;
  2. ``k_max ** e`` overcounts every mixed-domain cluster.  Pedigrees carry
     k in {2,3,4,5}; the true elimination grid is ``prod_{v in E} k_v``;
  3. ~94% of the measured time at the median cluster is fixed overhead
     (``T_gen = 8.92 ms + 3.65e-9 * ops`` on ops <= 1e8), so a *ratio* median is
     a measurement of the overhead, not of a per-op rate.

This module records the missing fields, plus the code revision that produced
each row, so that pre- and post-``perf/nn-eval-fixes`` numbers can never be
pooled by accident.

DESIGN CONSTRAINTS
------------------
* **Off by default.**  Tracing activates only when the config carries a
  ``gamma_trace_path``.  With the key absent every hook reduces to one dict
  ``.get()`` returning ``None``, and the traced code path is byte-for-byte the
  same sequence of tensor operations as the untraced one.
* **Numerically inert.**  The tracer only *reads* already-computed structure
  (labels, domain sizes, factor flags) and calls ``time.perf_counter`` /
  ``torch.cuda.synchronize``.  It never draws a random number, never seeds,
  never allocates a tensor, and never reorders an accumulation.  ``synchronize``
  changes *when* work happens, never *what* is computed.
* **Structured sink.**  Rows are appended as JSONL (one JSON object per
  cluster).  JSONL, not CSV, because the per-factor breakdown is a variable
  length list and because a schema addition must not invalidate old files.

USAGE
-----
Set ``gamma_trace_path`` in the nn_config (and optionally
``gamma_trace_per_factor: true``, ``gamma_trace_meta: {...}``).  Every cluster
that generates an NBE validation set then appends one row.  ``time_sample_gen``
keeps its original behaviour of printing a ``[GammaTiming]`` line and may be
used with or without ``gamma_trace_path``.

Reading rows back::

    from nce.utils.gamma_trace import read_rows
    rows = read_rows('/path/to/gamma_v2.jsonl')
"""

from __future__ import annotations

import atexit
import datetime as _dt
import json
import math
import os
import subprocess
import time
import uuid

SCHEMA_VERSION = 2

# --------------------------------------------------------------------------
# module state.  ``_CURRENT`` is the cluster record being filled, or None.
# Sample generation for a cluster is single-threaded and strictly nested inside
# the ``cluster()`` context, so a module global is the right scope here.
# --------------------------------------------------------------------------
_CURRENT = None          # dict | None  -- fast "is tracing on right now?" check
_SINKS = {}              # path -> file handle
_GIT = None              # cached git revision info
_RUN_ID = uuid.uuid4().hex[:12]


# ==========================================================================
# sinks
# ==========================================================================

def _sink(path):
    """Return an append-mode handle for ``path``, opening (and registering for
    close-at-exit) on first use."""
    h = _SINKS.get(path)
    if h is None:
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        h = open(path, 'a', buffering=1)     # line buffered: rows survive a crash
        _SINKS[path] = h
    return h


def close_all():
    for h in _SINKS.values():
        try:
            h.close()
        except Exception:
            pass
    _SINKS.clear()


atexit.register(close_all)


def read_rows(path):
    """Load a JSONL trace file into a list of dicts (skipping blank lines)."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ==========================================================================
# provenance
# ==========================================================================

def _git_info():
    """Revision of the checkout that *this file* lives in (not the cwd).

    Rows from before and after the perf fixes are not comparable, so every row
    carries the revision that produced it.  ``dirty`` is as important as the
    hash: an uncommitted working tree means the hash alone does not identify
    the code.
    """
    global _GIT
    if _GIT is not None:
        return _GIT
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def _run(*args):
        try:
            return subprocess.run(args, cwd=repo, capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None

    status = _run('git', 'status', '--porcelain')
    _GIT = {
        'git_repo': repo,
        'git_commit': _run('git', 'rev-parse', 'HEAD'),
        'git_branch': _run('git', 'rev-parse', '--abbrev-ref', 'HEAD'),
        'git_describe': _run('git', 'describe', '--always', '--dirty'),
        'git_dirty': bool(status) if status is not None else None,
        'git_dirty_files': len(status.splitlines()) if status else 0,
    }
    return _GIT


def _env_info(device):
    info = {'device': str(device), 'gpu_name': None, 'torch_version': None,
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    try:
        import torch
        info['torch_version'] = torch.__version__
        if torch.cuda.is_available() and 'cuda' in str(device):
            idx = torch.device(str(device)).index or 0
            info['gpu_name'] = torch.cuda.get_device_name(idx)
    except Exception:
        pass
    # the streaming knobs change chunk sizes and therefore launch counts
    for k in ('NCE_SAMPLE_BLOCK_LOG2', 'NCE_SAMPLE_ACHUNK',
              'NCE_SAMPLE_SMALL_LOG2', 'NCE_SAMPLE_MEM_FRAC'):
        info[k.lower()] = os.environ.get(k)
    return info


# ==========================================================================
# cluster structure extraction  (read-only)
# ==========================================================================

def _states_of(gm, label, cache):
    s = cache.get(label)
    if s is None:
        try:
            s = int(gm.matching_var(label).states)
        except Exception:
            s = 2
        cache[label] = s
    return s


def _describe_cluster(bucket):
    """Everything about the cluster's shape that the cost model might need.

    Deliberately records BOTH the old and the corrected definitions of every
    quantity that changed, so v2 rows can reproduce a v1 fit exactly and the
    difference between the two can be attributed rather than assumed.
    """
    gm = bucket.gm
    cache = {}
    elim_labels = [getattr(v, 'label', v) for v in bucket.elim_vars]
    elim_states = [_states_of(gm, l, cache) for l in elim_labels]

    try:
        sep = list(bucket.get_message_scope())
    except Exception:
        sep = []
    sep_states = [_states_of(gm, l, cache) for l in sep]

    scope_labels = sorted(set(elim_labels) | set(sep))
    scope_states = [_states_of(gm, l, cache) for l in scope_labels]

    def _prod(xs):
        p = 1
        for x in xs:
            p *= int(x)
        return p

    factors = list(bucket.factors)
    fdesc = []
    n_nn = 0
    sum_nn_scope_states = 0
    sum_nn_elim_states = 0
    elim_set = set(elim_labels)
    for i, f in enumerate(factors):
        is_nn = bool(getattr(f, 'is_nn', False))
        labels = [getattr(l, 'label', l) for l in getattr(f, 'labels', [])]
        f_states = [_states_of(gm, l, cache) for l in labels]
        # e_f: the elim vars this factor's scope actually contains.  The
        # redundancy bug is exactly k^(e - e_f) wasted rows per NN factor, so
        # e_f must be recorded to model (or to verify the removal of) it.
        f_elim = [l for l in labels if l in elim_set]
        f_elim_states = [_states_of(gm, l, cache) for l in f_elim]
        d = {
            'i': i,
            'is_nn': is_nn,
            'n_labels': len(labels),
            'states_prod': _prod(f_states),
            'e_f': len(f_elim),
            'elim_states_prod_f': _prod(f_elim_states),
        }
        if is_nn:
            n_nn += 1
            sum_nn_scope_states += d['states_prod']
            sum_nn_elim_states += d['elim_states_prod_f']
        fdesc.append(d)

    elim_prod = _prod(elim_states)
    r = len(factors)
    k_max_sep = max(sep_states) if sep_states else 2      # the OLD `k` definition
    k_max_scope = max(scope_states) if scope_states else 2

    return {
        'bucket': getattr(bucket, 'label', None),
        'r_functions': r,
        'n_nn': n_nn,
        'n_table': r - n_nn,
        'e_elim_vars': len(elim_labels),
        'elim_labels': elim_labels,
        'elim_domain_sizes': elim_states,
        'elim_states_prod': elim_prod,          # TRUE prod k_v over eliminated vars
        'k_max_elim': max(elim_states) if elim_states else 2,
        'w_sep': len(sep),
        'sep_states_prod': _prod(sep_states),
        'w_scope': len(sep) + len(elim_labels),  # same definition as the v1 CSV
        'scope_states_prod': _prod(scope_states),
        'k_max_domain': k_max_sep,               # v1-compatible (max over separator)
        'k_max_scope': k_max_scope,
        'sum_nn_scope_states': sum_nn_scope_states,
        'sum_nn_elim_states': sum_nn_elim_states,
        'mixed_domain': len(set(elim_states)) > 1,
        'factors': fdesc,
    }


def _describe_config(cfg):
    """Identifying info, enough to join a row against structure data."""
    if not isinstance(cfg, dict):
        return {}
    keys = ('problem_key', 'experiment_name', 'iB', 'ecl', 'seed',
            'max_merge_bound', 'num_samples', 'sampling_scheme',
            'use_reduce_nn_merge', 'reduce_nn_backtrack', 'stream_nn_exact',
            'dope_factors', 'neurobe_mode', 'hidden_sizes', 'num_epochs')
    out = {}
    for k in keys:
        if k in cfg:
            v = cfg[k]
            out[k] = v if isinstance(v, (int, float, str, bool, type(None), list)) else str(v)
    # canonical aliases used by the analysis
    out['e_max'] = cfg.get('max_merge_bound')
    out['problem'] = cfg.get('problem_key')
    out['strategy'] = cfg.get('gamma_trace_strategy') or _infer_strategy(cfg)
    return out


def _infer_strategy(cfg):
    """Best-effort merge-strategy tag.  Prefer an explicit
    ``gamma_trace_strategy`` in the config; this is only a fallback."""
    name = cfg.get('experiment_name') or ''
    for tag in ('nomerge', 'rnn', 'sub', 'mask', 'non'):
        if tag in str(name):
            return tag
    if not cfg.get('use_reduce_nn_merge', False):
        return 'nomerge'
    return 'rnn'


# ==========================================================================
# the cluster record
# ==========================================================================

class _ClusterRecord(dict):
    """Accumulates phase and per-factor timings for one cluster."""

    def __init__(self, path, per_factor, sync):
        super().__init__()
        self.path = path
        self.per_factor = per_factor
        self.sync = sync
        self._ft = {}          # factor index -> [seconds, n_calls]
        self._phase = {}       # phase name -> seconds

    # -- gpu-accurate clock ------------------------------------------------
    def _sync(self):
        if not self.sync:
            return
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    def tic(self):
        self._sync()
        return time.perf_counter()

    def toc_factor(self, idx, t0):
        self._sync()
        slot = self._ft.get(idx)
        dt = time.perf_counter() - t0
        if slot is None:
            self._ft[idx] = [dt, 1]
        else:
            slot[0] += dt
            slot[1] += 1

    def add_phase(self, name, dt):
        self._phase[name] = self._phase.get(name, 0.0) + dt

    def note(self, **kw):
        """Record extra scalars (streaming path, chunk sizes, ...)."""
        for k, v in kw.items():
            self[k] = v


# ==========================================================================
# public API -- hooks call these
# ==========================================================================

def active():
    """The in-flight cluster record, or ``None``.  This is the hot-path check;
    keep it a bare global read."""
    return _CURRENT


def per_factor_active():
    """Record iff tracing is on AND per-factor breakdown was requested."""
    rec = _CURRENT
    return rec if (rec is not None and rec.per_factor) else None


def phase(name):
    """Context manager timing a named sub-phase of sample generation.

    No-op (and no ``synchronize``) when tracing is off.
    """
    return _PhaseCtx(name)


class _PhaseCtx:
    __slots__ = ('name', 'rec', 't0')

    def __init__(self, name):
        self.name = name
        self.rec = _CURRENT

    def __enter__(self):
        if self.rec is not None:
            self.t0 = self.rec.tic()
        return self

    def __exit__(self, *exc):
        if self.rec is not None:
            self.rec._sync()
            self.rec.add_phase(self.name, time.perf_counter() - self.t0)
        return False


def trace_enabled(config):
    """True iff this config asks for structured tracing."""
    return bool(isinstance(config, dict) and config.get('gamma_trace_path'))


def timed_generate(bucket, gen_fn, phase_name='trainer.nbe_val', print_line=True,
                   m_of=None):
    """Run ``gen_fn()``, timing it as one cluster's sample generation.

    This is the single entry point used by both hook sites.  It subsumes the
    original ``time_sample_gen`` behaviour (the ``[GammaTiming]`` print) and
    adds the structured row when ``gamma_trace_path`` is configured.

    Args:
        bucket:     the ``FastBucket`` whose message is being generated.
        gen_fn:     zero-arg callable performing the generation; its return
                    value is passed straight through.
        phase_name: which call site this is (there is more than one).
        print_line: emit the legacy ``[GammaTiming]`` line.
        m_of:       callable mapping ``gen_fn``'s result to the sample count.
                    Defaults to ``len(result[0]['x'])``.

    Returns:
        Exactly what ``gen_fn()`` returned.  Raising is propagated unchanged
        after the record is torn down.
    """
    global _CURRENT
    cfg = getattr(bucket, 'config', None) or {}
    path = cfg.get('gamma_trace_path')
    tracing = bool(path)

    rec = None
    if tracing:
        rec = _ClusterRecord(
            path=path,
            per_factor=bool(cfg.get('gamma_trace_per_factor', False)),
            sync=bool(cfg.get('gamma_trace_sync', True)),
        )
        prev, _CURRENT = _CURRENT, rec
        rec._sync()

    t0 = time.perf_counter()
    try:
        result = gen_fn()
    finally:
        if tracing:
            rec._sync()
    t_gen = time.perf_counter() - t0

    if tracing:
        _CURRENT = prev

    # -- sample count ------------------------------------------------------
    try:
        m = int(m_of(result)) if m_of is not None else int(len(result[0]['x']))
    except Exception:
        m = None

    if print_line:
        _print_legacy(bucket, t_gen, m)

    if tracing:
        try:
            _emit(rec, bucket, cfg, t_gen, m, phase_name)
        except Exception as ex:                     # never fail a run for a log
            print(f"[gamma_trace] WARNING: failed to emit row: "
                  f"{type(ex).__name__}: {ex}", flush=True)

    return result


def _print_legacy(bucket, t_gen, m):
    """The original ``[GammaTiming]`` line, unchanged in format so existing
    log parsers keep working."""
    try:
        scope = bucket.get_message_scope()
        sset = set(scope)
        states = [v.states for v in bucket.gm.vars if v.label in sset]
        print(f"[GammaTiming] bucket={bucket.label} T_gen={t_gen:.4f} "
              f"m={m} r={len(bucket.factors)} "
              f"e={len(bucket.elim_vars)} k={max(states) if states else 2} "
              f"w_scope={len(scope) + len(bucket.elim_vars)}", flush=True)
    except Exception:
        pass


def _emit(rec, bucket, cfg, t_gen, m, phase_name):
    row = {
        'schema_version': SCHEMA_VERSION,
        'ts': _dt.datetime.now().isoformat(timespec='seconds'),
        'run_id': cfg.get('gamma_trace_run_id') or _RUN_ID,
        'call_site': phase_name,
        'T_gen_s': t_gen,
        'm_samples': m,
    }
    row.update(_git_info())
    row.update(_env_info(getattr(bucket.gm, 'device', cfg.get('device'))))
    row.update(_describe_config(cfg))
    row.update(_describe_cluster(bucket))

    # -- phase breakdown ---------------------------------------------------
    acct = 0.0
    for name, dt in rec._phase.items():
        row['t_%s_s' % name] = dt
        acct += dt
    row['t_accounted_s'] = acct
    row['t_unaccounted_s'] = t_gen - acct

    # -- per-factor breakdown ---------------------------------------------
    if rec.per_factor and rec._ft:
        t_nn = t_tab = 0.0
        calls = 0
        for d in row['factors']:
            slot = rec._ft.get(d['i'])
            if slot is None:
                continue
            d['t_s'], d['n_calls'] = slot[0], slot[1]
            calls += slot[1]
            if d['is_nn']:
                t_nn += slot[0]
            else:
                t_tab += slot[0]
        row['t_nn_factors_s'] = t_nn
        row['t_table_factors_s'] = t_tab
        row['n_factor_evals'] = calls
    row['per_factor'] = bool(rec.per_factor)

    # -- streaming path notes recorded via rec.note() ----------------------
    for k in ('sg_path', 'sg_a_chunk', 'sg_e_chunk', 'sg_n_chunks',
              'sg_elim_prod', 'sg_n_assignments', 'sg_n_nn_resolved_to_exact'):
        if k in rec:
            row[k] = rec[k]

    # -- ops definitions, old and corrected --------------------------------
    if m:
        r = row['r_functions']
        e = row['e_elim_vars']
        k = row['k_max_domain']
        try:
            row['ops_v1'] = float(m) * r * (float(k) ** e)      # the published one
        except OverflowError:
            row['ops_v1'] = math.inf
        row['ops_true'] = float(m) * r * float(row['elim_states_prod'])
        row['rows_table'] = float(m) * row['n_table'] * float(row['elim_states_prod'])
        row['rows_nn'] = float(m) * float(row['sum_nn_elim_states'])
        row['gamma_v1'] = (t_gen / row['ops_v1']) if row['ops_v1'] else None

    _sink(rec.path).write(json.dumps(row, default=str) + '\n')
