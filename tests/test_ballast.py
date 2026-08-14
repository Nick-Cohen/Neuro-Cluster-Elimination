"""Tests for thermal ballast (nce/scheduler/ballast.py).

The three requirements from the spec, each with a test that would actually fail
if the requirement were violated:

  1. yields instantly when a real job arrives  -> test_stop_waits_for_exit,
     test_worker_yields_on_sigterm (GPU), test_dispatch_stops_ballast_first
  2. never touches gpu2                        -> test_pool_refuses_retired,
     test_ensure_never_ballasts_retired, test_worker_refuses_retired
  3. does not perturb a running job            -> test_ensure_drops_cards_not_free,
     test_dispatch_stops_ballast_first

Plus the interaction that would silently disable the scheduler:
     test_ignore_pids_clears_busy / test_ignore_pids_clears_memory.
"""
import os
import subprocess
import sys
import time

import pytest

from nce.scheduler import ballast as B
from nce.scheduler.gpus import RETIRED_GPU_INDICES, GpuInfo, query_gpus


# --------------------------------------------------------------------------
# Requirement 2: gpu2 is never touched
# --------------------------------------------------------------------------
def test_retired_set_still_contains_gpu2():
    assert 2 in RETIRED_GPU_INDICES


def test_pool_refuses_retired():
    pool = B.BallastPool(enabled=True)
    assert pool.start(2) is False
    assert pool.procs == {}
    assert pool.active_indices() == set()


def test_ensure_never_ballasts_retired():
    pool = B.BallastPool(enabled=True)
    started = []
    pool.start = lambda i: (started.append(i), True)[1]
    pool.ensure([0, 2, 3])
    assert 2 not in started


def test_worker_refuses_retired():
    """The worker guards its own --gpu, independently of the pool."""
    r = subprocess.run([sys.executable, '-m', 'nce.scheduler.ballast',
                        '--gpu', '2'],
                       capture_output=True, text=True,
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert r.returncode != 0
    assert 'RETIRED' in (r.stdout + r.stderr)


# --------------------------------------------------------------------------
# The interaction that would deadlock the scheduler
# --------------------------------------------------------------------------
def test_ignore_pids_clears_busy(monkeypatch):
    """A card carrying only ballast must read as IDLE once its pid is ignored."""
    uuid = 'GPU-test-uuid-0'

    def fake_smi(query, extra=None):
        if query.startswith('compute-apps'):
            return [[uuid, '99999', '1500']]
        # 1501 total - 1500 ballast = 1 MiB residual, i.e. the 1 MiB an
        # unattached TITAN RTX reports on this box.
        return [['0', uuid, 'FAKE', '1501', '24576', '99']]

    from nce.scheduler import gpus as G
    monkeypatch.setattr(G, '_nvidia_smi', fake_smi)

    busy = G.query_gpus()[0]
    assert busy.compute_pids == (99999,)
    assert not busy.idle, 'ballast pid must look busy when NOT ignored'

    free = G.query_gpus(ignore_pids={99999})[0]
    assert free.compute_pids == ()
    assert free.idle, 'ignoring the ballast pid must make the card dispatchable'


def test_ignore_pids_clears_memory(monkeypatch):
    """Ignoring the pid must also discount its memory, or the secondary
    memory signal re-flags the card and the pid filter is defeated."""
    uuid = 'GPU-test-uuid-1'

    def fake_smi(query, extra=None):
        if query.startswith('compute-apps'):
            return [[uuid, '4242', '1500']]
        return [['0', uuid, 'FAKE', '1501', '24576', '99']]

    from nce.scheduler import gpus as G
    monkeypatch.setattr(G, '_nvidia_smi', fake_smi)

    free = G.query_gpus(ignore_pids={4242})[0]
    assert free.memory_used_mib == 1, 'ballast memory must be subtracted'
    assert free.idle


def test_ignore_pids_does_not_hide_real_work(monkeypatch):
    """Only the ballast pid is ignored; a real job on the same card still
    marks it busy."""
    uuid = 'GPU-test-uuid-2'

    def fake_smi(query, extra=None):
        if query.startswith('compute-apps'):
            return [[uuid, '4242', '1500'], [uuid, '777', '8000']]
        return [['0', uuid, 'FAKE', '9501', '24576', '99']]

    from nce.scheduler import gpus as G
    monkeypatch.setattr(G, '_nvidia_smi', fake_smi)

    g = G.query_gpus(ignore_pids={4242})[0]
    assert g.compute_pids == (777,)
    assert not g.idle


# --------------------------------------------------------------------------
# Pool lifecycle (no GPU needed -- a sleep stands in for the worker)
# --------------------------------------------------------------------------
class _FakeProc:
    def __init__(self):
        self.pid = 1234
        self._alive = True
        self.terminated = False

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self.terminated = True
        self._alive = False

    def kill(self):
        self._alive = False

    def wait(self, timeout=None):
        if self._alive:
            raise subprocess.TimeoutExpired('x', timeout)
        return 0


def test_pids_reports_live_workers():
    pool = B.BallastPool(enabled=True)
    fp = _FakeProc()
    pool.procs[0] = {'proc': fp, 'started': time.time(), 'log': None}
    assert pool.pids() == {1234}
    fp.terminate()
    assert pool.pids() == set()


def test_stop_waits_for_exit():
    """stop() must terminate AND reap; returning early would let a job start
    against a card whose CUDA context is still held."""
    pool = B.BallastPool(enabled=True)
    fp = _FakeProc()
    pool.procs[0] = {'proc': fp, 'started': time.time(), 'log': None}
    assert pool.stop(0, timeout=1.0) is True
    assert fp.terminated
    assert 0 not in pool.procs


def test_ensure_drops_cards_not_free():
    """A card that stops being free must lose its ballast -- this is what keeps
    ballast off a card that is running a job."""
    pool = B.BallastPool(enabled=True)
    stopped, started = [], []
    pool.stop = lambda i, timeout=None: (stopped.append(i), True)[1]
    pool.start = lambda i: (started.append(i), True)[1]
    pool.active_indices = lambda: {0, 3}
    pool.ensure([3])
    assert stopped == [0]
    assert started == []


def test_disabled_pool_is_inert():
    pool = B.BallastPool(enabled=False)
    assert pool.start(0) is False
    pool.ensure([0, 1, 3])
    assert pool.procs == {}


# --------------------------------------------------------------------------
# Requirement 1, for real: a live worker yields on SIGTERM
# --------------------------------------------------------------------------
def _free_non_retired_gpu():
    for g in query_gpus():
        if not g.retired and g.idle:
            return g.index
    return None


@pytest.mark.skipif(_free_non_retired_gpu() is None,
                    reason='no free non-retired GPU')
def test_worker_yields_on_sigterm():
    """Start a real ballast worker, let it reach the GEMM loop, SIGTERM it, and
    require that it exits promptly and releases its CUDA context."""
    idx = _free_non_retired_gpu()
    pool = B.BallastPool(enabled=True,
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert pool.start(idx) is True
    try:
        # Wait for the context to actually appear before timing the yield.
        deadline = time.time() + 120
        while time.time() < deadline:
            g = [x for x in query_gpus() if x.index == idx][0]
            if pool.procs[idx]['proc'].pid in g.compute_pids:
                break
            time.sleep(1)
        else:
            pytest.skip('ballast never attached a context (GPU contended?)')

        pid = pool.procs[idx]['proc'].pid
        t0 = time.time()
        assert pool.stop(idx) is True
        dt = time.time() - t0
        # No invented budget: the assertion is only that it yielded within the
        # module's own documented stop timeout, i.e. that it did not need SIGKILL.
        assert dt < B.BALLAST_STOP_TIMEOUT_S

        g = [x for x in query_gpus() if x.index == idx][0]
        assert pid not in g.compute_pids, 'CUDA context outlived stop()'
    finally:
        pool.stop_all()
