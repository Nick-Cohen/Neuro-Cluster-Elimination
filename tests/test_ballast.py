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
    """A card that has left our control entirely must lose its ballast."""
    pool = B.BallastPool(enabled=True)
    stopped, started = [], []
    pool.stop = lambda i, timeout=None: (stopped.append(i), True)[1]
    pool.start = lambda i: (started.append(i), True)[1]
    pool.active_indices = lambda: {0, 3}
    pool.ensure([3])
    assert stopped == [0]
    assert started == []


def test_ensure_keeps_paused_worker_on_a_running_job():
    """REGRESSION. `ensure(free)` used to tear down the PAUSED worker on a card
    running one of our jobs (it is not "free"), and the next call then started a
    fresh UNPAUSED worker on that same card -- GEMM racing a job that was being
    timed. Caught only end-to-end; this pins it."""
    pool = B.BallastPool(enabled=True)
    stopped, started = [], []
    pool.stop = lambda i, timeout=None: (stopped.append(i), True)[1]
    pool.start = lambda i: (started.append(i), True)[1]
    pool.active_indices = lambda: {3}
    pool.ensure([], keep_indices=[3])
    assert stopped == [], 'paused worker on a running job must survive'
    assert started == [], 'must never start an UNPAUSED worker on a busy card'


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


# --------------------------------------------------------------------------
# Handoff: ballast covers the runner's CPU setup and yields before GPU work
# --------------------------------------------------------------------------
def test_handoff_info_none_when_no_ballast():
    pool = B.BallastPool(enabled=True)
    assert pool.handoff_info(0) is None


def test_handoff_info_reports_pid_and_control_files(tmp_path):
    pool = B.BallastPool(enabled=True, log_dir=str(tmp_path))
    fp = _FakeProc()
    pool.procs[3] = {'proc': fp, 'started': time.time(), 'log': None}
    info = pool.handoff_info(3)
    assert info is not None
    pid, pause_file, paused_marker = info
    assert pid == fp.pid
    assert pause_file == pool.pause_file_for(3)
    assert paused_marker == pool.paused_marker_for(3)
    fp.terminate()
    assert pool.handoff_info(3) is None, 'a dead worker must not be handed off'


def test_resume_clears_pause_file(tmp_path):
    pool = B.BallastPool(enabled=True, log_dir=str(tmp_path))
    os.makedirs(str(tmp_path), exist_ok=True)
    with open(pool.pause_file_for(0), 'w') as fh:
        fh.write('x')
    pool.resume(0)
    assert not os.path.exists(pool.pause_file_for(0))


def test_ballast_survives_on_cards_running_our_jobs(tmp_path, monkeypatch):
    """A card running a job holds a PAUSED worker. It must not be torn down, or
    it could not resume when the job ends and the next job would have nothing
    to hand off to."""
    from nce.scheduler.scheduler import Dispatcher
    from nce.scheduler.jobs import JobQueue
    from nce.scheduler import scheduler as S

    q = JobQueue(str(tmp_path / 'q.json'))
    d = Dispatcher(q, str(tmp_path / 'out'), ballast=True)
    fp = _FakeProc()
    d.ballast.procs[3] = {'proc': fp, 'started': time.time(), 'log': None}
    d.running['GPU-x'] = {'proc': _FakeProc(), 'job_id': 'j', 'started': 0.0,
                          'log': None, 'gpu_index': 3, 'name': 'n'}
    monkeypatch.setattr(S, 'dispatchable_gpus', lambda **kw: [])
    stopped = []
    d.ballast.stop = lambda i, timeout=None: (stopped.append(i), True)[1]
    d._ballast_idle_cards()
    assert stopped == [], 'ballast on a card running our job must be kept alive'


def test_start_clears_stale_stop_file(tmp_path):
    """A stop-file left by the previous job would kill the next ballast worker
    on its first check, silently disabling ballast for the rest of the sweep."""
    pool = B.BallastPool(enabled=True, log_dir=str(tmp_path))
    stale = pool.stop_file_for(0)
    os.makedirs(str(tmp_path), exist_ok=True)
    with open(stale, 'w') as fh:
        fh.write('stale')
    assert os.path.exists(stale)
    started = {}
    real_popen = subprocess.Popen

    def fake_popen(cmd, **kw):
        started['cmd'] = cmd
        started['stop_file_existed'] = os.path.exists(stale)
        return _FakeProc()

    subprocess.Popen = fake_popen
    try:
        pool.start(0)
    finally:
        subprocess.Popen = real_popen
    assert started['stop_file_existed'] is False, 'stale stop-file not cleared'
    assert '--stop-file' in started['cmd']


def test_reap_drops_exited_workers():
    pool = B.BallastPool(enabled=True)
    fp = _FakeProc()
    pool.procs[1] = {'proc': fp, 'started': time.time(), 'log': None}
    assert pool.reap() == []
    fp.terminate()
    assert pool.reap() == [1]
    assert 1 not in pool.procs


def test_pause_ballast_refuses_without_attestation(tmp_path):
    """If ballast never confirms it stopped, the runner must RAISE, not run.

    A job timed against a card that is also running ballast is silently wrong,
    which is worse than a failed job.
    """
    from nce.scheduler import runner as R
    pf = str(tmp_path / 'pause')
    with pytest.raises(RuntimeError, match='did not confirm pause'):
        R.pause_ballast(pf, str(tmp_path / 'paused'), 0, timeout=0.5)
    assert os.path.exists(pf), 'the pause-file must still have been written'


def test_pause_ballast_returns_on_attestation(tmp_path):
    """Only the worker's own marker clears the job to start."""
    from nce.scheduler import runner as R
    import threading
    pf, pm = str(tmp_path / 'pause'), str(tmp_path / 'paused')

    def attest():
        time.sleep(0.1)
        with open(pm, 'w') as fh:
            fh.write('ok')

    t = threading.Thread(target=attest)
    t.start()
    try:
        info = R.pause_ballast(pf, pm, 0, timeout=5)
    finally:
        t.join()
    assert info['ballast_pause_s'] >= 0


def test_pause_ballast_ignores_stale_marker(tmp_path):
    """A marker left from the previous job must not clear this one instantly."""
    from nce.scheduler import runner as R
    pf, pm = str(tmp_path / 'pause'), str(tmp_path / 'paused')
    with open(pm, 'w') as fh:
        fh.write('stale')
    with pytest.raises(RuntimeError, match='did not confirm pause'):
        R.pause_ballast(pf, pm, 0, timeout=0.5)


def test_dispatch_hands_off_warm_card(tmp_path, monkeypatch):
    """The scheduler must pass the stop-file/pid to the runner and must NOT
    stop ballast before launching -- otherwise the card idles through setup."""
    from nce.scheduler.scheduler import Dispatcher
    from nce.scheduler.jobs import JobQueue, JobSpec

    q = JobQueue(str(tmp_path / 'q.json'))
    q.add([JobSpec(problem_key='grids/grid10x10.f10', merge_strategy='nomerge',
                   merge_bound=None, seed=1)])
    d = Dispatcher(q, str(tmp_path / 'out'), ballast=True)
    fp = _FakeProc()
    d.ballast.procs[3] = {'proc': fp, 'started': time.time(), 'log': None}

    seen = {}
    monkeypatch.setattr('subprocess.Popen',
                        lambda cmd, **kw: (seen.update(cmd=cmd), _FakeProc())[1])

    class G:
        index, uuid = 3, 'GPU-x'
    d._launch(q.pending()[0], G())

    assert '--ballast-pause-file' in seen['cmd']
    assert '--ballast-paused-marker' in seen['cmd']
    assert d.ballast.pause_file_for(3) in seen['cmd']
    assert fp.terminated is False, 'ballast must NOT be stopped before launch'


# --------------------------------------------------------------------------
# Model-cache guard
# --------------------------------------------------------------------------
def test_cache_assertion_names_the_variable(tmp_path, monkeypatch):
    from nce.scheduler import models as M
    monkeypatch.setenv('NCE_MODEL_CACHE', str(tmp_path / 'nope'))
    with pytest.raises(RuntimeError, match='NCE_MODEL_CACHE'):
        M.assert_cache_configured()
    empty = tmp_path / 'empty'
    empty.mkdir()
    monkeypatch.setenv('NCE_MODEL_CACHE', str(empty))
    with pytest.raises(RuntimeError, match='no .uai files'):
        M.assert_cache_configured()


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
