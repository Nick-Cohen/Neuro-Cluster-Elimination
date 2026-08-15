"""Reaper tests: PID reuse, zombies, slow-but-alive jobs, retry-once, resume.

The load-bearing case is PID REUSE. A reaper that checks only `os.kill(pid, 0)`
will, on a box that churns 3.9M-range PIDs for 13.5 days, eventually find an
unrelated process wearing a dead job's number and conclude the job is alive --
so it never fires, and the sweep stalls exactly as it would with no reaper at
all. That case is simulated explicitly here rather than argued about.

The second load-bearing case is the opposite error: reaping a job that is merely
SLOW. Some arms in this study run 5+ hours. A reaped-but-alive job would be
re-dispatched onto a card its own process still owns, and co-tenancy corrupts
the timings the rerun exists to measure.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

from nce.scheduler import reaper as rp
from nce.scheduler.jobs import (JobSpec, JobQueue, PENDING, RUNNING, DONE,
                                FAILED, BLOCKED, NEEDS_ATTENTION)


def _q(tmp_path, n=1):
    q = JobQueue(str(tmp_path / 'q.json'))
    q.add([JobSpec('grids/grid10x10.f10', 'reduce_nn', 4, 40 + i)
           for i in range(n)])
    return q


def _sleeper(marker: str = ''):
    """A live process whose cmdline we control. Not our child's identity."""
    code = 'import sys,time;time.sleep(300)'
    return subprocess.Popen([sys.executable, '-c', code, marker])


# ---------------------------------------------------------------------------
# process identity primitives
# ---------------------------------------------------------------------------
def test_identity_of_self_is_stable_and_has_a_start_time():
    me = rp.process_identity(os.getpid())
    assert me is not None
    assert me['pid'] == os.getpid()
    assert isinstance(me['starttime_ticks'], int) and me['starttime_ticks'] > 0
    assert rp.process_identity(os.getpid()) == me      # does not drift


def test_identity_is_none_for_a_dead_pid():
    p = _sleeper()
    pid = p.pid
    p.kill()
    p.wait()                       # reaped, so not even a zombie remains
    assert rp.process_identity(pid) is None


def test_zombie_is_treated_as_dead_even_though_os_kill_succeeds():
    """os.kill(pid, 0) succeeds on a zombie; the reaper must not be fooled."""
    p = subprocess.Popen([sys.executable, '-c', 'pass'])
    for _ in range(200):
        st = rp._read_stat(p.pid)
        if st and st[0] == 'Z':
            break
        time.sleep(0.01)
    else:
        p.wait()
        pytest.skip('could not observe the zombie window')
    os.kill(p.pid, 0)                          # the naive check: "alive"
    assert rp.process_identity(p.pid) is None  # ours: correctly dead
    p.wait()


# ---------------------------------------------------------------------------
# PID REUSE -- the case that decides whether this design is worth anything
# ---------------------------------------------------------------------------
def test_recycled_pid_is_not_mistaken_for_the_original_job(tmp_path):
    """Simulate reuse: record job A's identity, then let an IMPOSTOR hold the pid.

    Reuse is simulated rather than waited for (a real wrap takes days): the
    impostor is a genuinely different live process, and the queue record is made
    to point at it while carrying the ORIGINAL process's identity. That is
    precisely the state the box produces when a runner dies and its number is
    reissued.
    """
    q = _q(tmp_path)
    spec = q.pending()[0]

    original = _sleeper(marker='--job-file /runs/%s/job.json' % spec.name)
    orig_identity = rp.process_identity(original.pid)
    q.mark_running(spec.job_id, 'GPU-uuid-x', original.pid,
                   proc_identity=orig_identity)
    assert rp.job_liveness(q.get(spec.job_id), spec.job_id)[0] == rp.ALIVE

    original.kill()
    original.wait()

    impostor = _sleeper(marker='totally-unrelated')
    try:
        # The impostor is a DIFFERENT process; make the record name its pid while
        # keeping the identity recorded for the original.
        rec = dict(q.get(spec.job_id))
        rec['pid'] = impostor.pid
        state, reason = rp.job_liveness(rec, spec.job_id)

        # The naive check would say alive:
        os.kill(impostor.pid, 0)
        assert state == rp.DEAD, (
            'reaper treated a recycled pid as the original job: %s' % reason)
        assert 'PID REUSE' in reason
    finally:
        impostor.kill()
        impostor.wait()


def test_reboot_invalidates_a_recorded_identity(tmp_path):
    """starttime is measured FROM BOOT, so it is only meaningful with boot_id."""
    q = _q(tmp_path)
    spec = q.pending()[0]
    p = _sleeper(marker='--job-file /runs/%s/job.json' % spec.name)
    try:
        ident = rp.process_identity(p.pid)
        ident = dict(ident, boot_id='00000000-0000-0000-0000-000000000000')
        q.mark_running(spec.job_id, 'GPU-uuid-x', p.pid, proc_identity=ident)
        state, reason = rp.job_liveness(q.get(spec.job_id), spec.job_id)
        assert state == rp.DEAD and 'reboot' in reason
    finally:
        p.kill()
        p.wait()


def test_legacy_record_without_identity_uses_the_cmdline_binding(tmp_path):
    """Records written before this module existed (the live phase-1 queue).

    They carry a pid and nothing else, so the binding has to be the pid's own
    cmdline. A live process carrying the job id is the job; a live process not
    carrying it is a recycled pid.
    """
    q = _q(tmp_path)
    spec = q.pending()[0]

    carrier = _sleeper(marker='--job-file /runs/%s/job.json' % spec.name)
    other = _sleeper(marker='something-else')
    try:
        assert rp.job_liveness({'pid': carrier.pid}, spec.job_id)[0] == rp.ALIVE
        state, reason = rp.job_liveness({'pid': other.pid}, spec.job_id)
        assert state == rp.DEAD and 'PID REUSE' in reason
    finally:
        for p in (carrier, other):
            p.kill()
            p.wait()


def test_self_pid_is_never_reported_alive(tmp_path):
    state, reason = rp.job_liveness({'pid': os.getpid()}, 'deadbeefdeadbeef')
    assert state == rp.DEAD and 'itself' in reason


def test_unreadable_identity_is_unknown_not_reaped(monkeypatch, tmp_path):
    """When nothing binds a live pid to the job, do NOT reap.

    A wrong reap puts a second job on an occupied card and corrupts timings; a
    missed reap leaves a job for a human. The asymmetry decides.
    """
    p = _sleeper()
    try:
        monkeypatch.setattr(rp, 'read_cmdline', lambda pid: None)
        state, _ = rp.job_liveness({'pid': p.pid}, 'deadbeefdeadbeef')
        assert state == rp.UNKNOWN
    finally:
        p.kill()
        p.wait()


# ---------------------------------------------------------------------------
# Slow-but-alive must survive
# ---------------------------------------------------------------------------
def test_a_long_running_alive_job_is_never_reaped(tmp_path):
    """Reaping is on liveness, not elapsed time. 5+ hour arms exist."""
    q = _q(tmp_path)
    spec = q.pending()[0]
    p = _sleeper(marker='--job-file /runs/%s/job.json' % spec.name)
    try:
        q.mark_running(spec.job_id, 'GPU-uuid-x', p.pid,
                       proc_identity=rp.process_identity(p.pid),
                       started_at=time.time() - 40 * 3600)   # 40 hours in
        logs = []
        r = rp.Reaper(out_dir=str(tmp_path / 'runs'), log=logs.append)
        res = r.sweep(q, apply=True)
        assert q.get(spec.job_id)['status'] == RUNNING
        assert len(res['adopted']) == 1 and not res['reaped']
        # It IS flagged for a human -- warn only, no state change.
        assert any('STALE-WARN' in m for m in logs)
    finally:
        p.kill()
        p.wait()


def test_own_children_are_left_to_popen_poll(tmp_path):
    """A job we are tracking ourselves is never second-guessed by the reaper."""
    q = _q(tmp_path)
    spec = q.pending()[0]
    q.mark_running(spec.job_id, 'GPU-uuid-x', 999999999,
                   proc_identity={'boot_id': 'x', 'starttime_ticks': 1})
    r = rp.Reaper(out_dir=str(tmp_path / 'runs'), log=lambda *_a: None)
    res = r.sweep(q, owned_job_ids=[spec.job_id], apply=True)
    assert q.get(spec.job_id)['status'] == RUNNING
    assert len(res['owned']) == 1 and not res['reaped']


# ---------------------------------------------------------------------------
# Reaping, retry-once, and outcome recovery
# ---------------------------------------------------------------------------
def test_dead_job_is_requeued_with_the_same_id_and_output_dir(tmp_path):
    q = _q(tmp_path)
    spec = q.pending()[0]
    out = str(tmp_path / 'runs')
    before = spec.output_dir(out)

    p = _sleeper()
    pid = p.pid
    q.mark_running(spec.job_id, 'GPU-uuid-x', pid,
                   proc_identity=rp.process_identity(pid))
    p.kill()
    p.wait()

    r = rp.Reaper(out_dir=out, log=lambda *_a: None)
    res = r.sweep(q, apply=True)
    assert len(res['reaped']) == 1
    rec = q.get(spec.job_id)
    assert rec['status'] == PENDING and rec['reaps'] == 1
    assert rec['pid'] is None and rec['proc_identity'] is None
    # RESUME: same id => same directory => the journal is found and replayed.
    assert q.pending()[0].job_id == spec.job_id
    assert q.pending()[0].output_dir(out) == before


def test_retry_once_then_needs_attention(tmp_path):
    """A job that keeps dying must not loop forever."""
    q = _q(tmp_path)
    spec = q.pending()[0]
    r = rp.Reaper(out_dir=str(tmp_path / 'runs'), log=lambda *_a: None)

    for expected, status in ((1, PENDING), (2, NEEDS_ATTENTION)):
        p = _sleeper()
        pid = p.pid
        q.mark_running(spec.job_id, 'GPU-uuid-x', pid,
                       proc_identity=rp.process_identity(pid))
        p.kill()
        p.wait()
        r.sweep(q, apply=True)
        rec = q.get(spec.job_id)
        assert rec['reaps'] == expected
        assert rec['status'] == status, (expected, rec['status'])

    # Parked, and therefore not dispatchable.
    assert not q.pending()
    assert len(q.by_status(NEEDS_ATTENTION)) == 1
    assert len(q.get(spec.job_id)['reap_history']) == 2


def test_finished_job_is_recovered_not_rerun(tmp_path):
    """A scheduler killed just after a job finished must not re-run that job."""
    q = _q(tmp_path)
    spec = q.pending()[0]
    out = str(tmp_path / 'runs')

    p = _sleeper()
    pid = p.pid
    q.mark_running(spec.job_id, 'GPU-uuid-x', pid,
                   proc_identity=rp.process_identity(pid))
    d = spec.output_dir(out)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'result.json'), 'w') as fh:
        json.dump({'job_id': spec.job_id, 'status': 'done',
                   'log_z_repr': '-1.5'}, fh)
    p.kill()
    p.wait()

    r = rp.Reaper(out_dir=out, log=lambda *_a: None)
    res = r.sweep(q, apply=True)
    assert len(res['recovered']) == 1 and not res['reaped']
    assert q.get(spec.job_id)['status'] == DONE
    assert not q.pending()


def test_stale_result_json_from_a_previous_attempt_is_ignored(tmp_path):
    """result.json older than this attempt's dispatch must not be trusted."""
    q = _q(tmp_path)
    spec = q.pending()[0]
    out = str(tmp_path / 'runs')
    d = spec.output_dir(out)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'result.json'), 'w') as fh:
        json.dump({'status': 'failed'}, fh)
    old = time.time() - 10_000
    os.utime(os.path.join(d, 'result.json'), (old, old))

    p = _sleeper()
    pid = p.pid
    q.mark_running(spec.job_id, 'GPU-uuid-x', pid,
                   proc_identity=rp.process_identity(pid),
                   started_at=time.time())
    p.kill()
    p.wait()

    r = rp.Reaper(out_dir=out, log=lambda *_a: None)
    res = r.sweep(q, apply=True)
    assert len(res['reaped']) == 1 and not res['recovered']
    assert q.get(spec.job_id)['status'] == PENDING


def test_scan_is_read_only(tmp_path):
    q = _q(tmp_path)
    spec = q.pending()[0]
    q.mark_running(spec.job_id, 'GPU-uuid-x', 999999999,
                   proc_identity={'boot_id': 'x', 'starttime_ticks': 1})
    r = rp.Reaper(out_dir=str(tmp_path / 'runs'), log=lambda *_a: None)
    rows = r.scan(q)
    assert rows[0]['state'] == rp.DEAD
    assert q.get(spec.job_id)['status'] == RUNNING       # untouched
    # apply=False likewise changes nothing
    r.sweep(q, apply=False)
    assert q.get(spec.job_id)['status'] == RUNNING


def test_orphan_ballast_is_reported_not_killed_when_not_ours(monkeypatch):
    """Another agent's ballast must never be torn down by us."""
    from nce.scheduler import gpus as gpumod
    fake = [gpumod.GpuInfo(0, 'GPU-a', 'TITAN', 1, 24000, 0, (4242,))]
    monkeypatch.setattr(rp, 'read_cmdline',
                        lambda pid: 'python -m nce.scheduler.ballast --gpu 0 '
                                    '--stop-file /somewhere/else/b.stop')
    monkeypatch.setattr(gpumod, 'query_gpus', lambda **kw: fake)
    found = rp.orphan_ballast(log_dir='/home/x/runs/_ballast')
    assert len(found) == 1 and found[0]['ours'] is False
    assert rp.terminate_orphan_ballast(found[0], log=lambda *_a: None) is False
