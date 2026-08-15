"""End-to-end: kill a real scheduler mid-flight, restart it, prove nothing is lost.

WHY THIS FILE EXISTS SEPARATELY FROM test_scheduler_reaper.py
-------------------------------------------------------------
The unit tests exercise `job_liveness` against synthetic records. That is not
enough: the last scheduler bug on this project (a paused ballast worker torn
down on a busy card) passed every unit test and only appeared when the whole
loop ran. So this file drives the ACTUAL `python -m nce.scheduler.scheduler`
process, SIGKILLs it exactly as an OOM killer would, and inspects the queue
file that a human would inspect.

Three things are proved here, in order:

  A  a live orphan is ADOPTED, not reaped -- its card stays reserved, because
     dispatching a second timed job onto an occupied card corrupts timings;
  B  once its process is gone the job is REQUEUED and RE-DISPATCHED;
  C  the re-dispatched job RESUMES from its checkpoint journal instead of
     recomputing from scratch -- otherwise the reaper converts a crash into
     wasted GPU-hours rather than preventing a loss.

GPU POLICY. Dispatch needs a card only because the scheduler pins one per job;
the work here is CPU-bound or a no-op. The card is taken from
`dispatchable_gpus()` and `--only-gpus` restricts the scheduler to it, so a card
another agent is timing on is never touched. Ballast is off for the same reason.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import pytest

from nce.scheduler import gpus as gpumod
from nce.scheduler.jobs import (JobSpec, JobQueue, PENDING, RUNNING, DONE,
                                NEEDS_ATTENTION)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Stands in for `python -m nce.scheduler.runner`. It carries the real runner's
# argv (so its /proc/<pid>/cmdline binds it to the job id exactly as the real
# one does) and then simply waits, so the test controls when the "job" dies.
SHIM = r'''#!%(python)s
import os, sys, time
args = sys.argv
out = args[args.index('--out-dir') + 1]
os.makedirs(out, exist_ok=True)
open(os.path.join(out, 'shim_started'), 'a').write('%%f\n' %% time.time())
while not os.path.exists(os.path.join(out, 'shim_stop')):
    time.sleep(0.1)
sys.exit(0)
'''


def _free_gpu():
    free = [g for g in gpumod.dispatchable_gpus() if not g.retired]
    return free[0] if free else None


def _spawn_scheduler(queue, out_dir, gpu_index, python, extra=()):
    cmd = [sys.executable, '-u', '-m', 'nce.scheduler.scheduler',
           '--queue', queue, '--out-dir', out_dir,
           '--only-gpus', str(gpu_index), '--no-ballast',
           '--poll-interval', '1', '--python', python] + list(extra)
    env = dict(os.environ)
    env.setdefault('NCE_MODEL_CACHE', '/home/cohenn1/NCE/.model_cache')
    env['PYTHONPATH'] = REPO + os.pathsep + env.get('PYTHONPATH', '')
    return subprocess.Popen(cmd, cwd=REPO, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)


def _wait_for(fn, timeout, what):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = fn()
        if v:
            return v
        time.sleep(0.2)
    raise AssertionError('timed out after %.0fs waiting for %s' % (timeout, what))


def _status(queue_path, job_id):
    return JobQueue(queue_path).get(job_id)['status']


def test_kill_scheduler_then_restart_adopts_then_reaps_then_redispatches(tmp_path):
    gpu = _free_gpu()
    if gpu is None:
        pytest.skip('no free GPU right now (two cards are busy for days)')

    shim = tmp_path / 'shim.py'
    shim.write_text(SHIM % {'python': sys.executable})
    shim.chmod(0o755)

    qpath = str(tmp_path / 'q.json')
    out = str(tmp_path / 'runs')
    q = JobQueue(qpath)
    spec = JobSpec('grids/grid10x10.f10', 'reduce_nn', 4, 42, tag='reaper-e2e')
    q.add([spec])
    job_dir = spec.output_dir(out)

    sched = _spawn_scheduler(qpath, out, gpu.index, str(shim))
    shim_pid = None
    try:
        _wait_for(lambda: _status(qpath, spec.job_id) == RUNNING, 90,
                  'the job to be dispatched')
        rec = JobQueue(qpath).get(spec.job_id)
        shim_pid = rec['pid']
        # The identity that makes the reaper safe against PID reuse must have
        # been recorded AT DISPATCH -- there is no way to reconstruct it later.
        assert rec['proc_identity'] and rec['proc_identity']['starttime_ticks']
        assert rec['proc_identity']['boot_id']
        _wait_for(lambda: os.path.exists(os.path.join(job_dir, 'shim_started')),
                  60, 'the job process to start')

        # --- SIGKILL the scheduler: no cleanup, no signal handler, no mercy ---
        os.kill(sched.pid, signal.SIGKILL)
        sched.wait(timeout=30)
        assert _status(qpath, spec.job_id) == RUNNING, (
            'precondition: the dead scheduler leaves the job stuck at running')

        # ---- A. the job process SURVIVED its parent; it must be ADOPTED -----
        assert os.path.exists('/proc/%d' % shim_pid), (
            'precondition: Popen children are reparented, not killed')
        s2 = _spawn_scheduler(qpath, out, gpu.index, str(shim), extra=['--once'])
        log = s2.communicate(timeout=180)[0]
        assert 'ADOPTED' in log, 'live orphan was not adopted:\n%s' % log[-3000:]
        assert _status(qpath, spec.job_id) == RUNNING, (
            'a LIVE job was reaped -- it would now be double-dispatched onto '
            'its own card:\n%s' % log[-3000:])
        assert os.path.exists('/proc/%d' % shim_pid), 'adoption killed the job'
        # Its card must not have been handed to anything else.
        assert 'dispatch] cuda:%d' % gpu.index not in log, (
            'dispatched onto an adopted card:\n%s' % log[-3000:])

        # ---- B. now the process really dies: REQUEUE and RE-DISPATCH -------
        open(os.path.join(job_dir, 'shim_stop'), 'w').close()
        _wait_for(lambda: not os.path.exists('/proc/%d' % shim_pid), 60,
                  'the job process to exit')

        s3 = _spawn_scheduler(qpath, out, gpu.index, str(shim), extra=['--once'])
        log3 = s3.communicate(timeout=180)[0]
        assert 'REQUEUED' in log3, 'dead job not reaped:\n%s' % log3[-3000:]
        rec3 = JobQueue(qpath).get(spec.job_id)
        assert rec3['reaps'] == 1
        # Re-dispatched in the SAME sweep, into the SAME directory -> its
        # checkpoint journal is there to be replayed. (This shim exits at once
        # because its stop-file is already present, so the job may already read
        # `done` by the time the queue is inspected -- either is a re-dispatch.)
        assert '[dispatch] cuda:%d' % gpu.index in log3, (
            'reaped job was not re-dispatched:\n%s' % log3[-3000:])
        assert rec3['status'] in (RUNNING, DONE), rec3
        assert rec3['pid'] != shim_pid
        assert spec.output_dir(out) == job_dir
        shim_pid = rec3['pid']
    finally:
        for pid in [p for p in (shim_pid,) if p]:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except OSError:
                pass
        for p in (sched,):
            if p.poll() is None:
                p.kill()
        # Leave no worker behind on the borrowed card.
        try:
            open(os.path.join(job_dir, 'shim_stop'), 'w').close()
        except OSError:
            pass


def test_reaped_job_resumes_from_its_journal_instead_of_restarting(tmp_path):
    """C. The reaper is only worth having if a re-queued job RESUMES.

    Driven through the real `nce.scheduler.runner`, on CPU, so it is the exact
    code path a reaped job takes: same job_id -> same `output_dir` -> same
    `checkpoint/` -> `CheckpointStore` replays the clusters that already
    finished. If this ever regresses, the reaper turns a crash into a full
    recomputation instead of preventing a loss.
    """
    spec = JobSpec('pedigree/pedigree1', 'reduce_nn', 6, 42,
                   extra_config=dict(neurobe_mode=True, iB=8, ecl=1025,
                                     sampling_scheme='uniform',
                                     stream_nn_exact=True, device='cpu',
                                     dope_factors=True, num_samples=2048,
                                     num_epochs=5, verbose_merge=False),
                   tag='reaper-resume')
    out = spec.output_dir(str(tmp_path))
    os.makedirs(out, exist_ok=True)
    job_file = os.path.join(out, 'job.json')
    with open(job_file, 'w') as fh:
        json.dump(spec.to_dict(), fh, default=str)

    env = dict(os.environ)
    env.setdefault('NCE_MODEL_CACHE', '/home/cohenn1/NCE/.model_cache')
    env['PYTHONPATH'] = REPO + os.pathsep + env.get('PYTHONPATH', '')
    env['CUDA_VISIBLE_DEVICES'] = ''          # CPU only: touch no card at all
    cmd = [sys.executable, '-u', '-m', 'nce.scheduler.runner',
           '--job-file', job_file, '--out-dir', out, '--threads', '1']

    # Attempt 1: killed partway, exactly as a dying scheduler's child would be.
    p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.DEVNULL,
                         stderr=subprocess.STDOUT)
    journal = os.path.join(out, 'checkpoint', 'journal.json')
    t0 = time.time()
    steps = 0
    while time.time() - t0 < 900:
        if os.path.exists(journal):
            try:
                with open(journal) as fh:
                    steps = len(json.load(fh)['steps'])
            except (ValueError, OSError):
                steps = 0
            if steps >= 5:
                break
        if p.poll() is not None:
            break
        time.sleep(0.5)
    p.kill()
    p.wait()
    if steps < 5:
        pytest.skip('job did not journal enough clusters to test a resume')

    # Attempt 2: the re-dispatch. Same directory, no flags -- resume is default.
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    with open(os.path.join(out, 'result.json')) as fh:
        result = json.load(fh)
    with open(os.path.join(out, 'manifest.json')) as fh:
        manifest = json.load(fh)
    assert result['status'] == 'done', (r.stdout or '')[-3000:]
    replayed = manifest['result'].get('clusters_replayed', 0)
    computed = manifest['result'].get('clusters_computed', 0)
    assert replayed >= steps, (
        'the re-dispatched job RECOMPUTED work it had already finished: '
        'replayed=%s computed=%s, journal held %s'
        % (replayed, computed, steps))
    assert computed > 0, 'nothing left to compute; the test proved nothing'
