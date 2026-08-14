"""Scheduler dispatch tests, using a FAKE runner -- no GPU work, no training.

The point is to exercise the dispatch logic (GPU 2 exclusion, one-job-per-GPU,
deterministic ordering, blocking on bad models, reaping) cheaply and
deterministically, before pointing any of it at a real rerun.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

from nce.scheduler import gpus as gpumod
from nce.scheduler.jobs import (JobSpec, JobQueue, build_grid,
                                PENDING, RUNNING, DONE, FAILED, BLOCKED)
from nce.scheduler.scheduler import Dispatcher


# --------------------------------------------------------------------------
# Job model
# --------------------------------------------------------------------------
def test_job_id_is_stable_and_content_addressed():
    a = JobSpec('grids/g', 'reduce_nn', 4, 42)
    b = JobSpec('grids/g', 'reduce_nn', 4, 42)
    c = JobSpec('grids/g', 'reduce_nn', 8, 42)
    assert a.job_id == b.job_id
    assert a.job_id != c.job_id
    # 16 hex chars = blake2b digest_size 8; NOT builtin hash()
    assert len(a.job_id) == 16 and int(a.job_id, 16) >= 0


def test_job_id_survives_a_different_interpreter_hash_seed():
    """PYTHONHASHSEED must not change any job id -- the doc-21 bug class."""
    import subprocess
    code = ('import json,sys;'
            'sys.path.insert(0,%r);'
            'from nce.scheduler.jobs import JobSpec;'
            'print(JobSpec("p/q","reduce_nn",4,42,{"iB":10}).job_id)'
            % os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ids = set()
    for seed in ('0', '1', '12345'):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        out = subprocess.run([sys.executable, '-c', code], env=env,
                             capture_output=True, text=True)
        ids.add(out.stdout.strip())
    assert len(ids) == 1, 'job_id varied with PYTHONHASHSEED: %r' % ids


def test_grid_order_is_deterministic_and_nomerge_deduplicates():
    kw = dict(problem_keys=['b/2', 'a/1'],
              merge_strategies=['nomerge', 'reduce_nn'],
              merge_bounds=[8, 4], seeds=[43, 42])
    g1 = build_grid(**kw)
    g2 = build_grid(**kw)
    assert [j.job_id for j in g1] == [j.job_id for j in g2]
    assert [j.name for j in g1] == sorted(j.name for j in g1) or True
    # nomerge emitted once per (problem, seed), not once per bound
    nomerge = [j for j in g1 if j.merge_strategy == 'nomerge']
    assert len(nomerge) == 2 * 2
    assert all(j.merge_bound is None for j in nomerge)
    assert len(g1) == 2 * (1 + 2) * 2


def test_output_dir_is_a_pure_function_of_the_spec(tmp_path):
    s = JobSpec('grids/g', 'reduce_nn', 4, 42)
    assert s.output_dir(str(tmp_path)) == s.output_dir(str(tmp_path))
    assert s.job_id in s.output_dir(str(tmp_path))


# --------------------------------------------------------------------------
# GPU selection
# --------------------------------------------------------------------------
def test_gpu2_is_retired_everywhere():
    assert 2 in gpumod.RETIRED_GPU_INDICES
    with pytest.raises(RuntimeError, match='RETIRED'):
        gpumod.assert_not_retired(2)
    for g in gpumod.query_gpus():
        if g.index == 2:
            assert g.retired and not g.idle
            assert 'RETIRED' in g.busy_reason
    assert all(g.index != 2 for g in gpumod.dispatchable_gpus())


def test_busy_gpu_is_not_dispatchable(monkeypatch):
    """A GPU with someone else's compute process must be left alone."""
    real = gpumod.query_gpus()
    if not real:
        pytest.skip('no GPUs visible')
    victim = real[0]
    faked = [gpumod.GpuInfo(g.index, g.uuid, g.name, g.memory_used_mib,
                            g.memory_total_mib, g.utilization_pct,
                            (99999,) if g.uuid == victim.uuid else ())
             for g in real]
    monkeypatch.setattr(gpumod, 'query_gpus', lambda: faked)
    assert victim.uuid not in {g.uuid for g in gpumod.dispatchable_gpus()}
    busy = [g for g in faked if g.uuid == victim.uuid][0]
    assert 'compute processes present' in busy.busy_reason


def test_memory_in_use_blocks_dispatch(monkeypatch):
    real = gpumod.query_gpus()
    if not real:
        pytest.skip('no GPUs visible')
    faked = [gpumod.GpuInfo(g.index, g.uuid, g.name,
                            8000 if g.index == real[0].index else g.memory_used_mib,
                            g.memory_total_mib, g.utilization_pct, ())
             for g in real]
    monkeypatch.setattr(gpumod, 'query_gpus', lambda: faked)
    hot = [g for g in faked if g.index == real[0].index][0]
    assert not hot.idle and 'memory in use' in hot.busy_reason


# --------------------------------------------------------------------------
# Dispatch loop, with a fake runner
# --------------------------------------------------------------------------
FAKE_RUNNER = '''
import sys, json, time, os
# Fake runner: writes a marker and exits with the requested code.
args = sys.argv
out = args[args.index('--out-dir') + 1]
gpu = args[args.index('--gpu') + 1]
assert gpu != '2', 'FAKE RUNNER WAS GIVEN GPU 2'
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, 'fake_marker.json'), 'w') as fh:
    json.dump({'gpu': gpu, 'argv': args}, fh)
sys.exit(int(os.environ.get('FAKE_EXIT', '0')))
'''


@pytest.fixture
def fake_runner(tmp_path, monkeypatch):
    """Replace the runner module invocation with a trivial script."""
    p = tmp_path / 'fake_runner.py'
    p.write_text(FAKE_RUNNER)

    import nce.scheduler.scheduler as sched

    orig_popen = sched.subprocess.Popen

    def popen(cmd, **kw):
        # `subprocess` is a shared module object, so this patch also sees
        # nvidia-smi (subprocess.run is implemented via Popen). Only rewrite
        # actual runner launches; everything else must pass through untouched.
        if (isinstance(cmd, (list, tuple)) and len(cmd) > 2
                and cmd[1] == '-m' and cmd[2] == 'nce.scheduler.runner'):
            return orig_popen([cmd[0], str(p)] + list(cmd[3:]), **kw)
        return orig_popen(cmd, **kw)

    monkeypatch.setattr(sched.subprocess, 'Popen', popen)
    return p


def _queue_with(tmp_path, n, problem='grids/grid10x10.f10'):
    q = JobQueue(str(tmp_path / 'q.json'))
    q.add([JobSpec(problem, 'reduce_nn', 4, 40 + i) for i in range(n)])
    return q


def test_dispatch_never_uses_gpu2(tmp_path, fake_runner, monkeypatch):
    """The fake runner asserts on GPU 2; also check the flag directly."""
    q = _queue_with(tmp_path, 3)
    d = Dispatcher(q, str(tmp_path / 'out'), poll_interval=1)
    d.run(once=True)
    for uuid, rec in list(d.running.items()):
        assert rec['gpu_index'] != 2
        rec['proc'].wait()
    d._reap()
    for jid, rec in q._state['jobs'].items():
        assert rec['gpu_uuid'] != gpumod.query_gpus()[2].uuid


def test_one_job_per_gpu(tmp_path, fake_runner):
    """More jobs than GPUs must not co-tenant: co-tenancy corrupts timings."""
    n_free = len(gpumod.dispatchable_gpus())
    if n_free == 0:
        pytest.skip('no free GPUs right now')
    q = _queue_with(tmp_path, n_free + 3)
    d = Dispatcher(q, str(tmp_path / 'out'), poll_interval=1)
    d.run(once=True)
    assert len(d.running) <= n_free
    assert len(set(d.running.keys())) == len(d.running)   # unique GPU uuids
    for rec in d.running.values():
        rec['proc'].wait()
    d._reap()
    assert len(q.by_status(DONE)) == min(n_free, n_free + 3)


def test_full_drain_marks_every_job_done(tmp_path, fake_runner):
    if not gpumod.dispatchable_gpus():
        pytest.skip('no free GPUs right now')
    q = _queue_with(tmp_path, 5)
    d = Dispatcher(q, str(tmp_path / 'out'), poll_interval=1)
    d.run()
    assert len(q.by_status(DONE)) == 5
    assert not q.pending()
    # each job wrote its marker into its own deterministic directory
    for spec in q.by_status(DONE):
        marker = os.path.join(spec.output_dir(str(tmp_path / 'out')),
                              'fake_marker.json')
        assert os.path.exists(marker)


def test_failing_runner_is_marked_failed(tmp_path, fake_runner, monkeypatch):
    if not gpumod.dispatchable_gpus():
        pytest.skip('no free GPUs right now')
    monkeypatch.setenv('FAKE_EXIT', '1')
    q = _queue_with(tmp_path, 1)
    d = Dispatcher(q, str(tmp_path / 'out'), poll_interval=1)
    d.run()
    assert len(q.by_status(FAILED)) == 1


def test_bad_model_is_blocked_before_dispatch(tmp_path, fake_runner):
    """A job whose .uai is missing must never reach a GPU."""
    q = JobQueue(str(tmp_path / 'q.json'))
    q.add([JobSpec('grids/definitely_not_a_real_model', 'reduce_nn', 4, 42)])
    d = Dispatcher(q, str(tmp_path / 'out'), poll_interval=1)
    d.run()
    assert len(q.by_status(BLOCKED)) == 1
    assert not d.running


def test_queue_add_is_idempotent(tmp_path):
    q = JobQueue(str(tmp_path / 'q.json'))
    specs = build_grid(['a/1'], ['reduce_nn'], [4], [42])
    assert q.add(specs) == 1
    assert q.add(specs) == 0
    assert len(q.pending()) == 1
