"""Does a resumed run produce a BIT-IDENTICAL log Z to an uninterrupted one?

This is the load-bearing test of the checkpoint system. A resume that perturbs
results silently invalidates every rerun built on it, which is strictly worse
than having no checkpointing at all -- so the claim has to be measured, not
assumed.

Three runs, all in SEPARATE PROCESSES (a same-process test would share a warm
CUDA context and RNG history and could pass for the wrong reason):

  A  uninterrupted, checkpointing DISABLED   -> baseline truth
  B  uninterrupted, checkpointing ENABLED    -> proves journaling is inert
  C  crashed after K clusters, then resumed  -> proves resume is exact

The assertion is `repr(float(log_z))` equality, which is the exact-bits
comparison the determinism suite uses -- not `math.isclose`.

Default tier is CPU so this runs in the normal suite. `--gpu` runs the CUDA
tier, which is what the reruns actually use.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'resume_harness.py')

# pedigree1 at iB=8 gives ~10 NN clusters, so there is genuinely something to
# resume past; num_epochs is small to keep the test cheap.
CPU_CFG = dict(neurobe_mode=True, iB=8, ecl=1025, sampling_scheme='uniform',
               stream_nn_exact=True, device='cpu', seed=42,
               use_reduce_nn_merge=True, reduce_nn_backtrack=True,
               max_merge_bound=6, dope_factors=True, num_samples=2048,
               num_epochs=5, verbose_merge=False)
CPU_PROBLEM = 'pedigree/pedigree1'
CRASH_AFTER = 25          # clusters completed before the simulated crash


def _run(tmpdir, name, problem, cfg, checkpoint, crash_after=None,
         resume=False, gpu=None, threads=1):
    out = os.path.join(str(tmpdir), name)
    payload = {'problem_key': problem, 'config': cfg, 'out_dir': out,
               'checkpoint': checkpoint, 'crash_after': crash_after,
               'resume': resume, 'threads': threads}
    spec = os.path.join(str(tmpdir), name + '.json')
    with open(spec, 'w') as fh:
        json.dump(payload, fh)
    env = dict(os.environ)
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env['PYTHONPATH'] = REPO + os.pathsep + env.get('PYTHONPATH', '')
    proc = subprocess.run([sys.executable, HARNESS, spec], env=env,
                          capture_output=True, text=True, cwd=REPO)
    res_path = os.path.join(out, 'harness_result.json')
    result = None
    if os.path.exists(res_path):
        with open(res_path) as fh:
            result = json.load(fh)
    return proc, result


@pytest.fixture(scope='module')
def gpu_opt(request):
    return request.config.getoption('--gpu', default=False)


def test_resume_is_bit_identical_cpu(tmp_path):
    """CPU tier: A == B == C, bit for bit."""
    _, a = _run(tmp_path, 'A', CPU_PROBLEM, CPU_CFG, checkpoint=False)
    assert a and a['status'] == 'done', 'baseline run A failed: %r' % (a,)

    _, b = _run(tmp_path, 'B', CPU_PROBLEM, CPU_CFG, checkpoint=True)
    assert b and b['status'] == 'done', 'run B failed: %r' % (b,)
    assert b['log_z_repr'] == a['log_z_repr'], (
        'Enabling checkpointing CHANGED the answer.\n'
        '  no-checkpoint : %s\n  checkpointed  : %s'
        % (a['log_z_repr'], b['log_z_repr']))

    # C1: crash partway. Expected to fail; it must still leave a usable journal.
    _, c1 = _run(tmp_path, 'C', CPU_PROBLEM, CPU_CFG, checkpoint=True,
                 crash_after=CRASH_AFTER)
    assert c1 and c1['status'] == 'crashed', (
        'the simulated crash did not happen: %r' % (c1,))
    assert c1['journal_steps'] >= CRASH_AFTER, (
        'journal holds %d steps, expected >= %d'
        % (c1['journal_steps'], CRASH_AFTER))

    # C2: resume into the SAME out_dir, which is where the journal lives.
    _, c2 = _run(tmp_path, 'C', CPU_PROBLEM, CPU_CFG, checkpoint=True,
                 resume=True)
    assert c2 and c2['status'] == 'done', 'resume run failed: %r' % (c2,)
    assert c2['clusters_replayed'] >= CRASH_AFTER, (
        'resume recomputed work it should have replayed: replayed=%d'
        % c2['clusters_replayed'])
    assert c2['log_z_repr'] == a['log_z_repr'], (
        'RESUMED RUN IS NOT BIT-IDENTICAL.\n'
        '  uninterrupted : %s\n  resumed       : %s\n'
        '  replayed=%d computed=%d'
        % (a['log_z_repr'], c2['log_z_repr'],
           c2['clusters_replayed'], c2['clusters_computed']))


def test_resume_refuses_mismatched_journal(tmp_path):
    """A journal from a DIFFERENT config must be rejected, not replayed.

    Replaying someone else's messages would produce a confidently wrong log Z,
    so the failure mode has to be a loud exception.
    """
    cfg_a = dict(CPU_CFG)
    _, a = _run(tmp_path, 'M', CPU_PROBLEM, cfg_a, checkpoint=True,
                crash_after=10)
    assert a and a['status'] == 'crashed'

    # Same out_dir (=> same journal), different elimination structure.
    cfg_b = dict(CPU_CFG, iB=10, max_merge_bound=8)
    proc, b = _run(tmp_path, 'M', CPU_PROBLEM, cfg_b, checkpoint=True,
                   resume=True)
    assert b is None or b['status'] != 'done', (
        'resumed a journal built from a different config and reported success')
    combined = (proc.stdout or '') + (proc.stderr or '') + json.dumps(b or {})
    assert ('DIFFERENT computation' in combined or 'diverged' in combined), (
        'expected an explicit refusal naming the mismatch, got:\n%s'
        % combined[-2000:])


@pytest.mark.gpu
def test_resume_is_bit_identical_cuda(tmp_path, gpu_opt):
    """CUDA tier -- the configuration the reruns actually use.

    GPU 3 by default: 0 and 1 are in use by other agents and 2 is RETIRED.
    """
    if not gpu_opt:
        pytest.skip('needs --gpu')
    from nce.scheduler.gpus import assert_not_retired
    gpu = int(os.environ.get('NCE_TEST_GPU', '3'))
    assert_not_retired(gpu)
    cfg = dict(CPU_CFG, device='cuda')

    _, a = _run(tmp_path, 'GA', CPU_PROBLEM, cfg, checkpoint=False, gpu=gpu)
    assert a and a['status'] == 'done', 'baseline failed: %r' % (a,)

    _, c1 = _run(tmp_path, 'GC', CPU_PROBLEM, cfg, checkpoint=True,
                 crash_after=CRASH_AFTER, gpu=gpu)
    assert c1 and c1['status'] == 'crashed'

    _, c2 = _run(tmp_path, 'GC', CPU_PROBLEM, cfg, checkpoint=True,
                 resume=True, gpu=gpu)
    assert c2 and c2['status'] == 'done', 'resume failed: %r' % (c2,)
    assert c2['log_z_repr'] == a['log_z_repr'], (
        'RESUMED CUDA RUN IS NOT BIT-IDENTICAL.\n'
        '  uninterrupted : %s\n  resumed       : %s'
        % (a['log_z_repr'], c2['log_z_repr']))
