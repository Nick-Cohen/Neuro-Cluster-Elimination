"""Run ONE rerun job, with provenance, timing split, and checkpoint/resume.

This is a subprocess entry point, not a library call, for one specific reason:
`CUDA_VISIBLE_DEVICES` must be set BEFORE torch is imported, so pinning a job to
a GPU can only be done by a fresh process. It is also the isolation boundary
that keeps one job's crash from taking down the scheduler.

Usage (the scheduler does this for you):

    python -m nce.scheduler.runner --job-file job.json --out-dir DIR --gpu 3

Outputs, all under --out-dir:
    manifest.json    full provenance + timing + result (see provenance.py)
    result.json      just the outcome, for cheap aggregation
    checkpoint/      message journal (see checkpoint.py)
    runner.log       stdout/stderr of this process
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job-file', required=True,
                   help='JSON file holding one JobSpec dict')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--gpu', type=int, default=None,
                   help='PHYSICAL gpu index to pin to via CUDA_VISIBLE_DEVICES')
    p.add_argument('--threads', type=int, default=None,
                   help='torch.set_num_threads. Pin it: the determinism suite '
                        'MEASURED a 2-float32-ULP shift between 1 and 4 threads '
                        'on an otherwise identical run.')
    p.add_argument('--no-checkpoint', action='store_true')
    p.add_argument('--no-resume', action='store_true',
                   help='checkpoint but start from scratch, ignoring any journal')
    p.add_argument('--base-config', default=None,
                   help='JSON file of base config applied under the job spec')
    p.add_argument('--ballast-pause-file', default=None,
                   help='touch this to make thermal ballast stop feeding the '
                        'card. Set by the scheduler when it hands off a WARM '
                        'card (see pause_ballast).')
    p.add_argument('--ballast-paused-marker', default=None,
                   help='the worker writes this once it has actually stopped; '
                        'the job refuses to start GPU work until it appears.')
    p.add_argument('--launched-at', type=float, default=None,
                   help='epoch seconds at which the scheduler called Popen, so '
                        'process spawn + interpreter start can be attributed.')
    return p.parse_args(argv)


def _ballast_pids_on(gpu_index):
    """PIDs holding a CUDA context on this PHYSICAL gpu, straight from nvidia-smi.

    Deliberately not `os.kill(pid, 0)`: a ballast worker that has exited but not
    yet been reaped by the scheduler is a ZOMBIE, and `os.kill` succeeds for
    zombies. A zombie has already released its CUDA context, so waiting on
    os.kill would hang forever on a card that is in fact free. The driver's own
    compute-apps list is the authoritative answer to the only question that
    matters here -- "is a context still attached" -- and it is a fact, not a
    threshold.

    `gpu_index` must be the PHYSICAL index. The runner has already set
    CUDA_VISIBLE_DEVICES, but nvidia-smi goes through NVML and ignores it --
    verified on this box: with CUDA_VISIBLE_DEVICES=3, `-i 3` still returns
    GPU 3's uuid while `-i 0` returns GPU 0's. Passing a remapped index here
    would query the wrong card and could clear a job to start while ballast is
    still running on its own card.
    """
    import subprocess as _sp
    out = _sp.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                   '--format=csv,noheader,nounits'],
                  capture_output=True, text=True)
    if out.returncode != 0:
        return None
    uuid = None
    q = _sp.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader',
                 '-i', str(gpu_index)], capture_output=True, text=True)
    if q.returncode == 0:
        uuid = q.stdout.strip()
    pids = set()
    for line in out.stdout.splitlines():
        parts = [c.strip() for c in line.split(',')]
        if len(parts) >= 2 and (uuid is None or parts[0] == uuid):
            try:
                pids.add(int(parts[1]))
            except ValueError:
                pass
    return pids


def pause_ballast(pause_file, paused_marker, gpu_index, timeout=None):
    """Take the card from thermal ballast, and BLOCK until it has really let go.

    Called immediately before the job's first GPU work. Everything before this
    point -- process spawn, torch import, model validation, catalog load -- is
    CPU-bound, is outside every timed region the rerun reports, and is exactly
    the window in which the card would otherwise cool from ~83 C to ~72 C in
    8-9 s (doc 60 section 6). Letting ballast own that window is what makes a
    job start at equilibrium without a per-job warm-up.

    Ballast PAUSES rather than exits: restarting a worker costs ~5 s of process
    spawn plus torch import, and with a saturated queue the next job is
    dispatched ~0.4 s after the previous finishes, so an exiting worker could
    never be back in time to cover the next job's setup. A paused worker frees
    its matrices and queues no kernels, holding only the bare CUDA context.

    The contract this must not break: ballast must be queueing NO work before
    any timed work starts. So it does not merely signal -- it waits for the
    worker's own attestation (`paused_marker`, written after it has synchronised
    and freed), and raises if that never arrives. Running the job anyway would
    produce a silently perturbed timing, which is worse than a failed job.
    """
    from nce.scheduler.ballast import BALLAST_STOP_TIMEOUT_S
    timeout = BALLAST_STOP_TIMEOUT_S if timeout is None else timeout
    t0 = time.time()
    try:
        os.remove(paused_marker)
    except OSError:
        pass
    with open(pause_file, 'w') as fh:
        fh.write('%f\n' % t0)

    while time.time() - t0 < timeout:
        if os.path.exists(paused_marker):
            waited = time.time() - t0
            print('[runner] ballast paused cuda:%d in %.3fs'
                  % (gpu_index, waited), flush=True)
            return {'ballast_pause_s': round(waited, 4)}
        time.sleep(0.02)
    raise RuntimeError(
        'pause_ballast: ballast on cuda:%d did not confirm pause within %.1fs. '
        'Refusing to start GPU work -- a job timed against a card that is also '
        'running ballast is silently wrong.' % (gpu_index, timeout))


def yield_ballast(stop_file, ballast_pid, gpu_index, timeout=None):
    """Make thermal ballast let go of this card, and BLOCK until it really has.

    Called immediately before the job's first GPU work. Everything before this
    point -- process spawn, torch import, model validation, catalog load -- is
    CPU-bound, is outside every timed region the rerun reports, and is exactly
    the window in which the card would otherwise cool from ~83 C to ~49 C
    (doc 59, and doc 60 section 6). Letting ballast own that window is what makes
    a job start warm without a per-job warm-up.

    The contract this function must not break: ballast must be COMPLETELY off
    the card before any timed work starts. So it does not merely signal, it
    waits for the driver to stop reporting the ballast pid, and it raises if
    that never happens. Running the job anyway would produce a silently
    perturbed timing, which is worse than a failed job.

    Returns a dict of diagnostics for the manifest.
    """
    from nce.scheduler.ballast import BALLAST_STOP_TIMEOUT_S
    timeout = BALLAST_STOP_TIMEOUT_S if timeout is None else timeout
    t0 = time.time()
    with open(stop_file, 'w') as fh:
        fh.write('%f\n' % t0)

    waited = None
    while time.time() - t0 < timeout:
        pids = _ballast_pids_on(gpu_index)
        if pids is None:
            # nvidia-smi unavailable: cannot VERIFY the card is clear, so do not
            # pretend. Fail rather than time a job against an unknown card.
            raise RuntimeError(
                'yield_ballast: nvidia-smi unavailable, cannot verify that '
                'ballast released cuda:%d. Refusing to start timed work.'
                % gpu_index)
        if ballast_pid not in pids:
            waited = time.time() - t0
            break
        time.sleep(0.05)
    if waited is None:
        raise RuntimeError(
            'yield_ballast: ballast pid %d still holds cuda:%d after %.1fs. '
            'Refusing to start GPU work -- a job timed against a card that is '
            'also running ballast is silently wrong.'
            % (ballast_pid, gpu_index, timeout))
    print('[runner] ballast yielded cuda:%d in %.3fs' % (gpu_index, waited),
          flush=True)
    return {'ballast_yield_s': round(waited, 4), 'ballast_pid': ballast_pid}


def main(argv=None) -> int:
    args = _parse_args(argv)
    # Phase accounting. Doc 60 section 6.2 measured the GPU genuinely busy only
    # 43-45% of job wall time; these marks say where the rest goes, so per-job
    # overhead can be attacked on evidence instead of guesswork.
    t_entry = time.time()
    marks = {}
    if args.launched_at:
        marks['spawn_and_interpreter_s'] = t_entry - args.launched_at

    # ---- GPU pinning MUST happen before torch is imported ----------------
    if args.gpu is not None:
        # Import only the guard, which does not pull in torch.
        from nce.scheduler.gpus import assert_not_retired
        assert_not_retired(args.gpu)   # hard refusal for cuda:2
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    os.makedirs(args.out_dir, exist_ok=True)

    import torch
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    from nce.config_schema import prepare_config
    from nce.inference.graphical_model import FastGM
    from nce.benchmark_problems.catalog_utils import get_catalog
    from nce.scheduler.jobs import JobSpec
    from nce.scheduler.provenance import capture
    from nce.scheduler.timing import collect_timings
    from nce.scheduler.checkpoint import (CheckpointStore,
                                          checkpointed_elimination,
                                          journal_fingerprint,
                                          verify_environment)
    from nce.scheduler import models as modelval
    marks['imports_s'] = time.time() - t_entry

    # Fail here, loudly, rather than 8 s later with a per-job "model missing".
    modelval.assert_cache_configured()

    with open(args.job_file) as fh:
        spec = JobSpec.from_dict(json.load(fh))
    base = {}
    if args.base_config:
        with open(args.base_config) as fh:
            base = json.load(fh)

    manifest_path = os.path.join(args.out_dir, 'manifest.json')
    result_path = os.path.join(args.out_dir, 'result.json')

    # ---- validate the model BEFORE doing anything expensive --------------
    mv = modelval.validate(spec.problem_key)
    if not mv['ok']:
        payload = {'job_id': spec.job_id, 'status': 'blocked',
                   'reason': 'model validation failed', 'problems': mv['problems']}
        with open(result_path, 'w') as fh:
            json.dump(payload, fh, indent=2)
        print('BLOCKED: %s' % '; '.join(mv['problems']), file=sys.stderr)
        return 3

    raw_cfg = spec.to_config(base)
    raw_cfg.setdefault('problem_key', spec.problem_key)
    full = prepare_config(raw_cfg, strict=False)

    manifest = capture(run_id=spec.job_id, resolved_config=full,
                       job=spec.to_dict(),
                       model_info={k: mv[k] for k in
                                   ('paths', 'size_bytes', 'uai_check',
                                    'has_evidence')})
    manifest.write(manifest_path)

    # ---- resume bookkeeping ---------------------------------------------
    store = None
    ckpt_dir = os.path.join(args.out_dir, 'checkpoint')
    if not args.no_checkpoint:
        store = CheckpointStore(ckpt_dir)
        if store.n_steps and not args.no_resume:
            # A resume across a changed environment is not bit-identical.
            # Say so loudly rather than producing a quietly-wrong rerun.
            if os.path.exists(manifest_path):
                warns = verify_environment(json.load(open(manifest_path)))
                for w in warns:
                    print('RESUME WARNING: %s' % w, flush=True)
                manifest.result['resume_warnings'] = warns

    t_start = time.time()
    status, err = 'done', None
    log_z = None
    ballast_info = {}
    try:
        t0 = time.time()
        model = get_catalog()[spec.problem_key]
        marks['model_load_s'] = time.time() - t0

        # ---- LAST CPU-ONLY MOMENT: drop ballast, then touch the GPU -------
        # FastGM.__init__ uploads factors to the device, so it is the first GPU
        # work in this process. Everything above is CPU. Ballast has been
        # holding the card at working temperature through all of it.
        if args.ballast_pause_file and args.ballast_paused_marker:
            t0 = time.time()
            ballast_info = pause_ballast(args.ballast_pause_file,
                                         args.ballast_paused_marker, args.gpu)
            marks['ballast_pause_s'] = time.time() - t0

        t0 = time.time()
        gm = FastGM(model=model, nn_config=full, device=full['device'])
        t_build = time.time() - t0

        t0 = time.time()
        with collect_timings() as tc:
            if store is not None:
                fp = journal_fingerprint(spec.problem_key, full)
                with checkpointed_elimination(
                        gm, store, resume=not args.no_resume,
                        fingerprint=fp) as cstate:
                    gm.eliminate_variables(all=True)
                manifest.result['clusters_replayed'] = cstate['replayed']
                manifest.result['clusters_computed'] = cstate['computed']
            else:
                gm.eliminate_variables(all=True)
        t_elim = time.time() - t0

        # repr() not float(): the determinism suite compares log Z by repr,
        # which is the exact-bits comparison this rerun needs.
        log_z = repr(float(gm.log_partition_function))
        manifest.timing = dict(tc.report())
        manifest.timing['t_build_s'] = t_build
        manifest.timing['t_elim_s'] = t_elim
        manifest.timing['phase_times_s'] = dict(getattr(gm, 'phase_times', {}))
        manifest.result.update({
            'log_z_repr': log_z,
            'num_trained': getattr(gm, 'num_trained', None),
            'epochs': [d.get('epochs_trained')
                       for d in getattr(gm, 'per_bucket_training_log', [])
                       if d.get('epochs_trained') is not None],
        })
        if full.get('device') == 'cuda' and torch.cuda.is_available():
            manifest.result['cuda_peak_alloc_gb'] = (
                torch.cuda.max_memory_allocated() / 2 ** 30)
    except Exception as e:
        status, err = 'failed', traceback.format_exc()
        manifest.result['error'] = str(e)
        manifest.result['traceback'] = err
        print(err, file=sys.stderr)

    # Hand the card straight back to ballast. The scheduler only reaps on its
    # poll boundary (default 30 s), so waiting for it would leave the card idle
    # and cooling for up to a full interval after the job's GPU work is already
    # done -- measured at ~24 s on a 36 s job, which is far more idle time than
    # the setup window ballast was added to cover. Nothing timed remains at this
    # point: only manifest/result writes and interpreter teardown.
    if args.ballast_pause_file:
        try:
            os.remove(args.ballast_pause_file)
            print('[runner] released ballast on cuda:%s' % args.gpu, flush=True)
        except OSError:
            pass

    manifest.status = status
    manifest.timing['total_wall_s'] = time.time() - t_start
    marks['pre_timer_s'] = t_start - t_entry
    marks['post_elim_to_write_s'] = time.time() - t_start - marks.get(
        'model_load_s', 0.0) - manifest.timing.get('t_build_s', 0.0) \
        - manifest.timing.get('t_elim_s', 0.0) - marks.get('ballast_pause_s', 0.0)
    manifest.timing['phases_s'] = {k: round(v, 4) for k, v in marks.items()}
    manifest.timing.update(ballast_info)
    manifest.write(manifest_path)
    with open(result_path, 'w') as fh:
        json.dump({'job_id': spec.job_id, 'name': spec.name, 'status': status,
                   'log_z_repr': log_z,
                   'timing': manifest.timing.get('totals'),
                   'total_wall_s': manifest.timing['total_wall_s']},
                  fh, indent=2, default=str)
    print('%s %s log_z=%s' % (status.upper(), spec.name, log_z), flush=True)
    return 0 if status == 'done' else 1


if __name__ == '__main__':
    sys.exit(main())
