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
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

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
    try:
        model = get_catalog()[spec.problem_key]
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

    manifest.status = status
    manifest.timing['total_wall_s'] = time.time() - t_start
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
