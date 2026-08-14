"""Subprocess harness for the checkpoint/resume bit-identity test.

Runs one elimination in a fresh process, optionally journaling, optionally
crashing after K clusters, optionally resuming. Writes `harness_result.json`
into the run's out_dir.

Kept separate from the test module so each run really is an independent
process with a cold CUDA context and a fresh RNG -- an in-process test could
pass for reasons that have nothing to do with the checkpoint being correct.
"""

from __future__ import annotations

import json
import os
import sys
import traceback


class _InjectedCrash(RuntimeError):
    """Simulated mid-elimination death, raised at a deterministic cluster."""


def main(spec_path: str) -> int:
    with open(spec_path) as fh:
        spec = json.load(fh)

    out_dir = spec['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, 'harness_result.json')

    import torch
    torch.set_num_threads(int(spec.get('threads') or 1))

    from nce.config_schema import prepare_config
    from nce.inference.graphical_model import FastGM
    from nce.benchmark_problems.catalog_utils import get_catalog
    from nce.scheduler.checkpoint import (CheckpointStore,
                                          checkpointed_elimination,
                                          journal_fingerprint)

    cfg = prepare_config(dict(spec['config']), strict=False)
    model = get_catalog()[spec['problem_key']]

    out = {'status': 'unknown', 'log_z_repr': None,
           'clusters_replayed': 0, 'clusters_computed': 0,
           'journal_steps': 0}

    store = None
    if spec.get('checkpoint'):
        store = CheckpointStore(os.path.join(out_dir, 'checkpoint'))

    # Crash injection sits OUTSIDE the checkpoint wrapper, so the journal entry
    # for the cluster we die on has already been written -- which is what a real
    # crash between clusters looks like.
    crash_after = spec.get('crash_after')
    orig_process = FastGM.process_bucket
    counter = {'n': 0}

    def crashing_process(self, bucket, exact=False):
        if getattr(self, 'is_populating_backward_factors', False):
            return orig_process(self, bucket, exact=exact)
        if crash_after is not None and counter['n'] >= crash_after:
            raise _InjectedCrash('simulated crash after %d clusters' % crash_after)
        counter['n'] += 1
        return orig_process(self, bucket, exact=exact)

    try:
        gm = FastGM(model=model, nn_config=cfg, device=cfg['device'])
        if crash_after is not None:
            FastGM.process_bucket = crashing_process
        try:
            if store is not None:
                fp = journal_fingerprint(spec['problem_key'], cfg)
                with checkpointed_elimination(
                        gm, store, resume=bool(spec.get('resume')),
                        fingerprint=fp) as cstate:
                    gm.eliminate_variables(all=True)
                out['clusters_replayed'] = cstate['replayed']
                out['clusters_computed'] = cstate['computed']
            else:
                gm.eliminate_variables(all=True)
        finally:
            FastGM.process_bucket = orig_process
        out['log_z_repr'] = repr(float(gm.log_partition_function))
        out['status'] = 'done'
    except _InjectedCrash as e:
        out['status'] = 'crashed'
        out['error'] = str(e)
    except Exception as e:
        out['status'] = 'error'
        out['error'] = str(e)
        out['traceback'] = traceback.format_exc()
        print(out['traceback'], file=sys.stderr)

    if store is not None:
        try:
            out['journal_steps'] = CheckpointStore(
                os.path.join(out_dir, 'checkpoint')).n_steps
        except Exception:
            pass

    with open(result_path, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out))
    return 0 if out['status'] in ('done', 'crashed') else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
