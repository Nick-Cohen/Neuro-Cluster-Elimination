"""Checkpoint / resume for FastGM elimination, at cluster granularity.

THE DESIGN, AND WHY IT IS THIS ONE
----------------------------------
The obvious design -- serialise the whole `FastGM` after each cluster -- does
not work here. `FastGM.buckets` holds live `FactorNN` objects (torch Modules
plus preprocessors), and, worse, the accumulator that actually produces the
answer is `root_bucket`, a **function-local** in `eliminate_variables`
(`nce/inference/graphical_model.py:311`). A checkpoint that saved `self.buckets`
would silently lose every scalar message routed to the root and return a wrong
log Z -- with no error.

So this module checkpoints the MESSAGE JOURNAL instead:

    for each cluster processed, in elimination order, record the message(s)
    `process_bucket` returned.

On resume, `process_bucket` is intercepted: for clusters already in the journal
it returns the recorded messages instead of recomputing them. Every other line
of `eliminate_variables` -- routing, `find_next_bucket`, the scalar-forwarding
branch, `del self.buckets[var]`, the final log-space product over
`root_bucket.factors` -- runs completely unmodified, in the same order, on the
same values. Nothing about the accumulation path is reimplemented, so nothing
about it can drift.

Two properties give bit-identity:

  1. **Messages are restored bit-exactly.** `torch.save`/`torch.load` round-trip
     float32 tensors exactly; there is no re-derivation and no arithmetic.
     Association order of the final product is preserved because the journal is
     replayed in the original order.
  2. **RNG state is restored at the boundary.** Per-bucket reseeding
     (`Net.__init__`, `SampleGenerator._get_seed`) means this is *usually*
     unnecessary -- but only usually: several loss functions call `torch.randn`
     with no local reseed and therefore consume the global stream, so a resumed
     run would otherwise enter cluster k with a different stream position.
     Restoring CPU/CUDA/numpy/python RNG at the boundary removes that entire
     class of divergence rather than arguing about which configs are exempt.

WHAT THIS DOES **NOT** MAKE IDENTICAL
-------------------------------------
Wall-clock fields (`phase_times`, this module's own timings) are not and cannot
be reproducible. They are excluded from the identity check by construction.

Bit-identity is also conditional on the run resuming on the SAME GPU MODEL with
the SAME thread count -- both recorded in the manifest, and both verified on
resume by `verify_environment`. The determinism suite measured a 2-float32-ULP
shift between 1 and 4 CPU threads on an otherwise identical run, so this is a
real failure mode, not a theoretical one.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

JOURNAL_VERSION = 1


def journal_fingerprint(problem_key: str, resolved_config: Dict[str, Any]) -> str:
    """Identity of the computation a journal describes.

    WHY LABEL-MATCHING IS NOT ENOUGH -- this was a real bug, caught by
    `test_resume_refuses_mismatched_journal`. Changing `iB` / `max_merge_bound`
    changes which clusters are MERGED but leaves the elimination order, and
    therefore the early cluster labels, identical. A resume guarded only by
    "does the label at step k match" happily replayed 10 messages from a
    different configuration and reported a confidently wrong log Z with
    status=done. That is precisely the silent corruption this system exists to
    prevent, so the journal is bound to the full resolved config instead.

    Wall-clock and path-like keys are excluded: they vary between runs of the
    same computation and would cause spurious refusals.
    """
    import hashlib
    ignore = {'log_file', 'gamma_trace_path', 'gamma_trace_run_id',
              'experiment_name', 'verbose', 'verbose_merge'}
    payload = {k: v for k, v in sorted(resolved_config.items())
               if k not in ignore}
    blob = json.dumps({'problem_key': problem_key, 'config': payload},
                      sort_keys=True, default=str)
    return hashlib.blake2b(blob.encode('utf-8'), digest_size=16).hexdigest()


def _rng_state() -> Dict[str, Any]:
    """Snapshot every RNG that any code path on the elimination path touches."""
    import random
    import numpy as np
    import torch
    st = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        st['torch_cuda'] = torch.cuda.get_rng_state_all()
    return st


def _restore_rng(st: Dict[str, Any]) -> None:
    import random
    import numpy as np
    import torch
    random.setstate(st['python'])
    np.random.set_state(st['numpy'])
    # .cpu() is REQUIRED, not defensive: the journal is loaded with
    # map_location=<run device>, which moves these ByteTensors onto the GPU,
    # and set_rng_state rejects anything that is not a CPU ByteTensor
    # ("RNG state must be a torch.ByteTensor"). Found by the CUDA resume test.
    torch.set_rng_state(st['torch_cpu'].cpu())
    if 'torch_cuda' in st and torch.cuda.is_available():
        cuda_states = [s.cpu() for s in st['torch_cuda']]
        # Only restore as many device states as this process actually has.
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)
        else:
            torch.cuda.set_rng_state(cuda_states[0])


# Accumulator lists on FastGM whose per-cluster appends are output artifacts.
# Journaling their deltas means a resumed run reproduces the same logs, not just
# the same log Z. Wall-clock accumulators are deliberately absent.
_DELTA_LISTS = (
    'per_bucket_training_log',
    'bucket_complexities',
    'local_errors',
    'error_tracking_data',
    'nn_errors',
    'message_stats',
)
_DELTA_SCALARS = ('num_trained', 'wmb_fw_partitions')


class CheckpointStore:
    """Directory-backed message journal.

    Layout:
        <dir>/journal.json       ordered index: step -> bucket label, digests
        <dir>/step_<n>.pt        torch.save of that step's messages + RNG state
    One file per step, written atomically, so a kill mid-write can lose at most
    the in-flight step -- and that step is simply recomputed on resume.
    """

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.index: Dict[str, Any] = {'version': JOURNAL_VERSION, 'steps': []}
        self._index_path = os.path.join(self.path, 'journal.json')
        if os.path.exists(self._index_path):
            with open(self._index_path) as fh:
                self.index = json.load(fh)

    # -- paths -------------------------------------------------------------
    def _step_path(self, step: int) -> str:
        return os.path.join(self.path, 'step_%05d.pt' % step)

    @property
    def n_steps(self) -> int:
        return len(self.index['steps'])

    def bind(self, fingerprint: str) -> None:
        """Attach (or verify) the computation identity of this journal.

        A journal with no fingerprint predates this check; binding it is
        allowed. A journal whose fingerprint DISAGREES is a different
        computation and must never be replayed -- raise rather than resume.
        """
        existing = self.index.get('fingerprint')
        if existing is None:
            self.index['fingerprint'] = fingerprint
            self._save_index()
        elif existing != fingerprint:
            raise RuntimeError(
                'Checkpoint journal at %s was built from a DIFFERENT '
                'computation (fingerprint %s, this run is %s). Replaying it '
                'would silently produce a wrong log Z -- changing iB or '
                'max_merge_bound leaves early cluster labels identical, so '
                'label matching alone would not catch this. Delete the '
                'directory to rerun from scratch.'
                % (self.path, existing[:16], fingerprint[:16]))

    def _save_index(self) -> None:
        os.makedirs(self.path, exist_ok=True)
        tmp = self._index_path + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(self.index, fh, indent=2, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self._index_path)

    # -- write -------------------------------------------------------------
    def append(self, step: int, label: Any, messages: List[Any],
               rng: Dict[str, Any], deltas: Dict[str, Any],
               was_list: bool = False) -> None:
        import torch
        os.makedirs(self.path, exist_ok=True)
        payload = {'step': step, 'label': label, 'messages': messages,
                   'rng': rng, 'deltas': deltas, 'was_list': was_list}
        tmp = self._step_path(step) + '.tmp'
        torch.save(payload, tmp)
        os.replace(tmp, self._step_path(step))
        self.index['steps'].append({
            'step': step, 'label': str(label), 'n_messages': len(messages),
            'saved_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        })
        self._save_index()

    # -- read --------------------------------------------------------------
    def load(self, step: int, device: str = None) -> Dict[str, Any]:
        import torch
        # weights_only=False: the payload contains FastFactor / FactorNN
        # objects, not just tensors. The file is written by this process's own
        # run directory, so it is not untrusted input.
        obj = torch.load(self._step_path(step), map_location=device,
                         weights_only=False)
        return obj

    def truncate_to_complete(self) -> int:
        """Drop index entries whose step file is missing (killed mid-write)."""
        keep = []
        for rec in self.index['steps']:
            if os.path.exists(self._step_path(rec['step'])):
                keep.append(rec)
            else:
                break
        if len(keep) != len(self.index['steps']):
            self.index['steps'] = keep
            self._save_index()
        return len(keep)


def verify_environment(manifest: Dict[str, Any]) -> List[str]:
    """Compare the current environment against a manifest. Returns warnings.

    Bit-identity is only claimed within a matching environment, so a resume
    across a mismatch must be loud. Returns a list of human-readable problems;
    empty list means the environment matches.
    """
    import torch
    problems = []
    libs = manifest.get('libraries', {})
    if libs.get('torch') and libs['torch'] != torch.__version__:
        problems.append('torch version changed: %s -> %s'
                        % (libs['torch'], torch.__version__))
    if libs.get('torch_num_threads') not in (None, torch.get_num_threads()):
        problems.append('torch thread count changed: %s -> %s (MEASURED to '
                        'move results by 2 float32 ULP)'
                        % (libs['torch_num_threads'], torch.get_num_threads()))
    hw = manifest.get('hardware', {})
    if hw.get('gpu_uuid_in_use') and torch.cuda.is_available():
        try:
            from nce.scheduler.provenance import hardware_provenance
            now = hardware_provenance()
            if now.get('gpu_uuid_in_use') != hw['gpu_uuid_in_use']:
                problems.append('GPU changed: %s -> %s'
                                % (hw['gpu_uuid_in_use'],
                                   now.get('gpu_uuid_in_use')))
        except Exception:
            pass
    git = manifest.get('git', {})
    if git.get('commit'):
        from nce.scheduler.provenance import git_provenance
        now_git = git_provenance()
        if now_git.get('commit') != git['commit']:
            problems.append('git commit changed: %s -> %s'
                            % (git['commit_short'], now_git.get('commit_short')))
        if git.get('dirty') or now_git.get('dirty'):
            problems.append('working tree is/was DIRTY -- commit alone does '
                            'not identify the code that ran')
    return problems


@contextmanager
def checkpointed_elimination(gm, store: CheckpointStore,
                             resume: bool = True,
                             fingerprint: Optional[str] = None):
    """Wrap an elimination so it journals messages and can resume.

    Usage:
        with checkpointed_elimination(gm, store):
            gm.eliminate_variables(all=True)

    EVERY cluster is journaled, and there is deliberately no "checkpoint every
    N clusters" knob. Such a knob looks like a cheap way to shrink the journal
    but is incompatible with this design: replay works by feeding each recorded
    message back in, so a gap in the journal is a cluster whose message cannot
    be replayed AND whose inputs cannot be reconstructed -- there is no full
    state snapshot to fall back on. A thinned journal would therefore either
    fail loudly on the gap or, worse, silently misalign step indices with
    journal entries. If journal size becomes a problem on wide models, the fix
    is to compress or prune stored tensors, not to skip clusters.

    Per-cluster is also the right granularity on cost: a cluster is one NN
    training, so the journal write is negligible beside it, and there is no
    partial-cluster state worth preserving.

    Only calls made during THIS block are journaled, and calls made while
    `gm.is_populating_backward_factors` is set are ignored -- that pass also
    goes through `process_bucket` but is not part of the elimination sequence.
    """
    from nce.inference.graphical_model import FastGM

    # Bind BEFORE any replay: a fingerprint mismatch must abort the run, not
    # be discovered partway through replaying foreign messages.
    if fingerprint is not None:
        store.bind(fingerprint)

    orig = FastGM.process_bucket
    state = {'step': 0, 'replayed': 0, 'computed': 0}
    completed = store.truncate_to_complete() if resume else 0
    device = getattr(gm, 'device', None)

    def process_bucket(self, bucket, exact=False):
        # Not part of the elimination sequence -> never journal, never replay.
        if getattr(self, 'is_populating_backward_factors', False):
            return orig(self, bucket, exact=exact)

        step = state['step']
        state['step'] += 1

        if resume and step < completed:
            rec = store.load(step, device=device)
            if str(rec['label']) != str(bucket.label):
                raise RuntimeError(
                    'Checkpoint replay diverged at step %d: journal has bucket '
                    '%r, run reached bucket %r. The elimination order changed, '
                    'so this journal does not describe this run. Refusing to '
                    'produce a silently-wrong result -- delete the checkpoint '
                    'directory and rerun from scratch.'
                    % (step, rec['label'], bucket.label))
            # Replay the accumulator appends so logs match an uninterrupted run.
            for name, items in rec['deltas'].get('lists', {}).items():
                cur = getattr(self, name, None)
                if isinstance(cur, list):
                    cur.extend(items)
            for name, val in rec['deltas'].get('scalars', {}).items():
                if hasattr(self, name):
                    setattr(self, name, val)
            # Restore the RNG stream position as of the END of this cluster, so
            # the first recomputed cluster starts exactly where it would have.
            _restore_rng(rec['rng'])
            state['replayed'] += 1
            msgs = rec['messages']
            return msgs if rec.get('was_list') else msgs[0]

        # -- compute for real, then journal ------------------------------
        before_lists = {n: len(getattr(self, n, []) or [])
                        for n in _DELTA_LISTS if isinstance(getattr(self, n, None), list)}
        result = orig(self, bucket, exact=exact)
        state['computed'] += 1

        was_list = isinstance(result, list)
        msgs = result if was_list else [result]
        deltas = {
            'lists': {n: list(getattr(self, n)[before_lists[n]:])
                      for n in before_lists},
            'scalars': {n: getattr(self, n) for n in _DELTA_SCALARS
                        if hasattr(self, n)},
        }
        # Every cluster, unconditionally -- see the docstring on why there is no
        # skip interval. RNG is snapshotted AFTER the cluster's work, so a
        # resume that replays through here re-enters the stream where this
        # cluster left it.
        store.append(step, bucket.label, msgs, _rng_state(), deltas,
                     was_list=was_list)
        return result

    FastGM.process_bucket = process_bucket
    try:
        yield state
    finally:
        FastGM.process_bucket = orig
