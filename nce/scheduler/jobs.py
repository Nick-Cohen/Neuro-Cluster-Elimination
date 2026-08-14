"""Job specification and queue for paper-rerun work.

DETERMINISM CONTRACT
--------------------
This project lost days to a nondeterminism bug caused by `set()` over objects
with no `__hash__`, which made iteration order follow memory addresses. The
scheduler must not reintroduce that class of bug, so:

  * `JobSpec.job_id` is a BLAKE2b digest of the canonical JSON of the spec.
    Same spec => same id, in every process, on every run, forever. It does not
    use `hash()`, which is randomised per-process by PYTHONHASHSEED.
  * Job ordering is `sorted()` on an explicit total-order key of scalar fields.
    Never on a set, never on object identity, never on dict iteration order.
  * Output paths are a pure function of the job_id, so a resumed or re-queued
    job lands in exactly the same directory.
  * Seed assignment is an explicit field of the spec, never drawn from an RNG.

If you add a field to JobSpec, add it to `_canonical()` too, or two different
jobs will collide on one id and silently overwrite each other's results.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict, replace
from typing import Any, Dict, List, Optional, Tuple

SPEC_VERSION = 1


@dataclass(frozen=True)
class JobSpec:
    """One rerun cell: problem x merge strategy x merge bound x seed.

    `extra_config` carries any additional resolved-config overrides (iB, ecl,
    num_epochs, ...). It is part of the identity digest, so two jobs differing
    only in iB are correctly distinct jobs.
    """
    problem_key: str
    merge_strategy: str          # e.g. 'reduce_nn', 'nomerge', 'merge_degree'
    merge_bound: Optional[int]   # a.k.a. max_merge_bound; None for 'nomerge'
    seed: int
    extra_config: Dict[str, Any] = field(default_factory=dict)
    tag: str = ''                # free-form grouping label, part of identity

    # ---- identity ---------------------------------------------------------
    def _canonical(self) -> str:
        """Canonical JSON. Every identity-bearing field must appear here."""
        return json.dumps({
            'spec_version': SPEC_VERSION,
            'problem_key': self.problem_key,
            'merge_strategy': self.merge_strategy,
            'merge_bound': self.merge_bound,
            'seed': self.seed,
            'extra_config': self.extra_config,
            'tag': self.tag,
        }, sort_keys=True, separators=(',', ':'), default=str)

    @property
    def job_id(self) -> str:
        """Stable across processes: BLAKE2b, not the PYTHONHASHSEED-randomised
        builtin `hash()`."""
        return hashlib.blake2b(self._canonical().encode('utf-8'),
                               digest_size=8).hexdigest()

    @property
    def name(self) -> str:
        """Readable directory-safe name; still unique via the id suffix."""
        prob = self.problem_key.replace('/', '_')
        bound = 'none' if self.merge_bound is None else str(self.merge_bound)
        return '%s__%s%s__s%d__%s' % (prob, self.merge_strategy, bound,
                                      self.seed, self.job_id)

    @property
    def sort_key(self) -> Tuple:
        """Explicit total order over scalars -- deterministic queue ordering."""
        return (self.tag, self.problem_key, self.merge_strategy,
                -1 if self.merge_bound is None else self.merge_bound,
                self.seed, self.job_id)

    # ---- config ----------------------------------------------------------
    def to_config(self, base: Dict[str, Any] = None) -> Dict[str, Any]:
        """Build the raw (pre-`prepare_config`) config dict for this job.

        Merge strategy is expressed through the flags the codebase actually
        consumes. `extra_config` is applied LAST so a caller can always override.
        """
        cfg = dict(base or {})
        cfg['seed'] = self.seed
        if self.merge_strategy == 'reduce_nn':
            cfg['use_reduce_nn_merge'] = True
            cfg['max_merge_bound'] = self.merge_bound
        elif self.merge_strategy == 'nomerge':
            cfg['use_reduce_nn_merge'] = False
        elif self.merge_strategy == 'merge_degree':
            cfg['use_reduce_nn_merge'] = False
            cfg['max_merge_bound'] = self.merge_bound
        else:
            raise ValueError(
                'Unknown merge_strategy %r. Known: reduce_nn, nomerge, '
                'merge_degree. Add it here rather than smuggling it through '
                'extra_config, so it stays part of the job identity.'
                % (self.merge_strategy,))
        cfg.update(self.extra_config)
        return cfg

    def output_dir(self, root: str) -> str:
        """Pure function of the spec -- resume lands in the same place."""
        return os.path.join(root, self.name)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['job_id'] = self.job_id
        d['name'] = self.name
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'JobSpec':
        return JobSpec(
            problem_key=d['problem_key'],
            merge_strategy=d['merge_strategy'],
            merge_bound=d.get('merge_bound'),
            seed=int(d['seed']),
            extra_config=dict(d.get('extra_config') or {}),
            tag=d.get('tag', ''),
        )


def build_grid(problem_keys: List[str],
               merge_strategies: List[str],
               merge_bounds: List[Optional[int]],
               seeds: List[int],
               base_config: Dict[str, Any] = None,
               tag: str = '') -> List[JobSpec]:
    """Cartesian product -> deterministically ordered job list.

    'nomerge' takes no bound, so it is emitted once rather than once per bound;
    otherwise the same job would appear N times under N different ids.
    De-duplication is by job_id through a dict (insertion-ordered), never a set.
    """
    out: Dict[str, JobSpec] = {}
    for prob in problem_keys:
        for strat in merge_strategies:
            bounds = [None] if strat == 'nomerge' else merge_bounds
            for bound in bounds:
                for seed in seeds:
                    js = JobSpec(problem_key=prob, merge_strategy=strat,
                                 merge_bound=bound, seed=seed,
                                 extra_config=dict(base_config or {}), tag=tag)
                    out[js.job_id] = js
    return sorted(out.values(), key=lambda j: j.sort_key)


# ---------------------------------------------------------------------------
# Queue state
# ---------------------------------------------------------------------------
PENDING, RUNNING, DONE, FAILED, BLOCKED = (
    'pending', 'running', 'done', 'failed', 'blocked')
TERMINAL = (DONE, FAILED)


class JobQueue:
    """A directory-backed queue. One JSON file holds the whole queue state.

    Deliberately not a database: this runs on one box, and a file that a human
    can read and hand-edit while debugging is worth more here than concurrency
    features nobody needs. All writes are atomic (write-tmp + os.replace).
    """

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self._state: Dict[str, Any] = {'spec_version': SPEC_VERSION, 'jobs': {}}
        if os.path.exists(self.path):
            self.load()

    def load(self) -> None:
        with open(self.path) as fh:
            self._state = json.load(fh)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(self._state, fh, indent=2, sort_keys=True, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)

    def add(self, specs: List[JobSpec]) -> int:
        """Add jobs; existing job_ids are left untouched (idempotent re-queue)."""
        added = 0
        for js in specs:
            if js.job_id not in self._state['jobs']:
                self._state['jobs'][js.job_id] = {
                    'spec': js.to_dict(), 'status': PENDING, 'attempts': 0,
                    'gpu_uuid': None, 'note': '',
                }
                added += 1
        self.save()
        return added

    def _entries(self) -> List[Tuple[JobSpec, Dict[str, Any]]]:
        items = []
        for jid, rec in self._state['jobs'].items():
            items.append((JobSpec.from_dict(rec['spec']), rec))
        return sorted(items, key=lambda t: t[0].sort_key)

    def pending(self) -> List[JobSpec]:
        return [js for js, rec in self._entries() if rec['status'] == PENDING]

    def by_status(self, status: str) -> List[JobSpec]:
        return [js for js, rec in self._entries() if rec['status'] == status]

    def get(self, job_id: str) -> Dict[str, Any]:
        return self._state['jobs'][job_id]

    def set_status(self, job_id: str, status: str, **fields) -> None:
        rec = self._state['jobs'][job_id]
        rec['status'] = status
        rec.update(fields)
        self.save()

    def mark_running(self, job_id: str, gpu_uuid: str, pid: int) -> None:
        rec = self._state['jobs'][job_id]
        rec['attempts'] = rec.get('attempts', 0) + 1
        self.set_status(job_id, RUNNING, gpu_uuid=gpu_uuid, pid=pid)

    def counts(self) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for rec in self._state['jobs'].values():
            c[rec['status']] = c.get(rec['status'], 0) + 1
        return c

    def summary(self) -> str:
        c = self.counts()
        total = sum(c.values())
        parts = ['%s=%d' % (k, c[k]) for k in sorted(c)]
        return '%d jobs (%s)' % (total, ', '.join(parts) if parts else 'empty')
