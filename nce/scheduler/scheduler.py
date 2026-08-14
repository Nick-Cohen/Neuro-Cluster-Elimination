"""The dispatch loop: put queued rerun work on genuinely idle GPUs.

    python -m nce.scheduler.scheduler --queue Q.json --out-dir DIR
    python -m nce.scheduler.scheduler --status --queue Q.json

WHAT IT GUARANTEES
------------------
* **Never GPU 2.** Excluded in `gpus.py` and re-asserted in `runner.py` before
  torch is imported. GPU 2 is retired for silent data corruption.
* **Strictly one job per GPU.** Timings are a deliverable of the rerun, so
  co-tenancy would corrupt the data, not merely slow it down. An earlier
  cross-process comparison in this project reported a 22.5x speedup that was
  pure contention against a true 1.34x.
* **Idle is measured, not assumed.** A GPU is dispatchable only if nvidia-smi
  reports no compute processes on it (see gpus.py). Another agent's job is
  therefore visible and respected.
* **Deterministic.** Job order is `JobSpec.sort_key`; GPU choice is lowest free
  index; output paths derive from `job_id`. Nothing depends on dict order,
  `set()` iteration, or `hash()`.

WHAT IT IS NOT
--------------
Not a distributed system. One box, four GPUs, one of which is retired. State is
a JSON file a human can read and edit.

POLLING NOTE
------------
This process sleeps between sweeps. That is fine: it is a long-running
foreground process whose sleeping does not wake an agent. What is deliberately
avoided is `pgrep -f <script>` for liveness -- that pattern matches the
detecting shell's own command line and deadlocks. Liveness here is
`os.waitpid`/`Popen.poll` on children this process actually started.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from nce.scheduler.gpus import (dispatchable_gpus, query_gpus, describe,
                                RETIRED_GPU_INDICES)
from nce.scheduler.jobs import JobQueue, JobSpec, PENDING, RUNNING, DONE, FAILED, BLOCKED
from nce.scheduler import models as modelval
from nce.scheduler.ballast import BallastPool

# Seconds between sweeps. Not a timeout and not a resource limit: it only sets
# how quickly a freed GPU is noticed. Jobs here run for minutes to hours, so
# anything in the tens of seconds is immaterial; 30 s keeps `--status` feeling
# live without polling nvidia-smi hard. Override with --poll-interval.
DEFAULT_POLL_INTERVAL_S = 30


class Dispatcher:
    def __init__(self, queue: JobQueue, out_dir: str, python: str = None,
                 base_config: str = None, threads: int = 1,
                 poll_interval: int = DEFAULT_POLL_INTERVAL_S,
                 no_checkpoint: bool = False,
                 only_gpus: Optional[List[int]] = None,
                 ballast: bool = True):
        self.queue = queue
        # Restrict dispatch to these physical indices. Narrows the idle set; it
        # can never widen it, so GPU 2 stays excluded even if named here.
        self.only_gpus = set(only_gpus) if only_gpus else None
        self.out_dir = os.path.abspath(out_dir)
        self.python = python or sys.executable
        self.base_config = base_config
        self.threads = threads
        self.poll_interval = poll_interval
        self.no_checkpoint = no_checkpoint
        # gpu_uuid -> {'proc': Popen, 'job_id': str, 'started': float}
        self.running: Dict[str, Dict[str, Any]] = {}
        # Thermal ballast keeps idle cards at working temperature between jobs.
        # Without it every job starts cold and the cold-start penalty scales
        # with job length (doc 59: -5.59% at 30 s, -0.49% at 1500 s), i.e. a
        # bias correlated with the very quantity the paper measures.
        self.ballast = BallastPool(
            python=self.python, enabled=ballast,
            log_dir=os.path.join(self.out_dir, '_ballast'),
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))))

    # -- helpers -----------------------------------------------------------
    def _job_dir(self, spec: JobSpec) -> str:
        return spec.output_dir(self.out_dir)

    def _write_job_file(self, spec: JobSpec) -> str:
        d = self._job_dir(spec)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, 'job.json')
        with open(p, 'w') as fh:
            json.dump(spec.to_dict(), fh, indent=2, default=str)
        return p

    def _busy_uuids(self) -> set:
        return set(self.running.keys())

    # -- dispatch ----------------------------------------------------------
    def _launch(self, spec: JobSpec, gpu) -> None:
        d = self._job_dir(spec)
        os.makedirs(d, exist_ok=True)
        job_file = self._write_job_file(spec)
        cmd = [self.python, '-m', 'nce.scheduler.runner',
               '--job-file', job_file, '--out-dir', d,
               '--gpu', str(gpu.index), '--threads', str(self.threads)]
        if self.base_config:
            cmd += ['--base-config', self.base_config]
        if self.no_checkpoint:
            cmd += ['--no-checkpoint']
        log_path = os.path.join(d, 'runner.log')
        log = open(log_path, 'ab')
        log.write(('\n=== launch %s on cuda:%d (%s) ===\n'
                   % (time.strftime('%Y-%m-%dT%H:%M:%S'), gpu.index, gpu.uuid)
                   ).encode())
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                cwd=os.path.dirname(os.path.dirname(
                                    os.path.dirname(os.path.abspath(__file__)))))
        self.running[gpu.uuid] = {'proc': proc, 'job_id': spec.job_id,
                                  'started': time.time(), 'log': log,
                                  'gpu_index': gpu.index, 'name': spec.name}
        self.queue.mark_running(spec.job_id, gpu.uuid, proc.pid)
        print('[dispatch] cuda:%d <- %s (pid %d)'
              % (gpu.index, spec.name, proc.pid), flush=True)

    def _reap(self) -> None:
        """Collect finished children. Liveness via Popen.poll, never pgrep."""
        for uuid in list(self.running):
            rec = self.running[uuid]
            rc = rec['proc'].poll()
            if rc is None:
                continue
            rec['log'].close()
            elapsed = time.time() - rec['started']
            if rc == 0:
                status, note = DONE, ''
            elif rc == 3:
                status, note = BLOCKED, 'model validation failed'
            else:
                status, note = FAILED, 'runner exit code %d' % rc
            self.queue.set_status(rec['job_id'], status, note=note,
                                  elapsed_s=round(elapsed, 1))
            print('[%s] %s after %.1fs on cuda:%d %s'
                  % (status, rec['name'], elapsed, rec['gpu_index'], note),
                  flush=True)
            del self.running[uuid]

    def _validate_pending(self) -> None:
        """Block jobs whose model is missing/corrupt BEFORE they hold a GPU."""
        for spec in self.queue.pending():
            mv = modelval.validate(spec.problem_key)
            if not mv['ok']:
                self.queue.set_status(spec.job_id, BLOCKED,
                                      note='; '.join(mv['problems'])[:500])
                print('[blocked] %s: %s' % (spec.name, mv['problems'][0]),
                      flush=True)

    def sweep(self) -> int:
        """One pass: reap, then fill every free GPU. Returns jobs launched.

        Ballast interacts with this in two places and both matter:
          - our own ballast pids are excluded from the busy signal, or every
            ballasted card would look occupied and nothing would ever dispatch;
          - ballast is torn down on a card BEFORE a job launches there, and
            `BallastPool.stop` blocks until the CUDA context is actually gone,
            so a job never shares a card with ballast (requirement: ballast must
            not perturb a running job's timings).
        """
        self._reap()
        busy = self._busy_uuids()
        free = [g for g in dispatchable_gpus(ignore_pids=self.ballast.pids())
                if g.uuid not in busy]
        if self.only_gpus is not None:
            free = [g for g in free if g.index in self.only_gpus]
        pending = self.queue.pending()
        launched = 0
        for gpu, spec in zip(free, pending):
            # Yield the card before the job touches it. If ballast will not let
            # go, skip this card entirely rather than launch onto a busy GPU.
            if not self.ballast.stop(gpu.index):
                continue
            self._launch(spec, gpu)
            launched += 1
        # Ballast every dispatchable card that did NOT just receive a job, so it
        # is at working temperature when it does. Recomputed after dispatch so a
        # card we just launched onto is never in the set.
        self._ballast_idle_cards()
        return launched

    def _ballast_idle_cards(self) -> None:
        """Run ballast on dispatchable cards that carry no job of ours."""
        if not self.ballast.enabled:
            return
        busy = self._busy_uuids()
        free = [g for g in dispatchable_gpus(ignore_pids=self.ballast.pids())
                if g.uuid not in busy]
        if self.only_gpus is not None:
            free = [g for g in free if g.index in self.only_gpus]
        self.ballast.ensure(g.index for g in free)

    def run(self, once: bool = False, max_jobs: int = None) -> None:
        self._validate_pending()
        started = 0
        try:
            try:
                while True:
                    if max_jobs is not None and started >= max_jobs:
                        break
                    started += self.sweep()
                    if once:
                        break
                    if not self.running and not self.queue.pending():
                        print('[idle] queue drained: %s' % self.queue.summary(),
                              flush=True)
                        break
                    time.sleep(self.poll_interval)
            except KeyboardInterrupt:
                print('\n[interrupt] leaving %d job(s) running; their checkpoints '
                      'let them resume.' % len(self.running), flush=True)
                return
            # Drain: wait for stragglers so the final summary is truthful.
            while self.running:
                time.sleep(self.poll_interval)
                self._reap()
                # Keep the cards freed by finished jobs warm for whatever runs
                # next; drained cards otherwise cool 83->61 C in 60 s.
                self._ballast_idle_cards()
            print('[done] %s' % self.queue.summary(), flush=True)
        finally:
            # Never leave ballast behind. An orphaned ballast worker would hold
            # a card at full power indefinitely and, because it is a compute
            # process, would make that card look busy to the next scheduler.
            self.ballast.stop_all()


def _cmd_status(args) -> int:
    print('GPUs:')
    print(describe())
    print('  (cuda:%s excluded: RETIRED for silent data corruption)'
          % ','.join(str(i) for i in sorted(RETIRED_GPU_INDICES)))
    if args.queue and os.path.exists(args.queue):
        q = JobQueue(args.queue)
        print('\nQueue: %s' % q.summary())
        for status in (RUNNING, PENDING, BLOCKED, FAILED):
            items = q.by_status(status)
            if items:
                print('  %s:' % status)
                for s in items[:20]:
                    print('    %s' % s.name)
                if len(items) > 20:
                    print('    ... and %d more' % (len(items) - 20))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--queue', required=True, help='queue JSON file')
    p.add_argument('--out-dir', help='root for per-job output directories')
    p.add_argument('--status', action='store_true', help='print state and exit')
    p.add_argument('--once', action='store_true',
                   help='one dispatch sweep, then exit')
    p.add_argument('--max-jobs', type=int, default=None)
    p.add_argument('--threads', type=int, default=1,
                   help='torch threads per job (pinned: thread count changes '
                        'results by ~2 float32 ULP)')
    p.add_argument('--poll-interval', type=int, default=DEFAULT_POLL_INTERVAL_S)
    p.add_argument('--base-config', default=None)
    p.add_argument('--python', default=None)
    p.add_argument('--no-checkpoint', action='store_true')
    p.add_argument('--no-ballast', action='store_true',
                   help='disable thermal ballast on idle cards. Ballast is ON '
                        'by default: without it every job starts cold and the '
                        'cold-start penalty scales with job length (doc 59), '
                        'biasing exactly the timings the paper reports.')
    p.add_argument('--only-gpus', nargs='+', type=int, default=None,
                   help='restrict dispatch to these physical GPU indices '
                        '(e.g. reserve 0 and 1 for another agent). Can only '
                        'narrow the idle set; GPU 2 stays excluded regardless.')
    args = p.parse_args(argv)

    if args.status:
        return _cmd_status(args)
    if not args.out_dir:
        p.error('--out-dir is required unless --status')

    q = JobQueue(args.queue)
    d = Dispatcher(q, args.out_dir, python=args.python,
                   base_config=args.base_config, threads=args.threads,
                   poll_interval=args.poll_interval,
                   no_checkpoint=args.no_checkpoint,
                   only_gpus=args.only_gpus,
                   ballast=not args.no_ballast)
    print('Scheduler starting. %s' % q.summary())
    print(describe())
    d.run(once=args.once, max_jobs=args.max_jobs)
    return 0


if __name__ == '__main__':
    sys.exit(main())
