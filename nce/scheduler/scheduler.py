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
* **A dead scheduler cannot silently lose jobs.** On startup and on every sweep
  the reaper (`nce/scheduler/reaper.py`) checks each `running` job's recorded
  process identity. Dead ones go back to `pending` and resume from their
  checkpoint journal; live orphans from a previous scheduler are ADOPTED so
  nothing is dispatched onto their card. Reaping is on process liveness, never
  elapsed time -- some arms run 5+ hours.

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
from nce.scheduler.jobs import (JobQueue, JobSpec, PENDING, RUNNING, DONE,
                                FAILED, BLOCKED, NEEDS_ATTENTION)
from nce.scheduler import models as modelval
from nce.scheduler.ballast import BallastPool
from nce.scheduler.reaper import (Reaper, MAX_REAP_RETRIES, orphan_ballast,
                                  terminate_orphan_ballast,
                                  process_identity as reap_identity)

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
                 ballast: bool = True,
                 max_reap_retries: int = MAX_REAP_RETRIES):
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
        # gpu_uuid -> record, for jobs left RUNNING by a PREVIOUS scheduler whose
        # processes are still alive. Popen children are reparented, not killed,
        # when their parent dies, so after a restart these are genuinely still
        # computing. They are tracked here purely so their card counts as BUSY:
        # dispatching onto an occupied card would co-tenant two timed jobs, and
        # co-tenancy corrupts the timings this rerun exists to produce.
        self.adopted: Dict[str, Dict[str, Any]] = {}
        self.reaper = Reaper(out_dir=self.out_dir,
                             max_retries=max_reap_retries)
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
        """Cards this scheduler must not dispatch to.

        Includes ADOPTED cards -- a live orphan from a dead scheduler owns its
        card just as much as one of our own children does. Relying on
        `dispatchable_gpus()` alone would not be enough: a runner spends its
        first 8-9 s CPU-bound (imports, model load) with no CUDA context yet, so
        the driver reports its card as free and a fresh job would land on top of
        it.
        """
        return set(self.running.keys()) | set(self.adopted.keys())

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
        # BALLAST HANDOFF. The card is handed over WARM, with ballast still on
        # it, and the runner drops ballast itself immediately before its first
        # GPU work (runner.yield_ballast). Stopping ballast here instead would
        # leave the card idle for the whole of the runner's CPU-bound setup --
        # measured at 8-9 s, over which a card falls from ~83 C toward ~49 C --
        # which is precisely the cold start this exists to remove.
        handoff = self.ballast.handoff_info(gpu.index)
        if handoff is not None:
            _pid, pause_file, paused_marker = handoff
            cmd += ['--ballast-pause-file', pause_file,
                    '--ballast-paused-marker', paused_marker]
        cmd += ['--launched-at', repr(time.time())]
        log_path = os.path.join(d, 'runner.log')
        log = open(log_path, 'ab')
        log.write(('\n=== launch %s on cuda:%d (%s) ===\n'
                   % (time.strftime('%Y-%m-%dT%H:%M:%S'), gpu.index, gpu.uuid)
                   ).encode())
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                cwd=os.path.dirname(os.path.dirname(
                                    os.path.dirname(os.path.abspath(__file__)))))
        # Capture the process IDENTITY immediately, while the child is certainly
        # still the process we just started. A bare pid is not an identity: pids
        # are reused, and a reaper that trusted the number would mistake an
        # unrelated process for this job and never fire. See reaper.py.
        identity = reap_identity(proc.pid)
        started = time.time()
        self.running[gpu.uuid] = {'proc': proc, 'job_id': spec.job_id,
                                  'started': started, 'log': log,
                                  'gpu_index': gpu.index, 'name': spec.name}
        self.queue.mark_running(spec.job_id, gpu.uuid, proc.pid,
                                proc_identity=identity, started_at=started)
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
            # The job paused ballast before it touched the GPU; now that the
            # card is free again, let ballast feed it. Done here rather than at
            # end of sweep so the card starts re-warming the instant the job
            # ends, not one poll interval later.
            self.ballast.resume(rec['gpu_index'])

    def _reap_stale(self, startup: bool = False) -> None:
        """Repair jobs stuck at `running` because a previous scheduler died.

        Runs on startup AND on every sweep. On startup it is the thing that makes
        an unattended restart complete: without it those jobs stay `running`
        forever, are never re-dispatched, and the sweep finishes with a silently
        incomplete result set. On every sweep it is what notices an ADOPTED
        orphan finishing, so its card is released back to dispatch.

        Reaping is on PROCESS LIVENESS, never elapsed time -- some arms in this
        study run 5+ hours and a timeout-based reaper would murder them.
        """
        res = self.reaper.sweep(self.queue,
                                owned_job_ids=[r['job_id'] for r in
                                               self.running.values()])
        # Live orphans: hold their card so nothing is dispatched on top of them.
        uuid_to_index = {}
        if res['adopted']:
            uuid_to_index = {g.uuid: g.index for g in query_gpus()}
        adopted_now = {}
        for row in res['adopted']:
            rec, spec = row['rec'], row['spec']
            uuid = rec.get('gpu_uuid')
            if not uuid:
                # No card recorded: cannot protect a card we cannot name. Say so
                # rather than silently dispatching over it.
                print('[reaper] WARNING: %s is alive (pid %s) but records no '
                      'gpu_uuid; cannot reserve its card.'
                      % (spec.name, rec.get('pid')), flush=True)
                continue
            adopted_now[uuid] = {'job_id': spec.job_id, 'name': spec.name,
                                 'pid': rec.get('pid'),
                                 'gpu_index': uuid_to_index.get(uuid)}
            if uuid not in self.adopted:
                print('[reaper] ADOPTED %s (pid %s) still running on %s -- its '
                      'card is reserved; it will be marked from its own '
                      'result.json when it exits.'
                      % (spec.name, rec.get('pid'), uuid), flush=True)
        self.adopted = adopted_now
        if startup:
            n = sum(len(res[k]) for k in ('reaped', 'recovered', 'parked'))
            print('[reaper] startup scan: %d re-queued, %d recovered from disk, '
                  '%d parked for attention, %d still alive, %d unknown'
                  % (len(res['reaped']), len(res['recovered']),
                     len(res['parked']), len(res['adopted']),
                     len(res['unknown'])), flush=True)
            if n:
                print('[reaper] queue after repair: %s' % self.queue.summary(),
                      flush=True)
            # An orphaned ballast worker holds a CUDA context, so its card reads
            # BUSY and takes no jobs -- the same stall by another route. One
            # proven ours (its control files live under OUR out-dir) is torn
            # down; anything else is only reported, because it may be another
            # agent's and killing it would cool their card and bias their
            # timings.
            for orph in orphan_ballast(own_pids=self.ballast.pids(),
                                       log_dir=self.ballast.log_dir,
                                       gpu_indices=self.only_gpus):
                if orph['ours']:
                    print('[reaper] cuda:%s carries an ORPHANED ballast worker '
                          '(pid %s) from a previous run of THIS scheduler; '
                          'tearing it down so the card can take jobs.'
                          % (orph['gpu_index'], orph['pid']), flush=True)
                    terminate_orphan_ballast(orph)
                else:
                    print('[reaper] WARNING: cuda:%s carries a ballast worker '
                          '(pid %s) that is not ours. That card will look BUSY '
                          'and take no jobs. Nothing has been killed -- it may '
                          'belong to another agent. To clear it: kill %s'
                          % (orph['gpu_index'], orph['pid'], orph['pid']),
                          flush=True)

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
        self._reap_stale()
        busy = self._busy_uuids()
        free = [g for g in dispatchable_gpus(ignore_pids=self.ballast.pids())
                if g.uuid not in busy]
        if self.only_gpus is not None:
            free = [g for g in free if g.index in self.only_gpus]
        pending = self.queue.pending()
        # Ballast every free card BEFORE dispatching. Ordering matters: a worker
        # must already exist on the card for `_launch` to hand it to the runner,
        # and the runner is what keeps the card warm through its own setup. With
        # a saturated queue there is no idle window in which ballast could
        # otherwise ever start, so ballasting only after dispatch (the previous
        # ordering) meant ballast never ran at all and every job's setup window
        # stayed uncovered.
        self.ballast.ensure([g.index for g in free],
                            keep_indices=[rec['gpu_index']
                                          for rec in self.running.values()])
        launched = 0
        for gpu, spec in zip(free, pending):
            # NOTE: ballast is deliberately NOT stopped here. The card is handed
            # over warm and the runner pauses ballast itself, in the last
            # CPU-only instant before it touches the GPU. See _launch.
            self._launch(spec, gpu)
            launched += 1
        # Cards that did not receive a job keep (or gain) ballast.
        self._ballast_idle_cards()
        return launched

    def _ballast_idle_cards(self) -> None:
        """Keep a ballast worker on every card we own.

        "Own" means free-and-dispatchable OR currently running one of our jobs.
        The second half is essential: a card running a job has a ballast worker
        that the runner PAUSED, and that worker must stay alive so it can resume
        the instant the job ends and so the NEXT job has something to hand off
        to. Dropping it here (the obvious reading of "ballast idle cards") would
        kill the worker the running job just paused, and no job would ever have
        its setup window covered.

        Ballast is only torn down on cards that have left our control entirely.
        """
        if not self.ballast.enabled:
            return
        busy = self._busy_uuids()
        free = [g for g in dispatchable_gpus(ignore_pids=self.ballast.pids())
                if g.uuid not in busy]
        if self.only_gpus is not None:
            free = [g for g in free if g.index in self.only_gpus]
        self.ballast.ensure([g.index for g in free],
                            keep_indices=[rec['gpu_index']
                                          for rec in self.running.values()])

    def warm_cards(self, seconds: float) -> None:
        """Bring every dispatchable card to equilibrium BEFORE the first job.

        Ballast covers the gaps between jobs and each job's own setup window, so
        with it enabled the only job that can still start cold is the FIRST one
        on each card -- `sweep()` would otherwise dispatch in the same pass that
        starts ballast. This closes that last case.

        `seconds` is the caller's, not invented here. Doc 59 MEASURED
        time-to-equilibrium as 330 s on gpu0 and 180 s on gpu3 and recommends a
        360 s warm-up; this is opt-in because it delays the first job by that
        much and that is a scheduling decision, not a default I should impose.
        """
        if not self.ballast.enabled or seconds <= 0:
            return
        # Exclude adopted cards. A live orphan spends its first seconds
        # CPU-bound with no CUDA context, so the driver would call its card
        # free -- and warming it means running GEMM against a job that is being
        # timed. Same reasoning as _busy_uuids.
        busy = self._busy_uuids()
        free = [g for g in dispatchable_gpus(ignore_pids=self.ballast.pids())
                if g.uuid not in busy]
        if self.only_gpus is not None:
            free = [g for g in free if g.index in self.only_gpus]
        if not free:
            return
        for g in free:
            self.ballast.start(g.index)
        print('[warmup] holding %d card(s) at load for %.0fs before the first '
              'dispatch: %s' % (len(free), seconds,
                                ','.join('cuda:%d' % g.index for g in free)),
              flush=True)
        time.sleep(seconds)
        print('[warmup] done', flush=True)

    def run(self, once: bool = False, max_jobs: int = None,
            warmup_s: float = 0.0) -> None:
        # BEFORE validation and before the warm-up, so a restart after a crash
        # has its full pending set (and its true free-card set) from the first
        # sweep instead of one poll interval later.
        self._reap_stale(startup=True)
        self._validate_pending()
        self.warm_cards(warmup_s)
        started = 0
        try:
            try:
                while True:
                    if max_jobs is not None and started >= max_jobs:
                        break
                    started += self.sweep()
                    if once:
                        break
                    # `adopted` counts: exiting while an adopted orphan is still
                    # computing would leave it stuck at `running` again and
                    # report a drained queue that is not drained.
                    if (not self.running and not self.adopted
                            and not self.queue.pending()):
                        print('[idle] queue drained: %s' % self.queue.summary(),
                              flush=True)
                        break
                    time.sleep(self.poll_interval)
            except KeyboardInterrupt:
                print('\n[interrupt] leaving %d job(s) running; their checkpoints '
                      'let them resume. A later scheduler will re-queue any that '
                      'die (nce.scheduler.reaper).' % len(self.running),
                      flush=True)
                return
            # Drain: wait for stragglers so the final summary is truthful.
            # `adopted` counts only in a continuous run. A `--once` invocation
            # must not block on another scheduler's orphan, which can have hours
            # left to run and which we did not start.
            while self.running or (self.adopted and not once):
                time.sleep(self.poll_interval)
                self._reap()
                self._reap_stale()
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
        for status in (RUNNING, PENDING, BLOCKED, FAILED, NEEDS_ATTENTION):
            items = q.by_status(status)
            if items:
                print('  %s:' % status)
                for s in items[:20]:
                    print('    %s' % s.name)
                if len(items) > 20:
                    print('    ... and %d more' % (len(items) - 20))
        # Say, for each `running` job, whether its process actually still
        # exists. A `running` entry whose process is gone is the stall this
        # scheduler's reaper repairs on its next start; showing it here means
        # nobody has to guess.
        r = Reaper(out_dir=args.out_dir)
        rows = r.scan(q)
        if rows:
            print('\nLiveness of `running` jobs (dry run -- nothing changed):')
            for row in rows:
                print('  %-7s %s\n          %s'
                      % (row['state'].upper(), row['spec'].name, row['reason']))
            if any(row['state'] != 'alive' for row in rows):
                print('  -> repair with: python -m nce.scheduler.reaper '
                      '--queue %s --out-dir <runs> --apply' % args.queue)
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
    p.add_argument('--warmup-s', type=float, default=0.0,
                   help='hold every dispatchable card at load for this long '
                        'before the FIRST dispatch. Ballast covers inter-job '
                        'gaps and each job\'s setup, so only the first job per '
                        'card can still start cold. Doc 59 measured '
                        'time-to-equilibrium at 330 s (gpu0) / 180 s (gpu3) and '
                        'recommends 360. Off by default: it delays the first '
                        'job, which is a scheduling decision.')
    p.add_argument('--no-ballast', action='store_true',
                   help='disable thermal ballast on idle cards. Ballast is ON '
                        'by default: without it every job starts cold and the '
                        'cold-start penalty scales with job length (doc 59), '
                        'biasing exactly the timings the paper reports.')
    p.add_argument('--max-reap-retries', type=int, default=MAX_REAP_RETRIES,
                   help='how many times a job whose process died may be put '
                        'back on the queue before it is parked as '
                        '"needs_attention" for a human. Default 1 = retry once, '
                        'then require attention -- a job that keeps dying must '
                        'not be allowed to loop and consume the sweep.')
    p.add_argument('--only-gpus', nargs='+', type=int, default=None,
                   help='restrict dispatch to these physical GPU indices '
                        '(e.g. reserve 0 and 1 for another agent). Can only '
                        'narrow the idle set; GPU 2 stays excluded regardless.')
    args = p.parse_args(argv)

    if args.status:
        return _cmd_status(args)
    if not args.out_dir:
        p.error('--out-dir is required unless --status')

    # Fail before a single job is dispatched, naming the variable, rather than
    # letting an unattended sweep BLOCK every job on a missing model cache.
    print('model cache: %s' % modelval.assert_cache_configured())

    q = JobQueue(args.queue)
    d = Dispatcher(q, args.out_dir, python=args.python,
                   base_config=args.base_config, threads=args.threads,
                   poll_interval=args.poll_interval,
                   no_checkpoint=args.no_checkpoint,
                   only_gpus=args.only_gpus,
                   ballast=not args.no_ballast,
                   max_reap_retries=args.max_reap_retries)
    print('Scheduler starting. %s' % q.summary())
    print(describe())
    d.run(once=args.once, max_jobs=args.max_jobs, warmup_s=args.warmup_s)
    return 0


if __name__ == '__main__':
    sys.exit(main())
