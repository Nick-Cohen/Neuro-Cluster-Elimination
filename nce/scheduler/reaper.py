"""Stale-`running` reaper: return jobs whose process is gone to the queue.

THE FAILURE THIS EXISTS TO REMOVE
---------------------------------
`JobQueue.mark_running` writes `status='running'` to a file. Nothing ever
un-writes it. If the scheduler process dies -- SIGKILL, OOM killer, a lost
terminal, a reboot -- every job it had in flight stays `running` in the queue
forever and is never re-dispatched, because `Dispatcher.sweep` only ever looks
at `pending()`. On a 13.5-day unattended sweep that is how you lose days: the
scheduler is restarted, it happily drains the remaining 1300 jobs, and the two
that were in flight at 3 a.m. are silently missing from the result set.

Doc 61 section 7 documents the manual fix (hand-edit `running` -> `pending`).
This module is that fix, done automatically, correctly, and with the two
mistakes the manual recipe cannot avoid:

  * the manual recipe also resets jobs that had **already finished** and merely
    had not been marked yet -- it re-runs completed work;
  * the manual recipe cannot tell a dead job from a live one, so it must only be
    run when the scheduler is known dead. This module can run on every sweep.

LIVENESS IS PROCESS EXISTENCE, NEVER ELAPSED TIME
-------------------------------------------------
Phase-1 jobs run for minutes to hours and the longest arms in this study run
5+ hours. Any timeout-based reaper would murder legitimate work, and the
resulting job would be re-dispatched onto a card its predecessor still owns.
So the criterion here is strictly "does the process still exist", which is a
fact rather than a threshold -- the same principle `gpus.py` applies to card
idleness. The only time-based thing in this module is a WARNING that never
touches the queue (see `STALE_WARN_HOURS`).

HOW A PID IS CHECKED, AND WHY NOT THE OBVIOUS WAYS
--------------------------------------------------
1. **Never `pgrep -f <pattern>`.** The pattern matches the *detecting* shell's
   own command line, so the detector sees itself and concludes the job is
   alive. That deadlocked three jobs for another agent on this project. There
   is no pattern matching anywhere in this file: every check starts from a PID
   this scheduler recorded at dispatch.

2. **`os.kill(pid, 0)` alone is not enough**, for two independent reasons:

   * **PID reuse.** This box churns PIDs in the 3.9M range and this sweep runs
     for 13.5 days, so a dead runner's PID being reissued to an unrelated
     process is a routine event, not a corner case. A bare PID check would
     then report a dead job as alive and the reaper would never fire -- the
     exact stall it exists to prevent.
   * **Zombies.** `os.kill` succeeds for a process that has exited but not been
     reaped by its parent. `runner.py` already documents this trap for ballast.

   So liveness is bound to a **process identity**, not a PID:

       (boot_id, pid, starttime_ticks)

   `starttime_ticks` is field 22 of `/proc/<pid>/stat` -- the process's start
   time in clock ticks since boot. It is fixed for the life of the process and
   a recycled PID is essentially certain to differ (the clock tick is ~10 ms
   and the reissue happens millions of ticks later). `boot_id` is included
   because `starttime` is measured *from boot*: after a reboot a fresh process
   could in principle hold both the same PID and the same tick count, and a
   reboot is also, by itself, proof that every recorded job is dead.

3. **A corroborating cmdline check.** `/proc/<pid>/cmdline` for that ONE pid is
   read and must contain the job's `job_id` (it appears in `--job-file
   .../<name>/job.json`, and `JobSpec.name` ends in the id). This is not a
   pattern scan of the process table -- it is a targeted read of a specific
   PID's own cmdline, and the scheduler's own PID is excluded explicitly. It is
   what makes the check work for queue records written *before* this module
   existed, which have a `pid` but no recorded identity: those are exactly the
   records in the live phase-1 queue.

ADOPTION: A DEAD SCHEDULER DOES NOT KILL ITS JOBS
-------------------------------------------------
`subprocess.Popen` children are NOT killed when the parent dies; they are
reparented and keep running. So after a restart some `running` jobs are
genuinely still computing. Reaping those would be a disaster: two processes
would then share one card, and co-tenancy corrupts the timings this rerun
exists to measure. Live orphans are therefore **adopted** -- recorded as
occupying their card so nothing is dispatched there -- and only reaped when
their process actually goes away. Because they are not our children we cannot
read their exit code, so their outcome is taken from `result.json` on disk.

RESUME
------
A reaped job goes back to `pending` with its `job_id` unchanged, so
`JobSpec.output_dir` puts it back in the same directory, so
`CheckpointStore` finds the same journal and replays the clusters that already
completed instead of recomputing them (docs 55 and 61: measured 86 clusters
replayed, 14 computed, bit-identical log Z). The reaper deliberately does not
touch, move, or clear the run directory -- doing so is what would turn a crash
into wasted GPU-hours.

RETRY POLICY
------------
Nick's decision: retry once, then require attention. A job reaped for the first
time returns to `pending`. A job reaped a second time is parked in
`needs_attention`, which nothing dispatches and `--status` prints, so a
deterministically-crashing job cannot spin the sweep in a reap-retry loop.
Clearing it is a deliberate human act (set the status back to `pending`).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from nce.scheduler.jobs import (JobQueue, JobSpec, PENDING, RUNNING, DONE,
                                FAILED, BLOCKED, NEEDS_ATTENTION)

ALIVE, DEAD, UNKNOWN = 'alive', 'dead', 'unknown'

_BOOT_ID_PATH = '/proc/sys/kernel/random/boot_id'

# How many times a job may be reaped and re-queued before it is parked for a
# human. 1 = "retry once, then require attention" (Nick's decision). This is a
# policy constant, not a tuning knob: raising it re-admits the infinite
# reap-retry loop it exists to prevent.
MAX_REAP_RETRIES = 1

# WARN-ONLY, and it never touches the queue. A job is reaped on process
# liveness alone; nothing here can kill or re-queue slow work. This threshold
# only decides when the log says "this has been running a very long time, go
# look". 24 h is ~5x the longest arm measured anywhere in this study (5+ h), so
# it cannot fire on a merely slow job; it exists to surface a genuinely wedged
# process -- e.g. pyGMs' `requests.get` with no timeout, which this codebase has
# hit before and which would otherwise hold a card silently forever.
STALE_WARN_HOURS = 24.0


# ---------------------------------------------------------------------------
# Process identity
# ---------------------------------------------------------------------------
def boot_id() -> Optional[str]:
    """Identifier of the current boot. Changes on reboot; None if unavailable."""
    try:
        with open(_BOOT_ID_PATH) as fh:
            return fh.read().strip()
    except OSError:
        return None


def _read_stat(pid: int) -> Optional[Tuple[str, int]]:
    """(state, starttime_ticks) from /proc/<pid>/stat, or None if it is gone.

    `comm` (field 2) is user-controlled and may contain spaces and ')', so the
    only safe parse is to split *after the last* ')'. Everything after it is
    field 3 onward, hence starttime (field 22) is index 19.
    """
    try:
        with open('/proc/%d/stat' % pid, 'rb') as fh:
            raw = fh.read()
    except OSError:
        return None
    rparen = raw.rfind(b')')
    if rparen < 0:
        return None
    fields = raw[rparen + 1:].split()
    if len(fields) < 20:
        return None
    try:
        return fields[0].decode(), int(fields[19])
    except (ValueError, UnicodeDecodeError):
        return None


def process_identity(pid: int) -> Optional[Dict[str, Any]]:
    """Full identity of a live process, or None if there is no such process.

    Returns None for a ZOMBIE too: it has exited and released its resources
    (including any CUDA context), so for every purpose here it is dead, even
    though `os.kill(pid, 0)` would succeed on it.
    """
    if pid is None or pid <= 0:
        return None
    st = _read_stat(int(pid))
    if st is None:
        return None
    state, starttime = st
    if state == 'Z':
        return None
    return {'boot_id': boot_id(), 'pid': int(pid),
            'starttime_ticks': starttime, 'state': state}


def read_cmdline(pid: int) -> Optional[str]:
    """The one process's own argv, NUL-joined into a string. None if unreadable.

    Targeted read of a single known PID -- emphatically not a scan of the
    process table for a pattern. Unreadable (permissions) and empty (kernel
    thread / zombie) both return None, and a None here is never treated as
    evidence of death on its own.
    """
    try:
        with open('/proc/%d/cmdline' % int(pid), 'rb') as fh:
            raw = fh.read()
    except OSError:
        return None
    if not raw:
        return None
    return raw.replace(b'\x00', b' ').decode('utf-8', errors='replace')


def job_liveness(rec: Dict[str, Any], job_id: str) -> Tuple[str, str]:
    """Is the process recorded for this `running` job still that job?

    Returns (ALIVE | DEAD | UNKNOWN, human-readable reason). UNKNOWN means "the
    PID exists but nothing binds it to this job" and is NEVER reaped: a wrong
    reap dispatches a second job onto an occupied card, which corrupts timings,
    while a missed reap merely leaves a job for a human. The asymmetry decides.
    """
    pid = rec.get('pid')
    if pid is None:
        return DEAD, 'no pid recorded for a running job'
    pid = int(pid)
    if pid == os.getpid():
        return DEAD, ('recorded pid %d is this scheduler itself -- the runner '
                      'died and its pid was reissued to us' % pid)

    cur = process_identity(pid)
    if cur is None:
        return DEAD, 'no live process with pid %d (exited or zombie)' % pid

    recorded = rec.get('proc_identity') or {}
    cmd = read_cmdline(pid)
    has_id = bool(cmd and job_id in cmd)

    if recorded.get('starttime_ticks') is not None:
        if (recorded.get('boot_id') and cur.get('boot_id')
                and recorded['boot_id'] != cur['boot_id']):
            return DEAD, ('machine rebooted since dispatch (boot_id changed); '
                          'pid %d is a different process' % pid)
        if recorded['starttime_ticks'] != cur['starttime_ticks']:
            return DEAD, ('PID REUSE: pid %d exists but started at tick %s, '
                          'not %s -- it is not this job'
                          % (pid, cur['starttime_ticks'],
                             recorded['starttime_ticks']))
        if cmd is not None and not has_id:
            # (boot_id, pid, starttime) is ALREADY a complete identity, so this
            # branch means the two signals disagree -- a corrupt record, or a
            # process that re-exec'd. Deliberately UNKNOWN rather than DEAD:
            # making the weaker signal able to overrule the stronger one could
            # only ever produce a false reap, and a false reap puts a second job
            # on an occupied card.
            return UNKNOWN, ('pid %d matches the recorded start time but its '
                             'cmdline does not carry job id %s; signals '
                             'disagree, refusing to reap' % (pid, job_id))
        return ALIVE, 'pid %d matches recorded start time %s' % (
            pid, cur['starttime_ticks'])

    # Legacy record (dispatched before identities were recorded, e.g. the live
    # phase-1 queue). The cmdline IS the identity binding here, and it is a
    # strong one: a recycled pid will not be running our runner on our job id.
    if cmd is None:
        return UNKNOWN, ('pid %d exists but has no recorded start time and its '
                         'cmdline is unreadable -- cannot prove it is this job, '
                         'refusing to reap' % pid)
    if has_id:
        return ALIVE, 'pid %d cmdline carries job id %s' % (pid, job_id)
    return DEAD, ('PID REUSE: pid %d is a different process (cmdline does not '
                  'mention job id %s)' % (pid, job_id))


# ---------------------------------------------------------------------------
# Outcome recovery
# ---------------------------------------------------------------------------
_RESULT_STATUS = {'done': DONE, 'failed': FAILED, 'blocked': BLOCKED}


def finished_outcome(rec: Dict[str, Any], out_dir: Optional[str],
                     spec: JobSpec) -> Optional[Tuple[str, str]]:
    """(status, note) if the job actually FINISHED before its process vanished.

    A scheduler killed a second after a job wrote `result.json` and exited must
    not re-run that job. `runner.py` writes `result.json` last, so its presence
    is proof of completion -- provided it belongs to THIS attempt. Freshness is
    established by mtime against the attempt's dispatch time; where that is not
    recorded (legacy records) it is accepted only for a first attempt, where
    there is no earlier attempt it could have come from.
    """
    if not out_dir:
        return None
    path = os.path.join(spec.output_dir(out_dir), 'result.json')
    if not os.path.isfile(path):
        return None
    started = rec.get('started_at')
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    if started is not None:
        if mtime < float(started) - 1.0:
            return None                      # left over from a previous attempt
    elif int(rec.get('attempts', 1) or 1) > 1:
        return None                          # cannot tell which attempt it is
    try:
        with open(path) as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return None
    status = _RESULT_STATUS.get(str(payload.get('status', '')).lower())
    if status is None:
        return None
    return status, ('recovered from result.json written at %s'
                    % time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(mtime)))


# ---------------------------------------------------------------------------
# The reaper
# ---------------------------------------------------------------------------
class Reaper:
    """Scans a queue's `running` jobs and repairs the ones whose process died.

    Stateless with respect to the queue -- everything it needs is in the record
    -- so it is safe to run on startup, on every sweep, and from the CLI.
    """

    def __init__(self, out_dir: Optional[str] = None,
                 max_retries: int = MAX_REAP_RETRIES,
                 stale_warn_hours: float = STALE_WARN_HOURS,
                 log=print):
        self.out_dir = out_dir
        self.max_retries = max_retries
        self.stale_warn_hours = stale_warn_hours
        self.log = log
        self._warned: set = set()

    def scan(self, queue: JobQueue,
             owned_job_ids: Iterable[str] = ()) -> List[Dict[str, Any]]:
        """Classify every `running` job. Read-only: nothing is written."""
        owned = set(owned_job_ids or ())
        rows = []
        for spec in queue.by_status(RUNNING):
            rec = queue.get(spec.job_id)
            if spec.job_id in owned:
                # Our own child. `Popen.poll` in Dispatcher._reap is the
                # authority on it; a second opinion here could only race.
                rows.append({'spec': spec, 'rec': rec, 'state': ALIVE,
                             'reason': 'own child, tracked by Popen.poll',
                             'owned': True})
                continue
            state, reason = job_liveness(rec, spec.job_id)
            rows.append({'spec': spec, 'rec': rec, 'state': state,
                         'reason': reason, 'owned': False})
        return rows

    def sweep(self, queue: JobQueue, owned_job_ids: Iterable[str] = (),
              apply: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """Reap dead jobs; report live orphans so the caller can adopt them."""
        out: Dict[str, List[Dict[str, Any]]] = {
            'reaped': [], 'adopted': [], 'recovered': [], 'parked': [],
            'unknown': [], 'owned': []}
        for row in self.scan(queue, owned_job_ids):
            spec, rec, state = row['spec'], row['rec'], row['state']
            if row['owned']:
                out['owned'].append(row)
                self._maybe_warn_slow(spec, rec)
                continue
            if state == ALIVE:
                out['adopted'].append(row)
                self._maybe_warn_slow(spec, rec)
                continue
            if state == UNKNOWN:
                out['unknown'].append(row)
                self.log('[reaper] UNKNOWN %s: %s' % (spec.name, row['reason']))
                continue

            # -- dead ------------------------------------------------------
            done = finished_outcome(rec, self.out_dir, spec)
            if done is not None:
                status, note = done
                row['new_status'] = status
                out['recovered'].append(row)
                self.log('[reaper] %s %s: process gone but the job had already '
                         'finished (%s)' % (status, spec.name, note))
                if apply:
                    queue.set_status(spec.job_id, status,
                                     note=note, reaped_at=time.time())
                continue

            reaps = int(rec.get('reaps', 0) or 0) + 1
            history = list(rec.get('reap_history', []) or [])
            history.append({'at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                            'pid': rec.get('pid'), 'reason': row['reason']})
            if reaps > self.max_retries:
                row['new_status'] = NEEDS_ATTENTION
                out['parked'].append(row)
                self.log('[reaper] NEEDS ATTENTION %s: reaped %d times '
                         '(limit %d). Not re-queueing -- a job that keeps '
                         'dying will not be allowed to loop. Last reason: %s'
                         % (spec.name, reaps, self.max_retries, row['reason']))
                if apply:
                    queue.set_status(
                        spec.job_id, NEEDS_ATTENTION, reaps=reaps,
                        reap_history=history, reaped_at=time.time(),
                        note=('reaped %d times; set status back to "pending" '
                              'to retry: %s' % (reaps, row['reason']))[:500])
                continue

            row['new_status'] = PENDING
            out['reaped'].append(row)
            self.log('[reaper] REQUEUED %s: %s (attempt %s; its checkpoint '
                     'journal will be replayed, not recomputed)'
                     % (spec.name, row['reason'], rec.get('attempts')))
            if apply:
                queue.set_status(spec.job_id, PENDING, reaps=reaps,
                                 reap_history=history, reaped_at=time.time(),
                                 pid=None, proc_identity=None, gpu_uuid=None,
                                 note=('re-queued by reaper: %s'
                                       % row['reason'])[:500])
        return out

    def _maybe_warn_slow(self, spec: JobSpec, rec: Dict[str, Any]) -> None:
        """Log-only notice for an implausibly long-lived job. Never re-queues.

        See STALE_WARN_HOURS: this cannot reap anything, so a slow-but-healthy
        job is safe from it by construction.
        """
        started = rec.get('started_at')
        if not started or spec.job_id in self._warned:
            return
        hours = (time.time() - float(started)) / 3600.0
        if hours >= self.stale_warn_hours:
            self._warned.add(spec.job_id)
            self.log('[reaper] STALE-WARN %s has been running %.1f h (> %.0f h). '
                     'It is ALIVE so nothing has been done to it; check whether '
                     'it is wedged (e.g. a no-timeout download).'
                     % (spec.name, hours, self.stale_warn_hours))


# ---------------------------------------------------------------------------
# Orphaned ballast: report only, never kill without being asked
# ---------------------------------------------------------------------------
def orphan_ballast(own_pids: Iterable[int] = (),
                   log_dir: Optional[str] = None,
                   gpu_indices: Optional[Iterable[int]] = None
                   ) -> List[Dict[str, Any]]:
    """Ballast workers on a card that no live scheduler owns.

    A scheduler killed with SIGKILL never reaches `BallastPool.stop_all`, so its
    workers survive, keep a CUDA context, and make their cards look BUSY to the
    next scheduler -- which then dispatches nothing at all, on a card that is
    doing no work. And a worker the dead scheduler's runner had PAUSED will
    never be resumed or stopped by anyone, so that card is wedged permanently.
    Same stall as a stuck `running` job, different cause, so it is named here
    rather than left as a mystery.

    Detection starts from the driver's own compute-app list (the authoritative
    busy signal `gpus.py` uses) and then reads only those specific PIDs'
    cmdlines -- no pattern scan of the process table.

    `ours` is decided by whether the worker's control-file arguments point into
    THIS scheduler's `log_dir` (`<out-dir>/_ballast`), which is a path only our
    own pool constructs. Another agent's scheduler may legitimately own ballast
    on a card and killing it would cool their card and bias THEIR timings, so
    only workers proven ours are ever candidates for teardown.
    """
    from nce.scheduler.gpus import query_gpus
    own = set(own_pids or ())
    me = os.getpid()
    only = None if gpu_indices is None else set(gpu_indices)
    found = []
    for g in query_gpus():
        if only is not None and g.index not in only:
            # Cards we were told not to use are none of our business; reporting
            # on them only produces alarming noise about another agent's run.
            continue
        for pid in g.compute_pids:
            if pid in own or pid == me:
                continue
            cmd = read_cmdline(pid)
            if cmd and 'nce.scheduler.ballast' in cmd:
                found.append({'pid': pid, 'gpu_index': g.index,
                              'gpu_uuid': g.uuid, 'cmdline': cmd.strip(),
                              'ours': bool(log_dir
                                           and os.path.abspath(log_dir) in cmd)})
    return found


def terminate_orphan_ballast(orph: Dict[str, Any], timeout: float = 30.0,
                             log=print) -> bool:
    """SIGTERM one of OUR orphaned ballast workers and wait for it to exit.

    Waiting matters: a CUDA context is released at process exit, not at
    signal-delivery time, so returning early would let a job be dispatched onto
    a card the worker still holds. The worker's own SIGTERM handler exits
    between GEMM iterations (a few hundred ms), and it exits from the paused
    state too, so this is a sub-second operation in practice.
    """
    if not orph.get('ours'):
        return False
    pid = int(orph['pid'])
    try:
        os.kill(pid, 15)
    except OSError as e:
        log('[reaper] could not signal orphan ballast pid %d: %s' % (pid, e))
        return process_identity(pid) is None
    t0 = time.time()
    while time.time() - t0 < timeout:
        if process_identity(pid) is None:
            log('[reaper] orphaned ballast pid %d on cuda:%s released the card '
                'in %.2fs' % (pid, orph.get('gpu_index'), time.time() - t0))
            return True
        time.sleep(0.05)
    log('[reaper] orphaned ballast pid %d on cuda:%s did NOT exit in %.0fs; '
        'that card will keep looking busy' % (pid, orph.get('gpu_index'),
                                              timeout))
    return False


# ---------------------------------------------------------------------------
# CLI: `python -m nce.scheduler.reaper --queue Q --out-dir D [--apply]`
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        description='Inspect (and optionally repair) jobs stuck at "running". '
                    'Dry-run by default.')
    p.add_argument('--queue', required=True)
    p.add_argument('--out-dir', default=None,
                   help='run root, so a job that finished just before its '
                        'process died is recovered from result.json instead of '
                        'being re-run')
    p.add_argument('--apply', action='store_true',
                   help='actually write the queue (default: report only)')
    p.add_argument('--max-reap-retries', type=int, default=MAX_REAP_RETRIES)
    args = p.parse_args(argv)

    q = JobQueue(args.queue)
    r = Reaper(out_dir=args.out_dir, max_retries=args.max_reap_retries)
    if not args.apply:
        for row in r.scan(q):
            state = row['state']
            extra = ''
            if state == DEAD:
                fin = finished_outcome(row['rec'], args.out_dir, row['spec'])
                if fin is not None:
                    state, extra = 'FINISHED', ' -> would be marked %s' % fin[0]
            print('  %-8s %s\n           %s%s'
                  % (state.upper(), row['spec'].name, row['reason'], extra))
        print('(dry run -- pass --apply to repair)')
        return 0

    # `--apply` is for a queue whose scheduler is GONE. A running scheduler
    # reaps its own queue every sweep and, unlike this CLI, knows which jobs are
    # its own children; a second writer could re-queue a job it is about to mark
    # done. Say so rather than let it be discovered.
    print('NOTE: only use --apply when the scheduler for this queue is dead. A '
          'live scheduler already reaps this queue on every sweep.')
    res = r.sweep(q, apply=True)
    print('reaped=%d recovered=%d parked=%d still-alive=%d unknown=%d'
          % (len(res['reaped']), len(res['recovered']), len(res['parked']),
             len(res['adopted']), len(res['unknown'])))
    print(q.summary())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
