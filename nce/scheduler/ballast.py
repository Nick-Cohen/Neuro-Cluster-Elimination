"""Thermal ballast: keep an otherwise-idle GPU at its working temperature.

WHY THIS EXISTS
---------------
Doc 59 measured the problem this module removes. A TITAN RTX on this box cools
from 83 C to 61 C in 60 seconds. The scheduler's per-job setup (model load,
elimination order, checkpoint write, process teardown/startup) idles the card
for exactly that sort of interval, so without ballast EVERY job starts cold and
then warms up while it is being timed.

That would not matter if it were a constant offset. It is not. The cold-start
penalty is amortised over the job, so it scales with job length:

    30 s job    -5.59 %
    1500 s job  -0.49 %

i.e. a ~5 percentage-point bias that is CORRELATED WITH RUN TIME. Run time is
the quantity the paper measures. It systematically exaggerates the gap between
fast and slow merge bounds -- short jobs look disproportionately slow. Seeds and
medians cannot remove a bias that is correlated with the treatment; only keeping
the card warm can.

WHAT IT DOES
------------
When the scheduler has no job for a dispatchable card, it starts a ballast
process there: a saturating fp32 GEMM loop. When a real job arrives for that
card the ballast is torn down first and the scheduler waits for the CUDA context
to actually be released before launching.

THE THREE HARD REQUIREMENTS, AND HOW EACH IS MET
------------------------------------------------
1. "Yields instantly when a real job arrives."
   The worker installs SIGTERM/SIGINT handlers that set a flag checked between
   GEMM iterations, and it also polls a stop-file (belt and braces, for the case
   where the signal is lost). One iteration is a few hundred ms, so the yield is
   sub-second; `BallastPool.stop` then blocks on `proc.wait()`, which is what
   actually guarantees the CUDA context is gone -- a context is released at
   process exit, not at flag-set time. The scheduler never launches a job onto a
   card whose ballast has not been reaped.

2. "Never touches gpu2."
   Three independent guards: the pool filters `RETIRED_GPU_INDICES`, the worker
   calls `assert_not_retired` on its own `--gpu` before importing torch, and the
   pool refuses to start on a card it did not get from `dispatchable_gpus()`.
   gpu2 is retired for SILENT data corruption, so a ballast GEMM there would not
   even fail visibly -- it must simply never be sent.

3. "Does not perturb a running job's timings."
   Guaranteed by construction rather than by tuning: ballast only ever runs on a
   card with no job on it, and is stopped before a job starts on that card.
   Cards are independent thermal and compute domains, so ballast on cuda:0
   cannot affect a job on cuda:3. `BallastPool.ensure` is given the set of cards
   the scheduler knows to be job-free; it never widens that set.

THE INTERACTION THAT NEARLY BROKE THIS
--------------------------------------
`gpus.query_gpus` treats "a compute process is attached" as the authoritative
busy signal -- correctly, since that is a fact rather than a threshold. Ballast
attaches a compute process. Naively adding ballast therefore makes every card
look permanently BUSY and the scheduler stops dispatching entirely: a deadlock
in which the machine is 100% utilised and 0% productive. That is why
`dispatchable_gpus` and `query_gpus` take `ignore_pids`, and why the pool is the
authority on which pids are its own.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Optional

from nce.scheduler.gpus import (RETIRED_GPU_INDICES, assert_not_retired,
                                query_gpus)

# Doc 59 saturated both timing-pool cards with a 4 x 8192^2 fp32 GEMM and
# measured the full 280 W power cap with the thermal throttle bit never set on
# gpu0 or gpu3 across 25 minutes. These defaults are that measured workload, not
# invented numbers. Override per-site if the rerun's own load differs.
BALLAST_MATRIX_DIM = int(os.environ.get('NCE_BALLAST_DIM', '8192'))
BALLAST_BATCH = int(os.environ.get('NCE_BALLAST_BATCH', '4'))

# How long to wait for a ballast process to release its CUDA context before
# giving up and reporting the card unusable this sweep. A single GEMM iteration
# is a few hundred ms; 30 s is three orders of magnitude of headroom and exists
# only so a wedged worker cannot hang the scheduler forever.
BALLAST_STOP_TIMEOUT_S = float(os.environ.get('NCE_BALLAST_STOP_TIMEOUT_S', '30'))


# ---------------------------------------------------------------------------
# Worker: `python -m nce.scheduler.ballast --gpu N`
# ---------------------------------------------------------------------------
def _worker(gpu: int, stop_file: Optional[str], report_every: float,
            pause_file: Optional[str] = None,
            paused_marker: Optional[str] = None) -> int:
    # Guard BEFORE torch touches the device.
    assert_not_retired(gpu)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    import torch
    if not torch.cuda.is_available():
        print('[ballast] no CUDA visible for cuda:%d; exiting' % gpu, flush=True)
        return 1

    stopping = {'flag': False}

    def _handle(signum, frame):
        stopping['flag'] = True

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    dev = torch.device('cuda:0')  # remapped by CUDA_VISIBLE_DEVICES
    n = BALLAST_MATRIX_DIM

    def _alloc():
        return (torch.randn(n, n, device=dev, dtype=torch.float32),
                torch.randn(n, n, device=dev, dtype=torch.float32))

    a, b = _alloc()
    print('[ballast] cuda:%d up: %d x %dx%d fp32 GEMM loop (pid %d)'
          % (gpu, BALLAST_BATCH, n, n, os.getpid()), flush=True)

    def _mark_paused(state):
        """Attest, from inside the worker, that no more kernels are queued."""
        if not paused_marker:
            return
        if state:
            with open(paused_marker, 'w') as fh:
                fh.write('%f\n' % time.time())
        else:
            try:
                os.remove(paused_marker)
            except OSError:
                pass

    last_report = time.time()
    iters = 0
    paused = False
    while not stopping['flag']:
        # ---- PAUSE: stop feeding the card, but keep the process alive -----
        # Exiting instead would mean a fresh process (python + torch import +
        # CUDA context ~5 s) before the NEXT job's setup could be covered --
        # far too slow, since with a saturated queue the next job is dispatched
        # ~0.4 s after the last one finishes. So ballast idles in place: it
        # frees its matrices and stops submitting work, holding only the bare
        # CUDA context. No kernels queued => no SM contention => no perturbation
        # of the job that is about to be timed. (Verified, doc 60 section 6.4.)
        if pause_file and os.path.exists(pause_file):
            if not paused:
                del a, b
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                paused = True
                _mark_paused(True)
                print('[ballast] cuda:%d paused (context held, no work queued)'
                      % gpu, flush=True)
            time.sleep(0.05)
            if stop_file and os.path.exists(stop_file):
                break
            continue
        if paused:
            a, b = _alloc()
            paused = False
            _mark_paused(False)
            print('[ballast] cuda:%d resumed' % gpu, flush=True)
        for _ in range(BALLAST_BATCH):
            c = a @ b
        # One sync per batch bounds the yield latency: without it the queue
        # could run far ahead of the stop flag and the process would keep the
        # card busy long after it was told to stop.
        torch.cuda.synchronize()
        del c
        iters += 1
        # The stop-file is the RUNNER's channel (see runner.yield_ballast): it
        # is how a job about to touch the GPU tells ballast to get off the card.
        # It is checked every batch, right after the sync, so the flag can never
        # be observed while work is still queued.
        if stop_file and os.path.exists(stop_file):
            break
        now = time.time()
        if report_every and now - last_report >= report_every:
            last_report = now
            print('[ballast] cuda:%d alive, %d batches' % (gpu, iters), flush=True)

    print('[ballast] cuda:%d yielding after %d batches' % (gpu, iters), flush=True)
    return 0


# ---------------------------------------------------------------------------
# Pool: owned by the Scheduler
# ---------------------------------------------------------------------------
class BallastPool:
    """Starts/stops ballast workers and knows which pids are its own.

    The scheduler must consult `pids()` whenever it asks the driver which cards
    are busy, otherwise ballast masquerades as real work (see module docstring).
    """

    def __init__(self, python: str = None, enabled: bool = True,
                 log_dir: str = None, cwd: str = None):
        self.python = python or sys.executable
        self.enabled = enabled
        self.log_dir = log_dir
        self.cwd = cwd
        # gpu_index -> {'proc': Popen, 'started': float, 'log': fh|None}
        self.procs: Dict[int, Dict] = {}

    # -- introspection -----------------------------------------------------
    def pids(self) -> set:
        """PIDs of live ballast workers. Fed to gpus.query_gpus(ignore_pids=...)."""
        out = set()
        for rec in self.procs.values():
            if rec['proc'].poll() is None:
                out.add(rec['proc'].pid)
        return out

    def active_indices(self) -> set:
        return {i for i, rec in self.procs.items() if rec['proc'].poll() is None}

    def reap(self) -> list:
        """Drop records for workers that have already exited.

        A handed-off worker exits on the runner's stop-file, not on a stop()
        call from us, so without this its record (and its open log fd) would
        linger and `start()` would overwrite it, leaking the descriptor.
        """
        gone = [i for i, rec in self.procs.items() if rec['proc'].poll() is not None]
        for i in gone:
            self._close(i)
        return gone

    # -- lifecycle ---------------------------------------------------------
    def stop_file_for(self, gpu_index: int) -> str:
        """Path the RUNNER touches to make ballast yield this card.

        Deterministic and per-card so the runner can be told it on its command
        line and needs no other channel back to the scheduler.
        """
        d = self.log_dir or os.path.join(os.path.sep, 'tmp')
        return os.path.join(d, 'ballast-cuda%d.stop' % gpu_index)

    def pause_file_for(self, gpu_index: int) -> str:
        d = self.log_dir or os.path.join(os.path.sep, 'tmp')
        return os.path.join(d, 'ballast-cuda%d.pause' % gpu_index)

    def paused_marker_for(self, gpu_index: int) -> str:
        d = self.log_dir or os.path.join(os.path.sep, 'tmp')
        return os.path.join(d, 'ballast-cuda%d.paused' % gpu_index)

    def handoff_info(self, gpu_index: int):
        """(pid, pause_file, paused_marker) for a live worker, else None.

        This is what `Dispatcher._launch` passes to the runner so the runner can
        keep the card warm through its own CPU-bound setup and take the card
        itself, immediately before its first GPU work.
        """
        rec = self.procs.get(gpu_index)
        if rec is None or rec['proc'].poll() is not None:
            return None
        return (rec['proc'].pid, self.pause_file_for(gpu_index),
                self.paused_marker_for(gpu_index))

    def resume(self, gpu_index: int) -> None:
        """Let ballast start feeding the card again after a job has finished."""
        for p in (self.pause_file_for(gpu_index),):
            try:
                os.remove(p)
            except OSError:
                pass

    def start(self, gpu_index: int) -> bool:
        if not self.enabled:
            return False
        if gpu_index in RETIRED_GPU_INDICES:
            return False
        assert_not_retired(gpu_index)
        if gpu_index in self.procs and self.procs[gpu_index]['proc'].poll() is None:
            return False
        stop_file = self.stop_file_for(gpu_index)
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
        # Stale control files left by a previous job would either kill the new
        # worker on its first check or start it already paused, silently
        # disabling ballast for the rest of the sweep. Clear them before
        # starting, never after.
        for stale in (stop_file, self.pause_file_for(gpu_index),
                      self.paused_marker_for(gpu_index)):
            try:
                os.remove(stale)
            except OSError:
                pass
        cmd = [self.python, '-m', 'nce.scheduler.ballast', '--gpu', str(gpu_index),
               '--stop-file', stop_file,
               '--pause-file', self.pause_file_for(gpu_index),
               '--paused-marker', self.paused_marker_for(gpu_index)]
        log = None
        if self.log_dir:
            log = open(os.path.join(self.log_dir, 'ballast-cuda%d.log' % gpu_index), 'ab')
        proc = subprocess.Popen(cmd, stdout=log or subprocess.DEVNULL,
                                stderr=subprocess.STDOUT, cwd=self.cwd)
        self.procs[gpu_index] = {'proc': proc, 'started': time.time(), 'log': log,
                                 'stop_file': stop_file}
        print('[ballast] cuda:%d <- ballast (pid %d)' % (gpu_index, proc.pid),
              flush=True)
        return True

    def stop(self, gpu_index: int, timeout: float = None) -> bool:
        """Tear down ballast on one card and WAIT for the context to be released.

        Returns True if the card is now free of our ballast. The wait is the
        whole point: a CUDA context lives until process exit, so returning
        before the child is reaped would let a job start against a card that is
        still occupied.
        """
        rec = self.procs.get(gpu_index)
        if rec is None:
            return True
        proc = rec['proc']
        if proc.poll() is not None:
            self._close(gpu_index)
            return True
        timeout = BALLAST_STOP_TIMEOUT_S if timeout is None else timeout
        t0 = time.time()
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print('[ballast] cuda:%d did not yield in %.0fs; killing'
                  % (gpu_index, timeout), flush=True)
            proc.kill()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                print('[ballast] cuda:%d WEDGED; not dispatching there'
                      % gpu_index, flush=True)
                return False
        dt = time.time() - t0
        print('[ballast] cuda:%d yielded in %.2fs' % (gpu_index, dt), flush=True)
        self._close(gpu_index)
        return True

    def stop_all(self) -> None:
        for idx in list(self.procs):
            self.stop(idx)

    def _close(self, gpu_index: int) -> None:
        rec = self.procs.pop(gpu_index, None)
        if rec and rec.get('log'):
            try:
                rec['log'].close()
            except Exception:
                pass

    def ensure(self, free_indices: Iterable[int],
               keep_indices: Iterable[int] = ()) -> None:
        """Start ballast on job-free cards; keep it alive on cards we still own.

        Two sets, and conflating them is a real bug I shipped and had to catch
        end-to-end:

        `free_indices`  cards with no job -- START a worker here.
        `keep_indices`  cards running one of OUR jobs. Their worker has been
                        PAUSED by the runner and must be left completely alone:
                        not stopped (it has to resume the instant the job ends,
                        and the next job needs something to hand off to) and
                        emphatically not (re)started, since a fresh worker comes
                        up UNPAUSED and would run GEMM against a job that is
                        being timed.

        Anything in neither set has left our control and is torn down. This
        method never starts a worker on a card outside `free_indices`, which is
        what makes "ballast never perturbs a running job" structural rather than
        best-effort.
        """
        if not self.enabled:
            return
        self.reap()
        free = {i for i in free_indices if i not in RETIRED_GPU_INDICES}
        keep = {i for i in keep_indices if i not in RETIRED_GPU_INDICES}
        for idx in list(self.active_indices() - free - keep):
            self.stop(idx)
        for idx in sorted(free - self.active_indices()):
            self.start(idx)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--stop-file', default=None,
                    help='exit when this path appears (secondary to SIGTERM)')
    ap.add_argument('--pause-file', default=None,
                    help='while this path exists, hold the context but queue no '
                         'work (the runner uses this to take the card)')
    ap.add_argument('--paused-marker', default=None,
                    help='written by this worker once it has actually stopped')
    ap.add_argument('--report-every', type=float, default=60.0)
    args = ap.parse_args(argv)
    return _worker(args.gpu, args.stop_file, args.report_every,
                   args.pause_file, args.paused_marker)


if __name__ == '__main__':
    raise SystemExit(main())
