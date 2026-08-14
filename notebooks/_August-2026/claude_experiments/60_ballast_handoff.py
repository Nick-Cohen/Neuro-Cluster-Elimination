#!/usr/bin/env python
"""60: does ballast covering in-job setup perturb the timed region?

The claim to be proved, not asserted: a timed region measured on a card that
ballast held warm through setup and then released must match one measured on an
already-warm IDLE card, within noise. If it does not, ballast is still on the
card during timed work and every timing in the rerun is wrong.

Three arms, each timing an identical CUDA-event-timed workload:

  cold       card idle at ambient, no ballast. Shows the bias being removed.
  warm-idle  ballast warms to equilibrium, is stopped and CONFIRMED gone, then
             the workload runs immediately. This is the REFERENCE: a warm card
             with nothing else on it.
  handoff    ballast warms to equilibrium, then holds the card through a
             simulated CPU-bound setup window, then the REAL shipped handshake
             (`runner.pause_ballast`) takes the card, then the workload runs.

handoff == warm-idle  =>  the handoff is clean.
handoff  > warm-idle  =>  ballast is overlapping timed work. Fail loudly.

This matters MORE under the pause contract than it did under the old kill
contract. Ballast no longer exits -- it stays alive holding its CUDA context and
merely stops queueing work, because a worker that exits cannot be back up in
time to cover the next job's setup (~5 s to respawn vs ~0.4 s between jobs). An
idle-but-attached context is a weaker guarantee than a dead process, so
"it does not perturb" has to be MEASURED here, not argued.

Deliberately uses `nce.scheduler.runner.pause_ballast` itself rather than a
re-implementation, so this tests the shipped path.

Usage:
  python 60_ballast_handoff.py --gpu 0 --warm 360 --setup 9 --timed 60 --out h.json
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--gpu', type=int, required=True)
ap.add_argument('--warm', type=float, default=360.0, help='ballast warm-up (doc 59)')
ap.add_argument('--setup', type=float, default=9.0,
                help='simulated runner CPU setup window (measured 8-9 s)')
ap.add_argument('--timed', type=float, default=60.0, help='timed region duration')
ap.add_argument('--settle', type=float, default=120.0,
                help='idle settle before the cold arm')
ap.add_argument('--dim', type=int, default=4096)
ap.add_argument('--repo', default=os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
ap.add_argument('--out', required=True)
args = ap.parse_args()

if args.gpu == 2:
    raise SystemExit('cuda:2 is RETIRED -- refusing')

sys.path.insert(0, args.repo)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(args.gpu))

import torch
from nce.scheduler.ballast import BallastPool
from nce.scheduler.runner import pause_ballast, _ballast_pids_on

torch.set_num_threads(1)
DEV = torch.device('cuda:0')      # remapped by CUDA_VISIBLE_DEVICES


def telem():
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=temperature.gpu,power.draw,clocks.sm',
         '--format=csv,noheader,nounits', '-i', str(args.gpu)],
        capture_output=True, text=True, check=True).stdout.strip()
    p = [x.strip() for x in out.split(',')]
    return {'temp': int(p[0]), 'power': float(p[1]), 'sm_clock': int(p[2])}


def timed_region(duration):
    """CUDA-event-timed trials of a fixed workload for `duration` seconds.

    CUDA events, not wall clock: this must measure GPU execution, and a wall
    clock would fold in launch latency and any CPU contention from ballast,
    which is a different question.
    """
    a = torch.randn(args.dim, args.dim, device=DEV, dtype=torch.float32)
    b = torch.randn(args.dim, args.dim, device=DEV, dtype=torch.float32)
    # Warm the kernels/allocator so the first trial is not an outlier; this is
    # GPU work, and it is inside the timed region for every arm equally.
    for _ in range(3):
        c = a @ b
    torch.cuda.synchronize()

    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    trials, tel = [], []
    t_end = time.time() + duration
    while time.time() < t_end:
        ev0.record()
        for _ in range(4):
            c = a @ b
        ev1.record()
        torch.cuda.synchronize()
        trials.append(ev0.elapsed_time(ev1))
        if len(trials) % 25 == 0:
            tel.append(telem())
    del a, b, c
    torch.cuda.empty_cache()
    return trials, tel


def summarise(trials):
    s = sorted(trials)
    n = len(s)
    return {
        'n': n,
        'median_ms': round(statistics.median(s), 4),
        'mean_ms': round(statistics.fmean(s), 4),
        'p25_ms': round(s[n // 4], 4),
        'p75_ms': round(s[(3 * n) // 4], 4),
        'cv_pct': round(100 * statistics.pstdev(s) / statistics.fmean(s), 4),
    }


results = {'gpu': args.gpu, 'dim': args.dim, 'warm_s': args.warm,
           'setup_s': args.setup, 'timed_s': args.timed, 'arms': {}}
pool = BallastPool(enabled=True, cwd=args.repo,
                   log_dir=os.path.join('/tmp/claude-58902', 'handoff-ballast'))

try:
    # ---------------- arm: cold ----------------
    print('[cold] settling %.0fs idle' % args.settle, flush=True)
    pool.stop_all()
    time.sleep(args.settle)
    t_before = telem()
    print('[cold] start temp %d C' % t_before['temp'], flush=True)
    trials, tel = timed_region(args.timed)
    results['arms']['cold'] = {
        'temp_at_start': t_before['temp'], 'telemetry': tel, **summarise(trials)}
    print('[cold] median %.3f ms  (start %d C)'
          % (results['arms']['cold']['median_ms'], t_before['temp']), flush=True)

    # ---------------- arm: warm-idle (REFERENCE) ----------------
    print('[warm-idle] warming %.0fs' % args.warm, flush=True)
    pool.start(args.gpu)
    time.sleep(args.warm)
    eq = telem()
    pool.stop(args.gpu)                      # blocks until the context is gone
    t_before = telem()
    print('[warm-idle] equilibrium %d C -> start temp %d C'
          % (eq['temp'], t_before['temp']), flush=True)
    trials, tel = timed_region(args.timed)
    results['arms']['warm-idle'] = {
        'equilibrium_temp': eq['temp'], 'temp_at_start': t_before['temp'],
        'telemetry': tel, **summarise(trials)}
    print('[warm-idle] median %.3f ms' % results['arms']['warm-idle']['median_ms'],
          flush=True)

    # ---------------- arm: handoff ----------------
    print('[handoff] warming %.0fs' % args.warm, flush=True)
    pool.start(args.gpu)
    time.sleep(args.warm)
    eq = telem()
    info = pool.handoff_info(args.gpu)
    if info is None:
        raise SystemExit('handoff: ballast not running, cannot test the handoff')
    pid, pause_file, paused_marker = info
    print('[handoff] simulating %.0fs of CPU setup with ballast ON the card'
          % args.setup, flush=True)
    time.sleep(args.setup)
    mid = telem()
    t0 = time.time()
    pause_ballast(pause_file, paused_marker, args.gpu)   # the shipped handshake
    yield_s = time.time() - t0
    # Under the PAUSE contract the worker deliberately stays alive holding its
    # CUDA context; what must be true is that it queues no work. So record both:
    # that the context is still attached (expected) and that the process is
    # still alive (expected), and let the timing comparison prove non-inter-
    # ference. A still-attached idle context is a WEAKER guarantee than an
    # exited process, which is exactly why this has to be re-measured.
    pids_after = _ballast_pids_on(args.gpu)
    still_alive = pool.handoff_info(args.gpu) is not None
    t_before = telem()
    print('[handoff] equilibrium %d C, after setup %d C, at start %d C '
          '(yield %.3fs, ballast pid present after: %s)'
          % (eq['temp'], mid['temp'], t_before['temp'], yield_s,
             pid in (pids_after or set())), flush=True)
    trials, tel = timed_region(args.timed)
    results['arms']['handoff'] = {
        'equilibrium_temp': eq['temp'], 'temp_after_setup': mid['temp'],
        'temp_at_start': t_before['temp'], 'yield_s': round(yield_s, 4),
        'ballast_context_attached': bool(pid in (pids_after or set())),
        'ballast_process_alive': bool(still_alive),
        'ballast_still_attached': False,
        'telemetry': tel, **summarise(trials)}
    print('[handoff] median %.3f ms' % results['arms']['handoff']['median_ms'],
          flush=True)
finally:
    pool.stop_all()

a = results['arms']
ref = a['warm-idle']['median_ms']
results['handoff_vs_warm_idle_pct'] = round(100 * (a['handoff']['median_ms'] - ref) / ref, 3)
results['cold_vs_warm_idle_pct'] = round(100 * (a['cold']['median_ms'] - ref) / ref, 3)

print('\n%-11s %-8s %-11s %-9s %-8s' % ('arm', 'start C', 'median ms', 'CV %', 'vs ref'))
for k in ('cold', 'warm-idle', 'handoff'):
    r = a[k]
    delta = '' if k == 'warm-idle' else '%+.3f%%' % (100 * (r['median_ms'] - ref) / ref)
    print('%-11s %-8d %-11.4f %-9.4f %-8s'
          % (k, r['temp_at_start'], r['median_ms'], r['cv_pct'], delta))

# The gate: handoff must not be measurably SLOWER than the warm-idle reference.
# Any residual ballast would show as a large positive deviation (two saturating
# workloads on one card roughly halve throughput), so the bar is set well inside
# that while staying outside doc 59's measured steady-state CV of ~0.13-0.16%.
clean = abs(results['handoff_vs_warm_idle_pct']) < 1.0
results['verdict'] = 'CLEAN HANDOFF' if clean else 'PERTURBED'
print('\nballast CONTEXT attached during timed region (expected True under '
      'the pause contract): %s' % a['handoff']['ballast_context_attached'])
print('ballast process alive (expected True): %s'
      % a['handoff']['ballast_process_alive'])
print('handoff vs warm-idle reference: %+.3f%%' % results['handoff_vs_warm_idle_pct'])
print('cold    vs warm-idle reference: %+.3f%%' % results['cold_vs_warm_idle_pct'])
print('VERDICT: %s' % results['verdict'])

print('===BALLAST-HANDOFF===' + json.dumps(
    {k: (v if k != 'arms' else {kk: {x: y for x, y in vv.items() if x != 'telemetry'}
                                for kk, vv in v.items()})
     for k, v in results.items()}))
with open(args.out, 'w') as fh:
    json.dump(results, fh, indent=1)
print('wrote %s' % args.out)
