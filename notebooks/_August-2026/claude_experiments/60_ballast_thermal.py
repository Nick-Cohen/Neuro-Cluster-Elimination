#!/usr/bin/env python
"""60: does the ballast loop actually hold a card's temperature across a job boundary?

Runs the REAL scheduler on a REAL 2-job queue, once with ballast and once with
--no-ballast, sampling card telemetry every second throughout. The quantity of
interest is the temperature at the moment the SECOND job is dispatched: that is
the "every job starts cold" bias doc 59 measured, and it is the thing ballast
exists to remove.

Deliberately end-to-end rather than a microbenchmark of the GEMM: the failure
mode being guarded against is an integration one (ballast masquerading as a busy
GPU, or not yielding in time), which a microbenchmark cannot see.

Usage:
    python 60_ballast_thermal.py --arm ballast   --gpu 3 --out b.json
    python 60_ballast_thermal.py --arm noballast --gpu 3 --out n.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

ap = argparse.ArgumentParser()
ap.add_argument('--arm', required=True, choices=['ballast', 'noballast'])
ap.add_argument('--gpu', type=int, required=True)
ap.add_argument('--repo', default=os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
ap.add_argument('--problem', default='grids/grid10x10.f10')
ap.add_argument('--strategy', default='reduce_nn')
ap.add_argument('--bound', type=int, default=4)
ap.add_argument('--seeds', nargs='+', type=int, default=[42, 43])
ap.add_argument('--out', required=True)
ap.add_argument('--workdir', required=True)
args = ap.parse_args()

if args.gpu == 2:
    raise SystemExit('cuda:2 is RETIRED -- refusing')

sys.path.insert(0, args.repo)
os.makedirs(args.workdir, exist_ok=True)

# ---- build the queue -------------------------------------------------------
from nce.scheduler.jobs import JobQueue, build_grid

BASE_CONFIG = {
    # cuda_grid_plain from tests/test_determinism_regression.py, minus the
    # merge fields (JobSpec.to_config owns those) and minus problem_key/seed
    # (owned by the JobSpec). Config values are user intent: nothing here is
    # overridden by this harness.
    'neurobe_mode': True,
    'iB': 10,
    'ecl': 1025,
    'num_samples': 'nbe,0.1',
    'sampling_scheme': 'uniform',
    'stream_nn_exact': True,
    'dope_factors': True,
    'device': 'cuda',
}

qpath = os.path.join(args.workdir, 'Q.json')
if os.path.exists(qpath):
    os.remove(qpath)
q = JobQueue(qpath)
specs = build_grid([args.problem], [args.strategy], [args.bound], args.seeds,
                   base_config=BASE_CONFIG, tag='60-ballast-%s' % args.arm)
q.add(specs)
print('queue: %d jobs -> %s' % (len(specs), qpath), flush=True)

# ---- telemetry sampler -----------------------------------------------------
samples = []
stop = threading.Event()


def sample():
    q = ('--query-gpu=temperature.gpu,power.draw,utilization.gpu,clocks.sm,'
         'memory.used')
    while not stop.is_set():
        t = time.time()
        try:
            out = subprocess.run(
                ['nvidia-smi', q, '--format=csv,noheader,nounits',
                 '-i', str(args.gpu)],
                capture_output=True, text=True, check=True).stdout.strip()
            parts = [p.strip() for p in out.split(',')]
            samples.append({
                't': t, 'temp': int(parts[0]), 'power': float(parts[1]),
                'util': int(parts[2]), 'sm_clock': int(parts[3]),
                'mem': int(parts[4]),
            })
        except Exception as e:                                  # pragma: no cover
            samples.append({'t': t, 'error': str(e)})
        stop.wait(1.0)


th = threading.Thread(target=sample, daemon=True)
th.start()

# ---- run the scheduler -----------------------------------------------------
outdir = os.path.join(args.workdir, 'runs')
cmd = [sys.executable, '-m', 'nce.scheduler.scheduler',
       '--queue', qpath, '--out-dir', outdir,
       '--only-gpus', str(args.gpu), '--threads', '1']
if args.arm == 'noballast':
    cmd.append('--no-ballast')

print('launching: %s' % ' '.join(cmd), flush=True)
events = []
t_start = time.time()
proc = subprocess.Popen(cmd, cwd=args.repo, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, bufsize=1,
                        env=dict(os.environ))
log_lines = []
for line in proc.stdout:
    ts = time.time()
    line = line.rstrip()
    log_lines.append({'t': ts, 'line': line})
    print('  | %s' % line, flush=True)
    if line.startswith('[dispatch]'):
        events.append({'t': ts, 'kind': 'dispatch', 'line': line})
    elif line.startswith('[done]') or line.startswith('[DONE]'):
        events.append({'t': ts, 'kind': 'done', 'line': line})
    elif re.match(r'^\[(done|failed|blocked)\] ', line):
        events.append({'t': ts, 'kind': 'finish', 'line': line})
    elif line.startswith('[ballast]'):
        events.append({'t': ts, 'kind': 'ballast', 'line': line})
rc = proc.wait()
stop.set()
th.join(timeout=5)
t_end = time.time()
print('scheduler rc=%d, wall %.1fs' % (rc, t_end - t_start), flush=True)

# ---- analysis --------------------------------------------------------------
dispatches = [e for e in events if e['kind'] == 'dispatch']


def temp_at(t):
    """Telemetry sample nearest in time to t."""
    if not samples:
        return None
    best = min(samples, key=lambda s: abs(s['t'] - t))
    return best


res = {
    'arm': args.arm, 'gpu': args.gpu, 'rc': rc,
    'problem': args.problem, 'strategy': args.strategy, 'bound': args.bound,
    'seeds': args.seeds,
    'wall_s': round(t_end - t_start, 1),
    'n_dispatch': len(dispatches),
    'dispatch_temps': [],
    'samples': samples,
    'events': events,
}
for i, d in enumerate(dispatches):
    s = temp_at(d['t'])
    res['dispatch_temps'].append({
        'job_index': i, 't_rel': round(d['t'] - t_start, 1),
        'temp': s and s['temp'], 'power': s and s['power'],
        'sm_clock': s and s['sm_clock'], 'line': d['line'],
    })

# The inter-job gap: between the end of job 1 and the dispatch of job 2.
if len(dispatches) >= 2:
    t0, t1 = dispatches[0]['t'], dispatches[1]['t']
    finishes = [e for e in events
                if e['kind'] == 'finish' and t0 < e['t'] < t1]
    gap_start = finishes[0]['t'] if finishes else None
    if gap_start:
        gap = [s for s in samples if gap_start <= s['t'] <= t1 and 'temp' in s]
        res['gap'] = {
            'start_rel': round(gap_start - t_start, 1),
            'end_rel': round(t1 - t_start, 1),
            'duration_s': round(t1 - gap_start, 1),
            'min_temp': min((s['temp'] for s in gap), default=None),
            'max_temp': max((s['temp'] for s in gap), default=None),
            'temp_at_job2_dispatch': temp_at(t1)['temp'],
            'mean_power': (round(sum(s['power'] for s in gap) / len(gap), 1)
                           if gap else None),
            'n_samples': len(gap),
        }

with open(args.out, 'w') as fh:
    json.dump(res, fh, indent=1)

print('\n===BALLAST-THERMAL===' + json.dumps(
    {k: v for k, v in res.items() if k not in ('samples', 'events')}))
print('\nwrote %s' % args.out)
