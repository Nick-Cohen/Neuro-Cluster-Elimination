#!/usr/bin/env python
"""60: does ballast actually HOLD a card's temperature across an idle gap?

60_ballast_thermal.py ran the real scheduler and showed the integration works
(ballast starts when the queue drains, yields in 0.00 s before a dispatch), but
it could not answer the thermal question: its workload (grid10x10.f10 reduce_nn)
draws 62 W at 6% utilisation and never takes the card above 43 C, so there was
no heat to preserve. Doc 59's effect was measured on a card at 82 C.

This measures the mechanism directly, which is what the requirement asks for:
a warm card, an idle gap of the length a scheduler actually produces, and the
temperature at the far end of that gap -- with and without ballast.

  phase 1  WARM    ballast on, until equilibrium (doc 59: gpu0 reaches it in
                   330 s, gpu3 in 180 s; 360 s is doc 59's recommended warm-up).
  phase 2  GAP     `--arm noballast`: ballast stopped, card idle.
                   `--arm ballast`  : ballast left running.
  phase 3  REPORT  temperature at the end of the gap = the temperature a job
                   dispatched at that instant would start from.

Usage:
  python 60_ballast_hold.py --gpu 0 --warm 360 --gap 120 --out hold.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--gpu', type=int, required=True)
ap.add_argument('--warm', type=float, default=360.0)
ap.add_argument('--gap', type=float, default=120.0)
ap.add_argument('--repo', default=os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
ap.add_argument('--out', required=True)
args = ap.parse_args()

if args.gpu == 2:
    raise SystemExit('cuda:2 is RETIRED -- refusing')

sys.path.insert(0, args.repo)
from nce.scheduler.ballast import BallastPool


def telem():
    out = subprocess.run(
        ['nvidia-smi',
         '--query-gpu=temperature.gpu,power.draw,utilization.gpu,clocks.sm',
         '--format=csv,noheader,nounits', '-i', str(args.gpu)],
        capture_output=True, text=True, check=True).stdout.strip()
    p = [x.strip() for x in out.split(',')]
    return {'temp': int(p[0]), 'power': float(p[1]), 'util': int(p[2]),
            'sm_clock': int(p[3])}


def run_arm(arm, pool):
    """Warm with ballast, then hold the gap per `arm`. Returns the trace."""
    trace = []
    t0 = time.time()
    print('[%s] warming %.0fs' % (arm, args.warm), flush=True)
    pool.start(args.gpu)
    while time.time() - t0 < args.warm:
        s = telem()
        s['t'] = round(time.time() - t0, 1)
        s['phase'] = 'warm'
        trace.append(s)
        time.sleep(2.0)
    warm_end = telem()
    print('[%s] equilibrium: %d C, %.0f W, %d MHz'
          % (arm, warm_end['temp'], warm_end['power'], warm_end['sm_clock']),
          flush=True)

    if arm == 'noballast':
        pool.stop(args.gpu)
    tg = time.time()
    while time.time() - tg < args.gap:
        s = telem()
        s['t'] = round(time.time() - t0, 1)
        s['phase'] = 'gap'
        trace.append(s)
        time.sleep(2.0)
    gap_end = telem()
    pool.stop(args.gpu)
    print('[%s] end of %.0fs gap: %d C, %.0f W'
          % (arm, args.gap, gap_end['temp'], gap_end['power']), flush=True)
    return trace, warm_end, gap_end


results = {'gpu': args.gpu, 'warm_s': args.warm, 'gap_s': args.gap, 'arms': {}}
pool = BallastPool(enabled=True, cwd=args.repo)
try:
    for arm in ('noballast', 'ballast'):
        trace, warm_end, gap_end = run_arm(arm, pool)
        gap = [s for s in trace if s['phase'] == 'gap']
        results['arms'][arm] = {
            'equilibrium_temp': warm_end['temp'],
            'equilibrium_power': warm_end['power'],
            'gap_end_temp': gap_end['temp'],
            'gap_min_temp': min(s['temp'] for s in gap),
            'gap_mean_power': round(sum(s['power'] for s in gap) / len(gap), 1),
            'drop_C': warm_end['temp'] - gap_end['temp'],
            'trace': trace,
        }
        # Let the card return to ambient before the next arm so both arms warm
        # from a comparable starting point.
        if arm == 'noballast':
            print('[cooldown] 120s before next arm', flush=True)
            time.sleep(120)
finally:
    pool.stop_all()

a = results['arms']
print('\n%-12s %-14s %-14s %-10s' % ('arm', 'equilibrium', 'end of gap', 'drop'))
for arm in ('noballast', 'ballast'):
    r = a[arm]
    print('%-12s %-14s %-14s %-10s'
          % (arm, '%d C' % r['equilibrium_temp'], '%d C' % r['gap_end_temp'],
             '%+d C' % -r['drop_C']))
results['verdict'] = ('HOLDS' if a['ballast']['drop_C'] < a['noballast']['drop_C']
                      else 'NO EFFECT')
print('\nVERDICT: %s' % results['verdict'])
print('===BALLAST-HOLD===' + json.dumps(
    {k: (v if k != 'arms' else {kk: {x: y for x, y in vv.items() if x != 'trace'}
                                for kk, vv in v.items()})
     for k, v in results.items()}))
with open(args.out, 'w') as fh:
    json.dump(results, fh, indent=1)
print('wrote %s' % args.out)
