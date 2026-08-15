"""Phase-1 rerun status: queue progress, throughput, and accuracy so far.

    NCE_MODEL_CACHE=/home/cohenn1/NCE/.model_cache \
    python notebooks/_August-2026/claude_experiments/61_status.py \
        [--root /home/cohenn1/NCE-rerun/phase1] [--csv out.csv]

Reads only files on disk, so it is safe to run while the sweep is live.

Timings come from `manifest.timing.t_elim_s` / `total_wall_s`, NEVER from the
scheduler's `[done] ... after Xs` console line, which is quantised to the poll
interval and overstates job duration (doc 60 section 6.3).

`abs_err` is |log10 Z - reference|, with the reference taken from
`exact_logZ.json` where an exact solve exists and otherwise from
`benchmark_set.json`'s `ref_logZ` -- the same convention as the original study.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter

STUDY = '/home/cohenn1/NCE/notebooks/June-2026/claude_experiments/reduce_nn_experiment'


def refs():
    out = {}
    with open(os.path.join(STUDY, 'benchmark_set.json')) as fh:
        for g in json.load(fh)['groups']:
            for p in g['problems']:
                out.setdefault(p['key'], (p['ref_logZ'], 'benchmark_set'))
    path = os.path.join(STUDY, 'exact_logZ.json')
    if os.path.exists(path):
        with open(path) as fh:
            for k, v in json.load(fh).items():
                if v.get('exact_log10Z') is not None:
                    out[k] = (v['exact_log10Z'], 'exact_solve')
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/home/cohenn1/NCE-rerun/phase1')
    p.add_argument('--csv', default=None)
    p.add_argument('--worst', type=int, default=10)
    a = p.parse_args()

    with open(os.path.join(a.root, 'queue.json')) as fh:
        q = json.load(fh)['jobs']
    print('queue: %d jobs  %s'
          % (len(q), dict(Counter(r['status'] for r in q.values()))))

    R = refs()
    rows, wall = [], 0.0
    runs = os.path.join(a.root, 'runs')
    for name in sorted(os.listdir(runs)) if os.path.isdir(runs) else []:
        mpath = os.path.join(runs, name, 'manifest.json')
        if not os.path.exists(mpath):
            continue
        try:
            m = json.load(open(mpath))
        except (ValueError, OSError):
            continue
        if m.get('status') != 'done':
            continue
        spec = m['job']
        lz = m['result'].get('log_z_repr')
        if lz is None:
            continue
        lz = float(lz)
        ref, src = R.get(spec['problem_key'], (None, ''))
        t = m.get('timing', {})
        wall += t.get('total_wall_s', 0.0) or 0.0
        rows.append({
            'problem': spec['problem_key'],
            'iB': spec['extra_config'].get('iB'),
            'strategy': spec['merge_strategy'],
            'bound': spec['merge_bound'], 'seed': spec['seed'],
            'log_z': lz, 'ref': ref, 'ref_src': src,
            'abs_err': None if ref is None else abs(lz - ref),
            'num_trained': m['result'].get('num_trained'),
            't_elim_s': t.get('t_elim_s'), 'total_wall_s': t.get('total_wall_s'),
            'gpu': m.get('hardware', {}).get('gpu_physical_index'),
            'commit': m.get('git', {}).get('commit', '')[:8],
            'dirty': m.get('git', {}).get('dirty'),
            'replayed': m['result'].get('clusters_replayed'),
        })

    print('completed: %d   GPU-hours burned: %.1f' % (len(rows), wall / 3600))
    if not rows:
        return
    print('cards used: %s   commits: %s   dirty: %s'
          % (dict(Counter(r['gpu'] for r in rows)),
             dict(Counter(r['commit'] for r in rows)),
             dict(Counter(r['dirty'] for r in rows))))
    finite = [r for r in rows if r['abs_err'] is not None]
    bad = [r for r in rows if r['log_z'] != r['log_z']]
    print('non-finite log Z: %d' % len(bad))
    if finite:
        errs = sorted(r['abs_err'] for r in finite)
        print('abs_err: median %.4g  p90 %.4g  max %.4g'
              % (errs[len(errs) // 2], errs[int(0.9 * (len(errs) - 1))], errs[-1]))
        print('worst %d:' % a.worst)
        for r in sorted(finite, key=lambda r: -r['abs_err'])[:a.worst]:
            print('   %-26s %-16s s%d  logZ=%-14.4f ref=%-12.4f err=%.4g (%s)'
                  % (r['problem'],
                     '%s%s' % (r['strategy'],
                               '' if r['bound'] is None else r['bound']),
                     r['seed'], r['log_z'], r['ref'], r['abs_err'], r['ref_src']))
    if a.csv:
        with open(a.csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print('-> %s' % a.csv)


if __name__ == '__main__':
    main()
