#!/usr/bin/env python
"""Doc 50: does the benefit track the memorized MASS fraction?

Doc 44 s 4.2 stratified 44 cluster-runs by mass fraction and found the error
reduction jumps only above ~50% mass. This repeats that stratification on the
much larger sweep here -- and, unlike doc 44, keeps the two problems SEPARATE.

usage: python 50_mass.py <outdir> [<outdir> ...]
"""
import json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def main(paths):
    with open(os.path.join(HERE, '50-sweep.json')) as f:
        sweep = json.load(f)
    for cell, d in sweep.items():
        print('=' * 84)
        print('CELL', cell, '' if len(sys.argv) < 2
              else f'   [only K/2^w={sys.argv[1]} pool={sys.argv[2]}]')
        want = None
        if len(sys.argv) > 1:
            want = (float(sys.argv[1]), float(sys.argv[2]))
        buckets = defaultdict(list)
        for pt in d['points']:
            if want and (abs(pt['mem_frac'] - want[0]) > 1e-9
                         or abs(pt['sample_frac'] - want[1]) > 1e-9):
                continue
            for pc in pt['per_cluster']:
                if pc['mass_frac'] is None:
                    continue
                m = pc['mass_frac']
                key = ('>=50%' if m >= 0.5 else
                       '20-50%' if m >= 0.2 else
                       '5-20%' if m >= 0.05 else '<5%')
                buckets[key].append(pc)
        print(f'  {"mass band":<10} {"n":>5} {"wins":>5} '
              f'{"med|err| base":>14} {"med|err| hyb":>13} {"med rel. red.":>14}')
        for key in ['>=50%', '20-50%', '5-20%', '<5%']:
            rows = buckets.get(key, [])
            if not rows:
                continue
            n = len(rows)
            wins = sum(1 for r in rows if r['d_abs'] < 0)
            b = sorted(abs(r['err_base']) for r in rows)
            h = sorted(abs(r['err_mem']) for r in rows)
            rel = sorted((abs(r['err_base']) - abs(r['err_mem']))
                         / abs(r['err_base'])
                         for r in rows if abs(r['err_base']) > 0)
            print(f'  {key:<10} {n:>5} {wins:>5} {b[n//2]:>14.5g} '
                  f'{h[n//2]:>13.5g} {rel[len(rel)//2]*100:>13.1f}%')
        # correlation between mass fraction and relative reduction
        pts = [(r['mass_frac'],
                (abs(r['err_base']) - abs(r['err_mem'])) / abs(r['err_base']))
               for rows in buckets.values() for r in rows
               if abs(r['err_base']) > 0]
        n = len(pts)
        rx = sorted(range(n), key=lambda i: pts[i][0])
        ry = sorted(range(n), key=lambda i: pts[i][1])
        ranki = [0] * n
        rankj = [0] * n
        for k, i in enumerate(rx):
            ranki[i] = k
        for k, i in enumerate(ry):
            rankj[i] = k
        mx = sum(ranki) / n
        my = sum(rankj) / n
        num = sum((ranki[i] - mx) * (rankj[i] - my) for i in range(n))
        den = (sum((ranki[i] - mx) ** 2 for i in range(n)) *
               sum((rankj[i] - my) ** 2 for i in range(n))) ** 0.5
        print(f'  Spearman rho(mass fraction, relative error reduction) = '
              f'{num/den:+.3f}  over n={n} cluster-runs')


if __name__ == '__main__':
    main(sys.argv[1:])
