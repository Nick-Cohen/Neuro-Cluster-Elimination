#!/usr/bin/env python
"""Doc 50: paired comparison between two treatment arms (not against baseline).

Used for the fixed-K / different-pool question: at K/2^w = 1%, does a 10:1
candidate pool beat a 2:1 pool? Pairs by (seed, cluster) exactly as the
baseline comparison does.

usage: python 50_pairwise.py <outdir> <memfracA> <poolA> <memfracB> <poolB>
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '50_analyze.py')).read().split('def main(')[0])


def arm(runs, mf, pool):
    out = {}
    for r in runs:
        if r['arm'] != 'mem':
            continue
        if (abs(float(r['mem_frac']) - mf) < 1e-9
                and abs(float(r['sample_frac']) - pool) < 1e-9):
            out[r['seed']] = r
    return out


def main():
    outdir, mfa, pa, mfb, pb = (sys.argv[1], float(sys.argv[2]),
                                float(sys.argv[3]), float(sys.argv[4]),
                                float(sys.argv[5]))
    runs = load([outdir])
    A, B = arm(runs, mfa, pa), arm(runs, mfb, pb)
    diffs, rows = [], []
    for s in sorted(set(A) & set(B)):
        ea = {e['bucket']: e.get('signed_local_error')
              for e in A[s]['local_errors']}
        eb = {e['bucket']: e.get('signed_local_error')
              for e in B[s]['local_errors']}
        ma = {e['bucket']: e.get('memorized_mass_fraction')
              for e in A[s]['local_errors']}
        mb = {e['bucket']: e.get('memorized_mass_fraction')
              for e in B[s]['local_errors']}
        for k in sorted(set(ea) & set(eb)):
            if ea[k] is None or eb[k] is None:
                continue
            diffs.append(abs(eb[k]) - abs(ea[k]))   # <0 => B better
            rows.append((s, k, ma.get(k), mb.get(k), abs(ea[k]), abs(eb[k])))
    w, l, p = sign_test(diffs)
    _, pw = wilcoxon(diffs)
    ma_ = med([r[2] for r in rows if r[2] is not None])
    mb_ = med([r[3] for r in rows if r[3] is not None])
    print(f'A = K/2^w={mfa} pool={pa}   B = K/2^w={mfb} pool={pb}')
    print(f'  seeds paired: {sorted(set(A) & set(B))}   n = {len(diffs)}')
    print(f'  B better on {w}, worse on {l}   sign p={p:.4g}  '
          f'Wilcoxon p={pw:.4g}')
    print(f'  median |err|  A={med([r[4] for r in rows]):.6g}   '
          f'B={med([r[5] for r in rows]):.6g}')
    print(f'  median mass   A={ma_:.4g}   B={mb_:.4g}')


if __name__ == '__main__':
    main()
