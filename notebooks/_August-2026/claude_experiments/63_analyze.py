#!/usr/bin/env python
"""Doc 63 analysis: paired per-cluster local error, PER FAMILY, never pooled.

Pairing unit is (problem, seed, bucket). The response is |signed_local_error|,
the cluster's own contribution error to log Z in dex (doc 44/50's metric).

gain(A vs B) = median over paired clusters of [ log10|err_B| - log10|err_A| ]
so a POSITIVE gain means arm A is more accurate than arm B, in dex -- the same
sign convention as doc 31's "+0.515 dex".

Doc 31 §4: pooling across families hid an entire negative result, so every table
here is per problem, and the family roll-ups are labelled as such.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

FAMILY = {
    'grid10x10.f10.wrap': 'grid', 'grid20x20.f10': 'grid',
    'grid20x20.f10.wrap': 'grid', 'grid10x10.f10': 'grid',
    'rbm_20': 'rbm', 'rbm_21': 'rbm', 'rbm_22': 'rbm',
}
ARMS = ['base', 'residual', 'input', 'parts']


def sign_test(diffs):
    """Two-sided exact sign test. diffs > 0 == A better (convention below)."""
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    n = wins + losses
    if n == 0:
        return wins, losses, 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return wins, losses, min(1.0, p)


def wilcoxon(diffs):
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n < 6:
        return float('nan')
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w = sum(ranks[i] for i in range(n) if d[i] > 0)
    mu = n * (n + 1) / 4.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sd == 0:
        return float('nan')
    z = (w - mu + 0.5) / sd if w < mu else (w - mu - 0.5) / sd
    return math.erfc(abs(z) / math.sqrt(2))


def load(outdirs):
    """-> errs[(problem, arm, seed, bucket)] = |signed_local_error|, plus meta."""
    errs = {}
    meta = defaultdict(dict)
    for d in outdirs:
        for f in sorted(Path(d).glob('*.json')):
            r = json.loads(f.read_text())
            prob = r['problem'].split('/')[-1]
            arm, seed = r['arm'], r['seed']
            meta[(prob, arm, seed)] = {
                'wall': r.get('wall_seconds'), 'log_z': r.get('log_z'),
                'ref': r.get('ref_logZ'), 'n_clusters': len(r.get('local_errors', [])),
                'wmb': r.get('wmb_base_stats', []),
            }
            for rec in r.get('local_errors', []):
                e = rec.get('signed_local_error')
                if e is None:
                    continue
                errs[(prob, arm, seed, rec['bucket'])] = abs(float(e))
    return errs, meta


def compare(errs, prob, a, b, eps=1e-12):
    """gain(a vs b) in dex, positive = a more accurate."""
    keys = [k for k in errs if k[0] == prob and k[1] == a]
    diffs, na, nb = [], [], []
    for (p, _, s, bk) in keys:
        kb = (p, b, s, bk)
        if kb not in errs:
            continue
        ea, eb = errs[(p, a, s, bk)], errs[kb]
        diffs.append(math.log10(max(eb, eps)) - math.log10(max(ea, eps)))
        na.append(ea)
        nb.append(eb)
    if not diffs:
        return None
    w, l, p_sign = sign_test(diffs)
    return {
        'n': len(diffs), 'wins': w, 'losses': l, 'p_sign': p_sign,
        'p_wilcoxon': wilcoxon(diffs), 'gain_dex': float(np.median(diffs)),
        'med_abs_a': float(np.median(na)), 'med_abs_b': float(np.median(nb)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outdirs', nargs='+')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    errs, meta = load(args.outdirs)
    probs = sorted({k[0] for k in errs}, key=lambda p: (FAMILY.get(p, 'zz'), p))
    lines = []

    def emit(s=''):
        lines.append(s)
        print(s)

    emit('=' * 78)
    emit('DOC 63 -- WMB as INPUT vs residual learning vs baseline')
    emit('gain > 0 means the FIRST arm is more accurate, in dex of |local error|')
    emit('=' * 78)

    emit('\n## Runs')
    emit(f"{'problem':22s} {'arm':9s} {'seeds':7s} {'clusters':9s} {'wall(s) gpu1':>13s}")
    for p in probs:
        for a in ARMS:
            ss = sorted(s for (pp, aa, s) in meta if pp == p and aa == a)
            if not ss:
                continue
            nc = meta[(p, a, ss[0])]['n_clusters']
            wl = np.mean([meta[(p, a, s)]['wall'] for s in ss])
            emit(f"{p:22s} {a:9s} {str(ss):7s} {nc:<9d} {wl:13.0f}")

    pairs = [('input', 'residual'), ('parts', 'residual'),
             ('input', 'base'), ('parts', 'base'), ('residual', 'base'),
             ('parts', 'input')]

    for a, b in pairs:
        emit(f'\n## {a} vs {b}   (positive = {a} better)')
        emit(f"{'problem':22s} {'fam':5s} {'n':>4s} {'win/loss':>9s} {'gain dex':>9s} "
             f"{'p_sign':>9s} {'p_wilc':>9s} {'med|e| ' + a:>14s} {'med|e| ' + b:>14s}")
        for p in probs:
            r = compare(errs, p, a, b)
            if r is None:
                continue
            emit(f"{p:22s} {FAMILY.get(p, '?'):5s} {r['n']:4d} "
                 f"{str(r['wins']) + '/' + str(r['losses']):>9s} {r['gain_dex']:9.3f} "
                 f"{r['p_sign']:9.2e} {r['p_wilcoxon']:9.2e} "
                 f"{r['med_abs_a']:14.4g} {r['med_abs_b']:14.4g}")
        # family roll-up, still NOT cross-family
        for fam in ('grid', 'rbm'):
            fp = [p for p in probs if FAMILY.get(p) == fam]
            if not fp:
                continue
            alld = []
            for p in fp:
                r = compare(errs, p, a, b)
                if r:
                    alld.append((p, r['gain_dex'], r['n']))
            if alld:
                med = float(np.median([g for _, g, _ in alld]))
                emit(f"  -> {fam.upper()} family: per-cell gains "
                     f"{[f'{g:+.3f}' for _, g, _ in alld]}  median {med:+.3f} dex")

    emit('\n## WMB feature structure (input arms)')
    emit(f"{'problem':22s} {'arm':9s} {'k partitions (median)':>22s} {'features (median)':>18s}")
    for p in probs:
        for a in ('input', 'parts', 'residual'):
            ws = []
            for (pp, aa, s), m in meta.items():
                if pp == p and aa == a:
                    ws += m['wmb']
            if not ws:
                continue
            kk = [w['num_base_factors'] for w in ws]
            ff = [w.get('n_wmb_features', 0) for w in ws]
            emit(f"{p:22s} {a:9s} {np.median(kk):22.1f} {np.median(ff):18.1f}")

    txt = '\n'.join(lines)
    if args.out:
        Path(args.out).write_text(txt + '\n')
        print(f"\n[wrote {args.out}]")


if __name__ == '__main__':
    main()
