#!/usr/bin/env python
"""Doc 64 analysis: fw_true vs fw_bw memorization selection, paired per cluster.

Leads with the CONSTRAINT AUDIT, because if selection changed the NN training
target or the preprocessor's normalisation the accuracy comparison is worthless
and nothing below it should be read.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ARMS = ['base', 'fw_true', 'fw_bw']


def sign_test(diffs):
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    n = wins + losses
    if n == 0:
        return wins, losses, 1.0
    k = min(wins, losses)
    return wins, losses, min(1.0, sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outdirs', nargs='+')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    runs = []
    for d in args.outdirs:
        for f in sorted(Path(d).glob('*.json')):
            runs.append(json.loads(f.read_text()))

    errs, meta = {}, {}
    for r in runs:
        p = r['problem'].split('/')[-1]
        key = (p, r['arm'], r['seed'])
        meta[key] = r
        for rec in r.get('local_errors', []):
            e = rec.get('signed_local_error')
            if e is not None:
                errs[(p, r['arm'], r['seed'], rec['bucket'])] = abs(float(e))

    probs = sorted({k[0] for k in errs})
    lines = []

    def emit(s=''):
        lines.append(s)
        print(s)

    emit('=' * 78)
    emit('DOC 64 -- memorization selection: fw_true vs fw_bw')
    emit('=' * 78)

    # ---------------- constraint audit first ----------------
    emit('\n## CONSTRAINT AUDIT (read this before any accuracy number)')
    emit('The claim: selection changes ONLY which entries are memorized -- not the NN')
    emit('training target, not the preprocessor normalisation, and use_bw_approx stays off.')
    bad = []
    for k, r in sorted(meta.items()):
        aud = r.get('preproc_audit', [])
        unchanged = all(a.get('preproc_unchanged') for a in aud) if aud else None
        bw_set = any(a.get('dataloader_bw_factors_set') or a.get('dataloader_bw_modifier_set')
                     for a in aud)
        if r.get('use_bw_approx') or bw_set or unchanged is not True:
            bad.append(k)
        emit(f"  {k[0]:22s} {k[1]:8s} s{k[2]}  use_bw_approx={r.get('use_bw_approx')}  "
             f"dataloader_bw_ever_set={bw_set}  preproc_unchanged_all={unchanged}  "
             f"n_clusters={len(aud)}")
    emit(f"  => {'PASS' if not bad else 'FAIL: ' + str(bad)}")

    emit('\n## Do fw_true and fw_bw train against the SAME target, cluster by cluster?')
    emit('Identical where the upstream context is identical; divergence downstream is the')
    emit('legitimate consequence of the arms memorizing different entries, not a leak.')
    for p in probs:
        for s in sorted({k[2] for k in meta if k[0] == p}):
            a, b = meta.get((p, 'fw_true', s)), meta.get((p, 'fw_bw', s))
            if not a or not b:
                continue
            pa = {x['bucket']: x['preproc_before_memorization'] for x in a['preproc_audit']}
            pb = {x['bucket']: x['preproc_before_memorization'] for x in b['preproc_audit']}
            common = sorted(set(pa) & set(pb))
            same = [k for k in common if pa[k] == pb[k]]
            # elimination order == order of the audit records
            order = [x['bucket'] for x in a['preproc_audit']]
            first_diff = next((i for i, k in enumerate(order)
                               if k in pb and pa[k] != pb[k]), None)
            emit(f"  {p:22s} s{s}: identical normaliser on {len(same)}/{len(common)} clusters; "
                 f"first divergence at elimination position "
                 f"{first_diff if first_diff is not None else 'none'}")

    emit('\n## fw_bw effectiveness (a degraded arm looks exactly like a null result)')
    for k, r in sorted(meta.items()):
        if k[1] != 'fw_bw':
            continue
        ml = [m for m in r.get('memorization_log', []) if m.get('ok')]
        eff = [m for m in ml if m.get('selection_effective') == 'fw_bw']
        srcs = {m.get('bw_source') for m in eff}
        ov = [m.get('topk_overlap_frac') for m in eff if m.get('topk_overlap_frac') is not None]
        fails = [m for m in r.get('memorization_log', []) if not m.get('ok')]
        emit(f"  {k[0]:22s} s{k[2]}: fw_bw effective {len(eff)}/{len(ml)}  src={srcs}  "
             f"median top-K overlap with fw_true = {np.median(ov):.3f}  "
             f"table build failures={len(fails)}")

    # ---------------- accuracy ----------------
    def compare(prob, a, b, eps=1e-12):
        diffs, va, vb = [], [], []
        for (p, arm, s, bk) in list(errs):
            if p != prob or arm != a:
                continue
            kb = (p, b, s, bk)
            if kb not in errs:
                continue
            ea, eb = errs[(p, a, s, bk)], errs[kb]
            diffs.append(math.log10(max(eb, eps)) - math.log10(max(ea, eps)))
            va.append(ea)
            vb.append(eb)
        if not diffs:
            return None
        w, l, ps = sign_test(diffs)
        return {'n': len(diffs), 'w': w, 'l': l, 'p': ps, 'pw': wilcoxon(diffs),
                'gain': float(np.median(diffs)),
                'ma': float(np.median(va)), 'mb': float(np.median(vb))}

    for a, b in [('fw_bw', 'fw_true'), ('fw_true', 'base'), ('fw_bw', 'base')]:
        emit(f'\n## {a} vs {b}   (positive gain = {a} more accurate, dex)')
        emit(f"{'problem':22s} {'n':>4s} {'win/loss':>9s} {'gain dex':>9s} {'p_sign':>9s} "
             f"{'p_wilc':>9s} {'med|e| ' + a:>14s} {'med|e| ' + b:>14s}")
        for p in probs:
            r = compare(p, a, b)
            if r:
                emit(f"{p:22s} {r['n']:4d} {str(r['w']) + '/' + str(r['l']):>9s} "
                     f"{r['gain']:9.3f} {r['p']:9.2e} {r['pw']:9.2e} "
                     f"{r['ma']:14.4g} {r['mb']:14.4g}")

    emit('\n## Cost (gpu1 -- INDICATIVE ONLY, thermally throttled)')
    emit(f"{'problem':22s} {'arm':9s} {'wall(s) mean':>13s} {'x base':>8s}")
    for p in probs:
        bw = [meta[k]['wall_seconds'] for k in meta if k[0] == p and k[1] == 'base']
        for a in ARMS:
            ws = [meta[k]['wall_seconds'] for k in meta if k[0] == p and k[1] == a]
            if ws:
                emit(f"{p:22s} {a:9s} {np.mean(ws):13.0f} "
                     f"{(np.mean(ws) / np.mean(bw)) if bw else float('nan'):8.3f}")

    txt = '\n'.join(lines)
    if args.out:
        Path(args.out).write_text(txt + '\n')
        print(f"\n[wrote {args.out}]")


if __name__ == '__main__':
    main()
