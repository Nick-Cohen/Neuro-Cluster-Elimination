#!/usr/bin/env python
"""Docs 63/64 -- PRIMARY endpoint: end-to-end |log Z - reference|.

Why this file exists (and why it supersedes the pre-registered primary):
--------------------------------------------------------------------------
Both docs pre-registered *paired per-cluster local error* as primary. That endpoint is
NOT trustworthy for the memorization experiment (doc 64) and demonstrably misleads:
it reported fw_bw beating fw_true by +8.388 dex, 50/0, p=1.8e-15, with median |error|
of EXACTLY ZERO on two cells -- while end-to-end log Z had fw_true nominally BETTER on
both grids with per-seed ranges overlapping. The cause is structural: memorized entries
are exact BY CONSTRUCTION, so local error collapses toward zero on precisely the entries
the arm memorized, whether or not the final answer improves. On grid20 it got worse.

So, per Nick:
  1. PRIMARY = end-to-end |log Z - ref|, per problem, mean over seeds, WITH the per-seed
     spread printed so overlap is visible.
  2. Per-cluster local error is SECONDARY, and disagreements are called out explicitly.
  3. Significance is judged by whether PER-SEED RANGES SEPARATE, not by a p-value on
     cluster-level pairs -- cluster pairs are not independent within a run, which is
     exactly what manufactured p=1.8e-15 on a difference end-to-end does not support.
  4. References come from results_for_writeup/problem_overview_table.csv, column
     ref_log10Z, ONLY where ref_kind == 'exact' (17 problems). rbm_ferro_20 has no
     exact reference and is reported separately as reference-free.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

REF_CSV = ('/home/cohenn1/NCE/notebooks/June-2026/claude_experiments/'
           'reduce_nn_experiment/results_for_writeup/problem_overview_table.csv')

FAMILY_OF = {'pedigree': 'pedigree', 'grid': 'grid', 'rbm': 'rbm'}
ARMS = {63: ['base', 'residual', 'input', 'parts'], 64: ['base', 'fw_true', 'fw_bw']}


def cell_sign_p(deltas):
    """Two-sided exact sign test over CELLS.

    This is the one sign test that is legitimate here: distinct benchmark problems
    are independent units. The cluster-level sign/Wilcoxon tests in the old analysis
    are not -- clusters within a run share the same elimination and the same trained
    upstream messages, which is how doc 64 produced p=1.8e-15 for a difference the
    end-to-end answer does not support.
    """
    w = sum(1 for d in deltas if d > 0)
    l = sum(1 for d in deltas if d < 0)
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2)


def norm_name(problem):
    """'grids/grid10x10.f10.wrap' -> 'grid10x10f10wrap';  'dbn/rbm_20' -> 'rbm_20'."""
    return problem.split('/')[-1].replace('.', '')


def family(problem):
    b = problem.split('/')[-1]
    if b.startswith('pedigree'):
        return 'pedigree'
    if b.startswith('grid'):
        return 'grid'
    return 'rbm'


def load_refs():
    """name -> ref_log10Z, EXACT references only."""
    refs = {}
    for r in csv.DictReader(open(REF_CSV)):
        if r['ref_kind'].strip() == 'exact':
            refs[r['problem'].strip()] = float(r['ref_log10Z'])
    return refs


def load_runs(roots):
    runs = []
    for root in roots:
        for f in Path(root).rglob('*.json'):
            try:
                runs.append(json.loads(f.read_text()))
            except Exception:
                pass
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', type=int, required=True, choices=[63, 64])
    ap.add_argument('roots', nargs='+')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    refs = load_refs()
    runs = [r for r in load_runs(args.roots) if r['arm'] in ARMS[args.exp]]

    lines = []

    def emit(s=''):
        lines.append(s)
        print(s)

    # cell = (problem, iB): rbm_20/21/22 exist at BOTH i-bounds in the benchmark set
    # and share a basename, so keying on the problem alone would silently merge them.
    e2e = defaultdict(dict)      # (prob, iB) -> arm -> {seed: |logZ - ref|}
    lerr = {}                    # (prob, iB, arm, seed, bucket) -> |local err|
    noref = set()
    for r in runs:
        prob, ib = r['problem'], int(r['iB'])
        nm = norm_name(prob)
        cell = (prob, ib)
        if nm in refs:
            err = abs(float(r['log_z']) - refs[nm])
            e2e[cell].setdefault(r['arm'], {})[int(r['seed'])] = err
        else:
            noref.add(cell)
        for rec in r.get('local_errors', []):
            e = rec.get('signed_local_error')
            if e is not None:
                lerr[(prob, ib, r['arm'], int(r['seed']), rec['bucket'])] = abs(float(e))

    cells = sorted(set(e2e) | noref, key=lambda c: (family(c[0]), c[0], c[1]))
    arms = ARMS[args.exp]

    emit('=' * 100)
    emit(f'DOC {args.exp} -- PRIMARY ENDPOINT: end-to-end |log Z - reference|  (lower = better)')
    emit('Reference: problem_overview_table.csv ref_log10Z, ref_kind==exact only.')
    emit('Per-seed spread is printed because overlap is what decides significance here.')
    emit('=' * 100)

    emit('\n## Per-cell end-to-end error, mean over seeds [min .. max across seeds]')
    hdr = f"{'problem':24s} {'iB':>3s} {'fam':>8s} " + ' '.join(f'{a:>28s}' for a in arms)
    emit(hdr)
    for cell in cells:
        if cell in noref:
            continue
        prob, ib = cell
        row = f"{prob.split('/')[-1]:24s} {ib:3d} {family(prob):>8s} "
        for a in arms:
            d = e2e[cell].get(a, {})
            if not d:
                row += f"{'-':>28s} "
                continue
            v = [d[s] for s in sorted(d)]
            row += f"{np.mean(v):10.4f} [{min(v):7.4f}..{max(v):7.4f}] "
        emit(row)

    # ---------------- pairwise, judged by range separation ----------------
    def pairs(exp):
        if exp == 63:
            return [('input', 'residual'), ('parts', 'residual'), ('input', 'base'),
                    ('residual', 'base'), ('parts', 'input')]
        return [('fw_bw', 'fw_true'), ('fw_true', 'base'), ('fw_bw', 'base')]

    for a, b in pairs(args.exp):
        emit(f'\n## END-TO-END {a} vs {b}   (delta = mean|err_b| - mean|err_a|; '
             f'positive = {a} closer to reference)')
        emit(f"{'problem':24s} {'iB':>3s} {'fam':>8s} {'mean|e| ' + a:>13s} "
             f"{'mean|e| ' + b:>13s} {'delta':>9s} {'seeds':>6s}  verdict")
        fam_roll = defaultdict(list)
        for cell in cells:
            if cell in noref:
                continue
            da, db = e2e[cell].get(a, {}), e2e[cell].get(b, {})
            common = sorted(set(da) & set(db))
            if not common:
                continue
            va, vb = [da[s] for s in common], [db[s] for s in common]
            delta = float(np.mean(vb) - np.mean(va))
            # significance := do the per-seed RANGES separate?
            if max(va) < min(vb):
                verdict = f'SEPARATE: {a} better'
            elif max(vb) < min(va):
                verdict = f'SEPARATE: {b} better'
            else:
                verdict = 'overlap (within noise)'
            fam_roll[family(cell[0])].append(delta)
            emit(f"{cell[0].split('/')[-1]:24s} {cell[1]:3d} {family(cell[0]):>8s} "
                 f"{np.mean(va):13.4f} {np.mean(vb):13.4f} {delta:9.4f} "
                 f"{len(common):6d}  {verdict}")
        allcells = []
        for fam in ('pedigree', 'grid', 'rbm'):
            if fam_roll.get(fam):
                d = fam_roll[fam]
                allcells += d
                emit(f"  -> {fam.upper():9s} n_cells={len(d):2d}  median delta "
                     f"{float(np.median(d)):+.4f}  cells favouring {a}: "
                     f"{sum(1 for x in d if x > 0)}/{len(d)}"
                     f"   sign p={cell_sign_p(d):.3f}")
        if allcells:
            emit(f"  => ALL FAMILIES n_cells={len(allcells):2d}  median delta "
                 f"{float(np.median(allcells)):+.4f}  cells favouring {a}: "
                 f"{sum(1 for x in allcells if x > 0)}/{len(allcells)}"
                 f"   sign p={cell_sign_p(allcells):.3f}")

    # ---------------- secondary: per-cluster local error ----------------
    emit('\n\n' + '=' * 100)
    emit('SECONDARY ENDPOINT: paired per-cluster |local error|  (the OLD primary)')
    emit('Reported for continuity. Where it disagrees with end-to-end above, the')
    emit('end-to-end column wins -- see the header note on why this metric misleads.')
    emit('=' * 100)

    def local_gain(cell, a, b, eps=1e-12):
        prob, ib = cell
        diffs = []
        for k in lerr:
            if k[0] != prob or k[1] != ib or k[2] != a:
                continue
            kb = (prob, ib, b, k[3], k[4])
            if kb in lerr:
                diffs.append(math.log10(max(lerr[kb], eps))
                             - math.log10(max(lerr[k], eps)))
        if not diffs:
            return None
        w = sum(1 for d in diffs if d > 0)
        l = sum(1 for d in diffs if d < 0)
        return {'n': len(diffs), 'w': w, 'l': l, 'gain': float(np.median(diffs))}

    for a, b in pairs(args.exp):
        emit(f'\n## LOCAL {a} vs {b}   (positive gain dex = {a} more accurate locally)')
        emit(f"{'problem':24s} {'iB':>3s} {'n':>5s} {'win/loss':>10s} {'gain dex':>9s}"
             f"   {'agrees with end-to-end?':>24s}")
        for cell in cells:
            r = local_gain(cell, a, b)
            if not r:
                continue
            da, db = e2e[cell].get(a, {}), e2e[cell].get(b, {})
            common = sorted(set(da) & set(db))
            agree = 'no end-to-end ref'
            if common:
                delta = float(np.mean([db[s] for s in common])
                              - np.mean([da[s] for s in common]))
                if abs(delta) < 1e-9 or abs(r['gain']) < 1e-9:
                    agree = 'tie'
                else:
                    agree = 'AGREE' if (delta > 0) == (r['gain'] > 0) else '*** DISAGREE ***'
            emit(f"{cell[0].split('/')[-1]:24s} {cell[1]:3d} {r['n']:5d} "
                 f"{str(r['w']) + '/' + str(r['l']):>10s} {r['gain']:9.3f}   {agree:>24s}")

    if noref:
        emit('\n\n## Reference-free cells (no exact reference in the overview table)')
        emit('Reported separately: no |log Z - ref| is computable, so only the arms\'')
        emit('log Z values and their spread are shown.')
        emit(f"{'problem':24s} {'iB':>3s} {'arm':>10s} {'mean logZ':>13s} {'min..max':>26s}")
        for cell in sorted(noref):
            for a in arms:
                v = [float(r['log_z']) for r in runs
                     if r['problem'] == cell[0] and int(r['iB']) == cell[1] and r['arm'] == a]
                if v:
                    emit(f"{cell[0].split('/')[-1]:24s} {cell[1]:3d} {a:>10s} "
                         f"{np.mean(v):13.4f} {'[%.4f..%.4f]' % (min(v), max(v)):>26s}")

    # coverage
    emit('\n\n## Coverage')
    emit(f"{'problem':24s} {'iB':>3s} " + ' '.join(f'{a:>10s}' for a in arms))
    for cell in cells:
        counts = []
        for a in arms:
            n = len({int(r['seed']) for r in runs if r['problem'] == cell[0]
                     and int(r['iB']) == cell[1] and r['arm'] == a})
            counts.append(f'{n:>10d}')
        emit(f"{cell[0].split('/')[-1]:24s} {cell[1]:3d} " + ' '.join(counts))

    txt = '\n'.join(lines)
    if args.out:
        Path(args.out).write_text(txt + '\n')
        print(f'\n[wrote {args.out}]')


if __name__ == '__main__':
    main()
