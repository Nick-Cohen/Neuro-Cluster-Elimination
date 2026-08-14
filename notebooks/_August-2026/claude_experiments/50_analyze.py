#!/usr/bin/env python
"""Doc 50 analysis: memorization threshold sweep.

For each (problem, merge, D) cell and each sweep point (mem_frac), the paired
per-cluster |signed local error| against that cell's baseline, paired by
(seed, cluster). Sign test (exact binomial, two-sided) + Wilcoxon signed-rank,
matching 44_analyze.py so numbers are comparable.

Reports, per point: entry fraction K/2^w (realized), memorized MASS fraction
(realized), wall-time multiplier, and table-build seconds.

Problems are NEVER pooled: every cell is reported on its own.
"""
import glob, json, math, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def load(outdirs):
    runs = []
    for od in outdirs:
        for p in sorted(glob.glob(os.path.join(HERE, od, '*.json'))):
            with open(p) as f:
                runs.append(json.load(f))
    return runs


def sign_test(diffs):
    """Two-sided exact sign test. diffs > 0 == treatment worse."""
    wins = sum(1 for d in diffs if d < 0)
    losses = sum(1 for d in diffs if d > 0)
    n = wins + losses
    if n == 0:
        return wins, losses, 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return wins, losses, min(1.0, p)


def wilcoxon(diffs):
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n < 1:
        return None, 1.0
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
    w_pos = sum(r for r, x in zip(ranks, d) if x > 0)
    w_neg = sum(r for r, x in zip(ranks, d) if x < 0)
    w = min(w_pos, w_neg)
    mu = n * (n + 1) / 4.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sd == 0:
        return w, 1.0
    z = (w - mu + 0.5) / sd
    p = math.erfc(abs(z) / math.sqrt(2))
    return w, min(1.0, p)


def med(xs):
    xs = sorted(xs)
    if not xs:
        return float('nan')
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def main(outdirs):
    runs = load(outdirs)
    cells = defaultdict(lambda: defaultdict(dict))   # cell -> point -> seed -> run
    for r in runs:
        cell = (r['problem'], r['merge'], r['D'])
        # Key on (mem_frac, sample_frac): the anchor arm shares mem_frac=0.01
        # with a sweep point but uses a 10:1 pool instead of 2:1, so keying on
        # mem_frac alone would silently overwrite one with the other.
        point = ('base' if r['arm'] == 'base'
                 else (round(float(r['mem_frac']), 6),
                       round(float(r['sample_frac']), 6)))
        cells[cell][point][r['seed']] = r

    out = {}
    for cell, points in sorted(cells.items()):
        base = points.get('base', {})
        print('=' * 90)
        print(f'CELL {cell[0]}  merge={cell[1]} D={cell[2]}   '
              f'baseline seeds={sorted(base)}')
        if base:
            bw = [b['wall_time_s'] for b in base.values()]
            per = ', '.join('%s:%.0f' % (s, base[s]['wall_time_s'])
                            for s in sorted(base))
            print(f'  baseline wall (s): {per}   mean={sum(bw)/len(bw):.1f}')
            nb = [len(b["local_errors"]) for b in base.values()]
            ok = [sum(1 for e in b['local_errors']
                      if e.get('signed_local_error') is not None)
                  for b in base.values()]
            print(f'  NN clusters={nb}  with usable local error={ok}')
        rows = []
        for point in sorted([p for p in points if p != 'base']):
            mfrac, sfrac = point
            d = points[point]
            diffs, mass, entry, kact, walls, builds, samp, evals = \
                [], [], [], [], [], [], [], []
            per_cluster = []
            for s in sorted(d):
                b, m = base.get(s), d[s]
                if not b:
                    continue
                eb = {e['bucket']: e.get('signed_local_error')
                      for e in b['local_errors']}
                em = {e['bucket']: e.get('signed_local_error')
                      for e in m['local_errors']}
                mf = {e['bucket']: e.get('memorized_mass_fraction')
                      for e in m['local_errors']}
                mem = {x['bucket']: x for x in m.get('memorization_log', [])
                       if x.get('ok')}
                for k in sorted(set(eb) & set(em)):
                    if eb[k] is None or em[k] is None:
                        continue
                    ab, am = abs(eb[k]), abs(em[k])
                    diffs.append(am - ab)
                    mk = mem.get(k, {})
                    if mf.get(k) is not None:
                        mass.append(mf[k])
                    if mk.get('memorized_fraction') is not None:
                        entry.append(mk['memorized_fraction'])
                    kact.append(mk.get('k_actual'))
                    per_cluster.append(dict(
                        seed=s, bucket=k, w=mk.get('width'),
                        msg=mk.get('message_size'), k=mk.get('k_actual'),
                        entry_frac=mk.get('memorized_fraction'),
                        mass_frac=mf.get(k), err_base=eb[k], err_mem=em[k],
                        d_abs=am - ab))
                walls.append(m['wall_time_s'] / b['wall_time_s'])
                builds.append(m['phase_times_s'].get('memorize', 0.0))
                samp.append(sum(x.get('sample_seconds', 0.0)
                                for x in m.get('memorization_log', [])
                                if x.get('ok')))
                evals.append(sum(x.get('eval_seconds', 0.0)
                                 for x in m.get('memorization_log', [])
                                 if x.get('ok')))
            if not diffs:
                print(f'  point mem_frac={mfrac} pool={sfrac}: no paired data')
                continue
            w, l, p = sign_test(diffs)
            _, pw = wilcoxon(diffs)
            base_abs = [abs(r['err_base']) for r in per_cluster]
            mem_abs = [abs(r['err_mem']) for r in per_cluster]
            rec = dict(mem_frac=mfrac, sample_frac=sfrac, n=len(diffs), better=w, worse=l,
                       sign_p=p, wilcoxon_p=pw,
                       med_d=med(diffs),
                       med_abs_base=med(base_abs), med_abs_mem=med(mem_abs),
                       med_entry_frac=med(entry), med_mass_frac=med(mass),
                       mean_wall_mult=sum(walls) / len(walls),
                       wall_mults=[round(x, 3) for x in walls],
                       mean_build_s=sum(builds) / len(builds),
                       mean_sample_s=sum(samp) / len(samp) if samp else None,
                       mean_eval_s=sum(evals) / len(evals) if evals else None,
                       per_cluster=per_cluster)
            rows.append(rec)
            print(f'  K/2^w={mfrac:<7g} pool={sfrac:<6g} n={rec["n"]:<3d} '
                  f'{w}/{l} sign_p={p:.3g} wilc_p={pw:.3g} | '
                  f'med|err| {rec["med_abs_base"]:.4g}->{rec["med_abs_mem"]:.4g} '
                  f'(med d={rec["med_d"]:+.4g}) | '
                  f'entry={rec["med_entry_frac"]:.4g} mass={rec["med_mass_frac"]:.4g} | '
                  f'wall={rec["mean_wall_mult"]:.3f}x build={rec["mean_build_s"]:.1f}s '
                  f'(samp {rec["mean_sample_s"]:.1f}s / eval {rec["mean_eval_s"]:.1f}s)')
        # --- mass axis: per cluster, the smallest swept entry fraction whose
        # memorized entries already carry >= 50% of the exact message mass
        # (doc 44 s 4.2 threshold, restated as a question about K).
        knee = {}
        for rec in rows:
            for pc in rec['per_cluster']:
                if pc['mass_frac'] is None:
                    continue
                kk = (pc['seed'], pc['bucket'])
                if pc['mass_frac'] >= 0.5 and kk not in knee:
                    knee[kk] = rec['mem_frac']
        if rows:
            allk = {(pc['seed'], pc['bucket'])
                    for rec in rows for pc in rec['per_cluster']}
            print(f'  MASS KNEE: {len(knee)}/{len(allk)} (seed,cluster) reach '
                  f'>=50% mass somewhere in the sweep; smallest entry fraction '
                  f'that does: ' +
                  ', '.join(f'{v:g}:{sum(1 for x in knee.values() if x == v)}'
                            for v in sorted(set(knee.values()))))
        out[f'{cell[0]}|{cell[1]}|{cell[2]}'] = dict(
            mass_knee={f'{k[0]}_{k[1]}': v for k, v in knee.items()},
            baseline_wall=[base[s]['wall_time_s'] for s in sorted(base)],
            points=rows)
    with open(os.path.join(HERE, '50-sweep.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('\nwrote', os.path.join(HERE, '50-sweep.json'))


if __name__ == '__main__':
    main(sys.argv[1:] or ['results50'])
