"""Build the phase-1 queue for the corrected paper rerun.

PHASE 1 = every cell of the June-2026 reduce-NN benchmark set, at merge bounds
through e_max=14, at 3 seeds.

    29 cells x 16 arms x 3 seeds = 1392 jobs

Nothing here is invented. Every number comes from one of two frozen inputs:

  * `benchmark_set.json`  -- the 29 (problem, iB) cells and `ecl = 2**iB + 1`.
  * the study's own per-arm YAML in `reduce_nn_experiment/configs/` -- the
    config body and the merge flags for each arm. `--verify-configs` diffs the
    config this script would produce against those YAMLs, field for field, and
    refuses to enqueue on any mismatch.

ORDERING: shortest job first
---------------------------
`JobSpec.sort_key` leads with `tag`, and `JobQueue.pending()` is sorted by it,
so putting a zero-padded time estimate in the tag makes the scheduler dispatch
shortest-job-first. Early results then arrive in minutes rather than hours, so a
mistake is caught cheaply. The estimate comes from the ORIGINAL study's measured
`time_min` in `results_for_writeup/results_table.csv`; it affects dispatch order
only and never touches a config.

The original table covers 363 of the 464 (cell, arm) pairs. For the rest the
estimate is filled in, in this order, and the rule used is recorded per row in
the emitted CSV so the ordering is auditable:

    exact      the table has this (cell, arm)
    nearest    same cell, same arm family (rnn*/sub*), nearest bound
    cellmean   same cell, mean over all its arms
    analogue   cell absent from the table entirely (rbm_20 iB20,
               rbm_ferro_20 iB20): mean of the sibling rbm cells at the same iB
               and the same arm

Usage:
    NCE_MODEL_CACHE=/home/cohenn1/NCE/.model_cache \
    python notebooks/_August-2026/claude_experiments/61_enqueue_phase1.py \
        --queue /home/cohenn1/NCE-rerun/phase1/queue.json \
        --est-csv notebooks/_August-2026/claude_experiments/61-phase1-estimates.csv \
        --verify-configs [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

STUDY = '/home/cohenn1/NCE/notebooks/June-2026/claude_experiments/reduce_nn_experiment'
BENCH = os.path.join(STUDY, 'benchmark_set.json')
TABLE = os.path.join(STUDY, 'results_for_writeup', 'results_table.csv')
CONFIGS = os.path.join(STUDY, 'configs')

SEEDS = (42, 43, 44)
BOUNDS = (2, 4, 6, 8, 10, 12, 14)      # "merge bounds through e_max 14"
NONSUB_BOUND = 12                       # the study ran exactly one nonsub arm

# arm label -> (merge_strategy, merge_bound)
ARMS = {'nomerge': ('nomerge', None),
        'nonsub%d' % NONSUB_BOUND: ('non_subsumption', NONSUB_BOUND)}
for _d in BOUNDS:
    ARMS['rnn%d' % _d] = ('reduce_nn', _d)
    ARMS['sub%d' % _d] = ('subsumption', _d)


def cells():
    """The 29 (problem_key, iB, ecl) cells, in benchmark_set.json order."""
    with open(BENCH) as fh:
        bs = json.load(fh)
    out = []
    for grp in bs['groups']:
        iB, ecl = int(grp['iB']), int(grp['ecl'])
        assert ecl == 2 ** iB + 1, (grp['group'], iB, ecl)   # the set's own rule
        for pr in grp['problems']:
            out.append({'problem_key': pr['key'], 'iB': iB, 'ecl': ecl,
                        'group': grp['group'], 'ref_logZ': pr['ref_logZ'],
                        'table_name': table_name(pr['key'], iB)})
    return out


def table_name(problem_key: str, iB: int) -> str:
    """The name this cell has in the study's results_table.csv."""
    return '%s_iB%d' % (problem_key.rpartition('/')[2].replace('.', ''), iB)


def group_config(iB: int, ecl: int):
    """Config body shared by every arm of a cell.

    Verbatim from the study's YAML minus the per-job fields (experiment_name,
    problem_key, seed) and minus the merge flags, which JobSpec.to_config owns.
    """
    return {'neurobe_mode': True, 'iB': iB, 'ecl': ecl,
            'num_samples': 'nbe,0.1', 'sampling_scheme': 'uniform',
            'stream_nn_exact': True, 'dope_factors': True, 'device': 'cuda'}


# ---------------------------------------------------------------------------
# time estimates -> dispatch order
# ---------------------------------------------------------------------------
def load_table():
    t = {}
    with open(TABLE) as fh:
        for r in csv.DictReader(fh):
            try:
                t[(r['problem'], r['strategy'])] = float(r['time_min'])
            except (TypeError, ValueError):
                pass
    return t


def _family(arm):
    if arm.startswith('rnn'):
        return 'rnn'
    if arm.startswith('sub'):
        return 'sub'
    return arm


def estimate(tab, cell, arm, all_cells):
    name = cell['table_name']
    if (name, arm) in tab:
        return tab[(name, arm)], 'exact'
    fam = _family(arm)
    if fam in ('rnn', 'sub'):
        want = int(arm[len(fam):])
        cands = [(abs(int(k[1][len(fam):]) - want), v)
                 for k, v in tab.items()
                 if k[0] == name and k[1].startswith(fam)
                 and k[1][len(fam):].isdigit()]
        if cands:
            return min(cands)[1], 'nearest'
    mine = [v for k, v in tab.items() if k[0] == name]
    if mine:
        return sum(mine) / len(mine), 'cellmean'
    # cell absent from the table: use sibling cells of the same group and iB
    sibs = [c['table_name'] for c in all_cells
            if c['group'] == cell['group'] and c['iB'] == cell['iB']
            and c['table_name'] != name]
    vals = [tab[(s, arm)] for s in sibs if (s, arm) in tab]
    if not vals:
        vals = [v for k, v in tab.items() if k[0] in sibs]
    if not vals:
        raise SystemExit('no time estimate derivable for %s / %s -- refusing to '
                         'guess. Fix the estimator or the mapping.' % (name, arm))
    return sum(vals) / len(vals), 'analogue'


def tag_for(est_min):
    """Zero-padded hundredths of a minute: sorts lexicographically = SJF."""
    return 'p1-%08d' % int(round(est_min * 100))


# ---------------------------------------------------------------------------
# config fidelity check against the study's own YAML
# ---------------------------------------------------------------------------
IGNORE = {'experiment_name', 'problem_key'}


def verify_configs(specs_by_arm):
    """Diff our generated config against every study YAML we can pair with."""
    try:
        import yaml
    except ImportError:
        raise SystemExit('pyyaml needed for --verify-configs')
    checked = mismatched = unpaired = 0
    problems = []
    for fn in sorted(os.listdir(CONFIGS)):
        if not fn.endswith('.yaml'):
            continue
        stem = fn[:-5]
        if not stem.endswith(('_s42', '_s43', '_s44')):
            continue
        body, _, seedtok = stem.rpartition('_')
        cellname, _, arm = body.rpartition('_')
        key = (cellname, arm, int(seedtok[1:]))
        if key not in specs_by_arm:
            unpaired += 1
            continue
        with open(os.path.join(CONFIGS, fn)) as fh:
            want = yaml.safe_load(fh)
        got = specs_by_arm[key]
        checked += 1
        for k, v in want.items():
            if k in IGNORE:
                continue
            if got.get(k) != v:
                mismatched += 1
                problems.append('%s: %s want=%r got=%r' % (fn, k, v, got.get(k)))
        for k, v in got.items():
            if k in IGNORE or k in want:
                continue
            problems.append('%s: EXTRA %s=%r not in the study YAML' % (fn, k, v))
            mismatched += 1
    print('config fidelity: %d YAML(s) paired and checked, %d field mismatch(es), '
          '%d study YAML(s) outside phase 1 (ignored)'
          % (checked, mismatched, unpaired))
    for p in problems[:40]:
        print('   ! %s' % p)
    return checked, mismatched


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--queue', required=True)
    p.add_argument('--est-csv', default=None)
    p.add_argument('--verify-configs', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
    from nce.scheduler.jobs import JobSpec, JobQueue
    from nce.scheduler import models as modelval

    root = modelval.assert_cache_configured()
    print('model cache: %s' % root)

    cs = cells()
    tab = load_table()
    print('cells: %d   arms: %d   seeds: %d   -> %d jobs'
          % (len(cs), len(ARMS), len(SEEDS), len(cs) * len(ARMS) * len(SEEDS)))

    # ---- every model must load, validated by CONTENT not existence ----
    bad = []
    for key in sorted({c['problem_key'] for c in cs}):
        mv = modelval.validate(key)
        if not mv['ok']:
            bad.append((key, mv['problems']))
    print('model validation: %d distinct problems, %d bad'
          % (len({c['problem_key'] for c in cs}), len(bad)))
    for k, pr in bad:
        print('   ! %s: %s' % (k, pr))
    if bad:
        raise SystemExit('refusing to enqueue with unloadable models')

    specs, rows, by_arm = [], [], {}
    for cell in cs:
        base = group_config(cell['iB'], cell['ecl'])
        for arm, (strat, bound) in sorted(ARMS.items()):
            est, how = estimate(tab, cell, arm, cs)
            tag = tag_for(est)
            for seed in SEEDS:
                js = JobSpec(problem_key=cell['problem_key'],
                             merge_strategy=strat, merge_bound=bound,
                             seed=seed, extra_config=dict(base), tag=tag)
                specs.append(js)
                by_arm[(cell['table_name'], arm, seed)] = js.to_config()
            rows.append({'cell': cell['table_name'],
                         'problem_key': cell['problem_key'], 'iB': cell['iB'],
                         'arm': arm, 'strategy': strat,
                         'bound': '' if bound is None else bound,
                         'est_min': round(est, 3), 'est_rule': how, 'tag': tag,
                         'n_seeds': len(SEEDS)})

    total_min = sum(r['est_min'] * r['n_seeds'] for r in rows)
    from collections import Counter
    print('estimate provenance: %s' % dict(Counter(r['est_rule'] for r in rows)))
    print('estimated total: %.0f GPU-min = %.0f GPU-h  (2 cards -> %.1f days)'
          % (total_min, total_min / 60, total_min / 60 / 2 / 24))
    print('shortest 5 cells: %s'
          % [(r['cell'], r['arm'], r['est_min'])
             for r in sorted(rows, key=lambda r: r['est_min'])[:5]])
    print('longest 5 cells:  %s'
          % [(r['cell'], r['arm'], r['est_min'])
             for r in sorted(rows, key=lambda r: -r['est_min'])[:5]])

    if args.est_csv:
        with open(args.est_csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print('estimates -> %s' % args.est_csv)

    if args.verify_configs:
        checked, mismatched = verify_configs(by_arm)
        if mismatched or checked == 0:
            raise SystemExit('config fidelity check FAILED -- not enqueuing')

    ids = {s.job_id for s in specs}
    if len(ids) != len(specs):
        raise SystemExit('job_id collision: %d specs, %d ids'
                         % (len(specs), len(ids)))
    print('job ids: %d specs, %d distinct' % (len(specs), len(ids)))

    if args.dry_run:
        print('DRY RUN: nothing written to %s' % args.queue)
        for s in sorted(specs, key=lambda j: j.sort_key)[:8]:
            print('   %s' % s.name)
        return 0

    q = JobQueue(args.queue)
    added = q.add(sorted(specs, key=lambda j: j.sort_key))
    print('added %d new; queue now: %s' % (added, q.summary()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
