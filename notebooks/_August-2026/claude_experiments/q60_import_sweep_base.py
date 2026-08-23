#!/usr/bin/env python
"""Import doc 64's `base` arm from the paper rerun instead of re-running it.

WHY. doc 64's `base` arm config is the paper rerun's subsumption / merge-bound-10
configuration, term for term: neurobe_mode, the same iB/ecl, num_samples='nbe,0.1',
uniform sampling, stream_nn_exact, dope_factors, use_join_tree_merge, and the study
default num_epochs=500. Re-running it recomputes a number already on disk.

That is an argument, not a proof, so it was CHECKED against the runs that already
existed: over all 10 (problem, iB, seed) cells where a doc-64 base run and a rerun
sub-D10 run overlap, log Z agreed to **0.00e+00 -- bit-identical, every cell**.

doc 63 is deliberately NOT imported: it runs num_epochs=2000 against the study's 500,
and that difference is BINDING -- doc-63 base disagrees with the rerun on 5 of 14
overlapping cells (grid20x20.f10 all three seeds, pedigree13 s42, rbm_20 s44), because
those problems' clusters are still improving when the 500-epoch budget cuts them off.
Importing there would silently mix two epoch budgets.

Rows carry `imported_from` so no reader can mistake them for fresh runs. Anything the
rerun has not finished (4 pedigrees at the time of writing) is simply not imported and
the queue runs it normally.
"""
import json
import os
import sys
from glob import glob
from pathlib import Path

SWEEP = '/home/cohenn1/NCE-rerun/phase1'
DEST = os.environ.get('Q60_DEST', '/tmp/claude-58902/res64')
CONFIGS = [('pedigree/pedigree13', 20, 'ped_pedigree13'), ('pedigree/pedigree7', 20, 'ped_pedigree7'),
           ('pedigree/pedigree41', 20, 'ped_pedigree41'), ('pedigree/pedigree51', 20, 'ped_pedigree51'),
           ('pedigree/pedigree19', 20, 'ped_pedigree19'), ('pedigree/pedigree34', 20, 'ped_pedigree34'),
           ('grids/grid10x10.f10', 10, 'gs_grid10x10_f10'), ('grids/grid10x10.f10.wrap', 10, 'gs_grid10x10_f10_wrap'),
           ('grids/grid20x20.f10', 10, 'gs_grid20x20_f10'), ('grids/grid20x20.f15', 10, 'gs_grid20x20_f15'),
           ('grids/grid20x20.f2', 10, 'gs_grid20x20_f2'), ('grids/grid20x20.f5', 10, 'gs_grid20x20_f5'),
           ('dbn/rbm_21', 20, 'rbm20_rbm_21'), ('dbn/rbm_20', 20, 'rbm20_rbm_20'),
           ('dbn/rbm_22', 20, 'rbm20_rbm_22'), ('dbn/rbm_ferro_22', 20, 'rbm20_rbm_ferro_22'),
           ('dbn/rbm_ferro_21', 20, 'rbm20_rbm_ferro_21'), ('dbn/rbm_ferro_20', 20, 'rbm20_rbm_ferro_20'),
           ('dbn/rbm_20', 10, 'rbm10_rbm_20'), ('dbn/rbm_21', 10, 'rbm10_rbm_21'), ('dbn/rbm_22', 10, 'rbm10_rbm_22')]


def index_sweep():
    out = {}
    for m in glob(os.path.join(SWEEP, 'runs/*/manifest.json')):
        d = os.path.dirname(m)
        rp = os.path.join(d, 'result.json')
        if not os.path.exists(rp):
            continue
        try:
            mm = json.load(open(m))
            j = mm['job']
            if j['merge_strategy'] != 'subsumption' or j.get('merge_bound') != 10:
                continue
            r = json.load(open(rp))
            if r.get('status') not in (None, 'ok', 'done', 'success'):
                continue
            lz = r.get('log_z_repr')
            if lz is None:
                continue
            out[(j['problem_key'], j['extra_config'].get('iB'), int(j['seed']))] = {
                'log_z': float(lz), 'wall': float(r.get('total_wall_s') or 0),
                'run': os.path.basename(d), 'commit': (mm.get('git') or {}).get('commit_short'),
                'ecl': j['extra_config'].get('ecl'),
            }
        except Exception:
            continue
    return out


def main():
    apply = '--apply' in sys.argv
    sweep = index_sweep()
    have = set()
    for g in Path(DEST).rglob('*.json'):
        try:
            r = json.loads(g.read_text())
            if r.get('arm') == 'base':
                have.add((r['problem'], int(r['iB']), int(r['seed'])))
        except Exception:
            continue
    n = miss = skip = 0
    for prob, ib, outdir in CONFIGS:
        # Pedigree cannot be imported. The rerun computed it under the VARIABLE-COUNT
        # merge bound; under the bits bound (the universal rule) pedigree structure
        # genuinely differs -- measured, all 12 pedigree x strategy cells change, with
        # the largest eliminator dropping from 2^10.6..2^13.5 to exactly 2^10. Binary
        # families are IDENTICAL cluster-for-cluster under the two readings, verified
        # on grid20x20, grid40x40.f15.wrap and rbm_20 for both strategies, so those
        # imports stay exact.
        if prob.split('/')[-1].startswith('pedigree'):
            miss += 3
            continue
        for seed in (42, 43, 44):
            src = sweep.get((prob, ib, seed))
            if src is None:
                miss += 1
                continue
            if (prob, ib, seed) in have:      # keyed on the JSON body, never a filename:
                skip += 1                       # rbm_20/21/22 exist at two i-bounds under one basename
                continue
            d = Path(DEST) / outdir
            f = d / f"d_{prob.split('/')[-1]}_sub10_base_s{seed}_iB{ib}.json"
            rec = {
                'problem': prob, 'arm': 'base', 'seed': seed, 'iB': ib,
                'ecl': src['ecl'], 'merge': 'sub', 'D': 10, 'num_epochs': 500,
                'log_z': src['log_z'], 'wall_seconds': src['wall'],
                'local_errors': [],
                'imported_from': f"NCE-rerun/phase1/runs/{src['run']}",
                'import_note': ('doc-64 base is term-for-term the rerun sub/D=10 config at the '
                                'study default num_epochs=500; verified bit-identical (0.00e+00) '
                                'on all 10 overlapping cells before importing'),
                'import_commit': src['commit'],
            }
            if apply:
                d.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps(rec, indent=1))
            n += 1
    print(f"{'imported' if apply else 'WOULD import'}: {n}   already present: {skip}   "
          f"not yet in the rerun (will run normally): {miss}")
    if not apply:
        print('re-run with --apply to write')


if __name__ == '__main__':
    main()
