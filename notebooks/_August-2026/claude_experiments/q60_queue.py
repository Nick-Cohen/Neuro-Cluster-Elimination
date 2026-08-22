#!/usr/bin/env python
"""Restartable sequential queue driver for docs 63 (Q60 WMB-as-input) and 64 (fw_bw).

Full-benchmark extension: the 21 non-grid40 configs of the reduce-NN benchmark set
(pedigree x6 @iB20, grid-small x6 @iB10, rbm x6 @iB20, rbm x3 @iB10).

Why a python driver and not the old 63_drive.sh / 64_drive.sh:
  1. the shell drivers HARDCODE `--ib 10 --ecl 1025`; the benchmark set has per-GROUP
     iB/ecl (pedigree and the rbm-20 group are iB=20, ecl=2**20+1) and those are user
     intent, not ours to flatten;
  2. this run is ~7 days long and long-lived processes get reaped here, so the queue
     must be RESTARTABLE -- relaunching must skip everything already on disk;
  3. rbm_20/21/22 appear in the benchmark set at BOTH iB=10 and iB=20, which collide
     under the runner's filename scheme, so completion is keyed on the run's recorded
     (problem, iB, merge, D, arm, seed) read back out of the JSON, never on a path.

gpu1 ONLY. gpu0/gpu3 carry the paper rerun whose timings are a published result;
gpu2 is retired for silent data corruption.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- benchmark set
# 21 configs. grid40x40 (8 configs) is deliberately EXCLUDED: measured from the live
# sweep, grid40x40.f15.wrap at sub D=10 costs 41,734 s per run, so 8 configs x 21 runs
# is ~1,948 GPU-hours ~ 81 days on one card. That is a separate decision for Nick.
GROUPS = [
    # (family label, iB, ecl, [problem keys, cheapest first by reduce-NN sub_time])
    ('ped',   20, 1048577, ['pedigree/pedigree13', 'pedigree/pedigree7',
                            'pedigree/pedigree41', 'pedigree/pedigree51',
                            'pedigree/pedigree19', 'pedigree/pedigree34']),
    ('gs',    10, 1025,    ['grids/grid10x10.f10', 'grids/grid10x10.f10.wrap',
                            'grids/grid20x20.f10', 'grids/grid20x20.f15',
                            'grids/grid20x20.f2', 'grids/grid20x20.f5']),
    ('rbm20', 20, 1048577, ['dbn/rbm_21', 'dbn/rbm_20', 'dbn/rbm_22',
                            'dbn/rbm_ferro_22', 'dbn/rbm_ferro_21',
                            'dbn/rbm_ferro_20']),
    ('rbm10', 10, 1025,    ['dbn/rbm_20', 'dbn/rbm_21', 'dbn/rbm_22']),
]

ARMS63 = ['base', 'residual', 'input', 'parts']
ARMS64 = ['base', 'fw_true', 'fw_bw']
SEEDS = [42, 43, 44]
MERGE, D = 'sub', 10
EPOCHS63, EPOCHS64 = 2000, 500      # each doc's own established protocol -- untouched


def slug(problem):
    return problem.split('/')[-1].replace('.', '_')


def build_order(exp):
    """(config, seed-phase) blocks, families interleaved, cheapest-first within family.

    Ordering requirement: an interrupted run must still give interpretable PER-FAMILY
    coverage. Shortest-job-first end to end would finish every grid before touching
    pedigree, and pedigree is the biggest gap in both docs. So:
      * phase 1 emits seeds 42+43 for every config -> a 2-seed spread everywhere, which
        is the minimum that lets the per-seed ranges be compared at all;
      * phase 2 adds seed 44 in the same order;
      * within each phase, families round-robin (ped, gs, rbm20, rbm10), cheapest-first,
        so a day in you have some pedigree AND some rbm AND some grid.
    """
    arms = ARMS63 if exp == 63 else ARMS64
    epochs = EPOCHS63 if exp == 63 else EPOCHS64
    out = []
    for seeds in ([42, 43], [44]):
        rings = [[(fam, ib, ecl, p) for p in probs] for fam, ib, ecl, probs in GROUPS]
        i = 0
        while any(rings):
            ring = rings[i % len(rings)]      # fixed modulus: rings are emptied, never removed
            i += 1
            if not ring:
                continue
            fam, ib, ecl, prob = ring.pop(0)
            for seed in seeds:
                for arm in arms:
                    out.append({
                        'exp': exp, 'problem': prob, 'arm': arm, 'seed': seed,
                        'ib': ib, 'ecl': ecl, 'epochs': epochs, 'family': fam,
                        'outdir': f'{fam}_{slug(prob)}',
                    })
    return out


def index_done(root):
    """(problem, iB, merge, D, arm, seed) -> path, for every parseable run under root.

    Read out of the JSON body, not the filename: the pre-existing iB=10 runs use a
    different tag and rbm_2x exists at two i-bounds under the same basename.
    """
    done = {}
    for f in Path(root).rglob('*.json'):
        try:
            r = json.loads(f.read_text())
            done[(r['problem'], int(r['iB']), r['merge'], int(r['D']),
                  r['arm'], int(r['seed']))] = str(f)
        except Exception:
            continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root63', default='/tmp/claude-58902/res63')
    ap.add_argument('--root64', default='/tmp/claude-58902/res64')
    ap.add_argument('--log', default='/tmp/claude-58902/q60-full.log')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--plan-out', default=None)
    args = ap.parse_args()

    # Interleave the two experiments as well, so BOTH docs gain coverage together
    # rather than doc 64 waiting three days behind doc 63.
    o63, o64 = build_order(63), build_order(64)
    order, i, j = [], 0, 0
    while i < len(o63) or j < len(o64):
        for _ in range(len(ARMS63)):
            if i < len(o63):
                order.append(o63[i]); i += 1
        for _ in range(len(ARMS64)):
            if j < len(o64):
                order.append(o64[j]); j += 1

    roots = {63: args.root63, 64: args.root64}
    done = {63: index_done(roots[63]), 64: index_done(roots[64])}
    todo = [s for s in order
            if (s['problem'], s['ib'], MERGE, D, s['arm'], s['seed']) not in done[s['exp']]]

    n_skip = len(order) - len(todo)
    hdr = (f"[q60] planned={len(order)} already_done={n_skip} todo={len(todo)}")
    print(hdr)
    if args.plan_out:
        Path(args.plan_out).write_text(json.dumps(todo, indent=1))
    if args.dry_run:
        for s in todo[:40]:
            print('  ', s['exp'], s['family'], s['problem'], 'iB%d' % s['ib'], s['arm'], 's%d' % s['seed'])
        print('   ...')
        return

    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = '1'          # gpu1 ONLY
    env['NCE_MODEL_CACHE'] = '/home/cohenn1/NCE/.model_cache'
    env['NCE_THREADS'] = '6'
    py = '/home/cohenn1/NCE/venv/bin/python'

    with open(args.log, 'a') as log:
        log.write(f"\n===== q60 queue start {time.strftime('%FT%T')} {hdr} =====\n")
        log.flush()
        for n, s in enumerate(todo, 1):
            key = (s['problem'], s['ib'], MERGE, D, s['arm'], s['seed'])
            # cheap incremental re-check (the initial index is authoritative; each
            # completed run is folded in below, so no rglob per iteration)
            if key in done[s['exp']]:
                log.write(f"[skip already present] {key}\n"); log.flush()
                continue
            odir = Path(roots[s['exp']]) / s['outdir']
            tag = ('c%d' % s['ib']) if s['exp'] == 63 else ('d%d' % s['ib'])
            cmd = [py, str(HERE / f"{s['exp']}_run.py"),
                   '--problem', s['problem'], '--arm', s['arm'],
                   '--seed', str(s['seed']), '--ib', str(s['ib']), '--ecl', str(s['ecl']),
                   '--merge', MERGE, '--D', str(D), '--num-epochs', str(s['epochs']),
                   '--tag', tag, '--outdir', str(odir)]
            if s['exp'] == 63:
                cmd += ['--local-error', '1']
            log.write(f"=== {time.strftime('%FT%T')} [{n}/{len(todo)}] doc{s['exp']} "
                      f"{s['problem']} iB{s['ib']} {s['arm']} s{s['seed']} ===\n")
            log.flush()
            t0 = time.time()
            rc = subprocess.call(cmd, stdout=log, stderr=log, env=env)
            log.write(f"=== done rc={rc} in {time.time() - t0:.0f}s ===\n")
            log.flush()
            if rc == 0:
                done[s['exp']][key] = 'just-ran'
        log.write(f"ALL DONE {time.strftime('%FT%T')}\n")
        log.flush()


if __name__ == '__main__':
    main()
