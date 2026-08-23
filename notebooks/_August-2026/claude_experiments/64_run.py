#!/usr/bin/env python
"""Doc 64 -- memorization selection: fw_true vs fw_bw (exact forward x approximate backward).

Memorization currently picks the top-K separator entries by FORWARD message
value. What matters for log Z is the entry's CONTRIBUTION, forward x backward.
This runs the paired comparison.

The constraint: the approximate backward factors are built from the existing
cluster structure via FastBucket.get_backward_factor_list(), WITHOUT enabling
`use_bw_approx` and without changing the NN training target or the
preprocessor. Every run records `preproc_audit` so that can be checked rather
than asserted.

Arms: base (no memorization) | fw_true | fw_bw.
Everything else follows doc 50's recommended operating point (memorize 10% of
entries, 2:1 sample pool) and doc 50's runner config.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--problem', required=True)
    ap.add_argument('--ib', type=int, default=10)
    ap.add_argument('--ecl', type=int, default=1025)
    ap.add_argument('--merge', default='sub', choices=['sub', 'reducenn', 'none'])
    ap.add_argument('--D', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--arm', required=True, choices=['base', 'fw_true', 'fw_bw'])
    # Local error is the SECONDARY endpoint (end-to-end log Z is primary), and it costs
    # ~22 s/cluster -- ~23% of a pedigree run. The queue turns it on for seed 42 only,
    # which keeps the diagnostic while dropping two thirds of its cost.
    ap.add_argument('--local-error', type=int, default=1)
    # doc 50's recommendation: memorize 10% of entries, constant 2:1 oversample.
    ap.add_argument('--mem-frac', type=float, default=0.1)
    ap.add_argument('--sample-frac', type=float, default=0.2)
    ap.add_argument('--bw-ecl', type=int, default=1024)
    ap.add_argument('--num-epochs', type=int, default=500)
    ap.add_argument('--tag', default='run')
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    from nce.config_schema import prepare_config
    from nce.inference.graphical_model import FastGM
    from nce.benchmark_problems.catalog_utils import get_catalog
    import torch

    cfg = dict(
        neurobe_mode=True, iB=args.ib, ecl=args.ecl,
        num_samples='nbe,0.1', sampling_scheme='uniform',
        stream_nn_exact=True, dope_factors=True,
        device='cuda', seed=args.seed, approximation_method='nn',
        verbose_merge=False, num_epochs=args.num_epochs,
        compute_local_error=bool(args.local_error),
        # Backward factors are POPULATED for every arm, including fw_true and
        # base, so the arms differ only in whether the selection rule reads them.
        # (populate_bw_factors does not enable use_bw_approx; doc 57.)
        populate_bw_factors=True,
        populate_bw_via_tree_collect=True,   # doc 54 s4.3: far more accurate bw
        populate_bw_skip_non_nn=True,
        bw_ecl=args.bw_ecl,
    )
    if args.merge == 'sub':
        cfg['use_join_tree_merge'] = True
        cfg['max_merge_bound'] = args.D
    elif args.merge == 'reducenn':
        cfg['use_reduce_nn_merge'] = True
        cfg['max_merge_bound'] = args.D

    if args.arm != 'base':
        cfg.update(
            use_memorization_table=True,
            memorize_frac=args.mem_frac,
            memorize_sample_frac=args.sample_frac,
            memorize_top_k=0,
            memorize_num_samples=0,
            memorize_selection=args.arm,
        )

    full = prepare_config(cfg, strict=False)
    assert full.get('use_bw_approx', False) is False, \
        "use_bw_approx must stay OFF: it would change the training target."

    model = get_catalog()[args.problem]
    ref = getattr(model, 'PR', None)
    try:
        ref = float(ref)
        if ref != ref:
            ref = None
    except (TypeError, ValueError):
        ref = None

    t0 = time.time()
    gm = FastGM(model=model, nn_config=full, device=full['device'])
    gm.eliminate_variables(all=True)
    wall = time.time() - t0

    out = {
        'problem': args.problem, 'arm': args.arm, 'seed': args.seed,
        'iB': args.ib, 'ecl': args.ecl, 'merge': args.merge, 'D': args.D,
        'mem_frac': args.mem_frac, 'sample_frac': args.sample_frac,
        'bw_ecl': args.bw_ecl, 'num_epochs': args.num_epochs,
        'log_z': float(gm.log_partition_function), 'ref_logZ': ref,
        'wall_seconds': wall,
        'num_trained': int(getattr(gm, 'num_trained', 0)),
        'local_errors': getattr(gm, 'local_errors', []),
        'memorization_log': getattr(gm, 'memorization_log', []),
        'preproc_audit': getattr(gm, 'preproc_audit', []),
        'use_bw_approx': full.get('use_bw_approx', False),
        'cuda_device': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'cuda_peak_alloc_gb': (torch.cuda.max_memory_allocated() / 2 ** 30
                               if torch.cuda.is_available() else None),
    }
    odir = Path(args.outdir)
    odir.mkdir(parents=True, exist_ok=True)
    name = f"{args.tag}_{args.problem.split('/')[-1]}_{args.merge}{args.D}_{args.arm}_s{args.seed}"
    (odir / f'{name}.json').write_text(json.dumps(out, indent=1))

    # Loud check: a silently-degraded fw_bw arm looks exactly like a null result.
    if args.arm == 'fw_bw':
        eff = [m.get('selection_effective') for m in out['memorization_log'] if m.get('ok')]
        bad = [e for e in eff if e != 'fw_bw']
        print(f"[64] fw_bw effective on {len(eff) - len(bad)}/{len(eff)} clusters"
              + (f"  *** {len(bad)} DEGRADED TO fw_true ***" if bad else ""))
    nok = sum(1 for m in out['memorization_log'] if not m.get('ok'))
    print(f"[64] wrote {odir / (name + '.json')}  logZ={out['log_z']:.6f} "
          f"clusters={len(out['local_errors'])} memfail={nok} wall={wall:.1f}s")


if __name__ == '__main__':
    main()
