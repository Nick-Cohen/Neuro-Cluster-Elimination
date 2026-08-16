#!/usr/bin/env python
"""Doc 63 / Q60 -- one NeuroBE run under one WMB arm, recording per-cluster local error.

Arms
    base        plain NeuroBE (no WMB signal)
    residual    target = exact - WMB, residual normaliser  (doc 31 arm 5eps)
    input       WMB estimate as ONE extra input column
    parts       WMB estimate + each mini-bucket partition as input columns

Everything except the arm switch is byte-identical across arms, and CRN is
default-on in this build, so arms sharing a separator draw identical assignments.
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
    ap.add_argument('--arm', required=True,
                    choices=['base', 'residual', 'input', 'parts'])
    ap.add_argument('--num-epochs', type=int, default=2000)
    ap.add_argument('--local-error', type=int, default=1)
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
    )
    if args.merge == 'sub':
        cfg['use_join_tree_merge'] = True
        cfg['max_merge_bound'] = args.D
    elif args.merge == 'reducenn':
        cfg['use_reduce_nn_merge'] = True
        cfg['max_merge_bound'] = args.D

    # --- the ONLY difference between arms -------------------------------------
    if args.arm == 'residual':
        cfg['wmb_residual'] = True
        cfg['wmb_residual_norm'] = 'residual'
        cfg['wmb_residual_eps_compensate'] = True
    elif args.arm == 'input':
        cfg['wmb_input'] = 'combined'
    elif args.arm == 'parts':
        cfg['wmb_input'] = 'partitions'

    full = prepare_config(cfg, strict=False)

    catalog = get_catalog()
    model = catalog[args.problem]
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
    log_z = float(gm.log_partition_function)

    out = {
        'problem': args.problem, 'arm': args.arm, 'seed': args.seed,
        'iB': args.ib, 'ecl': args.ecl, 'merge': args.merge, 'D': args.D,
        'num_epochs': args.num_epochs,
        'log_z': log_z, 'ref_logZ': ref,
        'wall_seconds': wall,
        'num_trained': int(getattr(gm, 'num_trained', 0)),
        'local_errors': getattr(gm, 'local_errors', []),
        'wmb_base_stats': getattr(gm, 'wmb_base_stats', []),
        'cuda_device': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'cuda_peak_alloc_gb': (torch.cuda.max_memory_allocated() / 2 ** 30
                               if torch.cuda.is_available() else None),
        # Constraint audit (experiment 2 needs this; free to record here too):
        # the fitted target normaliser per cluster, so arms can be proved to have
        # trained against the same target.
        'preproc_stats': getattr(gm, 'preproc_audit', []),
        'config_echo': {k: full.get(k) for k in
                        ('num_epochs', 'loss_fn', 'hidden_sizes', 'lr', 'batch_size',
                         'device', 'ecl', 'iB', 'normalization_mode', 'sampling_scheme',
                         'num_samples', 'common_random_numbers', 'wmb_input',
                         'wmb_residual', 'wmb_residual_norm',
                         'wmb_residual_eps_compensate')},
    }
    odir = Path(args.outdir)
    odir.mkdir(parents=True, exist_ok=True)
    name = f"{args.tag}_{args.problem.split('/')[-1]}_{args.merge}{args.D}_{args.arm}_s{args.seed}"
    (odir / f'{name}.json').write_text(json.dumps(out, indent=1))
    print(f"[63] wrote {odir / (name + '.json')}  logZ={log_z:.6f} "
          f"clusters={len(out['local_errors'])} wall={wall:.1f}s")


if __name__ == '__main__':
    main()
