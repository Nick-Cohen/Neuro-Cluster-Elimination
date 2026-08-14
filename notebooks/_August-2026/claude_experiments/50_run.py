#!/usr/bin/env python3
"""Doc 50 runner: one arm of the memorization-threshold sweep on one cell.

Verbatim 44_run.py plus: the sweep point (sample_frac, mem_frac) is written into
the experiment name so points do not collide, and cuda device / peak memory are
recorded.

Usage:
  python 50_run.py --problem grids/grid20x20.f10 --iB 10 --ecl 1025 \
      --merge reducenn --D 10 --seed 42 --arm mem --mem-frac 0.01 \
      --sample-frac 0.1 --tag k0100 --outdir results50
"""
import argparse, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = str(HERE.parents[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--problem', required=True)
    ap.add_argument('--iB', type=int, default=10)
    ap.add_argument('--ecl', type=int, default=1025)
    ap.add_argument('--merge', default='reducenn', choices=['reducenn', 'sub', 'none'])
    ap.add_argument('--D', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--arm', default='base', choices=['base', 'mem'])
    ap.add_argument('--sample-frac', type=float, default=0.1)
    ap.add_argument('--mem-frac', type=float, default=0.01)
    ap.add_argument('--sample-cap', type=int, default=0)
    ap.add_argument('--mem-cap', type=int, default=0)
    ap.add_argument('--selection', default='fw_true')
    ap.add_argument('--bw-ecl', type=int, default=1024)
    ap.add_argument('--num-epochs', type=int, default=500)
    ap.add_argument('--local-error', type=int, default=1)
    ap.add_argument('--tag', default='run')
    ap.add_argument('--outdir', default='results44')
    args = ap.parse_args()

    sys.path.insert(0, REPO)
    import torch
    torch.set_num_threads(int(os.environ.get('NCE_THREADS', '14')))
    from nce.benchmark_problems.catalog_utils import get_catalog
    from nce.inference.graphical_model import FastGM
    from nce.config_schema import prepare_config

    catalog = get_catalog()
    model = catalog[args.problem]
    ref_logZ = getattr(model, 'PR', None)
    if ref_logZ is not None and ref_logZ != ref_logZ:
        ref_logZ = None

    cfg = dict(
        neurobe_mode=True, iB=args.iB, ecl=args.ecl,
        num_samples='nbe,0.1', sampling_scheme='uniform',
        stream_nn_exact=True, dope_factors=True,
        device='cuda', seed=args.seed, approximation_method='nn',
        verbose_merge=False, num_epochs=args.num_epochs,
        compute_local_error=bool(args.local_error),
        # backward factors: needed by the proposal tree, set on BOTH arms so the
        # timing comparison is meaningful.
        populate_bw_factors=True,
        populate_bw_via_tree_collect=True,
        populate_bw_skip_non_nn=True,
        bw_ecl=args.bw_ecl,
    )
    if args.merge == 'reducenn':
        cfg['use_reduce_nn_merge'] = True
        cfg['max_merge_bound'] = args.D
    elif args.merge == 'sub':
        cfg['use_join_tree_merge'] = True
        cfg['max_merge_bound'] = args.D

    if args.arm == 'mem':
        cfg.update(use_memorization_table=True,
                   memorize_sample_frac=args.sample_frac,
                   memorize_frac=args.mem_frac,
                   memorize_num_samples=args.sample_cap,
                   memorize_top_k=args.mem_cap,
                   memorize_selection=args.selection)

    full = prepare_config(cfg, strict=False)
    exp = (f"{args.tag}_{args.problem.split('/')[-1]}_{args.merge}{args.D}"
           f"_{args.arm}_s{args.seed}")
    out_dir = HERE / args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 64, flush=True)
    print(f'EXPERIMENT {exp} | arm={args.arm} | CUDA_VISIBLE_DEVICES='
          f'{os.environ.get("CUDA_VISIBLE_DEVICES")}', flush=True)
    print('=' * 64, flush=True)

    t0 = time.time()
    gm = FastGM(model=model, nn_config=full, device=full['device'])
    t_build = time.time() - t0
    n_clusters = len(gm.buckets)

    t0 = time.time()
    gm.eliminate_variables(all=True)
    t_elim = time.time() - t0
    log_z = float(gm.log_partition_function)

    epochs = [d.get('epochs_trained') for d in getattr(gm, 'per_bucket_training_log', [])
              if d.get('epochs_trained') is not None]
    result = {
        'experiment_name': exp, 'problem': args.problem, 'arm': args.arm,
        'iB': args.iB, 'ecl': args.ecl, 'merge': args.merge, 'D': args.D,
        'seed': args.seed, 'bw_ecl': args.bw_ecl,
        'sample_frac': args.sample_frac, 'mem_frac': args.mem_frac,
        'selection': args.selection,
        'log_z': log_z, 'ref_logZ': ref_logZ,
        'abs_err': abs(log_z - ref_logZ) if ref_logZ is not None else None,
        'num_trained': int(gm.num_trained), 'n_clusters_post_merge': n_clusters,
        'num_nn_buckets': len(epochs), 'eff_epochs_per_bucket': epochs,
        't_build_s': t_build, 't_elim_s': t_elim, 'wall_time_s': t_build + t_elim,
        'phase_times_s': getattr(gm, 'phase_times', {}),
        'local_errors': getattr(gm, 'local_errors', []),
        'memorization_log': getattr(gm, 'memorization_log', []),
        'sample_cap': args.sample_cap, 'mem_cap': args.mem_cap,
        'tag': args.tag,
        'cuda_device': (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else None),
        'cuda_peak_alloc_gb': (torch.cuda.max_memory_allocated() / 2 ** 30
                               if torch.cuda.is_available() else None),
    }
    with open(out_dir / f'{exp}.json', 'w') as f:
        json.dump(result, f, indent=2)
    print('=' * 64, flush=True)
    print(f'DONE {exp}: log_Z={log_z:.4f} trained={gm.num_trained} '
          f'build={t_build:.1f}s elim={t_elim:.1f}s '
          f'memorize={result["phase_times_s"].get("memorize", 0.0):.1f}s', flush=True)
    print(f'  -> {out_dir / (exp + ".json")}', flush=True)


if __name__ == '__main__':
    main()
