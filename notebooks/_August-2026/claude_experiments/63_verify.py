#!/usr/bin/env python
"""Doc 63 correctness gate: the WMB feature columns the net is EVALUATED with must
be the ones it was TRAINED with.

If the FactorNN evaluation path rebuilt the features differently -- wrong column
order, wrong normalisation, wrong clamp, wrong coordinate decoding -- the net
would be queried off-distribution and every local-error number would be garbage
while still looking plausible. So we check it directly:

  for each trained cluster,
      A = undo_normalization( net(x_train) )            # the training input rows
      B = FactorNN.to_exact()[ assignments_train ]      # the evaluation path
  and require A == B.

`to_exact()` is the exact code path the local-error metric uses, and it rebuilds
the feature columns from raw assignment coordinates via WMBFeatureSpec, so this
exercises the whole chain end to end.
"""
import argparse
import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, REPO)

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--problem', default='grids/grid10x10.f10.wrap')
    ap.add_argument('--arm', required=True, choices=['base', 'input', 'parts'])
    ap.add_argument('--num-epochs', type=int, default=5)
    ap.add_argument('--ib', type=int, default=10)
    ap.add_argument('--ecl', type=int, default=1025)
    ap.add_argument('--D', type=int, default=10)
    args = ap.parse_args()

    from nce.config_schema import prepare_config
    from nce.inference.graphical_model import FastGM
    from nce.benchmark_problems.catalog_utils import get_catalog
    from nce.data.data_loader import DataLoader
    from nce.inference.factor_nn import FactorNN

    # --- capture the last training load per cluster, and each emitted factor ---
    loads = []
    factors = []

    _orig_load = DataLoader.load

    def load(self, num_samples=0, all=False, is_validation=False):
        x, y, bw = _orig_load(self, num_samples=num_samples, all=all,
                              is_validation=is_validation)
        # Re-derive the assignments the same way load() did is not possible after
        # the fact, so stash them from the sample generator's last draw instead:
        # load() is the only caller and it samples exactly once per call.
        loads.append({'x': x.detach().clone(), 'n': x.shape[0],
                      'spec': self.wmb_spec, 'assign': self._last_assignments})
        return x, y, bw

    # make load() record the assignments it drew
    _orig_load_src = DataLoader.load

    def load2(self, num_samples=0, all=False, is_validation=False):
        sg = self.sample_generator
        _osa = sg.sample_assignments
        holder = {}

        def sa(*a, **k):
            out = _osa(*a, **k)
            holder['a'] = out
            return out
        sg.sample_assignments = sa
        try:
            x, y, bw = _orig_load_src(self, num_samples=num_samples, all=all,
                                      is_validation=is_validation)
        finally:
            sg.sample_assignments = _osa
        loads.append({'x': x.detach().clone(), 'assign': holder.get('a'),
                      'spec': self.wmb_spec, 'dp': self.data_preprocessor})
        return x, y, bw

    DataLoader.load = load2

    _orig_fn_init = FactorNN.__init__

    def fn_init(self, net, data_processor, losses=None, wmb_spec=None):
        _orig_fn_init(self, net, data_processor, losses=losses, wmb_spec=wmb_spec)
        factors.append(self)

    FactorNN.__init__ = fn_init

    cfg = dict(
        neurobe_mode=True, iB=args.ib, ecl=args.ecl,
        num_samples='nbe,0.1', sampling_scheme='uniform',
        stream_nn_exact=True, dope_factors=True,
        device='cuda', seed=42, approximation_method='nn',
        verbose_merge=False, num_epochs=args.num_epochs,
        compute_local_error=False,
        use_join_tree_merge=True, max_merge_bound=args.D,
    )
    if args.arm == 'input':
        cfg['wmb_input'] = 'combined'
    elif args.arm == 'parts':
        cfg['wmb_input'] = 'partitions'
    full = prepare_config(cfg, strict=False)

    model = get_catalog()[args.problem]
    gm = FastGM(model=model, nn_config=full, device=full['device'])
    gm.eliminate_variables(all=True)

    print(f"\n=== verifying {len(factors)} emitted NN factors ({args.arm}) ===")
    # Pair each factor with the loads that preceded it (clusters are sequential).
    n_bad = 0
    n_checked = 0
    li = 0
    for f in factors:
        # consume loads until the next factor; use the last one before this factor
        mine = []
        while li < len(loads):
            mine.append(loads[li])
            li += 1
            # heuristic: a cluster's loads all share the same spec object identity
            if li < len(loads) and loads[li]['spec'] is not mine[0]['spec'] \
                    and mine[0]['spec'] is not None:
                break
        if not mine:
            continue
        rec = mine[-1]
        assign = rec['assign']
        if assign is None:
            continue
        table = f.to_exact()
        table.order_indices()
        # index the dense table at the training assignments
        dom = [int(gm.matching_var(l).states) for l in table.labels]
        strides = []
        acc = 1
        for d in reversed(dom):
            strides.append(acc)
            acc *= d
        strides = list(reversed(strides))
        st = torch.tensor(strides, device=assign.device, dtype=torch.int64)
        flat = (assign.to(torch.int64) * st).sum(dim=1)
        eval_vals = table.tensor.reshape(-1)[flat]

        with torch.no_grad():
            direct = f.data_processor.undo_normalization(f.net(rec['x'])).reshape(-1)

        d = (eval_vals - direct).abs()
        fin = torch.isfinite(eval_vals) & torch.isfinite(direct)
        mx = float(d[fin].max()) if fin.any() else float('nan')
        rel = mx / max(1e-30, float(direct[fin].abs().max())) if fin.any() else float('nan')
        n_checked += 1
        ok = mx < 1e-3
        if not ok:
            n_bad += 1
        print(f"  scope={len(f.labels):2d} n={rec['x'].shape[0]:6d} "
              f"feat={0 if f.wmb_spec is None else f.wmb_spec.n_features} "
              f"max|eval-train| = {mx:.3e}  rel={rel:.2e}  {'OK' if ok else '*** MISMATCH ***'}")

    print(f"\nchecked {n_checked} clusters, {n_bad} mismatched")
    sys.exit(1 if n_bad else 0)


if __name__ == '__main__':
    main()
