#!/usr/bin/env python
"""Doc 63: is the net USING the WMB input feature?

If the net implemented residual learning it would satisfy, in normalised units,
    y_hat_norm = base_norm + correction        =>   d y_hat_norm / d base_norm = 1
because the feature was normalised with the TARGET's own affine map precisely so
that a unit coefficient reproduces the residual solution.

So measure that derivative directly, on the training inputs, per cluster:
  sens = mean over training rows of  d net(x) / d x[:, wmb_col]
Also report the first-layer weight norm on the WMB column against the median
one-hot column, since both live on the same 0/1-ish scale.

sens ~ 1  -> the net found residual learning and the loss is elsewhere
sens ~ 0  -> the net is IGNORING a feature it should be leaning on
0 < sens < 1 -> partial trust, which is what the design was supposed to buy
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--problem', required=True)
ap.add_argument('--arm', default='input', choices=['input', 'parts'])
ap.add_argument('--num-epochs', type=int, default=2000)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--out', required=True)
a = ap.parse_args()

from nce.config_schema import prepare_config
from nce.inference.graphical_model import FastGM
from nce.benchmark_problems.catalog_utils import get_catalog
from nce.data.data_loader import DataLoader
from nce.inference.factor_nn import FactorNN

rows, pending = [], {}

_orig_load = DataLoader.load
def load(self, num_samples=0, all=False, is_validation=False):
    x, y, bw = _orig_load(self, num_samples=num_samples, all=all, is_validation=is_validation)
    if self.wmb_spec is not None:
        pending['x'] = x.detach().clone()
        pending['nf'] = self.wmb_spec.n_features
    return x, y, bw
DataLoader.load = load

_orig_init = FactorNN.__init__
def init(self, net, dp, losses=None, wmb_spec=None):
    _orig_init(self, net, dp, losses=losses, wmb_spec=wmb_spec)
    if wmb_spec is None or 'x' not in pending:
        return
    x = pending['x']; nf = pending['nf']
    net.eval()
    xg = x.clone().requires_grad_(True)
    out = net(xg).sum()
    g, = torch.autograd.grad(out, xg)
    sens = g[:, -nf:]                      # d out / d each wmb feature column
    W = None
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            W = m.weight; break            # first layer
    wn_wmb = W[:, -nf:].norm(dim=0)
    wn_oh = W[:, :-nf].norm(dim=0)
    rows.append({
        'bucket': int(net.bucket.label),
        'scope': int(len(net.bucket.get_message_scope())),
        'n_features': int(nf),
        'n_rows': int(x.shape[0]),
        # combined column is index 0; for 'parts' the partition columns follow
        'sens_combined_mean': float(sens[:, 0].mean()),
        'sens_combined_sd': float(sens[:, 0].std()),
        'sens_total_mean': float(sens.sum(dim=1).mean()),
        'w_norm_wmb_col0': float(wn_wmb[0]),
        'w_norm_onehot_median': float(wn_oh.median()),
        'w_ratio': float(wn_wmb[0] / wn_oh.median()),
    })
    pending.clear()
FactorNN.__init__ = init

cfg = dict(neurobe_mode=True, iB=10, ecl=1025, num_samples='nbe,0.1',
           sampling_scheme='uniform', stream_nn_exact=True, dope_factors=True,
           device='cuda', seed=a.seed, approximation_method='nn', verbose_merge=False,
           num_epochs=a.num_epochs, compute_local_error=False,
           use_join_tree_merge=True, max_merge_bound=10,
           wmb_input='combined' if a.arm == 'input' else 'partitions')
full = prepare_config(cfg, strict=False)
gm = FastGM(model=get_catalog()[a.problem], nn_config=full, device=full['device'])
gm.eliminate_variables(all=True)

Path(a.out).write_text(json.dumps({'problem': a.problem, 'arm': a.arm,
                                   'seed': a.seed, 'clusters': rows}, indent=1))
import statistics as st
s = [r['sens_combined_mean'] for r in rows]
w = [r['w_ratio'] for r in rows]
print(f"\n=== {a.problem} {a.arm} seed{a.seed}: {len(rows)} clusters ===")
print(f"  d(out)/d(base_norm):  median {st.median(s):+.4f}   min {min(s):+.4f}   max {max(s):+.4f}")
print(f"  (residual learning would be exactly +1.0000)")
print(f"  |W[:,wmb]| / median |W[:,onehot]|:  median {st.median(w):.3f}")
for r in rows:
    print(f"   bucket {r['bucket']:4d} scope={r['scope']:2d} sens={r['sens_combined_mean']:+.4f} "
          f"(sd {r['sens_combined_sd']:.4f}) w_ratio={r['w_ratio']:.2f}")
