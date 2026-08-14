#!/usr/bin/env python
"""Doc 44: how much slower is a forward pass through the splice?

Isolates the inference-path cost of HybridMemorizerNet from everything else:
same base net, same 65 536-row one-hot batch (the MAX_QUERY_ROWS block size used
by factor_nn.py), with and without the table. gpu3 only.
"""
import os, sys, time
sys.path.insert(0, '/tmp/memthresh')
import torch

from nce.config_schema import prepare_config
from nce.inference.bucket import FastBucket
from nce.inference.factor import FastFactor
from nce.inference.graphical_model import FastGM
from nce.inference.elimination_order import wtminfill_order
from nce.neural_networks.net import Net
from nce.neural_networks.memorization_table import (
    HybridMemorizerNet, pack_assignments)

DEV = 'cuda'
ROWS = 65536
REPS = 30


def bench(w, k):
    """w = separator width (binary vars), k = memorized entries."""
    n = w + 1
    fs = [FastFactor(torch.rand(*([2] * n)).log10().to(DEV), list(range(n)))]
    cfg = prepare_config(dict(neurobe_mode=True, iB=3, ecl=8, device=DEV,
                              approximation_method='nn', seed=42, num_epochs=1,
                              num_samples=16, batch_size=8,
                              hidden_sizes=[3 * w, 3 * w], lower_dim=True,
                              dope_factors=False, verbose_merge=False),
                         strict=False)
    order = wtminfill_order(fs, variables_not_eliminated=[])
    gm = FastGM(factors=fs, elim_order=order, nn_config=cfg, device=DEV)
    bucket = FastBucket(gm, 0, fs, DEV, [gm.matching_var(0)])
    base = Net(bucket, hidden_sizes=[3 * w, 3 * w]).to(DEV)
    base.eval()

    g = torch.Generator(device='cpu').manual_seed(0)
    keys = torch.unique(torch.randint(0, 2 ** w, (k,), generator=g)).to(DEV)
    vals = torch.rand(keys.numel(), generator=g).to(DEV)
    hyb = HybridMemorizerNet(base, keys, vals, bucket, lower_dim=True).to(DEV)
    hyb.eval()

    x = (torch.rand(ROWS, w, generator=g) < 0.5).float().to(DEV)

    def timeit(net):
        with torch.no_grad():
            for _ in range(5):
                net(x)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(REPS):
                net(x)
            torch.cuda.synchronize()
            return (time.time() - t0) / REPS

    tb, th = timeit(base), timeit(hyb)
    print(f'w={w:2d} K={keys.numel():6d}  base {tb*1e3:7.3f} ms  '
          f'hybrid {th*1e3:7.3f} ms  ratio {th/tb:.3f}x  '
          f'(per 2^{w} message: {tb*2**w/ROWS:.3f}s -> {th*2**w/ROWS:.3f}s)',
          flush=True)
    return tb, th


if __name__ == '__main__':
    print(f'device={torch.cuda.get_device_name(0)} rows/call={ROWS} reps={REPS}')
    for w, k in [(20, 1049), (20, 5243), (20, 10486), (20, 26215), (20, 52429),
                 (20, 104858), (16, 66), (16, 328), (16, 655), (16, 1638),
                 (16, 3277), (16, 6554), (12, 5), (12, 21), (12, 41), (12, 103),
                 (12, 205), (12, 410)]:
        bench(w, k)
