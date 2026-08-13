#!/usr/bin/env python
"""Q30/Q31 cost measurement: where does FactorNN._eval_elim_block's time go?

Times the REAL ``FactorNN._eval_elim_block`` (the routine every streaming path
funnels through) on gpu0, split into:
  * encode  -- column gather + torch.where + scatter one-hot  (the part a
               SHARED ENCODING across co-resident NN factors could amortise)
  * forward -- net(one_hot) + undo_normalization              (the part a
               CROSS-BLOCK CACHE could skip entirely)

Nothing is trained; a randomly-initialised net of the production shape is used
(forward cost is weight-independent). Uses ~1 GB of gpu0 for a few seconds.

Usage: CUDA_VISIBLE_DEVICES=0 python 49_bench_eval_block.py
"""
import sys, time, types
sys.path.insert(0, '/home/cohenn1/NCE-wt-survey')

import torch
from torch import nn
from nce.inference.factor_nn import FactorNN


def make_factor(n_labels, domain, hidden, device='cuda', lower_dim=True):
    """A FactorNN with only the fields _eval_elim_block reads."""
    f = FactorNN.__new__(FactorNN)
    f.labels = list(range(n_labels))
    f.domain_sizes = [domain] * n_labels
    f.gm = types.SimpleNamespace(lower_dim=lower_dim, device=device)
    in_dim = n_labels * (domain - 1 if lower_dim else domain)
    trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                          nn.Linear(hidden, hidden), nn.ReLU(),
                          nn.Linear(hidden, 1)).to(device)
    trunk.device = device
    f.net = trunk
    f.data_processor = types.SimpleNamespace(undo_normalization=lambda x: x)
    return f


def timeit(fn, reps=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps


def split_cost(f, A, B, device='cuda'):
    """Return (total_s, encode_s, forward_s) for one A x B block."""
    n_labels = len(f.labels)
    msg_scope = f.labels[:n_labels // 2]
    elim_labels = f.labels[n_labels // 2:]
    d = f.domain_sizes[0]
    assignments = torch.randint(0, d, (A, len(msg_scope)), device=device)
    coords = torch.randint(0, d, (B, len(elim_labels)), device=device)

    total = timeit(lambda: f._eval_elim_block(assignments, coords,
                                              None, elim_labels, msg_scope))

    # rebuild the encode-only half exactly as _eval_elim_block does
    offsets, oh_w, oh_lo = f._onehot_layout(device)
    col_src = [(True, i) for i in range(len(msg_scope))] + \
              [(False, i) for i in range(len(elim_labels))]
    is_msg_t, msg_src_t, elim_src_t = FactorNN._col_src_tensors(col_src, device)
    msg_cols = FactorNN._gather_cols(assignments, msg_src_t, A, n_labels) + offsets
    pdt = next(f.net.parameters()).dtype

    def encode():
        elim_cols = FactorNN._gather_cols(coords, elim_src_t, B, n_labels) + offsets
        cols = torch.where(is_msg_t, msg_cols.unsqueeze(0), elim_cols.unsqueeze(1))
        return f._one_hot_from_cols(cols, offsets, oh_w, oh_lo, pdt, device)

    enc = timeit(encode)
    oh = encode()
    fwd = timeit(lambda: f.net(oh))
    return total, enc, fwd


def main():
    print(f'device: {torch.cuda.get_device_name(0)}')
    print(f"{'n_lab':>6}{'dom':>5}{'hid':>5}{'A':>7}{'B':>9}{'rows':>11}"
          f"{'total_ms':>10}{'enc_ms':>9}{'fwd_ms':>9}{'enc%':>7}{'ns/row':>9}")
    # (n_labels, domain, hidden, A, B) -- shapes spanning the surveyed clusters
    cases = [
        (20,  2,  60,  512,  4096),
        (20,  2,  60,  512, 16384),
        (30,  2,  90,  512,  4096),
        (36,  2, 108,  256,  8192),
        (12,  5,  60,  512,  4096),
        (16,  5,  72,  256,  8192),
        (20,  2,  60,    1, 65536),   # _nn_factor_slice shape (S4): A=1
        (30,  2,  90,    1, 65536),
    ]
    for n_labels, dom, hid, A, B in cases:
        f = make_factor(n_labels, dom, hid)
        tot, enc, fwd = split_cost(f, A, B)
        rows = A * B
        print(f'{n_labels:>6}{dom:>5}{hid:>5}{A:>7}{B:>9}{rows:>11}'
              f'{tot*1e3:>10.2f}{enc*1e3:>9.2f}{fwd*1e3:>9.2f}'
              f'{100*enc/tot:>7.1f}{tot/rows*1e9:>9.2f}')
        del f
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
