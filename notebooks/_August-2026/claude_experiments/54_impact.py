#!/usr/bin/env python3
"""Numerical consequence of the H1b defect: get_backward_message's downstream
GM re-merged, which changes which variables WMB partitions over at backward_ecl
and therefore changes the VALUE of the backward message.

Measured by computing the backward message twice from the SAME populated
approximate_downstream_factors and the SAME backward_ecl -- once with the merge
flags left on in the downstream config (pre-fix behaviour) and once with them
off (post-fix) -- and reporting max |delta log10| over the message tensor.
Also reports each against the bw_ecl = 2**22 (exact) reference.
"""
import sys, os, json, copy, traceback
ROOT = os.environ.get('NCE_ROOT', '/tmp/claude-58902/wt-wmbbw')
sys.path.insert(0, ROOT)
import torch
torch.set_num_threads(8)

from nce.inference.graphical_model import FastGM
from nce.inference.factor import FastFactor
from nce.config_schema import prepare_config
from nce.benchmark_problems.catalog_utils import get_catalog
import nce.utils.backward_message as BM

CACHE = '/home/cohenn1/NCE/.model_cache'
PROBLEM = os.environ.get('PROBLEM', 'grids/grid10x10.f10')
BOUND = int(os.environ.get('BOUND', '10'))
BW_ECL = int(os.environ.get('BW_ECL', '1024'))

STRATS = {
    'subsumption':     {'use_join_tree_merge': True},
    'non_subsumption': {'use_non_subsumption_merge': True},
    'reduce_nn':       {'use_reduce_nn_merge': True},
    'sub+nonsub':      {'use_join_tree_merge': True, 'use_non_subsumption_merge': True},
    'merge_degree':    {'merge_degree': BOUND},
}

_orig_get = BM.get_backward_message
_SRC = open(BM.__file__).read()
REMERGE = {'on': True, 'off': False}


def cfg(**extra):
    c = dict(neurobe_mode=True, iB=10, ecl=1025, bw_ecl=BW_ECL,
             num_samples='nbe,0.1', sampling_scheme='uniform',
             stream_nn_exact=True, dope_factors=False, device='cpu', seed=42,
             max_merge_bound=BOUND, verbose_merge=False,
             approximation_method='nn',
             populate_bw_factors=True, populate_bw_via_tree_collect=True,
             populate_bw_skip_non_nn=False)
    c.update(extra)
    return prepare_config(c, strict=False)


def bw_msg(gm, bucket, ecl, allow_remerge):
    """Reproduces get_backward_message's downstream elimination, with the
    pre-fix (allow_remerge=True) or post-fix (False) downstream config."""
    from nce.inference.elimination_order import wtminfill_order
    import math
    facs = [f.to_exact() if hasattr(f, 'to_exact') else f
            for f in bucket.approximate_downstream_factors]
    scope = bucket.get_message_scope()
    if not scope:
        return None
    scal = torch.tensor(0.0)
    ns = []
    for f in facs:
        if f.labels:
            ns.append(f)
        else:
            scal = scal + f.tensor.squeeze()
    if not ns:
        return None
    order = wtminfill_order(ns, variables_not_eliminated=scope)
    dc = copy.deepcopy(gm.config)
    dc['populate_bw_factors'] = False
    dc['approximation_method'] = 'wmb'
    dc['ecl'] = ecl
    dc['iB'] = int(math.log2(ecl)) if ecl > 0 else 0
    if not allow_remerge:
        for f in ('use_join_tree_merge', 'use_reduce_nn_merge',
                  'use_non_subsumption_merge'):
            dc[f] = False
        dc['merge_degree'] = 0
    g = FastGM(factors=ns, elim_order=order, reference_fastgm=gm,
               device=gm.device, nn_config=dc)
    g.is_primary = False
    g.eliminate_variables(all_but=scope)
    rest = g.get_all_factors()
    if not rest:
        return None
    p = rest[0]
    for f in rest[1:]:
        p = p * f
    out = FastFactor(p.tensor + scal, p.labels)
    out.order_indices()
    return out


def dmax(a, b):
    if a is None or b is None:
        return None
    if sorted(a.labels) != sorted(b.labels):
        return float('inf')
    d = a.tensor - b.tensor
    d = d[torch.isfinite(d)]
    return float(d.abs().max()) if d.numel() else 0.0


model = get_catalog(cache_dir=CACHE)[PROBLEM]
rows = []
for sname, sflags in STRATS.items():
    gm = FastGM(model=model, nn_config=cfg(**sflags), device='cpu')
    for kv, b in gm.buckets.items():
        if not b.approximate_downstream_factors:
            continue
        rec = {'strategy': sname, 'cluster': kv.label, 'n_elim': len(b.elim_vars),
               'sep': len(b.get_message_scope())}
        try:
            on = bw_msg(gm, b, BW_ECL, True)
            off = bw_msg(gm, b, BW_ECL, False)
            exact = bw_msg(gm, b, 2 ** 22, False)
            rec['delta_on_vs_off'] = dmax(on, off)
            rec['err_on_vs_exact'] = dmax(on, exact)
            rec['err_off_vs_exact'] = dmax(off, exact)
        except Exception as e:
            rec['error'] = f'{type(e).__name__}: {e}'
        rows.append(rec)
        print(json.dumps(rec), flush=True)
    del gm

json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '54-impact.json'), 'w'), indent=1)
d = [r['delta_on_vs_off'] for r in rows if r.get('delta_on_vs_off') is not None]
print(f"\nclusters compared: {len(d)}  nonzero delta: {sum(1 for x in d if x > 1e-4)}  "
      f"max delta: {max(d) if d else 0}")
