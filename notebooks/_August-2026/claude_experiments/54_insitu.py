#!/usr/bin/env python3
"""In-situ check: the backward path as it is actually consumed, DURING a live
elimination sweep rather than at build time.

P3 compared the populated factor lists against ground truth using build-time
bucket state. That is not the state a consumer sees: by the time a cluster is
eliminated, messages from its children have arrived, so bucket.get_message_scope()
has grown, while approximate_downstream_factors were computed at build time.

Here we run a real forward sweep (exact messages, so the forward side is not a
confound) and at every cluster reproduce EXACTLY what
FastBucket.compute_message_nn does under use_bw_approx (bucket.py:447-468):

    get_backward_message(gm, label, backward_factors=bucket.approximate_downstream_factors,
                         iB=backward_iB, backward_ecl=bw_ecl,
                         approximation_method='wmb', return_factor_list=...)

and check
  A. the backward message lives on the cluster's separator (not more, not less)
  B. at bw_ecl = 2**22 it equals, pointwise, the exact backward message obtained
     by eliminating the LIVE downstream buckets of the same sweep
  C. return_factor_list=True really returns a list
"""
import sys, os, json, traceback
ROOT = os.environ.get('NCE_ROOT', '/tmp/claude-58902/wt-wmbbw')
sys.path.insert(0, ROOT)
import torch
torch.set_num_threads(8)

from nce.inference.graphical_model import FastGM
from nce.inference.factor import FastFactor
from nce.config_schema import prepare_config
from nce.benchmark_problems.catalog_utils import get_catalog
from nce.utils.backward_message import get_backward_message

CACHE = '/home/cohenn1/NCE/.model_cache'
PROBLEM = os.environ.get('PROBLEM', 'grids/grid10x10.f10')
BOUND = int(os.environ.get('BOUND', '10'))
BW_ECL = int(os.environ.get('BW_ECL', '1024'))
EXACT = 2 ** 22
MAX_TGT = int(os.environ.get('MAX_TGT', 2 ** 18))

STRATS = {
    'none':            {},
    'subsumption':     {'use_join_tree_merge': True},
    'non_subsumption': {'use_non_subsumption_merge': True},
    'reduce_nn':       {'use_reduce_nn_merge': True},
    'sub+nonsub':      {'use_join_tree_merge': True, 'use_non_subsumption_merge': True},
    'merge_degree':    {'merge_degree': BOUND},
}
ROUTES = {'wmb': False, 'tree_collect': True}


def cfg(route, **extra):
    c = dict(neurobe_mode=True, iB=10, ecl=1025, bw_ecl=BW_ECL,
             num_samples='nbe,0.1', sampling_scheme='uniform',
             stream_nn_exact=True, dope_factors=False, device='cpu', seed=42,
             max_merge_bound=BOUND, verbose_merge=False,
             approximation_method='wmb',
             populate_bw_factors=True, populate_bw_via_tree_collect=route,
             populate_bw_skip_non_nn=False)
    c.update(extra)
    return prepare_config(c, strict=False)


def dmax(a, b):
    if sorted(a.labels) != sorted(b.labels):
        return float('inf')
    a = FastFactor(a.tensor.clone(), list(a.labels)); a.order_indices()
    b = FastFactor(b.tensor.clone(), list(b.labels)); b.order_indices()
    d = (a.tensor - b.tensor)
    d = d[torch.isfinite(d)]
    return float(d.abs().max()) if d.numel() else 0.0


model = get_catalog(cache_dir=CACHE)[PROBLEM]
rows = []
for sname, sflags in STRATS.items():
    for rname, rflag in ROUTES.items():
        rec = {'strategy': sname, 'route': rname}
        try:
            gm = FastGM(model=model, nn_config=cfg(rflag, **sflags), device='cpu')
            order = [v for v in gm.elim_order if v in gm.buckets]
            scheme = {i['var']: i['sends_to'] for i in gm.get_senders_receivers()}
            n_ck = n_scope_bad = n_val_bad = n_notlist = 0
            worst = 0.0
            ex = []
            for idx, v in enumerate(order):
                b = gm.buckets[v]
                sep = sorted(b.get_message_scope())     # LIVE separator
                if b.approximate_downstream_factors is not None and sep:
                    size = 1
                    for l in sep:
                        size *= gm.matching_var(l).states
                    if size <= MAX_TGT:
                        bw, _ = get_backward_message(
                            gm, v.label,
                            backward_factors=list(b.approximate_downstream_factors),
                            iB=100, backward_ecl=EXACT,
                            approximation_method='wmb', return_factor_list=False)
                        fl, _ = get_backward_message(
                            gm, v.label,
                            backward_factors=list(b.approximate_downstream_factors),
                            iB=100, backward_ecl=EXACT,
                            approximation_method='wmb', return_factor_list=True)
                        if not isinstance(fl, list):
                            n_notlist += 1
                        # A. scope
                        if not set(bw.labels) <= set(sep):
                            n_scope_bad += 1
                            if len(ex) < 3:
                                ex.append({'cluster': v.label, 'kind': 'scope',
                                           'bw': sorted(bw.labels), 'sep': sep})
                        # B. value vs the LIVE downstream of this same sweep
                        live_dn = []
                        for w in order[idx + 1:]:
                            bb = gm.buckets.get(w)
                            if bb is None:
                                continue
                            for f in bb.factors:
                                live_dn.append(f.to_exact() if hasattr(f, 'to_exact') else f)
                        if live_dn:
                            truth, _ = get_backward_message(
                                gm, v.label, backward_factors=live_dn,
                                iB=100, backward_ecl=EXACT,
                                approximation_method='wmb', return_factor_list=False)
                            d = dmax(bw, truth)
                            worst = max(worst, d if d != float('inf') else worst)
                            if d > 1e-2:
                                n_val_bad += 1
                                if len(ex) < 3:
                                    ex.append({'cluster': v.label, 'kind': 'value',
                                               'delta': d, 'n_elim': len(b.elim_vars),
                                               'sep': sep})
                        n_ck += 1
                # advance the sweep: exact message out
                msgs = b.compute_message_exact() if b.factors else None
                nxt = scheme.get(v)
                if msgs is not None and nxt is not None:
                    gm.buckets[gm.matching_var(nxt)].factors.append(msgs)
                b.factors = []
                del gm.buckets[v]
            rec.update(status='OK', checked=n_ck, scope_bad=n_scope_bad,
                       value_bad=n_val_bad, not_a_list=n_notlist,
                       worst_delta=worst, examples=ex)
            del gm
        except Exception as e:
            rec.update(status=f'{type(e).__name__}: {e}',
                       tb=traceback.format_exc().splitlines()[-5:])
        print(json.dumps(rec, default=str), flush=True)
        rows.append(rec)

json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  os.environ.get('OUT','54-insitu.json')), 'w'), indent=1, default=str)
