#!/usr/bin/env python3
"""H1b / H4: hazards that only fire at CONSUME time, not build time.

  H1b  get_backward_message() builds a downstream FastGM from a deepcopy of
       gm.config. Does it re-merge?
  H4   get_wmb_message_gradient_factors() / get_wmb_message_gradient() likewise.
"""
import sys, os, json, traceback
ROOT = os.environ.get('NCE_ROOT', '/tmp/claude-58902/wt-wmbbw')
sys.path.insert(0, ROOT)
import torch
torch.set_num_threads(8)

import nce.inference.graphical_model as GMM
from nce.inference.graphical_model import FastGM
from nce.config_schema import prepare_config
from nce.benchmark_problems.catalog_utils import get_catalog

CACHE = '/home/cohenn1/NCE/.model_cache'
PROBLEM = 'grids/grid10x10.f10'
BOUND = int(os.environ.get('BOUND', '10'))

merge_log = []
for name in ('merge_join_tree', 'reduce_nn_merge', 'merge_non_subsumption', 'merge_by_degree'):
    orig = getattr(FastGM, name)

    def wrap(orig=orig, name=name):
        def f(self, *a, **kw):
            merge_log.append({'pass': name, 'gm': id(self)})
            return orig(self, *a, **kw)
        return f
    setattr(FastGM, name, wrap())

STRATS = {
    'subsumption':     {'use_join_tree_merge': True},
    'non_subsumption': {'use_non_subsumption_merge': True},
    'reduce_nn':       {'use_reduce_nn_merge': True},
    'merge_degree':    {'merge_degree': BOUND},
}


def cfg(**extra):
    c = dict(neurobe_mode=True, iB=10, ecl=1025, bw_ecl=1024,
             num_samples='nbe,0.1', sampling_scheme='uniform',
             stream_nn_exact=True, dope_factors=False, device='cpu', seed=42,
             max_merge_bound=BOUND, verbose_merge=False,
             approximation_method='nn',
             populate_bw_factors=True, populate_bw_via_tree_collect=True,
             populate_bw_skip_non_nn=False)
    c.update(extra)
    return prepare_config(c, strict=False)


model = get_catalog(cache_dir=CACHE)[PROBLEM]
out = []
for sname, sflags in STRATS.items():
    gm = FastGM(model=model, nn_config=cfg(**sflags), device='cpu')
    primary = id(gm)
    # pick a merged cluster that has populated downstream factors
    target = None
    for kv, b in gm.buckets.items():
        if b.approximate_downstream_factors and len(b.elim_vars) > 1:
            target = (kv, b)
            break
    if target is None:
        for kv, b in gm.buckets.items():
            if b.approximate_downstream_factors:
                target = (kv, b)
                break
    kv, b = target
    merge_log.clear()
    rec = {'strategy': sname, 'cluster': kv.label, 'n_elim': len(b.elim_vars)}
    try:
        from nce.utils.backward_message import get_backward_message
        fl, _ = get_backward_message(gm, kv.label,
                                     backward_factors=list(b.approximate_downstream_factors),
                                     iB=10, backward_ecl=1024,
                                     approximation_method='wmb', return_factor_list=True)
        rec['n_bw_factors'] = len(fl)
    except Exception as e:
        rec['error'] = f'{type(e).__name__}: {e}'
        rec['tb'] = traceback.format_exc().splitlines()[-8:]
    extra = [m for m in merge_log if m['gm'] != primary]
    rec['H1b_merges_on_downstream_gm'] = len(extra)
    rec['H1b_passes'] = sorted({m['pass'] for m in extra})
    out.append(rec)
    print(json.dumps(rec), flush=True)

    # H4: get_wmb_message_gradient_factors
    merge_log.clear()
    rec4 = {'strategy': sname, 'probe': 'get_wmb_message_gradient_factors'}
    try:
        from nce.inference.message_gradient_factors import get_wmb_message_gradient_factors
        scope = b.get_message_scope()
        facs = [f.to_exact() if hasattr(f, 'to_exact') else f
                for f in list(b.approximate_downstream_factors)]
        get_wmb_message_gradient_factors(facs, scope, gm.config)
        rec4['ok'] = True
    except Exception as e:
        rec4['error'] = f'{type(e).__name__}: {e}'
    extra4 = [m for m in merge_log if m['gm'] != primary]
    rec4['H4_merges_on_temp_gm'] = len(extra4)
    rec4['H4_passes'] = sorted({m['pass'] for m in extra4})
    out.append(rec4)
    print(json.dumps(rec4), flush=True)
    del gm

json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '54-p2b.json'), 'w'), indent=1)
