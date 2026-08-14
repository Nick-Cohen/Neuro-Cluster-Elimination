#!/usr/bin/env python3
"""Systematic audit of WMB backward approximation over the cluster tree.

Three phases, all CPU, build-only unless stated:

  P1 BUILD MATRIX   6 merge strategies x 2 population routes x merge bounds.
  P2 HAZARD SCAN    the four known-hazard classes, instrumented:
                      H1 temp/copied GMs that re-merge
                      H2 scope caches read after merging changed the scope
                      H3 per-variable bookkeeping vs multi-elim-var clusters
                      H4 bucket-label keying that merging perturbs
  P3 CORRECTNESS    independent ground-truth upstream/downstream per cluster,
                    derived from the merged bucket tree + original factors only,
                    compared POINTWISE against what the populators produced.

Usage: 54_probe.py <phase> [problem] [iB] [ecl] [bw_ecl] [bound]
"""
import sys, os, json, traceback, itertools

ROOT = os.environ.get('NCE_ROOT', '/tmp/claude-58902/wt-wmbbw')
sys.path.insert(0, ROOT)

import torch
torch.set_num_threads(8)

from nce.inference.graphical_model import FastGM
from nce.inference.factor import FastFactor
from nce.inference.elimination_order import wtminfill_order
from nce.config_schema import prepare_config

CACHE = '/home/cohenn1/NCE/.model_cache'

PHASE   = sys.argv[1] if len(sys.argv) > 1 else 'p1'
PROBLEM = sys.argv[2] if len(sys.argv) > 2 else 'grids/grid10x10.f10'
IB      = int(sys.argv[3]) if len(sys.argv) > 3 else 10
ECL     = int(sys.argv[4]) if len(sys.argv) > 4 else 1025
BW_ECL  = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
BOUND   = int(sys.argv[6]) if len(sys.argv) > 6 else 10

MERGE_STRATEGIES = {
    'none':            {},
    'subsumption':     {'use_join_tree_merge': True},
    'non_subsumption': {'use_non_subsumption_merge': True},
    'reduce_nn':       {'use_reduce_nn_merge': True},
    'sub+nonsub':      {'use_join_tree_merge': True, 'use_non_subsumption_merge': True},
    'merge_degree':    {'merge_degree': None},   # filled with the bound
}
POP_ROUTES = {
    'wmb':          {'populate_bw_factors': True, 'populate_bw_via_tree_collect': False},
    'tree_collect': {'populate_bw_factors': True, 'populate_bw_via_tree_collect': True},
}


def load_model(key=None):
    from nce.benchmark_problems.catalog_utils import get_catalog
    cat = get_catalog(cache_dir=CACHE)
    return cat[key or PROBLEM]


def grid_factors(n=5, seed=0):
    """Self-contained n x n binary grid, log10 space (same as the tracked test)."""
    g = torch.Generator().manual_seed(seed)
    factors = []
    lab = lambda r, c: r * n + c
    for r in range(n):
        for c in range(n):
            factors.append(FastFactor(torch.rand(2, generator=g).log10(), [lab(r, c)]))
            if c + 1 < n:
                factors.append(FastFactor(torch.rand(2, 2, generator=g).log10(),
                                          [lab(r, c), lab(r, c + 1)]))
            if r + 1 < n:
                factors.append(FastFactor(torch.rand(2, 2, generator=g).log10(),
                                          [lab(r, c), lab(r + 1, c)]))
    return factors


def base_config(bound, **extra):
    cfg = dict(
        neurobe_mode=True, iB=IB, ecl=ECL, bw_ecl=BW_ECL,
        num_samples='nbe,0.1', sampling_scheme='uniform',
        stream_nn_exact=True, dope_factors=False, device='cpu', seed=42,
        max_merge_bound=bound, verbose_merge=False,
        approximation_method='nn',
        populate_bw_skip_non_nn=(os.environ.get('SKIP_NON_NN', '1') == '1'),
    )
    cfg.update(extra)
    return prepare_config(cfg, strict=False)


def make_cfg(strat, route, bound):
    extra = dict(MERGE_STRATEGIES[strat])
    if 'merge_degree' in extra and extra['merge_degree'] is None:
        extra['merge_degree'] = bound
    extra.update(POP_ROUTES[route])
    return base_config(bound, **extra)


def build(strat, route, bound, model=None, factors=None, elim_order=None):
    cfg = make_cfg(strat, route, bound)
    if factors is not None:
        return FastGM(factors=[FastFactor(f.tensor.clone(), list(f.labels)) for f in factors],
                      elim_order=elim_order, nn_config=cfg, device='cpu')
    return FastGM(model=model, nn_config=cfg, device='cpu')


# ---------------------------------------------------------------- P1
def phase1():
    model = load_model()
    bounds = [int(b) for b in os.environ.get('BOUNDS', '10,16,24').split(',')]
    rows = []
    for bound in bounds:
        for strat in MERGE_STRATEGIES:
            for route in POP_ROUTES:
                rec = {'bound': bound, 'strategy': strat, 'route': route}
                try:
                    gm = build(strat, route, bound, model=model)
                    n_clusters = len(gm.buckets)
                    n_merged = sum(1 for b in gm.buckets.values() if len(b.elim_vars) > 1)
                    max_elim = max((len(b.elim_vars) for b in gm.buckets.values()), default=0)
                    pop_up = sum(1 for b in gm.buckets.values()
                                 if b.approximate_upstream_factors is not None)
                    pop_dn = sum(1 for b in gm.buckets.values()
                                 if b.approximate_downstream_factors is not None)
                    rec.update(status='OK', clusters=n_clusters, merged=n_merged,
                               max_elim=max_elim, pop_up=pop_up, pop_dn=pop_dn)
                    # every populated cluster's upstream must cover its elim vars
                    bad_up = []
                    for kv, b in gm.buckets.items():
                        if b.approximate_upstream_factors is None:
                            continue
                        have = set()
                        for f in b.approximate_upstream_factors:
                            have.update(f.labels)
                        miss = {getattr(v, 'label', v) for v in b.elim_vars} - have
                        if miss:
                            bad_up.append((kv.label, sorted(miss)))
                    rec['bad_upstream'] = len(bad_up)
                    rec['bad_upstream_ex'] = bad_up[:3]
                    del gm
                except Exception as e:
                    rec.update(status=f'{type(e).__name__}: {e}',
                               tb=traceback.format_exc().splitlines()[-4:])
                print(json.dumps(rec), flush=True)
                rows.append(rec)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '54-p1.json')
    json.dump(rows, open(out, 'w'), indent=1)
    ok = sum(1 for r in rows if r['status'] == 'OK')
    print(f"\nBUILD MATRIX: {ok}/{len(rows)} OK")
    print(f"bad_upstream nonzero: {sum(1 for r in rows if r.get('bad_upstream'))}")


# ---------------------------------------------------------------- P2
def phase2():
    """Hazard scan. Instruments FastGM to record every merge pass that fires,
    tagged with whether the GM is the primary one."""
    import nce.inference.graphical_model as GMM

    merge_log = []
    for name in ('merge_join_tree', 'reduce_nn_merge', 'merge_non_subsumption',
                 'merge_by_degree'):
        orig = getattr(GMM.FastGM, name)

        def wrap(orig=orig, name=name):
            def f(self, *a, **kw):
                merge_log.append({'pass': name, 'gm': id(self),
                                  'primary': getattr(self, 'is_primary', True),
                                  'populating': getattr(self, 'is_populating_backward_factors', False)})
                return orig(self, *a, **kw)
            return f
        setattr(GMM.FastGM, name, wrap())

    model = load_model()
    findings = []

    for strat in ('reduce_nn', 'subsumption', 'non_subsumption', 'merge_degree'):
        for route in ('wmb', 'tree_collect'):
            merge_log.clear()
            gm = build(strat, route, BOUND, model=model)
            primary_id = id(gm)
            extra = [m for m in merge_log if m['gm'] != primary_id]
            findings.append({'phase': 'H1-build', 'strategy': strat, 'route': route,
                             'merges_on_temp_gms': len(extra),
                             'detail': extra[:5]})
            print(json.dumps(findings[-1]), flush=True)

            # ---- H2: is the build-time separator equal to the true one? ----
            # true separator = bucket.get_message_scope() at the moment the
            # cluster is actually processed in the forward sweep.
            sep_at_build = {}
            for kv, b in gm.buckets.items():
                sep_at_build[kv.label] = set(gm._cluster_separator(kv, b))
            h2 = []
            gm2 = build(strat, route, BOUND, model=model)
            # walk the tree the way elimination does, recording true scopes
            order = [v for v in gm2.elim_order if v in gm2.buckets]
            scheme = {i['var']: i['sends_to'] for i in gm2.get_senders_receivers()}
            for v in order:
                b = gm2.buckets[v]
                true_scope = set(b.get_message_scope())
                built = sep_at_build.get(v.label)
                if built is not None and not true_scope <= built:
                    h2.append({'cluster': v.label,
                               'missing_from_build_scope': sorted(true_scope - built),
                               'built': sorted(built), 'true': sorted(true_scope)})
                # push the message onward (scope only; use a stub factor)
                nxt = scheme.get(v)
                if nxt is not None and true_scope:
                    stub = FastFactor(torch.zeros(*[gm2.matching_var(l).states
                                                    for l in sorted(true_scope)]),
                                      sorted(true_scope))
                    gm2.buckets[gm2.matching_var(nxt)].factors.append(stub)
                b.factors = []
            findings.append({'phase': 'H2-scope-cache', 'strategy': strat, 'route': route,
                             'n_underestimated': len(h2), 'examples': h2[:3]})
            print(json.dumps(findings[-1]), flush=True)
            del gm, gm2

    # ---- H3: use_bw_approx WITHOUT populate_bw_factors, under merging ----
    for strat in ('reduce_nn', 'subsumption', 'merge_degree'):
        rec = {'phase': 'H3-bw-approx-no-populate', 'strategy': strat}
        try:
            extra = dict(MERGE_STRATEGIES[strat])
            if extra.get('merge_degree', 'x') is None:
                extra['merge_degree'] = BOUND
            cfg = base_config(BOUND, use_bw_approx=True, populate_bw_factors=False, **extra)
            gm = build.__wrapped__ if False else None
            gm = FastGM(model=model, nn_config=cfg, device='cpu')
            # emulate exactly what compute_message_nn does when there are no
            # pre-populated factors: get_backward_message(gm, label, None)
            from nce.utils.backward_message import _get_backward_factors
            first = [v for v in gm.elim_order if v in gm.buckets][3]
            _get_backward_factors(gm, first.label, None)
            rec['result'] = 'no error'
        except Exception as e:
            rec['result'] = f'{type(e).__name__}: {e}'
        findings.append(rec)
        print(json.dumps(rec), flush=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '54-p2.json')
    json.dump(findings, open(out, 'w'), indent=1)


# ---------------------------------------------------------------- P3
EXACT_BW_ECL = 2 ** 22


def _exact_elim_to(factors, target_scope, device='cpu'):
    """Exact elimination of `factors` down to `target_scope`; returns FastFactor."""
    factors = [f for f in factors]
    scal = torch.tensor(0.0, device=device)
    ns = []
    for f in factors:
        (ns.append(f) if f.labels else None)
        if not f.labels:
            scal = scal + f.tensor.squeeze()
    if not ns:
        return FastFactor(scal, [])
    allv = set()
    for f in ns:
        allv.update(f.labels)
    tgt = [l for l in target_scope if l in allv]
    if not (allv - set(tgt)):
        prod = ns[0]
        for f in ns[1:]:
            prod = prod * f
        return FastFactor(prod.tensor + scal, prod.labels)
    order = wtminfill_order(ns, variables_not_eliminated=tgt)
    cfg = prepare_config(dict(neurobe_mode=True, iB=64, ecl=EXACT_BW_ECL,
                              bw_ecl=EXACT_BW_ECL, device=device, seed=0,
                              approximation_method='wmb',
                              populate_bw_factors=False,
                              num_samples=1, sampling_scheme='uniform'), strict=False)
    g = FastGM(factors=ns, elim_order=order, nn_config=cfg, device=device)
    g.is_primary = False
    g.eliminate_variables(all_but=tgt)
    rest = g.get_all_factors()
    if not rest:
        return FastFactor(scal, [])
    prod = rest[0]
    for f in rest[1:]:
        prod = prod * f
    return FastFactor(prod.tensor + scal, prod.labels)


def phase3():
    """Independent ground truth for upstream/downstream, pointwise comparison."""
    n = int(os.environ.get('GRID_N', '5'))
    use_catalog = os.environ.get('P3_CATALOG', '0') == '1'
    bounds = [int(b) for b in os.environ.get('BOUNDS', '4,16').split(',')]
    rows = []

    for bound in bounds:
        for strat in MERGE_STRATEGIES:
            if strat == 'none':
                continue
            for route in POP_ROUTES:
                rec = {'bound': bound, 'strategy': strat, 'route': route}
                try:
                    if use_catalog:
                        model = load_model()
                        cfgk = dict(model=model)
                    else:
                        fl = grid_factors(n)
                        cfgk = dict(factors=fl,
                                    elim_order=wtminfill_order(fl, variables_not_eliminated=[]))
                    cfg = make_cfg(strat, route, bound)
                    cfg['bw_ecl'] = EXACT_BW_ECL          # make the populators exact
                    cfg['populate_bw_skip_non_nn'] = False
                    gm = FastGM(nn_config=cfg, device='cpu', **cfgk)

                    # --- ground truth from the merged tree + originals only ---
                    originals = {kv.label: list(b.factors) for kv, b in gm.buckets.items()}
                    scheme = {i['var'].label: i['sends_to'] for i in gm.get_senders_receivers()}
                    children = {k: [] for k in originals}
                    for c, p in scheme.items():
                        if p is not None and p in children:
                            children[p].append(c)

                    def subtree(k):
                        out, stack = set(), [k]
                        while stack:
                            x = stack.pop()
                            if x in out:
                                continue
                            out.add(x)
                            stack.extend(children.get(x, []))
                        return out

                    n_up_bad = n_dn_bad = n_checked = 0
                    worst_up = worst_dn = 0.0
                    ex = []
                    for kv, b in gm.buckets.items():
                        if b.approximate_downstream_factors is None:
                            continue
                        k = kv.label
                        st = subtree(k)
                        up_truth, dn_truth = [], []
                        for kk, fl in originals.items():
                            (up_truth if kk in st else dn_truth).extend(fl)
                        elim = sorted({getattr(v, 'label', v) for v in b.elim_vars})
                        sep = sorted(set(gm._cluster_separator(kv, b)))
                        tgt = sorted(set(sep) | set(elim))
                        size = 1
                        for l in tgt:
                            size *= gm.matching_var(l).states
                        if size > int(os.environ.get('MAX_TGT', 2 ** 18)):
                            continue

                        code_dn = _exact_elim_to(b.approximate_downstream_factors, tgt)
                        true_dn = _exact_elim_to(dn_truth, tgt)
                        d_dn = _cmp(code_dn, true_dn)

                        code_up = _exact_elim_to(b.approximate_upstream_factors or [], tgt)
                        true_up = _exact_elim_to(up_truth, tgt)
                        d_up = _cmp(code_up, true_up)

                        n_checked += 1
                        worst_dn = max(worst_dn, d_dn)
                        worst_up = max(worst_up, d_up)
                        if d_dn > 1e-3:
                            n_dn_bad += 1
                            if len(ex) < 4:
                                ex.append({'cluster': k, 'kind': 'downstream', 'delta': d_dn,
                                           'n_elim': len(elim), 'sep': len(sep),
                                           'code_labels': sorted(code_dn.labels),
                                           'true_labels': sorted(true_dn.labels)})
                        if d_up > 1e-3:
                            n_up_bad += 1
                            if len(ex) < 4:
                                ex.append({'cluster': k, 'kind': 'upstream', 'delta': d_up,
                                           'n_elim': len(elim), 'sep': len(sep),
                                           'code_labels': sorted(code_up.labels),
                                           'true_labels': sorted(true_up.labels)})
                    rec.update(status='OK', checked=n_checked,
                               up_mismatch=n_up_bad, dn_mismatch=n_dn_bad,
                               worst_up=worst_up, worst_dn=worst_dn, examples=ex)
                    del gm
                except Exception as e:
                    rec.update(status=f'{type(e).__name__}: {e}',
                               tb=traceback.format_exc().splitlines()[-5:])
                print(json.dumps(rec, default=str), flush=True)
                rows.append(rec)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '54-p3.json')
    json.dump(rows, open(out, 'w'), indent=1, default=str)


def _cmp(a, b):
    """max |a - b| after aligning labels; +inf if label sets differ."""
    if sorted(a.labels) != sorted(b.labels):
        return float('inf')
    if not a.labels:
        return float(abs(a.tensor.squeeze() - b.tensor.squeeze()))
    aa = FastFactor(a.tensor.clone(), list(a.labels)); aa.order_indices()
    bb = FastFactor(b.tensor.clone(), list(b.labels)); bb.order_indices()
    d = (aa.tensor - bb.tensor)
    d = d[torch.isfinite(d)]
    if d.numel() == 0:
        return 0.0
    return float(d.abs().max())


if __name__ == '__main__':
    {'p1': phase1, 'p2': phase2, 'p3': phase3}[PHASE]()
