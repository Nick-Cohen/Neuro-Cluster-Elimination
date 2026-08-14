"""Is the memorization NR sampler's generator actually consumed?

_phase1_vectorized takes no rng; only the phase-2 frontier pass does. If phase 1
resolves the whole budget, the seed -- legacy OR CRN -- is inert.

Sweeps doc 50's own (sample_frac, mem_frac) points.
"""
import contextlib, io
import torch
from nce.config_schema import prepare_config
from nce.inference.graphical_model import FastGM
from nce.benchmark_problems.catalog_utils import get_catalog
from nce.neural_networks import memorization_table as mt

torch.set_num_threads(1)

BASE = dict(neurobe_mode=True, ecl=1025, sampling_scheme='uniform', iB=10,
            stream_nn_exact=True, device='cpu', seed=42, dope_factors=True,
            num_epochs=1, verbose_merge=False, num_samples=256,
            populate_bw_factors=True, bw_ecl=1025, use_memorization_table=True,
            use_reduce_nn_merge=True, max_merge_bound=4, reduce_nn_backtrack=True)

# doc 50 POINTS, sample_frac = 2 * mem_frac, plus doc 44's anchor (10:1)
POINTS = [(2 * mf, mf) for mf in (0.001, 0.005, 0.01, 0.025, 0.05, 0.1)]
POINTS.append((0.1, 0.01))

cat = get_catalog()
for key in ('grids/grid10x10.f10', 'grids/grid10x10.f10.wrap'):
    for sf, mf in POINTS:
        cfg = prepare_config(dict(BASE, memorize_sample_frac=sf,
                                  memorize_frac=mf), strict=False)
        used = []
        real = mt._crn.no_replacement_generator

        class _Shim:
            def __getattr__(self, n):
                return getattr(mt._crn, n)

            def no_replacement_generator(self, config, scope, doms, device,
                                         draw_index=0, role=None):
                g = real(config, scope, doms, device, draw_index, role)
                before = g.get_state().clone()
                used.append([before, g])
                return g

        shim = _Shim()
        old = mt._crn
        mt._crn = shim
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                gm = FastGM(model=cat[key], nn_config=cfg, device='cpu')
                gm.eliminate_variables(all=True)
        finally:
            mt._crn = old
        n_used = sum(1 for b, g in used if not torch.equal(b, g.get_state()))
        widths = [r['width'] for r in gm.memorization_log if r.get('ok')]
        nfill = sum(r.get('n_scope_vars_not_in_tree', 0)
                    for r in gm.memorization_log if r.get('ok'))
        print(f"{key:28s} sf={sf:<6g} mf={mf:<6g} clusters={len(used)} "
              f"rng_consumed={n_used} widths={widths} scope_vars_not_in_tree={nfill}")
