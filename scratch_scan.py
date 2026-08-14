"""Find a (problem, arm-pair) where a separator is SHARED but sits on a
DIFFERENT key variable -- the only configuration in which the legacy
`seed*1000003 + bucket.label` memorization seeding can be shown to break
pairing."""
import contextlib, io, itertools, sys
import torch
from nce.config_schema import prepare_config
from nce.inference.graphical_model import FastGM
from nce.benchmark_problems.catalog_utils import get_catalog
import nce.inference.bucket as bmod

torch.set_num_threads(1)

ARMS = {
    'none': {},
    'sub4': {'use_join_tree_merge': True, 'max_merge_bound': 4},
    'sub8': {'use_join_tree_merge': True, 'max_merge_bound': 8},
    'nonsub4': {'use_non_subsumption_merge': True, 'max_merge_bound': 4},
    'rnn4': {'use_reduce_nn_merge': True, 'max_merge_bound': 4, 'reduce_nn_backtrack': True},
    'rnn8': {'use_reduce_nn_merge': True, 'max_merge_bound': 8, 'reduce_nn_backtrack': True},
    'md4': {'merge_degree': 4},
}
PROBLEMS = [('grids/grid10x10.f10', 10, 1025),
            ('grids/grid10x10.f10.wrap', 10, 1025),
            ('pedigree/pedigree1', 8, 1025)]

cat = get_catalog()
for key, iB, ecl in PROBLEMS:
    tables = {}
    for arm, extra in ARMS.items():
        cfg = prepare_config(dict(neurobe_mode=True, ecl=ecl,
                                  sampling_scheme='uniform', iB=iB,
                                  stream_nn_exact=True, device='cpu', seed=42,
                                  dope_factors=True, num_epochs=1,
                                  num_samples=256, verbose_merge=False,
                                  **extra), strict=False)
        rec = {}
        orig = bmod.FastBucket.compute_message_nn

        def probe(self, *a, **k):
            rec[tuple(sorted(self.get_message_scope()))] = self.label
            return orig(self, *a, **k)

        bmod.FastBucket.compute_message_nn = probe
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                gm = FastGM(model=cat[key], nn_config=cfg, device='cpu')
                gm.eliminate_variables(all=True)
        except Exception as e:
            print(f"  {key} {arm}: {type(e).__name__}: {e}")
            continue
        finally:
            bmod.FastBucket.compute_message_nn = orig
        tables[arm] = rec
    print(f"\n### {key}  (NN clusters per arm: "
          f"{ {a: len(t) for a, t in tables.items()} })")
    hits = []
    for a, b in itertools.combinations(sorted(tables), 2):
        for sep in set(tables[a]) & set(tables[b]):
            if tables[a][sep] != tables[b][sep]:
                hits.append((a, b, len(sep), tables[a][sep], tables[b][sep], sep))
    if not hits:
        print("  no shared separator on different key vars")
    for h in hits[:8]:
        print(f"  HIT {h[0]} vs {h[1]}: |sep|={h[2]} labels {h[3]} vs {h[4]}")
