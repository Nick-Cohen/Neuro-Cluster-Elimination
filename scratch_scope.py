"""Does the pre-merge message_scopes cache diverge from the live scope on the
problems the suites already use? Needed to keep the scope test non-vacuous."""
import contextlib, io
import torch
from nce.config_schema import prepare_config
from nce.inference.graphical_model import FastGM
from nce.benchmark_problems.catalog_utils import get_catalog
import nce.inference.bucket as bmod

torch.set_num_threads(1)

CASES = [
    ('dbn/rbm_20', 20, 'rnn8'),
]
ARMS = {
    'rnn4': dict(use_reduce_nn_merge=True, max_merge_bound=4, reduce_nn_backtrack=True),
    'rnn6': dict(use_reduce_nn_merge=True, max_merge_bound=6, reduce_nn_backtrack=True),
    'rnn8': dict(use_reduce_nn_merge=True, max_merge_bound=8, reduce_nn_backtrack=True),
    'jt8': dict(use_join_tree_merge=True, max_merge_bound=8),
}
cat = get_catalog()
for key, iB, arm in CASES:
    cfg = prepare_config(dict(neurobe_mode=True, ecl=1048577, sampling_scheme='uniform',
                              iB=iB, stream_nn_exact=True, device='cpu', seed=42,
                              dope_factors=True, num_epochs=1, num_samples=256,
                              verbose_merge=False, **ARMS[arm]), strict=False)
    div = []
    orig = bmod.FastBucket.compute_message_nn

    def probe(self, *a, **k):
        live = sorted(self.get_message_scope())
        cached = sorted(self.gm.message_scopes.get(self.label, []))
        n_elim = len(self.elim_vars)
        if live != cached:
            div.append((self.label, n_elim, sorted(set(live) - set(cached)),
                        sorted(set(cached) - set(live))))
        return orig(self, *a, **k)

    bmod.FastBucket.compute_message_nn = probe
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            gm = FastGM(model=cat[key], nn_config=cfg, device='cpu')
            gm.eliminate_variables(all=True)
    finally:
        bmod.FastBucket.compute_message_nn = orig
    print(f"{key} iB={iB} {arm}: {len(div)} NN buckets where live != pre-merge cache")
    for d in div[:6]:
        print(f"    bucket={d[0]} n_elim={d[1]} live-only={d[2]} cache-only={d[3]}")
