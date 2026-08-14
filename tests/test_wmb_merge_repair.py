"""Regression tests for WMB backward-factor population under bucket merging.

Failure being pinned (all four reproduced on grids/grid10x10.f10, iB=10,
ecl=1025, max_merge_bound=10, CPU, build only — see
notebooks/_August-2026/claude_experiments/10-wmb-merge-repair.md):

1. `_create_population_copy` disabled only `use_join_tree_merge`, while
   `FastGM.__init__` dispatches four independent merge passes, so under
   reduce-NN / non-subsumption / merge_degree the population copy re-merged and
   the copy-side lookups (which assume one elim var per bucket) raised KeyError.
2. `get_senders_receivers` raw-indexed `self.buckets[var]` while walking the full
   `self.elim_order`; absorbed buckets are removed from `self.buckets` but stay
   in `elim_order`. It also discarded only the key var from the outgoing scope.
3. A merged cluster's `approximate_upstream_factors` were read off the key var's
   single-variable copy bucket, in which the absorbed members' variables had
   already been summed out.
4. `build_proposal_for_bucket` derived a merged cluster's proposal scope by
   unioning the whole backward chain (measured 14/16/17 variables against true
   separators of 1/1/0).

These tests are build-only (no elimination, no NN training) and run on CPU.
"""
import itertools

import pytest
import torch

from nce.config_schema import prepare_config
from nce.inference.factor import FastFactor
from nce.inference.graphical_model import FastGM
from nce.sampling.proposal_sampler import proposal_scope_for_bucket


# Each merge strategy FastGM.__init__ can dispatch. reduce-NN is the acceptance
# criterion; the rest are included because they share the same defects.
MERGE_STRATEGIES = {
    'none': {},
    'subsumption': {'use_join_tree_merge': True},
    'non_subsumption': {'use_non_subsumption_merge': True},
    'reduce_nn': {'use_reduce_nn_merge': True},
    'sub_plus_nonsub': {'use_join_tree_merge': True, 'use_non_subsumption_merge': True},
    'merge_degree': {'merge_degree': 4},
}
POPULATION_ROUTES = {
    'wmb': False,
    'tree_collect': True,
}
CASES = list(itertools.product(sorted(MERGE_STRATEGIES), sorted(POPULATION_ROUTES)))


def _grid_factors(n=5, seed=0):
    """n x n binary grid with random pairwise factors, in log10 space."""
    g = torch.Generator().manual_seed(seed)
    factors = []
    label = lambda r, c: r * n + c
    for r in range(n):
        for c in range(n):
            factors.append(FastFactor(torch.rand(2, generator=g).log10(), [label(r, c)]))
            if c + 1 < n:
                factors.append(FastFactor(torch.rand(2, 2, generator=g).log10(),
                                          [label(r, c), label(r, c + 1)]))
            if r + 1 < n:
                factors.append(FastFactor(torch.rand(2, 2, generator=g).log10(),
                                          [label(r, c), label(r + 1, c)]))
    return factors


def _build(merge_flags, via_tree_collect, iB=3, ecl=8, bw_ecl=256):
    cfg = prepare_config(dict(
        neurobe_mode=False, iB=iB, ecl=ecl, bw_ecl=bw_ecl,
        device='cpu', approximation_method='nn', seed=42,
        loss_fn='logspace_mse_fdb', loss_fn2='',
        num_epochs=1, num_samples=32, batch_size=16, hidden_sizes=[],
        dope_factors=False, verbose_merge=False,
        max_merge_bound=4,
        populate_bw_factors=True,
        populate_bw_via_tree_collect=via_tree_collect,
        populate_bw_skip_non_nn=False,
        **merge_flags), strict=False)
    from nce.inference.elimination_order import wtminfill_order
    factors = _grid_factors()
    order = wtminfill_order(factors, variables_not_eliminated=[])
    return FastGM(factors=factors, elim_order=order, nn_config=cfg, device='cpu')


@pytest.fixture(scope='module')
def built():
    """Build every (merge strategy, population route) combination once."""
    out = {}
    for merge_name, route_name in CASES:
        out[(merge_name, route_name)] = _build(MERGE_STRATEGIES[merge_name],
                                               POPULATION_ROUTES[route_name])
    return out


@pytest.mark.parametrize('merge_name,route_name', CASES)
def test_build_succeeds_under_every_merge_strategy(built, merge_name, route_name):
    """Defects 1 and 2: construction used to raise KeyError on absorbed vars."""
    gm = built[(merge_name, route_name)]
    assert len(gm.buckets) > 0
    # get_senders_receivers must survive the merged tree too (it is what raised).
    scheme = gm.get_senders_receivers()
    assert len(scheme) == len(gm.buckets)


@pytest.mark.parametrize('merge_name,route_name', CASES)
def test_population_copy_is_never_merged(built, merge_name, route_name):
    """Defect 1: the copy must be a plain one-var-per-bucket tree."""
    gm = built[(merge_name, route_name)]
    copy_gm = gm._create_population_copy()
    assert len(copy_gm.buckets) == len(copy_gm.elim_order)
    assert all(len(b.elim_vars) == 1 for b in copy_gm.buckets.values())


@pytest.mark.parametrize('merge_name,route_name', CASES)
def test_cluster_upstream_covers_all_elim_vars(built, merge_name, route_name):
    """Defect 3: every variable the cluster eliminates must still be present.

    The old code handed a merged cluster the key var's copy bucket, in which the
    absorbed members' variables had already been summed out.
    """
    gm = built[(merge_name, route_name)]
    merged = [b for b in gm.buckets.values() if len(b.elim_vars) > 1]
    if not merged:
        pytest.skip(f'{merge_name} produced no merged clusters on this problem')
    for b in merged:
        labels = set()
        for f in (b.approximate_upstream_factors or []):
            labels.update(f.labels)
        missing = [getattr(v, 'label', v) for v in b.elim_vars
                   if getattr(v, 'label', v) not in labels]
        assert not missing, (f'{merge_name}/{route_name}: cluster upstream is '
                             f'missing elim vars {missing}')


@pytest.mark.parametrize('merge_name,route_name', CASES)
def test_cluster_proposal_scope_equals_separator(built, merge_name, route_name):
    """Defect 4: the proposal scope must be the cluster's separator, not the
    union of the whole downstream chain.

    HONESTY NOTE (doc 57, 2026-08-14). `proposal_scope_for_bucket` now returns
    `bucket.get_message_scope()` unconditionally -- the single-elim-var branch
    that read the pre-merge `gm.message_scopes` cache is gone -- so this
    assertion is now a tautology and pins a CONTRACT, not a behaviour. It is
    kept so that reintroducing any other scope source fails loudly. The defect-4
    content it used to carry (scope != downstream-chain union) is measured in
    doc 10; the live proposal-path coverage is in
    tests/test_proposal_sampling_path.py.
    """
    gm = built[(merge_name, route_name)]
    merged = [b for b in gm.buckets.values() if len(b.elim_vars) > 1]
    if not merged:
        pytest.skip(f'{merge_name} produced no merged clusters on this problem')
    for b in merged:
        assert sorted(proposal_scope_for_bucket(b, gm)) == sorted(b.get_message_scope())


@pytest.mark.parametrize('merge_name,route_name', CASES)
def test_upstream_and_downstream_decompose_the_model(merge_name, route_name):
    """upstream + downstream must be a complete, non-overlapping decomposition.

    With bw_ecl large enough that every WMB step in the population sweep is
    exact, eliminating their union over ALL variables must reproduce the model's
    exact log Z. Omitted content makes this too small, double-counted content
    (a cluster member leaking into the downstream snapshot) too large.
    """
    from nce.inference.elimination_order import wtminfill_order

    exact_cfg = prepare_config(dict(
        neurobe_mode=False, iB=30, ecl=2 ** 30, device='cpu',
        approximation_method='wmb', loss_fn='logspace_mse_fdb', loss_fn2='',
        populate_bw_factors=False, verbose_merge=False), strict=False)
    ref_gm = FastGM(factors=_grid_factors(),
                    elim_order=wtminfill_order(_grid_factors(),
                                               variables_not_eliminated=[]),
                    nn_config=exact_cfg, device='cpu')
    ref_gm.eliminate_variables(all=True)
    ref_logz = float(ref_gm.log_partition_function)

    gm = _build(MERGE_STRATEGIES[merge_name], POPULATION_ROUTES[route_name],
                bw_ecl=2 ** 20)
    merged = [b for b in gm.buckets.values() if len(b.elim_vars) > 1]
    if not merged:
        pytest.skip(f'{merge_name} produced no merged clusters on this problem')
    for b in merged:
        factors = list(b.approximate_upstream_factors or []) + \
            list(b.approximate_downstream_factors or [])
        factors = [f.to_exact() if hasattr(f, 'to_exact') else f for f in factors]
        if not factors:
            continue
        tmp = FastGM(factors=factors,
                     elim_order=wtminfill_order(factors, variables_not_eliminated=[]),
                     reference_fastgm=gm, nn_config=exact_cfg, device='cpu')
        tmp.is_primary = False
        tmp.eliminate_variables(all=True)
        assert float(tmp.log_partition_function) == pytest.approx(ref_logz, abs=1e-3)


# ---------------------------------------------------------------------------
# Doc 54: four further sites of the SAME defect class (temporary/copied GMs
# that re-merge, and merge-unaware per-variable bucket indexing), none of which
# doc 10 or doc 44 covered. All on the backward path.
# ---------------------------------------------------------------------------

def _merge_flags_are_off(cfg):
    return (not cfg.get('use_join_tree_merge', False)
            and not cfg.get('use_reduce_nn_merge', False)
            and not cfg.get('use_non_subsumption_merge', False)
            and not int(cfg.get('merge_degree', 0) or 0))


@pytest.mark.parametrize('merge_name,route_name', CASES)
def test_get_backward_factors_is_merge_aware(built, merge_name, route_name):
    """Doc 54 defect 1: `_get_backward_factors` indexed `gm.buckets[var]` for
    every var in `gm.elim_order`. Absorbed members are removed from
    `gm.buckets` but stay in `elim_order`, so it raised KeyError under every
    merge strategy. This is the path taken whenever `use_bw_approx=True`
    without pre-populated backward factors.
    """
    from nce.utils.backward_message import _get_backward_factors
    gm = _build(MERGE_STRATEGIES[merge_name], POPULATION_ROUTES[route_name])
    live = [v for v in gm.elim_order if v in gm.buckets]
    if len(live) < 3:
        pytest.skip('not enough surviving buckets')
    target = live[len(live) // 2]
    facs = _get_backward_factors(gm, target.label, None)   # used to KeyError
    assert isinstance(facs, list)


@pytest.mark.parametrize('merge_name,route_name', CASES)
def test_backward_message_downstream_gm_does_not_remerge(built, merge_name, route_name):
    """Doc 54 defect 2: `get_backward_message` built its downstream GM from a
    verbatim `deepcopy(gm.config)`, so all four merge passes re-ran inside the
    backward GM. Measured consequence on grids/grid10x10.f10 under subsumption:
    the returned backward message spanned 11 variables against a 2-variable
    separator, i.e. nine variables that should have been summed out were not.

    Asserted on the OBSERVABLE consequence -- the backward message must live on
    exactly the cluster's separator -- not on the config, so the test still
    means something if the implementation changes.
    """
    from nce.utils.backward_message import get_backward_message
    gm = built[(merge_name, route_name)]
    checked = 0
    for kv, b in gm.buckets.items():
        if not b.approximate_downstream_factors:
            continue
        scope = b.get_message_scope()
        if not scope:
            continue
        bw, _ = get_backward_message(
            gm, kv.label, backward_factors=list(b.approximate_downstream_factors),
            iB=10, backward_ecl=256, approximation_method='wmb',
            return_factor_list=False)
        assert set(bw.labels) <= set(scope), (
            f'{merge_name}/{route_name} cluster {kv.label}: backward message '
            f'spans {sorted(bw.labels)} outside separator {sorted(scope)}')
        checked += 1
    if checked == 0:
        pytest.skip('no populated clusters with a non-empty separator')


@pytest.mark.parametrize('merge_name,route_name', CASES)
def test_backward_message_honours_return_factor_list(built, merge_name, route_name):
    """Doc 54 defect 5: the three early returns in `get_backward_message`
    ignored `return_factor_list` and handed back a bare FastFactor, which every
    caller then feeds to SampleGenerator.sample_tensor_product (which iterates
    it). Reachable because merging creates empty-separator clusters.
    """
    from nce.utils.backward_message import get_backward_message
    gm = built[(merge_name, route_name)]
    checked = 0
    for kv, b in gm.buckets.items():
        if b.approximate_downstream_factors is None:
            continue
        fl, _ = get_backward_message(
            gm, kv.label, backward_factors=list(b.approximate_downstream_factors),
            iB=10, backward_ecl=256, approximation_method='wmb',
            return_factor_list=True)
        assert isinstance(fl, list), (
            f'{merge_name}/{route_name} cluster {kv.label} (sep='
            f'{sorted(b.get_message_scope())}): got {type(fl).__name__}')
        checked += 1
    if checked == 0:
        pytest.skip('no populated clusters')


@pytest.mark.parametrize('merge_name,route_name', CASES)
def test_message_gradient_factors_gm_does_not_remerge(built, merge_name, route_name):
    """Doc 54 defect 3: `get_wmb_message_gradient_factors` builds a temporary GM
    from a factor LIST using a copy of the caller's config, with the merge flags
    intact. It raised KeyError under subsumption / non-subsumption.
    """
    from nce.inference.message_gradient_factors import get_wmb_message_gradient_factors
    gm = built[(merge_name, route_name)]
    for kv, b in gm.buckets.items():
        if not b.approximate_downstream_factors:
            continue
        scope = b.get_message_scope()
        if not scope:
            continue
        facs = [f.to_exact() if hasattr(f, 'to_exact') else f
                for f in b.approximate_downstream_factors]
        get_wmb_message_gradient_factors(facs, scope, gm.config)   # used to KeyError
        return
    pytest.skip('no populated clusters with a non-empty separator')


@pytest.mark.parametrize('merge_name,route_name', CASES)
def test_backward_factors_available_without_use_bw_approx(built, merge_name, route_name):
    """Doc 54: `use_bw_approx` gated THREE things at once -- computing the
    backward factors, the preprocessor's normalising constant, and `bw_hat` in
    the loss -- so the `fw_bw` arm could not be isolated. `FastBucket.
    get_backward_factor_list()` provides the factors with the flag off.
    """
    gm = built[(merge_name, route_name)]
    assert gm.config.get('use_bw_approx', False) is False
    for kv, b in gm.buckets.items():
        if b.approximate_downstream_factors is None:
            continue
        fl = b.get_backward_factor_list(backward_iB=10, backward_ecl=256)
        assert isinstance(fl, list)
        return
    pytest.skip('no populated clusters')
