"""Correctness gate for the hybrid NN + memorization-table splice (doc 44).

The splice is the only genuinely new code: `HybridMemorizerNet` must return the
memorized value on memorized rows and the base net's value everywhere else, in
the batched 65k-row one-hot path that `factor_nn.py` uses. Everything here is
CPU-only, build-only where possible, and does not depend on the problem catalog.
"""
import itertools

import pytest
import torch

from nce.config_schema import prepare_config
from nce.inference.factor import FastFactor
from nce.inference.factor_nn import FactorNN
from nce.inference.graphical_model import FastGM
from nce.neural_networks.memorization_table import (
    HybridMemorizerNet, onehot_projection, pack_assignments)
from nce.neural_networks.net import Net


def _grid_factors(n=4, seed=0):
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


def _build_gm(lower_dim):
    cfg = prepare_config(dict(
        neurobe_mode=False, iB=2, ecl=4, device='cpu',
        approximation_method='nn', seed=42, lower_dim=lower_dim,
        loss_fn='logspace_mse_fdb', loss_fn2='', num_epochs=1,
        num_samples=16, batch_size=8, hidden_sizes=[4],
        dope_factors=False, verbose_merge=False), strict=False)
    from nce.inference.elimination_order import wtminfill_order
    factors = _grid_factors()
    order = wtminfill_order(factors, variables_not_eliminated=[])
    return FastGM(factors=factors, elim_order=order, nn_config=cfg, device='cpu')


def _wide_bucket(gm, min_width=3):
    """A bucket whose OUTGOING scope is {1,2,3} plus elim var 0.

    Buckets straight out of `_create_buckets_from_factors` only hold their
    original factors, so before any elimination none of them is wide. Build one
    explicitly instead -- the splice only cares about the scope and the domain
    sizes.
    """
    from nce.inference.bucket import FastBucket
    dev = 'cpu'
    factors = [FastFactor(torch.rand(2, 2, 2, 2).log10(), [0, 1, 2, 3])]
    for f in factors:
        f.tensor = f.tensor.to(dev)
    return FastBucket(gm, 0, factors, dev, [gm.matching_var(0)])


@pytest.mark.parametrize('lower_dim', [True, False])
def test_projection_inverts_the_onehot_encoding(lower_dim):
    """onehot_row @ proj == packed key, for every assignment of a 3x2x4 scope."""
    domain_sizes = [3, 2, 4]
    proj, total = onehot_projection(domain_sizes, lower_dim)
    assignments = torch.cartesian_prod(*[torch.arange(d) for d in domain_sizes])
    if lower_dim:
        cols = [torch.nn.functional.one_hot(assignments[:, i], d)[:, 1:]
                for i, d in enumerate(domain_sizes)]
    else:
        cols = [torch.nn.functional.one_hot(assignments[:, i], d)
                for i, d in enumerate(domain_sizes)]
    x = torch.cat(cols, dim=-1).to(torch.float64)
    packed = torch.round(x @ torch.tensor(proj, dtype=torch.float64)).long()
    assert total == 24
    assert torch.equal(packed, pack_assignments(assignments, domain_sizes))
    assert torch.equal(torch.sort(packed).values, torch.arange(24))


@pytest.mark.parametrize('lower_dim', [True, False])
def test_splice_returns_memorized_on_hits_and_base_elsewhere(lower_dim):
    gm = _build_gm(lower_dim)
    bucket = _wide_bucket(gm)
    scope = bucket.get_message_scope()
    domain_sizes = bucket.get_message_dimension()
    base = Net(bucket, hidden_sizes=[4])
    base.eval()

    assignments = torch.cartesian_prod(*[torch.arange(int(d)) for d in domain_sizes])
    total = assignments.shape[0]
    g = torch.Generator().manual_seed(7)
    sel = torch.randperm(total, generator=g)[: max(1, total // 4)]
    keys = pack_assignments(assignments[sel], domain_sizes)
    values = torch.arange(sel.numel(), dtype=torch.float32) * 1.5 + 100.0

    hybrid = HybridMemorizerNet(base, keys, values, bucket, lower_dim=lower_dim)
    hybrid.eval()

    if lower_dim:
        cols = [torch.nn.functional.one_hot(assignments[:, i], int(d))[:, 1:]
                for i, d in enumerate(domain_sizes)]
    else:
        cols = [torch.nn.functional.one_hot(assignments[:, i], int(d))
                for i, d in enumerate(domain_sizes)]
    x = torch.cat(cols, dim=-1).to(torch.float32)

    with torch.no_grad():
        base_out = base(x).reshape(-1)
        hyb_out = hybrid(x).reshape(-1)

    is_hit = torch.zeros(total, dtype=torch.bool)
    is_hit[sel] = True
    # memorized rows carry exactly the stored value ...
    order = torch.argsort(keys)
    expect = torch.empty(total)
    expect[sel] = values
    assert torch.allclose(hyb_out[is_hit], expect[is_hit], atol=1e-5)
    # ... and every other row is untouched.
    assert torch.equal(hyb_out[~is_hit], base_out[~is_hit])
    assert hybrid.n_memorized == sel.numel()


def test_empty_table_is_a_noop():
    gm = _build_gm(True)
    bucket = _wide_bucket(gm)
    base = Net(bucket, hidden_sizes=[4])
    base.eval()
    hybrid = HybridMemorizerNet(base, torch.zeros(0, dtype=torch.int64),
                               torch.zeros(0), bucket, lower_dim=True)
    hybrid.eval()
    x = torch.randint(0, 2, (32, bucket._get_nn_input_size())).float()
    with torch.no_grad():
        assert torch.equal(hybrid(x), base(x))


@pytest.mark.parametrize('lower_dim', [True, False])
def test_to_exact_round_trip_replaces_exactly_k_entries(lower_dim):
    """FactorNN(hybrid).to_exact() == base dense factor with K entries replaced."""
    gm = _build_gm(lower_dim)
    bucket = _wide_bucket(gm)
    domain_sizes = bucket.get_message_dimension()
    base = Net(bucket, hidden_sizes=[4])
    base.eval()

    from nce.data.data_preprocessor import DataPreprocessor
    dp = DataPreprocessor(lower_dim=lower_dim, device='cpu')
    dp.normalize(torch.tensor([-3.0, 0.0, 1.0]), None)  # initialize constants

    assignments = torch.cartesian_prod(*[torch.arange(int(d)) for d in domain_sizes])
    g = torch.Generator().manual_seed(11)
    sel = torch.randperm(assignments.shape[0], generator=g)[:5]
    keys = pack_assignments(assignments[sel], domain_sizes)
    values = torch.full((5,), 7.0)

    base_dense = FactorNN(base, dp).to_exact()
    hybrid = HybridMemorizerNet(base, keys, values, bucket, lower_dim=lower_dim)
    hybrid.eval()
    hyb_dense = FactorNN(hybrid, dp).to_exact()

    base_flat = base_dense.tensor.reshape(-1)
    hyb_flat = hyb_dense.tensor.reshape(-1)
    diff = (base_flat != hyb_flat).nonzero().reshape(-1)
    assert diff.numel() == 5, f"expected exactly 5 replaced entries, got {diff.numel()}"
    expected = dp.undo_normalization(values)
    assert torch.allclose(torch.sort(hyb_flat[diff]).values,
                          torch.sort(expected).values, atol=1e-5)
