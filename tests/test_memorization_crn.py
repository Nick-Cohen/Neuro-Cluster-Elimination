"""CRN pairing for the hybrid memorization table.

THE REQUIREMENT
---------------
Same problem, same seed, same merge structure and same sampling configuration
=> the same sampled assignments across arms. `build_memorization_table` used to
seed its no-repeat sampler as `seed * 1000003 + bucket.label`. That formula is
injective, so it had no collision defect -- its problem is that the BUCKET LABEL
is exactly the execution artefact common random numbers exists to remove. Two
merge strategies that build a cluster with the same separator on a different key
variable drew different memorization samples, so memorization arms could not be
paired across strategies at all.

Both sampling sites in `build_memorization_table` are now keyed on the
separator:

  1. the no-repeat sampler   -> crn.no_replacement_generator(..., role=ROLE_MEMO)
  2. the uniform fill for separator variables the proposal tree does not span
                             -> crn.proposal_uniform on the missing sub-scope

Site 2 is the easy one to miss (doc 56 section 4 flags it explicitly), and it
was doubly broken: it drew off the *consumed* no-repeat generator, so it moved
with `memorize_num_samples` and with the tree's shape as well as with the label.

NON-VACUITY -- READ THIS BEFORE TRUSTING THE FILE
-------------------------------------------------
MEASURED 2026-08-14, doc 57: across 7 merge strategies (none, sub4, sub8,
nonsub4, rnn4, rnn8, merge_degree 4) on grid10x10.f10, grid10x10.f10.wrap and
pedigree1, there is NOT ONE separator that two strategies share while sitting on
different key variables. The key variable of a cluster is pinned by the
elimination order, and every strategy that reaches a given separator reaches it
at the same variable. So the LEGACY `seed * 1000003 + bucket.label` seeding
already paired every shared separator on every problem in the suite, by
accident, and `test_memorization_arms_are_paired_across_merge_strategies` would
NOT distinguish it from CRN. This is the same accident that
tests/test_common_random_numbers.py records for the uniform sampler.

And it is weaker still than that. MEASURED 2026-08-14:
`sample_no_replacement_v3_recursive` **never consumes its generator** -- its
phase 1 over-delivers on every tree probed, so the `K_outer < N` branch that
reaches the RNG never runs. Site 1's seed, legacy or CRN, is not an input to
the draw at all. `test_the_no_replacement_sampler_never_consumes_its_generator`
records that as a TRIPWIRE so the day it changes, someone is told.

So this file does NOT claim to have repaired a measured break. It pins:

  test_memorization_arms_are_paired_across_merge_strategies
      the property itself, guarded on a nonzero count of shared separators.
  test_the_memorization_draw_does_not_depend_on_the_bucket_label
      that shimming the label back in changes nothing -- which is the pairing
      property, stated in the direction that is currently true.
  test_the_uniform_fill_is_keyed_on_the_missing_subscope
      site 2, which is NOT inert: it used to draw off the *consumed* no-repeat
      generator, so it moved with `memorize_num_samples` and with the tree's
      shape as well as with the label.
  test_memorization_digests_are_not_degenerate
      a constant assignment matrix would satisfy every equality here.
"""
import contextlib
import io

import pytest
import torch

from nce.config_schema import prepare_config
from nce.inference.graphical_model import FastGM
from nce.sampling import crn

PROBLEM_KEY = 'grids/grid10x10.f10'
SEED = 42

# Sampling budget is doc 50's "anchor" point (sample_frac 0.1 / mem_frac 0.01),
# which is doc 44's exact configuration -- not a number invented here. The rest
# is tests/test_common_random_numbers.py's BASE, the repo's own smallest config
# that trains at all, plus the backward-factor population the proposal tree
# needs (doc 50's runner sets the same flag, for the same reason).
BASE = dict(neurobe_mode=True, ecl=1025, sampling_scheme='uniform', iB=10,
            stream_nn_exact=True, device='cpu', seed=SEED, dope_factors=True,
            num_epochs=1, verbose_merge=False, num_samples=256,
            populate_bw_factors=True, bw_ecl=1025,
            use_memorization_table=True,
            memorize_sample_frac=0.1, memorize_frac=0.01)

ARMS = {
    'nomerge': dict(use_reduce_nn_merge=False),
    'rnn4': dict(use_reduce_nn_merge=True, max_merge_bound=4,
                 reduce_nn_backtrack=True),
    'jt4': dict(use_join_tree_merge=True, max_merge_bound=4),
}


@pytest.fixture(scope='session', autouse=True)
def _pin_threads():
    torch.set_num_threads(1)


def _load_model():
    from nce.benchmark_problems.catalog_utils import get_catalog
    return get_catalog()[PROBLEM_KEY]


def _run_arm(arm, label_keyed=False):
    """Run one arm and return {tuple(separator): (digest, bucket_label)}.

    `label_keyed=True` shims the BUCKET LABEL back into the no-repeat seed --
    which is what `seed * 1000003 + bucket.label` did. The label is captured
    through `_proposal_tree_over_scope`, which `build_memorization_table` calls
    on the line immediately before it builds the generator.
    """
    from nce.neural_networks import memorization_table as mt

    cfg = prepare_config(dict(BASE, **ARMS[arm]), strict=False)
    real_crn = mt._crn
    real_tree = mt._proposal_tree_over_scope

    class _LabelKeyedCrn:
        """Only the no-repeat seed differs; everything else defers."""
        label = None

        def __getattr__(self, name):
            return getattr(real_crn, name)

        def no_replacement_generator(self, config, scope, domain_sizes, device,
                                     draw_index=0, role=None):
            g = torch.Generator(device=device)
            g.manual_seed(int(config.get('seed', 42)) * 1000003 + int(self.label))
            return g

    shim = _LabelKeyedCrn()

    def _tree_probe(bucket, gm, scope, ecl=0, temperature=1.0):
        shim.label = bucket.label
        return real_tree(bucket, gm, scope, ecl=ecl, temperature=temperature)

    if label_keyed:
        mt._crn = shim
        mt._proposal_tree_over_scope = _tree_probe
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            gm = FastGM(model=_load_model(), nn_config=cfg, device=cfg['device'])
            gm.eliminate_variables(all=True)
    finally:
        mt._crn = real_crn
        mt._proposal_tree_over_scope = real_tree

    out = {}
    for rec in getattr(gm, 'memorization_log', []):
        assert rec.get('ok'), (f'{arm}: memorization failed on bucket '
                               f'{rec.get("bucket")}: {rec.get("error")}\n'
                               f'{rec.get("traceback", "")}')
        out[tuple(rec['separator'])] = (rec['assignments_digest'], rec['bucket'])
    assert out, f'{arm}: no memorization tables were built at all'
    return out


@pytest.fixture(scope='module')
def arm_tables():
    return {arm: _run_arm(arm) for arm in ARMS}


@pytest.fixture(scope='module')
def label_keyed_rnn4():
    return _run_arm('rnn4', label_keyed=True)


def _shared(tables):
    """[(separator, {arm: (digest, label)})] for separators seen in >1 arm."""
    seps = {}
    for arm, tbl in tables.items():
        for sep, val in tbl.items():
            seps.setdefault(sep, {})[arm] = val
    return [(s, v) for s, v in sorted(seps.items()) if len(v) > 1]


def test_the_arms_actually_share_separators(arm_tables):
    """Guard for every test below: nothing to pair means nothing is proved."""
    shared = _shared(arm_tables)
    assert shared, ('no separator is shared by two arms on this problem, so '
                    'the pairing tests would be vacuous')


def test_memorization_arms_are_paired_across_merge_strategies(arm_tables):
    """The property: shared separator => byte-identical memorization draw."""
    bad = []
    for sep, per_arm in _shared(arm_tables):
        digests = {d for d, _ in per_arm.values()}
        if len(digests) != 1:
            bad.append((sep, per_arm))
    assert not bad, 'memorization draws differ on a shared separator: %r' % (bad,)


def test_the_memorization_draw_does_not_depend_on_the_bucket_label(
        arm_tables, label_keyed_rnn4):
    """Re-keying the no-repeat seed on the bucket label changes NOTHING.

    Measured, and it is not the result the fix was aiming at: it holds because
    `sample_no_replacement_v3_recursive` never touches its generator in this
    regime (see `test_the_no_replacement_sampler_never_consumes_its_generator`),
    so neither the legacy seed nor the CRN one is an input to the draw at all.

    Asserted anyway, in the direction that is true, because it is the pairing
    property itself: the same separator gives the same assignments whatever key
    variable it sat on. If the sampler ever becomes seed-sensitive, the
    tripwire below fires first and tells the next reader that this assertion
    just became load-bearing.
    """
    real = arm_tables['rnn4']
    shared = set(real) & set(label_keyed_rnn4)
    assert shared, 'the shimmed run built no comparable table'
    moved = [s for s in shared if real[s][0] != label_keyed_rnn4[s][0]]
    assert not moved, (
        'the bucket label moved the memorization draw, so the CRN re-key is '
        'incomplete: %r' % (moved,))
    # ... and the same separator gives the same digest on every arm.
    by_sep = {}
    for tbl in arm_tables.values():
        for sep, (digest, _label) in tbl.items():
            by_sep.setdefault(sep, set()).add(digest)
    assert all(len(v) == 1 for v in by_sep.values())


def test_the_no_replacement_sampler_never_consumes_its_generator():
    """TRIPWIRE, not a property test. Records a measured fact so that the day it
    stops being true, someone is told.

    `sample_no_replacement_v3_recursive` only reaches its RNG when
    `K_outer < N` -- i.e. when phase 1 commits FEWER samples than the budget.
    Measured 2026-08-14 (doc 57): phase 1 over-delivers on every tree probed
    (2^12 / 2^16 / 2^20 states, N from 4 to 128, M from 1 to 1000, temperatures
    1.0 / 0.3 / 0.05), so the branch never runs and the generator is never
    advanced. The NR draw is therefore FULLY DETERMINISTIC given the tree.

    Consequences, both of which the rerun needs to know:
      * memorization arms pair across strategies automatically, not because of
        the CRN re-key. The re-key removes the bucket label from a key that is
        currently inert; it is insurance, not a repair.
      * the `no_replacement` and `half_nr` proposal mixes have NO Monte-Carlo
        variability from the sampler seed. Seed-replicate error bars on those
        arms measure only the NN's own randomness.

    If this test fails, the seeding suddenly matters and every pairing claim in
    this file has to be re-derived rather than assumed.
    """
    from nce.inference.factor import FastFactor
    from nce.sampling.proposal_sampler import ProposalTree, BucketRecord
    import nce.sampling.no_replacement_sampler_v3  # noqa: F401  (attaches)

    def _tree(nvars, D=2, temp=1.0, seed=0):
        g = torch.Generator().manual_seed(seed)
        levels = []
        for v in range(nvars):
            # Chain to a LATER-eliminated variable: levels are stored in
            # elimination order and sampled in reverse, so a factor may only
            # reference variables already drawn.
            fs = [FastFactor(torch.rand(D, generator=g).log10() / temp, [v])]
            if v + 1 < nvars:
                fs.append(FastFactor(
                    torch.rand(D, D, generator=g).log10() / temp, [v, v + 1]))
            levels.append(BucketRecord(v, D, fs))
        return ProposalTree(levels, 'cpu')

    for nvars in (12, 16):
        for temp in (1.0, 0.3):
            t = _tree(nvars, temp=temp)
            for n in (8, 32, 128):
                g = torch.Generator()
                g.manual_seed(1)
                before = g.get_state().clone()
                t.sample_no_replacement_v3_recursive(n, M=1, rng=g, mode='save')
                assert torch.equal(before, g.get_state()), (
                    f'the NR sampler consumed its generator at nvars={nvars} '
                    f'T={temp} N={n} -- the seeding is now load-bearing')


def test_memorization_digests_are_not_degenerate(arm_tables):
    """A constant assignment matrix would satisfy every equality above."""
    digests = {d for tbl in arm_tables.values() for d, _ in tbl.values()}
    assert len(digests) > 1, ('every cluster in every arm produced the same '
                              'assignment digest -- the sampler is degenerate')


def test_the_uniform_fill_is_keyed_on_the_missing_subscope():
    """Site 2, unit level: the fill is a pure function of the sub-scope.

    Keyed on the missing sub-scope rather than on the whole separator, because
    that sub-scope is what the uniform q is defined over. The property that
    matters is that it does not depend on the surrounding draw at all -- the
    legacy version pulled it off the *consumed* no-repeat generator, so it moved
    with `memorize_num_samples` and with the tree's shape.
    """
    cfg = dict(seed=SEED, common_random_numbers=True)
    sub, doms = [7, 13], [2, 3]
    a = crn.proposal_uniform(cfg, 64, sub, doms, 'cpu')
    b = crn.proposal_uniform(cfg, 64, sub, doms, 'cpu')
    assert torch.equal(a, b)
    # Prefix-closed: a bigger draw extends a smaller one.
    big = crn.proposal_uniform(cfg, 256, sub, doms, 'cpu')
    assert torch.equal(big[:64], a)
    # In range, and not constant.
    for j, d in enumerate(doms):
        assert int(a[:, j].min()) >= 0 and int(a[:, j].max()) < d
    assert int(a.unique().numel()) > 1
    # A different sub-scope is a different stream.
    c = crn.proposal_uniform(cfg, 64, [7, 14], doms, 'cpu')
    assert not torch.equal(a, c)


def test_memo_and_prop_nr_streams_stay_disjoint():
    """`memo` must not collide with the proposal path's own NR stream."""
    cfg = dict(seed=SEED, common_random_numbers=True)
    sep, doms = [3, 9, 21], [2, 2, 2]
    memo = crn.no_replacement_generator(cfg, sep, doms, 'cpu',
                                        role=crn.ROLE_MEMO).initial_seed()
    prop = crn.no_replacement_generator(cfg, sep, doms, 'cpu',
                                        role=crn.ROLE_PROP_NR).initial_seed()
    assert memo != prop
    # And neither is the formula they replaced, for any bucket label.
    legacy = {SEED * 1000003 + lab for lab in range(200)}
    assert int(memo) not in legacy
