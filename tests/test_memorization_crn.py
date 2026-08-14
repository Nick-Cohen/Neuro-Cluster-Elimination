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

NON-VACUITY
-----------
`test_memorization_arms_are_paired_across_merge_strategies` passing means
nothing on its own -- a run in which no two arms ever share a separator would
pass trivially, and so would one where every arm produced no table at all. So:

  * the test asserts a nonzero number of SHARED separators before comparing;
  * `test_the_legacy_label_keying_breaks_the_pairing` runs the same comparison
    with the legacy `seed * 1000003 + bucket.label` seeding shimmed back in and
    asserts it FAILS, which is the negative control;
  * `test_memorization_digests_are_not_degenerate` rejects an all-zeros or
    single-valued assignment matrix, which would satisfy every equality
    assertion here.
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


def _run_arm(arm, legacy=False):
    """Run one arm and return {tuple(separator): (digest, bucket_label)}.

    `legacy=True` shims the pre-CRN seeding back in: the no-repeat generator
    keyed on `seed * 1000003 + bucket.label`, and the uniform fill drawn off
    that same consumed generator. The bucket label is captured through
    `_proposal_tree_over_scope`, which `build_memorization_table` calls on the
    line immediately before it builds the generator.
    """
    from nce.neural_networks import memorization_table as mt

    cfg = prepare_config(dict(BASE, **ARMS[arm]), strict=False)
    real_crn = mt._crn
    real_tree = mt._proposal_tree_over_scope

    class _LegacyCrn:
        """Only the two seeding entry points differ; everything else defers."""
        label = None

        def __getattr__(self, name):
            return getattr(real_crn, name)

        def no_replacement_generator(self, config, scope, domain_sizes, device,
                                     draw_index=0, role=None):
            g = torch.Generator(device=device)
            g.manual_seed(int(config.get('seed', 42)) * 1000003 + int(self.label))
            self._g = g
            return g

        def proposal_uniform(self, config, n, scope, doms, device, draw_index=0):
            return torch.stack(
                [torch.randint(0, int(d), (int(n),), generator=self._g,
                               device=device) for d in doms], dim=1)

    shim = _LegacyCrn()

    def _tree_probe(bucket, gm, scope, ecl=0, temperature=1.0):
        shim.label = bucket.label
        return real_tree(bucket, gm, scope, ecl=ecl, temperature=temperature)

    if legacy:
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
def legacy_arm_tables():
    return {arm: _run_arm(arm, legacy=True) for arm in ARMS}


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


def test_the_pairing_survives_a_different_key_variable(arm_tables):
    """The specific thing the bucket label broke.

    Pairing is only interesting where two arms put the SAME separator on
    DIFFERENT key variables -- that is precisely the case the legacy
    `seed*1000003 + bucket.label` seeding got wrong. Skip rather than pass
    silently if this problem never produces one.
    """
    interesting = [(sep, per_arm) for sep, per_arm in _shared(arm_tables)
                   if len({lab for _, lab in per_arm.values()}) > 1]
    if not interesting:
        pytest.skip('no shared separator sits on different key variables here')
    for sep, per_arm in interesting:
        digests = {d for d, _ in per_arm.values()}
        assert len(digests) == 1, (
            f'separator {sep} sits on key vars '
            f'{sorted({l for _, l in per_arm.values()})} and drew different '
            f'assignments per arm: {per_arm}')


def test_the_legacy_label_keying_breaks_the_pairing(legacy_arm_tables,
                                                    arm_tables):
    """Negative control: the test above must be capable of failing.

    With the pre-CRN seeding shimmed back in, at least one shared separator
    must draw DIFFERENT assignments across arms. If this ever passes, the
    pairing test has stopped discriminating and is worthless.
    """
    shared = _shared(legacy_arm_tables)
    assert shared, 'legacy run shares no separator -- control is vacuous'
    differing = [sep for sep, per_arm in shared
                 if len({d for d, _ in per_arm.values()}) > 1]
    assert differing, (
        'the LEGACY seeding paired every shared separator too, so '
        'test_memorization_arms_are_paired_across_merge_strategies proves '
        'nothing on this problem')


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
