"""`proposal_sampling=True` must actually run, and stay CRN-paired while it does.

WHAT THIS PINS
--------------
Before this file, `config['proposal_sampling'] = True` could not complete a
single elimination: it died with `KeyError: <first separator var>` at
`proposal_in_elim.py`'s `samples_dict[v]`, on every combination of iB, bw_ecl
and merge arm probed (doc 56 section 7). The importance-sampling design's
WMB-proposal columns were therefore dead before common random numbers mattered.

The cause was a missing precondition rather than a scope defect: the WMB
proposal tree is built from `bucket.approximate_upstream_factors +
approximate_downstream_factors`, which ONLY the backward-factor population pass
fills in, so with `populate_bw_factors` off (its default) the tree was empty and
the empty samples dict surfaced ~100 lines downstream as an opaque KeyError.

NON-VACUITY
-----------
`test_the_empty_tree_now_raises_naming_the_flag` reproduces the old failure
state deliberately (blank the approximate factors on a correctly configured GM)
and asserts the guard fires with the flag's name in the message. Without it,
"the end-to-end tests pass" could mean the guard is unreachable dead code.
`test_the_uniform_fill_shifts_log_q_by_its_own_density` is the only check on the
fill's importance weights: a fill that forgot the density correction would still
produce assignments, still train, and still pass every other test here.

Design: notebooks/_August-2026/claude_experiments/57-proposal-and-memo-crn.md.
"""
import contextlib
import io
import math

import pytest
import torch

from nce.config_schema import prepare_config
from nce.inference.graphical_model import FastGM
from nce.sampling import crn

# tests/test_common_random_numbers.py's BASE -- the repo's own "smallest thing
# that trains at all" -- plus proposal_sampling. Nothing invented here.
PROBLEM_KEY = 'grids/grid10x10.f10'
SEED = 42
BASE = dict(neurobe_mode=True, ecl=1025, sampling_scheme='uniform', iB=10,
            stream_nn_exact=True, device='cpu', seed=SEED, dope_factors=True,
            num_epochs=1, verbose_merge=False, num_samples=256,
            proposal_sampling=True, bw_ecl=1025)

ARMS = {
    'nomerge': dict(use_reduce_nn_merge=False),
    'rnn4': dict(use_reduce_nn_merge=True, max_merge_bound=4,
                 reduce_nn_backtrack=True),
    'jt4': dict(use_join_tree_merge=True, max_merge_bound=4),
}
MIXES = ['full', 'half', 'no_replacement', 'half_nr', 'uniform']


@pytest.fixture(scope='session', autouse=True)
def _pin_threads():
    torch.set_num_threads(1)


def _load_model():
    from nce.benchmark_problems.catalog_utils import get_catalog
    return get_catalog()[PROBLEM_KEY]


def _run(arm, mix, record=None):
    """One full elimination. Returns (logZ, n_nn_clusters)."""
    import nce.benchmark.proposal_in_elim as pie

    cfg = prepare_config(dict(BASE, proposal_mix=mix, **ARMS[arm]), strict=False)
    orig = pie._stack_over_scope

    def probe(samples, msg_scope, domain_sizes, config, device, draw_index, lp):
        a, l = orig(samples, msg_scope, domain_sizes, config, device,
                    draw_index, lp)
        if record is not None:
            record[(tuple(msg_scope), draw_index)] = crn.assignments_digest(a)
        return a, l

    pie._stack_over_scope = probe
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            gm = FastGM(model=_load_model(), nn_config=cfg,
                        device=cfg['device'])
            gm.eliminate_variables(all=True)
    finally:
        pie._stack_over_scope = orig
    n_nn = buf.getvalue().count('[ProposalElim] Bucket')
    return float(gm.log_partition_function), n_nn


# --------------------------------------------------------------------------
# 1. The precondition
# --------------------------------------------------------------------------
def test_proposal_sampling_derives_populate_bw_factors():
    cfg = prepare_config(dict(BASE), strict=False)
    assert cfg['populate_bw_factors'] is True


def test_neurobe_modes_own_default_does_not_win():
    """NEUROBE_DEFAULTS carries populate_bw_factors=False. The derivation has to
    run BEFORE that expansion or it would be silently undone."""
    cfg = prepare_config(dict(BASE, neurobe_mode=True), strict=False)
    assert cfg['populate_bw_factors'] is True


def test_an_explicit_false_is_rejected_rather_than_flipped():
    with pytest.raises(ValueError, match='populate_bw_factors'):
        prepare_config(dict(BASE, populate_bw_factors=False), strict=False)


def test_the_internal_wmb_sub_gms_still_build():
    """The derivation makes `proposal_sampling=True` + an explicit
    `populate_bw_factors=False` an error, and five internal sites set exactly
    that flag on their derived configs. They must drop proposal_sampling too or
    every proposal run dies inside FastGM.__init__."""
    cfg = prepare_config(dict(BASE, **ARMS['rnn4']), strict=False)
    with contextlib.redirect_stdout(io.StringIO()):
        gm = FastGM(model=_load_model(), nn_config=cfg, device='cpu')
    # populate_backward_factors_wmb ran in __init__ via _create_population_copy.
    assert any(b.approximate_upstream_factors for b in gm.buckets.values())


# --------------------------------------------------------------------------
# 2. End to end
# --------------------------------------------------------------------------
@pytest.mark.parametrize('mix', MIXES)
def test_every_proposal_mix_completes_an_elimination(mix):
    logz, n_nn = _run('rnn4', mix)
    assert n_nn > 0, f'{mix}: no bucket took the proposal path at all'
    assert math.isfinite(logz), f'{mix}: logZ = {logz}'


@pytest.mark.parametrize('arm', sorted(ARMS))
def test_every_merge_arm_completes_an_elimination(arm):
    logz, n_nn = _run(arm, 'full')
    assert n_nn > 0, f'{arm}: no bucket took the proposal path at all'
    assert math.isfinite(logz), f'{arm}: logZ = {logz}'


def test_the_proposal_draws_are_paired_across_merge_arms():
    """Doc 56 could exercise the sampling FUNCTIONS but not the training loop
    that consumes them. This closes that gap: same separator, same draw index,
    two merge arms => byte-identical assignments after the whole loop."""
    per_arm = {}
    for arm in ARMS:
        rec = {}
        _run(arm, 'full', record=rec)
        per_arm[arm] = rec

    arms = sorted(ARMS)
    shared, bad = 0, []
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            a, b = arms[i], arms[j]
            for k in set(per_arm[a]) & set(per_arm[b]):
                shared += 1
                if per_arm[a][k] != per_arm[b][k]:
                    bad.append((k, per_arm[a][k], per_arm[b][k]))
    assert shared > 0, 'no separator is shared across arms -- vacuous'
    assert not bad, 'proposal draws differ on a shared separator: %r' % (bad,)


# --------------------------------------------------------------------------
# 3. The guards, and the fill's density
# --------------------------------------------------------------------------
def test_the_empty_tree_now_raises_naming_the_flag():
    """Reproduce the pre-fix state on a correctly configured GM.

    Blanking the approximate factors is exactly what `populate_bw_factors=False`
    used to leave behind. The old code turned that into
    `KeyError: <separator var>` a hundred lines away; it must now name the flag.
    """
    cfg = prepare_config(dict(BASE, **ARMS['rnn4']), strict=False)
    with contextlib.redirect_stdout(io.StringIO()):
        gm = FastGM(model=_load_model(), nn_config=cfg, device='cpu')
        for b in gm.buckets.values():
            b.approximate_upstream_factors = None
            b.approximate_downstream_factors = None
        with pytest.raises(RuntimeError, match='populate_bw_factors'):
            gm.eliminate_variables(all=True)


def test_the_uniform_fill_shifts_log_q_by_its_own_density():
    """The fill is a uniform draw on the missing sub-scope, so its density
    -sum(log d) must be ADDED to the tree's log q or every importance weight is
    off by a constant factor -- invisible to every equality test in this file."""
    from nce.benchmark.proposal_in_elim import _stack_over_scope

    scope, doms = [3, 7, 11], [2, 3, 2]
    n = 32
    samples = {3: torch.zeros(n, dtype=torch.long),
               11: torch.ones(n, dtype=torch.long)}   # 7 is missing
    lp = torch.zeros(n)
    cfg = dict(seed=SEED, common_random_numbers=True)

    with contextlib.redirect_stdout(io.StringIO()):
        a, lp_out = _stack_over_scope(samples, scope, doms, cfg, 'cpu',
                                      crn.DRAW_TRAIN, lp)
    assert a.shape == (n, 3)
    assert torch.equal(a[:, 0], samples[3]) and torch.equal(a[:, 2], samples[11])
    assert int(a[:, 1].min()) >= 0 and int(a[:, 1].max()) < 3
    assert int(a[:, 1].unique().numel()) > 1, 'the fill is constant'
    assert torch.allclose(lp_out, torch.full((n,), -math.log(3)))

    # Same fill is drawn whatever the surrounding draw did -- the property the
    # memorization site's `generator=rng` version did not have.
    with contextlib.redirect_stdout(io.StringIO()):
        b, _ = _stack_over_scope(
            {3: torch.ones(n, dtype=torch.long),
             11: torch.zeros(n, dtype=torch.long)}, scope, doms, cfg, 'cpu',
            crn.DRAW_TRAIN, lp)
    assert torch.equal(a[:, 1], b[:, 1])


def test_a_tree_that_spans_nothing_is_not_silently_uniform():
    """Partial coverage degrades to a partly-uniform proposal, which is a real
    proposal. Zero coverage is not a proposal at all and must raise rather than
    quietly turn a WMB-proposal arm into a uniform one."""
    from nce.benchmark.proposal_in_elim import _stack_over_scope

    with pytest.raises(RuntimeError, match='populate_bw_factors'):
        _stack_over_scope({}, [3, 7], [2, 2], dict(seed=SEED), 'cpu',
                          crn.DRAW_TRAIN, torch.zeros(4))


# --------------------------------------------------------------------------
# 4. The scope the tree is built over
# --------------------------------------------------------------------------
def test_proposal_scope_is_the_live_scope_not_the_pre_merge_cache():
    """`proposal_scope_for_bucket` used to read `gm.message_scopes` whenever the
    bucket had a single elim var. That cache is built by
    `calculate_message_scopes()` at construction -- before any merge pass and
    before elimination has moved one message -- so it is a pre-elimination
    artefact, and doc 10's defect-4 fix covered only the merged branch.
    """
    from nce.sampling.proposal_sampler import proposal_scope_for_bucket

    cfg = prepare_config(dict(BASE, **ARMS['rnn4']), strict=False)
    with contextlib.redirect_stdout(io.StringIO()):
        gm = FastGM(model=_load_model(), nn_config=cfg, device='cpu')
    for b in gm.buckets.values():
        assert sorted(proposal_scope_for_bucket(b, gm)) == \
            sorted(b.get_message_scope())
