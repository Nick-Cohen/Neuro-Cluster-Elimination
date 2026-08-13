"""Common-random-numbers regression suite.

THE REQUIREMENT THIS PINS
-------------------------
"For the same problem and experimental seed, whenever two strategies produce
exactly the same separator, generate exactly the same ordered separator
assignments regardless of strategy or execution order. Derive sampling RNG
streams from stable identifiers rather than a mutable global RNG; if sample
counts differ, use the same shared prefix. No pairing is required when
separators differ."

Design and justification: notebooks/_August-2026/claude_experiments/
46-common-random-numbers.md. Implementation: nce/sampling/crn.py.

NON-VACUITY -- READ THIS BEFORE TRUSTING THE FILE
-------------------------------------------------
MEASURED 2026-08-13: on grid10x10.f10 at iB=10, the LEGACY sampler already
produces identical assignments on the separators shared by the `nomerge`,
`rnn4` and `jt4` arms. That is an accident of these problems, not a property:
the legacy seed is `bucket_label + 10000*seed + 100*draw_index`, and in every
arm pair probed, a shared separator happened to sit on the same bucket label
with the same draw index. So `test_shared_separators_are_paired_across_arms`
alone would NOT distinguish CRN from no-CRN here, and would be worthless as a
guard on its own.

The tests that actually discriminate, and fail if `common_random_numbers` stops
working, are:

  test_larger_draw_extends_smaller_draw_in_pipeline
      Legacy draws column-by-column, so changing the row count shifts every
      column after the first. The test asserts the prefix property AND asserts
      that the legacy path violates it, so it cannot pass by both paths being
      equally good.
  test_stream_ignores_bucket_label_and_draw_index
      Legacy puts the bucket label and the draw counter INTO the seed. This is
      the mechanism by which the accidental pairing above would break the first
      time a merge strategy moves a separator to a different key variable.
  test_legacy_seed_formula_collides_across_buckets
      `bucket_label + 100*draw_index` is not injective: bucket 145 draw 0 and
      bucket 45 draw 1 get the SAME seed, so two different separators share a
      stream. Any problem with more than 100 variables can hit this.
  test_marginals_are_uniform
      A stream of zeros would satisfy every equality assertion in this file.
"""
import contextlib
import io
import math

import pytest
import torch

from nce.config_schema import prepare_config
from nce.inference.graphical_model import FastGM
from nce.sampling import crn
from nce.sampling.sample_generator import SampleGenerator

# ---------------------------------------------------------------------------
# Pipeline case. Deliberately the smallest thing that trains at all: the point
# is WHICH assignments are drawn, not what the network learns from them.
# grid10x10.f10 at iB=10 is the same problem the determinism suite uses.
# ---------------------------------------------------------------------------
PROBLEM_KEY = 'grids/grid10x10.f10'
IB = 10
SEED = 42
BASE = dict(neurobe_mode=True, ecl=1025, sampling_scheme='uniform', iB=IB,
            stream_nn_exact=True, device='cpu', seed=SEED, dope_factors=True,
            num_epochs=1, verbose_merge=False)

ARMS = {
    'nomerge': dict(use_reduce_nn_merge=False),
    'rnn4': dict(use_reduce_nn_merge=True, max_merge_bound=4, reduce_nn_backtrack=True),
    'jt4': dict(use_join_tree_merge=True, max_merge_bound=4),
}


@pytest.fixture(scope='session', autouse=True)
def _pin_threads():
    torch.set_num_threads(1)


def _load_model():
    from nce.benchmark_problems.catalog_utils import get_catalog
    return get_catalog()[PROBLEM_KEY]


def _run_arm(arm, use_crn, num_samples):
    """Run one arm end to end, recording every separator draw.

    Returns {(scope, role, draw_index): tensor}. The key is exactly the CRN
    stream key minus the seed, so "the same key in two arms" is precisely the
    case the requirement is about.
    """
    cfg = prepare_config(dict(BASE, common_random_numbers=use_crn,
                              num_samples=num_samples, **ARMS[arm]), strict=False)
    out = {}
    labels = {}
    original = SampleGenerator.sample_assignments

    def patched(self, num_samples=-1, sampling_scheme=None, is_validation=False):
        res = original(self, num_samples, sampling_scheme, is_validation)
        key = (tuple(self.message_scope), 'val' if is_validation else 'train',
               self._last_draw_index)
        assert key not in out, 'duplicate draw key %r -- the recorder is wrong' % (key,)
        out[key] = res.detach().to('cpu').clone()
        labels[key] = self.bucket.label
        return res

    SampleGenerator.sample_assignments = patched
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            gm = FastGM(model=_load_model(), nn_config=cfg, device=cfg['device'])
            gm.eliminate_variables(all=True)
    finally:
        SampleGenerator.sample_assignments = original
    return out, labels


@pytest.fixture(scope='module')
def arm_draws():
    """Every arm, CRN on, 256 samples. Cached: each run is a few seconds."""
    return {arm: _run_arm(arm, use_crn=True, num_samples=256)[0] for arm in ARMS}


# ---------------------------------------------------------------------------
# 1. The arithmetic the stream rests on
# ---------------------------------------------------------------------------
def test_int64_multiply_wraps_mod_2_64():
    """splitmix64 in torch relies on int64 multiplication wrapping.

    C++ signed overflow is formally undefined, so this is a MEASURED property of
    this torch build, not a guarantee. If it ever stops holding, the CRN stream
    silently becomes a different (still deterministic) stream, and this test is
    the only thing that would say so.
    """
    mask = (1 << 64) - 1
    const = 0x9E3779B97F4A7C15
    xs = [1, 2, 3, 7919, 2 ** 40 + 1, 2 ** 62 - 1]
    got = (torch.tensor(xs, dtype=torch.int64) * crn._signed(const)).tolist()
    want = [crn._signed((x * const) & mask) for x in xs]
    assert got == want, 'int64 multiply no longer wraps mod 2**64: %r vs %r' % (got, want)


def test_lshr_is_a_logical_shift():
    """`>>` on a signed dtype is arithmetic; splitmix64 needs logical."""
    xs = torch.tensor([-1, -2, crn._signed(0xFFFFFFFFFFFFFFF0)], dtype=torch.int64)
    for k in (27, 30, 31, 32):
        got = crn._lshr(xs, k).tolist()
        want = [((int(x) & ((1 << 64) - 1)) >> k) for x in xs]
        assert got == want, 'logical shift by %d wrong: %r vs %r' % (k, got, want)


def test_mix64_matches_python_reference():
    xs = [0, 1, 2, 12345, 2 ** 31, 2 ** 63 - 1, -1, -(2 ** 62)]
    got = crn.mix64(torch.tensor(xs, dtype=torch.int64)).tolist()
    want = [crn._signed(crn.mix64_py(x & ((1 << 64) - 1))) for x in xs]
    assert got == want, 'torch mixer and reference mixer disagree: %r vs %r' % (got, want)


# ---------------------------------------------------------------------------
# 2. The stream contract, at the level of nce/sampling/crn.py
# ---------------------------------------------------------------------------
SCOPE = [7, 2, 19, 5]
DOMS = [2, 3, 2, 4]
KW = dict(scope=SCOPE, domain_sizes=DOMS, seed=SEED, role='train', draw_index=0)


def test_stream_is_prefix_closed():
    """The property that forced a counter-based design (requirement: 'if sample
    counts differ, use the same shared prefix')."""
    small = crn.uniform_assignments(1000, **KW)
    large = crn.uniform_assignments(40000, **KW)
    assert torch.equal(large[:1000], small), (
        'the 40000-row draw does not start with the 1000-row draw. The stream is '
        'not a pure function of the row index, so a strategy that draws more '
        'samples is no longer paired with one that draws fewer.')
    # and every intermediate length, not just the two endpoints
    for n in (1, 2, 17, 999, 1001):
        assert torch.equal(crn.uniform_assignments(n, **KW), large[:n]), \
            'prefix property fails at n=%d' % n


def test_stream_ignores_scope_input_order():
    """The canonical key is the separator SET; column order is `sorted(scope)`."""
    perm = [2, 0, 3, 1]
    shuffled = crn.uniform_assignments(
        500, scope=[SCOPE[i] for i in perm], domain_sizes=[DOMS[i] for i in perm],
        seed=SEED, role='train', draw_index=0)
    assert torch.equal(shuffled, crn.uniform_assignments(500, **KW)), (
        'handing the same separator in a different input order produced a '
        'different stream, so the key is not the separator set.')


def test_distinct_separators_get_distinct_streams():
    a = crn.uniform_assignments(2000, **KW)
    variants = {
        'one more variable': dict(KW, scope=SCOPE + [23], domain_sizes=DOMS + [2]),
        'one different variable': dict(KW, scope=[7, 2, 19, 6]),
        'different domain size': dict(KW, domain_sizes=[2, 3, 2, 5]),
        'different seed': dict(KW, seed=SEED + 1),
        'validation role': dict(KW, role='val'),
        'next draw': dict(KW, draw_index=1),
    }
    for name, kw in variants.items():
        b = crn.uniform_assignments(2000, **kw)
        cols = min(a.shape[1], b.shape[1])
        assert not torch.equal(a[:, :cols], b[:, :cols]), (
            '%s produced the SAME stream. CRN must pair identical separators, '
            'not everything.' % name)


def test_marginals_are_uniform():
    """Guard against a degenerate stream.

    A stream of constants satisfies every equality assertion above. This checks
    each column's empirical distribution against uniform with a chi-square
    statistic. Threshold: the 1e-6 upper tail of chi2 with df = d-1, which for
    d <= 4 is comfortably below 35; a correct stream fails this with probability
    1e-6, a constant or badly biased one fails it by orders of magnitude.
    """
    n = 200000
    a = crn.uniform_assignments(n, **KW)
    for j, d in enumerate(sorted(zip(SCOPE, DOMS))):
        d = d[1]
        counts = torch.bincount(a[:, j], minlength=d).double()
        assert int(counts.sum()) == n
        assert (counts > 0).all(), 'column %d never took some value: %r' % (j, counts)
        expected = n / d
        chi2 = float(((counts - expected) ** 2 / expected).sum())
        assert chi2 < 35.0, (
            'column %d (domain %d) is not uniform: counts %r, chi2 %.1f'
            % (j, d, counts.tolist(), chi2))


def test_empty_and_zero_length_draws():
    assert crn.uniform_assignments(0, **KW).shape == (0, 4)
    assert crn.uniform_assignments(10, scope=[], domain_sizes=[], seed=SEED,
                                   role='train', draw_index=0).shape == (10, 0)


@pytest.mark.gpu
def test_cpu_and_cuda_streams_agree():
    """Integer-only arithmetic, so the stream must not depend on the device.

    torch's own RNG does not have this property (different engines), which is
    another reason `torch.manual_seed` + `randint` could not be the mechanism.
    """
    if not torch.cuda.is_available():
        pytest.skip('no CUDA device visible')
    cpu = crn.uniform_assignments(50000, device='cpu', **KW)
    gpu = crn.uniform_assignments(50000, device='cuda', **KW).cpu()
    assert torch.equal(cpu, gpu), 'CPU and CUDA CRN streams differ'


# ---------------------------------------------------------------------------
# 3. The stream contract as SampleGenerator applies it
# ---------------------------------------------------------------------------
class _FakeVar:
    def __init__(self, label, states):
        self.label, self.states = label, states


class _FakeGM:
    device = 'cpu'

    def __init__(self, config, doms):
        self.config = config
        self.iB = 10
        self._doms = doms

    def matching_var(self, label):
        return _FakeVar(label, self._doms[label])


class _FakeFactor:
    is_nn = False

    def __init__(self, labels):
        self.labels = labels

    def order_indices(self):
        pass


class _FakeBucket:
    def __init__(self, label, scope, elim_label):
        self.label = label
        self.factors = [_FakeFactor(list(scope) + [elim_label])]
        self.elim_vars = [_FakeVar(elim_label, 2)]


def _generator(bucket_label, scope, doms, use_crn, elim_label=999):
    cfg = {'sampling_scheme': 'uniform', 'num_samples': 256,
           'common_random_numbers': use_crn}
    all_doms = dict(zip(scope, doms))
    all_doms[elim_label] = 2
    gm = _FakeGM(cfg, all_doms)
    return SampleGenerator(gm=gm, bucket=_FakeBucket(bucket_label, scope, elim_label),
                           random_seed=SEED)


def test_stream_ignores_bucket_label_and_draw_index():
    """Same separator, different bucket label and different execution position.

    This is the case the accidental legacy pairing does NOT cover: the moment a
    merge strategy moves a separator onto a different key variable, or changes
    how many draws precede it, the legacy seed changes and the arms stop being
    paired. The second half of the test asserts the legacy path really does
    break here, so the first half cannot pass vacuously.
    """
    a = _generator(11, SCOPE, DOMS, use_crn=True)
    b = _generator(77, SCOPE, DOMS, use_crn=True)
    for _ in range(3):                       # advance b's draw counter
        b._compute_seed()
    b._training_sample_counter = 0           # ... then ask for draw 0 again
    b._last_draw_index = 0
    got_a = a.sample_assignments(500)
    got_b = b.sample_assignments(500)
    assert torch.equal(got_a, got_b), (
        'CRN assignments changed with the bucket label / execution position. '
        'payload A: %s\npayload B: %s'
        % (a.crn_stream_payload(), b.crn_stream_payload()))

    la = _generator(11, SCOPE, DOMS, use_crn=False).sample_assignments(500)
    lb = _generator(77, SCOPE, DOMS, use_crn=False).sample_assignments(500)
    assert not torch.equal(la, lb), (
        'the LEGACY sampler gave the same assignments for bucket 11 and bucket '
        '77 on the same separator, so this test would pass with CRN disabled '
        'and proves nothing. Legacy seeds bucket labels into the RNG; if that '
        'stopped being true, re-derive the non-vacuity argument.')


def test_legacy_seed_formula_collides_across_buckets():
    """`bucket_label + 100*draw_index` is not injective.

    Bucket 145 draw 0 and bucket 45 draw 1 collide, so under the legacy scheme
    two DIFFERENT separators are sampled from the same stream. Any model with
    more than 100 variables can hit this (pedigree1 has 334). CRN keys on the
    separator itself, so the same pair must not collide.
    """
    hi = _generator(145, SCOPE, DOMS, use_crn=False)
    lo = _generator(45, [8, 3, 20, 6], DOMS, use_crn=False)
    lo._compute_seed()                        # burn draw 0 -> next draw is 1
    assert hi._compute_seed() == lo._compute_seed(), (
        'the legacy collision this test documents no longer exists; the seed '
        'formula changed and the surrounding argument needs revisiting.')

    hi_c = _generator(145, SCOPE, DOMS, use_crn=True).sample_assignments(500)
    lo_c = _generator(45, [8, 3, 20, 6], DOMS, use_crn=True).sample_assignments(500)
    assert not torch.equal(hi_c, lo_c), (
        'CRN gave two different separators the same stream -- the key is not '
        'discriminating.')


def test_crn_off_leaves_the_legacy_path_untouched():
    """The default must be bit-identical to the pre-CRN sampler."""
    g = _generator(11, SCOPE, DOMS, use_crn=False)
    seed = g._compute_seed()
    g._set_seed(seed)
    expected = g.sample_uniform(500)
    g2 = _generator(11, SCOPE, DOMS, use_crn=False)
    assert torch.equal(g2.sample_assignments(500), expected)


# ---------------------------------------------------------------------------
# 4. End to end: the requirement as stated, on a real elimination
# ---------------------------------------------------------------------------
def test_arms_actually_share_separators(arm_draws):
    """Non-vacuity guard for the pairing test below.

    If the arms shared nothing, the pairing test would iterate an empty set and
    pass without checking anything.
    """
    pairs = [('nomerge', 'rnn4'), ('nomerge', 'jt4')]
    for x, y in pairs:
        shared = set(arm_draws[x]) & set(arm_draws[y])
        assert shared, (
            'arms %s and %s share no separator on %s at iB=%d, so the pairing '
            'test covers nothing. Pick arms that do overlap.\n  %s: %r\n  %s: %r'
            % (x, y, PROBLEM_KEY, IB, x, sorted(arm_draws[x]), y, sorted(arm_draws[y])))


def test_shared_separators_are_paired_across_arms(arm_draws):
    """The literal requirement, on a real elimination.

    NOTE (see module docstring): MEASURED, this particular case also passes with
    CRN off, because the shared separators happen to sit on the same bucket
    label in both arms. It is kept because it is the requirement as written and
    it would catch a CRN key that accidentally included strategy state; the
    tests that discriminate CRN from legacy are the prefix and bucket-label ones.
    """
    for x, y in [('nomerge', 'rnn4'), ('nomerge', 'jt4')]:
        for key in sorted(set(arm_draws[x]) & set(arm_draws[y])):
            a, b = arm_draws[x][key], arm_draws[y][key]
            assert a.shape == b.shape, (
                '%s vs %s: separator %r drew %r and %r rows' % (x, y, key[0], a.shape, b.shape))
            assert torch.equal(a, b), (
                'arms %s and %s produced DIFFERENT assignments for the same '
                'separator %r (role %s, draw %d). digests %s vs %s'
                % (x, y, key[0], key[1], key[2],
                   crn.assignments_digest(a), crn.assignments_digest(b)))


def test_larger_draw_extends_smaller_draw_in_pipeline():
    """Shared prefix across sample counts, measured on a real elimination --
    and the same measurement on the legacy path, which must fail it.

    This is the discriminating test of the file. Legacy fills column 0 with N
    draws and column 1 with the NEXT N, so changing N shifts every column after
    the first; the prefix property is not merely absent, it is structurally
    impossible without the counter-based redesign.
    """
    small_crn, _ = _run_arm('rnn4', use_crn=True, num_samples=256)
    large_crn, _ = _run_arm('rnn4', use_crn=True, num_samples=1024)
    shared = sorted(set(small_crn) & set(large_crn))
    assert shared, 'the two sample counts produced no comparable draw'

    grew = 0
    for key in shared:
        s, l = small_crn[key], large_crn[key]
        n = min(len(s), len(l))
        assert torch.equal(l[:n], s[:n]), (
            'separator %r (role %s, draw %d): the %d-row draw is not a prefix of '
            'the %d-row draw. digests %s vs %s'
            % (key[0], key[1], key[2], len(s), len(l),
               crn.assignments_digest(s[:n]), crn.assignments_digest(l[:n])))
        if len(l) > len(s):
            grew += 1
    assert grew, (
        'no separator actually drew more rows at num_samples=1024 than at 256, '
        'so the prefix assertions above compared equal-length draws and the '
        '"shared prefix" property was never exercised.'
    )

    small_legacy, _ = _run_arm('rnn4', use_crn=False, num_samples=256)
    large_legacy, _ = _run_arm('rnn4', use_crn=False, num_samples=1024)
    violated = 0
    for key in sorted(set(small_legacy) & set(large_legacy)):
        s, l = small_legacy[key], large_legacy[key]
        n = min(len(s), len(l))
        if len(l) > len(s) and not torch.equal(l[:n], s[:n]):
            violated += 1
    assert violated, (
        'the LEGACY sampler satisfied the shared-prefix property too, so the '
        'assertions above do not distinguish CRN from no-CRN and this test is '
        'vacuous. Investigate before trusting it.')


def test_crn_flag_defaults_off():
    """A silent default change would move every number in the project."""
    cfg = prepare_config(dict(BASE, num_samples=256, **ARMS['rnn4']), strict=False)
    assert cfg['common_random_numbers'] is False
    assert cfg['deterministic_guard'] is False
