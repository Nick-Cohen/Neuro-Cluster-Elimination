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
  test_seed_derivation_is_collision_free_on_the_legacy_collisions
  test_derive_seed_has_no_collisions_over_the_pedigree_domain
      `bucket_label + 100*draw_index` was not injective: bucket 145 draw 0 and
      bucket 45 draw 1 got the SAME seed, so two different separators shared a
      stream. Any problem with more than 100 variables can hit it (pedigree1 has
      334). `test_the_legacy_seed_formula_did_collide` keeps the old formula as
      a reference so the regression cannot become vacuous.
  test_marginals_are_uniform
      A stream of zeros would satisfy every equality assertion in this file.

2026-08-14 -- CRN IS NOW DEFAULT-ON, AND THE PROPOSAL PATH IS COVERED
---------------------------------------------------------------------
Section 5 covers the proposal / importance-sampling arms, which doc 46 left
out. See notebooks/_August-2026/claude_experiments/56-crn-complete.md. The
discriminating tests there are `test_proposal_uniform_half_is_paired_across_
merge_arms` (asserts the legacy torch.randint half fails the same check),
`test_proposal_tree_crn_is_independent_of_the_global_rng` (asserts the
multinomial path is not), and `test_proposal_tree_crn_samples_the_right_
distribution` (a deterministic stream that samples the WRONG q would pass every
equality assertion in section 5).
"""
import contextlib
import io
import math
import os

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
    meta = {}
    original = SampleGenerator.sample_assignments

    def patched(self, num_samples=-1, sampling_scheme=None, is_validation=False):
        res = original(self, num_samples, sampling_scheme, is_validation)
        key = (tuple(self.message_scope), 'val' if is_validation else 'train',
               self._last_draw_index)
        assert key not in out, 'duplicate draw key %r -- the recorder is wrong' % (key,)
        out[key] = res.detach().to('cpu').clone()
        meta[key] = (self.bucket.label, tuple(int(d) for d in self.domain_sizes))
        return res

    SampleGenerator.sample_assignments = patched
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            gm = FastGM(model=_load_model(), nn_config=cfg, device=cfg['device'])
            gm.eliminate_variables(all=True)
    finally:
        SampleGenerator.sample_assignments = original
    return out, meta


@pytest.fixture(scope='module')
def arm_runs():
    """Every arm, CRN on, 256 samples. Cached: each run is a few seconds."""
    return {arm: _run_arm(arm, use_crn=True, num_samples=256) for arm in ARMS}


@pytest.fixture(scope='module')
def arm_draws(arm_runs):
    return {arm: run[0] for arm, run in arm_runs.items()}


@pytest.fixture(scope='module')
def arm_meta(arm_runs):
    """{arm: {separator_tuple: domain_sizes_tuple}} from the real eliminations.

    Used by the proposal tests so that they run against separators the merge
    arms ACTUALLY produce, rather than against hand-written scopes.
    """
    res = {}
    for arm, (_, meta) in arm_runs.items():
        res[arm] = {key[0]: doms for key, (_, doms) in meta.items()}
    return res


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


def _legacy_compute_seed(bucket_label, run_seed, draw, is_validation=False):
    """The seed formula this branch REMOVED, kept as a reference.

    `bucket_label + 10000*run_seed + 100*draw + 50000000*is_validation`
    (`SampleGenerator._compute_seed` up to 2026-08-14). Reproduced here rather
    than imported, because the point of the tests below is that no code path
    computes it any more.
    """
    return (int(bucket_label) + 10000 * int(run_seed) + 100 * int(draw)
            + (50000000 if is_validation else 0))


def test_the_legacy_seed_formula_did_collide():
    """Non-vacuity anchor for the two tests below.

    If this ever stops holding, the "collision-free" claim is being tested
    against a defect that was never there, and the surrounding argument (doc 46
    appendix: 173 of 519 study cells, 63 byte-identical pairs) needs revisiting.
    """
    assert _legacy_compute_seed(145, SEED, 0) == _legacy_compute_seed(45, SEED, 1)
    assert _legacy_compute_seed(127, SEED, 2) == _legacy_compute_seed(327, SEED, 0)
    # ... and across run seeds, which the appendix did not call out:
    assert _legacy_compute_seed(0, 1, 0) == _legacy_compute_seed(0, 0, 100)


def test_seed_derivation_is_collision_free_on_the_legacy_collisions():
    """The pairs that used to share a stream must not share one now.

    `bucket_label + 100*draw_index` is not injective, so under the legacy scheme
    two DIFFERENT separators were sampled from the same stream. Any model with
    more than 100 variables reaches it (pedigree1 has 334).
    """
    hi = _generator(145, SCOPE, DOMS, use_crn=False)
    lo = _generator(45, [8, 3, 20, 6], DOMS, use_crn=False)
    lo._compute_seed()                        # burn draw 0 -> next draw is 1
    s_hi, s_lo = hi._compute_seed(), lo._compute_seed()
    assert s_hi != s_lo, (
        'bucket 145 draw 0 and bucket 45 draw 1 still share a global-RNG seed '
        '(%d). The collision-free derivation is not in force.' % s_hi)

    hi_c = _generator(145, SCOPE, DOMS, use_crn=True).sample_assignments(500)
    lo_c = _generator(45, [8, 3, 20, 6], DOMS, use_crn=True).sample_assignments(500)
    assert not torch.equal(hi_c, lo_c), (
        'CRN gave two different separators the same stream -- the key is not '
        'discriminating.')


def test_the_collision_no_longer_yields_byte_identical_assignments():
    """The old collision was not just equal seeds -- it was equal DATA.

    MEASURED in the study cell audit (doc 46 appendix): on grid20x20 at iB=10,
    cluster 127's draw 2 and cluster 327's draw 0 are separators of equal width
    over binary variables, so the nbe formula gave them the same row count and
    the shared seed made their 16640x17 assignment matrices byte-identical.
    Reproduced here in miniature: two DIFFERENT separators, labels 200 apart,
    draw 2 against draw 0.

    Both halves must now differ -- the CRN path because the separator is the
    key, and the LEGACY path (crn off) because `_compute_seed` is a field-tagged
    digest instead of a sum of scaled integers.
    """
    scope_a, scope_b = [1, 4, 9, 12], [2, 5, 10, 13]
    doms = [2, 2, 2, 2]
    assert _legacy_compute_seed(127, SEED, 2) == _legacy_compute_seed(327, SEED, 0), \
        'the collision being regressed is not the one the appendix measured'

    for use_crn in (False, True):
        a = _generator(127, scope_a, doms, use_crn=use_crn)
        for _ in range(2):
            a._compute_seed()                 # advance to draw 2
        b = _generator(327, scope_b, doms, use_crn=use_crn)
        assert not torch.equal(a.sample_assignments(4096),
                               b.sample_assignments(4096)), (
            'crn=%s: two different separators still drew byte-identical '
            'assignments.' % use_crn)


def test_derive_seed_has_no_collisions_over_the_pedigree_domain():
    """Exhaustive over the label/draw range the old formula broke on.

    The legacy formula collided for any two clusters whose labels differ by a
    multiple of 100 and whose draw indices differ correspondingly -- reachable
    on every problem with more than 100 variables, and pedigree1 has 334. This
    enumerates the whole (run seed, label, role, draw) grid over that range and
    demands 2 * 8 * 400 * 8 = 51200 distinct seeds.

    `derive_seed` is a truncated SHA-256, so it is collision-free by measurement
    over the domain that matters, not injective by construction; the expected
    number of collisions at this size is 51200**2 / 2**64 ~ 1.4e-10.
    """
    seen = {}
    for run_seed in range(8):
        for label in range(400):              # pedigree1 has 334 variables
            for draw in range(8):
                for role in ('train', 'val'):
                    s = crn.derive_seed('sample-generator', run_seed=run_seed,
                                        bucket=label, role=role, draw=draw)
                    assert 0 <= s < 2 ** 63
                    prev = seen.setdefault(s, (run_seed, label, role, draw))
                    assert prev == (run_seed, label, role, draw), (
                        'derive_seed collision: %r and %r both give %d'
                        % (prev, (run_seed, label, role, draw), s))
    assert len(seen) == 8 * 400 * 8 * 2

    # The same grid under the legacy formula, to show the test has teeth.
    legacy = set()
    for run_seed in range(8):
        for label in range(400):
            for draw in range(8):
                legacy.add(_legacy_compute_seed(label, run_seed, draw))
    assert len(legacy) < 8 * 400 * 8, (
        'the legacy formula was injective on this grid, so the exhaustive check '
        'above proves nothing about the defect it replaced.')


def test_seed_derivation_does_not_depend_on_pythonhashseed():
    """Non-int bucket labels used to go through `hash(str(label)) % 10000`.

    Python salts `hash` on str per process, so that branch was not reproducible
    across runs AT ALL -- a silent nondeterminism the collision audit did not
    cover. The digest is stable by construction; this pins it against a literal.
    """
    import subprocess
    import sys
    prog = ('import os,sys;sys.path.insert(0,%r);'
            'from nce.sampling import crn;'
            'print(crn.derive_seed("sample-generator", run_seed=42, '
            'bucket="cluster-a", role="train", draw=0))'
            % str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    outs = set()
    for salt in ('0', '1', '12345'):
        env = dict(os.environ, PYTHONHASHSEED=salt)
        outs.add(subprocess.run([sys.executable, '-c', prog], env=env,
                                capture_output=True, text=True,
                                check=True).stdout.strip())
    assert len(outs) == 1, 'seed derivation varies with PYTHONHASHSEED: %r' % outs


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


def test_crn_flag_defaults_on():
    """Default flipped to True on 2026-08-14 (Nick's directive).

    It moves every number in the project, which is why the determinism goldens
    were regenerated in the same commit rather than bypassed. If this assertion
    ever flips back, the goldens are wrong too.
    """
    cfg = prepare_config(dict(BASE, num_samples=256, **ARMS['rnn4']), strict=False)
    assert cfg['common_random_numbers'] is True
    assert cfg['deterministic_guard'] is False
    # An explicit False must still be honoured: the flag is not a no-op.
    off = prepare_config(dict(BASE, num_samples=256, common_random_numbers=False,
                              **ARMS['rnn4']), strict=False)
    assert off['common_random_numbers'] is False


def test_the_default_actually_reaches_the_sampler():
    """A default that the SampleGenerator's own `.get(..., False)` overrides
    would be a config-only change. Pin both ends."""
    cfg = {'sampling_scheme': 'uniform', 'num_samples': 256}   # no CRN key at all
    doms = dict(zip(SCOPE, DOMS))
    doms[999] = 2
    gm = _FakeGM(cfg, doms)
    g = SampleGenerator(gm=gm, bucket=_FakeBucket(11, SCOPE, 999), random_seed=SEED)
    assert g.use_crn is True
    assert torch.equal(g.sample_assignments(64),
                       crn.uniform_assignments(64, **KW)[:64])


# ---------------------------------------------------------------------------
# 5. The proposal path
#
# Nick's condition, 2026-08-14: "same seed + same merge structure/strategy +
# same relevant sampling configuration should produce the same sampled
# assignments across arms. If proposal hyperparameters genuinely change the
# proposal distribution, different samples are fine."
#
# So the key must contain what genuinely determines q and nothing else. The
# three mechanisms get three different treatments, argued in
# notebooks/_August-2026/claude_experiments/56-crn-complete.md:
#   uniform half   q is uniform on the separator  -> full CRN.
#   WMB tree half  q is the proposal tree         -> inverse CDF off a shared
#                                                    uniform; identical q gives
#                                                    identical samples, different
#                                                    q gives coupled ones.
#   no-replacement Gumbel-top-k, stateful         -> separator-keyed generator
#                                                    seed; no prefix property.
# ---------------------------------------------------------------------------
PROP_CFG = dict(seed=SEED, common_random_numbers=True)
PROP_CFG_OFF = dict(seed=SEED, common_random_numbers=False)


def _burn_global_rng(seed, n):
    """Put the global torch RNG in an arbitrary state.

    Stands in for "this cluster was reached after k other clusters had already
    drawn", which is exactly what differs between two merge arms.
    """
    torch.manual_seed(seed)
    torch.rand(n)


def _shared_separators(arm_meta, a, b):
    return sorted(set(arm_meta[a]) & set(arm_meta[b]))


def test_proposal_arms_actually_share_separators(arm_meta):
    """Non-vacuity guard for the proposal pairing tests below."""
    for x, y in [('nomerge', 'rnn4'), ('nomerge', 'jt4')]:
        assert _shared_separators(arm_meta, x, y), (
            'arms %s and %s share no separator, so the proposal pairing tests '
            'iterate nothing.' % (x, y))


def test_proposal_uniform_half_is_paired_across_merge_arms(arm_meta):
    """The motivating case, on the separators the arms really produce.

    Two merge strategies, same problem, same seed, a shared separator: the
    uniform half of a mixed proposal draw must be byte-identical AND in the same
    order. The two arms reach `crn.proposal_uniform` at different points in
    their elimination, so the global RNG state differs; that is simulated here
    by burning the global stream differently before each call.

    The second half asserts the LEGACY path (`common_random_numbers=False`,
    which is the bare per-column `torch.randint` the call sites used to inline)
    fails exactly this, so the test cannot pass by both paths being equally good.
    """
    checked = 0
    legacy_violations = 0
    for x, y in [('nomerge', 'rnn4'), ('nomerge', 'jt4')]:
        for sep in _shared_separators(arm_meta, x, y):
            doms = arm_meta[x][sep]
            assert doms == arm_meta[y][sep], (
                'same separator with different domain sizes in %s and %s: %r vs %r'
                % (x, y, doms, arm_meta[y][sep]))
            _burn_global_rng(1, 1234)
            a = crn.proposal_uniform(PROP_CFG, 512, list(sep), list(doms), 'cpu')
            _burn_global_rng(9, 7)
            b = crn.proposal_uniform(PROP_CFG, 512, list(sep), list(doms), 'cpu')
            assert a.shape == (512, len(sep))
            assert torch.equal(a, b), (
                'arms %s and %s drew DIFFERENT uniform halves for the shared '
                'separator %r. digests %s vs %s'
                % (x, y, sep, crn.assignments_digest(a), crn.assignments_digest(b)))
            checked += 1

            _burn_global_rng(1, 1234)
            la = crn.proposal_uniform(PROP_CFG_OFF, 512, list(sep), list(doms), 'cpu')
            _burn_global_rng(9, 7)
            lb = crn.proposal_uniform(PROP_CFG_OFF, 512, list(sep), list(doms), 'cpu')
            legacy_violations += int(not torch.equal(la, lb))
    assert checked, 'no shared separator was checked'
    assert legacy_violations == checked, (
        'the legacy torch.randint half paired too (%d of %d), so the assertions '
        'above do not discriminate.' % (legacy_violations, checked))


def test_proposal_uniform_half_is_prefix_closed(arm_meta):
    """Arms with different sample counts must share a prefix, not diverge."""
    grew = 0
    for sep, doms in sorted(arm_meta['rnn4'].items())[:6]:
        small = crn.proposal_uniform(PROP_CFG, 300, list(sep), list(doms), 'cpu')
        large = crn.proposal_uniform(PROP_CFG, 1700, list(sep), list(doms), 'cpu')
        assert torch.equal(large[:300], small), (
            'separator %r: the 300-row proposal uniform half is not a prefix of '
            'the 1700-row one.' % (sep,))
        grew += 1
    assert grew, 'no separator was exercised'


def test_proposal_uniform_and_separator_streams_are_disjoint(arm_meta):
    """The proposal role must not reuse the training role's stream.

    They are different sampling schemes over the same separator; sharing the
    stream would silently correlate a proposal arm's uniform half with a
    uniform-scheme arm's training set, which is not a pairing anyone asked for.
    """
    sep, doms = sorted(arm_meta['rnn4'].items())[0]
    prop = crn.proposal_uniform(PROP_CFG, 2000, list(sep), list(doms), 'cpu')
    train = crn.uniform_assignments(2000, list(sep), list(doms), SEED,
                                    crn.ROLE_TRAIN, crn.DRAW_TRAIN)
    assert not torch.equal(prop, train)


def test_correction_draw_does_not_collide_with_another_run_seed(arm_meta):
    """`seed + 1` for the correction draw was a cross-run collision.

    The old code seeded the correction no-replacement generator with
    `int(config['seed']) + 1`, so run 42's CORRECTION draw was run 43's TRAINING
    draw. Draw indices replace seed arithmetic; this pins that they do not
    collapse back onto each other.
    """
    sep, doms = sorted(arm_meta['rnn4'].items())[0]
    sep, doms = list(sep), list(doms)
    train_42 = crn.stream_seed(42, sep, doms, crn.ROLE_PROP_NR, crn.DRAW_TRAIN)
    corr_42 = crn.stream_seed(42, sep, doms, crn.ROLE_PROP_NR, crn.DRAW_CORRECTION)
    train_43 = crn.stream_seed(43, sep, doms, crn.ROLE_PROP_NR, crn.DRAW_TRAIN)
    assert len({train_42, corr_42, train_43}) == 3, (
        'the correction draw shares a stream with a training draw: %r'
        % [train_42, corr_42, train_43])
    # ... whereas the formula it replaced collapsed two of the three.
    assert (42 + 1) == 43, 'the legacy defect being regressed is seed + 1'

    u_train = crn.proposal_uniform(PROP_CFG, 500, sep, doms, 'cpu', crn.DRAW_TRAIN)
    u_corr = crn.proposal_uniform(PROP_CFG, 500, sep, doms, 'cpu', crn.DRAW_CORRECTION)
    assert not torch.equal(u_train, u_corr), (
        'the correction uniform half is the training uniform half')


def test_no_replacement_generator_is_separator_keyed(arm_meta):
    """Every bucket used to share ONE no-replacement stream (`manual_seed(seed)`).

    That is the opposite defect from a collision-prone formula and just as bad:
    perfectly correlated Monte-Carlo error across clusters. The generator seed
    must now be a function of the separator.
    """
    pooled = {}
    for arm in arm_meta:
        pooled.update(arm_meta[arm])
    seps = sorted(pooled.items())
    assert len(seps) >= 2, 'need two separators to compare'
    seeds = set()
    for sep, doms in seps:
        g = crn.no_replacement_generator(PROP_CFG, list(sep), list(doms), 'cpu')
        seeds.add(int(g.initial_seed()))
    assert len(seeds) == len(seps), (
        'two different separators got the same no-replacement generator seed')

    # Same separator, whatever the global RNG has been doing: same seed.
    sep, doms = seps[0]
    _burn_global_rng(3, 999)
    s1 = crn.no_replacement_generator(PROP_CFG, list(sep), list(doms), 'cpu').initial_seed()
    _burn_global_rng(77, 4)
    s2 = crn.no_replacement_generator(PROP_CFG, list(sep), list(doms), 'cpu').initial_seed()
    assert s1 == s2
    # ... and it is NOT the run seed, which is what the old code used.
    assert int(s1) != SEED


def test_the_proposal_call_sites_do_not_reintroduce_a_bare_rng():
    """Source guard: the mechanism only works if the call sites go through it.

    A future edit that inlines `torch.randint` or `manual_seed(int(seed))` back
    into the proposal sampling blocks would silently un-pair those arms while
    every behavioural test above still passed, because those tests exercise the
    helpers rather than the call sites.
    """
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parents[1]
    banned = [
        (re.compile(r'torch\.randint\(0,\s*d,'), 'inline per-column uniform draw'),
        (re.compile(r"manual_seed\(int\(config\.get\('seed'"), 'run seed as an RNG seed'),
        (re.compile(r'manual_seed\(int\(seed\)\)'), 'run seed as an RNG seed'),
        (re.compile(r'manual_seed\(int\(bucket\.label\)'), 'bucket label as an RNG seed'),
    ]
    hits = []
    for rel in ('nce/benchmark/training.py', 'nce/benchmark/proposal_in_elim.py'):
        text = (root / rel).read_text()
        for lineno, line in enumerate(text.splitlines(), 1):
            if 'crn-seed-ok:' in line:
                continue
            for pat, why in banned:
                if pat.search(line):
                    hits.append('%s:%d %s -- %s' % (rel, lineno, line.strip(), why))
    assert not hits, (
        'the proposal path grew a sampling site that bypasses nce/sampling/crn.py:\n  '
        + '\n  '.join(hits))


# ---------------------------------------------------------------------------
# 5b. ProposalTree.sample under CRN -- the inverse-CDF mechanism itself
# ---------------------------------------------------------------------------
def _toy_tensors(scale=1.0, seed=0):
    """log10 potentials for the toy proposal tree. One source for the sampler
    and for the brute-force reference, so the two cannot drift apart."""
    g = torch.Generator().manual_seed(seed)
    t12 = torch.rand((4,), generator=g) * scale
    t9 = torch.rand((2, 4), generator=g) * scale
    t5 = torch.rand((3, 2), generator=g) * scale
    return t5, t9, t12


def _toy_tree(device='cpu', scale=1.0, seed=0):
    """A 3-variable proposal tree with a genuinely non-uniform, correlated q.

    Built from real `FastFactor`s and real `BucketRecord`s so `sample()` runs
    its actual conditioning code, not a mock. Levels are in ELIMINATION order,
    and `sample()` traverses them in reverse, so level k's factors may only
    mention variables from levels >= k.
    """
    from nce.inference.factor import FastFactor
    from nce.sampling.proposal_sampler import BucketRecord, ProposalTree
    t5, t9, t12 = _toy_tensors(scale, seed)
    return ProposalTree([BucketRecord(5, 3, [FastFactor(t5.clone(), [5, 9])]),
                         BucketRecord(9, 2, [FastFactor(t9.clone(), [9, 12])]),
                         BucketRecord(12, 4, [FastFactor(t12.clone(), [12])])],
                        device)


def _toy_exact_q(scale=1.0, seed=0):
    """q over the 3*2*4 = 24 joint states, by brute force.

    The tree normalises LEVEL BY LEVEL, so q factorises as
    q(x12) q(x9 | x12) q(x5 | x9) with each conditional the normalised local
    potential -- NOT the joint normalisation of the product, which would be a
    different distribution.
    """
    t5, t9, t12 = _toy_tensors(scale, seed)
    ln = math.log(10)

    def norm(t, dim):
        e = (t.double() * ln).exp()
        return e / e.sum(dim=dim, keepdim=True)

    p12 = norm(t12, 0)                    # (4,)
    p9 = norm(t9, 0)                      # (2, 4), normalised over x9 | x12
    p5 = norm(t5, 0)                      # (3, 2), normalised over x5 | x9
    q = p5[:, :, None] * p9[None, :, :] * p12[None, None, :]
    assert abs(float(q.sum()) - 1.0) < 1e-9
    return q


TOY_KEY = crn.stream_key(SEED, [5, 9, 12], [3, 2, 4], crn.ROLE_PROP_TREE, 0)


def test_proposal_tree_crn_is_independent_of_the_global_rng():
    tree = _toy_tree()
    _burn_global_rng(5, 1000)
    a, lpa = tree.sample(3000, crn_key=TOY_KEY)
    _burn_global_rng(11, 3)
    b, lpb = tree.sample(3000, crn_key=TOY_KEY)
    assert all(torch.equal(a[v], b[v]) for v in a)
    assert torch.equal(lpa, lpb)

    # The multinomial path does depend on it -- otherwise the assertion above
    # would hold for reasons unrelated to CRN.
    _burn_global_rng(5, 1000)
    m1, _ = tree.sample(3000)
    _burn_global_rng(11, 3)
    m2, _ = tree.sample(3000)
    assert any(not torch.equal(m1[v], m2[v]) for v in m1), (
        'torch.multinomial gave the same samples from two different global RNG '
        'states, so this test does not discriminate CRN from the legacy path.')


def test_proposal_tree_crn_is_prefix_closed():
    tree = _toy_tree()
    small, _ = tree.sample(700, crn_key=TOY_KEY)
    large, _ = tree.sample(5000, crn_key=TOY_KEY)
    for v in small:
        assert torch.equal(large[v][:700], small[v]), (
            'variable %d: the 700-row proposal draw is not a prefix of the '
            '5000-row one' % v)


def test_proposal_tree_column_is_keyed_on_the_variable_not_the_depth():
    """A variable at a different depth in two arms' trees must draw the same column."""
    u_alone = crn.uniform01(50, [9], TOY_KEY)
    u_first = crn.uniform01(50, [9, 12], TOY_KEY)
    u_second = crn.uniform01(50, [12, 9], TOY_KEY)
    assert torch.equal(u_alone[:, 0], u_first[:, 0])
    assert torch.equal(u_alone[:, 0], u_second[:, 1])
    assert not torch.equal(u_first[:, 0], u_first[:, 1])


def test_proposal_tree_crn_samples_the_right_distribution():
    """The inverse CDF must reproduce q, not merely be deterministic.

    A `searchsorted` off by one, or a CDF built on the wrong axis, would pass
    every equality test above and silently sample the wrong proposal -- which
    biases nothing (the IS weights use the reported log q) but destroys the
    variance reduction the proposal exists for. Chi-square over all 24 joint
    states against the brute-force q, at the 1e-6 upper tail for df = 23
    (~68.0), and the same check on the multinomial path as a control.
    """
    n = 200000
    q = _toy_exact_q().reshape(-1)
    tree = _toy_tree()
    for label, kw in (('crn', dict(crn_key=TOY_KEY)), ('multinomial', {})):
        s, _ = tree.sample(n, **kw)
        flat = s[5] * 8 + s[9] * 4 + s[12]
        counts = torch.bincount(flat, minlength=24).double()
        expected = q * n
        chi2 = float(((counts - expected) ** 2 / expected).sum())
        assert chi2 < 68.0, (
            '%s path does not sample the proposal distribution: chi2 %.1f over '
            '24 states\n counts  %r\n expected %r'
            % (label, chi2, counts.tolist(), [round(float(e), 1) for e in expected]))


def test_proposal_tree_reports_the_log_prob_of_what_it_sampled():
    """log q must belong to the row that was drawn -- the IS weight depends on it."""
    tree = _toy_tree()
    s, lp = tree.sample(4000, crn_key=TOY_KEY)
    q = _toy_exact_q()
    want = torch.log10(q[s[5], s[9], s[12]].double())
    assert torch.allclose(lp.double(), want, atol=1e-4), (
        'reported log10 q does not match the brute-force joint probability of '
        'the sampled state; max err %.3e'
        % float((lp.double() - want).abs().max()))


def test_a_different_proposal_gives_different_but_coupled_samples():
    """The requirement's carve-out, measured.

    "If proposal hyperparameters genuinely change the proposal distribution,
    different samples are fine." They do differ -- but under a shared uniform
    they are far MORE likely to coincide than two independent draws would be,
    which is the whole point of driving the tree by inverse CDF instead of
    re-seeding a generator. The independence baseline is sum_x q1(x) q2(x),
    computed exactly, not estimated.
    """
    n = 40000
    a, _ = _toy_tree(scale=1.0).sample(n, crn_key=TOY_KEY)
    b, _ = _toy_tree(scale=2.5).sample(n, crn_key=TOY_KEY)
    agree = torch.stack([a[v] == b[v] for v in (5, 9, 12)]).all(dim=0)
    frac = float(agree.double().mean())

    q1 = _toy_exact_q(scale=1.0).reshape(-1)
    q2 = _toy_exact_q(scale=2.5).reshape(-1)
    baseline = float((q1 * q2).sum())

    assert frac < 1.0, (
        'the two proposals produced identical samples, so `scale` did not '
        'change q and this test measures nothing')
    assert frac > baseline + 0.05, (
        'shared-uniform coupling bought nothing: agreement %.3f vs independent '
        'baseline %.3f' % (frac, baseline))
