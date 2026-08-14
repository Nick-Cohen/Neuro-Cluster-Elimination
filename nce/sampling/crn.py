"""Common random numbers (CRN) for separator-assignment sampling.

WHY
---
The paper rerun compares merge strategies on the SAME problems. Those arms are
paired: the interesting quantity is (log Z of arm A) - (log Z of arm B) on one
problem. If each arm draws its training assignments from an independent stream,
the sampling noise of the two arms is independent and the variance of the
DIFFERENCE is the SUM of the two variances. If both arms evaluate the same
separator at the SAME assignment points, the sampling noise is shared and
largely cancels in the difference. That is common random numbers.

CRN only helps where the arms actually share something. Two arms share a
separator whenever their cluster trees happen to produce a cluster with the same
outgoing scope -- which is common, because merging changes some clusters and
leaves others alone. Where the separators differ there is nothing to pair and
this module makes no attempt to force one (see "no pairing when separators
differ" in the design doc).

THE STREAM IS A PURE FUNCTION, NOT A SEEDED GENERATOR
-----------------------------------------------------
`assignment[i][j] = f(key, i, j)` -- a counter-based RNG. Nothing is consumed,
nothing is advanced, no global state is touched. Three properties follow, and
all three are the reason this is not implemented as `torch.manual_seed(k)`
followed by `torch.randint`:

1. ORDER INDEPENDENCE. The value at row i does not depend on how many rows were
   drawn before it, nor on which OTHER separators were sampled earlier in the
   elimination. A seeded generator gives this only if it is reseeded per draw.

2. SHARED PREFIX. Drawing N rows and drawing M > N rows agree exactly on the
   first N rows, for every N and M. A reseeded `torch.randint` does NOT have
   this property: it fills column 0 with N draws, then column 1 with the NEXT N
   draws, so changing N shifts every column after the first. This is the single
   property that forced a counter-based design.

3. DEVICE INDEPENDENCE. Only integer arithmetic is used, so CPU and CUDA
   produce identical assignments. (torch's own RNG does not: the CUDA and CPU
   engines are different.)

THE KEY
-------
The canonical key is the SEPARATOR AS A SET OF (label, domain size) pairs,
together with the run's experimental seed, the role (train/val), and the draw
index. It deliberately does NOT include the bucket label, the eliminated
variables, the cluster scope, the merge strategy, or the elimination position.
Justification is in the design doc; the short form is that the separator set
already determines the column order, because `SampleGenerator` builds its
message scope as `sorted(scope)`, so "the separator set" and "the separator set
plus its ordering" are the same key. Including anything strategy-dependent
would break pairing exactly in the case CRN exists to serve.

64-BIT ARITHMETIC
-----------------
The mixer is splitmix64 on `torch.int64`, which relies on multiplication
wrapping modulo 2**64. MEASURED (`tests/test_common_random_numbers.py::
test_int64_multiply_wraps_mod_2_64`) rather than assumed, and the mixer output
is additionally pinned against a pure-Python reference implementation, so a
torch version that ever stopped wrapping would fail loudly instead of silently
producing a different stream.
"""
import hashlib

import torch

# splitmix64 constants, as SIGNED int64 (torch has no unsigned 64-bit dtype).
_MASK64 = (1 << 64) - 1


def _signed(x):
    """Reinterpret a 64-bit unsigned constant as the int64 torch will hold."""
    x &= _MASK64
    return x - (1 << 64) if x >= (1 << 63) else x


_GAMMA = _signed(0x9E3779B97F4A7C15)
_MUL1 = _signed(0xBF58476D1CE4E5B9)
_MUL2 = _signed(0x94D049BB133111EB)

STREAM_VERSION = 'nce-crn-v1'
SEED_VERSION = 'nce-seed-v1'

# Role tags. Kept as named constants because they are part of the stream
# identity: renaming one silently re-rolls every assignment drawn under it.
ROLE_TRAIN = 'train'            # sampling_scheme='uniform', training draw
ROLE_VAL = 'val'                # sampling_scheme='uniform', validation draw
ROLE_PROP_UNIFORM = 'prop-uniform'   # the uniform half of a mixed proposal
ROLE_PROP_TREE = 'prop-tree'         # the WMB-proposal half (inverse CDF)
ROLE_PROP_NR = 'prop-nr'             # the no-replacement half (generator seed)
ROLE_MEMO = 'memo'                   # build_memorization_table's NR sampler

# Draw indices for the proposal path. The training draw and the (optional)
# correction draw are two draws of the SAME role, separated by this index --
# never by perturbing the seed (`seed + 1`), which is the collision-prone
# family this branch removes: run seed 42's correction draw and run seed 43's
# training draw would share a stream.
DRAW_TRAIN = 0
DRAW_CORRECTION = 1


def _lshr(x, k):
    """Logical (unsigned) right shift of an int64 tensor by k bits.

    `x >> k` in torch is an ARITHMETIC shift on a signed dtype, so it smears the
    sign bit downwards and splitmix64 would be wrong for half of all inputs.
    Masking off the k high bits afterwards recovers the unsigned shift; the mask
    (1 << (64 - k)) - 1 is itself representable in int64 for every k >= 1.
    """
    return (x >> k) & ((1 << (64 - k)) - 1)


def mix64(x):
    """splitmix64 finalizer on an int64 tensor. Pure, elementwise, no state."""
    x = x + _GAMMA
    x = (x ^ _lshr(x, 30)) * _MUL1
    x = (x ^ _lshr(x, 27)) * _MUL2
    return x ^ _lshr(x, 31)


def mix64_py(x):
    """Pure-Python reference for `mix64`, used only by the tests.

    Kept next to the torch version so the two cannot drift apart unnoticed.
    """
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def stream_payload(seed, scope, domain_sizes, role, draw_index):
    """The exact string the stream key is derived from.

    Returned (not just hashed) so tests and debugging output can show WHY two
    streams are or are not identical -- a bare 64-bit key is undiagnosable.

    `scope` is sorted here rather than trusted, so the key is a function of the
    separator SET even if a caller ever hands over a differently ordered list.
    The (label, domain size) pairs are sorted together, so the payload also
    pins the column order the assignments will be produced in.
    """
    pairs = sorted((int(l), int(d)) for l, d in zip(scope, domain_sizes))
    return '%s|seed=%d|role=%s|draw=%d|sep=%s' % (
        STREAM_VERSION, int(seed), role, int(draw_index),
        ','.join('%d:%d' % p for p in pairs))


def stream_key(seed, scope, domain_sizes, role, draw_index):
    """64-bit signed stream key. SHA-256 of the payload, truncated.

    A cryptographic digest (rather than Python's `hash`) because it must be
    stable across processes and across PYTHONHASHSEED.
    """
    payload = stream_payload(seed, scope, domain_sizes, role, draw_index)
    digest = hashlib.sha256(payload.encode('utf-8')).digest()[:8]
    return int.from_bytes(digest, 'big', signed=True)


def stream_seed(seed, scope, domain_sizes, role, draw_index):
    """A non-negative 63-bit RNG seed for the same stream identity as `stream_key`.

    For samplers that cannot be re-expressed as a counter-based pure function --
    the no-replacement Gumbel-top-k sampler is the only one in this project --
    the next best thing is a stateful generator whose SEED is a function of the
    separator rather than of the bucket label or of the execution order. That
    buys arm pairing (same separator + same proposal tree => same samples) and
    collision-freedom; it does NOT buy the shared-prefix property, which for a
    stateful generator is unobtainable. Said plainly in the design doc.
    """
    return stream_key(seed, scope, domain_sizes, role, draw_index) & ((1 << 63) - 1)


def seed_payload(kind, **parts):
    """The exact string a `derive_seed` value is a digest of."""
    body = '|'.join('%s=%s' % (k, parts[k]) for k in sorted(parts))
    return '%s|kind=%s|%s' % (SEED_VERSION, kind, body)


def derive_seed(kind, **parts):
    """Collision-free replacement for arithmetic seed formulas.

    WHY THIS EXISTS. The pipeline used to build RNG seeds by adding scaled
    integers together -- `bucket_label + 10000*seed + 100*draw_index` in
    `SampleGenerator._compute_seed`, `seed*1000003 + bucket.label` in
    `build_memorization_table`, and a bare `seed` (or `seed + 1`) for the
    proposal generators. The first of those is not injective: bucket 145 draw 0
    and bucket 45 draw 1 produce the same seed, so two different separators are
    sampled from one stream. MEASURED (doc 46 appendix) on 173 of 519 study
    cells, with 63 pairs byte-identical. The third is worse in a different way:
    every bucket in a run shares one seed, and `seed + 1` for the correction
    draw is the training stream of the run at `seed + 1`.

    Field-tagged digest instead of arithmetic: distinct `(kind, parts)` tuples
    give unrelated 63-bit values, and there is no carry structure for two
    different tuples to exploit. Field names are sorted and the values are
    rendered as `name=value`, so `(label=1, draw=23)` and `(label=12, draw=3)`
    cannot collapse onto the same payload the way `1 + 100*23` and `12 + 100*3`
    almost can. Range is [0, 2**63), accepted by both `torch.manual_seed` and
    `torch.Generator.manual_seed`; `np.random.seed` still needs the caller's
    `% 2**31`.

    Not injective in the mathematical sense -- it is a truncated SHA-256, so
    collisions exist in principle at a rate of ~n**2 / 2**64. Exhaustively
    checked to have none over the pedigree-scale domain the old formula broke
    on: `tests/test_common_random_numbers.py::
    test_derive_seed_has_no_collisions_over_the_pedigree_domain`.
    """
    payload = seed_payload(kind, **parts)
    digest = hashlib.sha256(payload.encode('utf-8')).digest()[:8]
    return int.from_bytes(digest, 'big', signed=False) >> 1


def _hash_grid(key, num_rows, col_ids, device='cpu'):
    """h[i, j] = mix64(mix64(key ^ mix64(i)) ^ mix64(col_ids[j])), as int64.

    `col_ids` is the COLUMN IDENTITY, not merely a position. For separator
    assignments it is the position within `sorted(scope)` (positions and the
    set determine each other, see the module docstring). For the proposal path
    it is the VARIABLE LABEL being sampled at that level, so that a variable
    appearing in two arms' proposal trees at different depths still draws from
    the same column -- the same "key on the object, not on the execution
    position" argument that chose the separator over the bucket label.
    """
    rows = torch.arange(int(num_rows), dtype=torch.long, device=device)
    cols = torch.as_tensor(list(col_ids), dtype=torch.long, device=device)
    # Mix row and column indices SEPARATELY before combining, so that two keys
    # differing by a small amount cannot produce streams that are shifts of one
    # another (`mix64(key + row)` would have exactly that defect).
    row_seed = mix64(key ^ mix64(rows))
    return mix64(row_seed.unsqueeze(1) ^ mix64(cols).unsqueeze(0))


def uniform01(num_rows, col_ids, key, device='cpu'):
    """Rows 0..num_rows-1 of the CRN stream as float64 uniforms on [0, 1).

    Same pure-function contract as `uniform_assignments`: value (i, j) depends
    only on (key, i, col_ids[j]). Used to drive the WMB proposal tree by
    inverse CDF, which is what makes the proposal path paired: two arms whose
    proposal distribution is identical draw identical samples, and two arms
    whose proposal distribution genuinely differs draw DIFFERENT but
    inverse-CDF-COUPLED samples (the strongest coupling available), instead of
    the independent streams a global-RNG `torch.multinomial` gives.

    Exactly representable: `top32 / 2**32` with top32 < 2**32 is exact in
    float64, so the value is a deterministic function of the integer stream and
    cannot wobble with the FPU.
    """
    if num_rows <= 0 or len(col_ids) == 0:
        return torch.zeros((max(0, int(num_rows)), len(col_ids)),
                           dtype=torch.float64, device=device)
    h = _hash_grid(key, num_rows, col_ids, device=device)
    return _lshr(h, 32).to(torch.float64) / float(1 << 32)


def uniform_assignments(num_samples, scope, domain_sizes, seed, role,
                        draw_index, device='cpu'):
    """Rows 0..num_samples-1 of the CRN stream for one separator.

    Returns an int64 tensor of shape (num_samples, len(scope)); column j ranges
    over [0, domain_sizes[j]) and corresponds to `sorted(scope)[j]`.

    Uniformity: the value is `floor(u * d)` with u the top 32 bits of a
    splitmix64 output scaled to [0, 1). The modulo bias is at most d / 2**32
    (< 3e-8 for any domain size in this project) -- far below the sampling noise
    it lives inside, and the alternative (rejection sampling) would destroy the
    pure-function property this whole module exists for.
    """
    n_cols = len(scope)
    if num_samples <= 0 or n_cols == 0:
        return torch.zeros((max(0, int(num_samples)), n_cols),
                           dtype=torch.long, device=device)

    order = sorted(range(n_cols), key=lambda i: int(scope[i]))
    doms = torch.tensor([int(domain_sizes[i]) for i in order],
                        dtype=torch.long, device=device)
    if int(doms.min()) < 1:
        raise ValueError('CRN: non-positive domain size in %r' % (list(doms),))

    key = stream_key(seed, scope, domain_sizes, role, draw_index)
    h = _hash_grid(key, num_samples, range(n_cols), device=device)

    top32 = _lshr(h, 32)                      # uniform on [0, 2**32)
    return (top32 * doms) >> 32               # < 2**42, no overflow


# ---------------------------------------------------------------------------
# Proposal-path helpers.
#
# The proposal arms draw from three different mechanisms and each needs a
# different treatment. The rule that decides all three is the requirement's own
# carve-out: the key must contain whatever GENUINELY determines the proposal
# distribution and nothing else.
#
#   uniform half   q is uniform on the separator, so the separator determines q
#                  completely -> full CRN, same mechanism as the uniform
#                  sampling scheme, prefix property and all.
#   WMB tree half  q is determined by the proposal tree (the cluster's factors,
#                  proposal_ecl, proposal_temperature). Those are NOT in the
#                  key: putting them in would break pairing between arms whose
#                  q is in fact identical, and leaving them out cannot create
#                  false pairing, because the samples are the uniforms pushed
#                  through q -- a different q yields different samples on its
#                  own. Inverse CDF off the shared uniform.
#   no-replacement Gumbel-top-k over an algorithm-dependent frontier; it cannot
#                  be re-expressed as a counter-based pure function without
#                  redesigning that sampler. Its GENERATOR SEED is keyed on the
#                  separator instead, which gives pairing and collision-freedom
#                  but not the prefix property.
# ---------------------------------------------------------------------------
def enabled(config):
    """Is CRN on for this config? Default True since 2026-08-14."""
    return bool((config or {}).get('common_random_numbers', True))


def _run_seed(config):
    return int((config or {}).get('seed', 42))


def proposal_uniform(config, num_samples, scope, domain_sizes, device,
                     draw_index=DRAW_TRAIN):
    """The uniform half of a mixed proposal draw, CRN or legacy.

    Legacy behaviour (`common_random_numbers=False`) is the bare per-column
    `torch.randint` the call sites used to inline, kept byte-identical so the
    flag still means "the old sampler".
    """
    n = int(num_samples)
    if not enabled(config):
        return torch.stack(
            [torch.randint(0, int(d), (n,), device=device, dtype=torch.long)
             for d in domain_sizes], dim=1)
    return uniform_assignments(
        num_samples=n, scope=scope, domain_sizes=[int(d) for d in domain_sizes],
        seed=_run_seed(config), role=ROLE_PROP_UNIFORM, draw_index=draw_index,
        device=device)


def proposal_tree_key(config, scope, domain_sizes, draw_index=DRAW_TRAIN):
    """Stream key for `ProposalTree.sample`, or None to keep the legacy path."""
    if not enabled(config):
        return None
    return stream_key(_run_seed(config), scope, [int(d) for d in domain_sizes],
                      ROLE_PROP_TREE, draw_index)


def no_replacement_generator(config, scope, domain_sizes, device,
                             draw_index=DRAW_TRAIN, role=ROLE_PROP_NR):
    """A `torch.Generator` for the no-replacement sampler, seeded off the separator.

    Unconditional -- it does NOT check `common_random_numbers`. What it replaces
    (`rng.manual_seed(int(seed))`, and `int(seed) + 1` for the correction draw)
    is not a legacy behaviour worth preserving under any flag: it gave every
    cluster in a run the SAME stream, and it made run `seed`'s correction draw
    identical to run `seed + 1`'s training draw. Removing collision-prone seed
    derivation is a separate requirement from CRN and applies either way.
    """
    rng = torch.Generator(device=device)
    rng.manual_seed(stream_seed(_run_seed(config), scope,
                                [int(d) for d in domain_sizes], role, draw_index))
    return rng


def assignments_digest(assignments):
    """Stable content digest of an assignment tensor, for test reporting."""
    a = assignments.detach().to('cpu').contiguous().to(torch.int64)
    return hashlib.sha256(a.numpy().tobytes()).hexdigest()[:16]
