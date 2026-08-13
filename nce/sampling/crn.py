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

    rows = torch.arange(int(num_samples), dtype=torch.long, device=device)
    cols = torch.arange(n_cols, dtype=torch.long, device=device)
    # Mix row and column indices SEPARATELY before combining, so that two keys
    # differing by a small amount cannot produce streams that are shifts of one
    # another (`mix64(key + row)` would have exactly that defect).
    row_seed = mix64(key ^ mix64(rows))
    h = mix64(row_seed.unsqueeze(1) ^ mix64(cols).unsqueeze(0))

    top32 = _lshr(h, 32)                      # uniform on [0, 2**32)
    return (top32 * doms) >> 32               # < 2**42, no overflow


def assignments_digest(assignments):
    """Stable content digest of an assignment tensor, for test reporting."""
    a = assignments.detach().to('cpu').contiguous().to(torch.int64)
    return hashlib.sha256(a.numpy().tobytes()).hexdigest()[:16]
