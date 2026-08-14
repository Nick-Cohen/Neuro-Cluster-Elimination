"""Hybrid NN + memorization table.

Two pieces:

* :class:`HybridMemorizerNet` -- the *splice*. Wraps an already-trained ``Net``
  plus a sorted key/value table and, on every forward, substitutes the memorized
  value wherever the one-hot input row matches a memorized assignment. Fully
  vectorized (one fp64 mat-vec + one ``searchsorted``), no Python loop, so it is
  usable against the 65 536-row batched inference path in ``factor_nn.py``.

* :func:`build_memorization_table` -- draws no-repeat samples from the bucket's
  WMB proposal tree, evaluates the TRUE message value at them, and keeps the
  top-K.

Nothing here runs unless ``config['use_memorization_table']`` is true, and the
pre-existing ``Memorizer`` class in ``net.py`` is untouched.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn

from nce.sampling import crn as _crn


# --------------------------------------------------------------------------- #
# key packing
# --------------------------------------------------------------------------- #
def _radix_weights(domain_sizes):
    """Row-major mixed-radix weights: weight[i] = prod(domain_sizes[i+1:])."""
    w = [1] * len(domain_sizes)
    acc = 1
    for i in range(len(domain_sizes) - 1, -1, -1):
        w[i] = acc
        acc *= int(domain_sizes[i])
    return w, acc


def pack_assignments(assignments: torch.Tensor, domain_sizes) -> torch.Tensor:
    """(n, n_vars) integer assignments -> (n,) int64 mixed-radix key."""
    weights, _ = _radix_weights(domain_sizes)
    w = torch.tensor(weights, dtype=torch.int64, device=assignments.device)
    return (assignments.to(torch.int64) * w).sum(dim=1)


def onehot_projection(domain_sizes, lower_dim: bool):
    """Vector p with  onehot_row @ p == packed key.

    The one-hot layout is the one produced by
    ``DataPreprocessor.one_hot_encode`` / ``FactorNN._onehot_layout``: variable
    ``i`` owns a contiguous block of ``d_i - 1`` columns when ``lower_dim`` (the
    all-zero row meaning value 0, column ``j`` meaning value ``j + 1``) and of
    ``d_i`` columns otherwise (column ``j`` meaning value ``j``).
    """
    weights, total = _radix_weights(domain_sizes)
    p = []
    for d, wgt in zip(domain_sizes, weights):
        d = int(d)
        if lower_dim:
            p.extend((j + 1) * wgt for j in range(d - 1))
        else:
            p.extend(j * wgt for j in range(d))
    return p, total


# --------------------------------------------------------------------------- #
# the splice
# --------------------------------------------------------------------------- #
class HybridMemorizerNet(nn.Module):
    """A trained ``Net`` with a top-K exact lookup table spliced over it.

    ``forward`` returns the base net's output everywhere except on the memorized
    assignments, where it returns the stored (NORMALIZED-space) true value. The
    override is applied AFTER the base net's own output transform, so a
    ``masked_net`` -inf mask cannot zero a memorized entry.
    """

    def __init__(self, base_net, keys: torch.Tensor, values: torch.Tensor,
                 bucket, lower_dim: bool):
        super().__init__()
        self.base = base_net
        self.bucket = bucket
        self.gm = bucket.gm
        self.device = base_net.device

        domain_sizes = bucket.get_message_dimension()
        proj, total = onehot_projection(domain_sizes, lower_dim)
        if total > 2 ** 52:
            raise ValueError(
                f"message size {total} exceeds the fp64-exact packing range; "
                "HybridMemorizerNet cannot key this bucket")
        self.register_buffer('_proj',
                             torch.tensor(proj, dtype=torch.float64,
                                          device=self.device))
        order = torch.argsort(keys)
        self.register_buffer('_keys', keys[order].to(torch.int64).to(self.device))
        self.register_buffer('_vals',
                             values[order].reshape(-1).to(self.device))
        self.n_memorized = int(self._keys.numel())
        self.message_size = float(total)

    # nn.Module plumbing -------------------------------------------------- #
    def train(self, mode: bool = True):
        super().train(mode)
        self.base.train(mode)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.n_memorized == 0:
            return out
        packed = torch.round(x.to(torch.float64) @ self._proj).to(torch.int64)
        idx = torch.searchsorted(self._keys, packed).clamp_(max=self.n_memorized - 1)
        hit = self._keys[idx] == packed
        memo = self._vals[idx].to(out.dtype).reshape(out.shape)
        return torch.where(hit.reshape(out.shape), memo, out)


# --------------------------------------------------------------------------- #
# table construction
# --------------------------------------------------------------------------- #
def _proposal_tree_over_scope(bucket, gm, scope, ecl=0, temperature=1.0):
    """`build_proposal_for_bucket`, but over the scope WE ask for.

    The shipped function picks the scope itself via `proposal_scope_for_bucket`,
    which for a bucket with a SINGLE elim var reads the pre-merge
    `gm.message_scopes` cache. That cache is computed before any merge pass, so
    once a neighbour has been merged the messages actually arriving at an
    unmerged bucket can span variables the cache never predicted, and
    `build_proposal_tree` then raises `KeyError: <label>` on the missing domain
    size. Measured on `dbn/rbm_20` under reduce-NN: 11/11 NN clusters raise.
    Doc 10's defect-4 fix covered the MERGED branch of the same function and
    left this one alone.

    We already know the correct scope -- `bucket.get_message_scope()`, which is
    recomputed from the bucket's current factors -- so pass it in. Everything
    else mirrors `proposal_sampler.build_proposal_for_bucket` verbatim.
    """
    from nce.inference.factor import FastFactor
    from nce.sampling.proposal_sampler import ProposalTree, build_proposal_tree

    up = bucket.approximate_upstream_factors or []
    down = bucket.approximate_downstream_factors or []
    all_factors = list(up) + list(down)
    if not all_factors or not scope:
        return ProposalTree([], gm.device)
    if temperature != 1.0:
        all_factors = [FastFactor(f.tensor / temperature, list(f.labels))
                       for f in all_factors]

    elim_var_labels = sorted({getattr(v, 'label', v) for v in bucket.elim_vars})
    domain_sizes = {v: gm.matching_var(v).states for v in scope}
    for lab in elim_var_labels:
        domain_sizes[lab] = gm.matching_var(lab).states
    # Any label the backward chain carries that is neither in the separator nor
    # an elim var still needs a domain size for the WMB step below.
    for f in all_factors:
        for lab in f.labels:
            if lab not in domain_sizes:
                domain_sizes[lab] = gm.matching_var(lab).states

    msg_factors = gm._wmb_eliminate_to_scope(
        all_factors, list(scope), gm.matching_var(elim_var_labels[0]))
    return build_proposal_tree(
        factors=msg_factors, message_scope=list(scope),
        domain_sizes=domain_sizes, ecl=ecl, device=gm.device, reference_gm=gm)


def _budget(msg_size, abs_cap, frac, floor_at_least=1):
    """Resolve a budget that is expressed as an absolute cap and/or a fraction."""
    cands = [msg_size]
    if abs_cap:
        cands.append(int(abs_cap))
    if frac:
        cands.append(int(math.ceil(float(frac) * msg_size)))
    return max(floor_at_least, int(min(cands)))


def build_memorization_table(bucket, trainer, config):
    """Sample -> evaluate TRUE values -> keep top-K.

    Returns ``(keys, values_normalized, stats)``. ``values_normalized`` are in
    the same normalized space the network was trained in, because every
    consumer in ``factor_nn.py`` applies ``data_processor.undo_normalization``
    to the net output.
    """
    from nce.sampling.proposal_sampler import build_proposal_for_bucket
    import nce.sampling.no_replacement_sampler_v3  # attaches the sampler method

    gm = bucket.gm
    device = config['device']
    scope = bucket.get_message_scope()
    domain_sizes = bucket.get_message_dimension()
    msg_size = 1
    for d in domain_sizes:
        msg_size *= int(d)

    n_samp = _budget(msg_size,
                     config.get('memorize_num_samples', 0),
                     config.get('memorize_sample_frac', 0.0))
    k_mem = _budget(msg_size,
                    config.get('memorize_top_k', 0),
                    config.get('memorize_frac', 0.0))
    k_mem = min(k_mem, n_samp)

    stats = {
        'bucket': bucket.label,
        'width': len(scope),
        'message_size': msg_size,
        'n_samples_requested': n_samp,
        'k_requested': k_mem,
        'selection': config.get('memorize_selection', 'fw_true'),
    }

    dp = trainer.data_preprocessor
    if dp.normalization_mode == 'minmax_01':
        if getattr(dp, 'ln_min', None) is None:
            raise RuntimeError(
                "memorization table built before the preprocessor's minmax "
                "constants were initialized -- would change the training "
                "normalization")
    elif getattr(dp, 'normalizing_constant', None) is None:
        raise RuntimeError("memorization table built before normalization was "
                           "initialized")

    # ---- 1. no-repeat samples from the WMB proposal tree ------------------ #
    t0 = time.time()
    tree = _proposal_tree_over_scope(
        bucket, gm, scope, ecl=config.get('bw_ecl', 0),
        temperature=float(config.get('proposal_temperature', 1.0)))
    # CRN site 1/2 -- the no-repeat sampler.
    #
    # Was `manual_seed(seed * 1000003 + bucket.label)`. That formula is
    # injective, so it had no collision defect; its problem is that the BUCKET
    # LABEL is exactly the execution artefact CRN exists to remove. Two merge
    # strategies that build a cluster with the same separator on different key
    # variables drew different memorization samples, so memorization arms could
    # not be paired across strategies. `no_replacement_generator` keys the seed
    # on the separator (scope + domain sizes) instead, under a `memo` role that
    # is disjoint from the proposal path's `prop-nr` stream.
    #
    # This buys pairing, order-independence and collision-freedom, NOT the
    # shared-prefix property: `sample_no_replacement_v3_recursive` is Gumbel
    # top-k over an N-dependent frontier, so two runs with different
    # `memorize_num_samples` still share nothing (see doc 56 section 2c).
    rng = _crn.no_replacement_generator(config, scope, domain_sizes, device,
                                        role=_crn.ROLE_MEMO)
    nr_samples, _, _ = tree.sample_no_replacement_v3_recursive(
        n_samp, M=1, rng=rng, mode='save')
    # The proposal tree does not always span the whole separator (its variables
    # come from the upstream/downstream factors). proposal_in_elim.py raises a
    # KeyError in that case; here the missing coordinates are filled uniformly
    # at random instead. Correctness is unaffected -- the TRUE value is still
    # evaluated at whatever assignment comes out, and the top-K is still by true
    # value -- but the candidate pool for those variables is not proposal-guided
    # and no-repeat is no longer guaranteed (duplicates are dropped below).
    missing = [v for v in scope if v not in nr_samples]
    n_rows = int(next(iter(nr_samples.values())).shape[0]) if nr_samples else n_samp
    # CRN site 2/2 -- the uniform fill, and the easier of the two to miss.
    #
    # Was `torch.randint(..., generator=rng)`, i.e. drawn off the *consumed*
    # no-repeat generator. Two defects in one: the draw depended on how many
    # numbers the NR sampler had already pulled (so it moved with
    # `memorize_num_samples` and with the tree's shape), and after site 1 above
    # it would have inherited the NR stream rather than having one of its own.
    # Here q IS uniform on the missing sub-scope, so that sub-scope determines q
    # completely and the full counter-based stream applies -- same argument as
    # the uniform halves of the proposal mixes (doc 56 section 2a). Keyed on the
    # missing sub-scope, not the whole separator: it is the scope this draw is
    # actually defined over, and if two arms' trees cover different variables
    # then q genuinely differs and the arms are entitled to differ.
    missing_sorted = sorted(missing)
    fill = None
    if missing_sorted:
        dom_of = {int(v): int(d) for v, d in zip(scope, domain_sizes)}
        fill = _crn.proposal_uniform(
            config, n_rows, missing_sorted,
            [dom_of[int(v)] for v in missing_sorted], device, _crn.DRAW_TRAIN)
    cols = []
    for v, d in zip(scope, domain_sizes):
        if v in nr_samples:
            cols.append(nr_samples[v].to(device))
        else:
            cols.append(fill[:, missing_sorted.index(v)])
    assignments = torch.stack(cols, dim=1).to(device)
    t_sample = time.time() - t0
    stats['n_samples_actual'] = int(assignments.shape[0])
    stats['n_scope_vars_not_in_tree'] = len(missing)
    stats['n_scope_vars'] = len(scope)
    # CRN audit trail. `separator` + `assignments_digest` make the pairing
    # property checkable from any run's gm.memorization_log without re-running:
    # two arms that share a separator must show the same digest.
    stats['separator'] = [int(v) for v in scope]
    stats['assignments_digest'] = _crn.assignments_digest(assignments)

    # ---- 2. TRUE message values at those assignments --------------------- #
    t0 = time.time()
    y_log10 = trainer.sample_generator.compute_message_values(assignments)
    score = y_log10
    if stats['selection'] == 'fw_bw':
        bw_factors = getattr(trainer.dataloader, 'bw_factors', None)
        bw_mod = getattr(trainer.dataloader, 'bw_modifier', None)
        bw_list = bw_factors if bw_factors is not None else (
            [bw_mod] if bw_mod is not None else None)
        if bw_list is not None:
            bw_log10 = trainer.sample_generator.compute_backward_values(
                assignments, backward_factors=bw_list)
            score = y_log10 + bw_log10
            stats['selection_effective'] = 'fw_bw'
        else:
            stats['selection_effective'] = 'fw_true (no bw available)'
    t_eval = time.time() - t0

    # ---- 3. top-K by the selection score --------------------------------- #
    k = min(k_mem, int(score.numel()))
    top = torch.topk(score, k)
    sel = top.indices
    y_sel = y_log10[sel]
    finite = torch.isfinite(y_sel)
    sel, y_sel = sel[finite], y_sel[finite]

    # Sort by key and drop duplicates (the NR sampler should not emit any, but a
    # duplicate would silently corrupt the (key, value) pairing).
    keys_all = pack_assignments(assignments[sel], domain_sizes)
    order = torch.argsort(keys_all)
    keys_sorted = keys_all[order]
    vals_sorted = y_sel[order]
    keep = torch.ones_like(keys_sorted, dtype=torch.bool)
    keep[1:] = keys_sorted[1:] != keys_sorted[:-1]
    keys = keys_sorted[keep]
    y_final = vals_sorted[keep]

    values, _ = dp.normalize(y_final, None)

    stats.update({
        'k_actual': int(keys.numel()),
        'memorized_fraction': float(keys.numel()) / float(msg_size),
        'sample_seconds': t_sample,
        'eval_seconds': t_eval,
        'y_top_log10': float(y_final.max()) if y_final.numel() else None,
        'y_min_kept_log10': float(y_final.min()) if y_final.numel() else None,
    })
    return keys, values, stats
