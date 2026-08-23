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

    HISTORY. The shipped function used to pick the scope itself via
    `proposal_scope_for_bucket`, which for a bucket with a SINGLE elim var read
    the pre-merge `gm.message_scopes` cache -- a pre-elimination artefact. This
    copy was written to bypass that.

    AS OF DOC 57 the two are equivalent: `proposal_scope_for_bucket` now returns
    `bucket.get_message_scope()` unconditionally, and `build_proposal_for_bucket`
    unions in the extra domain sizes this copy always did. Kept rather than
    deleted only because collapsing it is a refactor on a path that is currently
    verified, and this file is on a branch that still has to merge. If you are
    touching this area, `build_proposal_for_bucket(bucket, gm, ecl, temperature)`
    should now do the same thing.
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
    # shared-prefix property: the sampler's phase-1 frontier schedule is a
    # function of N, so two runs with different `memorize_num_samples` share
    # nothing (doc 56 section 2c).
    #
    # MEASURED (doc 57), and it changes what this line is worth:
    # `sample_no_replacement_v3_recursive` reaches its RNG only when
    # `K_outer < N`, and phase 1 over-delivers on every tree probed, so the
    # generator is NEVER advanced and this seed is not an input to the draw at
    # all. The re-key is insurance against the day that changes, not a repair.
    # `test_the_no_replacement_sampler_never_consumes_its_generator` is the
    # tripwire.
    rng = _crn.no_replacement_generator(config, scope, domain_sizes, device,
                                        role=_crn.ROLE_MEMO)
    nr_samples, _, _ = tree.sample_no_replacement_v3_recursive(
        n_samp, M=1, rng=rng, mode='save')
    # The proposal tree does not always span the whole separator (its variables
    # come from the upstream/downstream factors); the missing coordinates are
    # filled uniformly at random. (proposal_in_elim.py used to raise a KeyError
    # here; as of doc 57 it fills too, and additionally corrects log q by the
    # fill's own density, which matters there because its weights are 1/q and
    # does not matter here because the top-K is by TRUE value.)
    # Correctness is unaffected -- the TRUE value is still
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
        # Selection by forward alone memorises the entries with the largest
        # message value; what actually matters for log Z is the entry's
        # CONTRIBUTION, forward x backward. Score on that instead.
        #
        # The hard part is getting a backward message without enabling
        # `use_bw_approx`, which doc 57 showed gates THREE things at once:
        #   (1) attaching bw factors to the dataloader,
        #   (2) flipping DataPreprocessor.use_bw_approx (the normalising
        #       constant becomes logsumexp(y+bw) - logsumexp(bw)), and
        #   (3) making `bw_hat` non-None in the loss, which literally rewrites
        #       `outputs` and `targets`.
        # Turning the flag on to obtain (1) also buys (2) and (3), so the arm
        # would no longer be a pure selection change and the comparison would be
        # confounded. `FastBucket.get_backward_factor_list()` (doc 54) returns
        # exactly (1) and nothing else: it reads the already-populated
        # `approximate_downstream_factors`, sets no trainer state, and -- because
        # it always passes a non-None `backward_factors` -- never takes
        # `_get_backward_factors`' branch that mutates the live GM.
        #
        # Note this only affects WHICH entries are chosen. `y_sel` below is
        # always the true FORWARD value, and `dp.normalize(y_final, None)` is
        # called with bw_vals=None, so neither the stored values nor the
        # preprocessor's constants can move.
        bw_factors = getattr(trainer.dataloader, 'bw_factors', None)
        bw_mod = getattr(trainer.dataloader, 'bw_modifier', None)
        bw_list = bw_factors if bw_factors is not None else (
            [bw_mod] if bw_mod is not None else None)
        bw_source = 'dataloader' if bw_list is not None else None
        if bw_list is None:
            # Independent approximate backward factors, built from the existing
            # cluster structure, without touching the training path.
            try:
                bw_list = bucket.get_backward_factor_list()
                if bw_list is not None:
                    bw_source = 'get_backward_factor_list'
            except Exception as _bwe:
                stats['bw_error'] = f"{type(_bwe).__name__}: {_bwe}"
                bw_list = None
        if bw_list is not None:
            bw_log10 = trainer.sample_generator.compute_backward_values(
                assignments, backward_factors=bw_list)
            score = y_log10 + bw_log10
            stats['selection_effective'] = 'fw_bw'
            stats['bw_source'] = bw_source
            stats['bw_num_factors'] = len(bw_list)
            with torch.no_grad():
                _f = torch.isfinite(bw_log10)
                stats['bw_finite_frac'] = float(_f.float().mean())
                if _f.any():
                    stats['bw_range_log10'] = [float(bw_log10[_f].min()),
                                               float(bw_log10[_f].max())]
                # How much does the backward actually re-order the entries? If
                # this is ~1.0 the two arms memorise the same set and a null
                # result means "no difference to make", not "no benefit".
                _k = min(int(_budget(msg_size,
                                     config.get('memorize_top_k', 0),
                                     config.get('memorize_frac', 0.0))),
                         int(score.numel()))
                _a = set(torch.topk(y_log10, _k).indices.tolist())
                _b = set(torch.topk(score, _k).indices.tolist())
                stats['topk_overlap_frac'] = len(_a & _b) / max(1, _k)
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

    # ---- 3b. how much MASS did we actually catch? ------------------------ #
    # The memoriser is a hedge against LOW-ENTROPY messages: cases where a
    # handful of entries hold most of the probability mass, which uniform
    # training samples are unlikely ever to draw. So the diagnostic that matters
    # is not "what fraction of entries did we cover" -- that necessarily falls as
    # messages grow, and is expected to -- but "what fraction of the MASS did the
    # kept entries hold". A cap of 100k entries covering 0.16% of a 2^26 message
    # is doing its job if those 100k carry most of the mass, and is not doing its
    # job if the message is near-uniform (in which case there was nothing to
    # hedge and the NN was already fine).
    #
    # Computed over the SAMPLED POOL, not the full message: these samples come
    # from the WMB proposal without replacement, so this is the captured share of
    # the sampled mass, not an unbiased estimate of the true share. It is still
    # the right shape -- near 1.0 means concentrated and caught, near k/n means
    # flat -- and it needs no extra message evaluations.
    with torch.no_grad():
        _fin = torch.isfinite(y_log10)
        if _fin.any():
            _m = float(y_log10[_fin].max())
            _lin_all = torch.exp((y_log10[_fin] - _m) * math.log(10.0))
            _ysel_fin = y_sel[torch.isfinite(y_sel)]
            _lin_sel = torch.exp((_ysel_fin - _m) * math.log(10.0))
            _tot = float(_lin_all.sum())
            stats['mass_captured_frac_of_sampled'] = (
                float(_lin_sel.sum()) / _tot if _tot > 0 else None)
            # Concentration of the sampled pool itself, as a reference point:
            # what share sits in the top 10 and top 100 entries.
            _srt, _ = torch.sort(_lin_all, descending=True)
            stats['sampled_mass_top10'] = float(_srt[:10].sum()) / _tot if _tot > 0 else None
            stats['sampled_mass_top100'] = float(_srt[:100].sum()) / _tot if _tot > 0 else None
            stats['entries_kept_frac_of_sampled'] = float(_ysel_fin.numel()) / float(_fin.sum())

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
