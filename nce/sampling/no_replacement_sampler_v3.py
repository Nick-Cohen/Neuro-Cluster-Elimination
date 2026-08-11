"""
Vectorized faithful re-implementation of v1's no-replacement algorithm.

Unlike v2, this preserves v1's iterative threshold-reduction semantics:

  Phase 1 (fixed point):
    while any unexpanded node has prob > 1/nws:
      - commit all above-threshold leaves at once (updates K, S → nws grows → threshold drops)
      - expand all above-threshold internals in parallel (batched by depth)
    Leaves below the final threshold plus frontier internal subtrees form the
    phase-2 frontier.

  Phase 2 (decimal resolution):
    Use v2's top-down walk starting from the frontier, distributing N - K samples.

Pool layout: parallel tensors with a preallocated, growable buffer.

  pool_log_probs   : (P,)            current node log-prob under proposal q
  pool_depths      : (P,)            depth (0 = root; num_levels = leaf)
  pool_partial     : (P, num_levels) padded partial assignment; only the first
                                     `depths[i]` columns of row i are valid
  pool_active      : (P,)            bool — node is still in play
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch

from nce.sampling.proposal_sampler import ProposalTree, BucketRecord
from nce.sampling.no_replacement_sampler_v2 import (
    _conditional_log_probs_batch,
    _systematic_round_batch,
)

FRAC_EPS = 1e-9


# ---------------------------------------------------------------------------
# Growable pool
# ---------------------------------------------------------------------------

class _Pool:
    """Preallocated buffer of unexpanded tree nodes, doubling when full.

    `partial_dtype` controls the storage dtype of the partial-state matrix.
    Default is int8 (fits domains up to 127, covers binary + most benchmarks).
    Pass torch.int16 / torch.int32 / torch.int64 if you need larger domains.
    """

    __slots__ = ('log_probs', 'depths', 'partial', 'active', 'num_levels',
                 'device', 'capacity', 'next_free', 'partial_dtype')

    def __init__(self, initial_capacity: int, num_levels: int, device: str,
                 partial_dtype: torch.dtype = torch.int8):
        self.num_levels = num_levels
        self.device = device
        self.partial_dtype = partial_dtype
        self.capacity = max(initial_capacity, 16)
        self.next_free = 0
        self.log_probs = torch.full(
            (self.capacity,), float('-inf'), device=device, dtype=torch.float64)
        self.depths = torch.zeros(self.capacity, dtype=torch.long, device=device)
        self.partial = torch.zeros(
            (self.capacity, num_levels), dtype=partial_dtype, device=device)
        self.active = torch.zeros(self.capacity, dtype=torch.bool, device=device)

    def _grow_to(self, new_capacity: int):
        assert new_capacity > self.capacity
        new_lp = torch.full(
            (new_capacity,), float('-inf'), device=self.device, dtype=torch.float64)
        new_de = torch.zeros(new_capacity, dtype=torch.long, device=self.device)
        new_pa = torch.zeros((new_capacity, self.num_levels),
                             dtype=self.partial_dtype, device=self.device)
        new_ac = torch.zeros(new_capacity, dtype=torch.bool, device=self.device)
        new_lp[:self.capacity] = self.log_probs
        new_de[:self.capacity] = self.depths
        new_pa[:self.capacity] = self.partial
        new_ac[:self.capacity] = self.active
        self.log_probs = new_lp
        self.depths = new_de
        self.partial = new_pa
        self.active = new_ac
        self.capacity = new_capacity

    def _ensure_space(self, num_new: int):
        if self.next_free + num_new > self.capacity:
            # First try to compact if that's enough
            if self.active.sum().item() + num_new <= self.capacity:
                self._compact()
            if self.next_free + num_new > self.capacity:
                target = max(self.capacity * 2, self.next_free + num_new)
                self._grow_to(target)

    def _compact(self):
        """Move all active nodes to the front, freeing inactive slots."""
        active_idx = self.active.nonzero(as_tuple=True)[0]
        n = active_idx.numel()
        if n == 0:
            self.next_free = 0
            return
        self.log_probs[:n] = self.log_probs[active_idx]
        self.depths[:n] = self.depths[active_idx]
        self.partial[:n] = self.partial[active_idx]
        # Clear inactive portion
        self.log_probs[n:].fill_(float('-inf'))
        self.depths[n:].zero_()
        self.partial[n:].zero_()
        self.active.zero_()
        self.active[:n] = True
        self.next_free = n

    def add(self, log_probs: torch.Tensor, depths: torch.Tensor,
            partial: torch.Tensor):
        num_new = log_probs.shape[0]
        if num_new == 0:
            return
        self._ensure_space(num_new)
        start = self.next_free
        end = start + num_new
        self.log_probs[start:end] = log_probs
        self.depths[start:end] = depths
        self.partial[start:end] = partial
        self.active[start:end] = True
        self.next_free = end


# ---------------------------------------------------------------------------
# Phase 1: iterative threshold-reduction expansion (vectorized)
# ---------------------------------------------------------------------------

def _phase1_vectorized(
    tree: ProposalTree,
    N: int,
    initial_pool_capacity: int = 8192,
    max_iters: int = 10_000,
    partial_dtype: torch.dtype = torch.int8,
    seed_log_prob: float = 0.0,
    seed_depth: int = 0,
    seed_partial: Optional[torch.Tensor] = None,
) -> Tuple[_Pool, int, float, torch.Tensor, torch.Tensor]:
    """
    Run v1's phase 1 in vectorized batches.

    Returns (pool, K, S) where:
      pool contains the remaining (below-threshold) frontier nodes
      K, S are the number and prob-mass of committed phase-1 leaves
    Phase-1 leaves themselves are stored inside the pool with active=False and
    depth=num_levels; callers should reconstruct them via compact_pool if needed.
    A separate `phase1_log_probs` / `phase1_partial` buffer is returned for them.
    """
    device = tree.device
    num_levels = len(tree.levels)

    pool = _Pool(initial_pool_capacity, num_levels, device, partial_dtype=partial_dtype)
    # Seed pool with starting node. Defaults to root (depth=0, log_prob=0, zeros),
    # but can be a subtree root when recursing.
    if seed_partial is None:
        seed_partial = torch.zeros(num_levels, dtype=partial_dtype, device=device)
    # Ensure seed_partial is (num_levels,) with correct dtype
    seed_partial_row = seed_partial.to(dtype=partial_dtype, device=device).view(num_levels)
    pool.add(
        log_probs=torch.tensor([seed_log_prob], dtype=torch.float64, device=device),
        depths=torch.tensor([seed_depth], dtype=torch.long, device=device),
        partial=seed_partial_row.unsqueeze(0),
    )

    # Phase-1 storage (separate from pool, to keep pool small)
    p1_log_probs: List[torch.Tensor] = []
    p1_partial: List[torch.Tensor] = []

    K = 0
    S = 0.0

    for _ in range(max_iters):
        # Stop once the budget is exhausted — going past N would over-commit
        # samples (e.g., enumerate the entire tree even when N << num_states).
        if K >= N:
            break

        active = pool.active
        if not active.any():
            break

        lp = pool.log_probs
        de = pool.depths

        # Compute current threshold based on remaining budget. When S is
        # already at (or numerically near) 1 there's no remaining mass to
        # gate on, so the threshold collapses to -inf and every active
        # item passes.
        if S >= 1.0 - FRAC_EPS:
            threshold_log = float('-inf')
        else:
            nws = (N - K) / (1.0 - S)
            if nws <= 0 or not math.isfinite(nws):
                threshold_log = float('-inf')
            else:
                threshold_log = -math.log(nws)

        above = active & (lp > threshold_log + FRAC_EPS)

        if not above.any():
            # No leaf currently passes the threshold, but the pool still
            # has actives. Force progress on the top-K highest-lp items
            # — committing/expanding them raises K/S, which lowers the
            # threshold. Without this, phase 1 stalls and dumps the rest
            # onto phase 2 even when N = num_states (where enumeration
            # is the right behavior).
            masked_lp = torch.where(
                active, lp, torch.full_like(lp, float('-inf')))
            n_active = int(active.sum().item())
            k_take = min(n_active, 4096)
            _, top_idx = masked_lp.topk(k_take)
            above = torch.zeros_like(active)
            above[top_idx] = True

        is_leaf = de == num_levels
        leaf_above = above & is_leaf
        internal_above = above & (~is_leaf)

        # (1) Commit all above-threshold leaves
        if leaf_above.any():
            idx = leaf_above.nonzero(as_tuple=True)[0]
            committed_lp = lp[idx].clone()
            committed_partial = pool.partial[idx].clone()

            K += idx.numel()
            S += committed_lp.exp().sum().item()

            p1_log_probs.append(committed_lp)
            p1_partial.append(committed_partial)

            pool.active[idx] = False
            # Go back to top: threshold just dropped
            continue

        # (2) Expand all above-threshold internals, batched by depth
        int_idx = internal_above.nonzero(as_tuple=True)[0]
        int_depths = de[int_idx]

        # Snapshot log_probs and partial states BEFORE deactivation so we can
        # keep using them. This also guards against mid-loop pool reallocation.
        int_lp_snapshot = lp[int_idx].clone()
        int_partial_snapshot = pool.partial[int_idx].clone()

        # Group by depth
        unique_d = int_depths.unique()
        # Deactivate parents first (they're being expanded away)
        pool.active[int_idx] = False

        # Accumulate all children across depth groups, then do ONE pool.add
        all_child_lp_list = []
        all_child_depth_list = []
        all_child_partial_list = []

        for d_t in unique_d:
            d = int(d_t.item())
            level = tree.levels[num_levels - d - 1]
            D = level.domain_size

            mask = (int_depths == d_t)
            B = int(mask.sum().item())

            batch_lp = int_lp_snapshot[mask]                 # (B,)
            batch_partial = int_partial_snapshot[mask]        # (B, num_levels)
            # Conditional requires partial restricted to the first d columns
            # (which are the variables assigned so far in traversal order)
            var_order_so_far = [
                tree.levels[num_levels - k - 1].elim_var_label
                for k in range(d)
            ]
            cond_ln = _conditional_log_probs_batch(
                level, batch_partial[:, :d], var_order_so_far, device
            ).to(torch.float64)  # (B, D)

            children_lp = (batch_lp.unsqueeze(1) + cond_ln).reshape(-1)     # (B*D,)
            # Children partial state: copy parent's (int8), set column d to child value
            children_partial = batch_partial.unsqueeze(1).repeat(1, D, 1)   # (B, D, num_levels)
            child_values = torch.arange(D, dtype=partial_dtype, device=device).view(1, D, 1).expand(B, D, 1)
            children_partial = children_partial.clone()
            children_partial[:, :, d] = child_values.squeeze(-1)
            children_partial = children_partial.reshape(B * D, num_levels)
            children_depths = torch.full(
                (B * D,), d + 1, dtype=torch.long, device=device)

            # Drop any children with -inf log_prob (prob 0 branches)
            keep = children_lp > float('-inf')
            if keep.any():
                all_child_lp_list.append(children_lp[keep])
                all_child_depth_list.append(children_depths[keep])
                all_child_partial_list.append(children_partial[keep])

        # Do ONE pool.add at the end (avoid mid-loop compaction invalidating refs)
        if all_child_lp_list:
            pool.add(
                log_probs=torch.cat(all_child_lp_list),
                depths=torch.cat(all_child_depth_list),
                partial=torch.cat(all_child_partial_list, dim=0),
            )

    # Concatenate phase-1 tensors
    if p1_log_probs:
        all_p1_lp = torch.cat(p1_log_probs)
        all_p1_partial = torch.cat(p1_partial, dim=0)
    else:
        all_p1_lp = torch.empty(0, device=device, dtype=torch.float64)
        all_p1_partial = torch.empty((0, num_levels), device=device, dtype=partial_dtype)

    return pool, K, S, all_p1_lp, all_p1_partial


# ---------------------------------------------------------------------------
# Phase 2: distribute N-K samples across the frontier
# ---------------------------------------------------------------------------

def _phase2_vectorized(
    tree: ProposalTree,
    pool: _Pool,
    S: float,
    phase2_budget: int,
    rng: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Top-down walk from the frontier. Returns (partial, log_probs_nat) for the
    selected phase-2 samples.
    """
    device = tree.device
    num_levels = len(tree.levels)

    partial_dtype = pool.partial_dtype

    if phase2_budget == 0:
        return (torch.empty((0, num_levels), dtype=partial_dtype, device=device),
                torch.empty(0, device=device, dtype=torch.float64))

    # Compact pool to get active frontier nodes
    active_idx = pool.active.nonzero(as_tuple=True)[0]
    if active_idx.numel() == 0:
        return (torch.empty((0, num_levels), dtype=partial_dtype, device=device),
                torch.empty(0, device=device, dtype=torch.float64))

    frontier_lp = pool.log_probs[active_idx]            # (F,)
    frontier_depths = pool.depths[active_idx]           # (F,)
    frontier_partial = pool.partial[active_idx]         # (F, num_levels)

    # Each frontier node has proposal prob = exp(lp). Total frontier prob = 1 - S.
    # If S is ≈ 1, there's no mass left — return empty.
    if S >= 1.0 - FRAC_EPS:
        return (torch.empty((0, num_levels), dtype=partial_dtype, device=device),
                torch.empty(0, device=device, dtype=torch.float64))

    # Compute allocation in log space.
    #   allocated[i] = phase2_budget × q(i) / (1 - S)
    #   log_alloc[i] = log phase2_budget + frontier_lp[i] - log(1 - S)
    log_alloc = (math.log(float(phase2_budget))
                 + frontier_lp
                 - math.log(max(1.0 - S, FRAC_EPS))).to(torch.float64)

    # Floors: only safe to materialize for log_alloc >= 0 (allocated >= 1).
    # For log_alloc < 0, the integer floor is 0 by definition, and the exp()
    # would underflow to literal zero in float64 once log_alloc < -745.
    is_large = log_alloc >= 0.0
    allocated_safe = torch.zeros_like(log_alloc)
    allocated_safe[is_large] = log_alloc[is_large].exp()
    floors = allocated_safe.floor()
    need = phase2_budget - int(floors.sum().item())

    if need > 0 and log_alloc.numel() > 0:
        # Residual allocation via Gumbel-top-k in log space — replaces the
        # linear-space systematic rounding so frontier nodes with very
        # negative log_alloc (where exp would underflow) still participate.
        # log_frac[i] = log of the *fractional* part of allocation:
        #   - log_alloc < 0 (underflow regime): frac = exp(log_alloc), so
        #     log_frac = log_alloc directly.
        #   - log_alloc >= 0 with floor < allocated: log_frac = log(allocated - floor)
        #   - log_alloc >= 0 with allocated exactly integer: log_frac = -inf (no extra)
        log_frac = torch.where(
            is_large,
            torch.log((allocated_safe - floors).clamp(min=torch.finfo(torch.float64).tiny)),
            log_alloc,
        )
        # Sentinel -inf where there's no fractional weight at all
        no_frac_mask = is_large & (allocated_safe == floors)
        log_frac = torch.where(
            no_frac_mask, torch.full_like(log_frac, float('-inf')), log_frac)

        u = torch.rand(log_frac.shape, generator=rng, device=device, dtype=torch.float64)
        u = u.clamp(min=torch.finfo(torch.float64).tiny)
        gumbel = -torch.log(-torch.log(u))
        keys = log_frac + gumbel
        actual_need = min(need, int((log_frac > float('-inf')).sum().item()))
        extras = torch.zeros_like(floors)
        if actual_need > 0:
            top_indices = torch.topk(keys, actual_need, largest=True).indices
            extras.scatter_(0, top_indices, 1.0)
    else:
        extras = torch.zeros_like(floors)

    integer_counts = (floors + extras).to(torch.long)   # (F,)

    # Filter frontier to nodes with count > 0
    keep = integer_counts > 0
    frontier_lp = frontier_lp[keep]
    frontier_depths = frontier_depths[keep]
    frontier_partial = frontier_partial[keep]
    integer_counts = integer_counts[keep]

    # For each frontier node, sample `count` path completions from its subtree
    # via WMB conditional sampling. We run this vectorized.
    out_partial_list = []
    out_lp_list = []

    # To vectorize: replicate each frontier node `count` times, then do top-down
    # independent sampling from each replica.
    if integer_counts.numel() > 0:
        replicated_partial = frontier_partial.repeat_interleave(integer_counts, dim=0)
        replicated_depths = frontier_depths.repeat_interleave(integer_counts)
        replicated_lp = frontier_lp.repeat_interleave(integer_counts)

        # Group by starting depth and sample down
        for start_d_t in replicated_depths.unique():
            start_d = int(start_d_t.item())
            mask_sd = replicated_depths == start_d_t
            cur_partial = replicated_partial[mask_sd].clone()         # (B, num_levels)
            cur_lp = replicated_lp[mask_sd].clone()                   # (B,)

            for d in range(start_d, num_levels):
                level = tree.levels[num_levels - d - 1]
                D = level.domain_size
                var_order_so_far = [
                    tree.levels[num_levels - k - 1].elim_var_label
                    for k in range(d)
                ]
                cond_ln = _conditional_log_probs_batch(
                    level, cur_partial[:, :d], var_order_so_far, device
                ).to(torch.float64)                                    # (B, D)

                # Gumbel-max sampling in log space — equivalent to multinomial
                # over softmax(cond_ln) but never exits log space, so children
                # with cond_ln < -745 below the max still participate in the
                # draw at the correct (vanishing) rate instead of being
                # outright dropped by exp() underflow.
                u = torch.rand(cond_ln.shape, generator=rng,
                               device=device, dtype=torch.float64)
                u = u.clamp(min=torch.finfo(torch.float64).tiny)
                gumbel = -torch.log(-torch.log(u))
                sampled = (cond_ln + gumbel).argmax(dim=1)
                cur_partial[:, d] = sampled.to(cur_partial.dtype)
                cur_lp = cur_lp + cond_ln.gather(1, sampled.unsqueeze(1)).squeeze(1)

            out_partial_list.append(cur_partial)
            out_lp_list.append(cur_lp)

    if out_partial_list:
        out_partial = torch.cat(out_partial_list, dim=0)
        out_lp = torch.cat(out_lp_list, dim=0)
    else:
        out_partial = torch.empty((0, num_levels), dtype=partial_dtype, device=device)
        out_lp = torch.empty(0, dtype=torch.float64, device=device)

    return out_partial, out_lp


def _systematic_round_1d(fracs: torch.Tensor, need: int,
                         rng: torch.Generator, device: str,
                         clamp_one: bool = True) -> torch.Tensor:
    """1-D systematic sampling: pick `need` indices from fracs to round up.

    When clamp_one=True (default): caps extras at 1 per node (used by phase 2
    where each frontier node represents one unique sample destination).
    When clamp_one=False: allows > 1 per node (used by recursive v3 where each
    frontier node represents a subtree that can absorb multiple samples).
    """
    assert fracs.ndim == 1
    L = fracs.shape[0]
    extras = torch.zeros_like(fracs)
    if need <= 0:
        return extras
    cum = fracs.cumsum(dim=0)
    u = torch.rand((), generator=rng, device=device).to(torch.float64)
    js = torch.arange(need, device=device, dtype=torch.float64)
    thresholds = u + js
    idx = torch.searchsorted(cum, thresholds)
    idx = idx.clamp(max=L - 1)
    # Mark each chosen index
    extras.scatter_add_(0, idx, torch.ones_like(idx, dtype=fracs.dtype))
    if clamp_one:
        return extras.clamp(max=1.0)
    return extras


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def sample_no_replacement_v3(
    tree: ProposalTree,
    N: int,
    rng: Optional[torch.Generator] = None,
    partial_dtype: torch.dtype = torch.int8,
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Vectorized faithful re-implementation of v1's no-replacement sampler.

    Args:
        partial_dtype: storage dtype for partial-state tensor (default int8;
            use int16/int32/int64 for domain sizes > 127).

    Returns (samples, log_probs_log10, effective_log_probs_log10) with the same
    contract as the other samplers.
    """
    device = tree.device
    num_levels = len(tree.levels)

    if N <= 0 or num_levels == 0:
        empty = {rec.elim_var_label: torch.zeros(0, dtype=torch.long, device=device)
                 for rec in tree.levels}
        return empty, torch.zeros(0, device=device), torch.zeros(0, device=device)

    if rng is None:
        rng = torch.Generator(device=device)
        rng.manual_seed(torch.randint(0, 2**31 - 1, (1,)).item())

    # Phase 1: iteratively commit above-threshold leaves (vectorized)
    pool, K, S, p1_lp_nat, p1_partial = _phase1_vectorized(
        tree, N, partial_dtype=partial_dtype
    )

    # Phase 2: distribute N - K samples across the frontier
    p2_partial, p2_lp_nat = _phase2_vectorized(
        tree, pool, S, phase2_budget=N - K, rng=rng
    )

    # Concatenate phase-1 + phase-2 samples
    all_partial = torch.cat([p1_partial, p2_partial], dim=0)
    all_lp_nat = torch.cat([p1_lp_nat, p2_lp_nat], dim=0)

    actual = all_partial.shape[0]
    if actual < N:
        # NR exhausted the reachable state space before reaching N. Refuse to
        # silently top up with with-replacement WMB samples (which would mix
        # estimator regimes and add bias-free but pure-noise duplicates).
        raise RuntimeError(
            f"sample_no_replacement_v3: requested {N} samples but only "
            f"{actual} unique states reachable (phase-1 K={K}, S={S:.6f}). "
            f"Caller must reduce num_samples or pick a different sampler."
        )
    elif actual > N:
        all_partial = all_partial[:N]
        all_lp_nat = all_lp_nat[:N]

    # Effective log probs (v1 convention):
    #   phase-1: eff = -log10(nws)
    #   phase-2: eff = log10(q)
    final_nws = float('inf') if S >= 1.0 - FRAC_EPS else (N - K) / (1.0 - S)
    log10_nws = math.log10(final_nws) if (final_nws > 0 and math.isfinite(final_nws)) \
                                       else 0.0
    log_probs_log10 = (all_lp_nat / math.log(10)).to(torch.float32)

    # Identify phase-1 samples: the first K are phase-1 by construction
    is_phase1 = torch.zeros(all_partial.shape[0], dtype=torch.bool, device=device)
    is_phase1[:K] = True
    eff_log_probs = torch.where(
        is_phase1,
        torch.full_like(log_probs_log10, -log10_nws),
        log_probs_log10,
    )

    var_order = [tree.levels[num_levels - d - 1].elim_var_label
                 for d in range(num_levels)]
    # Return per-var sample columns as long for downstream use (one-hot, indexing).
    # The storage cost saving was in the partial-state matrix; the final
    # per-var outputs are 1D and small.
    samples_dict = {var_order[d]: all_partial[:, d].to(torch.long) for d in range(num_levels)}
    return samples_dict, log_probs_log10, eff_log_probs


def _proposal_tree_sample_no_replacement_v3(self, N, rng=None):
    return sample_no_replacement_v3(self, N, rng=rng)

ProposalTree.sample_no_replacement_v3 = _proposal_tree_sample_no_replacement_v3


# ---------------------------------------------------------------------------
# Recursive no-replacement: outer budget N with per-subtree multiplier M
# ---------------------------------------------------------------------------

def sample_no_replacement_v3_recursive(
    tree: ProposalTree,
    N: int,
    M: int,
    rng: Optional[torch.Generator] = None,
    mode: str = 'save',
    callback: Optional['callable'] = None,
    partial_dtype: torch.dtype = torch.int8,
):
    """
    Recursive no-replacement sampling with outer budget N, per-subtree multiplier M.

    Outer v3 produces K_outer phase-1 samples + a set of frontier subtrees
    (unresolved, with small proposal probability). For each frontier subtree f
    with outer "node value" v_f = q(f) × nws_outer, we run an inner v3 with
    budget N_f = round(v_f × M). Total expected samples ≈ K_outer + M·(N - K_outer).

    Output convention: **pure Horvitz-Thompson**. For every returned sample,
    eff_log_prob_log10 = log10(π) where π is the marginal inclusion probability.
    Weight = 1/π = 10^(-eff_log_prob). Training wrappers that self-normalize
    work unchanged; Z estimators use `logsumexp(f_log10 - eff_log_prob)`.

    Args:
        N: outer budget.
        M: per-subtree multiplier (e.g., 10, 1000, 10000).
        mode:
            'save'    — materialize and return all samples (for NN training).
            'discard' — call callback(samples_dict, log_probs, eff_log_probs)
                        once per "batch" (outer phase-1 + one per subtree),
                        then drop them. For massive-N streaming estimators.
        callback: required if mode='discard'. Caller accumulates state itself.

    Returns (save mode): (samples_dict, log_probs_log10, eff_log_probs_log10).
    Returns (discard mode): None.
    """
    device = tree.device
    num_levels = len(tree.levels)
    ln10 = math.log(10)

    if mode == 'discard' and callback is None:
        raise ValueError("discard mode requires a callback")
    if mode not in ('save', 'discard'):
        raise ValueError(f"unknown mode {mode!r}")

    if rng is None:
        rng = torch.Generator(device=device)
        rng.manual_seed(torch.randint(0, 2**31 - 1, (1,)).item())

    var_order = [tree.levels[num_levels - d - 1].elim_var_label
                 for d in range(num_levels)]

    def _to_samples_dict(partial):
        return {var_order[d]: partial[:, d].to(torch.long) for d in range(num_levels)}

    saved_samples_cols: List[torch.Tensor] = []   # each is (B, num_levels)
    saved_lp: List[torch.Tensor] = []
    saved_eff: List[torch.Tensor] = []

    def _emit(partial, log_probs_log10, eff_log_probs_log10):
        if partial.shape[0] == 0:
            return
        if mode == 'save':
            saved_samples_cols.append(partial)
            saved_lp.append(log_probs_log10)
            saved_eff.append(eff_log_probs_log10)
        else:
            callback(_to_samples_dict(partial), log_probs_log10, eff_log_probs_log10)

    # ---- Step 1: outer v3 phase 1 ----
    pool, K_outer, S_outer, outer_p1_lp_nat, outer_p1_partial = _phase1_vectorized(
        tree, N, partial_dtype=partial_dtype)

    # Emit outer phase-1 samples: pure HT → π = 1 → eff = 0
    if K_outer > 0:
        outer_p1_lp_log10 = (outer_p1_lp_nat / ln10).to(torch.float32)
        outer_p1_eff_log10 = torch.zeros_like(outer_p1_lp_log10)
        _emit(outer_p1_partial, outer_p1_lp_log10, outer_p1_eff_log10)

    # ---- Step 2: iterate over frontier subtrees ----
    active_idx = pool.active.nonzero(as_tuple=True)[0]
    if (active_idx.numel() > 0 and S_outer < 1.0 - FRAC_EPS
            and K_outer < N):
        nws_outer = (N - K_outer) / (1.0 - S_outer)
        frontier_lp_nat = pool.log_probs[active_idx].clone()
        frontier_depths = pool.depths[active_idx].clone()
        frontier_partial = pool.partial[active_idx].clone()

        # Allocate M*(N - K_outer) integer samples across frontier subtrees,
        # in log space so exp-underflow doesn't drop low-q subtrees entirely.
        # Floors are computed only for subtrees with log_alloc >= 0 (where
        # exp is safe); the residual `need` samples go via with-replacement
        # Gumbel-max sampling on log_alloc.
        log_alloc = (frontier_lp_nat.to(torch.float64)
                     + math.log(float(nws_outer))
                     + math.log(float(M)))
        is_large = log_alloc >= 0.0
        allocated_safe = torch.zeros_like(log_alloc)
        allocated_safe[is_large] = log_alloc[is_large].exp()
        floors = allocated_safe.floor()
        target_total = int(round(M * (N - K_outer)))
        need = target_total - int(floors.sum().item())
        if need > 0 and log_alloc.numel() > 0:
            # With-replacement Gumbel-max in chunks to bound memory at
            # ~chunk * F float64 entries.
            F = log_alloc.shape[0]
            chunk = max(1, min(need, int(1.25e8 // max(F, 1))))
            extras = torch.zeros_like(floors)
            remaining = need
            while remaining > 0:
                n = min(chunk, remaining)
                u = torch.rand(
                    (n, F), generator=rng, device=device, dtype=torch.float64
                ).clamp_(min=torch.finfo(torch.float64).tiny)
                g = -torch.log(-torch.log(u))
                sampled = (log_alloc.unsqueeze(0) + g).argmax(dim=1)
                extras.scatter_add_(
                    0, sampled, torch.ones_like(sampled, dtype=floors.dtype))
                remaining -= n
        else:
            extras = torch.zeros_like(floors)
        frontier_Nf = (floors + extras).to(torch.int64)
        # Only recurse on subtrees that get at least 1 sample
        keep_idx = (frontier_Nf >= 1).nonzero(as_tuple=True)[0].tolist()

        # Drop the outer pool to free memory before doing many inner runs
        del pool

        for i in keep_idx:
            lp_f_nat = float(frontier_lp_nat[i].item())
            depth_f = int(frontier_depths[i].item())
            partial_f = frontier_partial[i]
            N_f = int(frontier_Nf[i].item())

            # Inner v3 starting from seed; seed_log_prob=0 so algorithm treats
            # the subtree as its own normalized world (q_inner = q/p_f).
            inner_pool, K_inner, S_inner, inner_p1_lp_nat, inner_p1_partial = \
                _phase1_vectorized(
                    tree, N_f,
                    partial_dtype=partial_dtype,
                    seed_log_prob=0.0,
                    seed_depth=depth_f,
                    seed_partial=partial_f,
                )

            # Phase 2 decimal resolution within the subtree
            inner_p2_partial, inner_p2_lp_nat = _phase2_vectorized(
                tree, inner_pool, S_inner,
                phase2_budget=max(0, N_f - K_inner), rng=rng,
            )

            # Concatenate and trim/pad to exactly N_f samples
            all_partial = torch.cat([inner_p1_partial, inner_p2_partial], dim=0)
            all_lp_nat_inner = torch.cat([inner_p1_lp_nat, inner_p2_lp_nat], dim=0)
            if all_partial.shape[0] > N_f:
                all_partial = all_partial[:N_f]
                all_lp_nat_inner = all_lp_nat_inner[:N_f]
            # (no top-up here — small deficits are acceptable per user spec)

            p_f_log10 = lp_f_nat / ln10   # = log10(q(f)) = log10(p_f)

            if S_inner >= 1.0 - FRAC_EPS or K_inner >= N_f:
                log10_nws_inner = 0.0  # degenerate: everything committed
            else:
                nws_inner = (N_f - K_inner) / (1.0 - S_inner)
                log10_nws_inner = math.log10(nws_inner) if nws_inner > 0 else 0.0

            # Convert to original (outer) frame + pure-HT eff encoding
            #   log10(q(x))  = inner log10(q_inner) + log10(p_f)
            #   inner phase-1 → π_inner = 1  → eff (pure HT) = 0
            #   inner phase-2 → π_inner = q_inner(x) × nws_inner
            #                 → eff = log10(q_inner(x)) + log10(nws_inner)
            #                 = (inner log10(q_inner) from output) + log10(nws_inner)
            K_inner_actual = inner_p1_partial.shape[0]
            K_inner_keep = min(K_inner_actual, all_partial.shape[0])
            # log_probs: inner returns log q_inner in natural log; convert to log10 and shift
            all_lp_log10 = (all_lp_nat_inner / ln10 + p_f_log10).to(torch.float32)

            eff = torch.empty(all_partial.shape[0], dtype=torch.float32, device=device)
            # Phase-1 portion: eff = 0
            eff[:K_inner_keep] = 0.0
            # Phase-2 portion: eff = log10(q_inner) + log10(nws_inner)
            #                  (and log10(q_inner) = all_lp_log10 - p_f_log10)
            if K_inner_keep < all_partial.shape[0]:
                eff[K_inner_keep:] = (all_lp_log10[K_inner_keep:] - p_f_log10
                                      + log10_nws_inner)

            _emit(all_partial, all_lp_log10, eff)

            # Drop the inner pool before next iteration
            del inner_pool

    if mode == 'save':
        if not saved_samples_cols:
            empty_samples = {v: torch.zeros(0, dtype=torch.long, device=device)
                             for v in var_order}
            return (empty_samples,
                    torch.zeros(0, device=device),
                    torch.zeros(0, device=device))
        all_partial = torch.cat(saved_samples_cols, dim=0)
        all_lp = torch.cat(saved_lp)
        all_eff = torch.cat(saved_eff)
        return _to_samples_dict(all_partial), all_lp, all_eff
    return None


def _proposal_tree_sample_no_replacement_v3_recursive(self, N, M, **kw):
    return sample_no_replacement_v3_recursive(self, N, M, **kw)

ProposalTree.sample_no_replacement_v3_recursive = \
    _proposal_tree_sample_no_replacement_v3_recursive
