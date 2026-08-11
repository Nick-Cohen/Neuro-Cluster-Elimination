"""
Multi-level recursive no-replacement sampling.

Extends v3 to arbitrary recursion depth via **batched-across-subtrees**
processing: at each level, all frontier subtrees are combined into a single
pool with a `subtree_id` column, so one batched tensor op covers all of them.

Memory model:
  - Per pool entry: (log_prob fp64, depth i64, subtree_id i32, partial i8×num_levels, active bool)
    ≈ 1.6 KB on grid40x40 (1600 levels) — same as v3.
  - Per subtree state: (K, S, N) — three small float/int values per subtree.

Algorithm at each level:
  1. For each subtree s with budget N_s, threshold_s = 1 / nws_s where
     nws_s = (N_s - K_s) / (1 - S_s).
  2. Above-threshold = per-entry log_prob > threshold[subtree_id].
  3. Commit above-threshold leaves (update K, S per their subtree_id).
  4. Expand above-threshold internals (children inherit subtree_id).
  5. Iterate until fixed point.

Non-terminal levels only do phase 1; the remaining frontier feeds the next
level with per-subtree budget `v_f × M_{next}` where v_f is the subtree's
expected sample count under its parent level's nws.

Terminal level runs phase 1 AND batched phase 2 (per-subtree decimal
resolution).

Output convention: pure Horvitz-Thompson. eff_log_prob = log10(π) where π
is the marginal inclusion probability.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import torch

from nce.sampling.proposal_sampler import ProposalTree
from nce.sampling.no_replacement_sampler_v2 import _conditional_log_probs_batch
from nce.sampling.no_replacement_sampler_v3 import _systematic_round_1d

FRAC_EPS = 1e-9


# ---------------------------------------------------------------------------
# Multi-subtree pool
# ---------------------------------------------------------------------------

class _MultiPool:
    """Batched pool: like v3's _Pool but with a subtree_id column."""

    __slots__ = ('log_probs', 'depths', 'subtree_ids', 'partial', 'active',
                 'num_levels', 'device', 'capacity', 'next_free',
                 'partial_dtype')

    def __init__(self, capacity: int, num_levels: int, device: str,
                 partial_dtype: torch.dtype = torch.int8):
        self.num_levels = num_levels
        self.device = device
        self.partial_dtype = partial_dtype
        self.capacity = max(capacity, 16)
        self.next_free = 0
        self.log_probs = torch.full(
            (self.capacity,), float('-inf'), device=device, dtype=torch.float64)
        self.depths = torch.zeros(self.capacity, dtype=torch.long, device=device)
        self.subtree_ids = torch.zeros(self.capacity, dtype=torch.int32, device=device)
        self.partial = torch.zeros(
            (self.capacity, num_levels), dtype=partial_dtype, device=device)
        self.active = torch.zeros(self.capacity, dtype=torch.bool, device=device)

    def _grow_to(self, new_capacity: int):
        new_lp = torch.full((new_capacity,), float('-inf'),
                            device=self.device, dtype=torch.float64)
        new_de = torch.zeros(new_capacity, dtype=torch.long, device=self.device)
        new_sid = torch.zeros(new_capacity, dtype=torch.int32, device=self.device)
        new_pa = torch.zeros((new_capacity, self.num_levels),
                             dtype=self.partial_dtype, device=self.device)
        new_ac = torch.zeros(new_capacity, dtype=torch.bool, device=self.device)
        new_lp[:self.capacity] = self.log_probs
        new_de[:self.capacity] = self.depths
        new_sid[:self.capacity] = self.subtree_ids
        new_pa[:self.capacity] = self.partial
        new_ac[:self.capacity] = self.active
        self.log_probs = new_lp
        self.depths = new_de
        self.subtree_ids = new_sid
        self.partial = new_pa
        self.active = new_ac
        self.capacity = new_capacity

    def _compact(self):
        active_idx = self.active.nonzero(as_tuple=True)[0]
        n = active_idx.numel()
        if n == 0:
            self.next_free = 0
            return
        self.log_probs[:n] = self.log_probs[active_idx]
        self.depths[:n] = self.depths[active_idx]
        self.subtree_ids[:n] = self.subtree_ids[active_idx]
        self.partial[:n] = self.partial[active_idx]
        self.log_probs[n:].fill_(float('-inf'))
        self.depths[n:].zero_()
        self.subtree_ids[n:].zero_()
        self.partial[n:].zero_()
        self.active.zero_()
        self.active[:n] = True
        self.next_free = n

    def _ensure_space(self, num_new: int):
        # Only compact when the occupied fraction is very low AND compaction
        # would avoid growth. Advanced-indexing compaction doubles peak memory
        # briefly; avoid it on big pools by just growing instead.
        if self.next_free + num_new > self.capacity:
            active_count = int(self.active.sum().item())
            # Compact only if active_count is much smaller than capacity and
            # the pool is small enough that the transient copy is cheap.
            if (active_count + num_new <= self.capacity
                    and self.capacity * self.num_levels < 2 ** 27):  # <~128M cells
                self._compact()
            if self.next_free + num_new > self.capacity:
                target = max(self.capacity * 2, self.next_free + num_new)
                self._grow_to(target)

    def add(self, log_probs: torch.Tensor, depths: torch.Tensor,
            subtree_ids: torch.Tensor, partial: torch.Tensor):
        num_new = log_probs.shape[0]
        if num_new == 0:
            return
        self._ensure_space(num_new)
        s = self.next_free
        e = s + num_new
        self.log_probs[s:e] = log_probs
        self.depths[s:e] = depths
        self.subtree_ids[s:e] = subtree_ids
        self.partial[s:e] = partial
        self.active[s:e] = True
        self.next_free = e


# ---------------------------------------------------------------------------
# Batched phase 1 across many subtrees
# ---------------------------------------------------------------------------

def _batched_phase1(
    tree: ProposalTree,
    seed_lp_nat: torch.Tensor,       # (S,) float64 — log_prob of each subtree's seed (natural log, inner frame)
    seed_depth: torch.Tensor,        # (S,) int64
    seed_partial: torch.Tensor,      # (S, num_levels) partial_dtype
    seed_N: torch.Tensor,            # (S,) int64 — budget per subtree
    partial_dtype: torch.dtype = torch.int8,
    max_iters: int = 10_000,
    initial_capacity: int = 8192,
):
    """
    Run phase 1 across S subtrees simultaneously.

    Each subtree has its own threshold = 1/nws_s where nws_s = (N_s-K_s)/(1-S_s).
    Commits above-threshold leaves per-subtree, updates K/S/nws, repeats.

    Returns:
        pool: _MultiPool with remaining active frontier (plus inactive expanded/committed slots)
        K_per_sub: (S,) int64 — phase-1 commits per subtree
        S_per_sub: (S,) float64 — phase-1 prob mass per subtree
        p1_log_probs: (total_K,) — log_prob (inner frame) of every committed phase-1 leaf
        p1_partial: (total_K, num_levels) — partial states of phase-1 leaves
        p1_subtree_ids: (total_K,) int32 — which subtree each phase-1 leaf belongs to
    """
    device = tree.device
    num_levels = len(tree.levels)
    S_count = seed_lp_nat.shape[0]

    pool = _MultiPool(initial_capacity, num_levels, device, partial_dtype)

    # Seed the pool with all subtree roots
    pool.add(
        log_probs=seed_lp_nat,
        depths=seed_depth,
        subtree_ids=torch.arange(S_count, dtype=torch.int32, device=device),
        partial=seed_partial,
    )

    # Per-subtree state
    K_per_sub = torch.zeros(S_count, dtype=torch.int64, device=device)
    S_per_sub = torch.zeros(S_count, dtype=torch.float64, device=device)
    N_per_sub = seed_N.to(torch.float64)

    # Collected phase-1 commits
    p1_log_probs_list: List[torch.Tensor] = []
    p1_partial_list: List[torch.Tensor] = []
    p1_subtree_ids_list: List[torch.Tensor] = []

    def compute_thresholds():
        # nws_s = (N_s - K_s) / (1 - S_s). threshold_log = -log(nws_s).
        remaining_N = N_per_sub - K_per_sub.to(torch.float64)
        remaining_S = 1.0 - S_per_sub
        # Degenerate subtrees: S >= 1 or K >= N → no more threshold crossings
        safe = (remaining_S > FRAC_EPS) & (remaining_N > FRAC_EPS)
        nws = torch.where(safe, remaining_N / remaining_S,
                          torch.full_like(remaining_N, float('inf')))
        # threshold_log = -log(nws). For inf nws, threshold_log = -inf (everything above)
        threshold_log = torch.where(
            nws > 0,
            -torch.log(nws.clamp(min=1e-300)),
            torch.full_like(nws, float('-inf')),
        )
        return threshold_log

    for _it in range(max_iters):
        if not pool.active.any():
            break

        threshold_log = compute_thresholds()  # (S_count,)
        # Per-entry threshold: look up by subtree_id
        active = pool.active
        sid = pool.subtree_ids.long()
        lp = pool.log_probs
        de = pool.depths

        entry_thresh = threshold_log[sid]    # (capacity,)
        above = active & (lp > entry_thresh + FRAC_EPS)

        if not above.any():
            break

        is_leaf = de == num_levels
        leaf_above = above & is_leaf
        internal_above = above & (~is_leaf)

        # ---- Commit all above-threshold leaves (per subtree accounting) ----
        if leaf_above.any():
            idx = leaf_above.nonzero(as_tuple=True)[0]
            lp_committed = lp[idx].clone()
            sid_committed = sid[idx]
            partial_committed = pool.partial[idx].clone()

            # Accumulate per-subtree K, S via scatter_add
            ones64 = torch.ones_like(lp_committed, dtype=torch.int64)
            K_per_sub.scatter_add_(0, sid_committed, ones64)
            probs_committed = lp_committed.exp()
            S_per_sub.scatter_add_(0, sid_committed, probs_committed)

            # Save for output
            p1_log_probs_list.append(lp_committed)
            p1_partial_list.append(partial_committed)
            p1_subtree_ids_list.append(sid_committed.to(torch.int32))

            pool.active[idx] = False
            # Thresholds dropped for these subtrees; re-check
            continue

        # ---- Expand all above-threshold internals (batched by depth) ----
        int_idx = internal_above.nonzero(as_tuple=True)[0]
        int_depths_snap = de[int_idx].clone()
        int_lp_snap = lp[int_idx].clone()
        int_partial_snap = pool.partial[int_idx].clone()
        int_sid_snap = sid[int_idx].clone()
        pool.active[int_idx] = False

        unique_d = int_depths_snap.unique()
        all_child_lp = []
        all_child_depth = []
        all_child_sid = []
        all_child_partial = []

        for d_t in unique_d:
            d = int(d_t.item())
            level = tree.levels[num_levels - d - 1]
            D = level.domain_size

            mask = (int_depths_snap == d_t)
            B = int(mask.sum().item())
            batch_lp = int_lp_snap[mask]
            batch_partial = int_partial_snap[mask]
            batch_sid = int_sid_snap[mask]

            var_order_so_far = [
                tree.levels[num_levels - k - 1].elim_var_label
                for k in range(d)
            ]
            cond_ln = _conditional_log_probs_batch(
                level, batch_partial[:, :d], var_order_so_far, device
            ).to(torch.float64)

            children_lp = (batch_lp.unsqueeze(1) + cond_ln).reshape(-1)
            child_values = torch.arange(D, dtype=partial_dtype, device=device).view(1, D, 1)
            children_partial = batch_partial.unsqueeze(1).repeat(1, D, 1).clone()
            children_partial[:, :, d] = child_values.expand(B, D, 1).squeeze(-1)
            children_partial = children_partial.reshape(B * D, num_levels)
            children_depth = torch.full((B * D,), d + 1, dtype=torch.long, device=device)
            children_sid = batch_sid.unsqueeze(1).expand(B, D).reshape(-1)

            keep = children_lp > float('-inf')
            if keep.any():
                all_child_lp.append(children_lp[keep])
                all_child_depth.append(children_depth[keep])
                all_child_sid.append(children_sid[keep])
                all_child_partial.append(children_partial[keep])

        if all_child_lp:
            pool.add(
                log_probs=torch.cat(all_child_lp),
                depths=torch.cat(all_child_depth),
                subtree_ids=torch.cat(all_child_sid),
                partial=torch.cat(all_child_partial, dim=0),
            )

    # Concatenate phase-1 outputs
    if p1_log_probs_list:
        p1_log_probs = torch.cat(p1_log_probs_list)
        p1_partial = torch.cat(p1_partial_list, dim=0)
        p1_subtree_ids = torch.cat(p1_subtree_ids_list)
    else:
        p1_log_probs = torch.empty(0, device=device, dtype=torch.float64)
        p1_partial = torch.empty((0, num_levels), device=device, dtype=partial_dtype)
        p1_subtree_ids = torch.empty(0, device=device, dtype=torch.int32)

    return pool, K_per_sub, S_per_sub, p1_log_probs, p1_partial, p1_subtree_ids


# ---------------------------------------------------------------------------
# Batched phase 2 across many subtrees
# ---------------------------------------------------------------------------

def _batched_phase2(
    tree: ProposalTree,
    pool: _MultiPool,
    K_per_sub: torch.Tensor,
    S_per_sub: torch.Tensor,
    N_per_sub: torch.Tensor,
    rng: torch.Generator,
):
    """
    Terminal level's decimal resolution. For each subtree s, distribute its
    phase-2 budget (N_s - K_s) across its remaining frontier nodes via
    systematic rounding, then sample completions for each selected frontier.

    Returns (partial, log_probs_nat, subtree_ids) for phase-2 samples.
    """
    device = tree.device
    num_levels = len(tree.levels)

    active_idx = pool.active.nonzero(as_tuple=True)[0]
    if active_idx.numel() == 0:
        return (torch.empty((0, num_levels), dtype=pool.partial_dtype, device=device),
                torch.empty(0, device=device, dtype=torch.float64),
                torch.empty(0, device=device, dtype=torch.int32))

    f_lp = pool.log_probs[active_idx]
    f_depth = pool.depths[active_idx]
    f_partial = pool.partial[active_idx]
    f_sid = pool.subtree_ids[active_idx].long()

    # Per-subtree phase-2 budget (remaining samples after phase-1)
    p2_budget_per_sub = (N_per_sub.to(torch.float64)
                         - K_per_sub.to(torch.float64)).clamp(min=0)

    # Allocate each subtree's frontier nodes
    # For subtree s, allocated_node = budget_s × q(node) / (1 - S_s) where q(node)
    # is the inner-frame prob, so (1 - S_s) is total remaining frontier mass.
    # Do this per subtree in a batched way:
    entry_budget = p2_budget_per_sub[f_sid]                                  # (F,)
    entry_remainder = (1.0 - S_per_sub[f_sid]).clamp(min=FRAC_EPS)           # (F,)
    # log_alloc = log(entry_budget) + f_lp - log(entry_remainder)
    log_alloc = (torch.log(entry_budget.clamp(min=FRAC_EPS))
                 + f_lp
                 - torch.log(entry_remainder))
    allocated = log_alloc.exp()
    # Skip subtrees with no budget
    allocated = torch.where(entry_budget > FRAC_EPS, allocated, torch.zeros_like(allocated))

    floors = allocated.floor()
    fracs = (allocated - floors).clamp(min=0, max=1)
    fracs = torch.where(fracs < FRAC_EPS, torch.zeros_like(fracs), fracs)
    fracs = torch.where(fracs > 1 - FRAC_EPS, torch.zeros_like(fracs), fracs)

    # For each subtree, systematic round its fractional parts so the sum equals
    # its integer phase-2 budget - floors_sum.
    # We do this per subtree in a loop (number of subtrees expected to be small-ish).
    integer_counts = floors.clone().to(torch.int64)

    # Group indices by subtree
    unique_sub, inv = torch.unique(f_sid, return_inverse=True)
    # For each unique subtree, compute need = budget - sum_floors and apply systematic round
    for sidx_i in range(unique_sub.numel()):
        s_mask = inv == sidx_i
        s = int(unique_sub[sidx_i].item())
        budget_s = int(p2_budget_per_sub[s].item())
        floors_s_sum = int(floors[s_mask].sum().item())
        need = budget_s - floors_s_sum
        if need <= 0:
            continue
        s_fracs = fracs[s_mask]
        # Systematic round
        extras = _systematic_round_1d(s_fracs, need, rng, device)
        integer_counts[s_mask] += extras.to(torch.int64)

    # Filter frontier nodes with count > 0
    keep = integer_counts > 0
    if not keep.any():
        return (torch.empty((0, num_levels), dtype=pool.partial_dtype, device=device),
                torch.empty(0, device=device, dtype=torch.float64),
                torch.empty(0, device=device, dtype=torch.int32))

    sel_lp = f_lp[keep]
    sel_depth = f_depth[keep]
    sel_partial = f_partial[keep]
    sel_sid = f_sid[keep]
    sel_count = integer_counts[keep]

    # Replicate each selected node `count` times
    rep_lp = sel_lp.repeat_interleave(sel_count)
    rep_depth = sel_depth.repeat_interleave(sel_count)
    rep_partial = sel_partial.repeat_interleave(sel_count, dim=0)
    rep_sid = sel_sid.repeat_interleave(sel_count)

    # Complete each replicated node to a full leaf by sampling through the
    # remaining levels. We batch by starting depth.
    cur_partial = rep_partial.clone()
    cur_lp = rep_lp.clone()
    cur_sid = rep_sid.clone()

    # Group by starting depth and sample down
    unique_start_depths = rep_depth.unique()
    # We'll process each starting-depth group; each advances to num_levels
    out_partial_chunks: List[torch.Tensor] = []
    out_lp_chunks: List[torch.Tensor] = []
    out_sid_chunks: List[torch.Tensor] = []

    for start_d_t in unique_start_depths:
        start_d = int(start_d_t.item())
        mask_sd = rep_depth == start_d_t
        p_slice = cur_partial[mask_sd].clone()
        lp_slice = cur_lp[mask_sd].clone()
        sid_slice = cur_sid[mask_sd]

        for d in range(start_d, num_levels):
            level = tree.levels[num_levels - d - 1]
            D = level.domain_size
            var_order_so_far = [
                tree.levels[num_levels - k - 1].elim_var_label
                for k in range(d)
            ]
            cond_ln = _conditional_log_probs_batch(
                level, p_slice[:, :d], var_order_so_far, device
            ).to(torch.float64)
            probs = cond_ln.exp().clamp(min=0)
            psum = probs.sum(dim=1, keepdim=True)
            probs = torch.where(psum > 0, probs / psum,
                                torch.full_like(probs, 1.0 / D))
            sampled = torch.multinomial(probs.to(torch.float32), 1,
                                        generator=rng).squeeze(1)
            p_slice[:, d] = sampled.to(p_slice.dtype)
            lp_slice = lp_slice + cond_ln.gather(1, sampled.unsqueeze(1)).squeeze(1)

        out_partial_chunks.append(p_slice)
        out_lp_chunks.append(lp_slice)
        out_sid_chunks.append(sid_slice)

    out_partial = torch.cat(out_partial_chunks, dim=0)
    out_lp = torch.cat(out_lp_chunks)
    out_sid = torch.cat(out_sid_chunks).to(torch.int32)
    return out_partial, out_lp, out_sid


# ---------------------------------------------------------------------------
# Multi-level driver
# ---------------------------------------------------------------------------

def sample_multilevel(
    tree: ProposalTree,
    level_budgets: List[int],         # e.g., [1000, 1000, 1000] for 3 levels
    rng: Optional[torch.Generator] = None,
    mode: str = 'save',
    callback: Optional[Callable] = None,
    partial_dtype: torch.dtype = torch.int8,
    max_seeds_per_batch: int = 256,  # chunk seeds to bound peak pool size
):
    """
    Multi-level recursive no-replacement sampling, batched across subtrees.

    Level 1: run phase-1 v3 with budget level_budgets[0] from root.
             Produce phase-1 leaves + frontier subtrees.
    Level k (2 ≤ k < L): batched phase-1 across all subtrees-from-level-(k-1).
             Each subtree's budget N_s = round(v_s × level_budgets[k-1]),
             where v_s = q_s × nws_{k-1}(parent of s) is the subtree's expected
             sample count under its parent's phase-2 scheme.
    Level L: batched phase-1 + batched phase-2.

    Output (pure Horvitz-Thompson):
      - Phase-1 leaves at any level: eff = 0  (weight 1, deterministic inclusion)
      - Phase-2 leaves at terminal level: eff = log10(q_inner × nws_terminal_for_that_subtree)

    mode='save': return (samples_dict, log_probs_log10, eff_log_probs_log10)
    mode='discard': call callback(...) per batch and drop; returns None.
    """
    device = tree.device
    num_levels = len(tree.levels)
    ln10 = math.log(10)

    if mode == 'discard' and callback is None:
        raise ValueError("discard mode requires a callback")

    if rng is None:
        rng = torch.Generator(device=device)
        rng.manual_seed(torch.randint(0, 2**31 - 1, (1,)).item())

    var_order = [tree.levels[num_levels - d - 1].elim_var_label
                 for d in range(num_levels)]

    def _to_samples_dict(partial):
        return {var_order[d]: partial[:, d].to(torch.long) for d in range(num_levels)}

    saved_parts: List[torch.Tensor] = []
    saved_lp: List[torch.Tensor] = []
    saved_eff: List[torch.Tensor] = []

    def _emit(partial, log_probs_log10, eff_log_probs_log10):
        if partial.shape[0] == 0:
            return
        if mode == 'save':
            saved_parts.append(partial)
            saved_lp.append(log_probs_log10)
            saved_eff.append(eff_log_probs_log10)
        else:
            callback(_to_samples_dict(partial), log_probs_log10, eff_log_probs_log10)

    # ----- Level 1: single seed at root -----
    L = len(level_budgets)
    assert L >= 1, "need at least one level"
    seed_lp_nat = torch.zeros(1, dtype=torch.float64, device=device)
    seed_depth = torch.zeros(1, dtype=torch.long, device=device)
    seed_partial = torch.zeros((1, num_levels), dtype=partial_dtype, device=device)
    seed_N = torch.tensor([int(level_budgets[0])], dtype=torch.int64, device=device)
    # Also track each subtree's "partial-state log_prob offset" — for level 1 this is 0 (root).
    seed_partial_offset_log10 = torch.zeros(1, dtype=torch.float64, device=device)

    def _process_seeds(seed_lp_nat_c, seed_depth_c, seed_partial_c, seed_N_c,
                       seed_offsets_c, is_terminal, next_M):
        """Run batched phase 1 on a chunk of seeds; optionally phase 2 if terminal.
        Returns (next_frontier_dict or None).
        """
        pool, K_sub, S_sub, p1_lp_nat, p1_partial, p1_sid = _batched_phase1(
            tree, seed_lp_nat_c, seed_depth_c, seed_partial_c, seed_N_c,
            partial_dtype
        )

        if p1_lp_nat.numel() > 0:
            offsets = seed_offsets_c[p1_sid.long()]
            p1_lp_log10 = (p1_lp_nat / ln10 + offsets).to(torch.float32)
            p1_eff = torch.zeros_like(p1_lp_log10)
            _emit(p1_partial, p1_lp_log10, p1_eff)

        if is_terminal:
            p2_partial, p2_lp_nat, p2_sid = _batched_phase2(
                tree, pool, K_sub, S_sub, seed_N_c.to(torch.float64), rng
            )
            if p2_partial.shape[0] > 0:
                offsets = seed_offsets_c[p2_sid.long()]
                p2_lp_log10 = (p2_lp_nat / ln10 + offsets).to(torch.float32)
                rem_N = (seed_N_c.to(torch.float64) - K_sub.to(torch.float64)).clamp(min=0)
                rem_S = (1.0 - S_sub).clamp(min=FRAC_EPS)
                nws_s = rem_N / rem_S
                log10_nws_s = torch.log10(nws_s.clamp(min=1.0))
                eff_at_sub = log10_nws_s.to(torch.float32)[p2_sid.long()]
                p2_lp_inner_log10 = (p2_lp_nat / ln10).to(torch.float32)
                p2_eff = p2_lp_inner_log10 + eff_at_sub
                _emit(p2_partial, p2_lp_log10, p2_eff)
            del pool
            return None

        # Non-terminal: extract frontier and prepare seeds for next level
        active_idx = pool.active.nonzero(as_tuple=True)[0]
        if active_idx.numel() == 0:
            del pool
            return None

        f_lp = pool.log_probs[active_idx].clone()
        f_depth = pool.depths[active_idx].clone()
        f_partial = pool.partial[active_idx].clone()
        f_sid = pool.subtree_ids[active_idx].long().clone()
        del pool  # free before computing next seeds

        rem_N = (seed_N_c.to(torch.float64) - K_sub.to(torch.float64)).clamp(min=0)
        rem_S = (1.0 - S_sub).clamp(min=FRAC_EPS)
        nws_cur = rem_N / rem_S
        f_nws = nws_cur[f_sid]
        v_f = f_lp.exp() * f_nws
        new_N_float = v_f * float(next_M)
        new_N = new_N_float.round().to(torch.int64)
        keep = new_N >= 1
        if not keep.any():
            return None
        f_lp = f_lp[keep]
        f_depth = f_depth[keep]
        f_partial = f_partial[keep]
        f_sid = f_sid[keep]
        new_N = new_N[keep]
        new_offsets = seed_offsets_c[f_sid] + (f_lp / ln10)

        return dict(
            lp_nat=torch.zeros_like(f_lp),  # seed in next level's inner frame
            depth=f_depth,
            partial=f_partial,
            N=new_N,
            offset_log10=new_offsets,
        )

    for level_i in range(L):
        is_terminal = (level_i == L - 1)
        next_M = float(level_budgets[level_i + 1]) if not is_terminal else None
        S_total = seed_lp_nat.shape[0]

        # Chunk the seeds to bound peak pool memory
        accumulated_frontier = []
        for start in range(0, S_total, max_seeds_per_batch):
            end = min(start + max_seeds_per_batch, S_total)
            result = _process_seeds(
                seed_lp_nat[start:end],
                seed_depth[start:end],
                seed_partial[start:end],
                seed_N[start:end],
                seed_partial_offset_log10[start:end],
                is_terminal, next_M,
            )
            if result is not None:
                accumulated_frontier.append(result)
            # Clear the CUDA cache between chunks to reclaim transient memory
            torch.cuda.empty_cache()

        if is_terminal or not accumulated_frontier:
            break

        # Combine frontier chunks into the next level's seeds
        seed_lp_nat = torch.cat([f['lp_nat'] for f in accumulated_frontier])
        seed_depth = torch.cat([f['depth'] for f in accumulated_frontier])
        seed_partial = torch.cat([f['partial'] for f in accumulated_frontier], dim=0)
        seed_N = torch.cat([f['N'] for f in accumulated_frontier])
        seed_partial_offset_log10 = torch.cat([f['offset_log10'] for f in accumulated_frontier])
        del accumulated_frontier

    # Return or finalize
    if mode == 'save':
        if not saved_parts:
            empty_samples = {v: torch.zeros(0, dtype=torch.long, device=device)
                             for v in var_order}
            return (empty_samples, torch.zeros(0, device=device),
                    torch.zeros(0, device=device))
        all_partial = torch.cat(saved_parts, dim=0)
        all_lp = torch.cat(saved_lp)
        all_eff = torch.cat(saved_eff)
        return _to_samples_dict(all_partial), all_lp, all_eff
    return None


def _proposal_tree_sample_multilevel(self, level_budgets, **kw):
    return sample_multilevel(self, level_budgets, **kw)

ProposalTree.sample_multilevel = _proposal_tree_sample_multilevel
