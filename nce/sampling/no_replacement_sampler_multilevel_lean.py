"""
Multi-level recursive no-replacement sampling with a parent-pointer pool.

Key memory optimization vs `no_replacement_sampler_multilevel.py`:

  Old layout per pool entry: (log_prob, depth, subtree_id, partial[num_levels], active)
    grid40x40 (num_levels=1600): ~1617 bytes/entry

  New layout per pool entry: (log_prob, depth, subtree_id, parent_idx, var_value, active)
    grid40x40: ~22 bytes/entry (~73x smaller)

Partial state is reconstructed lazily by walking up the parent_idx chain.
Cost: O(depth × batch_size) GPU ops per `materialize_partial` call (each step
is a vectorized scatter), but only paid when partial states are actually needed
(conditional compute and emit).

For seed roots (entries inserted with no parent), a separate per-session
`seed_partial` tensor holds their pre-set partial state; ancestors of those
seeds are inherited from earlier recursion levels.
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
# Lean pool: parent-pointer layout
# ---------------------------------------------------------------------------

class _LeanPool:
    """Pool with parent-pointer layout (~22 bytes/entry).

    partial states are reconstructed lazily via materialize_partial(indices).
    """

    __slots__ = ('log_probs', 'depths', 'subtree_ids', 'parent_idx',
                 'var_value', 'active', 'seed_partial',
                 'num_levels', 'device', 'capacity', 'next_free',
                 'partial_dtype')

    def __init__(self, capacity: int, num_levels: int, device: str,
                 partial_dtype: torch.dtype = torch.int8):
        self.num_levels = num_levels
        self.device = device
        self.partial_dtype = partial_dtype
        self.capacity = max(capacity, 16)
        self.next_free = 0
        self.log_probs = torch.full((self.capacity,), float('-inf'),
                                    device=device, dtype=torch.float64)
        self.depths = torch.zeros(self.capacity, dtype=torch.int32, device=device)
        self.subtree_ids = torch.zeros(self.capacity, dtype=torch.int32, device=device)
        # parent_idx = -1 means the entry is a seed (no parent in pool).
        self.parent_idx = torch.full((self.capacity,), -1, dtype=torch.int32, device=device)
        self.var_value = torch.zeros(self.capacity, dtype=partial_dtype, device=device)
        self.active = torch.zeros(self.capacity, dtype=torch.bool, device=device)
        # Seed partial states, set externally via set_seed_partial(...).
        # (num_seeds, num_levels) — only meaningful for seed entries.
        self.seed_partial: Optional[torch.Tensor] = None

    def set_seed_partial(self, partial: torch.Tensor):
        """Record the per-subtree seed partial states for this run.

        partial: (num_seeds, num_levels) of partial_dtype.
        """
        self.seed_partial = partial

    def _grow_to(self, new_capacity: int):
        new_lp = torch.full((new_capacity,), float('-inf'),
                            device=self.device, dtype=torch.float64)
        new_de = torch.zeros(new_capacity, dtype=torch.int32, device=self.device)
        new_sid = torch.zeros(new_capacity, dtype=torch.int32, device=self.device)
        new_par = torch.full((new_capacity,), -1, dtype=torch.int32, device=self.device)
        new_vv = torch.zeros(new_capacity, dtype=self.partial_dtype, device=self.device)
        new_ac = torch.zeros(new_capacity, dtype=torch.bool, device=self.device)
        new_lp[:self.capacity] = self.log_probs
        new_de[:self.capacity] = self.depths
        new_sid[:self.capacity] = self.subtree_ids
        new_par[:self.capacity] = self.parent_idx
        new_vv[:self.capacity] = self.var_value
        new_ac[:self.capacity] = self.active
        self.log_probs = new_lp
        self.depths = new_de
        self.subtree_ids = new_sid
        self.parent_idx = new_par
        self.var_value = new_vv
        self.active = new_ac
        self.capacity = new_capacity

    def _ensure_space(self, num_new: int):
        if self.next_free + num_new > self.capacity:
            target = max(self.capacity * 2, self.next_free + num_new)
            self._grow_to(target)

    def add(self, log_probs: torch.Tensor, depths: torch.Tensor,
            subtree_ids: torch.Tensor, parent_idx: torch.Tensor,
            var_value: torch.Tensor) -> torch.Tensor:
        """Add entries; return their indices in the pool."""
        num_new = log_probs.shape[0]
        if num_new == 0:
            return torch.empty(0, dtype=torch.int32, device=self.device)
        self._ensure_space(num_new)
        s = self.next_free
        e = s + num_new
        self.log_probs[s:e] = log_probs
        self.depths[s:e] = depths.to(torch.int32)
        self.subtree_ids[s:e] = subtree_ids.to(torch.int32)
        self.parent_idx[s:e] = parent_idx.to(torch.int32)
        self.var_value[s:e] = var_value.to(self.partial_dtype)
        self.active[s:e] = True
        self.next_free = e
        return torch.arange(s, e, dtype=torch.int32, device=self.device)

    def materialize_partial(self, indices: torch.Tensor,
                            max_depth: Optional[int] = None) -> torch.Tensor:
        """
        Reconstruct (B, max_depth) partial states for `indices`.

        Walks up parent_idx chain. When reaching a seed entry, copies the
        seed's partial state from `self.seed_partial[subtree_id]`.
        """
        B = indices.shape[0]
        if max_depth is None:
            max_depth = self.num_levels
        out = torch.zeros((B, max_depth), dtype=self.partial_dtype, device=self.device)
        if B == 0:
            return out

        cur = indices.to(torch.int64)  # walking pointer; -1 means done
        # Walk up
        for _ in range(max_depth + 1):  # +1 to allow reaching seeds
            valid = cur >= 0
            if not valid.any():
                break
            valid_cur = cur.clamp(min=0)
            cur_depth = self.depths[valid_cur].long()        # (B,)
            cur_val = self.var_value[valid_cur]              # (B,)
            cur_parent = self.parent_idx[valid_cur].to(torch.int64)  # (B,)

            # For non-seed entries (cur_depth > 0 AND has a parent):
            # write var_value at column (cur_depth - 1).
            non_seed = valid & (cur_depth > 0) & (cur_parent >= 0)
            if non_seed.any():
                rows = non_seed.nonzero(as_tuple=True)[0]
                cols = (cur_depth[rows] - 1).clamp(max=max_depth - 1)
                out[rows, cols] = cur_val[rows]

            # For seed entries (parent == -1), copy seed_partial[:, :seed_depth].
            # Use a column-mask so we only overwrite columns < seed_depth and
            # leave deeper columns (filled by descendants) untouched.
            is_seed = valid & (cur_parent < 0)
            if self.seed_partial is not None and is_seed.any():
                seed_rows = is_seed.nonzero(as_tuple=True)[0]
                sids = self.subtree_ids[valid_cur][seed_rows].to(torch.int64)
                sp = self.seed_partial[sids]   # (k, num_levels)
                seed_depths = cur_depth[seed_rows]  # (k,)
                j = torch.arange(max_depth, device=self.device).unsqueeze(0)  # (1, max_depth)
                write_mask = j < seed_depths.unsqueeze(1)  # (k, max_depth)
                out_subset = out[seed_rows]
                out_subset = torch.where(write_mask, sp[:, :max_depth], out_subset)
                out[seed_rows] = out_subset

            # Advance to parent (or -1 for seeds, terminating their walk)
            cur = torch.where(valid, cur_parent, torch.full_like(cur, -1))

        return out


# ---------------------------------------------------------------------------
# Batched phase 1 with lean pool
# ---------------------------------------------------------------------------

def _batched_phase1_lean(
    tree: ProposalTree,
    seed_lp_nat: torch.Tensor,       # (S,) float64
    seed_depth: torch.Tensor,        # (S,) int64
    seed_partial: torch.Tensor,      # (S, num_levels) partial_dtype — full seed states
    seed_N: torch.Tensor,            # (S,) int64
    partial_dtype: torch.dtype = torch.int8,
    max_iters: int = 10_000,
    initial_capacity: int = 8192,
):
    """Batched phase-1 across multiple subtree seeds, using parent-pointer pool."""
    device = tree.device
    num_levels = len(tree.levels)
    S_count = seed_lp_nat.shape[0]

    pool = _LeanPool(initial_capacity, num_levels, device, partial_dtype)
    pool.set_seed_partial(seed_partial.to(partial_dtype))

    # Seed: parent_idx = -1, var_value = 0 (unused), depth = seed_depth
    seed_indices = pool.add(
        log_probs=seed_lp_nat,
        depths=seed_depth.to(torch.int32),
        subtree_ids=torch.arange(S_count, dtype=torch.int32, device=device),
        parent_idx=torch.full((S_count,), -1, dtype=torch.int32, device=device),
        var_value=torch.zeros(S_count, dtype=partial_dtype, device=device),
    )

    # Per-subtree state
    K_per_sub = torch.zeros(S_count, dtype=torch.int64, device=device)
    S_per_sub = torch.zeros(S_count, dtype=torch.float64, device=device)
    N_per_sub = seed_N.to(torch.float64)

    p1_indices_list: List[torch.Tensor] = []
    p1_log_probs_list: List[torch.Tensor] = []
    p1_subtree_ids_list: List[torch.Tensor] = []

    def compute_thresholds():
        remaining_N = N_per_sub - K_per_sub.to(torch.float64)
        remaining_S = 1.0 - S_per_sub
        safe = (remaining_S > FRAC_EPS) & (remaining_N > FRAC_EPS)
        nws = torch.where(safe, remaining_N / remaining_S,
                          torch.full_like(remaining_N, float('inf')))
        threshold_log = torch.where(
            nws > 0,
            -torch.log(nws.clamp(min=1e-300)),
            torch.full_like(nws, float('-inf')),
        )
        return threshold_log

    for _ in range(max_iters):
        if not pool.active.any():
            break

        threshold_log = compute_thresholds()
        active = pool.active
        sid = pool.subtree_ids.long()
        lp = pool.log_probs
        de = pool.depths.long()

        entry_thresh = threshold_log[sid]
        above = active & (lp > entry_thresh + FRAC_EPS)
        if not above.any():
            break

        is_leaf = de == num_levels
        leaf_above = above & is_leaf
        internal_above = above & (~is_leaf)

        if leaf_above.any():
            idx = leaf_above.nonzero(as_tuple=True)[0]
            lp_committed = lp[idx].clone()
            sid_committed = sid[idx]
            ones64 = torch.ones_like(lp_committed, dtype=torch.int64)
            K_per_sub.scatter_add_(0, sid_committed, ones64)
            probs = lp_committed.exp()
            S_per_sub.scatter_add_(0, sid_committed, probs)

            p1_indices_list.append(idx.to(torch.int32).clone())
            p1_log_probs_list.append(lp_committed)
            p1_subtree_ids_list.append(sid_committed.to(torch.int32))

            pool.active[idx] = False
            continue

        # Expand internals — batched by depth
        int_idx = internal_above.nonzero(as_tuple=True)[0]
        int_depths_snap = de[int_idx].clone()
        int_lp_snap = lp[int_idx].clone()
        int_sid_snap = sid[int_idx].clone()
        # Save parent indices for children's parent_idx pointer
        parent_pool_idx = int_idx.clone().to(torch.int32)
        pool.active[int_idx] = False

        unique_d = int_depths_snap.unique()
        all_child_lp = []
        all_child_depth = []
        all_child_sid = []
        all_child_parent = []
        all_child_var = []

        for d_t in unique_d:
            d = int(d_t.item())
            level = tree.levels[num_levels - d - 1]
            D = level.domain_size

            mask = (int_depths_snap == d_t)
            B = int(mask.sum().item())
            batch_lp = int_lp_snap[mask]
            batch_sid = int_sid_snap[mask]
            batch_parent_pool_idx = parent_pool_idx[mask]
            batch_pool_idx_global = int_idx[mask]  # original pool indices

            # Materialize partial state[:, :d] for this batch
            # (only need the first d columns)
            batch_partial_d = pool.materialize_partial(
                batch_pool_idx_global, max_depth=d
            )  # (B, d)

            var_order_so_far = [
                tree.levels[num_levels - k - 1].elim_var_label
                for k in range(d)
            ]
            cond_ln = _conditional_log_probs_batch(
                level, batch_partial_d, var_order_so_far, device
            ).to(torch.float64)  # (B, D)

            # Children: (B × D)
            children_lp = (batch_lp.unsqueeze(1) + cond_ln).reshape(-1)
            child_var_values = (torch.arange(D, dtype=partial_dtype, device=device)
                                .unsqueeze(0).expand(B, D).reshape(-1))
            children_sid = batch_sid.unsqueeze(1).expand(B, D).reshape(-1)
            # Each child's parent is the parent's pool index, repeated D times
            children_parent = batch_parent_pool_idx.unsqueeze(1).expand(B, D).reshape(-1)
            children_depth = torch.full((B * D,), d + 1, dtype=torch.int32, device=device)

            keep = children_lp > float('-inf')
            if keep.any():
                all_child_lp.append(children_lp[keep])
                all_child_depth.append(children_depth[keep])
                all_child_sid.append(children_sid[keep])
                all_child_parent.append(children_parent[keep])
                all_child_var.append(child_var_values[keep])

        if all_child_lp:
            pool.add(
                log_probs=torch.cat(all_child_lp),
                depths=torch.cat(all_child_depth),
                subtree_ids=torch.cat(all_child_sid),
                parent_idx=torch.cat(all_child_parent),
                var_value=torch.cat(all_child_var),
            )

    if p1_indices_list:
        p1_indices = torch.cat(p1_indices_list)
        p1_log_probs = torch.cat(p1_log_probs_list)
        p1_subtree_ids = torch.cat(p1_subtree_ids_list)
        # Materialize partial states for the phase-1 commits (full depth)
        p1_partial = pool.materialize_partial(p1_indices, max_depth=num_levels)
    else:
        p1_log_probs = torch.empty(0, device=device, dtype=torch.float64)
        p1_partial = torch.empty((0, num_levels), device=device, dtype=partial_dtype)
        p1_subtree_ids = torch.empty(0, device=device, dtype=torch.int32)

    return pool, K_per_sub, S_per_sub, p1_log_probs, p1_partial, p1_subtree_ids


# ---------------------------------------------------------------------------
# Batched phase 2 with lean pool
# ---------------------------------------------------------------------------

def _batched_phase2_lean(
    tree: ProposalTree,
    pool: _LeanPool,
    K_per_sub: torch.Tensor,
    S_per_sub: torch.Tensor,
    N_per_sub: torch.Tensor,
    rng: torch.Generator,
):
    """Decimal resolution for the terminal level."""
    device = tree.device
    num_levels = len(tree.levels)
    partial_dtype = pool.partial_dtype

    active_idx = pool.active.nonzero(as_tuple=True)[0]
    if active_idx.numel() == 0:
        return (torch.empty((0, num_levels), dtype=partial_dtype, device=device),
                torch.empty(0, device=device, dtype=torch.float64),
                torch.empty(0, device=device, dtype=torch.int32))

    f_lp = pool.log_probs[active_idx]
    f_depth = pool.depths[active_idx].long()
    f_sid = pool.subtree_ids[active_idx].long()
    f_pool_idx = active_idx.to(torch.int32)

    p2_budget_per_sub = (N_per_sub.to(torch.float64)
                         - K_per_sub.to(torch.float64)).clamp(min=0)
    entry_budget = p2_budget_per_sub[f_sid]
    entry_remainder = (1.0 - S_per_sub[f_sid]).clamp(min=FRAC_EPS)
    log_alloc = (torch.log(entry_budget.clamp(min=FRAC_EPS))
                 + f_lp - torch.log(entry_remainder))
    allocated = log_alloc.exp()
    allocated = torch.where(entry_budget > FRAC_EPS, allocated, torch.zeros_like(allocated))

    floors = allocated.floor()
    fracs = (allocated - floors).clamp(min=0, max=1)
    fracs = torch.where(fracs < FRAC_EPS, torch.zeros_like(fracs), fracs)
    fracs = torch.where(fracs > 1 - FRAC_EPS, torch.zeros_like(fracs), fracs)
    integer_counts = floors.clone().to(torch.int64)

    unique_sub, inv = torch.unique(f_sid, return_inverse=True)
    for sidx_i in range(unique_sub.numel()):
        s_mask = inv == sidx_i
        s = int(unique_sub[sidx_i].item())
        budget_s = int(p2_budget_per_sub[s].item())
        floors_s_sum = int(floors[s_mask].sum().item())
        need = budget_s - floors_s_sum
        if need <= 0:
            continue
        s_fracs = fracs[s_mask]
        extras = _systematic_round_1d(s_fracs, need, rng, device)
        integer_counts[s_mask] += extras.to(torch.int64)

    keep = integer_counts > 0
    if not keep.any():
        return (torch.empty((0, num_levels), dtype=partial_dtype, device=device),
                torch.empty(0, device=device, dtype=torch.float64),
                torch.empty(0, device=device, dtype=torch.int32))

    sel_lp = f_lp[keep]
    sel_depth = f_depth[keep]
    sel_sid = f_sid[keep]
    sel_pool_idx = f_pool_idx[keep]
    sel_count = integer_counts[keep]

    rep_lp = sel_lp.repeat_interleave(sel_count)
    rep_depth = sel_depth.repeat_interleave(sel_count)
    rep_sid = sel_sid.repeat_interleave(sel_count)
    rep_pool_idx = sel_pool_idx.repeat_interleave(sel_count)

    out_partial_chunks: List[torch.Tensor] = []
    out_lp_chunks: List[torch.Tensor] = []
    out_sid_chunks: List[torch.Tensor] = []

    unique_start_depths = rep_depth.unique()
    for start_d_t in unique_start_depths:
        start_d = int(start_d_t.item())
        mask_sd = rep_depth == start_d_t
        # Materialize partial states for this group up to start_d
        idx_sd = rep_pool_idx[mask_sd]
        # Materialize FULL partial state (we'll fill columns >= start_d below)
        p_slice = pool.materialize_partial(idx_sd, max_depth=num_levels).clone()
        lp_slice = rep_lp[mask_sd].clone()
        sid_slice = rep_sid[mask_sd]

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
# Driver — same as multilevel.py but using the lean variants
# ---------------------------------------------------------------------------

def sample_multilevel_lean(
    tree: ProposalTree,
    level_budgets: List[int],
    rng: Optional[torch.Generator] = None,
    mode: str = 'save',
    callback: Optional[Callable] = None,
    partial_dtype: torch.dtype = torch.int8,
    max_seeds_per_batch: int = 4096,  # raised since pool is now ~70x leaner
):
    """Drop-in alternative to sample_multilevel using the parent-pointer pool."""
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

    L = len(level_budgets)
    assert L >= 1

    seed_lp_nat = torch.zeros(1, dtype=torch.float64, device=device)
    seed_depth = torch.zeros(1, dtype=torch.long, device=device)
    seed_partial = torch.zeros((1, num_levels), dtype=partial_dtype, device=device)
    seed_N = torch.tensor([int(level_budgets[0])], dtype=torch.int64, device=device)
    seed_partial_offset_log10 = torch.zeros(1, dtype=torch.float64, device=device)

    def _process_seeds(slp, sde, spar, sN, soff, is_terminal, next_M):
        pool, K_sub, S_sub, p1_lp_nat, p1_partial, p1_sid = _batched_phase1_lean(
            tree, slp, sde, spar, sN, partial_dtype
        )
        if p1_lp_nat.numel() > 0:
            offsets = soff[p1_sid.long()]
            p1_lp_log10 = (p1_lp_nat / ln10 + offsets).to(torch.float32)
            p1_eff = torch.zeros_like(p1_lp_log10)
            _emit(p1_partial, p1_lp_log10, p1_eff)

        if is_terminal:
            p2_partial, p2_lp_nat, p2_sid = _batched_phase2_lean(
                tree, pool, K_sub, S_sub, sN.to(torch.float64), rng
            )
            if p2_partial.shape[0] > 0:
                offsets = soff[p2_sid.long()]
                p2_lp_log10 = (p2_lp_nat / ln10 + offsets).to(torch.float32)
                rem_N = (sN.to(torch.float64) - K_sub.to(torch.float64)).clamp(min=0)
                rem_S = (1.0 - S_sub).clamp(min=FRAC_EPS)
                nws_s = rem_N / rem_S
                log10_nws_s = torch.log10(nws_s.clamp(min=1.0))
                eff_at_sub = log10_nws_s.to(torch.float32)[p2_sid.long()]
                p2_lp_inner_log10 = (p2_lp_nat / ln10).to(torch.float32)
                p2_eff = p2_lp_inner_log10 + eff_at_sub
                _emit(p2_partial, p2_lp_log10, p2_eff)
            del pool
            return None

        active_idx = pool.active.nonzero(as_tuple=True)[0]
        if active_idx.numel() == 0:
            del pool
            return None

        f_lp = pool.log_probs[active_idx].clone()
        f_depth = pool.depths[active_idx].long().clone()
        f_sid = pool.subtree_ids[active_idx].long().clone()
        # Materialize frontier partial states (needed as next-level seeds)
        f_partial = pool.materialize_partial(active_idx, max_depth=num_levels).clone()
        del pool

        rem_N = (sN.to(torch.float64) - K_sub.to(torch.float64)).clamp(min=0)
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
        new_offsets = soff[f_sid] + (f_lp / ln10)

        return dict(
            lp_nat=torch.zeros_like(f_lp),
            depth=f_depth,
            partial=f_partial,
            N=new_N,
            offset_log10=new_offsets,
        )

    for level_i in range(L):
        is_terminal = (level_i == L - 1)
        next_M = float(level_budgets[level_i + 1]) if not is_terminal else None
        S_total = seed_lp_nat.shape[0]

        accumulated = []
        for start in range(0, S_total, max_seeds_per_batch):
            end = min(start + max_seeds_per_batch, S_total)
            result = _process_seeds(
                seed_lp_nat[start:end], seed_depth[start:end],
                seed_partial[start:end], seed_N[start:end],
                seed_partial_offset_log10[start:end],
                is_terminal, next_M,
            )
            if result is not None:
                accumulated.append(result)
            torch.cuda.empty_cache()

        if is_terminal or not accumulated:
            break

        seed_lp_nat = torch.cat([f['lp_nat'] for f in accumulated])
        seed_depth = torch.cat([f['depth'] for f in accumulated])
        seed_partial = torch.cat([f['partial'] for f in accumulated], dim=0)
        seed_N = torch.cat([f['N'] for f in accumulated])
        seed_partial_offset_log10 = torch.cat([f['offset_log10'] for f in accumulated])
        del accumulated

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


def _proposal_tree_sample_multilevel_lean(self, level_budgets, **kw):
    return sample_multilevel_lean(self, level_budgets, **kw)

ProposalTree.sample_multilevel_lean = _proposal_tree_sample_multilevel_lean
