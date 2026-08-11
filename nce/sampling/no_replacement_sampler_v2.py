"""
Vectorized no-replacement sampler.

Walks the proposal tree top-down, one level at a time, processing all active
positions in parallel. At each level:
  1. Compute conditional distribution over the current variable for every
     active position (batched tensor op).
  2. Split each position's integer count among children via randomized
     integer rounding (batched systematic sampling).
  3. Form the next-level set of active positions from children with count > 0.

Phase-1 / phase-2 distinction is implicit: a final leaf's effective inclusion
probability π(x) = prod over path of min(1, parent_count × P(v|parent)).
Leaves on paths where every ancestor's allocation ≥ 1 are deterministically
included (π = 1, phase-1 style); others have π < 1 (phase-2).

The IS weight used by downstream code is ``1 / π(x)`` — equivalent to
Horvitz-Thompson over the sampler's marginal inclusion probabilities.
"""
import math
from typing import Dict, List, Optional, Tuple

import torch

from nce.sampling.proposal_sampler import ProposalTree, BucketRecord

FRAC_EPS = 1e-9


def _conditional_log_probs_batch(
    level: BucketRecord,
    partial_states: torch.Tensor,  # (B, d) int64
    var_order_so_far: List[int],   # length d
    device: str,
) -> torch.Tensor:
    """
    Compute conditional log probabilities P(elim_var | partial_state) in batch.

    Returns log-probs (natural log), shape (B, D) where D is level's domain size.
    """
    B = partial_states.shape[0]
    D = level.domain_size
    elim_var = level.elim_var_label
    label_to_col = {lbl: i for i, lbl in enumerate(var_order_so_far)}

    log10_unnorm = torch.zeros((B, D), device=device)

    for factor in level.factors:
        if elim_var in factor.labels:
            # Build per-dim index tensors of shape (B, D). PyTorch advanced
            # indexing requires long dtype, so cast any non-long column.
            indices = []
            for lbl in factor.labels:
                if lbl == elim_var:
                    idx = torch.arange(D, dtype=torch.long, device=device).unsqueeze(0).expand(B, D)
                else:
                    col = label_to_col[lbl]
                    idx = partial_states[:, col].to(torch.long).unsqueeze(1).expand(B, D)
                indices.append(idx)
            vals = factor.tensor[tuple(indices)]  # (B, D)
            log10_unnorm = log10_unnorm + vals
        else:
            # Factor doesn't involve elim_var
            if not factor.labels:
                log10_unnorm = log10_unnorm + factor.tensor.squeeze()
                continue
            indices = [partial_states[:, label_to_col[lbl]].to(torch.long)
                       for lbl in factor.labels]
            vals = factor.tensor[tuple(indices)]  # (B,)
            log10_unnorm = log10_unnorm + vals.unsqueeze(1)

    # Normalize to proper log-probabilities in natural log
    ln = log10_unnorm * math.log(10)
    ln = ln - torch.logsumexp(ln, dim=1, keepdim=True)
    return ln


def _systematic_round_batch(
    fracs: torch.Tensor,           # (B, D), each row's values in [0, 1), sum ≈ integer
    extras_needed: torch.Tensor,   # (B,), integer (as float or int)
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Batched Madow-style systematic rounding.

    For each row b, mark exactly ``extras_needed[b]`` cells to round up, with
    marginal P(cell i selected) = fracs[b, i]. Returns a (B, D) tensor of
    0s and 1s.
    """
    B, D = fracs.shape
    device = fracs.device

    extras_int = extras_needed.to(torch.int64)
    max_e = int(extras_int.max().item()) if B > 0 else 0
    if max_e == 0:
        return torch.zeros_like(fracs)

    # Per-row uniform random offset
    if generator is not None:
        u = torch.rand(B, 1, device=device, generator=generator)
    else:
        u = torch.rand(B, 1, device=device)

    # Thresholds u, u+1, ..., u+max_e-1
    js = torch.arange(max_e, device=device, dtype=fracs.dtype).unsqueeze(0)  # (1, max_e)
    thresholds = u + js                                                       # (B, max_e)

    # Only the first extras_needed[b] thresholds in each row are valid
    valid = js < extras_int.unsqueeze(1).to(fracs.dtype)                      # (B, max_e)

    # Cumulative fractional values per row (B, D)
    cum = fracs.cumsum(dim=1)

    # For each threshold, find the cell index via searchsorted
    indices = torch.searchsorted(cum, thresholds)                             # (B, max_e)
    indices = indices.clamp(max=D - 1)

    # Scatter 1s into the (B, D) extras tensor
    extras = torch.zeros_like(fracs)
    values = valid.to(fracs.dtype)
    extras.scatter_add_(dim=1, index=indices, src=values)
    # Multiple writes to the same cell are possible only under FP noise; clamp at 1
    return extras.clamp(max=1.0)


def sample_no_replacement_vectorized(
    tree: ProposalTree,
    N: int,
    rng: Optional[torch.Generator] = None,
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Vectorized exact-N no-replacement sampler.

    Returns:
        samples:             dict var_label -> LongTensor (N,)
        log_probs:           (N,) log10 proposal probability under WMB
        effective_log_probs: (N,) log10 effective q for IS weighting
                              (= -log10(marginal inclusion probability))
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

    # Traversal order: root (last-eliminated) first, leaves (first-eliminated) last
    var_order = [tree.levels[num_levels - d - 1].elim_var_label for d in range(num_levels)]
    domain_sizes = [tree.levels[num_levels - d - 1].domain_size for d in range(num_levels)]
    # Max number of leaves under each child at each level: product of domain
    # sizes from the next level onward. After the last level (d == num_levels-1),
    # children are leaves so max = 1.
    # Cap subtree-size at 1e18 — for any practical N we never need more.
    max_subtree_size_after = [1] * num_levels  # max_subtree[d] = max leaves in a child of level-d node
    running = 1
    for d in range(num_levels - 1, -1, -1):
        max_subtree_size_after[d] = running
        running = min(running * domain_sizes[d], int(1e18))

    # Active positions: partial_states[b, :] contains assignments for var_order[:d]
    partial_states = torch.zeros((1, 0), dtype=torch.long, device=device)
    counts = torch.tensor([float(N)], device=device, dtype=torch.float64)
    log_probs_nat = torch.zeros(1, device=device, dtype=torch.float64)
    # Phase-1 tracker: True if every ancestor's allocation was ≥ 1 (deterministic path)
    is_phase1 = torch.ones(1, dtype=torch.bool, device=device)

    for d in range(num_levels):
        level = tree.levels[num_levels - d - 1]
        D = domain_sizes[d]

        # Batched conditional: (B, D) natural-log probs
        cond_ln = _conditional_log_probs_batch(
            level, partial_states, var_order[:d], device
        ).to(torch.float64)

        # Allocate continuous counts per child cell
        probs = cond_ln.exp()                                     # (B, D)
        allocated = counts.unsqueeze(1) * probs                   # (B, D)

        # No subtree can hold more unique samples than the number of leaves
        # in it. Cap each cell at max_per_child, redistributing the excess
        # to siblings with remaining capacity (iteratively).
        max_per_child = float(max_subtree_size_after[d])
        for _ in range(8):  # water-filling loop; converges in <= D steps typically
            excess = (allocated - max_per_child).clamp(min=0)
            total_excess = excess.sum(dim=1, keepdim=True)           # (B, 1)
            if torch.all(total_excess < 1e-9):
                break
            allocated = allocated.clamp(max=max_per_child)
            available = (max_per_child - allocated).clamp(min=0)
            avail_sum = available.sum(dim=1, keepdim=True)
            # Guard: no sibling capacity → drop the excess (parent truly saturated)
            ratio = torch.where(
                avail_sum > 0,
                total_excess / (avail_sum + 1e-30),
                torch.zeros_like(avail_sum),
            )
            allocated = allocated + available * ratio
        allocated = allocated.clamp(max=max_per_child)

        floors = allocated.floor()
        fracs = allocated - floors
        # Clean near-boundary fractions
        fracs = torch.where(fracs < FRAC_EPS, torch.zeros_like(fracs), fracs)
        fracs = torch.where(fracs > 1 - FRAC_EPS, torch.zeros_like(fracs), fracs)
        floors = floors + (allocated - floors).gt(1 - FRAC_EPS).to(floors.dtype)

        # Extras to round up per row so child counts sum matches the capped
        # allocation (not counts, since some mass may have been lost to
        # subtree saturation).
        capped_sum = allocated.sum(dim=1)                         # (B,)
        extras_needed = (capped_sum - floors.sum(dim=1)).round()
        extras_needed = extras_needed.clamp(min=0)

        extras_mask = _systematic_round_batch(fracs, extras_needed, generator=rng)
        child_counts = floors + extras_mask                       # (B, D)

        # Per-cell phase-1 status: the cell is "phase-1" iff its parent was
        # phase-1 AND allocated[b, v] ≥ 1 (deterministic at this level).
        cell_is_phase1 = is_phase1.unsqueeze(1) & (allocated >= 1.0)          # (B, D)

        # Flatten (B, D) to new B' = B*D rows; then filter by count > 0
        new_partial_states = partial_states.repeat_interleave(D, dim=0)       # (B*D, d)
        new_var_col = torch.arange(D, device=device).repeat(partial_states.shape[0])
        new_partial_states = torch.cat(
            [new_partial_states, new_var_col.unsqueeze(1)], dim=1
        )                                                                     # (B*D, d+1)
        new_counts = child_counts.reshape(-1)
        new_log_probs = log_probs_nat.unsqueeze(1) + cond_ln                  # (B, D)
        new_log_probs = new_log_probs.reshape(-1)
        new_is_phase1 = cell_is_phase1.reshape(-1)

        keep_mask = new_counts > 0
        partial_states = new_partial_states[keep_mask]
        counts = new_counts[keep_mask]
        log_probs_nat = new_log_probs[keep_mask]
        is_phase1 = new_is_phase1[keep_mask]

    # Final: each position is a full sample with count 1 (after leaf capping).
    # If any still have count > 1 due to FP noise, duplicate them accordingly.
    counts_int = counts.round().to(torch.int64).clamp(min=0)
    if (counts_int == 1).all():
        final_partial = partial_states
        final_log_probs_nat = log_probs_nat
        final_is_phase1 = is_phase1
    else:
        final_partial = partial_states.repeat_interleave(counts_int, dim=0)
        final_log_probs_nat = log_probs_nat.repeat_interleave(counts_int)
        final_is_phase1 = is_phase1.repeat_interleave(counts_int)

    # Trim / top up to exactly N samples
    actual = final_partial.shape[0]
    topup_extra_partial = topup_extra_log_probs_nat = None
    topup_count = 0
    if actual > N:
        final_partial = final_partial[:N]
        final_log_probs_nat = final_log_probs_nat[:N]
        final_is_phase1 = final_is_phase1[:N]
    elif actual < N:
        topup_count = N - actual
        # Fall back to regular WMB samples for the deficit
        extra_samples, extra_lp_log10 = tree.sample(topup_count)
        topup_extra_partial = torch.stack(
            [extra_samples[v] for v in var_order], dim=1
        ).to(device=device)
        topup_extra_log_probs_nat = extra_lp_log10.to(torch.float64) * math.log(10)
        final_partial = torch.cat([final_partial, topup_extra_partial], dim=0)
        final_log_probs_nat = torch.cat([final_log_probs_nat, topup_extra_log_probs_nat])
        final_is_phase1 = torch.cat([
            final_is_phase1,
            torch.zeros(topup_count, dtype=torch.bool, device=device),
        ])

    # --- Compute effective log probs using global nws ---
    # The walk's `is_phase1` flag corresponds to threshold 1/N, but v1's
    # algorithm uses threshold 1/nws (with nws = (N-K)/(1-S)). Re-identify
    # phase-1 samples iteratively post-hoc: sort by q descending and add
    # samples to phase-1 as long as q × nws ≥ 1.
    log_probs_log10 = final_log_probs_nat / math.log(10)

    log_probs_sorted, sort_idx = log_probs_log10.sort(descending=True)
    qs_sorted = (log_probs_sorted * math.log(10)).exp()  # 10**log10(q)

    K = 0
    S = 0.0
    qs_cpu = qs_sorted.detach().cpu().numpy()
    for k in range(N):
        q_k = float(qs_cpu[k])
        new_S = S + q_k
        if new_S >= 1.0 - FRAC_EPS:
            break
        new_nws = (N - (k + 1)) / (1.0 - new_S)
        if q_k * new_nws >= 1.0 - FRAC_EPS:
            S = new_S
            K = k + 1
        else:
            break

    if K >= N or (1.0 - S) <= 0:
        final_nws = float(max(N, 1))
    else:
        final_nws = (N - K) / (1.0 - S)

    # Phase-1 mask in sorted order, then unsort
    sorted_p1_mask = torch.zeros(N, dtype=torch.bool, device=device)
    sorted_p1_mask[:K] = True
    final_is_phase1 = torch.zeros(N, dtype=torch.bool, device=device)
    final_is_phase1[sort_idx] = sorted_p1_mask

    # eff_log_prob convention (matches v1's training encoding):
    #   phase-1: eff = -log10(nws)   (so 1/exp(eff_nat) = nws)
    #   phase-2: eff = log10(q(x))   (so 1/exp(eff_nat) = 1/q)
    log10_nws = math.log10(final_nws) if final_nws > 0 else 0.0
    eff_log_probs = torch.where(
        final_is_phase1,
        torch.full_like(log_probs_log10, -log10_nws),
        log_probs_log10,
    )

    samples_dict = {var_order[d]: final_partial[:, d] for d in range(num_levels)}
    out_log_probs_log10 = log_probs_log10.to(torch.float32)
    out_eff_log_probs_log10 = eff_log_probs.to(torch.float32)
    return samples_dict, out_log_probs_log10, out_eff_log_probs_log10


# Attach as method for convenience
def _proposal_tree_sample_no_replacement_v2(self, N, rng=None):
    return sample_no_replacement_vectorized(self, N, rng=rng)

ProposalTree.sample_no_replacement_v2 = _proposal_tree_sample_no_replacement_v2
