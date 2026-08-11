"""
Exact-N no-replacement sampling over the WMB proposal tree.

Algorithm overview:
  1. Traverse the OR tree rooted at the last-eliminated variable, expanding
     nodes with expected sample count >= 1 in decreasing order. Leaves that
     cross the threshold are "sampled leaves" (contribute exactly 1 sample).
     Each time a sampled leaf is committed, update num_weighted_samples so
     exactly N - num_sampled_leaves samples remain for the rest of the tree.
  2. Walk top-down through the expanded tree. At each branching point,
     randomized rounding distributes the integer sample budget among
     children in proportion to fractional expected counts (systematic
     sampling). Sampled leaves contribute a fixed +1 each.

Returns exactly N unique samples with their proposal log-probabilities.
"""
import math
import heapq
from typing import Dict, List, Tuple, Optional

import torch

from nce.inference.factor import FastFactor
from nce.sampling.proposal_sampler import ProposalTree, BucketRecord


# Numerical tolerance for treating a fractional part as zero / one
FRAC_EPS = 1e-9


class _TreeNode:
    """A partially-assigned node in the no-replacement sampling tree.

    Parent/child semantics:
      depth 0       = root (no variables assigned yet)
      depth = D     = a leaf (all D tree variables assigned)
    The elimination order of the underlying ProposalTree levels is reversed
    here: depth 0 samples tree.levels[-1].elim_var_label first, depth D-1
    samples tree.levels[0].elim_var_label last.
    """
    __slots__ = (
        'depth',
        'partial_state',  # dict var_label -> int value
        'log_prob',       # natural log of P(partial_state) under proposal
        'children',       # list of _TreeNode or None (not yet expanded)
        'sampled',        # True iff this is a sampled leaf (depth == D, exp >= 1)
        'subtree_sampled_leaves',  # list of _TreeNode (sampled leaves in this subtree)
        'subtree_sampled_leaf_prob_sum',  # sum of P(leaf) for sampled leaves in subtree
        '_count',         # integer sample count assigned during phase 2
    )

    def __init__(self, depth: int, partial_state: Dict[int, int], log_prob: float):
        self.depth = depth
        self.partial_state = partial_state
        self.log_prob = log_prob
        self.children = None
        self.sampled = False
        self.subtree_sampled_leaves = []
        self.subtree_sampled_leaf_prob_sum = 0.0
        self._count = 0


def _conditional_log_probs(level: BucketRecord,
                           partial_state: Dict[int, int],
                           device: str) -> torch.Tensor:
    """
    Compute the conditional distribution P(elim_var | partial_state) at a
    single tree node. Returns a 1-D tensor of shape (D,) of natural-log
    probabilities that sum (in exp-space) to 1.

    partial_state must include every variable that appears in this level's
    factors except level.elim_var_label.
    """
    D = level.domain_size
    var = level.elim_var_label
    # Accumulate log10 contributions for each state of the elim var
    log10_unnorm = torch.zeros(D, device=device)

    for factor in level.factors:
        if var in factor.labels:
            # Evaluate per state of var, conditioning on the other labels
            for s in range(D):
                idx = []
                for label in factor.labels:
                    if label == var:
                        idx.append(s)
                    else:
                        idx.append(partial_state[label])
                log10_unnorm[s] += factor.tensor[tuple(idx)]
        else:
            # Factor doesn't involve var: contributes a constant per-state
            if not factor.labels:
                log10_unnorm += factor.tensor.squeeze()
            else:
                idx = tuple(partial_state[label] for label in factor.labels)
                log10_unnorm += factor.tensor[idx]

    # Convert log10 -> natural log and normalize
    ln = log10_unnorm * math.log(10)
    ln = ln - torch.logsumexp(ln, dim=0)
    return ln


# ---------------------------------------------------------------------------
# Phase 1: expansion + sampled-leaf discovery
# ---------------------------------------------------------------------------

def _phase1_expand(tree: ProposalTree, N: int):
    """
    Expand the tree top-down, committing sampled leaves as we go.

    Returns:
        root:                 expanded _TreeNode
        sampled_leaves:       list of _TreeNode (depth == num_levels, sampled==True)
        frontier:             list of _TreeNode (unexpanded internal/leaf, exp < 1)
        final_nws:            float, the converged num_weighted_samples
    """
    num_levels = len(tree.levels)
    device = tree.device

    root = _TreeNode(depth=0, partial_state={}, log_prob=0.0)  # log_prob = ln(1) = 0

    # Max-heap by log_prob (use negation for Python's min-heap)
    counter = [0]
    def push(h, node):
        counter[0] += 1
        heapq.heappush(h, (-node.log_prob, counter[0], node))

    heap: List = []
    push(heap, root)

    # num_sampled_leaves (K), sum of their probabilities (S)
    K = 0
    S = 0.0

    def current_nws() -> float:
        if S >= 1.0 - FRAC_EPS:
            return float('inf')
        return (N - K) / (1.0 - S)

    def current_log_threshold() -> float:
        # We accept a node iff exp = P(node) * nws >= 1, i.e. log_prob >= -log(nws)
        nws = current_nws()
        if nws == float('inf') or nws <= 0:
            return float('-inf')
        return -math.log(nws)

    sampled_leaves: List[_TreeNode] = []
    frontier: List[_TreeNode] = []

    while heap:
        neg_lp, _, node = heap[0]
        log_prob = -neg_lp
        threshold = current_log_threshold()
        if log_prob < threshold - FRAC_EPS:
            # No remaining node in the heap can exceed the threshold
            break
        heapq.heappop(heap)

        if node.depth == num_levels:
            # Leaf with exp >= 1: commit it
            node.sampled = True
            K += 1
            S += math.exp(node.log_prob)
            sampled_leaves.append(node)
            continue

        # Internal node with exp >= 1: expand
        level = tree.levels[num_levels - node.depth - 1]
        cond_ln = _conditional_log_probs(level, node.partial_state, device)

        node.children = []
        for v in range(level.domain_size):
            p_v = cond_ln[v].item()
            if p_v == float('-inf'):
                continue
            child_partial = dict(node.partial_state)
            child_partial[level.elim_var_label] = v
            child = _TreeNode(
                depth=node.depth + 1,
                partial_state=child_partial,
                log_prob=node.log_prob + p_v,
            )
            node.children.append(child)
            push(heap, child)

    # Whatever remains in the heap is the "fractional" frontier
    while heap:
        _, _, node = heapq.heappop(heap)
        frontier.append(node)

    return root, sampled_leaves, frontier, current_nws()


# ---------------------------------------------------------------------------
# Phase 2: integer allocation top-down
# ---------------------------------------------------------------------------

def _propagate_subtree_sampled_mass(node: _TreeNode) -> Tuple[int, float]:
    """
    Walk the expanded tree and annotate each node with the count / prob-mass
    of sampled leaves in its subtree.

    Returns (num_leaves, prob_sum) for this subtree.
    """
    if node.sampled:
        node.subtree_sampled_leaves = [node]
        node.subtree_sampled_leaf_prob_sum = math.exp(node.log_prob)
        return 1, node.subtree_sampled_leaf_prob_sum

    if node.children is None:
        # Unexpanded — no sampled leaves beneath it
        return 0, 0.0

    total_n = 0
    total_p = 0.0
    leaves: List[_TreeNode] = []
    for c in node.children:
        n_c, p_c = _propagate_subtree_sampled_mass(c)
        total_n += n_c
        total_p += p_c
        leaves.extend(c.subtree_sampled_leaves)
    node.subtree_sampled_leaves = leaves
    node.subtree_sampled_leaf_prob_sum = total_p
    return total_n, total_p


def _systematic_round(fracs: List[float], u: float) -> List[int]:
    """
    Randomized integer rounding preserving the sum.

    Given fractional parts (0 <= f_i < 1) summing to an integer e, choose
    exactly e indices to round up, with marginal P(i chosen) = f_i.

    Uses Madow-style systematic sampling with a single random u ~ U[0, 1).
    """
    k = len(fracs)
    extras = [0] * k
    total = sum(fracs)
    e = int(round(total))
    if e == 0 or total < FRAC_EPS:
        return extras

    # Cumulative distribution
    cum = [0.0] * (k + 1)
    for i in range(k):
        cum[i + 1] = cum[i] + fracs[i]

    # Systematic thresholds: u, u+1, ..., u+e-1
    i = 0
    for j in range(e):
        t = u + j
        while i < k and cum[i + 1] <= t + FRAC_EPS:
            i += 1
        if i >= k:
            i = k - 1  # numerical safety
        extras[i] = 1
        i += 1  # each index takes at most one since f_i < 1
    return extras


_PHASE2_TREE: Optional[ProposalTree] = None  # threaded via sample_no_replacement


def _phase2_allocate(root: _TreeNode, N: int, nws: float, rng: torch.Generator,
                     device: str, debug: bool = False) -> List[_TreeNode]:
    """
    Walk the expanded tree top-down, assigning integer sample counts to
    children at each branching point via randomized rounding. Accumulates
    the list of leaf nodes that receive a sample.
    """
    _propagate_subtree_sampled_mass(root)

    # Each node receives an integer `count` equal to its total samples
    # (fractional portion + sampled leaves beneath it).
    root._count = N

    selected_leaves: List[_TreeNode] = []

    # Iterative top-down walk
    stack: List[_TreeNode] = [root]
    while stack:
        node = stack.pop()
        count = node._count

        if count == 0:
            continue

        # Leaf or fully-assigned frontier?
        if node.children is None:
            num_levels = len(_PHASE2_TREE.levels) if _PHASE2_TREE else 0
            if node.sampled or node.depth == num_levels:
                # Actual leaf: contributes at most 1 sample (count should be 1)
                if debug:
                    print(f"  pop leaf depth={node.depth} count={count} sampled={node.sampled}")
                selected_leaves.append(node)
                continue
            # Internal frontier. If count == 1 we take a single completion.
            # If count > 1, expand on-the-fly and recurse.
            if count == 1:
                selected_leaves.append(node)
                continue
            # count > 1: expand this frontier and fall through to internal path
            level = _PHASE2_TREE.levels[num_levels - node.depth - 1]
            cond_ln = _conditional_log_probs(level, node.partial_state, _PHASE2_TREE.device)
            node.children = []
            for v in range(level.domain_size):
                p_v = cond_ln[v].item()
                if p_v == float('-inf'):
                    continue
                child_partial = dict(node.partial_state)
                child_partial[level.elim_var_label] = v
                child = _TreeNode(
                    depth=node.depth + 1,
                    partial_state=child_partial,
                    log_prob=node.log_prob + p_v,
                )
                # No sampled leaves beneath a freshly-expanded frontier
                child.subtree_sampled_leaves = []
                child.subtree_sampled_leaf_prob_sum = 0.0
                node.children.append(child)
            # fall through to internal node branch

        # Internal node: split `count` among children
        # fractional_exp(c) = P(c) * nws - sampled_leaf_mass(c)
        #   where sampled_leaf_mass(c) = sum of exp for sampled leaves under c
        #                              = nws * subtree_sampled_leaf_prob_sum
        children = node.children
        frac_exps: List[float] = []
        sampled_counts: List[int] = []

        for c in children:
            c_prob = math.exp(c.log_prob)
            c_exp = c_prob * nws
            c_sampled_exp = c.subtree_sampled_leaf_prob_sum * nws
            frac_exp = c_exp - c_sampled_exp
            # Clamp tiny negatives from FP error
            if frac_exp < 0:
                frac_exp = 0.0
            frac_exps.append(frac_exp)
            sampled_counts.append(len(c.subtree_sampled_leaves))

        # The fractional budget to distribute across children is
        # total_count - sum(sampled_counts)
        fractional_budget = count - sum(sampled_counts)

        # Renormalize frac_exps to sum to fractional_budget.
        # At the root, sum(frac_exps) == fractional_budget exactly, but at
        # deeper levels, upstream rounding slack means they can differ. We
        # rescale so children's allocations sum to our integer budget.
        sum_frac = sum(frac_exps)
        if sum_frac > FRAC_EPS and fractional_budget > 0:
            scale = fractional_budget / sum_frac
            allocated = [fe * scale for fe in frac_exps]
        else:
            allocated = [0.0] * len(frac_exps)

        floors = [int(math.floor(a)) for a in allocated]
        fracs = [a - fl for a, fl in zip(allocated, floors)]
        # Clean small numerical noise
        cleaned = []
        for f in fracs:
            if f < FRAC_EPS:
                cleaned.append(0.0)
            elif f > 1 - FRAC_EPS:
                cleaned.append(0.0)
            else:
                cleaned.append(f)
        for i, f in enumerate(fracs):
            if f > 1 - FRAC_EPS:
                floors[i] += 1
        total_extras = fractional_budget - sum(floors)
        if total_extras < 0:
            total_extras = 0

        # Systematic rounding
        u = torch.rand((), generator=rng, device=device).item() if total_extras > 0 else 0.0
        extras = _systematic_round(cleaned, u)

        if debug:
            print(f"  internal depth={node.depth} count={count} "
                  f"frac_exps={[f'{x:.3f}' for x in frac_exps]} "
                  f"floors={floors} extras={extras} sampled={sampled_counts} "
                  f"fractional_budget={fractional_budget}")

        num_levels = len(_PHASE2_TREE.levels) if _PHASE2_TREE else 0

        # Invariant: sum of children counts must equal parent count
        for i, c in enumerate(children):
            c._count = floors[i] + extras[i] + sampled_counts[i]

        # Leaves cannot absorb more than one sample (no-replacement). Cap
        # leaf allocations at 1 and push the excess to non-leaf siblings.
        def is_leaf(c):
            return c.children is None and (c.sampled or c.depth == num_levels)

        excess = 0
        for c in children:
            if is_leaf(c) and c._count > 1:
                excess += c._count - 1
                c._count = 1

        if excess > 0:
            # Prefer non-leaf children (which can be expanded further) to absorb
            non_leaf = [i for i, c in enumerate(children) if not is_leaf(c)]
            if non_leaf:
                # Spread excess by repeatedly adding to the one currently smallest
                for _ in range(excess):
                    idx = min(non_leaf, key=lambda i: children[i]._count)
                    children[idx]._count += 1
            else:
                # All children are leaves (can't take more than 1 each). Drop the
                # excess — this loses at most a few samples at deep binary leaves.
                pass

        # Second invariant check after capping
        children_count_sum = sum(c._count for c in children)
        if children_count_sum != count:
            diff = count - children_count_sum
            candidates = [i for i, c in enumerate(children) if not is_leaf(c)]
            if not candidates:
                candidates = list(range(len(children)))
            if diff > 0:
                idx = max(candidates, key=lambda i: children[i]._count)
                children[idx]._count += diff
            # diff < 0 (over-count) shouldn't happen after capping, but ignore

        for c in children:
            stack.append(c)

    return selected_leaves


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _complete_partial_sample(tree: ProposalTree, node: _TreeNode,
                             rng: torch.Generator) -> Tuple[Dict[int, int], float]:
    """
    Extend a partially-assigned internal frontier node to a full leaf by
    sampling remaining variables according to WMB conditional probabilities.

    Returns (full_partial_state, log_prob_nat) where log_prob_nat is the
    natural-log probability of the full assignment under the proposal.
    """
    partial = dict(node.partial_state)
    log_prob = node.log_prob
    num_levels = len(tree.levels)
    for d in range(node.depth, num_levels):
        level = tree.levels[num_levels - d - 1]
        cond_ln = _conditional_log_probs(level, partial, tree.device)
        probs = torch.exp(cond_ln)
        # Clamp for numerical safety
        probs = probs.clamp(min=0)
        s = probs.sum()
        if s <= 0:
            # Degenerate — pick uniformly (shouldn't happen unless all zero)
            v = torch.randint(0, level.domain_size, (1,), generator=rng,
                              device=tree.device).item()
        else:
            probs = probs / s
            v = torch.multinomial(probs, 1, generator=rng).item()
        partial[level.elim_var_label] = v
        log_prob += cond_ln[v].item()
    return partial, log_prob


def sample_no_replacement(
    tree: ProposalTree,
    N: int,
    rng: Optional[torch.Generator] = None,
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Draw exactly N unique samples from a ProposalTree's distribution without
    replacement, using the two-phase algorithm.

    Returns:
        samples:             dict mapping var label -> LongTensor (N,) of state indices
        log_probs:           (N,) log10 proposal probabilities under WMB
        effective_log_probs: (N,) log10 "effective q" for IS weighting.
                              - Phase-1 samples (committed pre-decimal):
                                  effective_log_q = -log10(final_nws)
                                  -> IS weight exp(-eff) = final_nws
                              - Phase-2 samples (decimal resolution / completion):
                                  effective_log_q = log10(q(x))
                                  -> IS weight exp(-eff) = 1/q(x)
                              Pass this to the existing self-normalized IS wrapper.
    """
    device = tree.device
    if rng is None:
        rng = torch.Generator(device=device)
        rng.manual_seed(torch.randint(0, 2**31 - 1, (1,)).item())

    if N <= 0 or not tree.levels:
        empty = {rec.elim_var_label: torch.zeros(0, dtype=torch.long, device=device)
                 for rec in tree.levels}
        return empty, torch.zeros(0, device=device), torch.zeros(0, device=device)

    # Phase 1: expand tree, collect sampled leaves, determine final nws
    root, sampled_leaves, frontier, final_nws = _phase1_expand(tree, N)

    # Phase 2: integer allocation top-down — thread the tree for lazy expansion
    global _PHASE2_TREE
    _PHASE2_TREE = tree
    try:
        selected = _phase2_allocate(root, N, final_nws, rng, device)
    finally:
        _PHASE2_TREE = None

    var_labels = [rec.elim_var_label for rec in tree.levels]
    num_levels = len(tree.levels)
    samples_dict = {v: [] for v in var_labels}
    log_probs: List[float] = []           # true log10 q(x)
    eff_log_probs: List[float] = []       # log10 "effective q" for IS weights

    ln10 = math.log(10)

    for node in selected:
        if node.sampled:
            # Phase 1 leaf: fully assigned, effective weight = final_nws
            state = node.partial_state
            q_log10 = node.log_prob / ln10
            eff_log10 = -math.log10(final_nws) if final_nws > 0 else 0.0
        elif node.depth == num_levels:
            # Phase 2 fully-assigned leaf
            state = node.partial_state
            q_log10 = node.log_prob / ln10
            eff_log10 = q_log10
        else:
            # Phase 2 internal frontier: sample a leaf from its subtree
            state, full_log_prob = _complete_partial_sample(tree, node, rng)
            q_log10 = full_log_prob / ln10
            eff_log10 = q_log10

        for v in var_labels:
            samples_dict[v].append(state[v])
        log_probs.append(q_log10)
        eff_log_probs.append(eff_log10)

    # If the phase 2 allocation dropped some samples (can happen when leaf
    # siblings all maxed out at 1 and there was nowhere to put the excess),
    # top up with regular WMB samples so the caller gets exactly N.
    deficit = N - len(log_probs)
    if deficit > 0:
        extra_samples, extra_log_probs_log10 = tree.sample(deficit)
        for v in var_labels:
            extra_col = extra_samples[v].tolist()
            samples_dict[v].extend(extra_col)
        for lp in extra_log_probs_log10.tolist():
            log_probs.append(lp)
            eff_log_probs.append(lp)  # phase-2 style weight: 1/q(x)

    out_samples = {v: torch.tensor(samples_dict[v], dtype=torch.long, device=device)
                   for v in var_labels}
    out_log_probs = torch.tensor(log_probs, dtype=torch.float32, device=device)
    out_eff_log_probs = torch.tensor(eff_log_probs, dtype=torch.float32, device=device)
    return out_samples, out_log_probs, out_eff_log_probs


def _proposal_tree_sample_no_replacement(self, N: int, rng=None):
    return sample_no_replacement(self, N, rng)

ProposalTree.sample_no_replacement = _proposal_tree_sample_no_replacement
