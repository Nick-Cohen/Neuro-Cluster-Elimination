"""
Proposal sampler using WMB-based importance sampling.

Builds a proposal tree by running WMB elimination on the approximate
distribution (upstream + downstream factors), then generates samples
via backward traversal through the tree.
"""
import torch
import math
import copy
from typing import List, Dict, Tuple

from nce.inference.factor import FastFactor
from nce.inference.elimination_order import wtminfill_order
from nce.sampling import crn


class BucketRecord:
    """Snapshot of a bucket's factors before elimination."""
    __slots__ = ('elim_var_label', 'domain_size', 'factors')

    def __init__(self, elim_var_label: int, domain_size: int, factors: List[FastFactor]):
        self.elim_var_label = elim_var_label
        self.domain_size = domain_size
        self.factors = factors


class ProposalTree:
    """
    A WMB-based proposal tree for importance sampling.

    Built by running WMB elimination on a set of factors and saving the
    bucket state at each level. Supports backward-traversal sampling to
    generate samples along with their log proposal probabilities.
    """

    def __init__(self, levels: List[BucketRecord], device: str):
        self.levels = levels  # In elimination order (first eliminated first)
        self.device = device

    def sample(self, num_samples: int, crn_key: int = None
               ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """
        Generate samples via backward traversal.

        Traverses the tree from last eliminated (root) to first eliminated,
        sampling each variable conditioned on those already sampled.

        Args:
            num_samples: Number of samples to generate
            crn_key: Optional common-random-numbers stream key
                (`nce.sampling.crn.stream_key`). When given, each level is
                sampled by INVERSE CDF from the counter-based CRN stream
                instead of by `torch.multinomial` on the global RNG.

                Why inverse CDF and not "seed the generator": the proposal
                distribution q is a function of the cluster's factors, so two
                merge arms that build genuinely different clusters have
                genuinely different q -- which the requirement explicitly
                permits to give different samples. What must NOT differ is the
                underlying randomness. Driving the multinomial by a shared
                uniform makes the arms coincide exactly when q coincides, and
                maximally coupled (the inverse-CDF / monotone coupling) when it
                does not, instead of independent. It also carries the row-wise
                purity of the uniform path through unchanged: row i depends
                only on row i's uniforms, so a larger draw still extends a
                smaller one, and the stream does not depend on the device or on
                how many other clusters were sampled first.

                The stream column is keyed on the eliminated VARIABLE LABEL,
                not on the level position, so a variable that sits at a
                different depth in two arms' trees still draws the same column.

        Returns:
            samples: Dict mapping variable label -> LongTensor of shape (num_samples,)
            log_probs: Tensor of shape (num_samples,) with cumulative log10 proposal
                       probability of each sample
        """
        samples = {}
        log_probs = torch.zeros(num_samples, device=self.device)

        # Traverse in reverse elimination order (root first)
        for record in reversed(self.levels):
            var_label = record.elim_var_label
            D = record.domain_size

            # Accumulate unnormalized log10 probabilities for each state
            log_unnorm = torch.zeros(num_samples, D, device=self.device)

            for factor in record.factors:
                if var_label in factor.labels:
                    # Factor involves elim var: condition on sampled vars, keep elim var
                    contrib = _condition_factor(factor, var_label, samples)
                else:
                    # Factor doesn't involve elim var: evaluate fully, broadcast
                    contrib = _evaluate_factor(factor, samples).unsqueeze(1)
                log_unnorm += contrib

            # Convert to natural log for numerics, normalize, sample
            ln_unnorm = log_unnorm * math.log(10)
            ln_Z = torch.logsumexp(ln_unnorm, dim=1, keepdim=True)
            ln_probs = ln_unnorm - ln_Z  # (num_samples, D)

            # Sample via multinomial
            probs = ln_probs.exp().clamp(min=0)
            # Renormalize for numerical safety
            probs = probs / probs.sum(dim=1, keepdim=True)
            if crn_key is None:
                sampled_states = torch.multinomial(probs, 1).squeeze(1)  # (num_samples,)
            else:
                # Inverse CDF against the CRN uniform for THIS variable.
                # float64 throughout: the uniform is exactly representable
                # (top32 / 2**32), and a float32 CDF would quantise nearby
                # uniforms onto the same state and lose the coupling.
                u = crn.uniform01(num_samples, [var_label], crn_key,
                                  device=self.device)          # (n, 1)
                cdf = probs.to(torch.float64).cumsum(dim=1).contiguous()
                # right=True gives the k with cdf[k-1] <= u < cdf[k]; the clamp
                # covers u >= cdf[-1], reachable when the row sums to 1 - eps.
                sampled_states = torch.searchsorted(
                    cdf, u, right=True).squeeze(1).clamp_(max=D - 1)

            samples[var_label] = sampled_states

            # Accumulate log10 proposal probability
            selected_ln_prob = ln_probs[torch.arange(num_samples, device=self.device), sampled_states]
            log_probs += selected_ln_prob / math.log(10)

        return samples, log_probs

    @property
    def variables(self) -> List[int]:
        """Variable labels in elimination order."""
        return [rec.elim_var_label for rec in self.levels]


def _condition_factor(factor: FastFactor, elim_var_label: int,
                      samples: Dict[int, torch.Tensor]) -> torch.Tensor:
    """
    Condition a factor on sampled values, keeping elim_var free.

    Args:
        factor: FastFactor whose labels include elim_var_label
        elim_var_label: Variable to keep free
        samples: Dict of already-sampled variable values

    Returns:
        Tensor of shape (num_samples, domain_size_of_elim_var) in log10 space
    """
    elim_dim = factor.labels.index(elim_var_label)
    D_elim = factor.tensor.shape[elim_dim]
    # Infer num_samples from any sampled variable
    N = next(iter(samples.values())).shape[0] if samples else 1

    result = torch.empty(N, D_elim, device=factor.tensor.device)

    for state in range(D_elim):
        indices = []
        for dim, label in enumerate(factor.labels):
            if label == elim_var_label:
                indices.append(state)
            else:
                indices.append(samples[label])
        result[:, state] = factor.tensor[tuple(indices)]

    return result


def _evaluate_factor(factor: FastFactor, samples: Dict[int, torch.Tensor]) -> torch.Tensor:
    """
    Evaluate a factor at sampled values (all variables sampled).

    Args:
        factor: FastFactor whose labels are all in samples
        samples: Dict of sampled variable values

    Returns:
        Tensor of shape (num_samples,) in log10 space
    """
    if not factor.labels:
        # Scalar factor
        N = next(iter(samples.values())).shape[0]
        return factor.tensor.squeeze().expand(N)

    indices = tuple(samples[label] for label in factor.labels)
    return factor.tensor[indices]


def build_proposal_tree(factors: List[FastFactor], message_scope: List[int],
                        domain_sizes: Dict[int, int], ecl: int, device: str,
                        reference_gm) -> ProposalTree:
    """
    Build a proposal tree for sampling message_scope variables.

    Runs WMB elimination on the given factors (which should already be
    over message_scope variables only), saving bucket state at each level
    for backward-traversal sampling.

    Args:
        factors: List of FastFactor objects over message_scope variables
        message_scope: List of variable labels to sample
        domain_sizes: Dict mapping variable label -> number of states
        ecl: WMB complexity limit
        device: Torch device string
        reference_gm: FastGM for variable resolution during temp GM construction

    Returns:
        ProposalTree ready for sampling
    """
    from nce.inference.graphical_model import FastGM

    if not factors or not message_scope:
        return ProposalTree([], device)

    # Filter out scalar factors and accumulate their constant
    scalar_constant = torch.tensor(0.0, device=device)
    non_scalar = []
    for f in factors:
        if not f.labels:
            scalar_constant += f.tensor.squeeze()
        else:
            non_scalar.append(f)

    if not non_scalar:
        return ProposalTree([], device)

    # Compute elimination order for message_scope variables
    elim_order_labels = wtminfill_order(non_scalar, variables_not_eliminated=[])

    # Build a temporary FastGM for WMB elimination
    config = dict(reference_gm.config)
    config['populate_bw_factors'] = False
    config['approximation_method'] = 'wmb'
    config['ecl'] = ecl
    config['iB'] = int(math.log2(ecl)) if ecl > 0 else 0
    # Don't re-merge inside the proposal-tree's temporary GM.
    # FastGM.__init__ dispatches FOUR independent merge passes; disabling two of
    # them left reduce-NN and non-subsumption free to merge here. A merged temp
    # cluster eliminates several variables at once while `levels` below records
    # exactly one elim var per level, so a level's saved factors end up
    # referencing variables that have not been sampled yet and the NR sampler
    # dies with `KeyError: <label>` in `_conditional_log_probs_batch`
    # (`no_replacement_sampler_v2.py:57`). Measured on `dbn/rbm_20` under
    # reduce-NN: 11/11 NN clusters. Same defect class as doc 10's defect 1 in
    # `_create_population_copy`, at a site doc 10 did not cover.
    config['use_join_tree_merge'] = False
    config['use_reduce_nn_merge'] = False
    config['use_non_subsumption_merge'] = False
    config['merge_degree'] = 0

    temp_gm = FastGM(
        factors=[copy.deepcopy(f) for f in non_scalar],
        elim_order=elim_order_labels,
        reference_fastgm=reference_gm,
        device=device,
        nn_config=config,
    )
    temp_gm.is_primary = False

    # Custom elimination loop: save bucket state, then process
    levels = []

    for var in list(temp_gm.elim_order):
        if var not in temp_gm.buckets:
            continue
        bucket = temp_gm.buckets[var]

        # Separate factors with/without elim var
        with_elim = [f for f in bucket.factors if var.label in f.labels]
        without_elim = [f for f in bucket.factors if var.label not in f.labels]

        # Save factors that involve the elim var (these define the conditional)
        saved = [FastFactor(f.tensor.clone(), list(f.labels)) for f in with_elim]

        # Add scalar constant to first saved factor (only once, at first level)
        if scalar_constant != 0.0 and saved:
            saved[0] = FastFactor(saved[0].tensor + scalar_constant, saved[0].labels)
            scalar_constant = torch.tensor(0.0, device=device)

        levels.append(BucketRecord(
            elim_var_label=var.label,
            domain_size=domain_sizes[var.label],
            factors=saved,
        ))

        # Compute message (WMB or exact)
        # Temporarily set factors to only those with elim var for processing
        bucket.factors = with_elim
        bucket_ec = bucket.get_ec()

        if not with_elim:
            messages = []
        elif bucket_ec <= ecl:
            messages = [bucket.compute_message_exact()]
        else:
            messages = bucket.compute_wmb_message(ecl=ecl)

        # Route messages + pass-through factors to next buckets
        all_outgoing = messages + without_elim
        for msg in all_outgoing:
            if msg.labels:
                next_bucket = temp_gm.find_next_bucket(msg.labels, var)
                if next_bucket:
                    next_bucket.receive_message(msg)

        del temp_gm.buckets[var]

    return ProposalTree(levels, device)


def proposal_scope_for_bucket(bucket, reference_gm):
    """The variable scope a proposal tree for `bucket` must be built over.

    Single-var buckets keep using the precomputed message_scopes cache (which is
    keyed per eliminated variable). Merged clusters must NOT: the cache holds the
    pre-merge, per-variable scopes. bucket.get_message_scope() is merge-correct —
    it unions the labels of the bucket's current factors (originals plus every
    message received so far, buckets being processed in elimination order) and
    discards ALL of the cluster's elim vars.

    Exposed separately from build_proposal_for_bucket so the scope can be
    asserted without building a tree.
    """
    elim_var_labels = {getattr(v, 'label', v) for v in bucket.elim_vars}
    if len(elim_var_labels) > 1:
        return bucket.get_message_scope()
    return list(reference_gm.message_scopes.get(bucket.label, []))


def build_proposal_for_bucket(bucket, reference_gm, ecl=None, temperature=1.0):
    """
    Convenience function: build a proposal tree for a single bucket.

    Combines the bucket's approximate upstream and downstream factors,
    eliminates the bucket's elim var, then builds a proposal tree over
    the message scope variables.

    Args:
        bucket: FastBucket with approximate_upstream_factors and
                approximate_downstream_factors populated
        reference_gm: The parent FastGM
        ecl: WMB complexity limit (defaults to config bw_ecl)
        temperature: Temperature T for the proposal. All factor log-values
                     are divided by T (equivalent to raising linear-space
                     values to power 1/T). T=1 leaves the distribution
                     unchanged; T>1 flattens toward uniform; T<1 sharpens.

    Returns:
        ProposalTree for sampling over the bucket's message scope
    """
    if ecl is None:
        ecl = reference_gm.config.get('bw_ecl', 0)

    up = bucket.approximate_upstream_factors or []
    down = bucket.approximate_downstream_factors or []
    all_factors = list(up) + list(down)

    if not all_factors:
        return ProposalTree([], reference_gm.device)

    # Apply temperature: scale each factor's log-values by 1/T so that
    # the implied joint becomes p(x)^{1/T}
    if temperature != 1.0:
        all_factors = [FastFactor(f.tensor / temperature, list(f.labels))
                       for f in all_factors]

    # Get message scope and domain sizes. For super-bucket clusters (multiple
    # elim_vars) the pre-merge message_scopes cache is keyed per variable and is
    # not the cluster's scope, so we take it from the bucket itself.
    #
    # bucket.get_message_scope() is already merge-correct: it unions the labels
    # of the bucket's factors (originals + every message received so far, since
    # buckets are processed in elimination order) and discards ALL of the
    # cluster's elim vars. Deriving the scope by unioning the backward chain
    # instead -- as this used to -- pulled in every variable of the whole
    # downstream chain: measured 14/16/17 variables against true separators of
    # 1/1/0 on grid10x10.f10 (see notebooks/_August-2026/claude_experiments/
    # 01-hybrid-memorization-table.md §0a F3).
    elim_var_labels = sorted({getattr(v, 'label', v) for v in bucket.elim_vars})
    msg_scope = proposal_scope_for_bucket(bucket, reference_gm)

    if not msg_scope:
        return ProposalTree([], reference_gm.device)

    domain_sizes = {v: reference_gm.matching_var(v).states for v in msg_scope}

    # Include all elim_vars in domain_sizes for the elimination step
    for lab in elim_var_labels:
        domain_sizes[lab] = reference_gm.matching_var(lab).states

    # Eliminate elim_var(s) from combined factors to get factors over message_scope
    message_scope_factors = reference_gm._wmb_eliminate_to_scope(
        all_factors, msg_scope, reference_gm.matching_var(elim_var_labels[0]))


    # Build proposal tree over message_scope
    return build_proposal_tree(
        factors=message_scope_factors,
        message_scope=msg_scope,
        domain_sizes=domain_sizes,
        ecl=ecl,
        device=reference_gm.device,
        reference_gm=reference_gm,
    )
