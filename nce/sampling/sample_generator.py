import torch
import math
import os
import numpy as np
from typing import List, Tuple
from nce.inference.graphical_model import FastGM
from nce.inference.bucket import FastBucket
from nce.inference.factor import FastFactor
from nce.inference.message_gradient_factors import get_wmb_message_gradient_factors
from nce.utils import gamma_trace
import copy

# Streaming sample-gen block knobs. Bigger blocks -> fewer kernel launches ->
# the launch-bound large-cluster path spends less time idle waiting on Python
# dispatch (see notebooks/.../reduce_nn_experiment/SPEEDUP_LARGER_BATCHES.md).
# Defaults reproduce the historical behavior exactly; override via env to trade
# GPU memory for speed (e.g. NCE_SAMPLE_BLOCK_LOG2=28 NCE_SAMPLE_ACHUNK=4096).
_SG_BLOCK_LOG2 = int(os.environ.get("NCE_SAMPLE_BLOCK_LOG2", "23"))   # large-path block budget UPPER cap = 2**this elems
_SG_ACHUNK     = int(os.environ.get("NCE_SAMPLE_ACHUNK", "512"))      # large-path assignment rows per chunk (cap)
_SG_SMALL_LOG2 = int(os.environ.get("NCE_SAMPLE_SMALL_LOG2", "22"))   # dense small-path block budget
_SG_MEM_FRAC   = float(os.environ.get("NCE_SAMPLE_MEM_FRAC", "0.25")) # frac of FREE gpu mem a streamed block may use


def _dyn_streaming_budget(dev, factors, cap_log2):
    """Number of rows (a_chunk * e_chunk) a streamed block may hold, sized from
    FREE gpu memory and the WIDEST NN factor's per-row footprint. The dominant
    transient is FactorNN._eval_elim_block's (rows x n_labels) int64 coord cube +
    the one-hot expansion. Floors at 2**12, capped at 2**cap_log2 (env). Falls
    back to a 2 GB budget off-cuda."""
    worst = 1
    for f in factors:
        if not getattr(f, 'is_nn', False):
            continue
        nlab = len(f.labels)
        try:
            onehot = int(sum(int(d) for d in f.domain_sizes))
        except Exception:
            onehot = 2 * nlab
        worst = max(worst, nlab * 8 + onehot * 4 + 4)   # coord cube + one_hot + accum
    try:
        free, _ = torch.cuda.mem_get_info(dev)
    except Exception:
        free = 2 * 1024 ** 3
    SLACK = 4                                            # net activations + transient copies
    budget = int(_SG_MEM_FRAC * free / max(1, worst * SLACK))
    return max(2 ** 12, min(budget, 2 ** cap_log2))

class SampleGenerator:
    def __init__(self, gm: FastGM, bucket: FastBucket, random_seed=None):
        self.config = gm.config
        self.gm = gm
        self.iB = gm.iB
        self.bucket = bucket
        self.random_seed = random_seed if random_seed is not None else 0
        self.factors = bucket.factors
        # Raw config entry: may be an int OR the per-cluster formula string
        # "nbe,<epsilon>[,<n_min>]".  Currently unread -- do not use it as a count.
        # For a resolved per-bucket integer use bucket.get_num_samples().
        self.num_samples = self.config.get('num_samples')
        self.sampling_scheme = self.config['sampling_scheme']
        self.message_scope, self.domain_sizes = self.get_message_scope_and_dims()
        self.message_size = np.prod([float(d.item()) for d in self.domain_sizes])

        # Backward factors - set externally via dataloader.bw_factors after get_backward_message() is called
        # Do NOT transform them here - they are used as-is
        self.backward_factors = None

        for factor in self.factors:
            factor.order_indices()
        self.elim_vars = sorted(self.bucket.elim_vars, key=lambda v: v.label)
        self.elim_domain_sizes = [v.states for v in self.elim_vars]

        # Sampling counters for deterministic seed generation
        # Each call to sample_assignments increments this to get unique but reproducible samples
        self._training_sample_counter = 0
        self._validation_sample_counter = 0

    def reset_sample_counters(self):
        """Reset sampling counters to reproduce the same samples.

        Call this to reset the internal counters so that subsequent calls
        to sample_assignments will produce the same samples as when the
        SampleGenerator was first created (with the same random_seed).
        """
        self._training_sample_counter = 0
        self._validation_sample_counter = 0

    def _compute_seed(self, is_validation: bool = False) -> int:
        """Compute deterministic seed based on bucket_id + global_seed + counter.

        The seed formula is:
            seed = bucket_label + global_seed * 10000 + counter * 100 + validation_offset

        This ensures:
        - Different buckets get different seeds
        - Different global seeds give different results
        - Multiple sampling calls get different but reproducible samples
        - Training and validation samples are different

        Args:
            is_validation: If True, add offset to separate from training samples

        Returns:
            Integer seed value
        """
        bucket_id = self.bucket.label if isinstance(self.bucket.label, int) else hash(str(self.bucket.label)) % 10000
        validation_offset = 50000000 if is_validation else 0

        if is_validation:
            counter = self._validation_sample_counter
            self._validation_sample_counter += 1
        else:
            counter = self._training_sample_counter
            self._training_sample_counter += 1

        seed = bucket_id + self.random_seed * 10000 + counter * 100 + validation_offset
        return seed

    def _set_seed(self, seed: int):
        """Set random seed for both CPU and GPU.

        Args:
            seed: Integer seed value
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        np.random.seed(seed % (2**31))  # numpy requires seed < 2^31

    def sample_assignments(self, num_samples: int = -1, sampling_scheme=None, is_validation: bool = False) -> torch.Tensor:
        """Sample assignments from the message scope.

        Args:
            num_samples: Number of samples to generate
            sampling_scheme: 'uniform' or 'all'. Defaults to config setting.
            is_validation: If True, use validation seed (different from training)

        Returns:
            Tensor of shape (num_samples, num_vars) with sampled assignments
        """
        if sampling_scheme is None:
            sampling_scheme = self.sampling_scheme
        # gamma_trace.phase() is a no-op (and performs no cuda sync) unless a
        # cluster trace is in flight; the seeding below is untouched by it.
        with gamma_trace.phase('assign'):
            if sampling_scheme == 'uniform':
                # Set deterministic seed before sampling
                seed = self._compute_seed(is_validation=is_validation)
                self._set_seed(seed)
                return self.sample_uniform(num_samples)
            elif sampling_scheme == 'all':
                return self.sample_all()
            else:
                raise ValueError(f"Unknown sampling scheme: {sampling_scheme}. Use 'uniform' or 'all'.")

    def sample_all(self) -> torch.Tensor:
        # Generate all possible assignments
        assignments = torch.cartesian_prod(*[torch.arange(size) for size in self.domain_sizes])
        if len(assignments.shape) == 1:
            assignments = assignments.view(-1,1)
        return assignments
    
    @staticmethod
    def _unravel_index(indices, shape):
        coord = []
        for dim in reversed(shape):
            coord.append(indices % dim)
            indices = indices // dim
        coord = torch.stack(coord[::-1], dim=-1)
        return coord
    
    def sample_uniform(self, num_samples) -> torch.Tensor:
        # dtype long used for compatibility with indexing
        samples = []
        for domain_size in self.domain_sizes:
            column_samples = torch.randint(
                low=0,
                high=domain_size.item(),
                size=(num_samples,),
                dtype=torch.long,
                device=self.gm.device,   # keep assignments on-device (sample gen is otherwise CPU-bound)
            )
            samples.append(column_samples)
        return torch.stack(samples, dim=1)
    
    def get_message_scope_and_dims(self) -> Tuple[List[int], torch.Tensor]:
        scope = set()
        for factor in self.bucket.factors:
            scope = scope.union(factor.labels)
        # Discard ALL elim_vars (super-bucket may eliminate multiple variables).
        for ev in self.bucket.elim_vars:
            scope.discard(getattr(ev, 'label', ev))
        scope.discard(self.bucket.label)
        scope = sorted(list(scope))
        if len(scope) == 0:
            return scope, torch.tensor([], dtype=torch.long)
        # matching_var returns a Var object (not an index), so use .states directly
        return scope, torch.tensor([self.gm.matching_var(v).states for v in scope])
    
    def compute_message_values(self, assignments: torch.Tensor) -> torch.Tensor:
        factors = self.factors
        if self.gm.config.get('fdb', False):
            return self.bucket.compute_message_exact().tensor.flatten()
        with gamma_trace.phase('eval'):
            return self.sample_tensor_product_elimination(self.factors, assignments)
    
    def compute_backward_values(self, assignments: torch.Tensor, backward_factors=None) -> torch.Tensor:
        """Compute backward message values at sampled assignments.

        Evaluates the product of backward factors at each assignment point.
        The backward factors should already be marginalized to only contain
        variables in the message scope (done by get_backward_message).

        Args:
            assignments: Tensor of shape (num_samples, num_vars) with sampled assignments
            backward_factors: List of FastFactors representing backward message.
                             If None, uses self.backward_factors.

        Returns:
            Tensor of shape (num_samples,) with backward message values in log10 space
        """
        if backward_factors is None:
            factors = self.backward_factors
        else:
            factors = backward_factors
        if factors is None:
            return None
        # Use sample_tensor_product - no marginalization needed since bw_factors
        # from get_backward_message already only contain message_scope variables
        with gamma_trace.phase('bw'):
            return self.sample_tensor_product(factors=factors, assignments=assignments)

    def sample_tensor_product_elimination(self, factors, assignments) -> torch.Tensor:
        for factor in factors:
            factor.order_indices()
        elim_doms = [int(d) for d in self.elim_domain_sizes]
        elim_prod = 1
        for d in elim_doms:
            elim_prod *= d
        n = len(assignments)
        ln10 = math.log(10)
        dev = self.gm.device

        # Resolve FactorNN-with-bw_inv to exact once.
        # NOTE for gamma tracing: fx is index-aligned with `factors` (==
        # bucket.factors), so per-factor timings key straight onto the
        # structure record.  A resolved bw_inv factor is counted as an NN by
        # bucket.factors but evaluated as a table here, so the count of such
        # resolutions is recorded and must be checked before attributing cost.
        fx = []
        _n_resolved = 0
        for f in factors:
            if hasattr(f, 'is_nn') and f.is_nn and hasattr(f, 'bw_inv') and f.bw_inv:
                f = f.to_exact()
                _n_resolved += 1
            fx.append(f)

        # Per-factor timing hook.  `_pf` is None unless a gamma trace is in
        # flight AND gamma_trace_per_factor is set; the loops below are then
        # byte-identical in their tensor operations and accumulation order to
        # the untraced ones, differing only by a perf_counter read and a
        # cuda synchronize between factors.
        _pf = gamma_trace.per_factor_active()
        _rec = gamma_trace.active()
        if _rec is not None:
            _rec.note(sg_n_nn_resolved_to_exact=_n_resolved)

        # --- small clusters: build the full (chunk x 2^#elim) and reduce (fast) ---
        if elim_prod <= 2 ** 18:
            unsummed_shape = tuple(elim_doms)
            a_chunk = max(1, (2 ** _SG_SMALL_LOG2) // max(1, elim_prod))
            if _rec is not None:
                _rec.note(sg_path='small', sg_a_chunk=a_chunk, sg_e_chunk=elim_prod,
                          sg_elim_prod=elim_prod, sg_n_assignments=n,
                          sg_n_chunks=(n + a_chunk - 1) // max(1, a_chunk))
            # Everything FastFactor._get_slices does except projecting/indexing the
            # chunk is invariant across chunks, so build it once per factor instead
            # of once per (factor, chunk). Bit-identical -- the same ops on the same
            # values, merely not recomputed. A cluster here can hold hundreds of
            # small table factors and hundreds of chunks, so this is the difference
            # between O(F) and O(F x chunks) Python bookkeeping. FactorNN returns
            # None from _slice_plan and keeps the unprepared path.
            plans = [f._slice_plan(self.elim_vars, self.elim_domain_sizes,
                                   self.message_scope) for f in fx]
            outs = []
            for start in range(0, n, a_chunk):
                ca = assignments[start:start + a_chunk]
                uv = torch.zeros((len(ca),) + unsummed_shape, device=dev, requires_grad=False)
                if _pf is None:
                    for f, plan in zip(fx, plans):
                        if plan is None:
                            uv += f._get_slices(assignments=ca, elim_vars=self.elim_vars,
                                                elim_domain_sizes=self.elim_domain_sizes,
                                                message_scope=self.message_scope)
                        else:
                            uv += f._get_slices_prepared(plan, ca)
                else:
                    for _fi, (f, plan) in enumerate(zip(fx, plans)):
                        _t0 = _pf.tic()
                        if plan is None:
                            uv += f._get_slices(assignments=ca, elim_vars=self.elim_vars,
                                                elim_domain_sizes=self.elim_domain_sizes,
                                                message_scope=self.message_scope)
                        else:
                            uv += f._get_slices_prepared(plan, ca)
                        _pf.toc_factor(_fi, _t0)
                outs.append(torch.logsumexp(uv * ln10, dim=tuple(range(1, uv.dim()))) / ln10)
            return torch.cat(outs, dim=0)

        # --- large clusters (2^#elim too big to materialize): stream over elim
        # blocks with an online log-sum-exp, so peak memory ~ a_chunk * e_chunk.
        elim_labels = [v.label for v in self.elim_vars]
        strides = [1] * len(elim_doms)
        acc = 1
        for i in range(len(elim_doms) - 1, -1, -1):
            strides[i] = acc
            acc *= elim_doms[i]
        strides_t = torch.tensor(strides, device=dev)
        doms_t = torch.tensor(elim_doms, device=dev)

        # Size the streamed block from FREE gpu memory (and the widest factor),
        # so wide / multi-valued NN factors never blow up the per-eval coord cube.
        budget = _dyn_streaming_budget(dev, fx, _SG_BLOCK_LOG2)
        a_chunk = max(1, min(_SG_ACHUNK, budget, n))
        e_chunk = max(1, min(budget // a_chunk, elim_prod))
        if _rec is not None:
            _rec.note(sg_path='large', sg_a_chunk=a_chunk, sg_e_chunk=e_chunk,
                      sg_elim_prod=elim_prod, sg_n_assignments=n,
                      sg_n_chunks=((n + a_chunk - 1) // max(1, a_chunk))
                                  * ((elim_prod + e_chunk - 1) // max(1, e_chunk)))
        outs = []
        for start in range(0, n, a_chunk):
            ca = assignments[start:start + a_chunk]
            A = len(ca)
            acc_lse = torch.full((A,), float('-inf'), device=dev)
            for e0 in range(0, elim_prod, e_chunk):
                e1 = min(elim_prod, e0 + e_chunk)
                flat = torch.arange(e0, e1, device=dev)
                coords = (flat.unsqueeze(1) // strides_t) % doms_t   # (B, n_elim_vars)
                block = torch.zeros((A, e1 - e0), device=dev)
                if _pf is None:
                    for f in fx:
                        block += f._eval_elim_block(ca, coords, self.elim_vars,
                                                    elim_labels, self.message_scope)
                else:
                    for _fi, f in enumerate(fx):
                        _t0 = _pf.tic()
                        block += f._eval_elim_block(ca, coords, self.elim_vars,
                                                    elim_labels, self.message_scope)
                        _pf.toc_factor(_fi, _t0)
                blk = torch.logsumexp(block * ln10, dim=1) / ln10     # (A,)
                acc_lse = torch.logaddexp(acc_lse * ln10, blk * ln10) / ln10
            outs.append(acc_lse)
        return torch.cat(outs, dim=0)

    def sample_tensor_product(self, factors, assignments) -> torch.Tensor:
        # Check for edge cases
        if factors is None or len(factors) == 0:
            return torch.zeros(len(assignments), device=self.gm.device)
        if len(factors) == 1 and factors[0].labels == []:
            return torch.zeros(len(assignments), device=self.gm.device)

        for factor in factors:
            factor.order_indices()
        output = torch.zeros((len(assignments),1), device=self.gm.device, requires_grad=False)
        for fast_factor in factors:
            # If this is a FactorNN with bw_inv, convert to exact first to apply inverse transformation
            if hasattr(fast_factor, 'is_nn') and fast_factor.is_nn and hasattr(fast_factor, 'bw_inv') and fast_factor.bw_inv:
                fast_factor = fast_factor.to_exact()
            output += fast_factor._get_values(assignments=assignments, message_scope=self.message_scope)
        assert not output.requires_grad
        return (output).squeeze(1)
