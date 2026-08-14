import itertools
import torch
import math


# Monotonic source of stable factor ids. See FastFactor.__hash__.
_FACTOR_ID_COUNTER = itertools.count()


class _SlicePlan:
    """Chunk-invariant precomputation for `FastFactor._get_slices` (see `_slice_plan`)."""
    __slots__ = ('scalar', 'base', 'flat_src', 'flat', 'proj_indices', 'proj_col',
                 'proj_idx_t', 'unexpanded_tail', 'expanded_tail', 'shapes', 'dbg')

    def __init__(self, scalar, base, flat, proj_indices,
                 unexpanded_tail, expanded_tail, dbg):
        self.scalar = scalar
        self.base = base
        # `flat_src` is the permuted tensor (a view, free); the flattened copy is
        # built on first use and only by factors that need it.
        self.flat_src = flat
        self.flat = None
        self.proj_indices = proj_indices
        # Single-column projection is the overwhelmingly common case (a small table
        # factor whose scope is one message variable plus elim vars). Indexing with
        # a Python list would rebuild an index tensor and copy it host->device on
        # every chunk; a plain column view costs nothing.
        self.proj_col = proj_indices[0] if len(proj_indices) == 1 else None
        self.proj_idx_t = (torch.as_tensor(proj_indices, dtype=torch.long,
                                           device=base.device)
                           if len(proj_indices) > 1 else None)
        self.unexpanded_tail = unexpanded_tail
        self.expanded_tail = expanded_tail
        # n -> (unexpanded_shape, expanded_shape); chunk sizes repeat, so building
        # these 17-element tuples once per distinct n instead of once per call.
        self.shapes = {}
        self.dbg = dbg

    def shapes_for(self, n):
        s = self.shapes.get(n)
        if s is None:
            s = ((n,) + self.unexpanded_tail, (n,) + self.expanded_tail)
            self.shapes[n] = s
        return s


class FastFactor:
    """A tensor-backed factor in log10 space.

    DETERMINISM -- WHY THIS CLASS HAS A `__hash__` BUT NOT AN `__eq__`
    ------------------------------------------------------------------
    Until 2026-08-12 `FastGM._create_buckets_from_factors` did `set(factors)`.
    With the default `object.__hash__` (derived from `id()`), the iteration
    order of that set followed memory addresses, so bucket factor order -- and
    hence the association order of a log-space product -- differed between
    processes and even between two builds in one process. See
    notebooks/_August-2026/claude_experiments/21-determinism.md.

    The fix at that site was to stop using a set. This adds a second, structural
    line of defence: every factor gets a stable creation-order integer id and
    hashes to it, so ANY set or dict keyed on factors iterates in an order that
    is a function of construction order rather than of the allocator.

    `__eq__` is deliberately left as identity. A value-based `__eq__` was
    considered and rejected on two grounds:

      * FastFactor is MUTABLE. `self.tensor` is reassigned in place by
        `order_indices`, `to`, `to_exact`, doping, and normalisation. An object
        whose hash tracks its value and which is mutated while sitting in a set
        is a well-known silent-corruption hazard: the object lands in the wrong
        bucket of the hash table and can no longer be found.
      * A value-based `__eq__` would change what `set(factors)` MEANS, from
        "deduplicate by identity" to "deduplicate by value". Duplicate factors
        are legitimate here (e.g. two copies of the same pairwise potential, or
        a doped factor equal to its neighbour), and silently dropping one is a
        correctness bug, not an ordering wobble.

    `copy.deepcopy` of a factor copies `_factor_id`, so a copy shares its
    original's id. That is harmless -- hash collisions are legal and equality is
    still identity -- but it means the id identifies a construction event, not a
    factor value.
    """

    def __init__(self, tensor, labels):
        self._factor_id = next(_FACTOR_ID_COUNTER)
        self.tensor = tensor
        if self.tensor is None:
            self.device = None
        else:
            self.device = self.tensor.device
        self.labels = labels
        self.is_nn = False
        if tensor is not None:
            self.shape = self.tensor.shape
        else:
            self.shape = None

    def __hash__(self):
        # Stable across processes: small ints hash to themselves and are not
        # affected by PYTHONHASHSEED. `_factor_id` is looked up defensively
        # because a few code paths build factors via __new__/deepcopy tricks.
        return hash(getattr(self, '_factor_id', 0))

    def __repr__(self):
        return f"FastFactor(tensor={self.tensor}, labels={self.labels})"

    def __mul__(self, other):
        if not isinstance(other, FastFactor):
            # If other is a scalar, just multiply the tensor and return
            return FastFactor(self.tensor + other, self.labels)

        # If both factors are scalars (no labels), just multiply the tensors
        if not self.labels and not other.labels:
            return FastFactor(self.tensor + other.tensor, [])

        # If one factor is a scalar and the other isn't, broadcast the scalar
        if not self.labels:
            return FastFactor(other.tensor + self.tensor.item(), other.labels)
        if not other.labels:
            return FastFactor(self.tensor + other.tensor.item(), self.labels)

        # Original multiplication logic for non-scalar factors
        common_labels = [label for label in self.labels if label in other.labels]
        self_unique = [label for label in self.labels if label not in other.labels]
        other_unique = [label for label in other.labels if label not in self.labels]

        self_perm = [self.labels.index(label) for label in common_labels + self_unique]
        other_perm = [other.labels.index(label) for label in common_labels + other_unique]

        self_tensor = self.tensor.permute(self_perm)
        other_tensor = other.tensor.permute(other_perm)

        self_shape = list(self_tensor.shape) + [1] * len(other_unique)
        other_shape = list(other_tensor.shape[:len(common_labels)]) + [1] * len(self_unique) + list(other_tensor.shape[len(common_labels):])

        result_tensor = self_tensor.reshape(self_shape) + other_tensor.reshape(other_shape)
        new_labels = common_labels + self_unique + other_unique

        return FastFactor(result_tensor, new_labels)
    
    def __matmul__(self, other): # does true multiplication of logspace elements
        if not isinstance(other, FastFactor):
            # If other is a scalar, just multiply the tensor and return
            return FastFactor(self.tensor + other, self.labels)

        # If both factors are scalars (no labels), just multiply the tensors
        if not self.labels and not other.labels:
            return FastFactor(self.tensor + other.tensor, [])

        # If one factor is a scalar and the other isn't, broadcast the scalar
        if not self.labels:
            return FastFactor(other.tensor + self.tensor.item(), other.labels)
        if not other.labels:
            return FastFactor(self.tensor + other.tensor.item(), self.labels)

        # Original multiplication logic for non-scalar factors
        common_labels = [label for label in self.labels if label in other.labels]
        self_unique = [label for label in self.labels if label not in other.labels]
        other_unique = [label for label in other.labels if label not in self.labels]

        self_perm = [self.labels.index(label) for label in common_labels + self_unique]
        other_perm = [other.labels.index(label) for label in common_labels + other_unique]

        self_tensor = self.tensor.permute(self_perm)
        other_tensor = other.tensor.permute(other_perm)

        self_shape = list(self_tensor.shape) + [1] * len(other_unique)
        other_shape = list(other_tensor.shape[:len(common_labels)]) + [1] * len(self_unique) + list(other_tensor.shape[len(common_labels):])

        result_tensor = self_tensor.view(self_shape) * other_tensor.view(other_shape)
        new_labels = common_labels + self_unique + other_unique

        return FastFactor(result_tensor, new_labels)

    def is_equal(self, other, rtol=1e-3, atol=1e-5):
        """
        Check if this FastFactor is approximately equal to another FastFactor.
        
        Args:
        other (FastFactor): The other FastFactor to compare with.
        rtol (float): Relative tolerance for numerical comparison.
        atol (float): Absolute tolerance for numerical comparison.
        
        Returns:
        bool: True if the factors are approximately equal, False otherwise.
        """
        # Check if the factors have the same variables (ignoring order)
        if set(self.labels) != set(other.labels):
            return False

        # Get the permutation to align the other factor's labels with this factor's labels
        perm = [other.labels.index(label) for label in self.labels]

        # Permute the other factor's tensor to match this factor's label order
        other_tensor_permuted = other.tensor.permute(*perm)

        # Reshape both tensors to 1D for easier comparison
        self_flat = self.tensor.reshape(-1)
        other_flat = other_tensor_permuted.reshape(-1)

        # Check if the tensors are approximately equal
        return torch.allclose(self_flat, other_flat, rtol=rtol, atol=atol)
    
    def order_indices(self):
        """
        Orders the labels from least to greatest and permutes the tensor accordingly.
        Assumes labels are integers.
        """
        
        # for identity factor
        if self.labels == []:
            return
        
        # Convert labels to integers and get the sorting order
        int_labels = [int(label) for label in self.labels]
        sorted_indices = torch.argsort(torch.tensor(int_labels))
        
        # Create the new order of labels
        new_labels = [self.labels[i] for i in sorted_indices]
        
        # Permute the tensor
        new_tensor = self.tensor.permute(tuple(sorted_indices.tolist()))
        
        # Update the FastFactor
        self.labels = new_labels
        self.tensor = new_tensor

        return
    
    def to_logspace(self, normalizing_constant):
        # make all negative values in tensor 0
        self.tensor[self.tensor < 0] = 0
        self.tensor = torch.log10(self.tensor + 1e-10) + normalizing_constant
            
    def eliminate(self, elim_labels, elimination_scheme = 'sum'):
        if elim_labels == 'all':
            elim_indices = list(range(len(self.labels)))
            new_labels = []
        else:
            elim_indices = [self.labels.index(label) for label in elim_labels]
            new_labels = [label for label in self.labels if label not in elim_labels]
        
        # Convert from log10 to natural log, perform logsumexp, then convert back to log10
        if elimination_scheme == 'sum':
            result_tensor = torch.logsumexp(self.tensor * math.log(10), dim=elim_indices) / math.log(10)
        elif elimination_scheme == 'max':
            result_tensor = torch.max(self.tensor, dim=elim_indices).values
        
        if type(result_tensor) == float:
            result_tensor = torch.Tensor([result_tensor])
        
        # If we've eliminated all variables, we need to ensure the result is a scalar
        if not new_labels:
            result_tensor = result_tensor.view(1)
        
        out = FastFactor(result_tensor, new_labels)
        out.order_indices()
        return out

    def sum_all_entries(self):
        return self.eliminate('all').tensor.item()

    def sum(self):
        return self.sum_all_entries()

    def to(self, device):
        self.tensor = self.tensor.to(device)
        return self

    def klargest(self, k):
        # Get top-k values and their indices from the flattened tensor
        topk_values, topk_flat_indices = torch.topk(self.tensor.flatten(), k)

        # Convert flat indices back to original shape
        topk_indices = torch.unravel_index(topk_flat_indices, self.tensor.shape)

        # Combine the indices into tuples (e.g., [(x1, y1), (x2, y2), ...])
        topk_indices = list(zip(*(dim.tolist() for dim in topk_indices)))
        topk_values = topk_values.tolist()  # Convert values to a plain list
        
        return topk_indices, topk_values
    
    def get_variance(self, ignore_inconsistencies = True):
        # Calculate the mean of the tensor
        mean = torch.mean(self.tensor)
        
        # Calculate the variance
        finite_tensor = self.tensor.reshape(-1)[torch.isfinite(self.tensor.reshape(-1))]
        variance = torch.var(finite_tensor, unbiased=False)
        
        return variance.item()
    
    def _slice_plan(self, elim_vars, elim_domain_sizes, message_scope):
        """Precompute the part of `_get_slices` that does not depend on the
        assignment chunk.

        `_get_slices` is called once per (factor, assignment chunk). Everything it
        does except projecting/indexing the chunk itself -- the label bookkeeping,
        the permutation, the flattened view of the factor tensor and the two output
        shapes' tails -- depends only on (self, elim_vars, message_scope), which are
        fixed for a whole cluster. Callers that loop over chunks should build the
        plan once and call `_get_slices_prepared`; that is bit-identical to calling
        `_get_slices` per chunk (the tensor ops performed are the same ops on the
        same values) and removes the repeated Python work.

        Returns an opaque plan object. `FactorNN` overrides this to return None,
        meaning "no prepared path -- use `_get_slices`".
        """
        tensor = self.tensor
        tensor_labels = self.labels

        expanded_tail = tuple([v.states for v in elim_vars])

        # Handle 0-dim tensors (scalar factors with empty labels)
        if tensor.dim() == 0 or len(tensor_labels) == 0:
            return _SlicePlan(scalar=True, base=tensor, flat=None,
                              proj_indices=(), unexpanded_tail=expanded_tail,
                              expanded_tail=expanded_tail, dbg=None)

        # indices of the assignment in the tensor
        tensor_assignment_indices = [i for i, idx in enumerate(tensor_labels) if idx not in elim_vars]
        # indices of eliminated variables in the tensor
        tensor_elim_indices = [i for i, idx in enumerate(tensor_labels) if idx in elim_vars]
        # Number of elim vars actually present in this factor (may be < len(elim_vars)
        # for super-buckets where a factor only touches a subset of cluster's elim_vars).
        n_elim_in_tensor = len(tensor_elim_indices)

        # put elimination indices at end of tensor
        permutation = (*tensor_assignment_indices, *tensor_elim_indices)
        # permute tensor
        tensor = tensor.permute(permutation)
        # reordered labels
        tensor_labels = [tensor_labels[i] for i in permutation]

        # get assignments from permuted tensor
        # assertation necessary for indexing
        assert(all(tensor_labels[i] < tensor_labels[i+1] for i in range(len(tensor_labels)-n_elim_in_tensor-1)))
        permuted_assignment_indices = [i for i, idx in enumerate(message_scope) if idx in tensor_labels]

        # stretch out the elim-vars-present-in-this-factor dimensions to 1d
        n_assign_dims = len(tensor.shape) - n_elim_in_tensor
        if n_elim_in_tensor == 0:
            view = tuple(int(dim) for dim in tensor.shape) + (1,)
        else:
            tail = 1
            for dim in tensor.shape[n_assign_dims:]:
                tail *= int(dim)
            view = tuple(int(dim) for dim in tensor.shape[:n_assign_dims]) + (tail,)

        # .view() requires contiguous strides which may not hold after permute()
        # above (especially for multi-elim_var super buckets). reshape() falls
        # back to a copy when needed -- and that copy is exactly what we hoist.
        # `flat` is only reached by factors whose whole scope is elim vars, so it
        # is built on demand: eagerly materialising it would hold a second copy of
        # every factor tensor alive for the whole chunk loop for nothing.
        base = tensor.reshape(view)

        # tail of the shape slices are reshaped to, e.g. (1,2,2,1) if the 2nd and
        # 3rd elim vars are in this tensor
        unexpanded_tail = tuple([v.states if v.label in tensor_labels else 1 for v in elim_vars])

        return _SlicePlan(scalar=False, base=base, flat=tensor,
                          proj_indices=permuted_assignment_indices,
                          unexpanded_tail=unexpanded_tail,
                          expanded_tail=expanded_tail,
                          dbg=(tuple(tensor.shape), list(tensor_labels), view,
                               n_elim_in_tensor,
                               [v.label for v in elim_vars], list(message_scope)))

    def _get_slices_prepared(self, plan, assignments):
        """`_get_slices` with the chunk-invariant work already done (see `_slice_plan`)."""
        n = assignments.shape[0]
        unexpanded_slice_shape, expanded_slice_shape = plan.shapes_for(n)
        if plan.scalar:
            return plan.base.expand(expanded_slice_shape)

        proj_indices = plan.proj_indices
        try:
            if n != 0 and len(proj_indices) != 0:
                if plan.proj_col is not None:
                    # base[assignments[:, c]] -- identical gather, no index tensor
                    # to materialise and no transpose/unbind round trip.
                    slices = plan.base.index_select(
                        0, assignments.select(1, plan.proj_col))
                else:
                    projected_assignments = assignments.index_select(1, plan.proj_idx_t)
                    slices = plan.base[tuple(projected_assignments.t())]
            else:
                # Factor has no assignment-projection dims (all its labels
                # are elim_vars). Flatten the elim dims and broadcast to
                # all assignments.
                flat = plan.flat
                if flat is None:
                    flat = plan.flat = plan.flat_src.reshape(-1)
                slices = flat.unsqueeze(0).expand(n, flat.numel())
        except Exception:
            tshape, tlabels, view, n_elim_in_tensor, elim_labels, message_scope = plan.dbg
            print(f"[_get_slices INDEX error] orig tensor.shape={self.tensor.shape} labels={self.labels}")
            print(f"  permuted tensor.shape={tshape}, tensor_labels={tlabels}")
            print(f"  view={view}, projected_assignments.shape={(n, len(proj_indices))}")
            print(f"  elim_var_labels={elim_labels}, n_elim_in_tensor={n_elim_in_tensor}")
            print(f"  message_scope={message_scope}, permuted_assignment_indices={proj_indices}")
            raise

        try:
            reshaped_slices = slices.reshape(unexpanded_slice_shape)
        except Exception:
            tshape, tlabels, view, n_elim_in_tensor, elim_labels, message_scope = plan.dbg
            print(f"[_get_slices reshape error] slices.shape={slices.shape} → unexpanded={unexpanded_slice_shape}")
            print(f"  tensor.shape={self.tensor.shape}, tensor_labels={self.labels}, elim_var_labels={elim_labels}")
            raise
        return reshaped_slices.expand(expanded_slice_shape)

    def _get_slices(self, assignments, elim_vars, elim_domain_sizes, message_scope):
        return self._get_slices_prepared(
            self._slice_plan(elim_vars, elim_domain_sizes, message_scope), assignments)

    def _eval_elim_block(self, assignments, elim_coords, elim_vars, elim_var_labels, message_scope):
        """Evaluate this factor at every (message assignment x elim assignment)
        pair for an explicit BLOCK of elim assignments.

        assignments : (A, len(message_scope));  elim_coords : (B, n_elim_vars)
        returns     : (A, B).

        Same math as _get_slices but returns just the requested elim block, so the
        caller can stream over the 2^#elim grid without materializing it. This
        factor only depends on the elim vars it actually contains, so we build the
        small assignment-indexed slice and flat-index it by the block.
        """
        tensor = self.tensor
        tensor_labels = self.labels
        A = len(assignments)
        B = len(elim_coords)

        if tensor.dim() == 0 or len(tensor_labels) == 0:
            return tensor.to(assignments.device).expand(A, B)

        tensor_assignment_indices = [i for i, idx in enumerate(tensor_labels) if idx not in elim_vars]
        tensor_elim_indices = [i for i, idx in enumerate(tensor_labels) if idx in elim_vars]
        n_elim_in_tensor = len(tensor_elim_indices)
        permutation = (*tensor_assignment_indices, *tensor_elim_indices)
        tensor_p = tensor.permute(permutation)
        tensor_labels_p = [tensor_labels[i] for i in permutation]
        permuted_assignment_indices = [i for i, idx in enumerate(message_scope) if idx in tensor_labels_p]
        projected = assignments[:, permuted_assignment_indices]

        n_assign_dims = len(tensor_p.shape) - n_elim_in_tensor
        if n_elim_in_tensor == 0:
            view = tuple(int(d) for d in tensor_p.shape) + (1,)
        else:
            view = tuple(int(d) for d in tensor_p.shape[:n_assign_dims]) + \
                   (int(torch.prod(torch.tensor(tensor_p.shape[n_assign_dims:]))),)
        if projected.numel() != 0:
            slices = tensor_p.reshape(view)[tuple(projected.t())]
        else:
            flat = tensor_p.reshape(-1)
            slices = flat.unsqueeze(0).expand(A, flat.numel())

        # (A, s1..sk) with si = states if elim var i is in this factor else 1
        present = tuple(v.states if v.label in tensor_labels_p else 1 for v in elim_vars)
        reshaped = slices.reshape((A,) + present)
        flat_rs = reshaped.reshape(A, -1)                        # (A, prod(present))

        coords = elim_coords.to(flat_rs.device).clone()
        for i, si in enumerate(present):
            if si == 1:
                coords[:, i] = 0                                  # absent dim -> broadcast index 0
        strides = [1] * len(present)
        acc = 1
        for i in range(len(present) - 1, -1, -1):
            strides[i] = acc
            acc *= int(present[i])
        strides_t = torch.tensor(strides, device=coords.device, dtype=coords.dtype)
        flat_idx = (coords * strides_t).sum(dim=1)               # (B,)
        return flat_rs[:, flat_idx]                              # (A, B)

    def _get_values(self, assignments, message_scope):
        tensor = self.tensor
        tensor_labels = self.labels

        # Ensure tensor_labels are ordered for correct indexing
        if tensor_labels != sorted(tensor_labels):
            # Need to reorder tensor to match sorted labels
            sorted_indices = [tensor_labels.index(label) for label in sorted(tensor_labels)]
            tensor = tensor.permute(sorted_indices)
            tensor_labels = sorted(tensor_labels)

        # Find labels in tensor that ARE in message_scope (for indexing)
        overlap_labels = [label for label in tensor_labels if label in message_scope]
        # Find labels in tensor that are NOT in message_scope (for marginalizing)
        marginalize_labels = [label for label in tensor_labels if label not in message_scope]

        # Get indices in message_scope that correspond to overlap labels
        assignment_indices = [i for i, idx in enumerate(message_scope) if idx in overlap_labels]
        projected_assignments = assignments[:, assignment_indices]

        try:
            if len(overlap_labels) == 0:
                # Factor has no variables in message_scope - marginalize over all dimensions
                # and return the same constant for all assignments
                marginal_value = torch.logsumexp(tensor.flatten(), dim=0)
                return marginal_value.expand(len(assignments), 1).reshape(-1, 1)

            if len(marginalize_labels) > 0:
                # Partial overlap: need to marginalize over non-message-scope dimensions
                # Permute tensor to put marginalize dimensions at the end
                overlap_indices = [tensor_labels.index(label) for label in overlap_labels]
                marginalize_indices = [tensor_labels.index(label) for label in marginalize_labels]
                permutation = overlap_indices + marginalize_indices
                tensor = tensor.permute(permutation)

                # Index into the overlap dimensions, then logsumexp over marginalize dimensions
                # tensor now has shape: [overlap_dim_1, ..., overlap_dim_k, marg_dim_1, ..., marg_dim_m]
                n_overlap = len(overlap_labels)
                n_marginalize = len(marginalize_labels)

                # Validate indices are within bounds before indexing
                overlap_shape = tensor.shape[:n_overlap]
                for dim_idx, (label, tensor_dim) in enumerate(zip(overlap_labels, overlap_shape)):
                    assignment_col = projected_assignments[:, dim_idx]
                    max_val = assignment_col.max().item()
                    if max_val >= tensor_dim:
                        raise IndexError(f"Assignment index {max_val} out of bounds for dimension {dim_idx} (label {label}) with size {tensor_dim}")

                # Index into overlap dimensions: result shape is [n_assignments, marg_dim_1, ..., marg_dim_m]
                indexed = tensor[tuple(projected_assignments.t())]  # shape: [n_assignments, marg_dim_1, ..., marg_dim_m]

                # Logsumexp over the marginalize dimensions (all dims except first)
                if n_marginalize > 0:
                    # Flatten marginalize dimensions and logsumexp
                    flat_indexed = indexed.reshape(len(assignments), -1)  # shape: [n_assignments, prod(marg_dims)]
                    values = torch.logsumexp(flat_indexed, dim=1, keepdim=True)  # shape: [n_assignments, 1]
                else:
                    values = indexed.reshape(-1, 1)
                return values

            else:
                # Full overlap: all tensor labels are in message_scope
                # Validate indices are within bounds before indexing
                for dim_idx, (label, tensor_dim) in enumerate(zip(tensor_labels, tensor.shape)):
                    assignment_col = projected_assignments[:, dim_idx]
                    max_val = assignment_col.max().item()
                    if max_val >= tensor_dim:
                        raise IndexError(f"Assignment index {max_val} out of bounds for dimension {dim_idx} (label {label}) with size {tensor_dim}")

                values = tensor[tuple(projected_assignments.t())].reshape(-1, 1)
                return values

        except Exception as e:
            print(f"ERROR in _get_values:")
            print(f"  Exception: {e}")
            print(f"  Tensor shape: {tensor.shape}")
            print(f"  Tensor labels: {tensor_labels}")
            print(f"  Message scope: {message_scope}")
            print(f"  Overlap labels: {overlap_labels}")
            print(f"  Marginalize labels: {marginalize_labels}")
            print(f"  Projected assignments shape: {projected_assignments.shape}")
            raise
    
    def to_exact(self):
        return self

    def get_factor_complexity(self):
        """
        Calculate the complexity of this factor.

        For regular FastFactor, this is simply the number of elements in the tensor.
        This method can be overridden in subclasses (e.g., FactorNN) to compute
        complexity without materializing the full tensor.

        Returns:
            int: Number of elements in the factor's tensor representation
        """
        if self.tensor is None:
            return 0
        return self.tensor.numel()

    def inverse(self):
        """
        Returns the inverse of this factor in log-space.
        In linear space this would be 1/f, but in log-space it's -f.
        Returns a deep copy with negated tensor values.
        """
        import copy
        ff_copy = copy.deepcopy(self)
        ff_copy.tensor = -ff_copy.tensor
        return ff_copy

    def shuffle(self):
        perm = torch.randperm(self.tensor.numel())
        self.tensor = self.tensor.reshape(-1)[perm].reshape(self.tensor.shape)