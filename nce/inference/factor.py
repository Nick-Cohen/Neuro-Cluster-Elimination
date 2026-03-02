import torch
import math

class FastFactor:
    def __init__(self, tensor, labels):
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
    
    def _get_slices(self, assignments, elim_vars, elim_domain_sizes, message_scope):
        tensor = self.tensor
        tensor_labels = self.labels

        # Handle 0-dim tensors (scalar factors with empty labels)
        # These represent constant factors that broadcast to all samples and all elimination states
        if tensor.dim() == 0 or len(tensor_labels) == 0:
            # Create output shape: (num_samples, elim_dim1, elim_dim2, ...)
            expanded_shape = (len(assignments),) + tuple([v.states for v in elim_vars])
            # Broadcast the scalar value to this shape
            return tensor.expand(expanded_shape)

        # indices in assignments that correspond to dimensions in the tensor
        assignment_indices = [i for i, idx in enumerate(message_scope) if idx in tensor_labels]
        
        # indices of the assignment in the tensor
        tensor_assignment_indices = [i for i, idx in enumerate(tensor_labels) if idx not in elim_vars]
        # indices of eliminated variables in the tensor
        tensor_elim_indices = [i for i, idx in enumerate(tensor_labels) if idx in elim_vars]
        
        # put elimination indices at end of tensor
        permutation = (*tensor_assignment_indices, *tensor_elim_indices)
        # permute tensor
        tensor = tensor.permute(permutation)
        # reordered labels
        tensor_labels = [tensor_labels[i] for i in permutation]
        
        # get assignments from permuted tensor
        # assertation necessary for indexing
        assert(all(tensor_labels[i] < tensor_labels[i+1] for i in range(len(tensor_labels)-len(elim_vars)-1)))
        permuted_assignment_indices = [i for i, idx in enumerate(message_scope) if idx in tensor_labels]
        projected_assignments = assignments[:,permuted_assignment_indices]

        # stretch out elimination indices to 1d
        # grab slices corresponding to assignments
        view = tuple(int(dim) for dim in tensor.shape[:len(tensor.shape) - len(elim_vars)]) + (int(torch.prod(torch.tensor(tensor.shape[len(tensor.shape) - len(elim_vars):]))),)
        try:
            if not projected_assignments.numel() == 0:
                slices = tensor.view(view)[tuple(projected_assignments.t())]
            else:
                slices = tensor.unsqueeze(0).expand(len(assignments), len(tensor))
        except Exception:
            print(f"_get_slices error: tensor.shape={tensor.shape}, view={view}, "
                  f"projected_assignments.shape={projected_assignments.shape}")
            raise
        
        # reshape slices to match elimination variables in order, e.g. (1,2,2,1) if 2nd and 3rd variables are in tensor
        unexpanded_slice_shape = (len(assignments),) + tuple([v.states if v.label in tensor_labels else 1 for v in elim_vars])
        reshaped_slices = slices.reshape(unexpanded_slice_shape)
        expanded_slice_shape = (len(assignments),) + tuple([v.states for v in elim_vars])
        return reshaped_slices.expand(expanded_slice_shape)
    
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