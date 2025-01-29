import torch
import math
import time
# from data import DataPreprocessor

class FastFactor:
    def __init__(self, tensor, labels):
        self.tensor = tensor
        self.labels = labels
        self.vars = labels
        self.is_nn = False
        if tensor is not None:
            self.shape = self.tensor.shape
        else:
            self.shape = None

    def __repr__(self):
        return f"FastFactor(tensor={self.tensor}, labels={self.labels})"

    @classmethod
    def check_tensor_size(cls, tensor):
        pass

    @classmethod
    def check_memory_usage(cls):
        pass
    
    @property
    def device(self):
        return self.tensor.device

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
        try:
            if not other.labels:
                return FastFactor(self.tensor + other.tensor.item(), self.labels)
        except:
            print("got here")
            raise(ValueError("Other is not a FastFactor"))

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

        result_tensor = self_tensor.view(self_shape) + other_tensor.view(other_shape)
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
        
        return FastFactor(result_tensor, new_labels)

    def sum_all_entries(self):
        return self.eliminate('all').tensor.item()

    def to(self, device):
        self.tensor = self.tensor.to(device)
        return self

    def nn_to_FastFactor(idx, fastGM, data_processor, jit_file = None, net = None, device='cuda', debug=False):
        if jit_file is None and net is None or jit_file is not None and net is not None:
            raise ValueError("Exacgtly one of a JIT file or a PyTorch net must be provided")
        if debug:
            start_time = time.time()

        # Load the JIT model
        if debug:
            load_start = time.time()
        if jit_file is not None:
            model = torch.jit.load(jit_file).to(device)
        else:
            model = net
        if debug:
            load_end = time.time()
            print(f"Loading model took {load_end - load_start:.4f} seconds")

        # Get the scope and domain sizes
        scope = fastGM.message_scopes[idx]
        domain_sizes = torch.tensor([fastGM.vars[fastGM.matching_var(v)].states for v in scope], device=device)
        
        # Calculate the total number of inputs
        total_inputs = (domain_sizes - 1).sum().item()
        
        if debug:
            input_creation_start = time.time()
        
        # Generate all possible assignments efficiently
        assignments = torch.cartesian_prod(*[torch.arange(size, device=device) for size in domain_sizes])
        
        if len(assignments.shape) == 1:
            assignments = assignments.view(-1,1)
        
        # Create the input tensor efficiently
        all_inputs = torch.zeros((assignments.shape[0], total_inputs), device=device)
        
        offset = 0
        for i, size in enumerate(domain_sizes):
            if size > 1:
                mask = assignments[:, i].unsqueeze(1) == torch.arange(1, size, device=device)
                all_inputs[:, offset:offset+size-1] = mask.float()
                offset += size - 1

        if debug:
            input_creation_end = time.time()
            print(f"Creating input tensor took {input_creation_end - input_creation_start:.4f} seconds")

        # Query the model for all inputs
        if debug:
            query_start = time.time()
        with torch.no_grad():
            outputs = model(all_inputs).reshape(tuple(domain_sizes.tolist()))
        if debug:
            query_end = time.time()
            print(f"Querying model took {query_end - query_start:.4f} seconds")
            
        # transform outputs back
        if data_processor is not None:
            if not data_processor.is_logspace:
                outputs = data_processor.convert_back_message_logspace(outputs)
            else:
                outputs += data_processor.y_max
                outputs /= torch.log(torch.tensor(10.0)).to(outputs.device)

        # Create and return the FastFactor
        if debug:
            factor_creation_start = time.time()
        fast_factor = FastFactor(outputs, scope)
        if debug:
            factor_creation_end = time.time()
            print(f"Creating FastFactor took {factor_creation_end - factor_creation_start:.4f} seconds")

        if debug:
            end_time = time.time()
            print(f"Total time for nn_to_FastFactor: {end_time - start_time:.4f} seconds")

        return fast_factor

    def klargest(self, k):
        # Get top-k values and their indices from the flattened tensor
        topk_values, topk_flat_indices = torch.topk(self.tensor.flatten(), k)

        # Convert flat indices back to original shape
        topk_indices = torch.unravel_index(topk_flat_indices, self.tensor.shape)

        # Combine the indices into tuples (e.g., [(x1, y1), (x2, y2), ...])
        topk_indices = list(zip(*(dim.tolist() for dim in topk_indices)))
        topk_values = topk_values.tolist()  # Convert values to a plain list
        
        return topk_indices, topk_values
    
    def _get_slices(self, assignments, elim_vars, elim_domain_sizes, message_scope):
        tensor = self.tensor
        tensor_labels = self.labels
        
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
        except:
            print(tensor.shape)
            print(view)
            print(projected_assignments.shape)
            print(projected_assignments.t().shape)
            exit(1)
        
        # reshape slices to match elimination variables in order, e.g. (1,2,2,1) if 2nd and 3rd variables are in tensor
        unexpanded_slice_shape = (len(assignments),) + tuple([v.states if v.label in tensor_labels else 1 for v in elim_vars])
        reshaped_slices = slices.reshape(unexpanded_slice_shape)
        expanded_slice_shape = (len(assignments),) + tuple([v.states for v in elim_vars])
        return reshaped_slices.expand(expanded_slice_shape)