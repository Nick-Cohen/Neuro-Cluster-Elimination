import torch
import time
from torch import nn
from typing import List
from inference.factor import FastFactor
import torch.nn.functional as F

class FactorNN(FastFactor):
    """
    A subclass of FastFactor that integrates a neural network for more flexible message computation.
    """
    def __init__(self, net, data_processor):
        labels = net.bucket.get_message_scope()
        super().__init__(None, labels)
        self.is_nn = True
        self.net = net
        self.data_processor = data_processor
        self.gm = self.net.gm
        self.device = self.gm.device
        self.domain_sizes = [self.gm.matching_var(label).states for label in self.labels]
        
    def eliminate(self, elim_labels, elimination_scheme='sum'):
        """
        Create a fast factor with labels = self.labels - elim_labels and an empty tensor of the appropriate size.
        Query self.net in batches (to reduce memory usage) and sum the elim_variables.
        Insert the sum into the proper indices of the fast factor tensor.
        Return the fast factor.
        """
        remaining_labels = [label for label in self.labels if label not in elim_labels]
        elim_indices = [self.labels.index(label) for label in elim_labels]
        remaining_indices = [self.labels.index(label) for label in remaining_labels]

        # Compute the shape of the reduced tensor
        domain_sizes = [var.states for var in [self.gm.matching_var(label) for label in self.labels]]
        reduced_shape = [domain_sizes[idx] for idx in remaining_indices]

        # Initialize reduced FastFactor tensor
        reduced_tensor = torch.zeros(reduced_shape, device=self.net.device)

        # Determine batch dimensions and their sizes
        batch_dims = [domain_sizes[idx] for idx in elim_indices]
        batch_assignments = torch.cartesian_prod(*[torch.arange(size, device=self.net.device) for size in batch_dims])

        # Process assignments iteratively
        batch_size = 1024  # Adjust based on memory constraints
        for start in range(0, batch_assignments.size(0), batch_size):
            batch = batch_assignments[start:start + batch_size]

            # Expand batch assignments to include non-eliminated variables
            full_assignments = torch.cartesian_prod(
                *[torch.arange(domain_sizes[idx], device=self.net.device) if idx in elim_indices else torch.tensor([0], device=self.net.device)
                  for idx in range(len(self.labels))]
            )

            # Generate inputs for the network
            inputs = self.prepare_inputs(full_assignments, domain_sizes)

            # Query the neural network
            with torch.no_grad():
                outputs = self.net(inputs)

            # Reshape and reduce along the eliminated dimensions
            output_shape = [domain_sizes[idx] if idx in elim_indices else 1 for idx in range(len(self.labels))]
            outputs = outputs.reshape([batch.shape[0]] + output_shape[len(remaining_indices):])

            if elimination_scheme == 'sum':
                reduced_values = torch.logsumexp(outputs, dim=tuple(range(1, len(output_shape) - len(remaining_indices) + 1)))
            elif elimination_scheme == 'max':
                reduced_values, _ = outputs.max(dim=tuple(range(1, len(output_shape) - len(remaining_indices) + 1)))
            else:
                raise ValueError(f"Unsupported elimination scheme: {elimination_scheme}")

            # Insert reduced values into the reduced tensor
            for i, assignment in enumerate(batch):
                index = tuple(assignment.tolist())
                reduced_tensor[index] = reduced_values[i]

        # Create and return the reduced FastFactor
        return FastFactor(reduced_tensor, remaining_labels)
        """
        Process a batch of remaining assignments, query the network, and update the reduced tensor.
        """
        batch_assignments = torch.stack(batch)

        # Expand batch assignments to include elim variables
        expanded_assignments = torch.cartesian_prod(
            *[torch.arange(domain_sizes[idx], device=self.net.device) if idx in elim_indices else batch_assignments[:, remaining_indices.index(idx)]
              for idx in range(len(self.labels))]
        )

        # Generate network inputs for the batch
        inputs = self.prepare_inputs(expanded_assignments, domain_sizes)

        # Query the neural network
        with torch.no_grad():
            outputs = self.net(inputs)

        # Reshape outputs to match elim dimensions
        output_shape = [domain_sizes[idx] if idx in elim_indices else 1 for idx in range(len(self.labels))]
        outputs = outputs.reshape([batch_assignments.shape[0]] + output_shape[len(remaining_indices):])

        # Reduce along elim dimensions
        if elimination_scheme == 'sum':
            reduced_values = torch.logsumexp(outputs, dim=tuple(range(1, len(output_shape) - len(remaining_indices) + 1)))
        elif elimination_scheme == 'max':
            reduced_values, _ = outputs.max(dim=tuple(range(1, len(output_shape) - len(remaining_indices) + 1)))
        else:
            raise ValueError(f"Unsupported elimination scheme: {elimination_scheme}")

        # Insert reduced values into the reduced tensor
        for i, assignment in enumerate(batch_assignments):
            index = tuple(assignment.tolist())
            reduced_tensor[index] = reduced_values[i]
        
    def _get_slices(self, assignments, elim_vars, elim_domain_sizes, message_scope):
        """
        Args:
            assignments (torch.tensor): _description_
            elim_vars (list[int]): _description_
            message_scope (list[int]): _description_
            self.labels is a list of variable indices in the NN input
            
        """
        all_elim_assignments = torch.cartesian_prod(*[torch.arange(size) for size in elim_domain_sizes])
        
        # indices in assignments that correspond to dimensions in the tensor
        assignment_indices = [i for i, idx in enumerate(message_scope) if idx in self.labels]
        
        # indices of the assignment in the factor
        factor_assignment_indices = [i for i, idx in enumerate(self.labels) if idx not in elim_vars]
        # indices of eliminated variables in the factor
        factor_elim_indices = [i for i, idx in enumerate(self.labels) if idx in elim_vars]
        
        # create a tensor that will be the pre-one-hot-encoded list of all NN queries we need to make. Built by stacking projected assingments for each elim assigment.
        nn_assignments = torch.empty((len(assignments)*len(all_elim_assignments), len(self.labels)), dtype=torch.int64)
        
        for i,elim_assignment in enumerate(all_elim_assignments):
            # get the assignments in the nn_factor we need for a given elim assignment
            fixed_factor_assignment = self._filter_and_fix_assignments(assignments_tensor=assignments, assignments_column_labels=message_scope, summation_assignment=elim_assignment, summation_assignment_labels=[var.label for var in elim_vars])
            # fill in the fixed elim assignment assignments in the nn-assignments
            nn_assignments[i*len(assignments):(i+1)*len(assignments)] = fixed_factor_assignment
        
        # convert nn_assignments to a one-hot-encoded version
        one_hot_encoded_samples = torch.cat([F.one_hot(nn_assignments[:, i], num_classes=self.domain_sizes[i])[:, 1:] for i in range(len(self.labels))], dim=-1)
        one_hot_encoded_samples = one_hot_encoded_samples.float().to(self.net.device)
        
        # get the values and convert them back to unnormalized version, all within logspace
        # print(self.labels)
        values = self.data_processor.undo_normalization(self.net(one_hot_encoded_samples))
        return values.view(len(all_elim_assignments), len(assignments)).T
        
        
    def _filter_and_fix_assignments(self, assignments_tensor: torch.Tensor,
                                assignments_column_labels: List[int],
                                summation_assignment: List[int],
                                summation_assignment_labels: List[int]) -> torch.Tensor:
        """
        Maps columns from assignments_tensor to a new tensor based on factor_column_labels.
        If a label in factor_column_labels is not found in assignments_column_labels,
        the corresponding column is populated from summation_assignment using summation_assignment_labels.

        Args:
            assignments_tensor (torch.Tensor): 2D tensor with shape (m, n_assignments)
            assignments_column_labels (list[int]): Labels for columns in assignments_tensor
            factor_column_labels (list[int]): Desired column labels for the output tensor
            summation_assignment (list[int]): List of values to populate columns for unmatched labels
            summation_assignment_labels (list[int]): Labels corresponding to summation_assignment values

        Returns:
            torch.Tensor: 2D tensor with shape (m, len(factor_column_labels))
        """
        
        factor_column_labels = self.labels
        
        # Ensure input tensor is 2D
        assert assignments_tensor.ndim == 2, "assignments_tensor must be 2-dimensional"

        # Prepare output tensor with shape (m, len(self.labels))
        output_tensor = torch.zeros(assignments_tensor.size(0), len(self.labels))

        # Create a mapping from assignments_column_labels to column indices
        assignments_label_to_index = {label: idx for idx, label in enumerate(assignments_column_labels)}

        # Create a mapping from summation_assignment_labels to values
        summation_label_to_value = {label: value for label, value in zip(summation_assignment_labels, summation_assignment.view(-1))}

        # Iterate over self.labels to populate the output tensor
        for i, slice_label in enumerate(self.labels):
            if slice_label in assignments_label_to_index:
                # If the label exists in assignments_column_labels, use the corresponding column
                output_tensor[:, i] = assignments_tensor[:, assignments_label_to_index[slice_label]]
            elif slice_label in summation_label_to_value:
                # If the label exists in summation_assignment_labels, use the corresponding value
                output_tensor[:, i] = summation_label_to_value[slice_label]
            else:
                # If no matching label is found, raise an error
                raise ValueError(f"Label {slice_label} not found in assignments_column_labels or summation_assignment_labels")

        return output_tensor
    
    def to_exact(self):
        return FactorNN.nn_to_FastFactor(fastGM=self.gm, jit_file = None, net = self.net, device=self.device, debug=False, data_processor=self.data_processor)
      
    def order_indices(self):
        assert self.labels == sorted(self.labels), "Labels must be sorted"
    
    @staticmethod    
    def nn_to_FastFactor(fastGM, jit_file = None, net = None, device='cuda', debug=False, data_processor=None):
        if jit_file is None and net is None or jit_file is not None and net is not None:
            raise ValueError("Exactly one of a JIT file or a PyTorch net must be provided")
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
        scope = net.bucket.get_message_scope()
        domain_sizes = torch.tensor([fastGM.vars[fastGM.matching_var(v)].states for v in scope], device=device)
        
        # Calculate the total number of inputs
        total_inputs = (domain_sizes - 1).sum().item()
        
        if debug:
            input_creation_start = time.time()
        
        # Generate all possible assignments efficiently
        assignments = torch.cartesian_prod(*[torch.arange(size, device=device) for size in domain_sizes])
        
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

        outputs = data_processor.undo_normalization(outputs)
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
