#%%
import sys
from typing import List
import re
import os
import math
from functools import reduce
import operator
import numpy as np
import importlib
# import psutil
import torch
import itertools
import gc
import time
# sys.path.append('/home/cohenn1/SDBE/PyGMs/pyGMs')

# # Path to the parent directory of the package
# package_parent_dir = '/home/cohenn1/SDBE/PyGMs'
# package_name = 'pyGMs'

# if package_parent_dir not in sys.path:
#     sys.path.append(package_parent_dir)
import pyGMs as gm
from pyGMs import wmb
from pyGMs.neuro import *
from pyGMs.graphmodel import eliminationOrder
from pyGMs import Var

dprint = print # used to control f for debugging statements

class FastFactor:
    def __init__(self, tensor, labels):
        self.tensor = tensor
        self.labels = labels

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
            
    def eliminate(self, elim_labels):
        if elim_labels == 'all':
            elim_indices = list(range(len(self.labels)))
            new_labels = []
        else:
            elim_indices = [self.labels.index(label) for label in elim_labels]
            new_labels = [label for label in self.labels if label not in elim_labels]
        
        # Convert from log10 to natural log, perform logsumexp, then convert back to log10
        result_tensor = torch.logsumexp(self.tensor * math.log(10), dim=elim_indices) / math.log(10)
        
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

    def nn_to_FastFactor(idx, fastGM, jit_file = None, net = None, device='cuda', debug=False):
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

class FastBucket:
    
    # TODO: Multiply factors in a more sensible order, e.g. subsumed multiplications first
    
    def __init__(self, label, factors, device, elim_vars, isRoot = False):
        self.label = label
        self.factors = factors
        self.device = device
        self.elim_vars = elim_vars
        self.isRoot = isRoot

        # Assert that all factors are on the specified device type
        for factor in self.factors:
            assert self.device in str(factor.device), f"Factor device {factor.device} does not match bucket device type {self.device}"
    
    def compute_message(self):
        # Multiply all factors
        if not self.factors:
            raise ValueError("No factors in the bucket to send message from")
        
        message = self.factors[0]
        for factor in self.factors[1:]:
            message = message * factor

        # Eliminate variables
        message = message.eliminate(self.elim_vars)
        return message

    def send_message(self, bucket: 'FastBucket'):
        """
        Multiply all factors, eliminate variables, and send the resulting message to another bucket.
        """
        # Multiply all factors
        if not self.factors:
            raise ValueError("No factors in the bucket to send message from")
        
        message = self.factors[0]
        for factor in self.factors[1:]:
            message = message * factor

        # Eliminate variables
        message = message.eliminate(self.elim_vars)

        # Send the message to the receiving bucket
        bucket.receive_message(message)

    def receive_message(self, message: FastFactor):
        """
        Receive a message (factor) from another bucket and append it to this bucket's factors.
        """
        # Assert that the incoming message is on the correct device
        assert str(self.device) in str(message.device), f"Message device {message.device} does not match bucket device {self.device}"

        # Append the message to the factors list
        self.factors.append(message)
        
    def get_message_scope(self):
        scope = set()
        for factor in self.factors:
            scope = scope.union(factor.labels)
        scope.discard(self.label)
        return sorted(list(scope))
        
def test_fast_bucket():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create factors in log10 space
    f1 = FastFactor(torch.log10(torch.tensor([[1., 2.], [3., 4.]])).to(device), ['a', 'b'])
    f2 = FastFactor(torch.log10(torch.tensor([[0.1, 0.2], [0.3, 0.4]])).to(device), ['b', 'c'])

    bucket1 = FastBucket('a', [f1], device, elim_vars=['a'])
    bucket2 = FastBucket('b', [f2], device, elim_vars=['b'])

    bucket1.send_message(bucket2)

    new_factor = bucket2.factors[-1]
    print("\nNew factor in Bucket 2:")
    print(f"Labels: {new_factor.labels}")
    print(f"Tensor:\n{new_factor.tensor}")

    # Correct calculation in log10 space
    expected_message = torch.log10(torch.tensor([1. + 3., 2. + 4.])).to(device)
    print(f"\nExpected message:\n{expected_message}")

    print(f"\nDifference:\n{new_factor.tensor - expected_message}")

    assert torch.allclose(new_factor.tensor, expected_message, atol=1e-6), "Message content is incorrect"

    print("\nTest passed successfully!")

def wtminfill_order(factors_or_buckets, variables_not_eliminated=None):
    """
    Find an elimination order using the weighted min-fill heuristic for a subset of variables.
    
    Args:
    factors_or_buckets (list): Either a list of FastFactor objects or a list of FastBucket objects.
    variables_not_eliminated (list): List of variable labels not eliminated. If None, all variables are considered.
    
    Returns:
    list: The elimination order for the specified variables (first eliminated first).
    """
    # Extract factors from buckets if necessary
    if isinstance(factors_or_buckets[0], FastBucket):
        factors = [factor for bucket in factors_or_buckets for factor in bucket.factors]
    else:
        factors = factors_or_buckets

    # Get all unique variables
    all_variables = set()
    for factor in factors:
        all_variables.update(factor.labels)
    all_variables = list(all_variables)

    # If variables_to_eliminate is not specified, use all variables
    if variables_not_eliminated is None:
        variables_to_eliminate = all_variables
    else:
        variables_to_eliminate = [var for var in all_variables if var not in variables_not_eliminated]

    # Create adjacency matrix
    n = len(all_variables)
    adj_matrix = np.zeros((n, n), dtype=int)
    var_to_index = {var: i for i, var in enumerate(all_variables)}

    for factor in factors:
        for i, var1 in enumerate(factor.labels):
            for var2 in factor.labels[i+1:]:
                idx1, idx2 = var_to_index[var1], var_to_index[var2]
                adj_matrix[idx1, idx2] = adj_matrix[idx2, idx1] = 1

    # Initialize priority queue for variables to eliminate
    pq = [(_compute_fill_weight(adj_matrix, var_to_index[var]), var) 
            for var in variables_to_eliminate]
    pq.sort(reverse=True)  # Higher weight = higher priority to eliminate

    elimination_order = []
    while pq:
        _, var = pq.pop()
        elimination_order.append(var)

        # Update adjacency matrix
        var_idx = var_to_index[var]
        neighbors = np.where(adj_matrix[var_idx] == 1)[0]
        for i in neighbors:
            for j in neighbors:
                if i != j:
                    adj_matrix[i, j] = adj_matrix[j, i] = 1
        adj_matrix[var_idx, :] = adj_matrix[:, var_idx] = 0

        # Update priorities for remaining variables to eliminate
        pq = [(_compute_fill_weight(adj_matrix, var_to_index[v]), v) 
                for _, v in pq if v in variables_to_eliminate]
        pq.sort(reverse=True)

    # Add variables not eliminated to the end of the elimination order
    elimination_order.extend(variables_not_eliminated)
    
    return elimination_order

def _compute_fill_weight(adj_matrix, var_idx):
    """
    Compute the weighted min-fill score for a variable.
    """
    neighbors = np.where(adj_matrix[var_idx] == 1)[0]
    fill_edges = 0
    for i in neighbors:
        for j in neighbors:
            if i < j and adj_matrix[i, j] == 0:
                fill_edges += 1
    return fill_edges * len(neighbors)  # weight by cluster size
   
class FastGM:
    def __init__(self, elim_order=None, buckets=None, factors=None, uai_file=None, device="cuda", reference_fastgm=None):
        self.device = device
        self.vars = []
        self.elim_order = None

        if uai_file is not None:
            self._load_from_uai(uai_file)
        elif buckets is not None:
            self.buckets = buckets
            self.elim_order = elim_order
        elif factors is not None:
            self._load_vars_from_factors(factors)
            if elim_order is not None:
                self.load_elim_order(elim_order, reference_fastgm)
                # print(self.elim_order)
            else:
                print("Computing elim order")
                self.load_elim_order(wtminfill_order(factors), reference_fastgm)
            self.buckets = self._create_buckets_from_factors(factors)
        else:
            raise ValueError("Either buckets, factors, or a UAI file must be provided")
        
        self.message_scopes = {}
        self.calculate_message_scopes()

    def __repr__(self):
        return f"FastGM(elim_order={self.elim_order}, buckets={self.buckets})"

    def __str__(self):
        output = []
        for var in self.elim_order:
            bucket = self.buckets[var]
            factor_strs = []
            for factor in bucket.factors:
                factor_strs.append(f"f({', '.join(map(str, factor.labels))})")
            output.append(f"Bucket {var}: {' '.join(factor_strs)}")
        return "\n".join(output)
    
    def _load_from_uai(self, uai_file):
        # Load the UAI file
        ord_file = uai_file + ".vo"
        gm_model = uai_to_GM(uai_file=uai_file, order_file=ord_file)
        self.vars = gm_model.vars
        
        # Convert PyGM factors to FastFactors
        fast_factors = []
        for factor in gm_model.factors:
            tensor = torch.tensor(factor.table, dtype=torch.float32).to(self.device)
            labels = [var.label for var in factor.vars]
            fast_factors.append(FastFactor(torch.log10(tensor), labels))
        
        # Set elimination order
        self.elim_order = gm_model.elim_order
        
        # Create buckets from FastFactors
        self.buckets = self._create_buckets_from_factors(fast_factors)

    def _load_vars_from_factors(self, factors):
        var_domains = {}
        for factor in factors:
            # print(factor.labels)
            if factor.labels:
                for i, label in enumerate(factor.labels):
                    if label not in var_domains:
                        var_domains[label] = factor.tensor.shape[i]
        for label, domain_size in var_domains.items():
            self.vars.append(Var(label, domain_size))

    def _create_buckets_from_factors(self, factors):
        if self.elim_order is None:
            raise ValueError("Elimination order must be set before creating buckets")
        
        # print(self.elim_order)
        buckets = {var: FastBucket(var.label, [], self.device, [var]) for var in self.elim_order}
        unplaced_factors = set(factors)

        for var in self.elim_order:
            factors_to_place = []
            for factor in unplaced_factors:
                if var in factor.labels:
                    factors_to_place.append(factor)
            
            for factor in factors_to_place:
                buckets[var].factors.append(factor)
                unplaced_factors.remove(factor)

        if unplaced_factors:
            raise ValueError(f"Some factors could not be placed in buckets: {unplaced_factors}")

        return buckets 

    def get_factors(self):
        factors = []
        for bucket in list(self.buckets.values()):
            for factor in bucket.factors:
                factors.append(factor)
        return factors

    def get_bucket(self, bucket_id):
        return self.buckets[self.matching_var(bucket_id)]

    def matching_var(self, var_index):
        for var in self.vars:
            if var_index == var.label:
                return var
    
    def load_elim_order(self, elim_order, reference_fastgm=None): 
        if reference_fastgm is not None:
            self.elim_order = [reference_fastgm.matching_var(var_index) for var_index in elim_order]
        elif isinstance(elim_order[0], Var):
            self.elim_order = elim_order
        elif isinstance(elim_order[0], int):
            self.elim_order = []
            for var_index in elim_order:
                if self.matching_var(var_index):
                    self.elim_order.append(self.matching_var(var_index))
                else:
                    raise ValueError("No matching var found for idx ", var_index)
                    
            # self.elim_order = [self.matching_var(var_index) for var_index in elim_order]
        else:
            raise ValueError("Elimination order must be a list of Var objects or integers")
        # print(self.elim_order)
            
    def eliminate_variables(self, elim_vars=None, up_to=None, through=None, all=False, all_but=None):
        if sum(map(bool, [elim_vars, up_to, through, all, all_but])) != 1:
            raise ValueError("Exactly one of elim_vars, up_to, through, or all must be specified")

        if all:
            vars_to_eliminate = self.elim_order
        elif elim_vars:
            vars_to_eliminate = elim_vars
        elif up_to:
            vars_to_eliminate = self.elim_order[:self.elim_order.index(up_to)]
        elif through:
            vars_to_eliminate = self.elim_order[:self.elim_order.index(through) + 1]
        elif all_but:
            vars_to_keep = set(all_but)
            vars_to_eliminate = [var for var in self.elim_order if var not in vars_to_keep]

        # Create a dummy root bucket to collect the final result
        root_bucket = FastBucket('root', [], self.device, [], isRoot=True)
        
        max_width = 0

        for var in vars_to_eliminate:
            # if type(var) != int:
            #     var = var.label
            current_bucket = self.buckets[var]
            # print(current_bucket.label)
            message = self.process_bucket(current_bucket)
            
            if message.labels:  # If the message is not a scalar
                # Find the next appropriate bucket
                if len(message.labels) > max_width:
                    max_width = len(message.labels)
                next_bucket = self.find_next_bucket(message.labels, var)
                if next_bucket:
                    next_bucket.receive_message(message)
                else:
                    # If no appropriate bucket found, send to root
                    root_bucket.receive_message(message)
            else:
                # If the message is a scalar, send to root
                root_bucket.receive_message(message)
            
            # Remove the eliminated variable's bucket
            del self.buckets[var]

        # Process the root bucket
        if root_bucket.factors:
            result = root_bucket.factors[0]
            for factor in root_bucket.factors[1:]:
                result = result * factor
        else:
            # If no factors were sent to the root, return an identity factor
            result = FastFactor(torch.tensor(0.0, device=self.device).reshape(1), [])

        # If we've eliminated all variables, save the partition function
        if all:
            assert len(result.labels) == 0, "Not all variables were eliminated"
            self.log_partition_function = result.tensor.item()

        return result

    def eliminate_max(self, elim_labels):
        if elim_labels == 'all':
            elim_indices = list(range(len(self.labels)))
            new_labels = []
        else:
            elim_indices = [self.labels.index(label) for label in elim_labels]
            new_labels = [label for label in self.labels if label not in elim_labels]
        
        result_tensor, _ = torch.max(self.tensor, dim=elim_indices[0])
        for dim in elim_indices[1:]:
            result_tensor, _ = torch.max(result_tensor, dim=dim-len(elim_indices)+1)
        
        if type(result_tensor) == float:
            result_tensor = torch.Tensor([result_tensor])
        
        # If we've eliminated all variables, we need to ensure the result is a scalar
        if not new_labels:
            result_tensor = result_tensor.view(1)
        
        return FastFactor(result_tensor, new_labels)
    
    def process_bucket(self, bucket):
        """Process a bucket by multiplying all factors and eliminating the bucket's variable."""
        if not bucket.factors:
            raise ValueError("No factors in the bucket to process")
        
        message = bucket.factors[0]
        for factor in bucket.factors[1:]:
            message = message * factor

        # Eliminate the bucket's variable
        message = message.eliminate(bucket.elim_vars)
        return message
    
    def calculate_message_scopes(self):
        """Calculate and save the message scope for each bucket."""
        elimination_scheme = []

        for var in self.elim_order:
            bucket_factors = self.buckets[var].factors
            incoming_messages = []
            outgoing_message_vars = set()

            # Determine incoming messages from previous buckets
            for prev_bucket in elimination_scheme:
                if prev_bucket['sends_to'] is not None and prev_bucket['sends_to'].label == var.label:
                    incoming_messages.append(prev_bucket['outgoing_message'])
                    outgoing_message_vars.update(prev_bucket['outgoing_message'])

            # Add variables from the bucket's own factors
            for factor in bucket_factors:
                outgoing_message_vars.update(factor.labels)

            # Remove the bucket's own variable
            outgoing_message_vars.discard(var.label)

            # Save the scope for this bucket
            self.message_scopes[var.label] = sorted(list(outgoing_message_vars))

            # Find the next bucket to send the message to
            next_var = self.find_next_bucket(list(outgoing_message_vars), var)

            bucket_info = {
                'var': var,
                'sends_to': next_var,
                'outgoing_message': sorted(list(outgoing_message_vars))
            }

            elimination_scheme.append(bucket_info)

    def show_message_scopes(self):
        """Display the calculated bucket scopes."""
        print("Bucket Scopes:")
        for var in self.elim_order:
            print(f"Bucket {var}: {self.message_scopes[var.label]}")
    
    def show_elimination(self, elim_vars=None, up_to=None, through=None, all=False):
        if sum(map(bool, [elim_vars, up_to, through, all])) != 1:
            raise ValueError("Exactly one of elim_vars, up_to, through, or all must be specified")

        if all:
            vars_to_eliminate = self.elim_order
        elif elim_vars:
            vars_to_eliminate = elim_vars
        elif up_to:
            vars_to_eliminate = self.elim_order[:self.elim_order.index(up_to)]
        elif through:
            vars_to_eliminate = self.elim_order[:self.elim_order.index(through) + 1]

        elimination_scheme = []
        max_width = 0

        for var in self.elim_order:
            if var not in vars_to_eliminate:
                continue
            
            bucket_factors = self.buckets[var].factors
            incoming_messages = []
            outgoing_message_vars = set()

            # Determine incoming messages from previous buckets
            for prev_bucket in elimination_scheme:
                if prev_bucket['sends_to'] is not None and prev_bucket['sends_to'].label == var.label:
                    incoming_messages.append(prev_bucket['outgoing_message'])
                    outgoing_message_vars.update(prev_bucket['outgoing_message'])

            # Determine variables in the outgoing message
            for factor in bucket_factors:
                outgoing_message_vars.update(factor.labels)
            outgoing_message_vars.discard(var)

            # Find the next bucket to send the message to
            next_var = self.find_next_bucket(list(outgoing_message_vars), var)

            width = len(list(outgoing_message_vars))
            max_width = max(max_width, width)

            bucket_info = {
                'var': var,
                'factors': [f"f{i}({', '.join(map(str, factor.labels))})" for i, factor in enumerate(bucket_factors)],
                'receives': [f"mess_to_{var}({', '.join(map(str, sorted(msg)))})" for msg in incoming_messages],
                'sends_to': next_var,
                'outgoing_message': sorted(list(outgoing_message_vars)),
                'width': width
            }

            elimination_scheme.append(bucket_info)

        # Print the elimination scheme
        print("Elimination Scheme:")
        for bucket in elimination_scheme:
            print(f"Bucket {bucket['var']}:")
            print(f"  Factors: {', '.join(bucket['factors'])}")
            print(f"  Receives: {', '.join(bucket['receives']) if bucket['receives'] else 'None'}")
            if bucket['sends_to'] is not None:
                print(f"  Sends: mess_{bucket['var']}_to_{bucket['sends_to']}({', '.join(map(str, bucket['outgoing_message']))}) to bucket {bucket['sends_to']}")
            else:
                print(f"  Sends: mess_{bucket['var']}_to_root({', '.join(map(str, bucket['outgoing_message']))}) to root")
            print(f"  Width: {bucket['width']}")
            print()

        print(f"Maximum width: {max_width}")

    def find_next_bucket(self, labels, current_var):
        """Find the next bucket that shares any variable with the given labels."""
        current_index = self.elim_order.index(current_var)
        for var in self.elim_order[current_index:]:  # Look at earlier variables
            if var in self.buckets and any(label in labels for label in self.buckets[var].elim_vars):
                return self.buckets[var]
        return None  # If no appropriate bucket found, this will send the message to root

    def get_joint_distribution(self):
        # used for message gradient
        joint = None
        for bucket in self.buckets.values():
            for factor in bucket.factors:
                if joint is None:
                    joint = factor
                else:
                    joint *= factor
        return joint

    def get_log_partition_function(self):
        """
        Returns the log partition function if it has been computed, otherwise computes it.
        """
        if hasattr(self, 'log_partition_function'):
            return self.log_partition_function
        else:
            result = self.eliminate_variables(all=True)
            return self.log_partition_function

    def get_message_gradient(self, bucket_var):
        # function should do elimination up to, but not including bucket_var.
        # function should gather all factors from all buckets that come after bucket_var and create a new fastGM from them.
        # function should eliminate all bucket the variables in the scope of bucket_var's bucket's scope
        self.eliminate_variables(up_to=bucket_var)
        gradient_factors = []
        for var in self.elim_order[self.elim_order.index(self.matching_var(bucket_var))+1:]:
            bucket = self.buckets[var]
            bucket_factors = bucket.factors
            for factor in bucket_factors:
                gradient_factors.append(factor)
        bucket = self.get_bucket(bucket_var)
        message = bucket.compute_message()
        bucket_scope = bucket.get_message_scope()
        downstream_elim_order = wtminfill_order(gradient_factors, variables_not_eliminated=bucket_scope)
        # print("deo is ", downstream_elim_order)
        # device_copy=str(self.device)
        downstream_gm = FastGM(factors=gradient_factors, elim_order=downstream_elim_order, reference_fastgm=self, device=self.device)
        downstream_gm.eliminate_variables(all_but=bucket_scope)
        return downstream_gm.get_joint_distribution(), message

    def removeFactors(self, factors_to_remove):
        for var in self.buckets:
            self.buckets[var].factors = [f for f in self.buckets[var].factors if f not in factors_to_remove]

    def addFactors(self, factors_to_add):
        for factor in factors_to_add:
            if not factor.labels:  # If the factor has no labels (scalar factor)
                # Add to a special 'constant' bucket or handle as needed
                if 'constant' not in self.buckets:
                    self.buckets['constant'] = FastBucket('constant', [], self.device, [])
                self.buckets['constant'].factors.append(factor)
            else:
                # find earliest bucket in elim order that contains any of the labels
                indices = [self.elim_order.index(self.matching_var(v)) for v in factor.labels]
                # find index of earliest bucket
                earliest_var = factor.labels[indices.index(min(indices))]
                # move to that bucket
                try:
                    self.buckets[self.matching_var(earliest_var)].factors.append(factor)
                except:
                    print('got here')
                    raise(ValueError("No matching var found for idx ", earliest_var))
                
                # self.buckets[var].factors.append(factor)

    def get_wmb_message_gradient(self, bucket_var, i_bound, weights='max'):
        self.eliminate_variables(up_to=bucket_var)

        gradient_factors = []
        
        
        for var in self.elim_order[self.elim_order.index(self.matching_var(bucket_var))+1:]:
            bucket = self.buckets[var]
            gradient_factors.extend(bucket.factors)

        # confirm no variables that should have been eliminated are in gradient factors
        should_have_been_eliminated = set([v.label for v in self.elim_order[0:self.elim_order.index(self.matching_var(bucket_var))]])
        
        for factor in gradient_factors:
            for v in factor.labels:
                if type(v) != int:
                    raise(ValueError("Variable ", v, " is not an int"))
                if v in should_have_been_eliminated:
                    raise ValueError("Variable ", v, " should have been eliminated")
        
        
        bucket = self.get_bucket(bucket_var)
        bucket_scope = bucket.get_message_scope()
        # dprint('bucket scope here reads', bucket_scope)

        # Get all variables involved in the gradient factors
        all_vars = set()
        for factor in gradient_factors:
            all_vars.update(factor.labels)

        # Create a new elimination order for the downstream graph
        downstream_elim_order = wtminfill_order(gradient_factors, variables_not_eliminated=bucket_scope)
        
        # Convert the elimination order to Var objects
        downstream_elim_order = [self.matching_var(var) for var in downstream_elim_order]

        # Create the downstream graphical model
        downstream_gm = FastGM(factors=gradient_factors, elim_order=downstream_elim_order, device=self.device)

        return self._wmb_eliminate(downstream_gm, bucket_scope, i_bound, weights)

    def _wmb_eliminate(self, gm, target_scope, i_bound, weights):
        """
        Perform Weighted Mini-Bucket elimination.

        Args:
        gm (FastGM): The graphical model to eliminate.
        target_scope (list): The variables to keep (not eliminate).
        i_bound (int): The maximum allowed scope size for mini-buckets.
        weights (str or list): Weights for WMB.

        Returns:
        FastFactor: The result of WMB elimination.
        """
        # dprint('target scope is ', target_scope)
        if isinstance(weights, str):
            if weights == 'max':
                weight_map = {var.label: 0.0 for var in gm.vars}
            elif weights == 'sum':
                weight_map = {var.label: 1.0 for var in gm.vars}
            else:
                raise ValueError("Unknown weight type. Use 'max', 'sum', or provide a list of weights.")
        else:
            weight_map = {var.label: weight for var, weight in zip(gm.vars, weights)}
        
        result = None
        # dprint('elim order is ', gm.elim_order)
        # dprint()
        # dprint('All bucket factors scopes are:')
        # for key in self.buckets.keys():
        #     dprint("Bucket = ", key)
        #     bucket = self.buckets[key]
        #     dprint('Factors scopes are:')
        #     for factor in bucket.factors:
        #         dprint(factor.labels)
        #     dprint()
                
        for var in gm.elim_order:
            # dprint('var ', var, ' considered')
            if var.label in target_scope:
                continue

            bucket = gm.get_bucket(var)
            
            if not bucket.factors:  # Skip empty buckets
                continue

            if len(bucket.factors[0].labels) <= i_bound:
                message = self._compute_weighted_message(bucket.factors, var, weight_map[var.label])
                gm.removeFactors(bucket.factors)
                gm.addFactors([message])
                # dprint(message.labels)
            else:
                mini_buckets = self._create_mini_buckets(bucket.factors, i_bound)
                for i, mini_bucket in enumerate(mini_buckets):
                    mini_weight = weight_map[var.label] / len(mini_buckets)
                    if i == len(mini_buckets) - 1:  # Adjust the last mini-bucket weight
                        mini_weight = weight_map[var.label] - (len(mini_buckets) - 1) * mini_weight
                    mini_message = self._compute_weighted_message(mini_bucket, var, mini_weight)
                    gm.removeFactors(mini_bucket)
                    gm.addFactors([mini_message])
                    # dprint(mini_message.labels)

        # After elimination, combine all remaining factors
        remaining_factors = []
        for bucket in gm.buckets.values():
            remaining_factors.extend(bucket.factors)
        
        if remaining_factors:
            result = remaining_factors[0]
            if (result.tensor == float('inf')).any():
                print("inf found")
                print(remaining_factors[0].labels)
                raise ValueError("inf found")
            for factor in remaining_factors[1:]:
                try:
                    result = result * factor
                    # if sum([0 if v in target_scope else 1 for v in factor.labels]) != 0:
                        # dprint('factor scope is ', factor.labels)
                    # dprint('result scope is ', result.labels)
                except:
                    print('err')
                    raise ValueError("scope incorrect")
                if (result.tensor == float('inf')).any():
                    print("inf found")
                    print(factor.labels)
                    raise ValueError("inf found")
        else:
            # If no factors remain, return a scalar factor with value 0 (in log space)
            result = FastFactor(torch.tensor([0.0], device=self.device), [])
        if (result.tensor == float('inf')).any():
            print("inf found")
            raise ValueError("inf found")
        return result

    def _create_mini_buckets(self, factors, i_bound):
        """
        Partition factors into mini-buckets respecting the i-bound.
        """
        mini_buckets = []
        sorted_factors = sorted(factors, key=lambda f: len(f.labels), reverse=True)
        for factor in sorted_factors:
            placed = False
            for bucket in mini_buckets:
                if len(set.union(*[set(f.labels) for f in bucket], set(factor.labels))) <= i_bound:
                    bucket.append(factor)
                    placed = True
                    break
            if not placed:
                mini_buckets.append([factor])
        return mini_buckets

    def _compute_weighted_message(self, factors, var, weight):
        # Multiply factors using the FastFactor multiplication method
        product = factors[0]
        for factor in factors[1:]:
            product = product * factor  # Correctly handles label alignment and tensor operations

        # Proceed with elimination based on the weight
        if weight == 0:  # max-product
            out = self._eliminate_max(product, var)
        elif weight == 1:  # sum-product
            out = product.eliminate([var])
        else:  # weighted sum-product
            out = self._eliminate_weighted_sum(product, var, weight)
        return out

    def _eliminate_max(self, factor, var):
        """Eliminate a variable using max-product in log10 space."""
        dim = factor.labels.index(var.label)
        max_values, _ = torch.max(factor.tensor, dim=dim)
        return FastFactor(max_values, [label for label in factor.labels if label != var.label])

    def _eliminate_weighted_sum(self, factor, var, weight):
        dim = factor.labels.index(var.label)
        # Convert from log10 to natural log, perform weighted sum, then convert back to log10
        natural_log_tensor = factor.tensor * math.log(10)
        weighted_sum = torch.logsumexp(natural_log_tensor * weight, dim=dim) / weight
        log10_result = weighted_sum / math.log(10)
        return FastFactor(log10_result, [label for label in factor.labels if label != var.label])
    
    @staticmethod
    def sample_output_function(factors, sum_vars, sample_assignments, device='cuda'):
        sum_vars_set = set(sum_vars)
        
        # Step 1: Create stacked slices for each factor
        stacked_factors = [FastGM._create_stacked_slice(factor, sample_assignments, device)
                        for factor in factors]
        
        # Step 2: Multiply (add in log space) all stacked factors
        result = torch.zeros(sample_assignments.shape[0], device=device)
        for stacked_factor in stacked_factors:
            if stacked_factor.dim() > 1:
                stacked_factor = torch.sum(stacked_factor, dim=tuple(range(1, stacked_factor.dim())))
            result += stacked_factor
        
        return result


    @staticmethod
    def _create_stacked_slice(factor, sample_assignments, device):
        # Create a list to hold slices for each sample
        slices = []

        for assignment in sample_assignments:
            # Create indexing tuple for this sample
            index = tuple(assignment.get(var, slice(None)) for var in factor.labels)
            
            # Extract the slice and add it to the list
            slices.append(factor.tensor[index])

        # Stack all slices
        return torch.stack(slices).to(device)
    
    @staticmethod
    def _get_exact_value(exact_result, assignment):
        index = tuple(assignment.get(i, 0) for i in range(exact_result.dim()))
        return exact_result[index].item()

    @staticmethod
    def generate_sample_assignments(var_dims, non_sum_var_nums, num_samples, device='cuda'):
        # Create a tensor of random integers for each variable
        sample_tensors = {
            var: torch.randint(0, var_dims[var], (num_samples,), device=device)
            for var in non_sum_var_nums
        }
        
        # Combine into a single tensor
        combined_samples = torch.stack([sample_tensors[var] for var in non_sum_var_nums], dim=1)
        
        return combined_samples 
        
    @staticmethod
    def tester_sample_output_function(factors, sum_var_nums, num_samples=100, device='cuda'):
        # Get all variable numbers
        all_var_nums = set()
        for factor in factors:
            all_var_nums.update(factor.labels)
        all_var_nums = list(all_var_nums)

        # Compute exact result
        exact_result = factors[0]
        for factor in factors[1:]:
            exact_result = exact_result * factor
        exact_result = exact_result.eliminate(sum_var_nums)

        # Generate sample assignments
        non_sum_var_nums = [var for var in all_var_nums if var not in sum_var_nums]
        var_dims = {}
        for factor in factors:
            for var in factor.labels:
                if var not in var_dims and var in non_sum_var_nums:
                    var_dims[var] = factor.tensor.shape[factor.labels.index(var)]

        sample_assignments = FastGM.generate_sample_assignments(var_dims, non_sum_var_nums, num_samples, device)

        # Compute sampled result
        sampled_result = FastGM.sample_output_function(factors, sum_var_nums, sample_assignments, device)

        # Compare results
        errors = []
        for i, assignment in enumerate(sample_assignments):
            sampled_value = sampled_result[i].item()
            exact_value = FastGM._get_exact_value(exact_result, assignment)
            error = abs(sampled_value - exact_value)
            errors.append(error)

            print(f"Sample {i}:")
            print(f"  Assignment: {assignment}")
            print(f"  Sampled value: {sampled_value}")
            print(f"  Exact value: {exact_value}")
            print(f"  Error: {error}")
            print()

        # Compute and print average error
        avg_error = sum(errors) / len(errors)
        print(f"Average error: {avg_error}")

        return avg_error  
        
class FastGMTester:
    def __init__(self, fastgm, max_memory_mb=100, max_tensor_size_mb=50):
        self.fastgm = fastgm
        self.max_memory_mb = max_memory_mb
        self.max_tensor_size_mb = max_tensor_size_mb
        self.max_tensor_size = 0
        self.max_width = 0
        self.tensor_size_history = []
        self.width_history = []
        self.memory_usage_history = []
        self.original_multiply = FastFactor.__mul__
        self.original_eliminate = FastFactor.eliminate
        self.step_counter = 0
        self.device = fastgm.device

    def wrap_fast_factor_methods(self):
        tester = self

        def wrapped_multiply(self, other):
            print(f"Multiply operation called")
            print(f"  Factor 1 labels: {self.labels}")
            print(f"  Factor 2 labels: {other.labels}")
            result = tester.original_multiply(self, other)
            print(f"  Result labels: {result.labels}")
            tester.check_tensor_size(result.tensor)
            tester.check_memory_usage()
            tester.check_treewidth()
            tester.step_counter += 1
            return result

        def wrapped_eliminate(self, elim_labels):
            print(f"Eliminate operation called")
            print(f"  Factor labels before elimination: {self.labels}")
            print(f"  Labels to eliminate: {elim_labels}")
            result = tester.original_eliminate(self, elim_labels)
            print(f"  Factor labels after elimination: {result.labels}")
            tester.check_tensor_size(result.tensor)
            tester.check_memory_usage()
            tester.check_treewidth()
            tester.step_counter += 1
            return result

        FastFactor.__mul__ = wrapped_multiply
        FastFactor.eliminate = wrapped_eliminate


    def unwrap_fast_factor_methods(self):
        FastFactor.__mul__ = self.original_multiply
        FastFactor.eliminate = self.original_eliminate

    def test_elimination(self, elim_vars=None, up_to=None, through=None, all=False):
        self.wrap_fast_factor_methods()

        try:
            result = self.fastgm.eliminate_variables(elim_vars, up_to, through, all)
        except MemoryError as e:
            print(f"Memory limit exceeded: {e}")
            self.report()
            raise
        finally:
            self.unwrap_fast_factor_methods()

        self.report()
        return result

    def check_tensor_size(self, tensor):
        tensor_size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
        self.max_tensor_size = max(self.max_tensor_size, tensor_size_mb)
        self.tensor_size_history.append(tensor_size_mb)
        if tensor_size_mb > self.max_tensor_size_mb:
            raise MemoryError(f"Tensor size ({tensor_size_mb:.2f} MB) exceeded limit of {self.max_tensor_size_mb} MB")
        print(f"Current tensor size: {tensor_size_mb:.2f} MB")

    def check_memory_usage(self):
        if torch.cuda.is_available() and self.device == 'cuda':
            # Check memory usage on the GPU
            memory_usage_mb = torch.cuda.memory_allocated() / 1024 / 1024
        else:
            # Check memory usage on the CPU
            memory_usage_mb = sum(tensor.element_size() * tensor.numel() for tensor in self.get_all_tensors()) / 1024 / 1024

        self.memory_usage_history.append(memory_usage_mb)
        if memory_usage_mb > self.max_memory_mb:
            raise MemoryError(f"Tensor memory usage ({memory_usage_mb:.2f} MB) exceeded limit of {self.max_memory_mb} MB")
        print(f"Current tensor memory usage: {memory_usage_mb:.2f} MB")

    def get_all_tensors(self):
        # Helper function to gather all tensors in memory
        return [obj for obj in gc.get_objects() if torch.is_tensor(obj)]

    def check_treewidth(self):
        if self.fastgm.buckets.values():
            current_width = max(len(bucket.factors) for bucket in self.fastgm.buckets.values())
        else:
            current_width = 0
        self.max_width = max(self.max_width, current_width)
        self.width_history.append(current_width)
        print(f"Current treewidth: {current_width}")

    def report(self):
        print(f"Maximum tensor size encountered: {self.max_tensor_size:.2f} MB")
        print(f"Maximum treewidth encountered: {self.max_width}")
        print("\nTensor size history (MB):")
        print(", ".join(f"{size:.2f}" for size in self.tensor_size_history))
        print("\nMemory usage history (MB):")
        print(", ".join(f"{size:.2f}" for size in self.memory_usage_history))
        print("\nTreewidth history:")
        print(", ".join(map(str, self.width_history)))

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 9))
        plt.subplot(3, 1, 1)
        plt.plot(self.tensor_size_history)
        plt.title("Tensor Size History")
        plt.ylabel("Size (MB)")
        
        plt.subplot(3, 1, 2)
        plt.plot(self.memory_usage_history)
        plt.title("Memory Usage History")
        plt.ylabel("Usage (MB)")
        
        plt.subplot(3, 1, 3)
        plt.plot(self.width_history)
        plt.title("Treewidth History")
        plt.ylabel("Width")
        plt.xlabel("Operation Step")
        plt.tight_layout()
        plt.show()

# test_message_gradient = lambda mg, message: (lambda product: product.eliminate(product.labels).tensor.item())(mg * message)

#%%

# jit_file = "/home/cohenn1/SDBE/Analysis/big_num_samples_experiement2/grid20x20.f2_iB_25_nSamples_10000_ecl_10_run_4/bucket-output-fn30/1000/nn_1000_0.jit"
# uai_file = "/home/cohenn1/SDBE/width20-30/grid20x20.f2.uai"
# idx = 30
# fastgm = FastGM(uai_file=uai_file, device='cuda')

# converted_nn = FastFactor.nn_to_FastFactor(jit_file, 30, fastgm, 'cuda')
# #%%
# mg, mess = fastgm.get_message_gradient(30)
# #%%
# test_message_gradient(mg, converted_nn)

# mg, message = fastgm.get_message_gradient(10)
# ord = wtminfill_order(fastgm.get_factors())
# fastgm.load_elim_order(ord)
# factors = fastgm.get_factors()
# newgm = FastGM(factors=factors)

# %%
