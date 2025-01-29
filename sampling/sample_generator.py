import torch
import math
import numpy as np
from typing import *
from inference.graphical_model import FastGM
from inference.bucket import FastBucket
from inference.factor import FastFactor
import copy

class SampleGenerator:
    def __init__(self, gm: FastGM, bucket: FastBucket, mess = None, mg = None, random_seed =None):
        self.config = gm.config
        self.gm = gm
        self.iB = gm.iB
        self.bucket = bucket
        self.mess = mess
        self.mg = mg
        self.random_seed = random_seed
        self.factors = bucket.factors
        self.num_samples = self.config['num_samples']
        self.sampling_scheme = self.config['sampling_scheme']
        self.message_scope, self.domain_sizes = self.get_message_scope_and_dims()
        self.message_size = np.prod([d.item() for d in self.domain_sizes])
        if bucket.approximate_downstream_factors is not None:
            # only consider factors that have a scope that intersects with the message scope
            self.gradient_factors = [factor for factor in bucket.approximate_downstream_factors if bool(set(factor.labels) & set(self.message_scope))]
        else:
            self.gradient_factors = None
        for factor in self.factors:
            factor.order_indices()
        if self.gradient_factors is not None:
            for factor in self.gradient_factors:
                factor.order_indices()
        self.elim_vars = sorted(self.bucket.elim_vars, key=lambda v: v.label)
        self.elim_domain_sizes = [v.states for v in self.elim_vars]
        
    def sample_assignments(self, num_samples: int = -1, sampling_scheme = None) -> torch.Tensor:
        self.random_seed += 1
        seed = self.random_seed
        if seed is not None:
            torch.manual_seed(seed)
            if self.gm.device == 'cuda':
                torch.cuda.manual_seed(seed)
        if sampling_scheme is None:
            sampling_scheme = self.sampling_scheme
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
        if sampling_scheme == 'uniform':
            return self.sample_uniform(num_samples)
        elif sampling_scheme == 'mg':
            return self.sample_from_mg_brute_force(self.mess, self.mg, num_samples)
        elif sampling_scheme == 'mess_times_mg':
            return self.sample_from_mess_times_mg_brute_force(self.mess, self.mg, num_samples)
        elif sampling_scheme == 'all':
            return self.sample_all()
        elif type(sampling_scheme) == tuple:
            samples = []
            for (scheme, ratio) in sampling_scheme:
                samples.append(self.sample_assignments(int(num_samples * ratio), self.random_seed, scheme))
            return torch.cat(samples, dim=0)
    
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
    
    def sample_from_mg_brute_force(self, mess: FastFactor, mg: FastFactor, num_samples: int, replacement = True) -> torch.Tensor:
        # normalize the message gradient to be a probability distribution
        # first ensure mg has every variable in it
        mess_copy = copy.deepcopy(mess)
        mess_copy.tensor = torch.zeros_like(mess_copy.tensor)
        mg_expanded = mess_copy * mg
        mg_expanded.order_indices()
        # convert tensor to base e
        ln10 = torch.tensor(math.log(10), device=self.gm.device)
        mg_expanded.tensor *= ln10
        logsumexp = torch.logsumexp(mg_expanded.tensor.reshape(-1), dim=0)
        dist = torch.exp(mg_expanded.tensor - logsumexp)
        # ensure the distribution is normalized
        assert(torch.allclose(dist.sum(), torch.tensor(1.0, device=self.gm.device)), f"Sum of distribution is {dist.sum()}")
        samples = torch.multinomial(dist.flatten(), num_samples, replacement=replacement)
        samples = SampleGenerator._unravel_index(samples, dist.shape)
        return samples

    def sample_from_mess_times_mg_brute_force(self, mess: FastFactor, mg: FastFactor, num_samples: int, replacement = True) -> torch.Tensor:
        # normalize the message gradient to be a probability distribution
        dist = mess * mg
        # convert tensor to base e
        ln10 = torch.tensor(math.log(10), device=self.gm.device)
        dist.tensor *= ln10
        logsumexp = torch.logsumexp(dist.tensor.reshape(-1), dim=0)
        dist = torch.exp(dist.tensor - logsumexp)
        samples = torch.multinomial(dist.flatten(), num_samples, replacement=replacement)
        samples = SampleGenerator._unravel_index(samples, dist.shape)
        return samples
        
    def sample_uniform(self, num_samples) -> torch.Tensor:
        # dtype long used for compatibility with indexing
        samples = []
        for domain_size in self.domain_sizes:
            column_samples = torch.randint(
                low=0, 
                high=domain_size.item(), 
                size=(num_samples,), 
                dtype=torch.long
            )
            samples.append(column_samples)
        return torch.stack(samples, dim=1)
    
    def get_message_scope_and_dims(self) -> Tuple[List[int], torch.Tensor]:
        scope = set()
        for factor in self.bucket.factors:
            scope = scope.union(factor.labels)
        scope.discard(self.bucket.label)
        return sorted(list(scope)), torch.tensor([self.gm.vars[self.gm.matching_var(v)].states for v in scope])
    
    def compute_message_values(self, assignments: torch.Tensor) -> torch.Tensor:
        factors = self.factors
        return self.sample_tensor_product_elimination(self.factors, assignments)
    
    def compute_gradient_values(self, assignments: torch.Tensor, gradient_factors=None) -> torch.Tensor:
        if gradient_factors is None:
            factors = self.gradient_factors
        else:
            factors = gradient_factors
        return self.sample_tensor_product(factors, assignments)
    
    # TODO: Combine factor slices in pairs to reduce number of tensor operations
    def sample_tensor_product_elimination(self, factors, assignments) -> torch.Tensor:
        for factor in factors:
            factor.order_indices()
        unsummed_shape = (*self.elim_domain_sizes,)
        unsummed_values = torch.zeros((len(assignments),) + unsummed_shape, device=self.gm.device)
        for fast_factor in factors:
            if True:
                unsummed_values += fast_factor._get_slices(assignments=assignments, elim_vars=self.elim_vars, elim_domain_sizes = self.elim_domain_sizes, message_scope=self.message_scope)
            if False:
                tensor = fast_factor.tensor
                tensor_labels = fast_factor.labels
                
                # indices in assignments that correspond to dimensions in the tensor
                assignment_indices = [i for i, idx in enumerate(scope) if idx in tensor_labels]
                
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
                permuted_assignment_indices = [i for i, idx in enumerate(scope) if idx in tensor_labels]
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
                unsummed_values += reshaped_slices.expand(expanded_slice_shape) # it will broadcast over dimensions not in the factor
        return torch.logsumexp(unsummed_values * math.log(10), dim=tuple(range(1, unsummed_values.dim()))) / math.log(10)

    def sample_tensor_product(self, factors, assignments) -> torch.Tensor:
        
        if len(factors) == 1 and factors[0].labels == [] or len(factors) == 0:
            return torch.zeros(len(assignments), device=self.gm.device)
        for factor in factors:
            factor.order_indices()
        scope = self.message_scope
        elim_vars = self.elim_vars
        unsummed_shape = (1,)
        unsummed_values = torch.zeros((len(assignments),) + unsummed_shape, device=self.gm.device)
        for fast_factor in factors:
            tensor = fast_factor.tensor
            tensor_labels = fast_factor.labels
            
            # indices in assignments that correspond to dimensions in the tensor
            assignment_indices = [i for i, idx in enumerate(scope) if idx in tensor_labels]
            
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
            permuted_assignment_indices = [i for i, idx in enumerate(scope) if idx in tensor_labels]
            projected_assignments = assignments[:,permuted_assignment_indices]

            # stretch out elimination indices to 1d
            # grab slices corresponding to slices
            view = tuple(int(dim) for dim in tensor.shape[:len(tensor.shape) - len(elim_vars)]) + (int(torch.prod(torch.tensor(tensor.shape[len(tensor.shape) - len(elim_vars):]))),)
            if not projected_assignments.numel() == 0:
                slices = tensor.view(view)[tuple(projected_assignments.t())]
            else:
                slices = tensor.unsqueeze(0).expand(len(assignments), len(tensor))
            
            # reshape slices to match elimination variables in order, e.g. (1,2,2,1) if 2nd and 3rd variables are in tensor
            unexpanded_slice_shape = (len(assignments),) + tuple([v.states if v.label in tensor_labels else 1 for v in elim_vars])
            reshaped_slices = slices.reshape(unexpanded_slice_shape)
            expanded_slice_shape = (len(assignments),) + tuple([1 for v in elim_vars])
            unsummed_values += reshaped_slices.expand(expanded_slice_shape)
        return torch.logsumexp(unsummed_values * math.log(10), dim=tuple(range(1, unsummed_values.dim()))) / math.log(10)
        
            
        
            
            
            
            
            
            