import torch
import math
import numpy as np
from typing import List, Tuple
from nce.inference.graphical_model import FastGM
from nce.inference.bucket import FastBucket
from nce.inference.factor import FastFactor
from nce.inference.message_gradient_factors import get_wmb_message_gradient_factors
import copy

class SampleGenerator:
    def __init__(self, gm: FastGM, bucket: FastBucket, random_seed=None):
        self.config = gm.config
        self.gm = gm
        self.iB = gm.iB
        self.bucket = bucket
        self.random_seed = random_seed if random_seed is not None else 0
        self.factors = bucket.factors
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
                dtype=torch.long
            )
            samples.append(column_samples)
        return torch.stack(samples, dim=1)
    
    def get_message_scope_and_dims(self) -> Tuple[List[int], torch.Tensor]:
        scope = set()
        for factor in self.bucket.factors:
            scope = scope.union(factor.labels)
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
        return self.sample_tensor_product(factors=factors, assignments=assignments)

    def sample_tensor_product_elimination(self, factors, assignments) -> torch.Tensor:
        for factor in factors:
            factor.order_indices()
        unsummed_shape = (*self.elim_domain_sizes,)
        unsummed_values = torch.zeros((len(assignments),) + unsummed_shape, device=self.gm.device, requires_grad=False)
        for fast_factor in factors:
            # If this is a FactorNN with bw_inv, convert to exact first to apply inverse transformation
            if hasattr(fast_factor, 'is_nn') and fast_factor.is_nn and hasattr(fast_factor, 'bw_inv') and fast_factor.bw_inv:
                fast_factor = fast_factor.to_exact()
            unsummed_values += fast_factor._get_slices(assignments=assignments, elim_vars=self.elim_vars, elim_domain_sizes = self.elim_domain_sizes, message_scope=self.message_scope)
        return torch.logsumexp(unsummed_values * math.log(10), dim=tuple(range(1, unsummed_values.dim()))) / math.log(10)

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
