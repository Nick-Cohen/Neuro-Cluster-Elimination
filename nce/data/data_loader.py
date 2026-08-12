from nce.inference import FastBucket
import torch
import torch.nn.functional as F
from .data_preprocessor import DataPreprocessor
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

class DataLoader:
    """DataLoader for sampling and loading training data for bucket message approximation.

    This class handles:
    - Sampling assignments from the message scope (uniform or all)
    - Computing message values at sampled assignments
    - Computing backward message values if bw_factors is set
    - Normalizing data using the DataPreprocessor

    Args:
        bucket: FastBucket being trained
        sample_generator: SampleGenerator for sampling assignments
        data_preprocessor: DataPreprocessor for normalization
    """
    def __init__(self, bucket: FastBucket, sample_generator=None, data_preprocessor=None):
        self.bucket = bucket
        self.lower_dim = self.bucket.gm.lower_dim
        self.sample_generator = sample_generator
        self.message_size = self.sample_generator.message_size
        self.data_preprocessor = data_preprocessor

        # Backward factors - set externally by get_backward_message() via bucket.compute_message_nn()
        self.bw_factors = None

        # WMB residual learning: forward WMB base factors for this cluster, set
        # externally by bucket.compute_message_nn() when config['wmb_residual'].
        # When set, load() appends the base value (in normalized units) as one
        # extra trailing column of x, consumed by nce.neural_networks.net.WMBResidualNet.
        self.base_factors = None
        # Diagnostics filled in by load(): stats of the log10 residual r = y - base.
        self.residual_stats = None
        
    # def __len__(self):
    #     """Required: Returns the total number of samples"""
    #     return len(self.assignments)
    
    def __getitem__(self, idx):
        if self.bw_hat is not None:
            return {
                'input': self.assignments[idx],
                'target': self.values[idx],
                'bw_hat': self.bw_hat[idx]
            }
        return {
            'input': self.assignments[idx],
            'target': self.values[idx]
        }
        
    def shuffle_data(self):
        indices = torch.randperm(len(self))
    
    def load(self, num_samples: int = 0, all: bool = False, is_validation: bool = False) -> tuple:
        """Load training data by sampling and computing message/backward values.

        Args:
            num_samples: Number of samples to generate (ignored if all=True)
            all: If True, enumerate all assignments instead of sampling
            is_validation: If True, use validation seed (different from training samples)

        Returns:
            Tuple of (x, y, bw) where:
            - x: One-hot encoded assignments (num_samples, input_dim)
            - y: Normalized message values (num_samples,) in natural log space
            - bw: Backward message values (num_samples,) in natural log space, or None
        """
        if self.sample_generator is None:
            raise ValueError("No sample generator provided")

        # Sample assignments
        if all:
            assignments = self.sample_generator.sample_assignments(sampling_scheme='all')
        else:
            assignments = self.sample_generator.sample_assignments(num_samples, is_validation=is_validation)

        # Compute message values
        mess_values = self.sample_generator.compute_message_values(assignments)

        # Compute backward message values if bw_factors or bw_modifier is set
        bw_values = None
        if self.bw_factors is not None:
            # Use compute_backward_values (no marginalization, just factor evaluation)
            bw_values = self.sample_generator.compute_backward_values(assignments, backward_factors=self.bw_factors)
        elif hasattr(self, 'bw_modifier') and self.bw_modifier is not None:
            # Full data batch mode: bw_modifier is a single factor
            bw_values = self.sample_generator.compute_backward_values(assignments, backward_factors=[self.bw_modifier])

        # Normalize using preprocessor (converts both to natural log space)
        normalized_y, normalized_bw = self.data_preprocessor.normalize(mess_values, bw_values)

        # One-hot encode assignments
        x = self.data_preprocessor.one_hot_encode(self.bucket, assignments)

        # WMB residual learning: append the WMB base value as an extra trailing
        # column of x, converted into the SAME normalized units the network
        # outputs, so that WMBResidualNet can simply add it to the net output.
        #
        #   minmax_01:      y_norm = (y*ln10 - ln_min) / ln_range
        #   logspace_mean:  y_norm = y*ln10 - normalizing_constant
        #
        # In both cases y_norm = residual_norm + base*ln10/scale with
        # scale = ln_range (minmax_01) or 1 (logspace_mean), and the network's
        # own undo_normalization() then maps its raw output back to exactly
        # r = log10(exact) - log10(wmb), i.e. the residual factor.
        if self.base_factors is not None:
            import math
            base_values = self.sample_generator.sample_tensor_product(self.base_factors, assignments)
            dp = self.data_preprocessor
            ln10 = math.log(10.0)
            if dp.normalization_mode == 'minmax_01':
                scale = dp.ln_range
            else:
                scale = 1.0
            base_col = (base_values * (ln10 / scale)).reshape(-1, 1).to(dtype=x.dtype, device=x.device)
            x = torch.cat([x, base_col], dim=1)
            # Diagnostics: the log10 residual actually being learned.
            with torch.no_grad():
                r = mess_values.reshape(-1) - base_values.reshape(-1)
                fin = torch.isfinite(r)
                if fin.any():
                    rf = r[fin]
                    yf = mess_values.reshape(-1)[fin]
                    self.residual_stats = {
                        'r_max': float(rf.max()), 'r_min': float(rf.min()),
                        'r_mean': float(rf.mean()),
                        'r_std': float(rf.std()) if rf.numel() > 1 else 0.0,
                        'sd_y': float(yf.std()) if yf.numel() > 1 else 0.0,
                        'y_range': float(yf.max() - yf.min()),
                        'n': int(rf.numel()),
                        'frac_positive': float((rf > 1e-6).float().mean()),
                    }

        return x, normalized_y, normalized_bw

    def load_batches(self, batch_size: int, num_batches: int, stratify_samples: bool = False) -> list:
        """Load multiple batches of training data.

        Args:
            batch_size: Number of samples per batch
            num_batches: Number of batches to generate
            stratify_samples: If True, distribute the top num_batches samples
                             (by y value) so each batch gets exactly one

        Returns:
            List of batch dicts with 'x', 'y', 'bw' keys
        """
        num_samples = num_batches * batch_size
        x, y, bw = self.load(num_samples)

        if stratify_samples and num_batches > 1:
            # Find indices of top num_batches samples by y value
            top_k = min(num_batches, len(y))
            top_indices = torch.topk(y, top_k).indices

            # Create a mask for non-top samples
            all_indices = torch.arange(len(y), device=y.device)
            mask = torch.ones(len(y), dtype=torch.bool, device=y.device)
            mask[top_indices] = False
            other_indices = all_indices[mask]

            # Shuffle other indices to randomize batch assignment
            other_indices = other_indices[torch.randperm(len(other_indices), device=y.device)]

            batches = []
            other_idx = 0
            samples_per_batch_other = batch_size - 1  # One slot reserved for top sample

            for i in range(num_batches):
                # Get the top sample for this batch
                top_idx = top_indices[i] if i < len(top_indices) else None

                # Get other samples for this batch
                end_other = other_idx + samples_per_batch_other
                batch_other_indices = other_indices[other_idx:end_other]
                other_idx = end_other

                # Combine: top sample first, then others
                if top_idx is not None:
                    batch_indices = torch.cat([top_idx.unsqueeze(0), batch_other_indices])
                else:
                    batch_indices = batch_other_indices

                batch = {
                    'x': x[batch_indices],
                    'y': y[batch_indices],
                    'bw': bw[batch_indices] if bw is not None else None
                }
                batches.append(batch)
            return batches
        else:
            # Original behavior: sequential split
            batches = []
            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size
                batch = {
                    'x': x[start:end],
                    'y': y[start:end],
                    'bw': bw[start:end] if bw is not None else None
                }
                batches.append(batch)
            return batches

    def load_all(self, all: bool = True) -> list:
        """Load all assignments as a single batch.

        Args:
            all: Must be True (kept for API compatibility)

        Returns:
            List with single batch dict containing all data
        """
        x, y, bw = self.load(all=True)
        return [{
            'x': x,
            'y': y,
            'bw': bw
        }]

