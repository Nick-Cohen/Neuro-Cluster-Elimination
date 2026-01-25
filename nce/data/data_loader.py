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

        return x, normalized_y, normalized_bw

    def load_old(self, num_samples=0, all=False, scaling_factor=None):
        """OLD VERSION - kept for reference."""
        # generate the samples with the sample generator
        if self.sample_generator is None:
            raise ValueError("No sample generator provided")
        if all:
            assignments = self.sample_generator.sample_assignments(sampling_scheme='all')
        else:
            assignments = self.sample_generator.sample_assignments(num_samples) # config in sg gives sample scheme
        mess_values = self.sample_generator.compute_message_values(assignments) #debug
        assert not mess_values.requires_grad

        # Compute backward message values if bw_factors or bw_modifier is set
        # These will be passed to the loss function, NOT used to modify the target
        bw_message_values = None

        # Check for batched mode (bw_factors is a list)
        if hasattr(self, 'bw_factors') and self.bw_factors is not None:
            # Batched mode: use sample_tensor_product_elimination to properly marginalize factors
            # bw_factors is a list of FastFactors
            bw_message_values = self.sample_generator.sample_tensor_product_elimination(self.bw_factors, assignments)

            # sample_tensor_product returns 1D tensor, already in correct shape
            assert bw_message_values.dim() == 1, "sample_tensor_product should return 1D tensor"

        # Backward compatibility: support bw_modifier (single factor) for full_data_batch mode
        elif hasattr(self, 'bw_modifier') and self.bw_modifier is not None:
            # Full data batch mode: use _get_values (materializes full tensor)
            # bw_modifier is a FastFactor - use _get_values with only its own labels
            # This properly handles cases where bw_modifier has smaller scope than message_scope
            message_scope = self.sample_generator.message_scope

            # Only project assignments onto variables that exist in bw_modifier
            # This handles the case where bw_modifier scope is subset of message_scope
            bw_labels = sorted(self.bw_modifier.labels)

            if set(bw_labels) != set(message_scope):
                # bw_modifier has different scope - only use variables it contains
                # Get indices in message_scope for variables in bw_modifier
                bw_assignment_indices = [i for i, label in enumerate(message_scope) if label in bw_labels]
                projected_assignments = assignments[:, bw_assignment_indices]

                # Now use bw_labels as the "message_scope" for _get_values
                bw_message_values = self.bw_modifier._get_values(projected_assignments, bw_labels)
            else:
                bw_message_values = self.bw_modifier._get_values(assignments, message_scope)

            # Flatten bw_message_values to match mess_values shape if needed
            if bw_message_values is not None and bw_message_values.dim() > 1:
                bw_message_values = bw_message_values.squeeze()

        # Determine what to pass as bw_hat to the loss function
        # Only compute/use backward messages if use_bw_approx is enabled
        if self.bucket.gm.config.get('use_bw_approx', False):
            if bw_message_values is not None:
                # use_bw_approx mode: pass backward message as bw_hat
                bw_hat_for_loss = bw_message_values
            else:
                # Fallback: compute gradient values if bw_approx is enabled but no pre-computed values
                bw_hat_for_loss = self.sample_generator.compute_gradient_values(assignments)
        else:
            # use_bw_approx is False: don't compute or use backward messages at all
            bw_hat_for_loss = None

        # format the samples with the data preprocessor
        # Only normalize forward message values (y_vals), backward messages (bw_hat_for_loss) are used as-is
        normalized_y_vals, _ = self.data_preprocessor._normalize_logspace2(y_vals=mess_values, mg_vals=bw_hat_for_loss, scaling_factor=scaling_factor)
        # Return backward messages WITHOUT normalization (ignore the normalized version from _normalize_logspace2)
        return self.data_preprocessor.one_hot_encode(self.bucket, assignments), normalized_y_vals, bw_hat_for_loss
    
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

    def load_batches_old(self, batch_size, num_batches, scaling_factor=None):
        """OLD VERSION - kept for reference."""
        num_samples = num_batches * batch_size
        data = self.load_old(num_samples, scaling_factor=scaling_factor)
        batches = []
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            batch = {
                'x': data[0][start:end],
                'y': data[1][start:end],
                'mgh': data[2][start:end] if data[2] is not None else None
            }
            batches.append(batch)
        return batches

    def load_all_old(self, num_samples=0, grad_informed=True, all=True, scaling_factor=None):
        """OLD VERSION - kept for reference."""
        data = self.load_old(num_samples, all, scaling_factor=scaling_factor)
        return [{
            'x': data[0],
            'y': data[1],
            'mgh': data[2] if data[2] is not None else None
        }]
    
def create_data_loaders(signatures, values, bw_hat=None, batch_size=32, split_point=0.8):
    
    #debug
    device = 'cpu'
    
    if bw_hat is not None:
        bw_hat_arg = bw_hat.to(device)
    else:
        bw_hat_arg = None
    dataset = Data(signatures.to(device), values.to(device), bw_hat_arg)
    
    # Calculate split sizes (e.g., 80% train, 20% validation)
    train_size = int(split_point * len(dataset))
    val_size = len(dataset) - train_size
    
    # Random split
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # debug
    train_dataset
    val_dataset
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader

def shuffle_batches(set_batches):
    """Shuffle samples across batches.

    Args:
        set_batches: List of batch dicts with 'x', 'y', 'bw' keys

    Returns:
        New list of batch dicts with shuffled samples
    """
    # Concatenate all data into one large batch
    x_all = torch.cat([batch['x'] for batch in set_batches], dim=0)
    y_all = torch.cat([batch['y'] for batch in set_batches], dim=0)

    # Handle bw (may be None)
    has_bw = set_batches[0].get('bw') is not None
    if has_bw:
        bw_all = torch.cat([batch['bw'] for batch in set_batches], dim=0)

    # Get the total number of samples
    num_samples = x_all.shape[0]

    # Generate shuffled indices using PyTorch (efficient on GPU)
    shuffled_indices = torch.randperm(num_samples, device=x_all.device)

    # Shuffle the data
    x_all = x_all[shuffled_indices]
    y_all = y_all[shuffled_indices]
    if has_bw:
        bw_all = bw_all[shuffled_indices]

    # Determine the batch size from the original batches
    batch_size = set_batches[0]['x'].shape[0]

    # Split back into batches
    new_set_batches = []
    for i in range(0, num_samples, batch_size):
        new_set_batches.append({
            'x': x_all[i:i + batch_size],
            'y': y_all[i:i + batch_size],
            'bw': bw_all[i:i + batch_size] if has_bw else None
        })

    return new_set_batches
