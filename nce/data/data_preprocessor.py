import torch
import torch.nn.functional as F
from typing import List, Tuple
from nce.inference.bucket import FastBucket

class DataPreprocessor:
    """Preprocessor for normalizing message and backward message data.

    Normalization depends on whether backward message (bw) is used:

    WITHOUT bw (use_bw_approx=False):
        normalizing_constant = logsumexp(y) - log(N)  (the training mean)

    WITH bw (use_bw_approx=True):
        normalizing_constant = logsumexp(y + bw) - logsumexp(bw) - log(N)
        bw_normalizing_constant = bw[argmax(y + bw)]  (passed to loss function)

    The normalizing constant is computed lazily on first normalize() call,
    using the actual training samples for accuracy.

    Args:
        y: Initial message samples in log10 space (or None for deferred init)
        bw: Initial backward message samples in log10 space (or None if not using backward)
        lower_dim: If True, use n-1 dimensional one-hot encoding
        device: Torch device
        use_bw_approx: If True, use bw-aware normalization
    """
    def __init__(self, y: torch.Tensor = None, bw: torch.Tensor = None, lower_dim: bool = True,
                 device=None, use_bw_approx: bool = False) -> None:
        self.y = y
        self.bw = bw
        if device is None:
            self.device = y.device if y is not None else 'cpu'
        else:
            self.device = device
        self.lower_dim = lower_dim
        self.use_bw_approx = use_bw_approx

        # Normalizing constant (in natural log space) - computed lazily
        self.normalizing_constant = None

        # For use_bw_approx mode: bw value at argmax(y + bw), passed to loss function
        self.bw_normalizing_constant = None

        # Global max of targets for UKL numerical stability (computed from full training data)
        # CRITICAL: This must be computed ONCE from all training data and used for ALL batches
        # Using per-batch max causes gradient inconsistency and training divergence
        self.global_max_targets = None

        # Initialize normalizing constant from provided samples if available
        if y is not None:
            self._initialize_normalizing_constant(y, bw)

    def _initialize_normalizing_constant(self, y_vals: torch.Tensor, bw_vals: torch.Tensor = None):
        """Compute normalizing constant from training samples.

        WITHOUT bw (use_bw_approx=False):
            normalizing_constant = logsumexp(y) - log(N)  (the training mean)

        WITH bw (use_bw_approx=True):
            normalizing_constant = logsumexp(y + bw) - logsumexp(bw) - log(N)
            bw_normalizing_constant = bw[argmax(y + bw)]

        Args:
            y_vals: Message values in log10 space
            bw_vals: Backward message values in log10 space (or None)
        """
        # Check for NaN in inputs
        if torch.isnan(y_vals).any():
            print(f"[DataPreprocessor] ERROR: NaN in y_vals input to _initialize_normalizing_constant!")

        ln10 = torch.log(torch.tensor(10.0)).to(self.device)
        N = torch.tensor(float(len(y_vals)), device=self.device)

        # Convert to natural log space
        y_ln = y_vals * ln10

        if self.use_bw_approx and bw_vals is not None:
            # WITH bw: normalize by logsumexp(y + bw) - logsumexp(bw) - log(N)
            bw_ln = bw_vals * ln10
            combined = y_ln + bw_ln

            logsumexp_combined = torch.logsumexp(combined, dim=0)
            logsumexp_bw = torch.logsumexp(bw_ln, dim=0)
            self.normalizing_constant = logsumexp_combined - logsumexp_bw - torch.log(N)

            # Also store bw value at argmax(y + bw) for the loss function
            max_idx = torch.argmax(combined)
            self.bw_normalizing_constant = bw_ln[max_idx]
        else:
            # WITHOUT bw: normalize by logsumexp(y) - log(N) (the training mean)
            self.normalizing_constant = torch.logsumexp(y_ln, dim=0) - torch.log(N)

    def reinitialize_with_backward(self, y_vals: torch.Tensor, bw_vals: torch.Tensor):
        """Reinitialize normalizing constant with backward message values.

        Call this after bw_factors are set to update the normalizing constant
        to use y[argmax(y + bw)] instead of just max(y).

        Args:
            y_vals: Message values in log10 space
            bw_vals: Backward message values in log10 space
        """
        self.use_bw_approx = True
        self._initialize_normalizing_constant(y_vals, bw_vals)

    def normalize(self, y_vals: torch.Tensor, bw_vals: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize message and backward message values.

        On first call, computes the normalizing constant from the provided samples.
        This enables lazy initialization where normalization happens after all
        training samples are generated.

        Args:
            y_vals: Message values in log10 space
            bw_vals: Backward message values in log10 space (or None)

        Returns:
            Tuple of (normalized_y, normalized_bw) where both are in natural log space.
            normalized_y is centered (normalizing constant subtracted).
            normalized_bw is converted to natural log but NOT centered (used as-is by loss functions).
        """
        # Lazy initialization: compute normalizing constant on first call
        if self.normalizing_constant is None:
            self._initialize_normalizing_constant(y_vals, bw_vals)

        ln10 = torch.log(torch.tensor(10.0)).to(self.device)

        # Convert message to natural log space and subtract normalizing constant
        y_ln = y_vals * ln10
        y_normalized = y_ln - self.normalizing_constant

        # Convert backward message to natural log space (but don't subtract normalizing constant)
        # Loss functions like unnormalized_kl add outputs + bw, so they need to be in same base
        if bw_vals is not None:
            bw_normalized = bw_vals * ln10
        else:
            bw_normalized = None

        # Debug: check for NaN
        if torch.isnan(y_normalized).any():
            print(f"[DataPreprocessor] WARNING: NaN in y_normalized!")
            print(f"  normalizing_constant={self.normalizing_constant}")
            print(f"  y_ln: min={y_ln.min()}, max={y_ln.max()}, has_nan={torch.isnan(y_ln).any()}")
        if bw_normalized is not None and torch.isnan(bw_normalized).any():
            print(f"[DataPreprocessor] WARNING: NaN in bw_normalized!")
            print(f"  bw_vals: min={bw_vals.min()}, max={bw_vals.max()}, has_nan={torch.isnan(bw_vals).any()}")

        return y_normalized, bw_normalized

    def undo_normalization(self, outputs: torch.Tensor) -> torch.Tensor:
        """Convert normalized outputs back to log10 space.

        Args:
            outputs: Normalized outputs from neural network (natural log space, centered)

        Returns:
            Values in log10 space
        """
        ln10 = torch.log(torch.tensor(10.0)).to(outputs.device)

        # Add back normalizing constant
        outputs = outputs + self.normalizing_constant

        # Convert back to log10 space
        outputs = outputs / ln10

        return outputs

    def one_hot_encode(self, bucket: FastBucket, assignments: torch.IntTensor) -> torch.Tensor:
        """One-hot encode variable assignments.

        Args:
            bucket: FastBucket containing variable domain information
            assignments: Tensor of shape (num_samples, num_vars) with integer assignments

        Returns:
            One-hot encoded tensor
        """
        domain_sizes = bucket.get_message_dimension()
        num_samples, num_vars = assignments.shape

        if self.lower_dim:  # send n domain variables to n-1 vector
            one_hot_encoded_samples = torch.cat(
                [F.one_hot(assignments[:, i], num_classes=domain_sizes[i])[:, 1:]
                 for i in range(num_vars)], dim=-1)
        else:
            one_hot_encoded_samples = torch.cat(
                [F.one_hot(assignments[:, i], num_classes=domain_sizes[i])
                 for i in range(num_vars)], dim=-1)
        return one_hot_encoded_samples.float().to(self.device)


# ============================================================================
# OLD VERSION - kept for reference
# ============================================================================
class DataPreprocessor_old:
    def __init__(self, y: torch.Tensor, mg: torch.Tensor, is_logspace: bool, lower_dim = True, device = None, fdb: bool = False) -> None:
        self.y = y
        self.mg = mg
        if device is None:
            self.device = mg.device
        else:
            self.device = device
        self.is_logspace = is_logspace
        self.lower_dim = lower_dim
        self.fdb = fdb  # Mean-based normalization flag (works with full or sampled data)

        # Original normalization parameters
        self.y_max = None
        self.mg_max = None
        self.y_mean = None  # For linear space
        self.y_std = None   # For linear space
        self.mg_mean = None # For linear space scaling

        # FDB-specific normalization parameters (mean-based)
        self.fdb_y_mean = None
        self.fdb_mg_mean = None

        # Flag to track if normalization constants were initialized from sampled data
        self.initialized_from_samples = False

        # CRITICAL: Track scaling factors for proper reverse normalization
        self.mg_scaling_factor = None
        self.mg_scaling_type = None  # 'mean', 'std', or 'none'

        # NEW: Exponential preprocessing mode (for scaled_mse/linspace_mse)
        self.exp_preprocessing = False  # True when using exponential targets
        self.scaling_factor = None  # For scaled_mse only (sigma_f^2 / (sigma_f^2 + sigma_g^2))
        self.y_max_after_scaling = None  # Max in natural log-space AFTER scaling, before exponentiating

        # Initialize normalization constants from the provided samples if fdb=True
        # This ensures constants are available even before first load() call
        if self.fdb and self.is_logspace:
            self._initialize_fdb_constants_from_samples(y, mg)

    def _initialize_fdb_constants_from_samples(self, y_vals, mg_vals):
        """
        Initialize FDB normalization constants from sample data.

        This is called during __init__ when fdb=True to compute normalization
        constants from the initial sample data (typically 1000 samples).
        These constants are then used consistently throughout training and inference.

        Args:
            y_vals: Sample message values in log10 space
            mg_vals: Sample message gradient values in log10 space
        """
        # Convert to log base e
        ln10 = torch.log(torch.tensor(10.0)).to(self.device)

        # Compute mean of y values
        y_vals_ln = y_vals * ln10
        self.fdb_y_mean = y_vals_ln.mean()

        # Only compute mg-related constants if mg_vals is provided
        if mg_vals is not None:
            # Compute mean of mg values
            mg_vals_ln = mg_vals * ln10
            self.fdb_mg_mean = mg_vals_ln.mean()

            # Compute mg scaling factor
            # This is needed for proper denormalization of mg values
            # First center the mg values
            mg_centered = mg_vals_ln - self.fdb_mg_mean

            # Determine scaling type and factor
            mg_mean_after_centering = mg_centered.mean()

            if torch.abs(mg_mean_after_centering) < 1e-8:
                # Mean is essentially zero, use std for scaling
                mg_std_for_scaling = mg_centered.std()
                if mg_std_for_scaling > 1e-8:
                    self.mg_scaling_factor = mg_std_for_scaling
                    self.mg_scaling_type = 'std'
                else:
                    # Both mean and std are tiny - no scaling needed
                    self.mg_scaling_factor = torch.tensor(1.0, device=self.device)
                    self.mg_scaling_type = 'none'
            else:
                # Use the (small) mean for scaling
                self.mg_scaling_factor = mg_mean_after_centering
                self.mg_scaling_type = 'mean'
        else:
            # No mg_vals provided - set default values
            self.fdb_mg_mean = None
            self.mg_scaling_factor = None
            self.mg_scaling_type = None

        self.initialized_from_samples = True

        # Optional debug output
        if hasattr(self, 'debug') and self.debug:
            print(f"Initialized FDB constants from {len(y_vals)} samples:")
            print(f"  fdb_y_mean: {self.fdb_y_mean:.6f}")
            if mg_vals is not None:
                print(f"  fdb_mg_mean: {self.fdb_mg_mean:.6f}")
                print(f"  mg_scaling_factor: {self.mg_scaling_factor:.6f} (type: {self.mg_scaling_type})")
            else:
                print(f"  mg_vals: None (backward approx disabled)")

    def convert_data(self, mess_normalizing_constant = None, mg_normalizing_constant = None) -> Tuple[torch.Tensor]:
        if not self.is_logspace:
            return self._convert_to_lin_space()
        else:
            if self.fdb:
                return self._normalize_logspace_fdb(mess_normalizing_constant, mg_normalizing_constant)
            else:
                return self._normalize_logspace(mess_normalizing_constant, mg_normalizing_constant)
    
    def _normalize_logspace_fdb(self, mess_normalizing_constant = None, mg_normalizing_constant = None):
        """
        Mean-based normalization (operates on self.y and self.mg).

        NOTE: This method is deprecated in favor of _normalize_logspace2_fdb.
        It's kept for backward compatibility with convert_data() calls.
        """
        if mess_normalizing_constant is not None or mg_normalizing_constant is not None:
            raise ValueError("Normalizing constant changes are not yet implemented for FDB")

        self._normalize_logspace_message_fdb(mess_normalizing_constant)
        self._normalize_logspace_mg_fdb(mg_normalizing_constant)
        return self.y, self.mg

    def _normalize_logspace_message_fdb(self, normalizing_constant = None):
        """
        FDB message normalization: subtract mean instead of max (operates on self.y).

        If fdb_y_mean was already initialized from samples, it will be used.
        Otherwise, it will be computed from self.y.
        """
        # convert to log base e
        ln10 = torch.log(torch.tensor(10.0)).to(self.device)
        self.y = self.y * ln10

        # Use pre-initialized mean if available, otherwise compute from self.y
        if self.fdb_y_mean is None:
            self.fdb_y_mean = self.y.mean()

        self.y = self.y - self.fdb_y_mean

    def _normalize_logspace_mg_fdb(self, normalizing_constant=None):
        """
        FDB mg normalization: subtract mean and scale (operates on self.mg).

        If fdb_mg_mean and mg_scaling_factor were already initialized from samples,
        they will be used. Otherwise, they will be computed from self.mg.
        """
        # convert to log base e
        ln10 = torch.log(torch.tensor(10.0)).to(self.device)
        self.mg = self.mg * ln10

        # Use pre-initialized mean if available, otherwise compute from self.mg
        if self.fdb_mg_mean is None:
            self.fdb_mg_mean = self.mg.mean()

        self.mg = self.mg - self.fdb_mg_mean

        # Compute scaling factor if not already set
        if self.mg_scaling_factor is None:
            mg_mean_after_centering = self.mg.mean()

            # Handle edge cases for scaling
            if torch.abs(mg_mean_after_centering) < 1e-8:
                # Mean is essentially zero, use std for scaling
                mg_std_for_scaling = self.mg.std()
                if mg_std_for_scaling > 1e-8:
                    self.mg_scaling_factor = mg_std_for_scaling
                    self.mg_scaling_type = 'std'
                else:
                    # Both mean and std are tiny - no scaling needed
                    self.mg_scaling_factor = torch.tensor(1.0, device=self.device)
                    self.mg_scaling_type = 'none'
            else:
                # Use the (small) mean for scaling
                self.mg_scaling_factor = mg_mean_after_centering
                self.mg_scaling_type = 'mean'

        # Apply the scaling
        self.mg = self.mg / self.mg_scaling_factor

        assert torch.all(self.mg >= -torch.inf)  # Allow negative values with FDB
    
    def convert_back_message_logspace_fdb(self, outputs):
        """
        Convert back from FDB normalized logspace to original logspace
        """
        # Add back the mean instead of max
        outputs = outputs + self.fdb_y_mean
        # Convert back to log base 10
        outputs = outputs / torch.log(torch.tensor(10.0)).to(outputs.device)
        return outputs
    
    def convert_back_mg_logspace_fdb(self, bw_hat):
        # Step 1: Reverse the scaling 
        bw_hat = bw_hat * self.mg_scaling_factor
        # Step 2: Add back the mean
        bw_hat = bw_hat + self.fdb_mg_mean  
        # Step 3: Convert back to log base 10
        bw_hat = bw_hat / torch.log(torch.tensor(10.0))
    
    def _convert_to_lin_space(self, mess_normalizing_constant = None, mg_normalizing_constant = None):
        # update the max values if new data has larger max by factor of 32,000
        if mess_normalizing_constant is not None or mg_normalizing_constant is not None:
            raise ValueError("Normalizing constant changes are not yet implemented correctly")
        self.y_max = self.y.max() if mess_normalizing_constant is None or mess_normalizing_constant + 5 < self.y.max() else mess_normalizing_constant
        self.mg_max = self.mg.max() if mg_normalizing_constant is None or mg_normalizing_constant + 5 < self.mg.max() else mg_normalizing_constant
        
        # subtract the maxes
        self.y = self.y - self.y_max
        self.mg = self.mg - self.mg_max
        
        # exponentiate
        self.y = torch.pow(10, self.y)
        self.mg = torch.pow(10, self.mg)
        
        # make the mean 0 of just y, mg needs to keep proportionality
        self.y_mean = self.y.mean()
        self.y = self.y - self.y_mean
        
        # make the std 1 of just y
        self.y_std = self.y.std()
        self.y = self.y / self.y_std
        
        # make average value of mg to standardize mg loss weights to an average of 1
        self.mg_mean = self.mg.mean()
        self.mg = self.mg / self.mg_mean
        assert torch.all(self.mg >= 0)
        
        # to convert NN input back:
        # multiply by std, add mean, exponentiate, add max
        
        return self.y, self.mg
    
    def undo_normalization(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        Undo normalization based on preprocessing mode.

        This method reverses the normalization applied during preprocessing.
        Three modes are supported:
        1. Exponential preprocessing (exp_preprocessing=True)
        2. Mean-based normalization (fdb=True) - for batched learning
        3. Max-based normalization (legacy, fdb=False)
        """
        with torch.no_grad():
            if self.exp_preprocessing:
                # Reverse exponential preprocessing
                # Forward was: log10 → ln → scale → subtract_max → exp
                # Reverse is: log → add_max → unscale → convert_to_log10

                # 1. Take log (outputs are in linear space from NN)
                outputs = torch.log(outputs.clamp(min=1e-30))  # Clamp to avoid log(0)

                # 2. Add back the max (that was subtracted after scaling)
                outputs = outputs + self.y_max_after_scaling

                # 3. Reverse scaling if it was applied (for scaled_mse)
                if self.scaling_factor is not None:
                    outputs = outputs / self.scaling_factor

                # 4. Convert back to log base 10
                outputs = outputs / torch.log(torch.tensor(10.0)).to(outputs.device)
                return outputs

            elif self.fdb:
                # Mean-based normalization (for batched learning and full data batch)
                return self.convert_back_message_logspace_fdb(outputs)

            else:
                # Legacy max-based normalization
                # Check if normalization constants were initialized
                if self.y_max is None:
                    raise ValueError(
                        "Cannot undo normalization: y_max is None. "
                        "This usually means:\n"
                        "1. You're using mini-batch learning but mean-based normalization wasn't enabled\n"
                        "2. Or normalization was never performed (no data was loaded)\n"
                        "\nSolution: Ensure 'batch_size' is specified in your config.\n"
                        "   The system will automatically enable mean-based normalization for mini-batches.\n"
                        "\nExample config:\n"
                        "  config = {\n"
                        "      'batch_size': 512,           # ← Enables mini-batch mode automatically\n"
                        "      'sampling_scheme': 'uniform', # Mini-batch sampling\n"
                        "      # ... other parameters\n"
                        "  }\n"
                        f"\nCurrent settings: fdb={self.fdb}, exp_preprocessing={self.exp_preprocessing}, "
                        f"initialized_from_samples={self.initialized_from_samples}"
                    )

                outputs += self.y_max
                outputs /= torch.log(torch.tensor(10.0)).to(outputs.device)
                return outputs
    
    def convert_back_message_logspace(self, outputs):
        outputs = outputs * self.y_std
        outputs = outputs + self.y_mean
        outputs[outputs < 0] = 0
        outputs = torch.log10(outputs)
        outputs = outputs + self.y_max
        return outputs
    
    # just needed for testing
    def convert_back_mg_logspace(self, bw_hat):
        bw_hat = bw_hat * self.mg_mean
        bw_hat = torch.log10(bw_hat)
        return bw_hat
    
    def _normalize_logspace(self, mess_normalizing_constant = None, mg_normalizing_constant = None):
        if mess_normalizing_constant is not None or mg_normalizing_constant is not None:
            raise ValueError("Normalizing constant changes are not yet implemented")
        self._normalize_logspace_message(mess_normalizing_constant)
        self._normalize_logspace_mg(mg_normalizing_constant)
        return self.y, self.mg
    
    def _normalize_logspace2(self, y_vals, mg_vals, scaling_factor=None):
        """
        Updated _normalize_logspace2 to support FDB normalization and exponential preprocessing

        Args:
            y_vals: Message values in log10 space
            mg_vals: Message gradient values in log10 space
            scaling_factor: Optional scaling factor for exponential preprocessing
        """
        if self.exp_preprocessing:
            # Use exponential preprocessing (for scaled_mse/linspace_mse)
            # If scaling_factor is None but we have a stored one, use the stored value
            if scaling_factor is None and self.scaling_factor is not None:
                scaling_factor = self.scaling_factor
            return self._preprocess_exponential(y_vals, scaling_factor), mg_vals
        elif self.fdb:
            # Use FDB normalization (mean-based)
            return self._normalize_logspace2_fdb(y_vals, mg_vals)
        else:
            # Use original normalization (max-based)
            self._set_normalizing_constants(y_vals, mg_vals)
            return self._normalize_logspace_message2(y_vals), self._normalize_logspace_mg2(mg_vals)

    def _normalize_logspace2_fdb(self, y_vals, mg_vals):
        """
        FDB version of _normalize_logspace2 - uses mean instead of max
        """
        self._set_normalizing_constants_fdb(y_vals, mg_vals)
        return self._normalize_logspace_message2_fdb(y_vals), self._normalize_logspace_mg2_fdb(mg_vals)

    def _set_normalizing_constants_fdb(self, y_vals, mg_vals):
        """
        FDB version: set normalizing constants using mean.

        If constants were already initialized from samples in __init__,
        this method will NOT overwrite them (to ensure consistency).
        Otherwise, it will compute them from the provided data.
        """
        ln10 = torch.log(torch.tensor(10.0)).to(self.device)

        if self.fdb_y_mean is None:
            # Convert to log base e and calculate mean
            y_vals_ln = y_vals * ln10
            self.fdb_y_mean = y_vals_ln.mean()

        # Only compute mg constants if mg_vals is provided
        if mg_vals is not None:
            if self.fdb_mg_mean is None:
                # Convert to log base e and calculate mean
                mg_vals_ln = mg_vals * ln10
                self.fdb_mg_mean = mg_vals_ln.mean()

            # Also set mg_scaling_factor if not already set
            if self.mg_scaling_factor is None:
                mg_vals_ln = mg_vals * ln10
                mg_centered = mg_vals_ln - self.fdb_mg_mean
                mg_mean_after_centering = mg_centered.mean()

                if torch.abs(mg_mean_after_centering) < 1e-8:
                    mg_std_for_scaling = mg_centered.std()
                    if mg_std_for_scaling > 1e-8:
                        self.mg_scaling_factor = mg_std_for_scaling
                        self.mg_scaling_type = 'std'
                    else:
                        self.mg_scaling_factor = torch.tensor(1.0, device=self.device)
                        self.mg_scaling_type = 'none'
                else:
                    self.mg_scaling_factor = mg_mean_after_centering
                    self.mg_scaling_type = 'mean'

        return self.fdb_y_mean, self.fdb_mg_mean

    def _normalize_logspace_message2_fdb(self, y_vals):
        """
        FDB version: normalize y_vals using mean instead of max
        """
        # convert to log base e
        ln10 = torch.log(torch.tensor(10.0)).to(self.device)
        y_vals = y_vals * ln10
        
        # subtract the mean instead of max
        y_vals -= self.fdb_y_mean
        return y_vals

    def _normalize_logspace_mg2_fdb(self, mg_vals):
        """
        FDB version: normalize mg_vals using mean instead of max, then scale.

        Normalization process:
        1. Convert to log base e
        2. Subtract mean
        3. Scale by mg_scaling_factor (for consistent loss weighting)
        """
        # If mg_vals is None (backward approx disabled), return None
        if mg_vals is None:
            return None

        # convert to log base e
        ln10 = torch.log(torch.tensor(10.0)).to(self.device)
        mg_vals = mg_vals * ln10

        # subtract the mean
        mg_vals -= self.fdb_mg_mean

        # scale by the factor (if it was computed)
        if self.mg_scaling_factor is not None:
            mg_vals = mg_vals / self.mg_scaling_factor

        return mg_vals
    
    def _set_normalizing_constants2(self, y_vals, mg_vals):
        ln10 = torch.log(torch.tensor(10.0)).to(self.device)
        if self.y_max is None:
            self.y_max = y_vals.max() * ln10
        if mg_vals is not None and self.mg_max is None:
            self.mg_max = mg_vals.max() * ln10
        return self.y_max, self.mg_max
    def _set_normalizing_constants(self, y_vals, mg_vals):
        ln10 = torch.log(torch.tensor(10.0)).to(self.device)
        if self.y_max is None:
            self.y_max = y_vals.max() * ln10 - torch.log(torch.tensor([len(y_vals)])).to(self.device)
        if mg_vals is not None and self.mg_max is None:
            self.mg_max = mg_vals.max() * ln10 - torch.log(torch.tensor([len(mg_vals)])).to(self.device)
        return self.y_max, self.mg_max
    
    def _normalize_logspace_message(self, normalizing_constant = None):
        # convert to log base e
        ln10 = torch.log(torch.tensor(10.0)).to(self.device)
        self.y = self.y * ln10
        
        # calculate normalizing constant
        if normalizing_constant is None:
            self.y_max = self.y.max()
        elif normalizing_constant < self.y.max():
            self.y_max = self.y.max()
        else:
            self.y_max = normalizing_constant
        self.y -= self.y_max
        
    def _normalize_logspace_message2(self, y_vals):
        # convert to log base e
        ln10 = torch.log(torch.tensor(10.0)).to(self.device)
        y_vals = y_vals * ln10
        
        # calculate normalizing constant
        y_vals -= self.y_max
        return y_vals
    
    def _normalize_logspace_mg(self, normalizing_constant = None):
        # convert to log base e
        ln10 = torch.log(torch.tensor(10.0)).to(self.device)
        self.mg = self.mg * ln10
        
        # calculate normalizing constant
        if normalizing_constant is None:
            self.mg_max = self.mg.max()
        elif normalizing_constant < self.mg.max():
            self.mg_max = self.mg.max()
        else:
            self.mg_max = normalizing_constant
        self.mg -= self.mg_max
        
    def _normalize_logspace_mg2(self, mg_vals):
        # If mg_vals is None (backward approx disabled), return None
        if mg_vals is None:
            return None

        # convert to log base e
        ln10 = torch.log(torch.tensor(10.0)).to(self.device)
        mg_vals = mg_vals * ln10

        # calculate normalizing constant
        # In batched mode with use_bw_approx, mg_max might not match the scale of mg_vals
        # So recompute it from the current batch to avoid scale mismatch
        if self.mg_max is None or (hasattr(self, 'use_bw_approx') and self.use_bw_approx):
            # Recompute mg_max from current batch for consistent normalization
            batch_mg_max = mg_vals.max()
            mg_vals -= batch_mg_max
        else:
            mg_vals -= self.mg_max
        return mg_vals

    def _preprocess_exponential(self, y_vals, scaling_factor=None):
        """
        Exponential preprocessing for scaled_mse and linspace_mse:
        1. Convert to natural log (from log10)
        2. Apply scaling factor if provided (for scaled_mse only)
        3. Subtract max (computed AFTER scaling for numerical stability)
        4. Exponentiate to linear space

        Args:
            y_vals: Message values in log10 space
            scaling_factor: Optional scaling factor (sigma_f^2 / (sigma_f^2 + sigma_g^2))
                           None for linspace_mse, scalar for scaled_mse

        Returns:
            Exponential targets in linear space
        """
        # Step 1: Convert to log base e
        ln10 = torch.log(torch.tensor(10.0)).to(self.device)
        y_vals = y_vals * ln10

        # Step 2: Check if scaling_factor changed (should be consistent within a bucket)
        if self.scaling_factor is not None and scaling_factor is not None:
            if abs(self.scaling_factor - scaling_factor) > 1e-10:
                raise ValueError(
                    f"Scaling factor changed between batches! "
                    f"Previous: {self.scaling_factor}, New: {scaling_factor}. "
                    f"This indicates inconsistent preprocessing."
                )

        # Store scaling factor for reverse transformation (first time only)
        if self.scaling_factor is None:
            self.scaling_factor = scaling_factor

        # Step 3: Apply scaling if provided (for scaled_mse)
        if self.scaling_factor is not None:
            y_vals = y_vals * self.scaling_factor

        # Step 4: Calculate and store max AFTER scaling, before exponentiating
        # This is critical - we subtract max from the SCALED values
        # We compute max from first batch, then reuse it for consistency
        if self.y_max_after_scaling is None:
            self.y_max_after_scaling = y_vals.max()

        # Subtract max for numerical stability
        y_vals = y_vals - self.y_max_after_scaling

        # Step 5: Exponentiate to linear space
        y_vals = torch.exp(y_vals)

        return y_vals


    def one_hot_encode(self, bucket: FastBucket, assignments: torch.IntTensor):
        domain_sizes = bucket.get_message_dimension()
        num_samples, num_vars = assignments.shape

        if self.lower_dim: # send n domain variables to n-1 vector
            one_hot_encoded_samples = torch.cat([F.one_hot(assignments[:, i], num_classes=domain_sizes[i])[:, 1:] for i in range(num_vars)], dim=-1)
        else:
            one_hot_encoded_samples = torch.cat([F.one_hot(assignments[:, i], num_classes=domain_sizes[i]) for i in range(num_vars)], dim=-1)
        return one_hot_encoded_samples.float().to(self.device)