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

        # --- WMB base as a learning signal (doc 63 / Q60) -------------------
        # Both mechanisms are driven by the same object: the cluster's WMB base
        # factors, set externally by bucket.compute_message_nn().
        #
        # RESIDUAL arm (`wmb_residual`): `base_factors` is set and `load()`
        # appends ONE trailing column carrying the affine offset that turns the
        # inner net's output into the reconstructed message, consumed additively
        # by nce.neural_networks.net.WMBResidualNet.
        #
        # INPUT arm (`wmb_input`): `wmb_spec` is set and `load()` appends the
        # spec's feature columns as ordinary net INPUTS. The target stays the
        # true message, so the net learns how much to trust the base rather than
        # being forced to trust it.
        #
        # The two are mutually exclusive; bucket.compute_message_nn() enforces it.
        self.base_factors = None
        # Set by bucket.compute_message_nn() for the INPUT arm. The spec itself is
        # built lazily on the first load(), because it freezes the target
        # normaliser's affine constants and those only exist once
        # DataPreprocessor.normalize() has seen the first batch of message values.
        self.wmb_input_mode = 'off'
        self.wmb_base_factors = None
        self.wmb_spec = None
        # Diagnostics filled in by load(): stats of the log10 residual r = y - base.
        self.residual_stats = None
        # 'message' (default) keeps the normaliser fitted to y; 'residual' fits a
        # SECOND DataPreprocessor to r itself (doc 02 s2.2 option 3).
        self.residual_norm = 'message'
        # Set on the first load() when residual_norm == 'residual'. This is the
        # preprocessor that FactorNN must use to undo-normalise the emitted factor.
        self.residual_dp = None
        # The WMBResidualNet wrapper, so load() can install the unit-conversion
        # scale once the residual normaliser has been fitted. Set by bucket.py.
        self.residual_net = None

    def _affine(self, p):
        from nce.data.wmb_features import affine_of
        return affine_of(p)

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

        # --- WMB-as-INPUT (doc 63 / Q60) ------------------------------------
        # Append the spec's feature columns as ordinary net inputs. The target
        # `normalized_y` is untouched: the net still predicts the TRUE message,
        # it simply gets to see the cheap bound while doing so. The clamp bounds
        # are fitted on this first (training) load and then frozen, so the
        # FactorNN reproduces these columns exactly at evaluation time.
        if self.wmb_input_mode != 'off' and self.wmb_base_factors:
            if self.wmb_spec is None:
                from nce.data.wmb_features import WMBFeatureSpec
                off_y, scale_y = self._affine(self.data_preprocessor)
                self.wmb_spec = WMBFeatureSpec(
                    self.wmb_base_factors, self.wmb_input_mode,
                    off_y, scale_y, list(self.sample_generator.message_scope))
            self.wmb_spec.fit(assignments)
            feats = self.wmb_spec.columns(assignments, dtype=x.dtype, device=x.device)
            x = torch.cat([x, feats], dim=1)

        # --- WMB RESIDUAL (doc 02 / doc 31 arm 5) ---------------------------
        # Append an extra trailing column of x carrying the affine offset that
        # turns the inner net's output into the RECONSTRUCTED message in
        # y-normalised units, so WMBResidualNet computes
        #     y_hat_norm = scale * inner(x[:, :-1]) + x[:, -1:]
        # and every loss / validation / early-stopping site in Trainer compares
        # y_hat_norm against y_norm. The importance weights of
        # neurobe_weighted_mse therefore stay keyed on the message mass, never
        # on the correction.
        #
        # Write the two normalisers as affine maps in natural-log space:
        #     y_norm = (y*ln10 - off_y) / scale_y
        #     r_norm = (r*ln10 - off_r) / scale_r
        # Since y = r + base (log10 space),
        #     y_norm = r_norm*(scale_r/scale_y) + (off_r + base*ln10 - off_y)/scale_y
        # =>  net scale  = scale_r/scale_y
        #     trailing col = (off_r + base*ln10 - off_y)/scale_y
        if self.base_factors is not None:
            import math
            base_values = self.sample_generator.sample_tensor_product(self.base_factors, assignments)
            dp = self.data_preprocessor
            ln10 = math.log(10.0)
            off_y, scale_y = self._affine(dp)

            if self.residual_norm == 'residual':
                if self.residual_dp is None:
                    # Fit the second normaliser to r, once, on this (training) load.
                    from nce.data.data_preprocessor import DataPreprocessor
                    r_fit = (mess_values.reshape(-1) - base_values.reshape(-1))
                    rdp = DataPreprocessor(
                        y=None, bw=None, lower_dim=dp.lower_dim, device=dp.device,
                        use_bw_approx=False, normalization_mode=dp.normalization_mode,
                        dtype=dp.dtype,
                    )
                    rdp._initialize_normalizing_constant(r_fit, None)
                    self.residual_dp = rdp
                off_r, scale_r = self._affine(self.residual_dp)
                net_scale = scale_r / scale_y
                if self.residual_net is not None:
                    self.residual_net.scale = float(net_scale)
                col = ((base_values.reshape(-1) * ln10) + (off_r - off_y)) / scale_y
            else:
                net_scale = 1.0
                col = base_values.reshape(-1) * (ln10 / scale_y)

            base_col = col.reshape(-1, 1).to(dtype=x.dtype, device=x.device)
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
                        'norm_mode': self.residual_norm,
                        'net_scale': float(net_scale),
                        'off_y': float(off_y), 'scale_y': float(scale_y),
                        'off_r': float(self._affine(self.residual_dp)[0]) if self.residual_dp else None,
                        'scale_r': float(self._affine(self.residual_dp)[1]) if self.residual_dp else None,
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

