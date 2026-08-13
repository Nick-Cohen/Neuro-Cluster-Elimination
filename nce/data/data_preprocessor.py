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
                 device=None, use_bw_approx: bool = False, normalization_mode: str = 'logspace_mean',
                 dtype=None) -> None:
        self.y = y
        self.bw = bw
        if device is None:
            self.device = y.device if y is not None else 'cpu'
        else:
            self.device = device
        self.lower_dim = lower_dim
        self.use_bw_approx = use_bw_approx
        self.normalization_mode = normalization_mode
        self.dtype = dtype or torch.float32

        # Normalizing constant (in natural log space) - computed lazily
        self.normalizing_constant = None

        # For use_bw_approx mode: bw value at argmax(y + bw), passed to loss function
        self.bw_normalizing_constant = None

        # Global max of targets for UKL numerical stability (computed from full training data)
        # CRITICAL: This must be computed ONCE from all training data and used for ALL batches
        # Using per-batch max causes gradient inconsistency and training divergence
        self.global_max_targets = None

        # Perf: ln(10) cached per device. The old code rebuilt
        # `torch.log(torch.tensor(10.0)).to(dev)` on every call -- a pageable
        # host->device copy, which synchronizes the CUDA stream and so drains
        # the queue inside the hot sample-generation loop (doc 03 sec 2.3, item 3).
        # MEASURED 32 us/call to construct vs 0.33 us to look up.
        # A cached float32 DEVICE TENSOR is used rather than doc 03's suggested
        # Python float: `t / <python float>` is rewritten by PyTorch into a
        # multiply-by-reciprocal and is NOT bit-identical to `t / <0-dim tensor>`
        # (MEASURED max diff 3.05e-05 at magnitude 700 -- exactly the deviation
        # doc 03 sec 2.7 reported for its V1). The cached tensor holds the
        # identical bit pattern and reproduces the current values exactly.
        self._ln10_cache = {}

        # minmax_01 mode attributes (populated by _initialize_normalizing_constant)
        self.ln_min = None
        self.ln_max = None
        self.sum_ln = None
        self.ln_range = None

        # Initialize normalizing constant from provided samples if available
        if y is not None:
            self._initialize_normalizing_constant(y, bw)

    def _ln10(self, device):
        """ln(10) as a 0-dim float32 tensor on `device`, built once per device.

        Bit-identical to the `torch.log(torch.tensor(10.0)).to(device)` it
        replaces (verified: same int32 bit pattern, and mul/div over 2^22
        random values at magnitudes 1/300/700 are exactly equal).
        """
        # `torch.device('cuda')` and `torch.device('cuda:0')` are distinct keys
        # but both are valid; caching per key is correct, just occasionally
        # stores two entries.
        # getattr guard: DataPreprocessor instances are pickled into checkpoints,
        # so an object unpickled from a pre-cache checkpoint has no _ln10_cache.
        cache = getattr(self, '_ln10_cache', None)
        if cache is None:
            cache = {}
            self._ln10_cache = cache
        key = str(device)
        t = cache.get(key)
        if t is None:
            t = torch.log(torch.tensor(10.0)).to(device)
            cache[key] = t
        return t

    def _initialize_normalizing_constant(self, y_vals: torch.Tensor, bw_vals: torch.Tensor = None):
        """Compute normalizing constant from training samples.

        The normalizing constant is the WEIGHTED MEAN of y (weighted by exp(bw)).
        This ensures that if the NN predicts 0 everywhere, the reconstructed
        message equals the constant estimator, giving near-zero log Z error.

        WITHOUT bw (use_bw_approx=False):
            normalizing_constant = logsumexp(y)  (effectively the max)

        WITH bw (use_bw_approx=True):
            normalizing_constant = logsumexp(y + bw) - logsumexp(bw)
            This is the weighted mean of y, weighted by exp(bw).
            bw_normalizing_constant = bw[argmax(y + bw)]

        Args:
            y_vals: Message values in log10 space
            bw_vals: Backward message values in log10 space (or None)
        """
        # Check for NaN in inputs
        if torch.isnan(y_vals).any():
            print(f"[DataPreprocessor] ERROR: NaN in y_vals input to _initialize_normalizing_constant!")

        ln10 = self._ln10(self.device)

        # Convert to natural log space
        y_ln = y_vals * ln10

        if self.normalization_mode == 'minmax_01':
            # Finite-aware: with masked_net + doping off, true-zero targets are
            # -inf and must be ignored when fitting the [0,1] range. No-op when
            # all targets are finite (the usual doped/no-zeros case).
            finite_ln = y_ln[torch.isfinite(y_ln)]
            if finite_ln.numel() == 0:
                finite_ln = y_ln
            self.ln_min = finite_ln.min().item()
            self.ln_max = finite_ln.max().item()
            self.ln_range = max(self.ln_max - self.ln_min, 1e-10)
            # sum_ln = sum(y_ln_i - ln_min) for IS weights in loss function
            self.sum_ln = (finite_ln - self.ln_min).sum().item()
            if self.ln_max == self.ln_min:
                print(f"[DataPreprocessor minmax_01] WARNING: all targets identical (ln_max == ln_min), using epsilon guard")
            print(f"[DataPreprocessor minmax_01] ln_min={self.ln_min:.4f}, ln_max={self.ln_max:.4f}, sum_ln={self.sum_ln:.4f}")
            return

        if self.use_bw_approx and bw_vals is not None:
            # WITH bw: normalize by logsumexp(y + bw) - logsumexp(bw)
            # This is the weighted mean of y (constant estimator)
            bw_ln = bw_vals * ln10
            combined = y_ln + bw_ln

            logsumexp_combined = torch.logsumexp(combined, dim=0)
            logsumexp_bw = torch.logsumexp(bw_ln, dim=0)
            self.normalizing_constant = logsumexp_combined - logsumexp_bw

            # Also store bw value at argmax(y + bw) for the loss function
            max_idx = torch.argmax(combined)
            self.bw_normalizing_constant = bw_ln[max_idx]
        else:
            # WITHOUT bw: normalize by logsumexp(y)
            self.normalizing_constant = torch.logsumexp(y_ln, dim=0)

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
        if self.normalization_mode == 'minmax_01':
            if self.ln_min is None:
                self._initialize_normalizing_constant(y_vals, bw_vals)
        elif self.normalizing_constant is None:
            self._initialize_normalizing_constant(y_vals, bw_vals)

        ln10 = self._ln10(self.device)

        # minmax_01 mode: normalize to [0, 1] range in natural log space
        if self.normalization_mode == 'minmax_01':
            y_ln = y_vals * ln10
            y_normalized = (y_ln - self.ln_min) / self.ln_range
            return y_normalized, None

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
            outputs: Normalized outputs from neural network.
                     For logspace_mean: natural log space, centered.
                     For minmax_01: [0,1] normalized values.

        Returns:
            Values in log10 space
        """
        ln10 = self._ln10(outputs.device)

        if self.normalization_mode == 'minmax_01':
            # Undo: y_ln = ln_min + outputs * ln_range, then convert to log10
            y_ln = self.ln_min + outputs * self.ln_range
            return y_ln / ln10

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
        return one_hot_encoded_samples.to(dtype=self.dtype, device=self.device)

