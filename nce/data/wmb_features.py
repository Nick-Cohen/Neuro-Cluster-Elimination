"""WMB-derived input features for the NeuroBE cluster nets (doc 63 / Q60).

Doc 43 measured *why* residual learning (target = exact - WMB) wins on grids and
loses on RBMs: not because WMB is inaccurate on RBMs -- per assignment it is
*tighter* there -- but because it stops **ranking**.  Spearman rho(WMB, exact) is
0.91-0.92 on the grids and 0.51-0.52 on the RBMs.  Residual learning *forces* the
net to trust that base; feeding the base as an **input** instead lets the net
learn how much to trust it.

Two feature modes:

``combined``
    one column: the cluster's WMB message estimate at the assignment.

``partitions``
    ``1 + k`` columns: the same combined estimate, plus one column per
    mini-bucket partition factor.  WMB's factorisation *discards* the dependence
    between partitions (it multiplies their separately-eliminated estimates);
    exposing the partitions individually is what lets the net learn that
    dependence back.

Normalisation (the choice, and why)
-----------------------------------
Every column is put through the **target's own affine normaliser**, the one
``DataPreprocessor`` already fitted to y:

    c0 = (base*ln10 - off_y) / scale_y

with ``(off_y, scale_y) = (ln_min, ln_range)`` for ``minmax_01``.  Reasons:

1. ``base`` and ``y`` are the same physical quantity (log of a message value)
   differing only by the WMB slack, so the map that puts y in [0, 1] puts the
   base in approximately [0, 1] too.  The feature is therefore O(1) and
   commensurate with the one-hot columns, which are exactly 0/1.  A raw log
   value would span tens of dex against 0/1 inputs and swamp the first layer.
2. It makes the residual hypothesis **exactly representable**: a unit weight
   from this column to the output plus the inner net's correction *is* residual
   learning.  So whatever WMB-as-input loses to the residual arm is
   optimisation, not expressiveness -- which is the cleanest form the comparison
   can take.
3. It re-uses an already-fitted normaliser.  No new statistic is estimated, so
   nothing new can leak between the arms, and the affine map is exactly the one
   whose inverse ``undo_normalization`` applies.

For ``partitions`` the per-partition columns use ``off_y / k`` so that they sum
**exactly** to the combined column::

    sum_i (p_i*ln10 - off_y/k)/scale_y == (base*ln10 - off_y)/scale_y

i.e. an all-ones weight vector over the partition columns reproduces the
combined feature, so ``partitions`` strictly contains ``combined``'s hypothesis
class, which in turn contains the residual's.  The three arms are nested.

Train/eval identity
-------------------
The spec is built **once**, on the first (training) load, and then carried on the
``FactorNN`` so that evaluation replays the identical recipe: the same frozen
list of base factors *in the same order*, the same ``(off_y, scale_y)``, and the
same clamp bounds.  There is exactly one implementation of the column
arithmetic (``WMBFeatureSpec.columns``) and both paths call it.
"""

import math

import torch

LN10 = math.log(10.0)


def affine_of(dp):
    """The preprocessor's normalisation written as ``y_norm = (y*ln10 - off)/scale``.

    ``minmax_01`` maps natural-log values through ``(v - ln_min)/ln_range``;
    ``logspace_mean`` subtracts a constant and does not rescale.
    """
    if getattr(dp, 'normalization_mode', None) == 'minmax_01':
        return float(dp.ln_min), float(dp.ln_range)
    nc = dp.normalizing_constant
    if isinstance(nc, torch.Tensor):
        nc = float(nc.item())
    return float(nc), 1.0


class WMBFeatureSpec:
    """Frozen recipe turning a cluster's WMB base factors into net input columns.

    Args:
        base_factors: list of ``FastFactor`` whose log10 values SUM to the WMB
            message estimate over ``labels``.  The list order is frozen here and
            defines the column order for ``partitions``.
        mode: ``'combined'`` or ``'partitions'``.
        off_y, scale_y: the target normaliser's affine constants (``affine_of``).
        labels: the cluster's message scope, i.e. the column order of the
            assignment coordinates handed to ``columns()``.
    """

    def __init__(self, base_factors, mode, off_y, scale_y, labels):
        if mode not in ('combined', 'partitions'):
            raise ValueError(f"unknown wmb_input mode {mode!r}")
        self.base_factors = list(base_factors)     # ORDER IS FROZEN
        self.mode = mode
        self.off_y = float(off_y)
        self.scale_y = float(scale_y) or 1.0
        self.labels = list(labels)
        # Clamp bounds, fitted once on the training load and then reused so the
        # evaluation columns are built by exactly the same map.
        self.lo = None
        self.hi = None
        self.n_nonfinite = 0
        self.n_rows_seen = 0

    @property
    def n_features(self):
        return 1 if self.mode == 'combined' else 1 + len(self.base_factors)

    # -- the single implementation of the column arithmetic -------------------
    def _raw_columns(self, coords):
        """(n, n_features) pre-clamp feature block for assignments `coords`."""
        vals = [f._get_values(assignments=coords, message_scope=self.labels).reshape(-1)
                for f in self.base_factors]
        k = len(vals)
        total = vals[0]
        for v in vals[1:]:
            total = total + v
        cols = [(total * LN10 - self.off_y) / self.scale_y]
        if self.mode == 'partitions':
            off_i = self.off_y / k
            for v in vals:
                cols.append((v * LN10 - off_i) / self.scale_y)
        return torch.stack(cols, dim=1)

    def fit(self, coords):
        """Fit the clamp bounds on the training assignments. Idempotent."""
        if self.lo is not None:
            return
        raw = self._raw_columns(coords)
        fin = torch.isfinite(raw)
        n_cols = raw.shape[1]
        lo = torch.zeros(n_cols, dtype=raw.dtype, device=raw.device)
        hi = torch.zeros(n_cols, dtype=raw.dtype, device=raw.device)
        for j in range(n_cols):
            col = raw[:, j][fin[:, j]]
            if col.numel():
                lo[j], hi[j] = col.min(), col.max()
        self.lo, self.hi = lo, hi

    def columns(self, coords, dtype=None, device=None):
        """(n, n_features) finite feature block for assignments `coords`.

        `coords` is (n, len(self.labels)) int64 assignment coordinates in the
        message-scope column order.
        """
        raw = self._raw_columns(coords)
        if self.lo is None:
            self.fit(coords)
        lo = self.lo.to(raw.device)
        hi = self.hi.to(raw.device)
        bad = ~torch.isfinite(raw)
        if bad.any():
            self.n_nonfinite += int(bad.sum())
            # -inf (a zero-probability WMB entry) -> the column's training floor;
            # +inf / nan -> its ceiling. Both bounds were frozen at fit time.
            neg = bad & (raw < 0)
            raw = torch.where(bad, hi.expand_as(raw), raw)
            raw = torch.where(neg, lo.expand_as(raw), raw)
        raw = torch.max(torch.min(raw, hi), lo)
        self.n_rows_seen += int(raw.shape[0])
        if dtype is not None or device is not None:
            raw = raw.to(dtype=dtype or raw.dtype, device=device or raw.device)
        return raw

    def stats(self):
        return {
            'mode': self.mode,
            'n_features': self.n_features,
            'n_partitions': len(self.base_factors),
            'off_y': self.off_y,
            'scale_y': self.scale_y,
            'lo': [float(v) for v in self.lo] if self.lo is not None else None,
            'hi': [float(v) for v in self.hi] if self.hi is not None else None,
            'n_nonfinite': int(self.n_nonfinite),
            'n_rows_seen': int(self.n_rows_seen),
        }
