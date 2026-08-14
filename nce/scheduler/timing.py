"""Per-bucket timing that SEPARATES sample generation from NN training.

WHY THIS EXISTS
---------------
`FastGM.phase_times['nn_path']` bundles sample generation, training and message
construction into one number. Its own comment in
`nce/inference/graphical_model.py::process_bucket` admits as much:

    # NN-path wall time (sample gen + training + message construction);

A prior measurement used that bundled number and produced a conclusion that
stood, wrong, for weeks. So this module measures the two separately and reports
them per problem AND per network (per bucket), which is the granularity Nick
asked for.

HOW
---
Non-invasive monkeypatching, installed and removed by a context manager, so no
edit to inference code and no collision with agents working in
`factor_nn.py` / `factor.py` / the WMB code.

  sample_gen_s   summed wall time across ALL THREE phases of sample
                 generation, which is the part that is easy to get wrong:
                   * `sample_assignments`      -- drawing the assignments
                   * `compute_message_values`  -- evaluating the forward factor
                                                  product at those assignments
                   * `compute_backward_values` -- evaluating backward factors
                 Timing only `sample_assignments` undercounts badly: drawing
                 indices is trivial next to evaluating the factor product on
                 them. A first version of this module made exactly that mistake
                 and reported a sample-gen : train ratio of 0.0009.
  nn_train_s     wall time inside Trainer.train MINUS the sample generation
                 that happened inside it. This is the pure optimisation loop.
  overhead_s     nn_path total minus (sample_gen + nn_train): message
                 construction, preprocessing, net init.

CUDA ASYNCHRONY -- READ THIS BEFORE TRUSTING THE NUMBERS
--------------------------------------------------------
CUDA kernels are asynchronous, so a wall-clock interval that does not
synchronise attributes the cost of a kernel to whichever later call happens to
block. `synchronize=True` (the default when the model is on CUDA) inserts
`torch.cuda.synchronize()` at each boundary so the split is truthful. That is
itself a measurable perturbation, so it is a documented, overridable flag rather
than a hidden default -- and timing is a deliverable of the rerun, so the
default favours correctness over speed.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional


class TimingCollector:
    """Accumulates per-bucket sample-gen / train splits."""

    def __init__(self, synchronize: bool = True):
        self.synchronize = synchronize
        self.buckets: Dict[Any, Dict[str, float]] = {}
        self._current_bucket: Any = None
        # sample-gen time accrued inside the currently-running Trainer.train
        self._sg_inside_train: float = 0.0

    # -- internals ---------------------------------------------------------
    def _sync(self):
        if not self.synchronize:
            return
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    def _rec(self, label) -> Dict[str, float]:
        return self.buckets.setdefault(label, {
            'sample_gen_s': 0.0, 'nn_train_s': 0.0, 'nn_path_s': 0.0,
            'exact_s': 0.0, 'n_sample_calls': 0, 'n_train_calls': 0,
        })

    def add_sample_gen(self, label, dt: float):
        r = self._rec(label)
        r['sample_gen_s'] += dt
        r['n_sample_calls'] += 1
        self._sg_inside_train += dt

    def add_train(self, label, dt: float, sg_inside: float):
        r = self._rec(label)
        r['nn_train_s'] += max(0.0, dt - sg_inside)
        r['n_train_calls'] += 1

    def add_nn_path(self, label, dt: float):
        self._rec(label)['nn_path_s'] += dt

    def add_exact(self, label, dt: float):
        self._rec(label)['exact_s'] += dt

    # -- reporting ---------------------------------------------------------
    def per_network(self) -> List[Dict[str, Any]]:
        """One row per bucket that did NN work, in deterministic label order."""
        rows = []
        for label in sorted(self.buckets, key=lambda x: (str(type(x)), str(x))):
            r = dict(self.buckets[label])
            r['bucket_label'] = label
            r['overhead_s'] = max(
                0.0, r['nn_path_s'] - r['sample_gen_s'] - r['nn_train_s'])
            rows.append(r)
        return rows

    def totals(self) -> Dict[str, float]:
        t = {'sample_gen_s': 0.0, 'nn_train_s': 0.0, 'nn_path_s': 0.0,
             'exact_s': 0.0, 'n_nn_buckets': 0}
        for r in self.buckets.values():
            for k in ('sample_gen_s', 'nn_train_s', 'nn_path_s', 'exact_s'):
                t[k] += r[k]
            if r['n_train_calls']:
                t['n_nn_buckets'] += 1
        t['overhead_s'] = max(
            0.0, t['nn_path_s'] - t['sample_gen_s'] - t['nn_train_s'])
        if t['nn_train_s'] > 0:
            t['sample_gen_to_train_ratio'] = t['sample_gen_s'] / t['nn_train_s']
        return t

    def report(self) -> Dict[str, Any]:
        return {'totals': self.totals(), 'per_network': self.per_network(),
                'synchronized': self.synchronize}


@contextmanager
def collect_timings(synchronize: bool = True):
    """Install the timing patches for the duration of the block.

    Patches are restored in a `finally`, so an exception mid-elimination cannot
    leave the process with a monkeypatched SampleGenerator.
    """
    from nce.sampling.sample_generator import SampleGenerator
    from nce.neural_networks.train import Trainer
    from nce.inference.graphical_model import FastGM

    tc = TimingCollector(synchronize=synchronize)

    # All three sample-generation phases, not just assignment drawing.
    _SG_METHODS = ('sample_assignments', 'compute_message_values',
                   'compute_backward_values')
    orig_sg = {name: getattr(SampleGenerator, name) for name in _SG_METHODS}
    orig_train = Trainer.train
    orig_process = FastGM.process_bucket

    def _make_sg_wrapper(orig_fn):
        def wrapper(self, *a, **kw):
            label = getattr(getattr(self, 'bucket', None), 'label', None)
            tc._sync()
            t0 = time.perf_counter()
            try:
                return orig_fn(self, *a, **kw)
            finally:
                tc._sync()
                tc.add_sample_gen(label, time.perf_counter() - t0)
        return wrapper

    def process_bucket(self, bucket, exact=False):
        """Charge each cluster's total to nn_path or exact, so `overhead_s`
        (message construction, preprocessing, net init) is derivable."""
        label = getattr(bucket, 'label', None)
        before = tc._rec(label)['nn_train_s']
        tc._sync()
        t0 = time.perf_counter()
        try:
            return orig_process(self, bucket, exact=exact)
        finally:
            tc._sync()
            dt = time.perf_counter() - t0
            # A cluster that ran the optimiser is an NN cluster; otherwise the
            # exact path handled it. Decided by observation, not by re-deriving
            # process_bucket's own iB/ecl gate, which would drift out of sync.
            if tc._rec(label)['nn_train_s'] > before:
                tc.add_nn_path(label, dt)
            else:
                tc.add_exact(label, dt)

    def train(self, *a, **kw):
        label = getattr(getattr(self, 'bucket', None), 'label', None)
        # Save/restore so nested trainers (loss_fn2 second pass) accrue
        # their own sample-gen time rather than the outer call's.
        outer = tc._sg_inside_train
        tc._sg_inside_train = 0.0
        tc._sync()
        t0 = time.perf_counter()
        try:
            return orig_train(self, *a, **kw)
        finally:
            tc._sync()
            dt = time.perf_counter() - t0
            inside = tc._sg_inside_train
            tc.add_train(label, dt, inside)
            tc._sg_inside_train = outer + inside

    for name, fn in orig_sg.items():
        setattr(SampleGenerator, name, _make_sg_wrapper(fn))
    Trainer.train = train
    FastGM.process_bucket = process_bucket
    try:
        yield tc
    finally:
        for name, fn in orig_sg.items():
            setattr(SampleGenerator, name, fn)
        Trainer.train = orig_train
        FastGM.process_bucket = orig_process
