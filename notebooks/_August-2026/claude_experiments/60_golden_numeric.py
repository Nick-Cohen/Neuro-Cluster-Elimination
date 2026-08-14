#!/usr/bin/env python
"""60: how large a numeric perturbation does the log_z golden actually detect?

The one-ULP probe in 60_golden_perturb.py did NOT move the golden. That has two
possible explanations and they are very different:
  (a) the perturbation never applied -- the golden's numeric sensitivity is
      untested and the probe is worthless;
  (b) it applied and was washed out -- one ULP on ONE entry of ONE factor is
      genuinely below the resolution of a float32 log Z aggregated over ~180
      factors and 100 variables.

This distinguishes them by ASSERTING the perturbation applied (it counts and
verifies the mutated element) and then sweeping the relative magnitude to find
where detection begins.

Usage: python 60_golden_numeric.py --case cpu_grid_plain --out numeric.json
"""
import argparse
import importlib.util
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--repo', default=os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
ap.add_argument('--case', default='cpu_grid_plain')
ap.add_argument('--rels', nargs='+', type=float,
                default=[0.0, 1e-7, 1e-6, 1e-4, 1e-2])
ap.add_argument('--out', default='')
args = ap.parse_args()

sys.path.insert(0, args.repo)
import torch

spec = importlib.util.spec_from_file_location(
    'detreg', os.path.join(args.repo, 'tests', 'test_determinism_regression.py'))
detreg = importlib.util.module_from_spec(spec)
sys.modules['detreg'] = detreg
spec.loader.exec_module(detreg)
torch.set_num_threads(1)

with open(detreg.GOLDENS_PATH) as fh:
    gold = json.load(fh)['cases'][args.case]['golden']
print('golden log_z = %s' % gold['log_z_repr'], flush=True)

from nce.inference.graphical_model import FastGM

_orig_init = FastGM.__init__
STATE = {'rel': 0.0, 'applied': 0, 'before': None, 'after': None}


def _perturb_init(self, *a_, **k_):
    _orig_init(self, *a_, **k_)
    rel = STATE['rel']
    if rel == 0.0:
        return
    for key in sorted(self.buckets, key=lambda v: getattr(v, 'label', v)):
        for f in self.buckets[key].factors:
            t = getattr(f, 'tensor', None)
            if t is None or t.numel() == 0 or not t.is_floating_point():
                continue
            flat = t.reshape(-1)
            before = float(flat[0])
            if rel is None:                      # one ULP
                flat[0] = torch.nextafter(
                    flat[0], torch.tensor(float('inf'), dtype=t.dtype,
                                          device=t.device))
            else:
                flat[0] = flat[0] * (1.0 + rel) if before != 0.0 else rel
            after = float(t.reshape(-1)[0])
            STATE['applied'] += 1
            STATE['before'], STATE['after'] = before, after
            return
    raise SystemExit('no float factor tensor found to perturb')


FastGM.__init__ = _perturb_init
rows = []
try:
    for rel in args.rels:
        STATE['rel'] = rel
        STATE['applied'] = 0
        r = detreg.run_case(args.case)
        detected = (r['log_z_repr'] != gold['log_z_repr'])
        # For rel != 0 the mutation MUST have happened and MUST have changed
        # the stored value, or the probe proves nothing.
        applied_ok = True
        if rel != 0.0:
            applied_ok = (STATE['applied'] > 0
                          and STATE['before'] != STATE['after'])
        rows.append({'rel': rel, 'log_z': r['log_z_repr'],
                     'detected': detected, 'applied': STATE['applied'],
                     'before': STATE['before'], 'after': STATE['after'],
                     'applied_ok': applied_ok})
        print('  rel=%-8g log_z=%-22s detected=%-5s applied=%d (%r -> %r)'
              % (rel, r['log_z_repr'], detected, STATE['applied'],
                 STATE['before'], STATE['after']), flush=True)
finally:
    FastGM.__init__ = _orig_init

res = {'case': args.case, 'golden_log_z': gold['log_z_repr'], 'rows': rows}
bad = [r for r in rows if r['rel'] != 0.0 and not r['applied_ok']]
detected = [r['rel'] for r in rows if r['rel'] != 0.0 and r['detected']]
res['all_perturbations_applied'] = not bad
res['smallest_detected_rel'] = min(detected) if detected else None
print('\nall perturbations verifiably applied : %s' % res['all_perturbations_applied'])
print('smallest relative change detected    : %r' % res['smallest_detected_rel'])
print('\n===GOLDEN-NUMERIC===' + json.dumps(res))
if args.out:
    with open(args.out, 'w') as fh:
        json.dump(res, fh, indent=1)
