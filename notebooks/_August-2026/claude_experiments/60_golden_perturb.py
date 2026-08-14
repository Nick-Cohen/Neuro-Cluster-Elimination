#!/usr/bin/env python
"""60: are the regenerated CRN goldens still SENSITIVE to nondeterminism?

A golden file that matches proves nothing on its own -- a golden regenerated
against a broken build matches that broken build perfectly. The question is
whether the golden would still CATCH a defect. That must be established by
PERTURBATION, not inferred from a passing run.

Checks per case, on the frozen build:

  A  UNPERTURBED       must MATCH the committed golden.
  B  REPEAT in-process must be identical to A (the run is deterministic at all).
  G1 crn.stream_key +1 must MISMATCH. This is the CRN keying defect class --
                       the exact risk introduced by making CRN default-on. The
                       uniform sampling scheme draws through `stream_key`
                       (NOT `stream_seed`, which serves the no-replacement
                       generator only), so this is the live hook for this case.
  G2 one-ULP bump on   must MISMATCH. Numeric sensitivity: if a single ULP on a
     one input factor   single factor cannot move the golden, the golden is too
                       coarse to police the numerics.

  I1 extra global-RNG  INFORMATIONAL, expected NO change. Doc 56 replaced
     draw per cluster   collision-prone global seeding with derivation from
                       (config seed, scope, role). If that worked, global RNG
                       state is not an input and this perturbation is inert.
                       Reported either way; it does not gate.

Gate: A matches, B identical, G1 and G2 both detected.

Usage: python 60_golden_perturb.py --case cpu_grid_plain --out perturb.json
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
ap.add_argument('--out', default='')
args = ap.parse_args()

sys.path.insert(0, args.repo)

import torch

spec = importlib.util.spec_from_file_location(
    'detreg', os.path.join(args.repo, 'tests', 'test_determinism_regression.py'))
detreg = importlib.util.module_from_spec(spec)
sys.modules['detreg'] = detreg
spec.loader.exec_module(detreg)

torch.set_num_threads(1)   # CPU goldens are 1-thread values

with open(detreg.GOLDENS_PATH) as fh:
    GOLDENS = json.load(fh)

case = args.case
gold = GOLDENS['cases'][case]['golden']
print('case %s\n  golden log_z = %s\n  golden epochs = %r'
      % (case, gold['log_z_repr'], gold['epochs']), flush=True)

results = {'case': case, 'golden': gold, 'checks': {}}


def compare(tag, got):
    same_lz = (got['log_z_repr'] == gold['log_z_repr'])
    same_ep = (got['epochs'] == gold['epochs'])
    same_le = (got.get('local_errors') == gold.get('local_errors'))
    matches = bool(same_lz and same_ep and same_le)
    print('  %-26s log_z=%-22s matches_golden=%s'
          % (tag, got['log_z_repr'], matches), flush=True)
    return {'log_z_repr': got['log_z_repr'], 'match_log_z': same_lz,
            'match_epochs': same_ep, 'match_local_errors': same_le,
            'matches_golden': matches}


# ---- A: unperturbed --------------------------------------------------------
a = detreg.run_case(case)
results['checks']['A_unperturbed'] = compare('A unperturbed', a)

# ---- B: repeat in-process --------------------------------------------------
b = detreg.run_case(case)
results['checks']['B_repeat'] = compare('B repeat', b)
results['checks']['B_repeat']['identical_to_A'] = (
    b['log_z_repr'] == a['log_z_repr'] and b['epochs'] == a['epochs']
    and b.get('local_errors') == a.get('local_errors'))

# ---- G1: CRN stream_key off by one ----------------------------------------
from nce.sampling import crn

_orig_key = crn.stream_key


def _key_off_by_one(seed, scope, domain_sizes, role, draw_index):
    return _orig_key(seed, scope, domain_sizes, role, draw_index) ^ 1


crn.stream_key = _key_off_by_one
try:
    g1 = detreg.run_case(case)
    results['checks']['G1_crn_key_perturbed'] = compare('G1 crn.stream_key^1', g1)
finally:
    crn.stream_key = _orig_key

# ---- G2: one ULP on one input factor --------------------------------------
from nce.inference.graphical_model import FastGM

_orig_init = FastGM.__init__


def _ulp_init(self, *a_, **k_):
    _orig_init(self, *a_, **k_)
    for key in sorted(self.buckets, key=lambda v: getattr(v, 'label', v)):
        for f in self.buckets[key].factors:
            t = getattr(f, 'tensor', None)
            if t is not None and t.numel() > 0 and t.is_floating_point():
                flat = t.reshape(-1)
                flat[0] = torch.nextafter(flat[0],
                                          torch.tensor(float('inf'),
                                                       dtype=t.dtype,
                                                       device=t.device))
                return
    raise SystemExit('G2: found no float factor tensor to perturb')


FastGM.__init__ = _ulp_init
try:
    g2 = detreg.run_case(case)
    results['checks']['G2_one_ulp_factor'] = compare('G2 one-ULP factor', g2)
finally:
    FastGM.__init__ = _orig_init

# ---- I1 (informational): extra global-RNG draw per NN cluster --------------
from nce.inference.bucket import FastBucket

_orig_nn = FastBucket.compute_message_nn


def _extra_draw(self, *a_, **k_):
    torch.rand(1)
    return _orig_nn(self, *a_, **k_)


FastBucket.compute_message_nn = _extra_draw
try:
    i1 = detreg.run_case(case)
    results['checks']['I1_extra_global_draw'] = compare('I1 extra global draw', i1)
finally:
    FastBucket.compute_message_nn = _orig_nn

# ---- verdict ---------------------------------------------------------------
c = results['checks']
ok = (c['A_unperturbed']['matches_golden']
      and c['B_repeat']['identical_to_A']
      and not c['G1_crn_key_perturbed']['matches_golden']
      and not c['G2_one_ulp_factor']['matches_golden'])
results['verdict'] = 'SENSITIVE' if ok else 'PROBLEM'

print('\nVERDICT: %s' % results['verdict'])
print('  A matches golden           : %s' % c['A_unperturbed']['matches_golden'])
print('  B identical to A           : %s' % c['B_repeat']['identical_to_A'])
print('  G1 CRN key   -> DETECTED   : %s' % (not c['G1_crn_key_perturbed']['matches_golden']))
print('  G2 one ULP   -> DETECTED   : %s' % (not c['G2_one_ulp_factor']['matches_golden']))
print('  I1 global RNG draw moved it: %s  (informational; False = seeding is '
      'insulated from global RNG state, as doc 56 intended)'
      % (not c['I1_extra_global_draw']['matches_golden']))

print('\n===GOLDEN-PERTURB===' + json.dumps(results))
if args.out:
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=1)
