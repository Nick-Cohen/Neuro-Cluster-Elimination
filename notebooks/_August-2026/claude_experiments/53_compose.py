#!/usr/bin/env python3
"""53: does the COMBINATION of perf/table-path + perf/elim-block-projection stay
bit-identical to the pre-merge base?

Both branches were proven bit-identical against 9dbc2d0 *independently*. They meet in
`SampleGenerator.sample_tensor_product_elimination`'s SMALL path, where table factors
now go through table-path's `_get_slices_prepared` and NN factors go through
elim-block's rewritten `FactorNN._get_slices`, accumulating into the same `uv`.
Neither branch's own corpus covers that: 52_ab.py replays `kind == 'table'` factors
only, and doc 51's evidence is all on `_eval_elim_block` / the streaming path.

This runs the MERGED build on a real config and, for every real call, re-executes the
VERBATIM 9dbc2d0 implementation (extracted with `git show`, not re-typed) on the same
inputs in the same process, and compares with torch.equal.

Wrapped:
  FastFactor._get_slices_prepared  vs base FastFactor._get_slices   (table-path)
  FastFactor._get_slices           vs base FastFactor._get_slices   (any other caller)
  FactorNN._get_slices             vs base FactorNN._get_slices     (elim-block)
  FactorNN._eval_elim_block        vs base FactorNN._eval_elim_block(elim-block)

Usage: python 53_compose.py --repo <merged-wt> --case cuda_rbm --gpu 0 --emax 16
"""
import argparse, ast, importlib.util, json, math, os, subprocess, sys, textwrap

ap = argparse.ArgumentParser()
ap.add_argument('--repo', required=True)
ap.add_argument('--base-rev', default='9dbc2d0')
ap.add_argument('--case', required=True)
ap.add_argument('--gpu', type=int, required=True)
ap.add_argument('--emax', type=int, default=None)
ap.add_argument('--out', default='')
args = ap.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
sys.path.insert(0, args.repo)

import torch
torch.set_num_threads(1)
from nce.inference.factor import FastFactor
from nce.inference.factor_nn import FactorNN


def extract(path, cls_name, fn_name):
    """Verbatim method source from the base revision, exec'd as a plain function."""
    src = subprocess.check_output(
        ['git', '-C', args.repo, 'show', f'{args.base_rev}:{path}'], text=True)
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for c in node.body:
                if isinstance(c, ast.FunctionDef) and c.name == fn_name:
                    fs = textwrap.dedent(ast.get_source_segment(src, c))
                    ns = {'torch': torch, 'math': math}
                    exec(compile(fs, f'<base {args.base_rev} {cls_name}.{fn_name}>',
                                 'exec'), ns)
                    return ns[fn_name], len(fs.splitlines())
    raise SystemExit(f'not found: {cls_name}.{fn_name}')


base_ff_gs, n1 = extract('nce/inference/factor.py', 'FastFactor', '_get_slices')
base_nn_gs, n2 = extract('nce/inference/factor_nn.py', 'FactorNN', '_get_slices')
base_nn_eb, n3 = extract('nce/inference/factor_nn.py', 'FactorNN', '_eval_elim_block')
print(f'base impls extracted from {args.base_rev}: '
      f'FastFactor._get_slices {n1}L, FactorNN._get_slices {n2}L, '
      f'FactorNN._eval_elim_block {n3}L', flush=True)

# name -> [n_compared, n_bit_equal, max_abs_dev, n_shape_mismatch]
STATS = {}
PLAN_CTX = {}   # id(plan) -> (elim_vars, elim_domain_sizes, message_scope)


def record(name, old, new):
    s = STATS.setdefault(name, [0, 0, 0.0, 0])
    s[0] += 1
    if tuple(old.shape) != tuple(new.shape):
        s[3] += 1
        return
    if torch.equal(old, new):
        s[1] += 1
    else:
        o, n = old.double(), new.double()
        d = torch.where(torch.isfinite(o) | torch.isfinite(n),
                        (o - n).abs(), torch.zeros_like(o))
        # -inf == -inf must not count as a deviation
        d = torch.nan_to_num(d, nan=0.0, posinf=float('inf'))
        s[2] = max(s[2], float(d.max().item()))


# ---- FastFactor: capture plan context, then compare the prepared path ----------
_orig_plan = FastFactor._slice_plan
_orig_prep = FastFactor._get_slices_prepared
_orig_ff_gs = FastFactor._get_slices


def plan_w(self, elim_vars, elim_domain_sizes, message_scope):
    p = _orig_plan(self, elim_vars, elim_domain_sizes, message_scope)
    if p is not None:
        # Keep a reference to `p` itself: without it a collected plan's id can be
        # reused by a later plan and the stale context would produce a bogus
        # mismatch. Bounded by the number of (factor, cluster) plans.
        PLAN_CTX[id(p)] = (p, elim_vars, elim_domain_sizes, message_scope)
    return p


def prep_w(self, plan, assignments):
    new = _orig_prep(self, plan, assignments)
    ctx = PLAN_CTX.get(id(plan))
    if ctx is not None:
        old = base_ff_gs(self, assignments, *ctx[1:])
        record('FastFactor._get_slices_prepared', old, new)
    return new


def ff_gs_w(self, assignments, elim_vars, elim_domain_sizes, message_scope):
    new = _orig_ff_gs(self, assignments, elim_vars, elim_domain_sizes, message_scope)
    old = base_ff_gs(self, assignments, elim_vars, elim_domain_sizes, message_scope)
    record('FastFactor._get_slices', old, new)
    return new


FastFactor._slice_plan = plan_w
FastFactor._get_slices_prepared = prep_w
FastFactor._get_slices = ff_gs_w

# ---- FactorNN: the elim-block branch's two rewritten entry points --------------
_orig_nn_gs = FactorNN._get_slices
_orig_nn_eb = FactorNN._eval_elim_block


def nn_gs_w(self, assignments, elim_vars, elim_domain_sizes, message_scope):
    new = _orig_nn_gs(self, assignments, elim_vars, elim_domain_sizes, message_scope)
    old = base_nn_gs(self, assignments, elim_vars, elim_domain_sizes, message_scope)
    record('FactorNN._get_slices', old, new)
    return new


def nn_eb_w(self, assignments, elim_coords, elim_vars, elim_var_labels, message_scope):
    new = _orig_nn_eb(self, assignments, elim_coords, elim_vars, elim_var_labels,
                      message_scope)
    old = base_nn_eb(self, assignments, elim_coords, elim_vars, elim_var_labels,
                     message_scope)
    record('FactorNN._eval_elim_block', old, new)
    return new


FactorNN._get_slices = nn_gs_w
FactorNN._eval_elim_block = nn_eb_w

# ---- run the real case --------------------------------------------------------
spec = importlib.util.spec_from_file_location(
    'detreg', os.path.join(args.repo, 'tests', 'test_determinism_regression.py'))
detreg = importlib.util.module_from_spec(spec)
sys.modules['detreg'] = detreg
spec.loader.exec_module(detreg)
from nce.config_schema import prepare_config
from nce.inference.graphical_model import FastGM

cfg = dict(detreg.CASES[args.case]['config'])
pk = cfg.pop('problem_key')
if args.emax is not None:
    cfg['max_merge_bound'] = args.emax
full = prepare_config(cfg, strict=False)
model = detreg._load_model(pk)
gm = FastGM(model=model, nn_config=full, device=full['device'])
gm.eliminate_variables(all=True)

res = {'case': args.case, 'emax': args.emax,
       'log_z': repr(float(gm.log_partition_function)),
       'stats': {k: {'compared': v[0], 'bit_equal': v[1], 'max_dev': v[2],
                     'shape_mismatch': v[3]} for k, v in STATS.items()}}
print('===COMPOSE===' + json.dumps(res))
tot = sum(v[0] for v in STATS.values())
eq = sum(v[1] for v in STATS.values())
print(f'\nTOTAL {eq}/{tot} calls bit-identical to {args.base_rev}; '
      f'max deviation {max([v[2] for v in STATS.values()] + [0.0])!r}')
for k, v in sorted(STATS.items()):
    print(f'  {k:<38} {v[1]:>8}/{v[0]:<8} maxdev={v[2]!r} shape_mismatch={v[3]}')
if args.out:
    open(args.out, 'w').write(json.dumps(res, indent=1))
