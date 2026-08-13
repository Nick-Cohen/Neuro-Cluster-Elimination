"""Audit: does the legacy sample seed `bucket_label + 10000*seed + 100*draw`
collide between NN clusters in the completed study's configurations?

Two modes:

  --validate <problem> <iB> <D>
      Run the REAL pipeline (structure-only epochs) and compare the draws it
      actually takes against the cheap structural prediction below. Prints both.

  --audit
      For every (problem, iB, D) cell in the study, build the FastGM (no
      training at all) and enumerate the predicted NN clusters, then report
      every colliding (bucket, draw) pair and whether the collision would
      actually produce identical assignments.

The prediction reuses exactly the `is_nn` / `_scope_at_elim` logic that
`FastGM._reduce_nn_core` uses to decide the merge, so it is the code's own
notion of "this cluster gets a network", not a reimplementation.
"""
import argparse
import contextlib
import io
import itertools
import json
import sys

import torch

from nce.config_schema import prepare_config
from nce.inference.bucket import FastBucket
from nce.inference.graphical_model import FastGM
from nce.benchmark_problems.catalog_utils import get_catalog
from nce.sampling.sample_generator import SampleGenerator

torch.set_num_threads(1)

# Draw structure of the uniform path, read off nce/neural_networks/train.py:
#   draw 0  init_batches       = load_batches(bs, set_size // bs)
#   draw 1  nbe validation set = load(num_samples = total // 9)     [is_validation=False!]
#   draw 2+ per training set   = load_batches(bs, set_size // bs)   x num_sets
# `is_validation=True` is never reached on this path, so the 50,000,000
# validation offset never fires and every draw shares one counter.
N_DRAWS_DEFAULT = 3


# Use the main checkout's populated model cache; this worktree's own cache is
# empty and the on-demand downloader stalls on it.
CACHE_DIR = '/home/cohenn1/NCE/.model_cache'
_CAT = [None]


def catalog():
    if _CAT[0] is None:
        _CAT[0] = get_catalog(cache_dir=CACHE_DIR)
    return _CAT[0]


def build(problem_key, cfg_over):
    cfg = prepare_config(dict(cfg_over), strict=False)
    with contextlib.redirect_stdout(io.StringIO()):
        gm = FastGM(model=catalog()[problem_key], nn_config=cfg,
                    device=cfg['device'])
    return gm, cfg


def predicted_nn_clusters(gm):
    """(label, separator, elim_labels) for every cluster the code will train.

    Mirrors `_reduce_nn_core.is_nn`: a cluster is NN when its outgoing scope
    exceeds the i-bound in width or the exact-computation limit in size.
    """
    iB = float(gm.iB) if gm.iB else float('inf')
    ecl = float(gm.ecl) if gm.ecl else float('inf')
    states = {v.label: gm.matching_var(v.label).states for v in gm.vars}
    out = []
    for key_var, b in gm.buckets.items():
        scope = getattr(b, '_scope_at_elim', None)
        if scope is None:
            scope = set(gm.message_scopes.get(key_var.label, []))
            scope.update(v.label for v in b.elim_vars)
        elim = {v.label for v in b.elim_vars}
        sep = sorted(set(scope) - elim)
        prod = 1.0
        for l in sep:
            prod *= states[l]
        if len(sep) > iB or prod > ecl:
            out.append({'label': key_var.label, 'sep': sep, 'elim': sorted(elim),
                        'doms': [states[l] for l in sep]})
    return sorted(out, key=lambda d: d['label'])


def nbe_counts(gm, sep, doms):
    """Sample counts for draws 0/1/2, from the code's own resolver."""
    cfg_val = gm.config.get('num_samples')
    w = len(sep)
    l = max(doms) if doms else 2
    if isinstance(cfg_val, str) and cfg_val.startswith('nbe'):
        parts = cfg_val.split(',')
        eps = float(parts[1]) if len(parts) > 1 and parts[1] != '' else 0.25
        total = FastBucket.compute_nbe_num_samples(w, l, eps)['total']
    else:
        total = int(cfg_val)
    bs = gm.config.get('batch_size') or 256
    set_size = gm.config.get('set_size') or total
    n_batched = (set_size // bs) * bs
    msg_size = 1.0
    for d in doms:
        msg_size *= d
    n_val = min(max(1, total // 9), msg_size)
    return {0: n_batched, 1: int(n_val), 2: n_batched}


def collisions(clusters, gm, n_draws=N_DRAWS_DEFAULT):
    """Every pair of DRAWS whose legacy seed is equal.

    seed = label + 10000*run_seed + 100*draw, so two draws collide iff
    label_a + 100*draw_a == label_b + 100*draw_b.
    """
    rows = []
    for c in clusters:
        counts = nbe_counts(gm, c['sep'], c['doms'])
        for d in range(n_draws):
            rows.append({'label': c['label'], 'draw': d,
                         'seed_part': c['label'] + 100 * d,
                         'n': counts[d], 'doms': tuple(c['doms']),
                         'sep': tuple(c['sep'])})
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r['seed_part'], []).append(r)
    out = []
    for seed_part, group in sorted(by_seed.items()):
        if len(group) < 2:
            continue
        for a, b in itertools.combinations(group, 2):
            if a['label'] == b['label'] and a['draw'] == b['draw']:
                continue                       # same draw, not a collision
            same_n = a['n'] == b['n']
            # torch.randint is called per column with high=domain_size, so two
            # streams agree on a leading run of columns only while the domain
            # sizes agree.
            shared = 0
            for da, db in zip(a['doms'], b['doms']):
                if da != db:
                    break
                shared += 1
            overlap = measure_overlap(a, b, run_seed=int(gm.config.get('seed', 42)))
            out.append({'seed_part': seed_part, 'a': a, 'b': b,
                        'same_n': same_n, 'shared_cols': shared,
                        'identical': same_n and a['doms'] == b['doms'],
                        'same_sep': a['sep'] == b['sep'],
                        'overlap': overlap})
    return rows, out


def _legacy_draw(seed, n, doms):
    """Byte-exact reproduction of the pre-CRN sampler for one draw.

    Same two lines as SampleGenerator._set_seed + sample_uniform, so what this
    measures is the real thing, not a model of it.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return torch.stack([torch.randint(0, int(d), (int(n),), dtype=torch.long)
                        for d in doms], dim=1)


def measure_overlap(a, b, run_seed):
    """Actually generate both colliding draws and measure how much they share.

    Reported:
      full_equal      the two assignment matrices are byte-identical
      eq_cols         number of LEADING columns that agree on all shared rows
      match_frac      fraction of entries agreeing in the shared (rows x cols) block
      chance_frac     the same fraction expected for two independent draws
    A collision that produces match_frac ~ chance_frac shared nothing in
    practice, however equal the seeds were.
    """
    # Row cap: a column's agreement under a shared seed is all-or-nothing (the
    # two draws either consume the same stream positions or they do not), so a
    # 20k-row prefix decides it, and `full_equal` does not depend on the cap at
    # all (equal row count + equal domain sizes + equal seed == equal matrix). The cap only exists because the nbe formula
    # asks for millions of rows on grid40x40 and materialising that is pointless
    # here. `full_equal` is still reported honestly as unknown-beyond-the-cap.
    ROW_CAP = 20000
    seed = 10000 * run_seed + a['seed_part']
    na, nb = min(a['n'], ROW_CAP), min(b['n'], ROW_CAP)
    capped = (a['n'] > ROW_CAP) or (b['n'] > ROW_CAP)
    xa = _legacy_draw(seed, na, a['doms'])
    xb = _legacy_draw(seed, nb, b['doms'])
    rows = min(xa.shape[0], xb.shape[0])
    cols = min(xa.shape[1], xb.shape[1])
    if rows == 0 or cols == 0:
        return {'full_equal': False, 'eq_cols': 0, 'match_frac': 0.0,
                'chance_frac': 0.0, 'rows': rows, 'cols': cols, 'capped': capped}
    ca, cb = xa[:rows, :cols], xb[:rows, :cols]
    eq = (ca == cb)
    eq_cols = 0
    for j in range(cols):
        if bool(eq[:, j].all()):
            eq_cols += 1
        else:
            break
    chance = sum(1.0 / max(a['doms'][j], b['doms'][j]) for j in range(cols)) / cols
    return {'full_equal': bool(a['n'] == b['n'] and a['doms'] == b['doms']
                               and xa.shape == xb.shape and torch.equal(xa, xb)),
            'eq_cols': eq_cols, 'match_frac': float(eq.double().mean()),
            'chance_frac': chance, 'rows': rows, 'cols': cols, 'capped': capped}


# ---------------------------------------------------------------------------
def cmd_validate(problem_key, iB, ecl, D, num_epochs):
    over = dict(neurobe_mode=True, iB=iB, ecl=ecl, num_samples='nbe,0.1',
                sampling_scheme='uniform', stream_nn_exact=True,
                dope_factors=True, device='cpu', seed=42,
                use_reduce_nn_merge=True, max_merge_bound=D,
                reduce_nn_backtrack=True, verbose_merge=False,
                num_epochs=num_epochs)
    gm0, _ = build(problem_key, over)
    pred = predicted_nn_clusters(gm0)
    print('PREDICTED %d NN clusters: %s'
          % (len(pred), [(c['label'], len(c['sep'])) for c in pred]))
    for c in pred:
        print('   label=%-5s w=%-3d counts=%s' % (c['label'], len(c['sep']),
                                                  nbe_counts(gm0, c['sep'], c['doms'])))

    seen = []
    original = SampleGenerator.sample_assignments

    def patched(self, num_samples=-1, sampling_scheme=None, is_validation=False):
        res = original(self, num_samples, sampling_scheme, is_validation)
        seen.append({'label': self.bucket.label, 'draw': self._last_draw_index,
                     'val': bool(is_validation), 'n': int(res.shape[0]),
                     'sep': tuple(self.message_scope),
                     'doms': tuple(int(d) for d in self.domain_sizes)})
        return res

    SampleGenerator.sample_assignments = patched
    try:
        cfg = prepare_config(dict(over), strict=False)
        with contextlib.redirect_stdout(io.StringIO()):
            gm = FastGM(model=catalog()[problem_key], nn_config=cfg, device='cpu')
            gm.eliminate_variables(all=True)
    finally:
        SampleGenerator.sample_assignments = original
    print('ACTUAL %d draws' % len(seen))
    for r in seen:
        print('   label=%-5s draw=%d val=%s n=%-7d w=%d' % (r['label'], r['draw'],
                                                            r['val'], r['n'], len(r['sep'])))
    act_labels = sorted({r['label'] for r in seen})
    print('predicted labels %r' % [c['label'] for c in pred])
    print('actual    labels %r' % act_labels)
    print('MATCH labels: %s' % (act_labels == [c['label'] for c in pred]))
    pred_counts = {c['label']: nbe_counts(gm0, c['sep'], c['doms']) for c in pred}
    ok = all(r['n'] == pred_counts.get(r['label'], {}).get(r['draw']) for r in seen)
    print('MATCH per-draw sample counts: %s' % ok)
    for r in seen:
        exp = pred_counts.get(r['label'], {}).get(r['draw'])
        if r['n'] != exp:
            print('   MISMATCH label=%s draw=%d actual=%s predicted=%s'
                  % (r['label'], r['draw'], r['n'], exp))
    return seen


OUT_PATH = [None]


def cmd_audit(cells, device):
    report = []
    for cell in cells:
        over = dict(neurobe_mode=True, iB=cell['iB'], ecl=cell['ecl'],
                    num_samples='nbe,0.1', sampling_scheme='uniform',
                    stream_nn_exact=True, dope_factors=True, device=device,
                    seed=42, verbose_merge=False, num_epochs=1,
                    reduce_nn_backtrack=bool(cell.get('backtrack')),
                    masked_net=bool(cell.get('masked')))
        arm = cell.get('arm', 'rnn')
        if cell['D'] is not None:
            over['max_merge_bound'] = cell['D']
        if arm == 'rnn':
            over['use_reduce_nn_merge'] = True
        elif arm == 'jt':
            over['use_join_tree_merge'] = True
        elif arm == 'nonsub':
            over['use_non_subsumption_merge'] = True
        elif arm == 'deg':
            over['merge_degree'] = cell['D']
        try:
            gm, _ = build(cell['key'], over)
        except Exception as e:
            print('SKIP %s %s D=%s: %s: %s' % (cell['key'], arm, cell['D'],
                                               type(e).__name__, str(e)[:90]), flush=True)
            continue
        clusters = predicted_nn_clusters(gm)
        rows, cols = collisions(clusters, gm)
        entry = {'key': cell['key'], 'iB': cell['iB'], 'D': cell['D'],
                 'arm': arm, 'backtrack': cell.get('backtrack'),
                 'masked': cell.get('masked'),
                 'n_vars': len(gm.vars), 'n_nn': len(clusters),
                 'labels': [c['label'] for c in clusters],
                 'n_collisions': len(cols),
                 'n_identical': sum(1 for c in cols if c['identical']),
                 'collisions': [
                     {'seed_part': c['seed_part'],
                      'a': (c['a']['label'], c['a']['draw'], c['a']['n'], len(c['a']['doms'])),
                      'b': (c['b']['label'], c['b']['draw'], c['b']['n'], len(c['b']['doms'])),
                      'identical': c['identical'], 'shared_cols': c['shared_cols'],
                      'same_sep': c['same_sep'], 'overlap': c['overlap']}
                     for c in cols]}
        report.append(entry)
        if OUT_PATH[0]:
            json.dump(report, open(OUT_PATH[0], 'w'))
        print('%-26s iB=%-3s %-7s D=%-5s nv=%-5d nn=%-3d collisions=%-3d identical=%d'
              % (cell['key'], cell['iB'], arm, cell['D'], entry['n_vars'],
                 entry['n_nn'], entry['n_collisions'], entry['n_identical']),
              flush=True)
        for c in entry['collisions']:
            o = c['overlap']
            print('      seed_part=%-6d  (label %s, draw %d, n=%d, w=%d)  vs  '
                  '(label %s, draw %d, n=%d, w=%d)  full_equal=%s eq_cols=%d/%d '
                  'match=%.4f chance=%.4f'
                  % (c['seed_part'], c['a'][0], c['a'][1], c['a'][2], c['a'][3],
                     c['b'][0], c['b'][1], c['b'][2], c['b'][3],
                     o['full_equal'], o['eq_cols'], o['cols'],
                     o['match_frac'], o['chance_frac']), flush=True)
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate', nargs=4, metavar=('KEY', 'IB', 'D', 'EPOCHS'))
    ap.add_argument('--audit', action='store_true')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out')
    args = ap.parse_args()

    if args.validate:
        key, iB, D, ep = args.validate
        iB = int(iB)
        cmd_validate(key, iB, 2 ** iB + 1, int(D), int(ep))
        sys.exit(0)

    if args.audit:
        # The faithful cell list: every distinct (problem, iB, ecl, merge arm,
        # bound, backtrack, masked_net) combination that appears among the
        # study's 2198 config yamls. 641 cells.
        raw = json.load(open('/tmp/claude-58902/study_cells.json'))
        cells = [{'key': k, 'iB': ib, 'ecl': ecl, 'arm': arm, 'D': D,
                  'backtrack': bt, 'masked': mask}
                 for (k, ib, ecl, arm, D, bt, mask) in raw]
        OUT_PATH[0] = args.out
        rep = cmd_audit(cells, args.device)
        if args.out:
            json.dump(rep, open(args.out, 'w'), indent=1)
            print('wrote %s' % args.out)
