"""Summarise the collision audit and measure the CONSEQUENCE of a collision.

The audit answers "do two draws share a seed". This answers "so what": when two
different clusters share a seed, how much of their sampled data is actually
shared, and does the sharing line up on the variables they have in common?
"""
import collections
import json
import sys

import torch

rep = json.load(open(sys.argv[1]))

print('=' * 78)
print('CELLS AUDITED: %d   (distinct (problem, iB, arm, bound, backtrack, masked)'
      ' combinations appearing in the study\'s 2198 config yamls)' % len(rep))

by_problem = collections.defaultdict(lambda: {'cells': 0, 'coll_cells': 0,
                                              'coll': 0, 'ident': 0, 'nv': 0})
tot_coll = tot_ident = coll_cells = 0
eqcol_hist = collections.Counter()
for e in rep:
    p = by_problem[e['key']]
    p['cells'] += 1
    p['nv'] = e['n_vars']
    p['coll'] += e['n_collisions']
    p['ident'] += e['n_identical']
    if e['n_collisions']:
        p['coll_cells'] += 1
        coll_cells += 1
    tot_coll += e['n_collisions']
    tot_ident += e['n_identical']
    for c in e['collisions']:
        eqcol_hist[c['overlap']['eq_cols']] += 1

print('CELLS WITH >=1 SEED COLLISION: %d / %d' % (coll_cells, len(rep)))
print('TOTAL COLLIDING DRAW PAIRS   : %d' % tot_coll)
print('OF WHICH BYTE-IDENTICAL      : %d' % tot_ident)
print()
print('%-26s %6s %6s %8s %8s %8s' % ('problem', 'nvars', 'cells', 'coll_cells',
                                     'coll', 'identical'))
for k in sorted(by_problem):
    v = by_problem[k]
    print('%-26s %6d %6d %8d %8d %8d'
          % (k, v['nv'], v['cells'], v['coll_cells'], v['coll'], v['ident']))

print()
print('LEADING COLUMNS SHARED, over all colliding pairs:')
for k in sorted(eqcol_hist):
    print('   eq_cols=%-3d : %d pairs' % (k, eqcol_hist[k]))

# which (draw_a, draw_b) role pairs actually occur, and which are byte-identical
roles = collections.Counter()
roles_ident = collections.Counter()
for e in rep:
    for c in e['collisions']:
        key = tuple(sorted((c['a'][1], c['b'][1])))
        roles[key] += 1
        if c['identical']:
            roles_ident[key] += 1
ROLE = {0: 'normalisation-init', 1: 'nbe validation', 2: 'training'}
print()
print('COLLIDING DRAW-ROLE PAIRS (draw 0 = normalisation sample, discarded; '
      'draw 1 = NeuroBE early-stopping validation set; draw 2 = training set):')
for k in sorted(roles):
    print('   draw %d (%s) x draw %d (%s): %d pairs, %d byte-identical'
          % (k[0], ROLE[k[0]], k[1], ROLE[k[1]], roles[k], roles_ident[k]))

# --- match fraction vs chance -------------------------------------------
excess = []
for e in rep:
    for c in e['collisions']:
        o = c['overlap']
        excess.append(o['match_frac'] - o['chance_frac'])
if excess:
    excess.sort()
    print()
    print('match_frac - chance_frac over %d colliding pairs:' % len(excess))
    print('   min %.4f   median %.4f   max %.4f'
          % (excess[0], excess[len(excess) // 2], excess[-1]))
    print('   (0 == the two draws share nothing beyond chance; 0.5 == they are '
          'the same matrix)')

# --- identical pairs: does the sharing line up on shared variables? -----
ident_pairs = [(e, c) for e in rep for c in e['collisions'] if c['identical']]
print()
if not ident_pairs:
    print('NO byte-identical colliding pair anywhere in the study cell list.')
else:
    print('BYTE-IDENTICAL PAIRS: %d. Consequence check below.' % len(ident_pairs))
    for e, c in ident_pairs[:20]:
        print('   %s iB=%s %s D=%s : labels %s/%s draws %s/%s'
              % (e['key'], e['iB'], e.get('arm'), e['D'],
                 c['a'][0], c['b'][0], c['a'][1], c['b'][1]))
