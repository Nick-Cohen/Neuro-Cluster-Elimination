"""Emit the doc-46 appendix A.3 markdown table from the audit shards."""
import collections
import json
import sys

rows = {}
for path in sys.argv[1:]:
    try:
        rep = json.load(open(path))
    except Exception:
        continue
    for e in rep:
        rows[(e['key'], e['iB'], e.get('arm'), e['D'], e.get('backtrack'),
              e.get('masked'))] = e
rep = list(rows.values())

agg = collections.defaultdict(lambda: {'cells': 0, 'coll_cells': 0, 'coll': 0,
                                       'ident': 0, 'nv': 0, 'arms': set()})
for e in rep:
    a = agg[e['key']]
    a['cells'] += 1
    a['nv'] = e['n_vars']
    a['coll'] += e['n_collisions']
    a['ident'] += e['n_identical']
    if e['n_collisions']:
        a['coll_cells'] += 1
        a['arms'].add(e.get('arm'))

print('| problem | vars | cells audited | cells with a collision | colliding draw pairs | byte-identical |')
print('|---|---|---|---|---|---|')
tot = collections.Counter()
for k in sorted(agg):
    v = agg[k]
    tot['cells'] += v['cells']
    tot['coll_cells'] += v['coll_cells']
    tot['coll'] += v['coll']
    tot['ident'] += v['ident']
    print('| `%s` | %d | %d | %d | %d | %d |'
          % (k, v['nv'], v['cells'], v['coll_cells'], v['coll'], v['ident']))
print('| **total** | | **%d** | **%d** | **%d** | **%d** |'
      % (tot['cells'], tot['coll_cells'], tot['coll'], tot['ident']))

roles = collections.Counter()
roles_ident = collections.Counter()
for e in rep:
    for c in e['collisions']:
        key = tuple(sorted((c['a'][1], c['b'][1])))
        roles[key] += 1
        if c['identical']:
            roles_ident[key] += 1
print()
print('role pairs:')
for k in sorted(roles):
    print('   draw %d x draw %d : %d pairs, %d byte-identical' % (k[0], k[1], roles[k], roles_ident[k]))

n_meas = sum(1 for e in rep for c in e['collisions'] if c['overlap'] is not None)
eq = collections.Counter()
excess = []
viol = 0
for e in rep:
    for c in e['collisions']:
        o = c['overlap']
        if o is None:
            continue
        eq[o['eq_cols']] += 1
        excess.append(o['match_frac'] - o['chance_frac'])
        if c['identical'] != o['full_equal'] or (not c['identical'] and o['eq_cols'] > 1):
            viol += 1
excess.sort()
print()
print('measured pairs: %d ; eq_cols histogram %r ; rule violations %d'
      % (n_meas, dict(sorted(eq.items())), viol))
if excess:
    print('excess over chance: min %.4f median %.4f max %.4f'
          % (excess[0], excess[len(excess) // 2], excess[-1]))
