"""Merge the shard JSONs from tools_seed_collision_audit.py --only runs."""
import json
import sys

out = {}
for path in sys.argv[1:-1]:
    try:
        rep = json.load(open(path))
    except Exception as exc:
        print('skip %s (%s)' % (path, exc))
        continue
    for e in rep:
        key = (e['key'], e['iB'], e.get('arm'), e['D'], e.get('backtrack'),
               e.get('masked'))
        out[key] = e
    print('%-52s %4d cells' % (path, len(rep)))
merged = [out[k] for k in sorted(out, key=str)]
json.dump(merged, open(sys.argv[-1], 'w'), indent=1)
print('merged %d distinct cells -> %s' % (len(merged), sys.argv[-1]))
