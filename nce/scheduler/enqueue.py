"""Build or extend a rerun queue.

    python -m nce.scheduler.enqueue --queue Q.json \
        --problems grids/grid10x10.f10 pedigree/pedigree1 \
        --strategies reduce_nn nomerge \
        --bounds 4 8 --seeds 42 43 \
        --base-config base_config.json

Adding is idempotent: job ids are content-addressed, so re-running this with a
superset of arguments adds only the new cells and leaves completed work alone.

`--validate` (default on) reports which problems are missing from the local
`.model_cache` BEFORE anything is queued, because a missing model is a job that
would either hang on a no-timeout HTTP GET or silently re-download.
"""

from __future__ import annotations

import argparse
import json
import sys

from nce.scheduler.jobs import JobQueue, build_grid
from nce.scheduler import models as modelval


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--queue', required=True)
    p.add_argument('--problems', nargs='+', required=True)
    p.add_argument('--strategies', nargs='+', default=['reduce_nn'],
                   help='reduce_nn | nomerge | merge_degree')
    p.add_argument('--bounds', nargs='+', type=int, default=[4],
                   help='max_merge_bound values (ignored for nomerge)')
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    p.add_argument('--base-config', default=None,
                   help='JSON file of config applied to every job')
    p.add_argument('--tag', default='')
    p.add_argument('--no-validate', action='store_true')
    args = p.parse_args(argv)

    modelval.assert_cache_configured()

    base = {}
    if args.base_config:
        with open(args.base_config) as fh:
            base = json.load(fh)

    if not args.no_validate:
        rep = modelval.validate_many(args.problems)
        print('model cache: %s' % rep['cache_root'])
        for k in rep['ok']:
            print('  OK      %s' % k)
        for k, probs in rep['bad'].items():
            print('  MISSING %s' % k)
            for pr in probs:
                print('            %s' % pr)
        if rep['bad']:
            print('\n%d problem(s) will be BLOCKED at dispatch until fixed.'
                  % len(rep['bad']), file=sys.stderr)

    specs = build_grid(args.problems, args.strategies, args.bounds,
                       args.seeds, base_config=base, tag=args.tag)
    q = JobQueue(args.queue)
    added = q.add(specs)
    print('\nbuilt %d job(s), added %d new; queue now: %s'
          % (len(specs), added, q.summary()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
