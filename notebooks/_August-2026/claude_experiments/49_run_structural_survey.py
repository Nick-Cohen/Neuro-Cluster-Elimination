#!/usr/bin/env python
"""Q30/Q31 structural survey over the paper benchmark set.

Build-only: no training, no message computation, no GPU. Uses the reusable
machinery in nce/analysis/structure_survey.py.

Usage:  python 49_run_structural_survey.py [out.json]
"""
import sys, json, time, traceback

sys.path.insert(0, '/home/cohenn1/NCE-wt-survey')

from nce.analysis import survey_problem                      # noqa: E402
from nce.benchmark_problems.catalog_utils import get_catalog  # noqa: E402

BENCH = '/home/cohenn1/NCE/notebooks/June-2026/claude_experiments/reduce_nn_experiment/benchmark_set.json'
D_VALUES = [8, 16, 24]

STRATS = [('nomerge', None)] + [(s, D) for s in ('subsumption', 'reduce_nn') for D in D_VALUES]


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/survey49.json'
    # optional: restrict to one group index (for trivial process-level parallelism)
    only = int(sys.argv[2]) if len(sys.argv) > 2 else None
    only_p = int(sys.argv[3]) if len(sys.argv) > 3 else None
    bench = json.load(open(BENCH))
    # the cache resolves relative to the package dir; when running from a worktree
    # that is a partial copy, so point at the main checkout's cache explicitly.
    cat = get_catalog(cache_dir='/home/cohenn1/NCE/.model_cache')

    rows, cluster_rows, failures = [], [], []
    for gi, g in enumerate(bench['groups']):
        if only is not None and gi != only:
            continue
        iB, ecl = g['iB'], g['ecl']
        for pi, p in enumerate(g['problems']):
            if only_p is not None and pi != only_p:
                continue
            key = p['key']
            for strat, D in STRATS:
                tag = f'{key}|iB{iB}|{strat}' + (f'|D{D}' if D else '')
                t0 = time.time()
                try:
                    s = survey_problem(key, iB=iB, ecl=ecl, strategy=strat,
                                       max_merge_bound=D, catalog=cat)
                except Exception as e:
                    failures.append({'tag': tag, 'error': repr(e)})
                    print(f'FAIL {tag}: {e}', flush=True)
                    traceback.print_exc()
                    continue
                d = s.summary()
                d.update({'group': g['group'], 'D': D, 'secs': round(time.time() - t0, 2)})
                rows.append(d)
                # keep the interesting clusters in full
                for c in s.clusters:
                    if c.n_nn_factors >= 1 or c.s1_streams or c.s4_streams:
                        cluster_rows.append({
                            'tag': tag, 'group': g['group'], 'problem': key,
                            'strategy': strat, 'D': D, 'iB': iB,
                            'key_label': c.key_label, 'path': c.path,
                            'n_elim': len(c.elim_labels), 'width': c.width,
                            'n_factors': c.n_factors, 'n_nn': c.n_nn_factors,
                            'elim_prod': c.elim_prod, 'joint_numel': c.joint_numel,
                            's1_streams': c.s1_streams, 's4_streams': c.s4_streams,
                            's4_n_blocks': c.s4_n_blocks,
                            'n_enc_groups': c.n_shared_encoding_groups,
                            'nn_facs': [{'w': f.width,
                                         's1_elim_in_scope': f.s1_elim_vars_in_scope,
                                         's1_distinct': f.s1_distinct_points,
                                         's1_red': f.s1_redundancy,
                                         's4_rows': f.s4_rows_per_eval,
                                         's4_distinct': f.s4_distinct_blocks,
                                         's4_red': f.s4_redundancy}
                                        for f in c.nn_factors],
                        })
                print(f'{tag:<62} nn={d["n_nn_clusters"]:<3} maxNNco={d["max_co_resident_nn"]:<3} '
                      f'S1j={d["n_s1_joint"]:<3} S4j={d["n_s4_joint"]:<3} '
                      f'maxS1red={d["max_s1_redundancy"]:.3g} ({d["secs"]}s)', flush=True)

    json.dump({'rows': rows, 'clusters': cluster_rows, 'failures': failures},
              open(out_path, 'w'), indent=1)
    print(f'\nwrote {out_path}: {len(rows)} rows, {len(cluster_rows)} clusters, '
          f'{len(failures)} failures')


if __name__ == '__main__':
    main()
