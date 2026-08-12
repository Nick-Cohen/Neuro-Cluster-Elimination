#!/usr/bin/env python3
"""Worker subprocess for run_small_problems.py.

Runs a single problem and writes the result to a JSON file.
CUDA_VISIBLE_DEVICES should be set by the coordinator.

Usage (called by run_small_problems.py, not directly):
    CUDA_VISIBLE_DEVICES=0 python scripts/run_small_problems_worker.py \
        --problem-index 3 --config-json '{"num_epochs": 10000, ...}' \
        --result-path /tmp/result_3.json
"""
import argparse
import copy
import json
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem-index', type=int, required=True)
    parser.add_argument('--config-json', type=str, required=True)
    parser.add_argument('--result-path', type=str, required=True)
    parser.add_argument('--ecl-override', type=int, default=None)
    args = parser.parse_args()

    from nce.benchmark_problems.small_problems import small_problems, _AUTO_ECL
    from nce.config_schema import prepare_config
    from nce.inference.graphical_model import FastGM

    idx = args.problem_index
    model = small_problems.problems[idx]
    base_config = json.loads(args.config_json)
    ecl = args.ecl_override if args.ecl_override is not None else _AUTO_ECL[model.modelfile]

    # Merge config with per-problem ecl and default iB=100
    config = copy.deepcopy(base_config)
    if any(k in config for k in ('inference', 'nn', 'training', 'sampling', 'backward', 'output')):
        config.setdefault('inference', {})
        config['inference']['ecl'] = ecl
        config['inference'].setdefault('i_bound', 100)
    else:
        config['ecl'] = ecl
        config.setdefault('iB', 100)

    config = prepare_config(config, strict=False)
    device = config.get('device', 'cuda')

    start = time.time()
    try:
        fastgm = FastGM(model=model, nn_config=config, device=device)
        log_Z = fastgm.get_log_partition_function()
        duration = time.time() - start

        result = {
            'problem': model.modelfile,
            'index': idx,
            'log_Z': float(log_Z),
            'ecl': ecl,
            'duration_seconds': round(duration, 2),
            'status': 'ok',
        }
    except Exception as e:
        duration = time.time() - start
        result = {
            'problem': model.modelfile,
            'index': idx,
            'log_Z': None,
            'ecl': ecl,
            'duration_seconds': round(duration, 2),
            'status': f'error: {e}',
        }

    with open(args.result_path, 'w') as f:
        json.dump(result, f)

    sys.exit(0 if result['status'] == 'ok' else 1)


if __name__ == '__main__':
    main()
