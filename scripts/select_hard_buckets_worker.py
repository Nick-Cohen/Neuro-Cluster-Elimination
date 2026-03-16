#!/usr/bin/env python
"""Phase 1 worker: train one problem with error tracking, write per-bucket error data.

Subprocess entry point — launched by select_hard_buckets.py with CUDA_VISIBLE_DEVICES
already set. Imports torch only after env is configured.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/select_hard_buckets_worker.py \
        --problem-index 3 --output-path /tmp/results/problem_3.json
"""
import argparse
import json
import sys
import traceback
import copy


def main():
    parser = argparse.ArgumentParser(description="Train one small_problems model with error tracking")
    parser.add_argument('--problem-index', type=int, required=True,
                        help='Index into small_problems.problems (0-23)')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Path for JSON results file')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Hard bucket threshold (for summary reporting only)')
    args = parser.parse_args()

    idx = args.problem_index

    try:
        # Imports after CUDA_VISIBLE_DEVICES is set
        from nce.benchmark_problems.small_problems import small_problems
        from nce.config_schema import prepare_config
        from nce.inference.graphical_model import FastGM

        model = small_problems.problems[idx]
        config = copy.deepcopy(small_problems.configs['default'][idx])

        problem_key = None
        for i, m in enumerate(small_problems.problems):
            if i == idx:
                # Reconstruct key from model
                problem_key = model.modelfile
                break

        auto_ecl = config['ecl']

        # Override config for error-tracked selection run
        config['error_tracking'] = True
        config['loss_fn'] = 'unnormalized_kl'
        config['sampling_scheme'] = 'all'
        config['num_epochs'] = 10000
        config['bw_ecl'] = config['ecl']  # backward info at auto_ecl level
        config['populate_bw_factors'] = True
        config['use_bw_approx'] = True

        config = prepare_config(config)

        fastgm = FastGM(model=model, nn_config=config, device='cuda')
        fastgm.eliminate_variables(all=True)

        # Extract error tracking data
        # Format: list of (bucket_label, [(epoch, loss, log_Z_err, abs_log_Z_err), ...])
        buckets_data = []
        for bucket_label, epoch_data in fastgm.error_tracking_data:
            final_entry = epoch_data[-1] if epoch_data else None
            final_abs_log_Z_err = final_entry[3] if final_entry else None
            buckets_data.append({
                'label': int(bucket_label),
                'error_data': [(int(e), float(l), float(z), float(a)) for e, l, z, a in epoch_data],
                'final_abs_log_Z_err': float(final_abs_log_Z_err) if final_abs_log_Z_err is not None else None,
                'num_epochs': int(final_entry[0]) if final_entry else 0,
            })

        num_hard = sum(1 for b in buckets_data
                       if b['final_abs_log_Z_err'] is not None
                       and b['final_abs_log_Z_err'] > args.threshold)

        result = {
            'problem_index': idx,
            'problem_key': problem_key,
            'model_file': model.modelfile,
            'auto_ecl': auto_ecl,
            'buckets': buckets_data,
        }

        with open(args.output_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"[Problem {idx}] {problem_key} — {len(buckets_data)} NN buckets, {num_hard} hard")

    except Exception as e:
        error_result = {
            'problem_index': idx,
            'error': str(e),
            'traceback': traceback.format_exc(),
        }
        with open(args.output_path, 'w') as f:
            json.dump(error_result, f, indent=2)
        print(f"[Problem {idx}] FAILED: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
