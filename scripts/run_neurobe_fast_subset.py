#!/usr/bin/env python
"""Run neurobe_mode on the fast subset of problems to validate sampling fix.

These are problems that completed in ~5 seconds with sampling_scheme='all'.
With the fixed uniform sampling + 80/20 split, we expect them to take
a similar order of magnitude to NeuroBE C++ (seconds to low minutes).
"""
import time
import torch
from nce.benchmark_problems.neurobe_binary import neurobe_binary, _MODEL_KEYS, NEUROBE_NN_COUNTS
from nce.inference.graphical_model import FastGM

# Fast subset: small ecl (<=32767) and few NNs
FAST_INDICES = [2, 3, 7, 11]  # BN_3(1NN), BN_5(1NN), BN_10(2NN), 10_14_s(3NN)

def main():
    problems = neurobe_binary.problems
    configs = neurobe_binary.configs['neurobe']

    print(f"{'='*70}")
    print(f"Neurobe-mode FAST SUBSET validation (fixed uniform sampling)")
    print(f"{'='*70}\n")

    for idx in FAST_INDICES:
        key = _MODEL_KEYS[idx]
        model = problems[idx]
        config = configs[idx]
        nn_count = NEUROBE_NN_COUNTS.get(key, '?')

        print(f"[{key}]  ecl={config['ecl']}  expected_NNs={nn_count}")
        print(f"-" * 50)

        t0 = time.time()
        fastgm = FastGM(model=model, nn_config=config, device=config['device'])
        fastgm.eliminate_variables(all=True)
        elapsed = time.time() - t0

        log_z = fastgm.log_partition_function
        num_trained = fastgm.num_trained
        print(f"  ✓ log_Z={log_z:.6f}  NNs={num_trained}  time={elapsed:.1f}s\n")

        # Free GPU memory
        del fastgm
        torch.cuda.empty_cache()

    print("Done.")

if __name__ == '__main__':
    main()
