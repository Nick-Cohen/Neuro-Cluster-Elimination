# %% [markdown]
# # Phase 3: Full Benchmark Evaluation on All 5 Problems
#
# Run NBE on all 5 benchmark problems using the benchmark NBE config.
# Records per-problem: num_vars, num_trained, log_z, time.
#
# NOTE: With ecl=2^22, all 5 problems may run exactly (num_trained=0) because
# no mini-bucket message exceeds the ecl threshold. This is a valid baseline
# result showing the exact log Z from WMB elimination with iB as the i-bound.
#
# Index mapping (from nbe_sanity_check):
#   0 = pedigree13      (1077 vars, width 32, iB=20)
#   1 = grid40x40.f10   (1600 vars, width 54, iB=20)
#   2 = grid20x20.f10   (400 vars, width 26, iB=10)
#   3 = rbm_20          (40 vars, width 20, iB=20)
#   4 = grid10x10.f5.wrap (100 vars, width 21, iB=10)

# %%
import time
import traceback
from nce.benchmark_problems import nbe_sanity_check
from nce.inference.graphical_model import FastGM

print("=" * 70)
print("Phase 3: Full Benchmark Evaluation")
print("=" * 70)
print(f"Problems: {len(nbe_sanity_check.problems)}")
print(f"Using NBE benchmark config (ecl=2^22, device=cuda)")
print()

# %%
# Results storage
results = []

PROBLEM_NAMES = [
    'pedigree13',
    'grid40x40.f10',
    'grid20x20.f10',
    'rbm_20',
    'grid10x10.f5.wrap',
]

# %%
# Run each problem
for idx in range(len(nbe_sanity_check.problems)):
    model = nbe_sanity_check.problems[idx]
    problem_name = PROBLEM_NAMES[idx]

    print(f"\n{'=' * 70}")
    print(f"Problem {idx+1}/5: {problem_name}")
    print(f"Num vars: {model.num_vars}, Width: {model.width}")
    print('=' * 70)

    try:
        # Create a fresh copy of the benchmark NBE config
        config = dict(nbe_sanity_check.configs['nbe'][idx])
        config['device'] = 'cuda'

        print(f"Config: iB={config['iB']}, ecl=2^{config['ecl'].bit_length()-1}, "
              f"num_epochs={config['num_epochs']}")
        print(f"  hidden_sizes={config['hidden_sizes']}, num_samples={config['num_samples']}")

        # Create FastGM and run inference
        t0 = time.time()
        fastgm = FastGM(model=model, nn_config=config, device='cuda')
        t_create = time.time() - t0
        print(f"FastGM created in {t_create:.2f}s")

        t0 = time.time()
        log_z = fastgm.get_log_partition_function()
        t_infer = time.time() - t0

        result = {
            'problem_name': problem_name,
            'num_vars': model.num_vars,
            'width': model.width,
            'iB': config['iB'],
            'log_z': float(log_z),
            'num_trained': fastgm.num_trained,
            'time': t_infer,
            'status': 'OK',
            'error': None,
        }
        print(f"Result: log Z = {log_z:.4f}, num_trained = {fastgm.num_trained}, time = {t_infer:.1f}s")

    except Exception as e:
        print(f"\nERROR in problem {problem_name}: {e}")
        traceback.print_exc()
        result = {
            'problem_name': problem_name,
            'num_vars': model.num_vars if hasattr(model, 'num_vars') else 'N/A',
            'width': model.width if hasattr(model, 'width') else 'N/A',
            'iB': 'N/A',
            'log_z': None,
            'num_trained': None,
            'time': None,
            'status': 'ERROR',
            'error': str(e),
        }

    results.append(result)

# %%
# Print summary table (matching the eval plan format)
print()
print("=" * 90)
print("SUMMARY TABLE - Phase 3: Full Benchmark Evaluation")
print("=" * 90)
header = (f"{'Problem':<22} {'num_vars':>9} {'width':>6} {'iB':>4} "
          f"{'num_trained':>12} {'log_Z_estimate':>16} {'time(s)':>9} {'status':>8}")
print(header)
print("-" * 90)
for r in results:
    log_z_str = f"{r['log_z']:.4f}" if r['log_z'] is not None else "ERROR"
    num_trained_str = str(r['num_trained']) if r['num_trained'] is not None else "N/A"
    time_str = f"{r['time']:.1f}" if r['time'] is not None else "N/A"
    width_str = str(r['width']) if r['width'] is not None else "N/A"
    ib_str = str(r['iB']) if r['iB'] is not None else "N/A"
    print(f"{r['problem_name']:<22} {r['num_vars']:>9} {width_str:>6} {ib_str:>4} "
          f"{num_trained_str:>12} {log_z_str:>16} {time_str:>9} {r['status']:>8}")
print()

# Note about num_trained
all_zero = all(r['num_trained'] == 0 for r in results if r['num_trained'] is not None)
if all_zero:
    print("Note: num_trained=0 for all problems.")
    print("With ecl=2^22, all mini-bucket messages fit within the exact computation limit.")
    print("The results above are WMB exact eliminations (WMB-BE with given iB), not NBE approximations.")
    print()
    print("To test actual NN training, lower ecl (e.g., ecl=2^15) or use larger models.")

print()
print("Phase 3 complete.")
