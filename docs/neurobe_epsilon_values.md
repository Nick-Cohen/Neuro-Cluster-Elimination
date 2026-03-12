# NeuroBE Epsilon Values by Benchmark

## Epsilon Values from Run Scripts

| Benchmark | Epsilon | i-Bound | width_problem | var_dim (b) | Script |
|-----------|---------|---------|---------------|-------------|--------|
| Grid 20x20, f2 | **0.05** | 999 | 10 | 1 | `Sakshis-Superbuckets.../build_old/grid20_f2.sh` |
| Grid 20x20, f10 | **0.35** | 999 | 10 | 1 | `build_old/grid20_f10.sh` |
| Grid 40x40, f10 | **0.35** | 20 | 10 | 3 | `_build/grid40_f10.sh` |
| Grid 40x40, f10 (old) | **0.35** | 999 | 20 | 3 | `build_old/grid40_f10.sh` |
| DBN Ferro 22 | **0.1** | 999 | 20 | 3 | `build_old/dbn_ferro_22.sh` |
| Pedigree 41 | **0.1** | 999 | 20 | 3 | `Clean-NeuroBE/build/ped_41.sh` |
| Generic (run_problem.sh) | **0.35** | 20 | 10 | 3 | `_build/run_problem.sh` |

## Summary: Epsilon by Problem Type

| Problem Type | Epsilon | Rationale |
|-------------|---------|-----------|
| Grid (easy, f2) | 0.05 | Tighter bound → more samples → higher accuracy for easy problems |
| Grid (easy/hard, f10) | 0.35 | Looser bound → fewer samples; standard for grid benchmarks |
| Pedigree | 0.1 | Moderate bound; deterministic structure needs precision |
| DBN | 0.1 | Moderate bound; similar complexity profile to pedigrees |
| Default | 0.25 | Hardcoded in Config.h when not overridden by command line |

## Paper Context (η instead of ε)

The paper (NeuroBE, Agarwal et al.) uses a different parameterization:
`N = η * (L * b * w)^2 * log(b * w)` where η is tuned per benchmark.

The paper's tuning procedure:
1. Pick a representative problem instance with induced width w*
2. Set N(w*/2) ≈ 300k for hard problems, ≈ 100k for easy problems
3. Derive η from Eq. 3
4. Use the same η for all instances in that benchmark

Reported per-benchmark average sample counts (from paper):
- Pedigrees: h = 3w, N_avg ∈ [149k, 350k]
- DBN: h = {3w, 5w}, N_avg ∈ [80k, 180k]
- Grid-easy: h = w, N_avg ∈ [12k, 121k]
- Grid-hard: h = w, N_avg ∈ [60k, 209k]

## How Epsilon Maps to Sample Counts

Formula: `nSamples = floor((pd + ln(1000)) / epsilon)`

where `pd = temp * ln(temp/l)` and `temp = (l-1)*w^2 + l*w + 4`.

Example for w=20, l=3 (pd ≈ 4893):

| Epsilon | nSamples | n_train (80%) | n_val (20%) |
|---------|----------|---------------|-------------|
| 0.05 | 97,998 | 78,398 | 19,600 |
| 0.10 | 48,999 | 39,199 | 9,800 |
| 0.25 | 19,600 | 15,680 | 3,920 |
| 0.35 | 14,000 | 11,200 | 2,800 |

## Source Directories

All scripts are under `/home/cohenn1/SDBE/NeuroBE/`:
- `BE-sampling-project/BESampling/_build/` — current build scripts
- `BE-sampling-project/BESampling/build_old/` — legacy scripts
- `Clean-NeuroBE/NeuroBE/BE-sampling-project/BESampling/build/` — cleaned version
- `Sakshis-Superbuckets-ARE-NN-source-2021-12-08/` — historical with grid20_f2

Config default: `Config.h` sets `epsilon = 0.25` (overridden by `--epsilon` CLI arg).
