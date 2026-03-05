# Quick Task 9: Change ecl from 2^iB to 2^(iB-1)

Changed `ecl: 2**_IB_MAP[key]` → `ecl: 2**(_IB_MAP[key] - 1)` in nbe_sanity_check.py.

rbm_20 now has 20 NN-eligible buckets (was 0). All 5 models confirmed with large_buckets > 0.

Commit: 59d5dc2
