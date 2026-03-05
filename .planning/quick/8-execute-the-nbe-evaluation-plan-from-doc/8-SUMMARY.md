# Quick Task 8 Summary: Execute NBE Evaluation Plan

## Bugs Fixed (4)

1. **ecl=2^22 hardcoded** → `2**_IB_MAP[key]` per-model in nbe_sanity_check.py
2. **approximation_method='wmb'** → `'nn'` so large buckets train NNs
3. **set_size=50000** → `None` (defaults to num_samples); train.py handles None
4. **num_trained double-counted** → removed duplicate increment from process_bucket

Also: try/except for catalog model.order (network failures), copied grid40x40 ord file.

## Phase Results

| Phase | Model | Log Z | Num Trained | Time | Status |
|-------|-------|-------|-------------|------|--------|
| 0a (exact) | grid10x10 | 169.408 | 0 | 6.7s | OK |
| 0b (1 bucket) | pedigree13 | N/A | 1 | 29.7s | OK |
| 1 practice | grid10x10 | 263.86 | 26 | ~60s | OK |
| 1 full (500ep) | grid10x10 | 154.29 | 26 | 428s | OK |

Phase 1 absolute error vs exact: **15.12**

## Open Issues

- **rbm_20**: iB=20 = model width → 0 NN buckets. Need correct iB from user.
- **Phase 1 error**: 15.12 gap needs investigation (loss fn tuning, more samples?)
- **Phases 2-5**: Not yet executed.

## Commits

- `541a170` fix(nbe): correct ecl, approximation_method, and set_size
- `ea208ef` fix(nbe): remove duplicate num_trained increment
