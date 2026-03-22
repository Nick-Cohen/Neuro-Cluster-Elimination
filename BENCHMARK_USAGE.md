# NCE Benchmark Config Usage Guide

This guide demonstrates the cleaner config interface introduced in M005 using width-based threshold parameters for binary-domain problems.

## Width-Based Parameters for Binary Domains

**Old interface** (from `small_problems` benchmark, circa M001-M004):
```python
config = {
    'ecl': 524287,            # What width does this correspond to? (2^19-1 → width 19)
    'iB': 100,
    'approximation_method': 'nn',
    'bw_ecl': 8388607,        # What width does this correspond to? (2^23-1 → width 23)
    'backward_iB': 100,
    # ... rest of config
}
```

**New interface** (M005 width-based parameters):
```python
config = {
    'inference': {
        'ib2': 19,                    # Binary width 19 → ecl = 2^19-1 = 524287
        'i_bound': 100,
        'approximation_method': 'nn',
    },
    'backward': {
        'bw_ib2': 23,                 # Binary width 23 → bw_ecl = 2^23-1 = 8388607
        'backward_i_bound': 100,
    },
    # ... rest of config in nested sections
}
```

The width-based parameters (`ib2`, `bw_ib2`) make the config self-documenting: width 23 means "approximate buckets wider than 23 variables in binary domains." No mental arithmetic converting powers of 2.

## When to Use Width Parameters

- ✅ **Use `ib2`/`bw_ib2`** for binary-domain problems (all variables have 2 states)
- ❌ **Use `ecl`/`bw_ecl`** for multi-valued problems (domain size ≥ 3)

## Backward Compatibility

All existing configs using the old flat format with direct `ecl` values continue to work unchanged. The schema auto-detects flat vs nested format and handles both.
