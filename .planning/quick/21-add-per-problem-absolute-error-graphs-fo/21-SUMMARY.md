---
phase: quick-21
status: complete
duration: ~2 min
commits:
  - hash: pending
    scope: "feat(quick-21): add no_exact_bw graphs for 9 pattern-1 problems"
---

## Summary

Added per-problem absolute error bar charts for 9 pattern-1 problems (where only `ukl_bw30` failed due to CUDA OOM but the other 4 configs completed successfully).

### What was done

- Extended `visualize_updated.py` with a new section that identifies pattern-1 problems and generates graphs using only the 4 non-bw30 configs
- Created `no_exact_bw/` subfolder under `results/updated_graphs_and_table/`
- Generated 18 PNG files (9 problems x linear + symlog scale)

### Pattern-1 problems (9 total)

1. 10_14_s.binary.uai
2. 10_16_s.binary.uai
3. 11_4_s.binary.uai
4. 29.wcsp.uai
5. 404.wcsp.uai
6. deer_rescaled_0034.K15.F1.5.model.uai
7. deer_rescaled_0034.K20.F1.25.model.uai
8. deer_rescaled_0294.K10.F1.75.model.uai
9. or_chain_10.fg.uai

### Output

- 18 PNG files in `results/updated_graphs_and_table/no_exact_bw/`
- Same color scheme and label format as clean-problem graphs (minus blue UKL bw=exact bar)
