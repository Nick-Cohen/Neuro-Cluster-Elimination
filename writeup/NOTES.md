# Paper draft — working notes & provenance

Companion to `paper.md`. Tracks where every number came from and what still needs
author input before this is submission-ready.

## Provenance of the headline numbers (paper §5.2)

The four-row headline table uses the **hand-verified curated numbers** recorded in
`lab_notebook.txt` (entries 2026-06-03 and 2026-06-03 cont. / cont. 2 / 2026-06-04),
which are the values Nick checked at the end of the final experiment. These are the
authoritative story:

- `grid10x10.f10.wrap` iB10: D=1 err 3.81 → D≈8–16 err 0.11 (29 NN), 6.7×, faster.
- `grid20x20.f10` iB10: D=1 err 15.84 → D≈8–16 err 0.16 (7 NN), ~100×, faster.
- `pedigree13` iB20: D=1 err 2.49 (126 NN) → D=16 err 0.37 (26 NN), 6.7×.
  - full curve: cap1 2.49(126) · cap4 0.74(39) · cap8 0.42(29) · cap12 0.38(26) ·
    cap16 0.37(26)* · cap24 0.38(26).
- `rbm_22` iB10: D=1 err 0.648 (33 NN, 22.6m) → D=6 err 0.094 (23 NN, 10.2m), 7×.
  - low-cap curve: cap1 0.648(33) · cap2 0.165(27) · cap4 0.162(24) · cap6 0.094(23).

⚠️ **Discrepancy to resolve.** The automated `reduce_nn_experiment/results/` JSONs
(aggregated below) are NOT always identical to the curated numbers — they come from a
later/broader sweep with the `reduce_nn` backtracking variants and different
seeds/sampling, so some cells are non-monotone in D (e.g. grid10x10.f10.wrap D=16
shows 10 NN / err 1.59, an over-merge artifact of the automated cap logic). Before
publishing, **pick one canonical run per cell** and regenerate the table + figures
from it. Do not mix the two sources in the final paper.

## §5.3 cost-cliff table

From `notebooks/May-2026/claude_experiments/subsumption_merge_analysis/merge_degree_sweep.csv`
(pedigree13, iB20, ecl=1048577). Columns used: `cap`, `nn_merged`, `max_elim`,
`log2_total_sc`. This is a **static topology** computation (no training), so it is
exact and reproducible.

## §5.4 full automated sweep (77 cells)

Aggregated from `notebooks/June-2026/claude_experiments/reduce_nn_experiment/results/**/*.json`
(fields: `problem`, `iB`, `max_merge_bound`/`max_cluster_size`, `num_trained`,
`abs_err`, `wall_time_s`). Deduped to one row per (problem, iB, cap):

```
problem                       iB   cap   NN      err   time_s
dbn/rbm_20                    10     4   19   0.4259    342.5
dbn/rbm_20                    10    12    9   0.2771    232.0
dbn/rbm_20                    10    16   19   0.0991    184.1
dbn/rbm_20                    20     4    0   0.0000      2.1
dbn/rbm_20                    20    12    0   0.0000      2.2
dbn/rbm_20                    20    16    0   0.0000      2.2
dbn/rbm_21                    10     4   20   0.4074    365.8
dbn/rbm_21                    10    12   10   0.9544    252.7
dbn/rbm_21                    10    16   20   0.1812    231.1
dbn/rbm_21                    20     4   18   0.9091    216.1
dbn/rbm_21                    20    12   10   0.5058    261.4
dbn/rbm_21                    20    16   20   0.1468    358.1
dbn/rbm_22                    10     4   21   0.0477    385.1
dbn/rbm_22                    10    12   11   0.0189    333.6
dbn/rbm_22                    10    16   21   0.0205    281.3
dbn/rbm_22                    10    18    5   0.2307   2573.1
dbn/rbm_22                    10    24    1   0.8936  47072.0
dbn/rbm_22                    10    28    1   0.2358  61506.7
dbn/rbm_22                    20     4   19   0.3301    266.6
dbn/rbm_22                    20    12   11   0.1093    313.7
dbn/rbm_22                    20    16   21   0.1497    335.2
dbn/rbm_ferro_20             20     4    0   0.0002      2.2
dbn/rbm_ferro_20             20    12    0   0.0002      2.1
dbn/rbm_ferro_20             20    16    0   0.0001      2.2
dbn/rbm_ferro_21             20     4   18   0.4309    449.5
dbn/rbm_ferro_21             20    12   10   0.0550    331.6
dbn/rbm_ferro_21             20    16   20   0.0492    367.6
dbn/rbm_ferro_22             20     4   19   1.7817    381.2
dbn/rbm_ferro_22             20    12   11   2.6922    472.2
dbn/rbm_ferro_22             20    16   21   1.6019    305.0
grids/grid10x10.f10          10     4    1   0.5830     19.9
grids/grid10x10.f10          10    12    0   0.0002      2.2
grids/grid10x10.f10          10    16    0   0.0002      2.1
grids/grid10x10.f10.wrap     10     4    7   0.2363    427.8
grids/grid10x10.f10.wrap     10     8    3   0.4160     63.5
grids/grid10x10.f10.wrap     10    12    2   0.1042    110.3
grids/grid10x10.f10.wrap     10    16   10   1.5857    209.3
grids/grid10x10.f10.wrap     10    20    1   0.0514    146.2
grids/grid10x10.f10.wrap     10    24    1   0.2900   1281.3
grids/grid10x10.f10.wrap     10    28    0   0.0002     42.7
grids/grid20x20.f10          10     4   28   3.0503    347.7
grids/grid20x20.f10          10    12   10   0.2602    380.0
grids/grid20x20.f10          10    16   36   0.7682    621.4
grids/grid20x20.f10          10    20    6   1.1695   3239.2
grids/grid20x20.f10          10    24    4   0.2236  28569.7
grids/grid20x20.f15          10     4   28   0.9628    315.2
grids/grid20x20.f15          10    12   10   0.2816    357.5
grids/grid20x20.f15          10    16    7   0.0405    277.7
grids/grid20x20.f2           10     4   28   0.3247    412.8
grids/grid20x20.f2           10    12   10   0.1665    332.1
grids/grid20x20.f2           10    16   36   0.1203    627.4
grids/grid20x20.f5           10     4   28   2.9929    332.2
grids/grid20x20.f5           10    12   10   0.0393    392.3
grids/grid20x20.f5           10    16   36   0.1119    678.3
grids/grid40x40.f10          20     4   69  54.5879   1647.6
grids/grid40x40.f10          20    12   22  21.8975   1216.3
grids/grid40x40.f10          20    22   12   1.1768 168935.8
pedigree/pedigree13          20     4   30   1.4675   1012.1
pedigree/pedigree13          20    12    8   0.1620   3154.6
pedigree/pedigree13          20    16   26   0.4455   1524.8
pedigree/pedigree13          20    20    4   0.2665   6031.2
pedigree/pedigree13          20    24    3   0.1159  21786.7
pedigree/pedigree19          20     4   18   0.1541   1827.6
pedigree/pedigree19          20    12    4   0.0497    887.9
pedigree/pedigree19          20    16   21   0.8297   3988.5
pedigree/pedigree34          20     4   28   1.1119   2064.3
pedigree/pedigree34          20    12    7   0.4897   3255.6
pedigree/pedigree34          20    16   30   0.5143   4138.4
pedigree/pedigree41          20     4   21   0.9740   2702.4
pedigree/pedigree41          20    12    6   0.4150   1074.8
pedigree/pedigree41          20    16   19   0.6369   1761.8
pedigree/pedigree51          20     4   32   0.9073   3429.4
pedigree/pedigree51          20    12   10   2.7928   3993.5
pedigree/pedigree51          20    16   29   2.9952   3107.1
pedigree/pedigree7           20     4   30   2.9340    942.9
pedigree/pedigree7           20    12    9   1.5239   3192.2
pedigree/pedigree7           20    16   35   1.9145   2344.5
```

Note the recurring pattern that D=12 (a true degree-bounded merge) often beats both
D=4 and the D=16 `reduce_nn` rows — consistent with the 8–16 sweet-spot claim, but it
shows the automated cap semantics differ across rows. Reconcile before publishing.

## Open items (mirror of the [TODO]s in paper.md)

1. **Canonical run set.** Choose one experiment configuration per (problem, D) cell;
   re-run any gaps so the main table and figures come from a single consistent sweep.
   Resolve the curated-vs-automated discrepancy (see warning above).
2. **Reference-Z.** Document the ground-truth Z source per problem (exact / converged
   solver / high-iB WMB). Some hard problems (grid40x40, pedigree51) had nan refs
   historically — see memory `reference_grid40x40_refz`.
3. **Citations.** NeuroBE original paper; Dechter bucket elimination; Dechter & Rish
   mini-bucket; Liu & Ihler weighted mini-bucket; join-graph / IJGP; learned-inference
   related work.
4. **Sampling details.** State sampling scheme + sample count + seeds/trials for the
   headline runs.
5. **Figures.** Regenerate publication-quality versions of accuracy/time-vs-merge and
   NN-saturation plots (current PNGs are working plots).
6. **Framing/positioning.** Confirm the intended venue and whether this is a standalone
   paper or a section of a larger NeuroBE paper. Adjust scope (e.g. how much systems
   detail in §6) accordingly.
7. **Author list / title.** Placeholder title; confirm with Nick.

## Source map (where to look in the repo)

- Merge code: `nce/inference/graphical_model.py` (`merge_join_tree` ~L557,
  `merge_by_degree` ~L634, `reduce_nn_merge` ~L713).
- Systems fixes: `nce/inference/factor_nn.py` (`nn_to_FastFactor`),
  `nce/sampling/sample_generator.py` (`_get_slices`),
  `nce/inference/bucket.py` (`_compute_message_exact_chunked`).
- NeuroBE repro: milestone M003 docs under `.gsd/milestones/M003/`.
- Lab notebook: `lab_notebook.txt`, entries 2026-05-31 → 2026-06-04.
