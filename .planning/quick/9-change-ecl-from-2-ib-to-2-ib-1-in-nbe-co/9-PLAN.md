---
phase: quick
plan: 9
type: execute
wave: 1
depends_on: []
files_modified:
  - nce/benchmark_problems/nbe_sanity_check.py
autonomous: true
---

<objective>
Change ecl from 2**iB to 2**(iB-1) in NBE configs. NeuroBE's implementation uses iB-1 internally, so ecl should be one power lower. This also fixes rbm_20 (width=20, iB=20) which currently has 0 NN-eligible buckets.
</objective>

<tasks>
<task type="auto">
  <name>Task 1: Change ecl formula in nbe_sanity_check.py</name>
  <files>nce/benchmark_problems/nbe_sanity_check.py</files>
  <action>Change `'ecl': 2**_IB_MAP[key]` to `'ecl': 2**(_IB_MAP[key]-1)` and verify all 5 models now have NN-eligible buckets.</action>
  <verify>Import nbe_sanity_check and confirm ecl values are halved. Check rbm_20 has large_buckets > 0.</verify>
  <done>ecl uses 2**(iB-1) for all models. rbm_20 now triggers NN training.</done>
</task>
</tasks>
