# Open Questions: WMSE vs UKL Benchmark Experiment

Questions discovered during experiment design. Each includes impact analysis, options, and a recommendation.

---

## Q1: batch_size='all' support in worker.py

**Context:** The user wants `batch_size = message_size` (one batch containing the entire message). `train.py` supports `batch_size='all'` as a string (line 259), which internally sets `batch_size = int(self.message_size)`. However, `worker.py`'s `build_nn_config()` reads `batch_size` as an integer from YAML config: `config.get('batch_size', 100000)`. Passing `'all'` through YAML would require worker.py modification.

**Impact:** If batch_size < message_size, data is split into multiple mini-batches. With `sampling_scheme='all'`, the full message is enumerated, but it would be split across batches. This changes training dynamics (per-batch gradient updates vs. full-batch gradient).

**Options:**

| Option | Approach | Pros | Cons |
|--------|----------|------|------|
| A | Set batch_size=10,000,000 (very large int) | No code changes; guaranteed > any message_size in small_problems | Not semantically explicit; technically wasteful allocation check |
| B | Modify `build_nn_config()` to pass through `'all'` string | Correct semantics; matches train.py API | Requires code change to worker.py |
| C | Run via Python script (bypass YAML/worker.py) | Full control; can set any config value | Doesn't use the existing experiment infrastructure |

**Recommendation:** **Option A** -- set `batch_size=10000000`. The largest `auto_ecl` in small_problems is 19,487,170 (deer_rescaled_0034.K10.F2.model), but actual `message_size` for NN-eligible buckets will be much smaller than `ecl` (message_size is the number of entries in a bucket's message, which depends on scope size, not ecl). In `train.py` (line 263): `num_batches_per_set = ceil(set_size / batch_size)` will be 1 whenever `batch_size >= message_size`, achieving the same effect as `batch_size='all'`.

---

## Q2: bw_ecl=ecl (Config 4) per-problem variation

**Context:** Configuration 4 sets `bw_ecl` equal to each problem's `auto_ecl`. The `auto_ecl` varies per problem (range: 16,383 to 19,487,170). The YAML config format specifies a single `bw_ecl` list that applies to all experiments generated from that config.

**Impact:** Cannot express Config 4 in a single YAML file because each of the 24 problems needs a different `bw_ecl` value.

**Options:**

| Option | Approach | Pros | Cons |
|--------|----------|------|------|
| A | Generate 24 individual YAML configs (one per problem) | Uses existing infrastructure | 24 files to manage; manual ecl-to-bw_ecl mapping |
| B | Add `'ecl'` as special bw_ecl keyword in worker.py | Clean API; single config | Requires code change |
| C | Run Config 4 via Python script | No code changes; most flexible | Different execution path than Configs 1-3 |
| D | Run ALL configs via Python script | Uniform execution; full control | Doesn't test YAML infrastructure |

**Recommendation:** **Option D** -- run all configurations via a Python script. This provides uniform execution across all 5 configs and avoids the per-problem YAML problem entirely. The YAML/worker.py system is designed for GPU-distributed execution; for a benchmark comparison where we want precise control, a Python script is more appropriate.

---

## Q3: Architecture choice

**Context:** The user specified the experiment but did not specify architecture. The `small_problems` default config uses `hidden_sizes=[3, 3]`. Other architectures available: `[]` (linear), `[2]` (simple_nn).

**Impact:** Each additional architecture multiplies total experiments by 1 (adding 24 more experiments per architecture per config). With 3 architectures: 5 x 24 x 3 = 360 experiments total.

**Recommendation:** Start with `[3, 3]` only (the small_problems default) to keep the experiment focused on the loss function / backward info comparison. Add architectures in a follow-up experiment if results warrant it.

---

## Q4: Number of runs (statistical significance)

**Context:** The user did not specify `num_runs`. The default is 1. Multiple runs with different seeds would provide statistical confidence intervals.

**Impact:** Each additional run multiplies wall-clock time by approximately 1x. With 120 experiments at ~5000 epochs each, a single run may take many hours.

**Recommendation:** 1 run first (seed=42). If results look noisy or close between configurations, add 2 more runs (seeds 1042, 2042) for statistical significance.

---

## Q5: ECL values (auto_ecl vs fixed ecl=1024)

**Context:** `small_problems` uses per-problem `auto_ecl` values (range 16,383 to 19,487,170). The existing `probs12_4` YAML configs use `ecl: 1024` (2^10). These are very different.

**Impact:** Higher ecl means fewer NN-trained buckets (more computation is exact). With `auto_ecl`, most computation is exact and only a few large-scope buckets are NN-trained. With `ecl=1024`, many more buckets are NN-trained.

**Recommendation:** Use `auto_ecl` from small_problems. These values were computed specifically for each problem's variable scope widths. The `ecl=1024` in probs12_4 examples was a conservative default, not problem-tuned.

---

## Q6: iB20 problems

**Context:** The benchmarks_12_4_2025 dataset has iB10, iB15, and iB20 problem sets. The `small_problems` module only includes iB10 and iB15 (24 problems total). iB20 problems are much larger and some lack known ground truth partition functions.

**Impact:** Including iB20 would add more problems but with less reliable ground truth for evaluation.

**Recommendation:** Use only `small_problems` (24 problems, iB10 + iB15). The `small_problems` module was specifically designed to exclude iB20 for this reason.

---

## Q7: WMSE loss and backward info interaction

**Context:** The `weighted_logspace_mse` function signature accepts `bw_hat` but ignores it completely:
```python
def weighted_logspace_mse(outputs, targets, bw_hat=None):
    # bw_hat is never used in the function body
    ...
```
When `bw_ecl=0`, no backward factors are populated, so `bw_hat=None`. This is correct behavior.

**Question:** Should we also test WMSE WITH backward info (bw_ecl > 0) to verify it truly ignores bw_hat at runtime?

**Recommendation:** Not needed. Code inspection confirms `bw_hat` is unused in the loss computation. Adding this test would add experiments without providing insight into the loss function comparison.

---

## Q8: 5000 epochs vs defaults

**Context:** The user specified 5000 epochs. Previous experiments used:
- `small_problems` default: 10,000 epochs
- `worker.py` default: 30,000 epochs
- Existing benchmark YAMLs: 10,000 epochs

**Impact:** Shorter training may not converge on harder problems (especially larger problems with more complex message structure). 5000 epochs is half the small_problems default.

**Recommendation:** Honor the user's specification of 5000 epochs. If convergence is poor on specific problems, those can be re-run with more epochs. The `skip_early_stopping=True` setting ensures all 5000 epochs run regardless.

---

## Q9: Output format

**Context:** The existing `experiment_runner.py` saves results as JSON (per-experiment `results.json` with aggregated `summary.json`). Should we also produce CSV or pickle files for analysis?

**Recommendation:** Use JSON format for raw results. Post-processing into pandas DataFrames / CSV can be done after the experiment completes. Keep the experiment script focused on running inference and saving raw results.

---

## Q10: dope_factors setting

**Context:** The `small_problems` configs have `dope_factors=False`. NeuroBE's original configuration uses `dope_factors=True`. Doping replaces `-inf` values with finite values in the factor table, which affects training stability for losses that operate in log-space.

**Impact:**
- `dope_factors=True`: All factor entries are finite, no `-inf` values in training data. Required by some loss functions for numerical stability.
- `dope_factors=False`: Factor entries may contain `-inf` (log of 0). Both WMSE and UKL handle this through their max-value normalization.

**Question:** Should we use `dope_factors=True` (NeuroBE convention) or `False` (small_problems convention)?

**Recommendation:** Use `dope_factors=False` for consistency with the `small_problems` defaults. Both losses (WMSE and UKL) see the same training data, so the comparison is fair. If WMSE shows numerical issues with `-inf` entries, this itself is a useful finding about the loss function's robustness.
