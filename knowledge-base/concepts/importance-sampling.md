---
type: concept
title: Importance Sampling (for message learning)
status: seedling
tags: [sampling, neural, this-project]
created: 2026-06-12
updated: 2026-06-12
---

# Importance Sampling (for message learning)

**Importance sampling** draws training configurations for a bucket's message from a
proposal distribution (rather than uniformly), so that high-probability / high-magnitude
regions of the message are better represented in the training set. In
[[neural-bucket-elimination]], the faithful mode samples the message's output scope this
way and uses **importance weights** in the (weighted-MSE) loss.

## Key points

- Goal: spend the sample budget where it reduces message-approximation error most.
- Contrasts with NCE's simpler `'uniform'` and `'all'` schemes — see [[sample-generation]].
- The proposal choice (uniform vs WMB-based) is an open comparison in the project; see
  [[open-questions]] and the importance-sampling plan in project memory.

## In NCE

- Sampling lives in `SampleGenerator` ([`nce/sampling/sample_generator.py`](../../nce/sampling/sample_generator.py));
  see [[sample-generation]], [[mini-bucket-sampling]].

## Sources

- [[@agarwal2022neurobe]] (NeuroBE's sampling + weighted loss).

## Related

- [[sample-generation]] · [[neural-bucket-elimination]] · [[weighted-mini-bucket]] · [[open-questions]]
