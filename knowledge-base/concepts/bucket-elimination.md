---
type: concept
title: Bucket Elimination
status: growing
tags: [inference, exact, graphical-models, foundations]
created: 2026-06-12
updated: 2026-06-12
---

# Bucket Elimination

**Bucket elimination (BE)** is Dechter's unifying organization of [[variable-elimination]]:
fix an [[elimination-ordering|elimination order]], place each factor in the *bucket* of
its earliest-eliminated variable, then process buckets in order — multiply the bucket's
factors, sum out (marginalize) the bucket's variable, and place the resulting **message**
in the bucket of the next-earliest variable in its scope. The same template solves many
tasks (belief updating, MPE, MAP, MEU, constraint reasoning) by swapping the
combine/marginalize operators ([[@dechter1999bucket]]).

## Key points

- BE and [[variable-elimination]] are the **same algorithm** in different vocabularies
  (VE: Zhang & Poole, [[@zhang1996exploiting]]). "Bucket" names the per-variable container.
- Cost of a bucket is exponential in its **width** (variables in the combined function);
  the max width over the run is the [[induced-width]], which bounds total cost.
- The bucket structure forms a tree (the *bucket tree*), a special [[tree-decomposition]].
- Exact but memory-bound: a single wide bucket can need a table exponential in its width —
  the bottleneck [[deep-bucket-elimination]] and [[neural-bucket-elimination]] attack.

## In NCE

- Driven by `FastGM.eliminate_variables()` /  bucket dispatch in
  [`nce/inference/graphical_model.py`](../../nce/inference/graphical_model.py); buckets are
  `FastBucket` ([`nce/inference/bucket.py`](../../nce/inference/bucket.py)) — see
  [[bucket-structure]].
- A bucket is computed **exactly** when its message table fits under the limits
  (`iB` width and `ecl` table size); otherwise it is approximated. See [[iB-parameter]],
  [[terminology-map]] (for `ecl`).

## Sources

- [[@dechter1999bucket]] (canonical; the 1996 UAI and 1998 chapter variants are titled
  "...for probabilistic inference"). Textbook: [[@koller2009pgm]].

## Related

- [[variable-elimination]] · [[induced-width]] · [[mini-bucket-elimination]] ·
  [[weighted-mini-bucket]] · [[super-bucket]] · [[bucket-merging]]
