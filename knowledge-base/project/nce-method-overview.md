---
type: project
title: NCE — Method Overview
status: growing
tags: [this-project, method, overview]
created: 2026-06-12
updated: 2026-06-12
---

# NCE — Method Overview

**NCE** estimates the [[partition-function]] $Z$ of a discrete
[[discrete-graphical-model|graphical model]] by running [[bucket-elimination]] and
replacing the messages of high-width buckets with **trained neural networks** — a faithful
reproduction of [[neural-bucket-elimination]] (NeuroBE, [[@agarwal2022neurobe]]) plus the
project's own contribution, **[[bucket-merging]]**. Lineage:
[[deep-bucket-elimination]] (2021) → NeuroBE (2022) → NCE.

## The pipeline

1. Load the model (UAI format / `pyGMs`) → compute an [[elimination-ordering|elimination order]]
   (weighted min-fill) → organize factors into buckets ([[bucket-structure]]).
2. **(NCE contribution)** Optionally **merge** adjacent NN-eligible buckets into
   super-clusters, capped by the [[merge-bound]] — see [[bucket-merging]].
3. Walk the order. For each bucket, decide **exact vs learned**:
   - table fits under the limits (`iB` width, `ecl` table size) → compute exactly;
   - otherwise → sample the message scope, label with exact log-values, normalize, train a
     small MLP, and use it as the message ([[neural-network-factors]],
     [[sample-generation]], [[loss-functions]]).
4. Messages propagate; the final scalar is $\log_{10} Z$.

## Why it works / where it fails

- **Failure mode:** [[error-accumulation]] — final $Z$ error scales with the *number* of
  learned messages, not their individual loss.
- **Fix:** [[bucket-merging]] cuts that number (and sometimes collapses clusters back to
  exact), buying ~6–7× (up to ~100×) accuracy at a [[merge-bound]] sweet spot of 8–16.

## Map of the system

- Code: [[codebase-map]]. Terminology vs the literature: [[terminology-map]].
- Architecture: [[three-layer-architecture]] · [[graphical-model-structure]].
- The paper: [`../../writeup/paper.md`](../../writeup/paper.md) ("Bucket Merging Improves
  Neural Bucket Elimination"); provenance in `../../writeup/NOTES.md`.

## Sources

- [[@agarwal2022neurobe]] · [[@razeghi2021deep]] · [[@liu2011holder]] · [[@dechter1999bucket]].

## Related

- [[bucket-merging]] · [[neural-bucket-elimination]] · [[error-accumulation]] ·
  [[weighted-mini-bucket]] · [[open-questions]]
