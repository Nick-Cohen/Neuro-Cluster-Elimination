---
type: glossary
title: Glossary
status: growing
tags: [glossary, terminology, writing]
created: 2026-06-12
updated: 2026-06-12
---

# Glossary

Quick definitions with the canonical term and a pointer to the full note. For the
NCE-code-term ↔ standard-term mapping and citation gotchas, see [[terminology-map]].

| Term | Definition | More |
|---|---|---|
| **Partition function ($Z$)** | Sum of the unnormalized distribution over all configurations; the PR inference task. | [[partition-function]] |
| **Factor** | Non-negative function over a small subset (scope) of variables; model = product of factors. | [[discrete-graphical-model]], [[factor-operations]] |
| **Variable elimination (VE)** | Exact inference by summing out variables one at a time in an order. | [[variable-elimination]] |
| **Bucket elimination (BE)** | VE organized into per-variable buckets; Dechter's unifying framework. | [[bucket-elimination]] |
| **Message** | Factor produced by eliminating a bucket's variable(s); passed downstream. | [[bucket-elimination]] |
| **Induced width / treewidth** | Max bucket width along an order; governs exact cost (exponential in it). | [[induced-width]] |
| **Elimination order** | Sequence of variable elimination; weighted min-fill heuristic in NCE. | [[elimination-ordering]] |
| **Tree decomposition** | Tree of variable clusters with running intersection; width = max cluster − 1. | [[tree-decomposition]] |
| **Junction / join / clique tree** | Tree of clusters satisfying running intersection; basis of exact propagation. | [[junction-tree]], [[join-tree]], [[clique-tree]] |
| **Running intersection property** | Each variable's clusters form a connected subtree. | [[running-intersection-property]] |
| **Separator** | Intersection of two adjacent clusters (= a message's scope). | [[junction-tree]] |
| **Mini-bucket elimination (MBE)** | Approx BE: split wide buckets into width-≤$i$ mini-buckets. | [[mini-bucket-elimination]] |
| **i-bound (`iB`)** | Max mini-bucket width; accuracy/cost knob. | [[iB-parameter]] |
| **Weighted mini-bucket (WMB)** | MBE with Hölder weights → tighter $Z$ bound. | [[weighted-mini-bucket]] |
| **IJGP** | Iterative join-graph propagation; bounded + iterative inference. | [[iterative-join-graph-propagation]] |
| **AND/OR search** | Search paradigm exploiting decomposition via a pseudo-tree. | [[and-or-search]] |
| **Super-bucket / super-cluster** | Bucket eliminating several variables at once (merged cluster). *Non-canonical term — prefer "merged cluster".* | [[super-bucket]] |
| **Deep Bucket Elimination (DBE)** | First to replace wide BE messages with trained NNs. | [[deep-bucket-elimination]] |
| **NeuroBE** | Personalized neural message approximation (UAI 2022). | [[neural-bucket-elimination]] |
| **Neural network factor** | A trained NN used as a bucket's outgoing message. | [[neural-network-factors]] |
| **Error accumulation** | Final $Z$ error grows with the *number* of learned messages. | [[error-accumulation]] |
| **Bucket merging** | NCE: merge adjacent NN-eligible buckets to cut that count. | [[bucket-merging]] |
| **Merge bound ($D$, `max_merge_bound`)** | Max vars eliminated per merged cluster; sweet spot 8–16. | [[merge-bound]] |
| **`ecl`** | *Project term:* exact-computation table-size limit; can override `iB`. | [[terminology-map]] |
| **`fdb`** | *Project term:* "forward diff barrier" — detached normalizer in a loss. | [[loss-functions]] |

## Related

- [[terminology-map]] · [[MOC-home]] · [`../references.bib`](../references.bib)
