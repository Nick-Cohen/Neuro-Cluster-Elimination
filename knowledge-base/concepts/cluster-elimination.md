---
type: concept
title: Cluster Elimination
created: 2026-03-01
tags: [inference, graphical-models, approximate]
---

# Cluster Elimination

Cluster elimination is the process of eliminating **more than one variable at a time** from a graphical model. Instead of a single elimination variable per bucket, a cluster (or "super bucket") has a **set** of elimination variables.

## Concept

In standard variable elimination, each bucket eliminates exactly one variable. Cluster elimination groups multiple variables together into a cluster and eliminates them simultaneously. This can be more efficient or produce better approximations in certain graph structures.

### Super Buckets

A super bucket is a bucket whose elimination target is a *set* of variables rather than a single variable. The scope of a super bucket is the union of scopes of all factors assigned to it, and all variables in the elimination set are marginalized out together.

### Merge Criteria

Clusters are formed according to merge criteria that determine which variables should be eliminated together. Common criteria include:
- **Shared scope**: Variables that appear together in many factors
- **Graph structure**: Variables that form cliques or near-cliques in the interaction graph
- **Complexity bounds**: Merge variables only if the resulting cluster stays within computational bounds

## Relationship to Mini-Bucket

Mini-bucket and cluster elimination are related but different:
- **Mini-bucket**: Splits a single bucket's factors into partitions, eliminates the *same* variable from each partition independently.
- **Cluster elimination**: Groups *multiple variables* into one elimination step.

They can be combined: a cluster could be too large for exact elimination, requiring mini-bucket approximation within the cluster.

## In NCE

The NCE codebase does not currently have a dedicated cluster elimination implementation. However, the concept is relevant for future development:
- `FastGM.eliminate_variables()` processes one variable at a time.
- The `wtminfill_order` heuristic could be extended to identify beneficial clusters.
- Cluster elimination could improve the quality of backward messages by reducing the number of elimination steps.

## Research Direction

From the project's prompt.txt, cluster elimination (along with its merge criteria) is a topic to be researched and added to the knowledge graph. Key questions:
- What merge criteria work best for the NCE use case?
- Can cluster elimination reduce the approximation error of backward messages?
- How does cluster elimination interact with WMB partitioning?

## Related

- [[variable-elimination]]
- [[weighted-mini-bucket]]
