---
type: moc
title: MOC — Home
status: evergreen
tags: [moc, index]
created: 2026-06-12
updated: 2026-06-12
---

# MOC — Home

Entry point for the NCE knowledge base. See [`../README.md`](../README.md) for conventions.
NCE = neural-network approximation of inference in discrete graphical models. Start with
[[nce-method-overview]].

## Maps of Content

- [[MOC-exact-inference]] — graphical models, partition function, variable/bucket
  elimination, junction trees, AND/OR search.
- [[MOC-approximate-inference]] — mini-bucket, weighted mini-bucket, join-graph propagation,
  super-buckets.
- [[MOC-neural-inference]] — Deep Bucket Elimination, NeuroBE, learned messages, error
  accumulation.
- [[MOC-this-project]] — NCE method, bucket merging (our contribution), code & terminology
  maps, open questions, progress.

## Fast lookups

- **Citations:** [`../references.bib`](../references.bib) · literature notes in
  [`../literature/`](../literature/) (`@key.md`, one per source).
- **Terminology when writing:** [[terminology-map]] · [[glossary]].
- **Where is X in the code?** [[codebase-map]].
- **What changed recently / advisor updates:** [`../progress/`](../progress/),
  `lab_notebook.txt`.
- **Open problems:** [[open-questions]].

## The story in one line

Bucket elimination is exact but memory-bound; NeuroBE learns the wide messages; **NCE merges
adjacent learned buckets to cut [[error-accumulation]]**, gaining 6–7× (up to ~100×) accuracy
at a [[merge-bound]] sweet spot historically quoted as 8–16 — but see
[[time-optimal-merge-bound]]: the *time*-optimal bound is a property of the code version and
moved in 2026-08.

## Maintenance

- Migrated code-concept notes live in [`../concepts/`](../concepts/) and
  [`../patterns/`](../patterns/) (formerly `.knowledge/`).
- When adding a source: add to `references.bib`, create `literature/@key.md`, link it from
  the relevant concept note's `## Sources`.
