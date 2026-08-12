---
type: literature
title: UAI Text File Format for Graphical Models
citekey: uaifileformat
authors: []
year: 2010
venue: PASCAL/UAI challenge spec (URL)
status: growing
tags: [literature, data]
created: 2026-06-12
updated: 2026-06-12
---

# UAI (Simple Text) File Format for Graphical Models

> **Citation.** "The UAI File Format for Graphical Models." Specification for the
> PASCAL/UAI inference challenges, 2010.
> **Link.** <https://www.cs.huji.ac.il/project/PASCAL/fileFormat.php>
> **BibTeX key.** `uaifileformat` (see [`../references.bib`](../references.bib))

## Contribution (in our words)
The plain-text format used to distribute benchmark graphical models: a preamble
(`BAYES`/`MARKOV`), variable counts and cardinalities, factor scopes, then factor tables.
A generalization of the Ergo format (Noetic Systems Inc.). No formal paper exists — cite the
URL. Related: the UAI inference competition (`uai2022competition`) and Ihler's UCI model
repository (`ihler_uai_models`).

## Why it matters to NCE
This is the format NCE loads (via `pyGMs`) for all grid / pedigree / RBM benchmark instances.
See [[discrete-graphical-model]].

## Related
- [[discrete-graphical-model]] · [[partition-function]]
