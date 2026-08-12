---
type: project
title: Open Questions
status: growing
tags: [this-project, research, questions]
created: 2026-06-12
updated: 2026-08-12
---

# Open Questions

Live unknowns and decisions in flight. Each links to where it's tracked. Keep this short
and current — resolved items move to a literature/concept note or the lab notebook.

## Paper-blocking (from `writeup/NOTES.md`)

- **Canonical run set.** Curated lab-notebook numbers vs the automated
  `reduce_nn_experiment/` JSONs disagree on some cells (different seeds/sampling, non-monotone
  in $D$). Pick **one** run per (problem, $D$) cell before publishing; don't mix sources.
- **Reference $Z$ per problem.** Document ground-truth source (exact / converged solver /
  high-iB WMB). grid40x40 and pedigree51 historically had `nan` refs — see project memory
  `reference_grid40x40_refz`. Affects [[partition-function]] error reporting.
- **Sampling details for headline runs** — scheme + sample count + seeds/trials. See
  [[importance-sampling]], [[sample-generation]].
- **Figures** — regenerate publication-quality accuracy/time-vs-merge and NN-saturation plots.
- **Venue / framing** — standalone "bucket merging" paper vs a section of a larger NeuroBE
  paper. Affects how much systems detail (§6) to keep.

## Opened 2026-W33 (see [[2026-W33]])

- **Seven unmerged branches, and the working tree is still the only source of truth.** Doc 10
  found that no branch contained the merge passes, the tree-collect populator or
  `proposal_sampler.py`. Same blocker as 2026-06-12; now blocking merges, not just refactors.
- **`e_max` semantics.** Nick's rule is a state-space product; the code caps a variable count;
  the ~6–15 line fix was scoped twice and applied neither time. Docs 25 and 27 disagree on
  whether it matters. → [[terminology-map]]
- **Published numbers need a correction pass** (doc 27): the median optimal $e_{\max}$ sentence
  (published 10 IQR [6,12] reproduces as 8 IQR [6,10] and is internally impossible), γ's headline
  provenance (two runs five days apart; the shipped CSV is the *other* one), the README error
  geomean (1.55 → 1.8126), and two four-arm cost-table rows. → [[time-optimal-merge-bound]]
- **grid40x40's time-optimal $e_{\max}$ is only a lower bound (≥16).** Needs rnn20/rnn24, ~2–4 h.
- **`FastFactor` still has no `__hash__`/`__eq__`**, so the [[bit-exact-reproducibility]] bug's
  *class* is unfixed, with no lint or CI rule against it. Several audited-but-unexercised
  nondeterminism hazards remain (float `scatter_add_`, `nn.Embedding` backward, ungenerated
  `randperm`/`randint`, `manual_seed` calls inside `losses.py`).
- **Can the residual result be reproduced on a second problem family?** Everything so far is two
  grid cells. → [[wmb-residual-learning]], doc 31's sweep design.
- **Should `adam_eps` default change?** 1e-10 makes the NeuroBE baseline 0.75× cheaper at no
  measured accuracy cost. → [[adam-eps-and-loss-scale]]
- **Does the training loss have to underflow?** Nothing fixes it; the workaround is to read the
  validation trajectory. → [[convergence-diagnostic-gap]]

## Method questions

- **Proposal distribution for [[importance-sampling]]** — uniform vs WMB-based proposal;
  6-condition comparison planned (project memory `project_importance_sampling_plan`).
- **Why do dense RBMs gain least from [[bucket-merging]]?** (smallest NN-count reduction).
- **Does merging compose with non-MLP backends** (decision trees, [[memorizer-nn]])?

## Terminology / citation

- Confirm secondary page ranges flagged in literature notes (Zhang–Poole 1994;
  Liu–Ihler marginal-MAP UAI 2011; Marinescu–Dechter AOBB companion). See those
  `literature/@*.md` "Citation confidence" sections.

## Related

- [[nce-method-overview]] · [[bucket-merging]] · [[error-accumulation]] · [[terminology-map]]
