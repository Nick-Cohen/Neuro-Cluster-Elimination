---
type: concept
title: Partition Function (Z)
status: growing
tags: [inference, graphical-models, foundations]
created: 2026-06-12
updated: 2026-06-12
---

# Partition Function (Z)

The partition function $Z$ is the normalizing constant of a [[discrete-graphical-model]]:
the sum of the unnormalized distribution over **all** joint configurations,
$$ Z = \sum_{x} \prod_j f_j(x_{S_j}). $$
Computing $Z$ (the **PR**, probability-of-evidence, task in the UAI competition) is the
canonical #P-hard inference problem and the quantity NCE estimates.

## Key points

- Exactly computable by [[variable-elimination]] / [[bucket-elimination]] at cost
  exponential in [[induced-width]] — infeasible for wide models.
- Usually reported in log scale ($\log_{10} Z$ in NCE, natural log elsewhere); errors
  are measured as absolute error in $\log Z$ (nats or dex).
- Bounded above/below by [[weighted-mini-bucket]] (Hölder bound, [[@liu2011holder]]).
- Approximated by a single point estimate in [[neural-bucket-elimination]] and NCE.

## In NCE

- The final scalar message after eliminating every variable **is** $\log_{10} Z$.
- All factor arithmetic is in log base 10 ([[log-space-convention]]); marginalization
  is log-sum-exp.
- Reference $Z$ values per benchmark are an open issue — some hard instances
  (grid40x40, pedigree51) historically had `nan` references; see [[open-questions]].

## Sources

- [[@dechter1999bucket]] (exact via bucket elimination) · [[@liu2011holder]] (bounds) ·
  [[@koller2009pgm]].

## Related

- [[discrete-graphical-model]] · [[variable-elimination]] · [[error-accumulation]]
