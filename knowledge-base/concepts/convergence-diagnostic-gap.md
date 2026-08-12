---
type: concept
title: The Convergence-Diagnostic Gap
status: budding
tags: [this-project, training, instrumentation, loss-functions]
created: 2026-08-12
updated: 2026-08-12
---

# The Convergence-Diagnostic Gap

For most of the project's life there was **no usable way to tell whether an NCE neural factor had
converged**. Two independent defects combined: the logged *training* loss underflows to exactly
`0.0`, and the *validation* loss — which the early-stopping rule computes every epoch — was thrown
away without ever being recorded. Every "did it converge?" question before 2026-08-11 was answered
by proxy (epoch count, early-stop firing) rather than by measurement.

## Defect 1: the training loss underflows to zero

Under `neurobe_weighted_mse` at float32 the reported training loss is **exactly `0.0`** for most of
training, in **every** arm including baseline — 382 of 425 epochs on one cluster (doc 16), 82–99%
of epochs across the runs doc 19 examined.

Gradients are unaffected: autograd differentiates the per-element expression, not the underflowed
sum. It is the *logged scalar* that is destroyed, past roughly epoch 50–100. Doc 16 is explicit
that this is not a property of the residual experiment: "it applies to every NeuroBE experiment…
Any past or future reading of those curves as a convergence signal is reading quantisation noise."

The underflow itself has **not** been fixed anywhere. It was worked around.

## Defect 2: `Trainer.val_losses` was write-only

`Trainer.val_losses` is initialised in `__init__` and plumbed all the way out into `result.json` —
and **nothing ever appended to it** on the `use_neurobe_early_stopping` path. The patience rule
computed a validation loss every epoch, compared it, and discarded it (doc 18).

A precision detail worth carrying: the `use_nbe_early_stopping` path (a *different* rule) does
append. The bug is specific to the `neurobe_` branch, which is the one every NeuroBE study config
uses — so the array was empty exactly where it mattered.

**Fix**: three lines in the patience block appending `(global_epoch_nb, val_loss)` plus a
`log_val_loss` call. No control flow change, no value recomputed. Landed as `2ae8cc1` on
`feat/wmb-residual`; re-applied on `fix/determinism` (doc 21). Doc 19 could only argue its
inertness statically, because at the time no A/A control could be bit-exact; doc 21 turned that
into a measurement — with ordering fixed, runs with and without the block give an identical
$\log Z$ and `len(val_losses)` of 0 vs 500. See [[bit-exact-reproducibility]].

## What became visible the moment the gap closed

The instrumentation immediately overturned a conclusion. Doc 16 had reported the WMB-residual
arm's epoch appetite as an unexplained cost; doc 18 read the validation trajectory and found the
arm was **genuinely still improving**: at epoch 500 its validation loss sat a median **2.60×**
above the best it eventually reached (range 1.47–4.93), and epochs 500→2000 cut it a further 61%.
Ground truth agreed — per-cluster $|$local error$|$ improved 22/30, median 0.0259 → 0.0138, with no
divergence between validation loss and local error. Doc 16's numbers were therefore a **lower
bound**, and 19/30 clusters were *still* improving at the 2000-epoch cap.

That is the practical value of the diagnostic: it distinguishes "this arm is expensive" from "this
arm has not finished", which the epoch count alone cannot do. See [[wmb-residual-learning]].

## Why it belongs in the knowledge base

Any comparison that treats `epochs_trained` as a convergence measure inherits both defects, and
one is compounded by [[num-samples-freeze]] — doc 04's convergence-speed study found grids where
a *larger* separator predicts *fewer* epochs, plausibly because every bucket trained on the same
frozen sample count and the net simply memorised faster. Before quoting an epoch number from
before 2026-08-11, check what it was actually measuring.

Doc 18's own recommendation: the validation-loss instrumentation should land on its own merits,
independently of whether any of the experiments that motivated it ship.

## Related

- [[loss-functions]] · [[neural-bucket-elimination]] · [[neural-network-factors]]
- [[wmb-residual-learning]] · [[num-samples-freeze]] · [[bit-exact-reproducibility]]
- [[codebase-map]] · [[2026-W33]]
