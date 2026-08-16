# Doc 63 pre-registration (written before any sweep result was inspected)

Timestamp: see git commit date. No 63-sweep result had been analysed at this point;
only the g10w calibration wall times (base 274s / residual 300s / input 200s /
parts 226s) had been read, and no accuracy number from any cell.

## Hypothesis (Nick's, from doc 43's mechanism)
Residual learning forces the net to trust the WMB base. Doc 43 measured that the
base RANKS well on grids (Spearman 0.91-0.92) and badly on RBMs (0.51-0.52), which
is why the residual wins on grids (+0.515 dex) and loses on RBMs (-0.206, -0.230).
Feeding the base as an INPUT lets the net learn how much to trust it.

**Prediction under test:** `input` >= `residual` on grids, and `input` > `residual`
strictly on RBMs.

## Primary endpoint
Paired per-cluster |signed_local_error|, pairing unit (problem, seed, bucket).
gain(A vs B) = median[ log10|err_B| - log10|err_A| ], positive = A more accurate.
Two-sided exact sign test + Wilcoxon signed-rank. Reported PER FAMILY, never pooled
(doc 31 s4: pooling previously hid an entire negative result).

## Cells
grids: grid10x10.f10.wrap, grid20x20.f10
rbm:   dbn/rbm_20, dbn/rbm_21
All at iB=10, ecl=1025, subsumption merge D=10, num_epochs cap 2000 with NeuroBE
early stopping -- doc 31's protocol, i.e. the protocol that produced the residual
numbers being compared against.

## Stopping rule (pre-registered)
Seeds 42 and 43 are run in full. **Seed 44 is run only if the sign of the
`input` vs `residual` comparison disagrees between seeds 42 and 43 in any cell.**
If the two seeds agree in every cell, seed 44 is dropped and the GPU goes to
experiment 2. This is stated before any accuracy number has been read.

## What counts as a negative result
If `input` does not beat `residual` on the grids, that is reported plainly as a
failure of the prediction. A clean negative closes the direction off.
