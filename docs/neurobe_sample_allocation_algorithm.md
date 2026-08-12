# NeuroBE Sample Allocation Algorithm

## Overview

NeuroBE determines **per-bucket** how many uniform samples to draw for training
the neural network that approximates each bucket's output message. The sample
count scales with the bucket's width (number of input variables to the NN),
based on a pseudo-dimension estimate of the neural network's complexity.

This document describes the algorithm as implemented in the code, notes where
the paper's description differs, and provides sufficient detail for
reimplementation in the NCE codebase.

---

## Algorithm: Determine Sample Count for a Bucket

```
INPUT:
  w_in    — number of input variables to the NN (= message scope size,
             i.e., bucket width minus the eliminated variable)
  L       — number of hidden layers (config param, default = 2)
  epsilon — precision parameter (config param, default = 0.25)

CONSTANTS:
  delta = 0.001     (confidence parameter, hardcoded)
  train_split = 0.8
  max_samples = 1,000,000
  n_test = 50,000   (fixed test set size)

STEPS:

1. Compute the number of weight-matrix layers:
       l = L + 1
   (For L=2 hidden layers, l=3: input→hidden1, hidden1→hidden2, hidden2→output)

2. Approximate the total parameter count of the NN (assuming h = w_in, i.e., b=1):
       temp = (l - 1) * w_in^2  +  l * w_in  +  4
   Breakdown with l=3:  temp = 2*w_in^2 + 3*w_in + 4
   This approximates: 2 hidden-to-hidden/input weight matrices of size w_in^2,
   plus bias terms and output layer weights.

3. Compute the pseudo-dimension estimate (lower bound from Bartlett et al. 2019):
       pd = temp * ln(temp / l)
   This follows the form: Pdim >= W * log(W/L) where W = total parameters.

4. Compute total sample count using PAC-learning-style bound:
       nSamples = floor( (pd + ln(1/delta)) / epsilon )

5. Cap at maximum:
       if nSamples > 1,000,000:
           nSamples = 1,000,000

6. Split into train/validation/test:
       n_train = floor(0.8 * nSamples)
       n_val   = floor(0.2 * nSamples)
       n_test  = 50,000                    (fixed, added on top)
       total   = n_train + n_val + n_test
```

---

## Sampling Procedure (per bucket)

For each of the `total` samples:

```
1. For each input variable x_i in the message scope:
       sample val_i ~ Uniform(0, domain_size(x_i) - 1)    (integer)

2. Compute the bucket output value by marginalizing the eliminated variable:
       value = SUM over x_elim in domain(X_elim):
                   PRODUCT over f in bucket_functions:
                       f(val_1, ..., val_w_in, x_elim)

3. Store (input_assignment, value) as a training sample.

4. Route to train/val/test set based on sample index:
       samples 0..n_train-1           → training set
       samples n_train..n_train+n_val-1  → validation set
       samples n_train+n_val..total-1    → test set
```

Input normalization: values are mapped to [-1, 1] via `2*val/domain_size - 1`.
Output normalization: log-transform for large values, then normalize to [0, 1].

---

## Worked Example

For a bucket with `w_in = 20` input variables, `L = 2` hidden layers, `epsilon = 0.25`:

```
l = 3
temp = 2*(20^2) + 3*20 + 4 = 800 + 60 + 4 = 864
pd = 864 * ln(864/3) = 864 * ln(288) = 864 * 5.663 = 4893
nSamples = floor((4893 + ln(1000)) / 0.25) = floor((4893 + 6.908) / 0.25) = floor(19600) = 19600
n_train = 15680
n_val   = 3920
n_test  = 50000
total   = 69600
```

For `w_in = 50`:
```
temp = 2*2500 + 150 + 4 = 5154
pd = 5154 * ln(5154/3) = 5154 * 7.449 = 38387
nSamples = floor((38387 + 6.908) / 0.25) = 153576
n_train = 122861, n_val = 30715, n_test = 50000
```

---

## Paper vs Code: Key Discrepancies

### 1. Pseudo-dimension Formula

**Paper (Eq. 2):**
```
rho_c(w_c) proportional_to (L * b * w_c)^2 * log(b * w_c)
```
This is a simplified proportionality involving the hidden-unit multiplier `b`.

**Code (MiniBucket-NN.cpp:73-77):**
```cpp
int l = global_config.n_layers + 1;
float temp = ((l-1)*pow(w_in,2) + l*w_in + 4);
float pd = temp * log(temp/l);
```
This directly estimates the total parameter count W and applies `Pdim ~ W * log(W/L)`
from Bartlett et al. 2019. Note: `b` (var_dim) is NOT used in the sample count formula
— the pseudo-dimension is always computed as if `b = 1`, even when hidden units are
set to `b * w` for the actual network architecture.

### 2. Sample Size Formula

**Paper (Eq. 3):**
```
N = eta * (L * b * w)^2 * log(b * w)
```
Where eta is a tuned constant.

**Code (MiniBucket-NN.cpp:81):**
```
nSamples = (pd + log(1/delta)) / epsilon
```
This is a PAC-learning bound: `N = (Pdim + log(1/delta)) / epsilon`.
The paper's `eta` is effectively `1/epsilon` in the code, but the formulas
for the pseudo-dimension itself differ as noted above.

### 3. Validation Set Size

**Paper (page 7):** "validation set of size N(w_c)/9" (≈ 11.1%)

**Code:** `n_val = 0.2 * nSamples` (20%)

### 4. Hidden Unit Multiplier (b)

**Paper:** `h = b * w` where `b >= 1` is tuned per benchmark (b = 1, 3, or 5).

**Code:** `h_dim = _nArgs * var_dim` where `var_dim` defaults to 1.
The hidden units DO scale with `var_dim`, but the sample count formula ignores it.

---

## Implementation Notes for NCE Codebase

To implement this in the NCE `SampleGenerator`:

1. **Inputs needed per bucket:**
   - `w_in`: number of variables in the message scope (bucket width - number of
     eliminated variables). In NCE, this is `len(bucket.scope) - len(bucket.elim_vars)`.
   - `n_layers`: number of hidden layers in the NN config (from `nn_config`).
   - `epsilon`: new config parameter to add (controls sample budget; typical range 0.1-0.5).

2. **Sample count function:**
   ```python
   import math

   def neurobe_sample_count(w_in, n_layers=2, epsilon=0.25, delta=0.001,
                            max_samples=1_000_000, train_split=0.8, n_test=50_000):
       l = n_layers + 1
       temp = (l - 1) * w_in**2 + l * w_in + 4
       pd = temp * math.log(temp / l)
       n_samples = int((pd + math.log(1.0 / delta)) / epsilon)
       n_samples = min(n_samples, max_samples)

       n_train = int(train_split * n_samples)
       n_val = n_samples - n_train
       # n_test is fixed at 50,000

       return n_train, n_val, n_test
   ```

3. **Integration point:** Call this function in `SampleGenerator.sample_assignments()`
   (or wherever the number of samples is currently determined) to set the sample count
   dynamically per bucket based on its width.

4. **Config addition:** Add `epsilon` to the NN config dictionary. Consider also
   exposing `delta`, `max_samples`, and `train_split` as config options.

---

## Source Files

- **Sample count calculation:** `/home/cohenn1/SDBE/NeuroBE/BE-sampling-project/ARP/BE/MiniBucket-NN.cpp` (lines 66-94)
- **Hidden unit setup:** `/home/cohenn1/SDBE/NeuroBE/BE-sampling-project/ARP/Problem/Function-NN.hxx` (line 165)
- **NN architecture:** `/home/cohenn1/SDBE/NeuroBE/BE-sampling-project/ARP/Problem/Net.h`
- **Config defaults:** `/home/cohenn1/SDBE/NeuroBE/BE-sampling-project/ARP/Problem/Config.h`
- **Run scripts:** `/home/cohenn1/SDBE/NeuroBE/BE-sampling-project/BESampling/_build/grid40_f10.sh`
- **Paper:** NeuroBE: Escalating Neural Network Approximations of Bucket Elimination (Agarwal, Kask, Ihler & Dechter, UAI 2022; PMLR 180, pp. 11–21)
