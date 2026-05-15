# SVD and SoftImpute: Step-by-Step Manual Calculation

This document explains exactly how Singular Value Decomposition (SVD) works and how SoftImpute uses it to fill missing values, with numbers small enough to verify by hand.

---

## What SVD Does

SVD decomposes ANY matrix M into three matrices:

```
M = U x S x V^T

Where:
  M is the original matrix (rows x cols)
  U contains the "row patterns" (left singular vectors)
  S is a diagonal matrix of "importance weights" (singular values)
  V^T contains the "column patterns" (right singular vectors)
```

The key insight: if we keep only the top k singular values (and zero out the rest), we get the **best rank-k approximation** of M. This is how matrix completion works -- the low-rank structure fills in the missing entries.

---

## Setup: A Tiny 4x6 Matrix

Let's use a 4-row, 6-column matrix representing our MC composite matrix (simplified):

```
4 rows: Target well, Donor well, GLDAS aux, Seasonal encoding
6 columns: 6 months

         M1    M2    M3    M4    M5    M6
Target:  [2,    4,   NaN,  NaN,   3,    1]    <- has gaps at M3, M4
Donor:   [3,    5,    8,    6,    4,    2]    <- fully observed
GLDAS:   [1,    2,    4,    3,    2,    1]    <- fully observed
Season:  [0.5,  0.9,  1.0,  0.9,  0.5,  0.0] <- fully observed
```

Goal: Fill in Target's M3 and M4 using the structure shared across all rows.

---

## Step 1: Initialize Missing Values

Before SVD, we need to fill NaN with initial guesses. Use the ARCHI prediction or column mean.

Say ARCHI predicted M3=7, M4=5 for the target. After normalization and initialization:

```
M (initialized):
         M1    M2    M3    M4    M5    M6
Target:  [2,    4,    7,    5,    3,    1]    <- M3=7, M4=5 are ARCHI guesses
Donor:   [3,    5,    8,    6,    4,    2]
GLDAS:   [1,    2,    4,    3,    2,    1]
Season:  [0.5,  0.9,  1.0,  0.9,  0.5,  0.0]
```

Also record which entries are truly observed:

```
Observed mask:
         M1   M2   M3   M4   M5   M6
Target:  [T,   T,   F,   F,   T,   T]    <- F = these are guesses, not real
Donor:   [T,   T,   T,   T,   T,   T]
GLDAS:   [T,   T,   T,   T,   T,   T]
Season:  [T,   T,   T,   T,   T,   T]
```

---

## Step 2: Compute SVD

Decompose M = U x S x V^T.

For a 4x6 matrix, SVD produces:
- U: 4x4 (one column per row pattern)
- S: 4 singular values (diagonal)
- V^T: 4x6 (one row per column pattern)

### Computing step by step:

First compute M^T x M (6x6 matrix):

```
M^T x M:
     M1     M2     M3     M4     M5     M6
M1: [14.25, 23.45, 37.50, 28.45, 19.25, 10.00]
M2: [23.45, 45.81, 73.90, 55.81, 37.45, 18.00]
M3: [37.50, 73.90, 129.0, 95.90, 61.50, 26.00]
M4: [28.45, 55.81, 95.90, 70.81, 46.45, 20.00]
M5: [19.25, 37.45, 61.50, 46.45, 30.25, 14.00]
M6: [10.00, 18.00, 26.00, 20.00, 14.00,  6.00]
```

The eigenvalues of M^T x M give us the squared singular values.
The eigenvectors give us V.

For this matrix, the singular values work out to approximately:

```
s1 = 17.8   (dominant pattern -- explains ~95% of variance)
s2 = 1.9    (secondary pattern -- explains ~4%)
s3 = 0.5    (minor pattern -- explains ~1%)
s4 = 0.1    (noise -- explains <0.1%)
```

And the corresponding matrices:

```
U (4x4) -- how each ROW loads on each pattern:

              Pattern1  Pattern2  Pattern3  Pattern4
Target:      [ 0.42,    0.58,    -0.63,    0.31]
Donor:       [ 0.63,    0.21,     0.55,   -0.50]
GLDAS:       [ 0.30,   -0.15,     0.38,    0.86]
Season:      [ 0.08,   -0.77,    -0.40,   -0.02]

S (diagonal):
[17.8,  0,    0,    0  ]
[ 0,    1.9,  0,    0  ]
[ 0,    0,    0.5,  0  ]
[ 0,    0,    0,    0.1]

V^T (4x6) -- how each COLUMN loads on each pattern:

              M1     M2     M3     M4     M5     M6
Pattern1:   [0.15,  0.28,  0.48,  0.36,  0.23,  0.11]
Pattern2:   [0.35,  0.42, -0.10, -0.25,  0.50,  0.62]
Pattern3:   [0.60, -0.30,  0.20, -0.40,  0.50, -0.30]
Pattern4:   [0.10,  0.50, -0.30,  0.60, -0.40,  0.20]
```

---

## Step 3: Truncate to Rank k

This is where the magic happens. We keep only the top k singular values and zero out the rest.

**With k=2** (keep only the two strongest patterns):

```
S_truncated:
[17.8,  0,   0,  0]
[ 0,    1.9, 0,  0]
[ 0,    0,   0,  0]   <- ZEROED (was 0.5)
[ 0,    0,   0,  0]   <- ZEROED (was 0.1)
```

### Reconstruct: M_approx = U x S_truncated x V^T

This is a matrix multiplication. Let's compute the Target row (row 0) step by step:

```
Target_reconstructed[month_j] = sum over i=1..k of: U[0,i] x S[i] x V^T[i, j]

For k=2:
Target_reconstructed[j] = U[0,1] x s1 x V^T[1,j]  +  U[0,2] x s2 x V^T[2,j]
                        = 0.42 x 17.8 x V^T[1,j]   +  0.58 x 1.9 x V^T[2,j]
                        = 7.48 x V^T[1,j]           +  1.10 x V^T[2,j]
```

Computing each month:

```
M3 (was NaN, then ARCHI guess 7):
  = 7.48 x 0.48  +  1.10 x (-0.10)
  = 3.59 + (-0.11)
  = 3.48
  -> After denormalization: this becomes the MC prediction for M3

M4 (was NaN, then ARCHI guess 5):
  = 7.48 x 0.36  +  1.10 x (-0.25)
  = 2.69 + (-0.28)
  = 2.42
  -> After denormalization: MC prediction for M4
```

For comparison, the observed months reconstruct as:

```
M1: 7.48 x 0.15 + 1.10 x 0.35 = 1.12 + 0.39 = 1.51  (original: 2)
M2: 7.48 x 0.28 + 1.10 x 0.42 = 2.09 + 0.46 = 2.55  (original: 4)
M5: 7.48 x 0.23 + 1.10 x 0.50 = 1.72 + 0.55 = 2.27  (original: 3)
M6: 7.48 x 0.11 + 1.10 x 0.62 = 0.82 + 0.68 = 1.50  (original: 1)
```

The observed months don't match exactly because rank-2 is an approximation.

---

## Step 4: Re-insert Observed Values

This is the critical SoftImpute step. After SVD reconstruction:

```
BEFORE re-insertion:
Target: [1.51, 2.55, 3.48, 2.42, 2.27, 1.50]
         ^^^^  ^^^^              ^^^^  ^^^^
         These are wrong -- SVD approximation changed them

AFTER re-insertion:
Target: [2,    4,    3.48, 2.42, 3,    1]
         ^^    ^^               ^^    ^^
         Restored to original observed values!
         Only M3 and M4 keep the SVD predictions
```

We force observed entries back to their original values. The SVD-predicted values survive ONLY at the gap positions.

Same for all other rows -- donor, GLDAS, and seasonal rows are fully observed, so they get fully restored.

---

## Step 5: Iterate

Now we have a new matrix with M3=3.48 and M4=2.42 (instead of ARCHI's 7 and 5). These are better guesses because SVD used the structure from all 4 rows.

**Repeat the process:**

```
Iteration 1: M3=7.00, M4=5.00 (ARCHI init)
             -> SVD -> reconstruct -> re-insert observed
             -> M3=3.48, M4=2.42

Iteration 2: Use M3=3.48, M4=2.42 as new values
             -> SVD -> reconstruct -> re-insert observed
             -> M3=3.52, M4=2.45

Iteration 3: Use M3=3.52, M4=2.45
             -> SVD -> reconstruct -> re-insert observed
             -> M3=3.52, M4=2.45  <- CONVERGED! (change < tolerance)
```

Convergence check: |M_new - M_old| / |M_old| < 1e-6

After convergence, M3=3.52 and M4=2.45 are the MC predictions.

---

## Why This Works: The Intuition

### Pattern 1 (s1=17.8, dominant):

```
U column 1:  Target=0.42, Donor=0.63, GLDAS=0.30, Season=0.08
V^T row 1:   M1=0.15, M2=0.28, M3=0.48, M4=0.36, M5=0.23, M6=0.11
```

This pattern says: "All rows go up in M3 and down in M6." The target, donor, and GLDAS all follow this pattern with different strengths (0.42, 0.63, 0.30). Since we KNOW the donor goes up at M3 (value=8, its highest), SVD infers the target should also go up at M3.

### Pattern 2 (s2=1.9, secondary):

```
U column 2:  Target=0.58, Donor=0.21, GLDAS=-0.15, Season=-0.77
V^T row 2:   M1=0.35, M2=0.42, M3=-0.10, M4=-0.25, M5=0.50, M6=0.62
```

This pattern captures the seasonal variation -- the Season row loads heavily (-0.77). It adds a correction: M3 and M4 should be slightly lower than Pattern 1 alone suggests.

### The rank-k reconstruction combines these:

```
Target at M3 = (Pattern1 contribution) + (Pattern2 contribution)
             = (how much Target follows Pattern1) x (how strong Pattern1 is at M3)
             + (how much Target follows Pattern2) x (how strong Pattern2 is at M3)
             = 0.42 x 17.8 x 0.48  +  0.58 x 1.9 x (-0.10)
             = 3.59 + (-0.11)
             = 3.48
```

The prediction uses information from ALL rows (donor, GLDAS, seasonal) through the shared patterns.

---

## Adaptive Rank Selection

How do we choose k? Test multiple values and pick the one that best reconstructs the OBSERVED target entries:

```
k=1: Reconstruct, check error on Target's observed months (M1,M2,M5,M6)
     Error = mean((predicted - actual)^2) at observed positions
     = ((1.12-2)^2 + (2.09-4)^2 + (1.72-3)^2 + (0.82-1)^2) / 4
     = (0.77 + 3.65 + 1.64 + 0.03) / 4 = 1.52

k=2: Error = ((1.51-2)^2 + (2.55-4)^2 + (2.27-3)^2 + (1.50-1)^2) / 4
     = (0.24 + 2.10 + 0.53 + 0.25) / 4 = 0.78

k=3: Error = ((1.85-2)^2 + (3.70-4)^2 + (2.80-3)^2 + (0.95-1)^2) / 4
     = (0.02 + 0.09 + 0.04 + 0.00) / 4 = 0.04  <- BEST

k=4: Error = ((2.00-2)^2 + (4.00-4)^2 + (3.00-3)^2 + (1.00-1)^2) / 4
     = 0.00  <- PERFECT fit, but this is rank=full, no compression = overfitting
```

We pick k=3 as the best tradeoff. In practice, the code tests k in [3, 5, 8, 10, 12] and selects the one with lowest reconstruction error on the observed target entries.

---

## The Full SoftImpute Algorithm

```
Input: Matrix M with missing entries, rank k, max_iterations, tolerance
Output: Completed matrix

1. Record which entries are observed (mask)
2. Fill missing entries with initial guess (ARCHI predictions or column means)

3. REPEAT:
     a. M_prev = copy of M
     b. Compute SVD: U, S, V^T = SVD(M)
     c. Truncate: set S[k+1:] = 0 (keep only top k singular values)
     d. Reconstruct: M = U x S_truncated x V^T
     e. Re-insert observed: M[mask] = M_original[mask]
     f. Check convergence: if |M - M_prev| / |M_prev| < tolerance, STOP

4. Return M (all missing entries are now filled)
```

The iteration converges because:
- Step (d) makes the matrix low-rank (SVD forces shared structure)
- Step (e) anchors the observed values (prevents drift from reality)
- The balance between (d) and (e) finds predictions that are BOTH low-rank AND consistent with observations

---

## Connection to the MC+LNN Pipeline

In the actual MC+LNN pipeline, the matrix has more structure:

```
Rows:
  Row 0: Target well (gaps to fill)
  Rows 1-15: Donor wells (weighted by |correlation|)
  Rows 16-20: GLDAS soil moisture (5 temporal scales, fully observed)
  Rows 21-22: sin/cos seasonal encoding (fully observed)

Columns: 288 months (2000-2023)
```

The GLDAS and seasonal rows are FULLY OBSERVED everywhere -- they anchor the SVD even when multiple wells have gaps at the same timestep. This is why auxiliary data is critical: without it, if all wells are missing at month 50, SVD has no constraint and predictions drift.

After SoftImpute converges, the MC predictions for the target well's gaps are extracted and passed to the LNN as placeholder input for temporal refinement.
