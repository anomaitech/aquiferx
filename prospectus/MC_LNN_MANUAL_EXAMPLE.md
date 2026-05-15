# MC+LNN Imputation: Step-by-Step Manual Example

This document walks through the entire MC+LNN pipeline with small numbers you can verify by hand.

---

## Setup: 5 wells, 12 months

We have 5 wells monitored over 12 months. **Well C** is our target -- it has a 6-month gap (M3-M8).

```
Raw observations (WTE in feet):

         M1    M2    M3    M4    M5    M6    M7    M8    M9    M10   M11   M12
Well A:  100   102   105   103   101   99    97    98    100   103   106   104
Well B:  200   203   207   204   ---   ---   198   199   201   205   208   206
Well C:  150   153   ---   ---   ---   ---   ---   ---   152   155   158   156  <- TARGET
Well D:  300   304   310   306   302   298   294   296   300   306   312   308
Well E:  250   ---   260   257   253   249   ---   ---   252   257   262   259

GLDAS:   60    65    85    100   91    75    70    65    60    58    62    68
```

Well C has 6 observed months and a 6-month gap (M3-M8).

---

## STAGE 1: PCHIP Small-Gap Fill

PCHIP fills gaps <= 24 months using shape-preserving cubic interpolation from bounding observations.

**Well B** has a 2-month gap (M5-M6). PCHIP uses M4=204 and M7=198 as anchors:

```
Well B M5: PCHIP interpolates between 204 and 198 -> ~202
Well B M6: PCHIP interpolates -> ~200
```

**Well E** has gaps at M2 (1 month) and M7-M8 (2 months). PCHIP fills these:

```
Well E M2: between 250 and 260 -> ~255
Well E M7-M8: between 249 and 252 -> ~250, ~251
```

**Well C** has a 6-month gap (M3-M8). If our threshold is 4 months, this is a **large gap** -- PCHIP does NOT fill it.

After PCHIP, the densified matrix:

```
         M1    M2    M3    M4    M5    M6    M7    M8    M9    M10   M11   M12
Well A:  100   102   105   103   101   99    97    98    100   103   106   104  (complete)
Well B:  200   203   207   204   *202  *200  198   199   201   205   208   206  (now complete)
Well C:  150   153   ---   ---   ---   ---   ---   ---   152   155   158   156  (still has gap)
Well D:  300   304   310   306   302   298   294   296   300   306   312   308  (complete)
Well E:  250   *255  260   257   253   249   *250  *251  252   257   262   259  (now complete)

* = PCHIP-filled values
```

PCHIP densified Wells B and E. Now the donor pool has better coverage for Well C.

---

## STAGE 2a: ARCHI Donor Regression

Select top donors by **Pearson correlation** with Well C on their common observed months.

Well C is observed at months: 1, 2, 9, 10, 11, 12

Compute correlation with each donor on these 6 months:

```
Well C values:     [150, 153, 152, 155, 158, 156]

Well A at M1,2,9,10,11,12: [100, 102, 100, 103, 106, 104]
  Correlation with C: both go up/down together -> r = 0.98

Well D at M1,2,9,10,11,12: [300, 304, 300, 306, 312, 308]
  Correlation with C: same pattern, different scale -> r = 0.97

Well B at M1,2,9,10,11,12: [200, 203, 201, 205, 208, 206]
  Correlation with C: -> r = 0.96

Well E at M1,2,9,10,11,12: [250, 255, 252, 257, 262, 259]
  Correlation with C: -> r = 0.95
```

All 4 donors qualify (r > 0.3). Select all 4 (our max is 15, we only have 4).

**Per-donor OLS regression** on common months:

```
For Well A: C = a x A + b
  Fit on 6 common months:
  a = 1.50, b = 0.0
  So C ~ 1.50 x A

  Predict gap months:
  M3: 1.50 x 105 = 157.5
  M4: 1.50 x 103 = 154.5
  M5: 1.50 x 101 = 151.5
  M6: 1.50 x 99  = 148.5
  M7: 1.50 x 97  = 145.5
  M8: 1.50 x 98  = 147.0
```

Do the same for Wells B, D, E. Then **weighted average** (weight = r-squared):

```
Weight A = 0.98^2 = 0.960
Weight B = 0.96^2 = 0.922
Weight D = 0.97^2 = 0.941
Weight E = 0.95^2 = 0.903

ARCHI prediction for M5:
  = (0.960x151.5 + 0.922x152.0 + 0.941x151.0 + 0.903x151.5) / (0.960+0.922+0.941+0.903)
  = 151.5 (approximately)
```

ARCHI gives a **trend-aware initial estimate** for the gap: [157, 154, 151, 148, 146, 147]

---

## STAGE 2b: Matrix Completion (SoftImpute)

### Step 1: Build raw composite matrix

Rows = target + donors + GLDAS + seasonal. Columns = 12 months.

```
Row 0 (Target C):  150  153  NaN  NaN  NaN  NaN  NaN  NaN  152  155  158  156
Row 1 (Donor A):   100  102  105  103  101   99   97   98  100  103  106  104
Row 2 (Donor B):   200  203  207  204  202  200  198  199  201  205  208  206
Row 3 (Donor D):   300  304  310  306  302  298  294  296  300  306  312  308
Row 4 (Donor E):   250  255  260  257  253  249  250  251  252  257  262  259
Row 5 (GLDAS):      60   65   85  100   91   75   70   65   60   58   62   68
Row 6 (sin):      0.50 0.87 1.00 0.87 0.50 0.00 -.50 -.87 -1.0 -.87 -.50 0.00
Row 7 (cos):      0.87 0.50 0.00 -.50 -.87 -1.0 -.87 -.50 0.00 0.50 0.87 1.00
```

### Step 2: Z-score normalize each row

For each row, compute mean and std from observed values, then normalize:

```
Row 0 (Target C): mean=154, std=2.8
  Normalized: [-1.43, -0.36, NaN, NaN, NaN, NaN, NaN, NaN, -0.71, 0.36, 1.43, 0.71]

Row 1 (Donor A): mean=101.5, std=2.7
  Normalized: [-0.56, 0.19, 1.30, 0.56, -0.19, -0.93, -1.67, -1.30, -0.56, 0.56, 1.67, 0.93]

(similarly for all rows...)

Row 5 (GLDAS): mean=71.6, std=13.2
  Normalized: [-0.88, -0.50, 1.02, 2.15, 1.47, 0.26, -0.12, -0.50, -0.88, -1.03, -0.73, -0.27]

Row 6 (sin): normalize to mean=0, std=0.71
Row 7 (cos): normalize to mean=0, std=0.71
```

### Step 3: Weight donor rows by |r|

```
Row 1 (Donor A): multiply all values by |0.98| = 0.98
Row 2 (Donor B): multiply all values by |0.96| = 0.96
Row 3 (Donor D): multiply all values by |0.97| = 0.97
Row 4 (Donor E): multiply all values by |0.95| = 0.95
```

Highly correlated donors get amplified; weak donors get dampened.

### Step 4: Initialize target row NaN with ARCHI predictions (normalized)

```
ARCHI predicted: [_, _, 157, 154, 151, 148, 146, 147, _, _, _, _]
Normalized: (157-154)/2.8 = 1.07, (154-154)/2.8 = 0, (151-154)/2.8 = -1.07, ...

Row 0 becomes: [-1.43, -0.36, 1.07, 0.00, -1.07, -2.14, -2.86, -2.50, -0.71, 0.36, 1.43, 0.71]
```

### Step 5: Iterative SVD (SoftImpute)

Repeat until convergence:

```
Iteration 1:
  1. SVD: M = U x S x V^T
     With rank k=3, keep only top 3 singular values

  2. Reconstruct: M_new = U_3 x S_3 x V_3^T
     This gives a low-rank approximation of the full matrix

  3. Re-insert observed values:
     M_new[observed positions] = M[observed positions]
     (Target's 6 observed months are restored exactly)
     (All donor observed values are restored)
     (GLDAS and sin/cos rows are fully restored)

  4. The only values that CHANGED are the Target's gap months (M3-M8)
     These are now influenced by the shared structure from donors + GLDAS + seasonal

Iteration 2:
  Repeat SVD -> reconstruct -> re-insert observed
  Gap values shift closer to the true low-rank structure

...after 50-100 iterations, convergence...
```

**What SVD does intuitively:** It finds that all 5 wells share a common seasonal pattern (up in spring, down in summer). The GLDAS row reinforces this. The sin/cos rows encode the 12-month cycle. SVD discovers this shared pattern and fills the target's gap consistently.

### Step 6: Denormalize

```
MC output (normalized): [-1.43, -0.36, 0.95, 0.12, -0.83, -1.90, -2.50, -2.10, -0.71, 0.36, 1.43, 0.71]
Denormalize: value = normalized x std + mean = normalized x 2.8 + 154

MC predictions for gap:
  M3: 0.95 x 2.8 + 154 = 156.7
  M4: 0.12 x 2.8 + 154 = 154.3
  M5: -0.83 x 2.8 + 154 = 151.7
  M6: -1.90 x 2.8 + 154 = 148.7
  M7: -2.50 x 2.8 + 154 = 147.0
  M8: -2.10 x 2.8 + 154 = 148.1
```

### Step 7: MOVE.1 bias correction

Ensure MC predictions match the observed mean and variance:

```
Observed mean = 154.0, observed std = 2.8
MC predicted at observed months: check if mean/std match
If not, scale: pred_corrected = obs_mean + (obs_std/pred_std) x (pred - pred_mean)
```

MC output: **[150, 153, 156.7, 154.3, 151.7, 148.7, 147.0, 148.1, 152, 155, 158, 156]**

---

## STAGE 2c: LNN CFC Temporal Refinement

Now the LNN refines the MC predictions using continuous-time dynamics.

**Key principle:** MC predictions are **reservoir input** during gaps, but the LNN readout is trained **only on real observations**.

For this detailed example, we use: **3 neurons, 4 inputs, 6 months** (3 observed, 3 gap).

### Setup

```
6 months. Months 1-3 observed, months 4-6 are a gap (MC filled).

Observations:    M1=100, M2=103, M3=105, M4=MC:104, M5=MC:102, M6=MC:101
GLDAS soilw:     60,     65,     85,     75,         65,         60
sin(season):     0.50,   0.87,   1.00,   0.87,       0.50,       0.00
cos(season):     0.87,   0.50,   0.00,  -0.50,      -0.87,      -1.00
```

**4 inputs per timestep:** [value, soilw, sin, cos]
**3 neurons:** x1, x2, x3

**Hyperparameters:**
- leak rate lambda = 0.3
- input_scaling = 0.2
- spectral_radius = 0.9
- ridge_alpha = 0.0001
- dt = 1 (one month per step)

### Step 1: Normalize inputs

Normalize the value input to [-0.8, 0.8]:

```
All values: [100, 103, 105, 104, 102, 101]
min = 100, max = 105, range = 5

normalize(v) = ((v - 100) / 5) x 1.6 - 0.8

normalize(100) = -0.80
normalize(103) =  0.16
normalize(105) =  0.80
normalize(104) =  0.48
normalize(102) = -0.16
normalize(101) = -0.48
```

Normalize GLDAS: (v - mean) / std

```
GLDAS mean=68.3, std=9.8
soilw_norm: [-0.85, -0.34, 1.71, 0.68, -0.34, -0.85]
```

**Normalized input vectors:**

```
M1: [-0.80, -0.85,  0.50,  0.87]
M2: [ 0.16, -0.34,  0.87,  0.50]
M3: [ 0.80,  1.71,  1.00,  0.00]
M4: [ 0.48,  0.68,  0.87, -0.50]  <- MC placeholder
M5: [-0.16, -0.34,  0.50, -0.87]  <- MC placeholder
M6: [-0.48, -0.85,  0.00, -1.00]  <- MC placeholder
```

### Step 2: Initialize random weights

**W_in** (3 neurons x 4 inputs), scaled by input_scaling = 0.2:

```
W_in =
         value  soilw   sin    cos
neuron1: [ 0.14, -0.08,  0.18,  0.06]
neuron2: [-0.10,  0.16,  0.04, -0.12]
neuron3: [ 0.06,  0.12, -0.14,  0.10]
```

**W_res** (3 x 3 recurrent), sparse, scaled to spectral_radius = 0.9:

```
W_res =
         x1     x2     x3
neuron1: [ 0.00,  0.45,  0.00]
neuron2: [ 0.60,  0.00, -0.30]
neuron3: [ 0.00,  0.35,  0.00]
```

**Initial state:** x = [0, 0, 0]

### Step 3: CFC update equation

At each timestep:

```
pre = W_in x input
b = tanh(pre + W_res x x)
x_new = x x exp(-lambda x dt) + (b/lambda) x (1 - exp(-lambda x dt))
```

With lambda=0.3, dt=1:

```
exp(-0.3 x 1) = 0.741
(1 - 0.741) / 0.3 = 0.863
```

So: **x_new = x x 0.741 + b x 0.863**

### Step 4: Run reservoir forward (all 6 months)

#### Month 1: input = [-0.80, -0.85, 0.50, 0.87], x = [0, 0, 0]

```
pre = W_in x input:
  pre1 = 0.14x(-0.80) + (-0.08)x(-0.85) + 0.18x(0.50) + 0.06x(0.87)
       = -0.112 + 0.068 + 0.090 + 0.052 = 0.098
  pre2 = (-0.10)x(-0.80) + 0.16x(-0.85) + 0.04x(0.50) + (-0.12)x(0.87)
       = 0.080 - 0.136 + 0.020 - 0.104 = -0.140
  pre3 = 0.06x(-0.80) + 0.12x(-0.85) + (-0.14)x(0.50) + 0.10x(0.87)
       = -0.048 - 0.102 - 0.070 + 0.087 = -0.133

recurrent = W_res x [0,0,0] = [0, 0, 0]

b = tanh(pre + recurrent):
  b1 = tanh(0.098) = 0.098
  b2 = tanh(-0.140) = -0.139
  b3 = tanh(-0.133) = -0.132

x_new = x x 0.741 + b x 0.863:
  x1 = 0 x 0.741 + 0.098 x 0.863 = 0.085
  x2 = 0 x 0.741 + (-0.139) x 0.863 = -0.120
  x3 = 0 x 0.741 + (-0.132) x 0.863 = -0.114

State after M1: x = [0.085, -0.120, -0.114]
Store: states[1] = [1, 0.085, -0.120, -0.114]
```

#### Month 2: input = [0.16, -0.34, 0.87, 0.50], x = [0.085, -0.120, -0.114]

```
pre:
  pre1 = 0.14x0.16 + (-0.08)x(-0.34) + 0.18x0.87 + 0.06x0.50
       = 0.022 + 0.027 + 0.157 + 0.030 = 0.236
  pre2 = (-0.10)x0.16 + 0.16x(-0.34) + 0.04x0.87 + (-0.12)x0.50
       = -0.016 - 0.054 + 0.035 - 0.060 = -0.096
  pre3 = 0.06x0.16 + 0.12x(-0.34) + (-0.14)x0.87 + 0.10x0.50
       = 0.010 - 0.041 - 0.122 + 0.050 = -0.103

recurrent = W_res x x:
  rec1 = 0.00x0.085 + 0.45x(-0.120) + 0.00x(-0.114) = -0.054
  rec2 = 0.60x0.085 + 0.00x(-0.120) + (-0.30)x(-0.114) = 0.051 + 0.034 = 0.085
  rec3 = 0.00x0.085 + 0.35x(-0.120) + 0.00x(-0.114) = -0.042

b = tanh(pre + recurrent):
  b1 = tanh(0.236 - 0.054) = tanh(0.182) = 0.180
  b2 = tanh(-0.096 + 0.085) = tanh(-0.011) = -0.011
  b3 = tanh(-0.103 - 0.042) = tanh(-0.145) = -0.144

x_new:
  x1 = 0.085 x 0.741 + 0.180 x 0.863 = 0.063 + 0.155 = 0.218
  x2 = (-0.120) x 0.741 + (-0.011) x 0.863 = -0.089 - 0.009 = -0.098
  x3 = (-0.114) x 0.741 + (-0.144) x 0.863 = -0.084 - 0.124 = -0.209

State after M2: x = [0.218, -0.098, -0.209]
Store: states[2] = [1, 0.218, -0.098, -0.209]
```

#### Month 3: input = [0.80, 1.71, 1.00, 0.00], x = [0.218, -0.098, -0.209]

```
pre:
  pre1 = 0.14x0.80 + (-0.08)x1.71 + 0.18x1.00 + 0.06x0.00
       = 0.112 - 0.137 + 0.180 + 0 = 0.155
  pre2 = (-0.10)x0.80 + 0.16x1.71 + 0.04x1.00 + (-0.12)x0.00
       = -0.080 + 0.274 + 0.040 + 0 = 0.234
  pre3 = 0.06x0.80 + 0.12x1.71 + (-0.14)x1.00 + 0.10x0.00
       = 0.048 + 0.205 - 0.140 + 0 = 0.113

recurrent:
  rec1 = 0.45x(-0.098) = -0.044
  rec2 = 0.60x0.218 + (-0.30)x(-0.209) = 0.131 + 0.063 = 0.194
  rec3 = 0.35x(-0.098) = -0.034

b:
  b1 = tanh(0.155 - 0.044) = tanh(0.111) = 0.111
  b2 = tanh(0.234 + 0.194) = tanh(0.428) = 0.404
  b3 = tanh(0.113 - 0.034) = tanh(0.079) = 0.079

x_new:
  x1 = 0.218 x 0.741 + 0.111 x 0.863 = 0.162 + 0.096 = 0.257
  x2 = (-0.098) x 0.741 + 0.404 x 0.863 = -0.073 + 0.349 = 0.276
  x3 = (-0.209) x 0.741 + 0.079 x 0.863 = -0.155 + 0.068 = -0.087

State after M3: x = [0.257, 0.276, -0.087]
Store: states[3] = [1, 0.257, 0.276, -0.087]
```

#### Month 4 (GAP): input = [0.48, 0.68, 0.87, -0.50], x = [0.257, 0.276, -0.087]

**Note: input[0] = 0.48 is the MC placeholder, not a real observation!**

```
pre:
  pre1 = 0.14x0.48 + (-0.08)x0.68 + 0.18x0.87 + 0.06x(-0.50)
       = 0.067 - 0.054 + 0.157 - 0.030 = 0.139
  pre2 = (-0.10)x0.48 + 0.16x0.68 + 0.04x0.87 + (-0.12)x(-0.50)
       = -0.048 + 0.109 + 0.035 + 0.060 = 0.156
  pre3 = 0.06x0.48 + 0.12x0.68 + (-0.14)x0.87 + 0.10x(-0.50)
       = 0.029 + 0.082 - 0.122 - 0.050 = -0.061

recurrent:
  rec1 = 0.45x0.276 = 0.124
  rec2 = 0.60x0.257 + (-0.30)x(-0.087) = 0.154 + 0.026 = 0.180
  rec3 = 0.35x0.276 = 0.097

b:
  b1 = tanh(0.139 + 0.124) = tanh(0.263) = 0.257
  b2 = tanh(0.156 + 0.180) = tanh(0.336) = 0.323
  b3 = tanh(-0.061 + 0.097) = tanh(0.036) = 0.036

x_new:
  x1 = 0.257 x 0.741 + 0.257 x 0.863 = 0.190 + 0.222 = 0.412
  x2 = 0.276 x 0.741 + 0.323 x 0.863 = 0.205 + 0.279 = 0.483
  x3 = (-0.087) x 0.741 + 0.036 x 0.863 = -0.064 + 0.031 = -0.033

State after M4: x = [0.412, 0.483, -0.033]
Store: states[4] = [1, 0.412, 0.483, -0.033]
```

#### Month 5 (GAP): input = [-0.16, -0.34, 0.50, -0.87], x = [0.412, 0.483, -0.033]

```
pre:
  pre1 = 0.14x(-0.16) + (-0.08)x(-0.34) + 0.18x0.50 + 0.06x(-0.87)
       = -0.022 + 0.027 + 0.090 - 0.052 = 0.043
  pre2 = (-0.10)x(-0.16) + 0.16x(-0.34) + 0.04x0.50 + (-0.12)x(-0.87)
       = 0.016 - 0.054 + 0.020 + 0.104 = 0.086
  pre3 = 0.06x(-0.16) + 0.12x(-0.34) + (-0.14)x0.50 + 0.10x(-0.87)
       = -0.010 - 0.041 - 0.070 - 0.087 = -0.208

recurrent:
  rec1 = 0.45x0.483 = 0.217
  rec2 = 0.60x0.412 + (-0.30)x(-0.033) = 0.247 + 0.010 = 0.257
  rec3 = 0.35x0.483 = 0.169

b:
  b1 = tanh(0.043 + 0.217) = tanh(0.260) = 0.254
  b2 = tanh(0.086 + 0.257) = tanh(0.343) = 0.330
  b3 = tanh(-0.208 + 0.169) = tanh(-0.039) = -0.039

x_new:
  x1 = 0.412 x 0.741 + 0.254 x 0.863 = 0.305 + 0.219 = 0.524
  x2 = 0.483 x 0.741 + 0.330 x 0.863 = 0.358 + 0.285 = 0.643
  x3 = (-0.033) x 0.741 + (-0.039) x 0.863 = -0.024 - 0.034 = -0.058

State after M5: x = [0.524, 0.643, -0.058]
Store: states[5] = [1, 0.524, 0.643, -0.058]
```

#### Month 6 (GAP): input = [-0.48, -0.85, 0.00, -1.00], x = [0.524, 0.643, -0.058]

```
pre:
  pre1 = 0.14x(-0.48) + (-0.08)x(-0.85) + 0.18x0.00 + 0.06x(-1.00)
       = -0.067 + 0.068 + 0 - 0.060 = -0.059
  pre2 = (-0.10)x(-0.48) + 0.16x(-0.85) + 0.04x0.00 + (-0.12)x(-1.00)
       = 0.048 - 0.136 + 0 + 0.120 = 0.032
  pre3 = 0.06x(-0.48) + 0.12x(-0.85) + (-0.14)x0.00 + 0.10x(-1.00)
       = -0.029 - 0.102 + 0 - 0.100 = -0.231

recurrent:
  rec1 = 0.45x0.643 = 0.289
  rec2 = 0.60x0.524 + (-0.30)x(-0.058) = 0.314 + 0.017 = 0.332
  rec3 = 0.35x0.643 = 0.225

b:
  b1 = tanh(-0.059 + 0.289) = tanh(0.230) = 0.226
  b2 = tanh(0.032 + 0.332) = tanh(0.364) = 0.349
  b3 = tanh(-0.231 + 0.225) = tanh(-0.006) = -0.006

x_new:
  x1 = 0.524 x 0.741 + 0.226 x 0.863 = 0.388 + 0.195 = 0.583
  x2 = 0.643 x 0.741 + 0.349 x 0.863 = 0.476 + 0.301 = 0.777
  x3 = (-0.058) x 0.741 + (-0.006) x 0.863 = -0.043 - 0.005 = -0.048

State after M6: x = [0.583, 0.777, -0.048]
Store: states[6] = [1, 0.583, 0.777, -0.048]
```

### Step 5: Full state matrix

```
                bias    x1      x2      x3
states[1]:    [ 1,    0.085, -0.120, -0.114 ]   <- observed
states[2]:    [ 1,    0.218, -0.098, -0.209 ]   <- observed
states[3]:    [ 1,    0.257,  0.276, -0.087 ]   <- observed
states[4]:    [ 1,    0.412,  0.483, -0.033 ]   <- GAP (MC input)
states[5]:    [ 1,    0.524,  0.643, -0.058 ]   <- GAP (MC input)
states[6]:    [ 1,    0.583,  0.777, -0.048 ]   <- GAP (MC input)
```

### Step 6: Train readout (ONLY on observed months 1-3)

Training matrix S (only rows where we have real observations):

```
S = [ 1,  0.085, -0.120, -0.114 ]   -> target: -0.80  (normalized 100)
    [ 1,  0.218, -0.098, -0.209 ]   -> target:  0.16  (normalized 103)
    [ 1,  0.257,  0.276, -0.087 ]   -> target:  0.80  (normalized 105)

Shape: 3 x 4
y = [-0.80, 0.16, 0.80]
Shape: 3 x 1
```

**Ridge regression:** W_out = (S^T S + alpha x I)^(-1) x S^T x y

```
S^T S (4x4):
        bias    x1      x2      x3
bias: [ 3.000,  0.560,  0.058, -0.410 ]
x1:   [ 0.560,  0.120,  0.035, -0.071 ]
x2:   [ 0.058,  0.035,  0.100, -0.021 ]
x3:   [-0.410, -0.071, -0.021,  0.069 ]

S^T y (4x1):
  bias: 1x(-0.80) + 1x(0.16) + 1x(0.80) = 0.160
  x1:   0.085x(-0.80) + 0.218x(0.16) + 0.257x(0.80) = 0.173
  x2:   (-0.120)x(-0.80) + (-0.098)x(0.16) + 0.276x(0.80) = 0.301
  x3:   (-0.114)x(-0.80) + (-0.209)x(0.16) + (-0.087)x(0.80) = -0.012

Solve: W_out = (S^T S + alpha x I)^(-1) x S^T y

W_out ~ [-1.25, 1.80, 2.15, 0.42]
          bias   w1    w2    w3
```

### Step 7: Predict ALL 6 months

```
prediction[t] = W_out . states[t] = -1.25 + 1.80 x x1 + 2.15 x x2 + 0.42 x x3

Month 1: -1.25 + 1.80x0.085 + 2.15x(-0.120) + 0.42x(-0.114)
       = -1.25 + 0.153 - 0.258 - 0.048 = -1.403
       -> denormalize: ((-1.403+0.8)/1.6) x 5 + 100 = 98.1

Month 2: -1.25 + 1.80x0.218 + 2.15x(-0.098) + 0.42x(-0.209)
       = -1.25 + 0.392 - 0.211 - 0.088 = -1.157
       -> denormalize: 98.9

Month 3: -1.25 + 1.80x0.257 + 2.15x0.276 + 0.42x(-0.087)
       = -1.25 + 0.463 + 0.593 - 0.037 = -0.231
       -> denormalize: 101.8

Month 4 (GAP): -1.25 + 1.80x0.412 + 2.15x0.483 + 0.42x(-0.033)
       = -1.25 + 0.742 + 1.039 - 0.014 = 0.517
       -> denormalize: 104.1

Month 5 (GAP): -1.25 + 1.80x0.524 + 2.15x0.643 + 0.42x(-0.058)
       = -1.25 + 0.943 + 1.382 - 0.024 = 1.051
       -> denormalize: 105.8

Month 6 (GAP): -1.25 + 1.80x0.583 + 2.15x0.777 + 0.42x(-0.048)
       = -1.25 + 1.049 + 1.671 - 0.020 = 1.450
       -> denormalize: 107.0
```

### Step 8: Compare all stages

```
              M1     M2     M3     M4     M5     M6
Observed:    100    103    105     ?      ?      ?
ARCHI:        -      -      -    154    151    148
MC:           -      -      -    154.3  151.7  148.7
LNN:         98.1   98.9  101.8  104.1  105.8  107.0
```

Each stage refines the previous:
- **ARCHI:** Donor regression gives a reasonable first guess
- **MC:** SVD sharpens it using full spatial+auxiliary structure
- **LNN:** Continuous-time dynamics add nonlinear temporal refinement, trained only on ground truth

---

## Why the LNN Design Works

1. **MC placeholders keep the reservoir alive during gaps.** Without them, the reservoir state would decay to zero (leak rate kills it), and the readout would have no meaningful state to map from.

2. **Training only on observations prevents error propagation.** If the readout were trained on MC placeholders too, any MC error would be baked into the weights. By training only on real observations, the readout learns the true input->output mapping.

3. **GLDAS and seasonal inputs are always available.** Even during multi-year gaps, the reservoir receives real climate information through inputs 2-8, keeping its dynamics grounded in physical reality.

4. **The leak rate controls memory.** With lambda=0.3, the reservoir retains 74% of its state each month. After a 12-month gap, it still carries 0.741^12 = 4% of the pre-gap state -- weak but nonzero. The MC placeholders and GLDAS inputs compensate by providing fresh input throughout.
