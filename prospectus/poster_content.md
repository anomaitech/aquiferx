# POSTER CONTENT — MC+LNN Groundwater Imputation

Use this content to design a 24x36 inch research poster.

---

## TITLE

A Hybrid Matrix Completion + Liquid Neural Network Framework for Groundwater Level Imputation

**Authors:** Henok Teklu
**Affiliation:** Department of Civil & Construction Engineering, Brigham Young University
**Advisor:** Dr. Norm Jones

---

## SECTION 1: The Problem

Groundwater monitoring wells are the primary source of water-table elevation (WTE) data, yet public databases worldwide suffer from long temporal gaps -- periods of months to years with no measurements. Jasechko et al. (2024) analyzed 170,000 wells across 1,693 aquifer systems and found rapid declines in 30%, but most records are too incomplete for trend analysis.

This work introduces the first framework that jointly exploits spatial cross-well correlations and continuous-time temporal dynamics to reconstruct groundwater records with multi-year gaps.

**FIGURE: Gap Illustration**
Show a well time series with:
- Observed data points (solid dots connected by line)
- A SMALL GAP (8 months) labeled "PCHIP fills this" with gold/yellow shading
- A LARGE GAP (3 years) labeled "MC+LNN required" with red shading
- Dashed lines across the gaps
- X-axis: Time (2000-2023), Y-axis: WTE (ft)

---

## SECTION 2: What Is Matrix Completion?

Arrange all wells as columns and all months as rows in a matrix. Most entries are observed, but gap periods are missing. If wells share common patterns -- seasonal cycles, drought responses, recharge trends -- the matrix is approximately low-rank: a few shared patterns explain most of the variation.

Singular Value Decomposition (SVD) recovers these patterns. By iteratively decomposing the matrix, re-inserting known values, and repeating until convergence (SoftImpute), the missing entries are inferred from the structure of observed entries across all 592 wells simultaneously.

The composite matrix includes:
- Target well row -- observed values + gaps to fill
- 15 donor well rows -- most correlated wells, weighted by Pearson r
- 5 GLDAS auxiliary rows -- soil moisture at multiple time scales (globally available)
- Seasonal encoding rows -- sin/cos at 12-month period

The auxiliary and seasonal rows are fully observed for all time steps, anchoring the SVD and constraining predictions during gaps.

**FIGURE: Composite Matrix**
Show a matrix visualization with labeled rows:
- Row 0: Target well (blue cells = observed, gray cells with red border = gaps)
- Rows 1-15: Donor wells (blue, fully observed)
- Rows 16-20: GLDAS aux (green, fully observed)
- Rows 21-22: sin/cos (gold, fully observed)
- Column axis: "288 months (2000-2023)"
- Annotations: "Gaps to fill", "Weighted by correlation", "Anchors the SVD"

---

## SECTION 3: What Is a Liquid Neural Network?

A Liquid Neural Network (LNN) is a continuous-time dynamical system whose hidden state evolves according to an ordinary differential equation (Hasani et al., 2022). Unlike discrete-time networks (LSTM, GRU) that update at fixed steps, LNN's state flows continuously -- naturally accommodating the irregular sampling intervals of groundwater wells without resampling.

**EQUATION:**
x(t + dt) = x * exp(-lambda * dt) + (b / lambda) * (1 - exp(-lambda * dt))
where b = tanh(W_in * input + W_res * x)

The leak rate lambda controls how quickly the state forgets (fast leak = responsive to recent input; slow leak = long memory). The input vector concatenates the current observation (or MC placeholder during gaps), GLDAS soil moisture, and seasonal encoding. The readout is trained via ridge regression on observed values only -- ensuring fidelity to ground truth while using MC's spatial context to drive the reservoir through gap periods.

Hyperparameters (reservoir size 10-80 neurons, leak rate 0.05-0.95, input scaling 0.01-0.40) are auto-optimized via 8 trials per well using Kling-Gupta Efficiency. An ensemble of 3 models (different random seeds) selects the best.

---

## SECTION 4: The Combined Pipeline

**FIGURE: Pipeline Flow Diagram**
Three connected boxes with arrows:

Box 1 (gray): STAGE 1: PCHIP
- Fills small gaps (<=24 months)
- Shape-preserving interpolation
- Densifies the network for better MC donor correlations

Arrow -->

Box 2 (blue): STAGE 2: Matrix Completion
- ARCHI selects top-15 correlated donors via OLS
- SoftImpute on composite matrix (donors + GLDAS + seasonal)
- Adaptive SVD rank selection (3-12)
- Provides spatial context from the full well network

Arrow -->

Box 3 (gold): STAGE 3: LNN Refinement
- Continuous-time CFC dynamics
- MC output drives the reservoir as input
- Readout trains only on real observations
- Captures nonlinear temporal patterns

**KEY DESIGN PRINCIPLE:** MC provides spatial context (what do nearby wells say?). LNN provides temporal dynamics (how does the system evolve between observations?). Neither alone achieves what the coupled framework delivers.

---

## SECTION 5: Cross-Validation Results

Validated on 592 wells, Great Salt Lake Basin, 2000-2023. Most complete well used as target.

### Random Missing Data (50 trials per level)

| % Removed | KGE         | R-squared | RMSE (ft) |
|-----------|-------------|-----------|-----------|
| 5%        | 0.837       | 0.771     | 2.53      |
| 20%       | 0.853       | 0.787     | 2.64      |
| 30%       | 0.844       | 0.770     | 2.79      |
| **50%**   | **0.847**   | **0.788** | **2.74**  |

### Consecutive Year Gaps (20 trials per level)

| Gap Length | KGE         | R-squared | RMSE (ft) |
|------------|-------------|-----------|-----------|
| 1 year     | 0.783       | 0.703     | 2.98      |
| 3 years    | 0.802       | 0.730     | 3.02      |
| **5 years**| **0.815**   | **0.744** | **2.91**  |

### KEY NUMBERS (large, prominent):
- **0.85** KGE at 50% data removed
- **0.82** KGE at 5-year consecutive gap

Performance remains stable across all missing-data rates and gap lengths.

---

## SECTION 6: Conclusions

- **First hybrid spatial-temporal imputation framework** for groundwater: matrix completion exploits cross-well correlations while liquid neural networks capture continuous-time dynamics conditioned on climate auxiliary data

- **KGE > 0.84** under random data loss up to 50%; **KGE > 0.78** for consecutive gaps up to 5 years -- substantially exceeding prior ELM-based methods that degrade below KGE 0.3 for gaps beyond 2 years

- **PCHIP small-gap filling** improves the full pipeline by 10-19% KGE by providing denser donor correlations for the MC stage

- **General-purpose:** Requires only well locations, irregular observations, and globally available GLDAS data. No basin-specific calibration. Applicable to any monitoring network worldwide

---

## SECTION 7: Applicability

The framework is general-purpose, not basin-specific. It requires only:
- Well locations (latitude, longitude) for donor correlation
- Irregular observations at any cadence (daily, monthly, quarterly)
- GLDAS auxiliary data -- globally available satellite-derived soil moisture (no local calibration needed)

All hyperparameters are auto-optimized per well. No manual tuning, no basin-specific training.

---

## REFERENCES

- Hasani, R., et al. (2022). Closed-form continuous-time neural networks. Nature Machine Intelligence, 4, 992-1003.
- Candes, E. J. & Recht, B. (2009). Exact matrix completion via convex optimization. Foundations of Computational Mathematics, 9(6), 717-772.
- Evans, S., et al. (2020). Exploiting Earth Observation Data to Impute Groundwater Level Measurements with an ELM. Remote Sensing, 12, 2044.
- Ramirez, S. G., et al. (2023). Improving Groundwater Imputation through Iterative Refinement. Water, 15, 1236.
- Sharma, Y. K., et al. (2024). Strategic imputation of groundwater data using machine learning. Groundwater for Sustainable Development, 27, 101300.
- Jasechko, S., et al. (2024). Rapid groundwater decline and some cases of recovery in aquifers globally. Nature, 625, 715-720.

---

## DESIGN NOTES

- Poster size: 24 inches tall x 36 inches wide (landscape)
- Color scheme: Navy (#162d50), Gold (#e8b830), White background
- Title font: ~40pt bold sans-serif
- Section headers: ~24pt bold sans-serif
- Body text: ~16pt serif
- Table text: ~14pt sans-serif
- Layout: 3 columns
- Left column: Problem + Gap figure + Matrix Completion explanation
- Center column: LNN explanation + Pipeline + Applicability
- Right column: Results tables + Key numbers + Conclusions
