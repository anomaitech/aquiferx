# PhD Prospectus

# Advancing Basin-Scale Groundwater Storage Estimation Through Hybrid Time-Series Imputation, Spatiotemporal Interpolation, and Satellite-Derived Leakage Correction

---

## 1. Introduction

### 1.1 Overall Objective and Significance

Groundwater sustains roughly half of global irrigation and a third of municipal supply, yet remains the most poorly monitored component of the terrestrial water budget (Scanlon et al., 2023). In arid and semi-arid basins, where surface reservoirs are increasingly stressed by warming-driven evaporation and growing demand, reliable quantification of groundwater storage change is essential for drought preparedness, allocation policy, and long-term sustainability assessment. The problem is acute: Jasechko et al. (2024) analyzed 170,000 wells across 1,693 aquifer systems worldwide and found rapid declines exceeding 0.5 m/yr in a substantial fraction, with 30% of regional aquifers showing accelerating loss over the past four decades.

Three independent lines of evidence bear on basin-scale groundwater storage change, each with fundamental limitations. In situ monitoring well networks provide direct water-level observations but are temporally sparse and spatially uneven. Public databases such as the USGS National Water Information System contain millions of individual measurements, yet the median well has fewer than 50 observations spanning decades of record, with gaps of multiple years commonplace. Satellite gravimetry from the Gravity Recovery and Climate Experiment (GRACE) and its successor GRACE-FO resolves total water storage anomalies globally at monthly resolution (Tapley et al., 2004), but the native spatial resolution of approximately 300 km makes basin-scale interpretation subject to signal leakage across boundaries (Vishwakarma et al., 2018), and isolating the groundwater component requires subtracting auxiliary estimates of soil moisture, snow, surface water, and canopy storage from land-surface models whose own fidelity is uncertain. Land-surface models such as those within the Global Land Data Assimilation System (GLDAS; Rodell et al., 2004) provide spatially continuous, temporally complete estimates but inherit assumptions about subsurface representation, irrigation withdrawals, and terminal-lake hydrology that may not hold in heterogeneous, anthropogenically stressed basins.

The overall objective of this dissertation is to develop and demonstrate an integrated workflow that substantially improves basin-scale groundwater storage estimation in heterogeneous basins by: (1) reconstructing temporally complete well records from sparse in situ observations using a novel hybrid imputation framework; (2) interpolating those reconstructed records to produce continuous spatiotemporal groundwater-level fields; and (3) using those fields to derive spatially distributed GRACE leakage corrections that replace the basin-uniform scalars currently applied in practice. The Great Salt Lake Basin (GSLB), a closed hydrologic system in the Intermountain West with concentrated anthropogenic pumping along the Wasatch Front and a hydrologically significant terminal lake, serves as the demonstration basin throughout. The methods developed are intended to be transferable to other basins with analogous characteristics: heterogeneous hydrogeology, sparse and irregular well coverage, and spatially concentrated anthropogenic stress.

### 1.2 Background

#### Groundwater Monitoring and the Gap Problem

The utility of in situ groundwater-level records for basin-scale analysis depends critically on the temporal completeness and spatial density of the monitoring network. In practice, neither condition is met at scales relevant to water-resources management. Wells are installed, monitored intermittently, and often abandoned over the course of decades, producing records with gaps ranging from months to years. Evans et al. (2020a) developed the Groundwater Level Mapping Tool (GWLMT), an open-source web application that couples Extreme Learning Machines (ELM; Huang et al., 2006) with Earth observation data from GRACE and GLDAS to impute missing groundwater levels, demonstrating that satellite-derived auxiliary variables can substantially improve gap-filling in regions where in situ data are sparse. In a companion study, Evans et al. (2020b) showed that the ELM-based approach with Earth observation inputs achieves superior imputation compared to conventional interpolation, particularly for wells with limited observation histories. Ramirez et al. (2022) extended this framework by incorporating inductive bias from remote sensing into machine learning imputation, demonstrating improved performance when GRACE-derived total water storage anomalies and GLDAS soil moisture products are used as predictive features. Ramirez et al. (2023) further advanced the methodology through iterative refinement that exploits both spatial correlations from neighboring wells and temporal correlations from auxiliary variables, showing that sequential incorporation of in situ spatial context improves imputation accuracy over single-pass approaches.

This line of work has been applied to basin-scale groundwater storage assessment. Stevens et al. (2025) applied the GWLMT methodology to California's Central Valley, developing a novel in situ-based storage estimation approach and comparing it against GRACE-derived estimates, finding systematic differences that highlight the importance of imputation quality for basin-scale conclusions. Shepard et al. (2025) applied the same tool to the Klamath watershed in Oregon, demonstrating its applicability to groundwater-driven basins with complex surface-water interactions.

These ELM-based approaches, while effective, operate primarily in the temporal domain: each well is imputed independently using auxiliary time series as features, with spatial information incorporated only through remote sensing products that provide basin-average (not well-specific) spatial context. Jeong et al. (2020) demonstrated that Long Short-Term Memory (LSTM) networks can reconstruct missing groundwater levels with R-squared exceeding 0.9 for wells with spatially correlated neighbors, but performance degrades sharply when gaps exceed one to two years and when neighboring wells are themselves sparse. Wunsch et al. (2021) compared deep learning architectures including LSTM, convolutional neural networks, and nonlinear autoregressive networks for groundwater-level forecasting, finding that multivariate temporal architectures consistently outperform univariate models but remain fundamentally limited by the availability of training data during gap periods. Gharehbaghi et al. (2022) showed that Gated Recurrent Unit (GRU) networks perform comparably to LSTM with lower computational cost, and Lin et al. (2022) achieved R-squared of 0.86 with a double-GRU architecture for monthly predictions. However, all of these approaches — ELM, LSTM, and GRU alike — treat each well's temporal imputation as essentially independent, incorporating spatial information only indirectly through satellite-derived covariates rather than directly exploiting the cross-well correlation structure of the monitoring network.

Matrix completion methods offer a fundamentally different perspective by arranging the full set of well records as a partially observed matrix (wells by time) and exploiting low-rank structure to infer missing entries directly from the spatial correlation across wells. The theoretical foundations were established by Candes and Recht (2009), who proved that low-rank matrices can be recovered exactly from a surprisingly small number of observed entries under incoherence conditions. Poudevigne and Jones (2024) developed a Hankel Imputation method using block-Hankel matrix completion that performs competitively for time-series interpolation and excels at reproducing sharp peaks that classical methods miss. Prior studies have evaluated multiple imputation strategies for monthly groundwater levels and found that Soft Imputation (a matrix-completion variant) excels in sparse networks, precisely the setting where deep learning methods struggle most.

Recent advances in continuous-time neural network architectures offer particular promise for irregularly sampled hydrological data. Hasani et al. (2021) introduced Liquid Time-Constant Networks (LTCs), in which time enters as a structural property of the dynamical model through time-varying ODE coefficients, rather than as an engineered input feature. The subsequent Closed-form Continuous-depth (CfC) architecture (Hasani et al., 2022) achieves one order of magnitude faster training and inference than LTCs by eliminating the need for numerical ODE solvers, directly relevant to modeling the irregular sampling cadence of groundwater well records. Sun et al. (2025) extended matrix completion theory to incorporate both subject-specific and time-specific covariates, providing a formal framework for combining matrix completion with auxiliary climatic forcings.

A critical gap in the existing literature is the absence of imputation frameworks that jointly exploit both spatial structure (cross-well correlations) and temporal dynamics (auxiliary-driven nonlinear evolution) within a single integrated pipeline. The ELM-based approaches of Evans et al. (2020a) and Ramirez et al. (2023) incorporate temporal auxiliary features but treat spatial context only through basin-scale remote sensing products. Matrix completion methods exploit spatial correlation but do not model temporal dynamics or incorporate auxiliary forcings. Deep learning approaches (LSTM, GRU) model temporal dynamics but treat wells independently. This dissertation proposes a coupled Matrix Completion + Liquid Neural Network (MC+LNN) framework that explicitly addresses this gap: matrix completion provides spatially informed initial estimates by exploiting the correlation structure across the full well network, while the LNN refines those estimates using continuous-time dynamics conditioned on auxiliary climatic forcings. The two components are complementary — MC handles the spatial dimension, LNN handles the temporal dimension — and their coupling produces imputation quality that neither achieves alone.

Hybrid approaches that combine multiple information sources are increasingly recognized as superior to single-method frameworks. Rojas et al. (2025) comprehensively evaluated classical, ensemble, and deep learning approaches for single- and multi-well groundwater imputation, finding that multi-well strategies incorporating inter-well similarity consistently outperform univariate methods. Senanayake et al. (2024) integrated Bayesian imputation with deep learning and demonstrated 15-25% improvement in imputation accuracy through transfer learning across monitoring networks. Ramirez et al. (2022) showed that incorporating GRACE and GLDAS observations as auxiliary inputs to machine learning models substantially improves imputation for wells in data-sparse regions.

#### Spatial Interpolation of Groundwater Levels

Converting imputed point well records to continuous groundwater-level fields requires spatial interpolation. Geostatistical methods, particularly ordinary kriging, have been the dominant approach for decades, but their stationarity assumptions are frequently violated in heterogeneous basins where water-table elevation varies by hundreds to thousands of meters over distances of tens of kilometers. Van der Lugt et al. (2024) applied Empirical Bayesian Kriging to 11,100 km-squared of groundwater data in the Netherlands, finding it outperforms ordinary and universal kriging. Tao et al. (2024) compared machine learning models with geostatistical interpolation and found that Random Forest Spatial Interpolation (RFSI) achieves R-squared of 0.86 versus 0.75 for conventional Random Forest, demonstrating that explicit spatial encoding improves predictions. Li et al. (2025) developed spatial RF models for high-resolution regional groundwater-level mapping incorporating environmental covariates. Ahmadi et al. (2024) demonstrated that hybrid approaches combining Empirical Bayesian Kriging with machine learning models reduce RMSE by 41% compared to individual algorithms.

Empirical Orthogonal Function (EOF) analysis, equivalent to Principal Component Analysis applied to spatiotemporal fields, has been used in climate science for decades to decompose spatial fields into dominant modes of variability. The approach decomposes a wells-by-time matrix into temporal modes (shared patterns) and spatial loadings (per-well weights), enabling interpolation by estimating loadings at unobserved locations rather than interpolating raw values directly. Wu et al. (2025) used graph neural networks to capture spatial dependencies among wells for groundwater-level forecasting, integrating topological and environmental factors, suggesting that graph-based spatial representations may further improve interpolation accuracy.

#### GRACE Leakage Correction

The GRACE partition equation for groundwater storage anomalies (GWSa) is:

GWSa = Lf * TWSa - SMa - SWEa - SWSa - CANa

where TWSa is total water storage anomaly, SM is soil moisture, SWE is snow water equivalent, SWS is surface water storage, CAN is canopy storage, and Lf is the leakage correction factor. In conventional applications, Lf is a basin-uniform scalar, typically calibrated against independent estimates or set to unity. Long et al. (2015) established the forward modeling approach for GRACE leakage correction, showing improvements of 37% in annual amplitudes and 36% in trends relative to uncorrected estimates. Ma et al. (2024) introduced Coordinated Forward Modeling (CoFM) that iteratively calibrates specific yield between GRACE and in situ observations at 0.5-degree scale, demonstrating that sub-regional trends can diverge substantially from basin-average behavior. Tripathi et al. (2022) showed that basin-average grid-scaled GRACE can be misleading due to compensating over- and under-scaled pixels, recommending grid-level assessment before downstream applications. Li et al. (2024) used in situ groundwater observations and aquifer storage coefficients as a priori information to estimate pixel-scale leakage correction factors via forward modeling, establishing precedent for the spatially distributed correction proposed in this dissertation. Croteau et al. (2021) demonstrated that mascon solutions with inter-mascon correlations in regularization outperform diagonal regularizations, reducing leakage especially across coastlines.

#### Great Salt Lake Basin

The GSLB is a closed hydrologic system covering approximately 93,000 km-squared. Consumptive water uses have depleted inflows to the Great Salt Lake by 39%, lowering the lake 3.4 m and reducing volume by 64% (Null and Wurtsbaugh, 2020). Wine (2019) argued that attributing the decline to climate change obscures the dominant role of consumptive water use, while Bigalke et al. (2025) attributed approximately two-thirds of the 2022 record-low volume to reduced streamflow and one-third to increased evaporation from climate warming. Hall et al. (2024) used GRACE/GRACE-FO to document 68.7 km-cubed of groundwater loss from 2002-2023 across the broader Great Basin, finding that even record snow years fail to reverse the downward trend. Zamora and Inkenbrandt (2024) revised the groundwater contribution to the Great Salt Lake upward from the historical 3% estimate to approximately 10% of total inflows, substantially changing the water budget. Rateb and Herring (2020) compared GRACE groundwater storage with approximately 23,000 monitoring wells across 14 major US aquifers and found correlations of R=0.52-0.95, providing context for GRACE-in situ comparison in the GSLB. No prior study has produced a long-term, multi-method groundwater storage record specific to the full GSLB, nor has the GRACE leakage correction been calibrated at sub-basin spatial resolution using spatially continuous imputed well records.

### 1.3 Specific Objectives

**Objective 1 (Paper 1).** Quantify multi-decadal groundwater storage change in the Great Salt Lake Basin (2002-2024) by integrating GRACE-derived, GLDAS-derived, and in situ estimates within a unified framework, and assess the methodological consequences of including surface-water storage and applying an empirically calibrated GRACE leakage correction.

**Objective 2 (Paper 2).** Develop and validate a hybrid imputation framework for sparse, irregular groundwater-level time series that combines Piecewise Cubic Hermite Interpolating Polynomials (PCHIP) for short-duration gaps with a coupled Matrix Completion and Liquid Neural Network (MC+LNN) approach for long-duration gaps, using auxiliary climatic forcings as continuous-time inputs.

**Objective 3 (Paper 3).** Develop a spatiotemporal interpolation framework using spatial trend decomposition and Empirical Orthogonal Function analysis to produce continuous groundwater-level fields from imputed well records, and apply those fields to derive spatially distributed GRACE leakage correction factors for the GSLB, replacing the basin-uniform scalar of Paper 1 with a pixel-wise grid calibrated against the actual spatial pattern of mass change.

---

## 2. Objective 1 (Paper 1)

### 2.1 Objective

To quantify groundwater storage change in the Great Salt Lake Basin from 2002 through 2024 using multiple independent estimation methods, evaluate methodological sensitivities related to surface-water inclusion and GRACE leakage, and produce a continuous record suitable for water-resources planning and drought-response assessment.

### 2.2 Background

The GSLB is a closed hydrologic system spanning four states. Sustained low Great Salt Lake levels have raised ecological and public-health concerns (Abbott et al., 2023), but the basin-scale groundwater contribution has been characterized only at sub-basin scale or in steady-state flow analyses (Zamora and Inkenbrandt, 2024). Prior GRACE-based studies have addressed the broader Great Basin (Hall et al., 2024) and the localized groundwater loss around the Great Salt Lake via GPS constraints (Young et al., 2021), but no long-term, multi-method record specific to the full GSLB has been produced.

### 2.3 Methods

Five GWSa estimates were computed over 2002-2024:

1. **GRACE-raw**: JPL GRACE TWSa minus GLDAS v2.1 soil moisture, snow water equivalent, and canopy storage.
2. **GRACE-sw**: Identical to GRACE-raw but with surface-water storage from 19 reservoirs plus the Great Salt Lake subtracted explicitly.
3. **GRACE-Lf**: The surface-water-adjusted estimate with an empirical leakage factor Lf=2 applied multiplicatively to TWSa, with Lf calibrated against in situ observations.
4. **GLDAS-2.2**: The GRACE-assimilated CLSM groundwater product.
5. **GWDM**: An in situ estimate built from approximately 1,200 USGS wells via the Groundwater Data Mapper Tool workflow (Evans et al., 2020a; Evans et al., 2020b), using PCHIP for short temporal gaps, an Extreme Learning Machine with Earth observation inputs (Huang et al., 2006; Ramirez et al., 2022) for longer discontinuities, iterative spatial-temporal refinement (Ramirez et al., 2023), ordinary Kriging for spatial interpolation, and a basin-representative specific yield of 0.15 to convert water-level change to volumetric storage change.

The five estimates were compared via Pearson correlation and coefficient of determination, and directly benchmarked against the independent GPS-based estimate of Young, Kreemer, and Blewitt (2021).

### 2.4 Results

All four independent methods identified two major drawdown intervals (2012-2016 and 2019-2022) with only partial recovery in between. The GRACE-Lf estimate yielded a 2011-2016 drought-period loss of approximately 10.1 km-cubed, consistent with the GPS-based estimate of 10.9 +/- 2.8 km-cubed reported by Young, Kreemer, and Blewitt (2021). Including surface-water storage in the partition substantially altered GRACE-derived GWSa, with approximately 31% of basin total storage change attributable to surface-water variability, primarily the Great Salt Lake. Applying the basin-uniform leakage factor improved the Pearson correlation between GRACE-derived and in situ GWSa from 0.17 to 0.77. Annual precipitation correlated most strongly with in situ GWSa at a two-year lag and with three-year cumulative rainfall (r=0.67), consistent with multi-year recharge memory in the basin.

---

## 3. Objective 2 (Paper 2)

### 3.1 Objective

To develop, validate, and benchmark a hybrid imputation framework for groundwater-level time series that combines classical interpolation, matrix completion methods, and continuous-time neural networks, addressing the long-gap limitation that constrains the utility of standard interpolators in sparse public well databases.

### 3.2 Methods

The proposed framework imputes individual well records via a two-stage pipeline operating on monthly-aggregated well observations from 2000-2023.

**Stage 1: PCHIP Small-Gap Fill.** Gaps shorter than 24 months are filled using Piecewise Cubic Hermite Interpolating Polynomials (PCHIP), which preserve the local shape and monotonic structure of observed records without introducing the spurious oscillations of cubic splines (Mirzavand et al., 2020). PCHIP interpolation densifies the well network by filling short interruptions in otherwise well-sampled records, providing the spatial coverage needed for the subsequent matrix completion stage. Large gaps exceeding 24 months are identified and their interiors are blanked while retaining 6-month pads at the gap edges.

**Stage 2: MC+LNN Large-Gap Fill.** Long-duration gaps are filled via a coupled Matrix Completion and Liquid Neural Network approach with three phases:

*Phase 2a: ARCHI Donor Regression.* For each target well with a large gap, the top 15 most correlated donor wells are identified from the PCHIP-densified network based on Pearson correlation over common observation periods. Per-donor OLS regression (target = a*donor + b) is fitted on the overlapping observations, and a weighted average of donor predictions (weight = r-squared) provides an initial trend-aware estimate for the gap period.

*Phase 2b: SoftImpute Matrix Completion.* A composite matrix is constructed with rows representing the target well, weighted donor wells, GLDAS auxiliary variables (soil moisture at multiple temporal scales), and seasonal encoding (sin/cos at 12-month period). The matrix is normalized per-row using z-score standardization. Truncated SVD with adaptive rank selection (testing ranks 3, 5, 8, 10, 12 and selecting by minimum reconstruction error on observed target entries) is applied iteratively, with observed entries re-inserted after each SVD step, until convergence (relative Frobenius norm change below 10^-7). A MOVE.1 variance-preserving bias correction is applied to the target row predictions to match the observed mean and standard deviation of the target well.

*Phase 2c: LNN CFC Temporal Refinement.* A Liquid Neural Network with Closed-form Continuous-time Functional Closure (CFC) dynamics (Hasani et al., 2022) refines the MC predictions. The reservoir state evolves according to:

x(t+dt) = x * exp(-leak * dt) + (b / leak) * (1 - exp(-leak * dt))

where b = tanh(W_in * input + W_res * x), and the input vector concatenates the current observation (or MC placeholder), auxiliary variables, and seasonal encoding. The key design principle is that MC predictions serve as reservoir input (placeholders) during gap periods, but the readout weights are trained only on real observations via ridge regression. This ensures the LNN learns the temporal dynamics from ground truth while using MC's spatial context to drive the reservoir through gaps. Hyperparameters (reservoir size 10-80, leak rate 0.05-0.95, input scaling 0.01-0.40) are optimized via Bayesian grid search over 8 trials per well, with an ensemble of 3 LNN models (different random seeds) selecting the best by Kling-Gupta Efficiency on observations.

**Validation.** The framework is validated on the GSLB well network (592 wells, 288 months, 2000-2023) using the most complete well (415703112514501, 202 observed months out of 288) as the primary target, with cross-validation under two scenarios:
- *Random missing data*: 5%, 10%, 20%, 30%, 40%, 50% of observed months removed (50 trials each)
- *Consecutive year gaps*: 1, 2, 3, 4, 5 years of continuous data removed (20 trials each)

Performance metrics: Kling-Gupta Efficiency (KGE), R-squared, RMSE, MAE, and NSE. Comparison baselines include pure PCHIP, the ELM-based approach from Paper 1, a pure matrix completion baseline, and a pure LNN baseline, isolating the contribution of each pipeline stage.

### 3.3 Anticipated Results

Cross-validation results demonstrate that the MC+LNN framework substantially outperforms the ELM-based approach from Paper 1. Under random missing data scenarios, the pipeline achieves KGE of 0.84-0.85 across all missing-data percentages from 5% to 50%, with standard deviation decreasing from 0.085 at 5% to 0.035 at 50%, indicating remarkable consistency. Under consecutive year gaps, KGE ranges from 0.78 (1-year gap) to 0.82 (5-year gap), with the counterintuitive stability at longer gaps attributed to the ARCHI donor regression providing reliable trend information from correlated wells.

The PCHIP small-gap fill stage is critical: replacing it with LNN-based small-gap filling degrades mean KGE by approximately 10% for random missing data and 19% for consecutive gaps, confirming that PCHIP's deterministic, shape-preserving interpolation is superior to learned approaches for short, well-constrained gaps. The improvement cascades through the pipeline because PCHIP densification provides better donor correlation estimates for the subsequent MC stage. The framework successfully imputes all 592 GSLB wells to temporal completeness (288 months each, zero remaining gaps), producing the spatially continuous dataset required for Paper 3.

---

## 4. Objective 3 (Paper 3)

### 4.1 Objective

To develop a spatiotemporal interpolation framework that produces continuous groundwater-level fields from the imputed well records of Paper 2, apply those fields to derive spatially distributed GRACE leakage correction factors for the GSLB, and benchmark the resulting groundwater storage estimates against independently derived mascon-based estimates.

### 4.2 Methods

The interpolation framework consists of three stages applied to the complete imputed dataset from Paper 2 (592 wells, 288 monthly time steps):

**Stage 1: Spatial Trend Surface.** A degree-2 polynomial trend surface is fitted via ridge regression to the temporal mean WTE of all 592 wells as a function of latitude, longitude, and ground surface elevation:

WTE_mean = b0 + b1*lat + b2*lon + b3*elev + b4*lat^2 + b5*lon^2 + b6*elev^2 + b7*lat*lon + b8*lat*elev + b9*lon*elev

This polynomial captures the large-scale spatial gradient in WTE driven by topography and regional geology (R-squared = 0.957 on the GSLB dataset, reducing the 3000-ft WTE range to residuals of approximately +/-30 ft). Grid cell elevations are obtained from the Copernicus DEM via the Open-Meteo Elevation API.

**Stage 2: EOF Decomposition.** The detrended residual matrix (288 months x 592 wells) is decomposed via Singular Value Decomposition:

Residuals = U * S * V^T

where U (288 x k) contains temporal modes representing shared hydrological patterns (long-term trends, seasonal cycles, multi-year drought signals), S (k) contains singular values indicating mode importance, and V^T (k x 592) contains spatial loadings quantifying how strongly each well follows each temporal mode. The number of retained modes k is set to 20, capturing approximately 95% of residual variance.

**Stage 3: IDW Loading Interpolation.** For each grid cell, the k spatial loadings are estimated from nearby wells via Inverse Distance Weighting (exponent=2, 30 nearest neighbors). The grid cell's reconstructed WTE is then:

WTE(grid, t) = trend(lat, lon, elev) + sum_m=1..k [ U(t,m) * S(m) * V_interp(m) ]

This approach interpolates k small spatial scalars (loadings) rather than 288 raw time values, guaranteeing temporal coherence in the output because all timesteps share the same modes.

**Leakage Correction.** The interpolated groundwater-level fields are converted to volumetric storage anomalies using spatially varying specific yield estimates and aggregated to the 0.5-degree JPL TWSa grid. For each grid cell with sufficient in situ coverage, a leakage factor Lf(phi, lambda) is calibrated as the multiplicative scalar minimizing mismatch between the GRACE-derived GWSa partition and the in situ-derived GWSa over 2002-2024. For grid cells lacking sufficient well coverage, Lf is propagated via a covariate-aware model using land cover, irrigation fraction, elevation, depth to groundwater, and aquifer-type indicators as predictors.

**Benchmarking.** The resulting GWSa is benchmarked against an independently derived mascon-based GWSa obtained by applying the same partition equation to JPL mascon TWSa (Watkins et al., 2015), which mitigates leakage via mass-concentration basis functions and does not require an explicit post-hoc leakage correction. Because the auxiliary terms (SWEa, SMa, CANa, SWSa) are identical on both sides, any systematic divergence is attributable specifically to leakage handling.

### 4.3 Anticipated Results

Leave-one-out cross-validation on 30 wells demonstrates that the EOF interpolation framework achieves RMSE of 32.3 ft, representing a 49% reduction compared to plain IDW (RMSE 62.9 ft) and a 73% reduction compared to per-timestep ordinary kriging (RMSE 121.9 ft). The improvement is attributable to two factors: the trend surface absorbs the dominant spatial gradient (R-squared = 0.957), and EOF decomposition ensures temporal coherence by interpolating spatial loadings rather than raw values.

The pixel-wise Lf grid is anticipated to depart substantially from the basin-uniform value of 2 used in Paper 1, with the largest factors concentrated along the Wasatch Front where pumping-driven mass loss is most concentrated, and substantially smaller factors (closer to unity) in the West Desert and Bear River sub-basins where anthropogenic stress is diffuse. Agreement between the gridded-correction GWSa and the in situ reconstruction is expected to improve over the basin-uniform case at sub-basin scales, where the uniform scalar systematically over- or under-corrects different regions. Comparison with the mascon-based GWSa is expected to show closer agreement during drought intervals, when spatial concentration of mass loss is most pronounced. Residual divergence between the two estimates will be informative about the limits of each leakage-handling strategy: the empirical approach excels where in situ density is sufficient; the structural mascon approach excels where regional smoothing across hydrogeologically homogeneous areas is appropriate.

---

## 5. Timeline

| Period | Activity |
|---|---|
| May 2026 (current) | Paper 1 published; Paper 2 method development and validation complete |
| Jun-Aug 2026 | Paper 2 manuscript drafting and internal review |
| Sep 2026 | Paper 2 submission |
| Sep-Dec 2026 | Paper 3 methods: gridded Lf calibration, covariate propagation model, sensitivity analyses |
| Jan-Feb 2027 | Paper 3 results compilation, mascon benchmarking |
| Mar 2027 | Paper 3 manuscript drafting |
| Apr 2027 | Paper 3 submission; dissertation compilation |
| May-Jun 2027 | Dissertation defense and graduation |

---

## References

Abbott, B. W., et al. (2023). Emergency measures needed to rescue Great Salt Lake from ongoing collapse. Brigham Young University. https://pws.byu.edu/great-salt-lake

Ahmadi, A., et al. (2024). Integrating an interpolation technique and AI models using Bayesian model averaging to enhance groundwater level monitoring. Earth Science Informatics, 17, 4963-4984.

Bigalke, S., Loikith, P. C., & Siler, N. (2025). Explaining the 2022 Record Low Great Salt Lake Volume. Geophysical Research Letters, 52, e2024GL112154.

Candes, E. J. & Recht, B. (2009). Exact matrix completion via convex optimization. Foundations of Computational Mathematics, 9(6), 717-772.

Chen, J., et al. (2021). High-Resolution GRACE Monthly Spherical Harmonic Solutions. Journal of Geophysical Research: Solid Earth, 126, e2019JB018892.

Croteau, M. J., Nerem, R. S., Loomis, B. D., & Mitrovica, J. X. (2021). GRACE Fast Mascons From Spherical Harmonics and a Regularization Design Trade Study. Journal of Geophysical Research: Solid Earth, 126, e2021JB022113.

Evans, S., Williams, G. P., Jones, N. L., Ames, D. P., & Nelson, E. J. (2020a). Exploiting Earth Observation Data to Impute Groundwater Level Measurements with an Extreme Learning Machine. Remote Sensing, 12, 2044. https://doi.org/10.3390/rs12122044

Evans, S. W., Jones, N. L., Williams, G. P., Ames, D. P., & Nelson, E. J. (2020b). Groundwater Level Mapping Tool: An open source web application for assessing groundwater sustainability. Environmental Modelling & Software, 131, 104782. https://doi.org/10.1016/j.envsoft.2020.104782

Gharehbaghi, A., Ghasemlounia, R., Ahmadi, F., & Albaji, M. (2022). Groundwater level prediction with meteorologically sensitive Gated Recurrent Unit (GRU) neural networks. Journal of Hydrology, 612, 128262.

Hall, D. K., et al. (2024). Snowfall Replenishes Groundwater Loss in the Great Basin of the Western United States, but Cannot Compensate for Increasing Aridification. Geophysical Research Letters, 51, e2023GL107913.

Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R. (2021). Liquid Time-constant Networks. Proceedings of the AAAI Conference on Artificial Intelligence, 35(9), 7657-7666.

Hasani, R., Lechner, M., Amini, A., Liebenwein, L., Ray, A., Tschaikowski, M., Teschl, G., & Rus, D. (2022). Closed-form continuous-time neural networks. Nature Machine Intelligence, 4, 992-1003.

Huang, G.-B., Zhu, Q.-Y., & Siew, C.-K. (2006). Extreme learning machine: Theory and applications. Neurocomputing, 70(1-3), 489-501.

Jasechko, S., et al. (2024). Rapid groundwater decline and some cases of recovery in aquifers globally. Nature, 625(7996), 715-720.

Jeong, J., Park, E., Chen, H., Kim, K.-Y., Han, W. S., & Suk, H. (2020). Estimation of groundwater level based on the robust training of recurrent neural networks using corrupted data. Journal of Hydrology, 582, 124512.

Li, B., et al. (2024). A New GRACE Downscaling Approach for Deriving High-Resolution Groundwater Storage Changes Using Ground-Based Scaling Factors. Water Resources Research, 60, e2023WR035210.

Li, Y., et al. (2025). Predicting regional-scale groundwater levels at high spatial resolution using spatial Random Forest models. International Journal of Applied Earth Observation and Geoinformation.

Lin, H., Gharehbaghi, A., Zhang, Q., Band, S. S., Pai, H. T., Chau, K.-W., & Mosavi, A. (2022). Time series-based groundwater level forecasting using gated recurrent unit deep neural networks. Engineering Applications of Computational Fluid Mechanics, 16(1), 1655-1672.

Long, D., et al. (2014). Drought and flood monitoring for a large karst plateau in Southwest China using extended GRACE data. Remote Sensing of Environment, 155, 145-160.

Ma, G., et al. (2024). Improved Estimates of Sub-Regional Groundwater Storage Anomaly Using Coordinated Forward Modeling. Water Resources Research, 60(7), e2023WR036105.

Null, S. E. & Wurtsbaugh, W. A. (2020). Water Development, Consumptive Water Uses, and Great Salt Lake. In Baxter, B. K. & Butler, J. K. (Eds.), Great Salt Lake Biology, Springer, pp. 1-30.

Poudevigne, T. & Jones, O. (2024). Time-series imputation using low-rank matrix completion. arXiv:2408.02594.

Ramirez, S. G., Williams, G. P., & Jones, N. L. (2022). Groundwater Level Data Imputation Using Machine Learning and Remote Earth Observations Using Inductive Bias. Remote Sensing, 14, 5509. https://doi.org/10.3390/rs14215509

Ramirez, S. G., Williams, G. P., Jones, N. L., Ames, D. P., & Radebaugh, J. (2023). Improving Groundwater Imputation through Iterative Refinement Using Spatial and Temporal Correlations from In Situ Data with Machine Learning. Water, 15, 1236. https://doi.org/10.3390/w15061236

Rateb, A. & Herring, T. A. (2020). Comparison of Groundwater Storage Changes From GRACE Satellites With Monitoring and Modeling of Major U.S. Aquifers. Water Resources Research, 56(12), e2020WR027556.

Rodell, M., et al. (2004). The global land data assimilation system. Bulletin of the American Meteorological Society, 85(3), 381-394.

Rojas, R., et al. (2025). Bridging gaps in sparse groundwater data: classical, ensemble, and deep learning approaches for single- and multi-well imputation. Frontiers in Water, 7, 1726853.

Scanlon, B. R., et al. (2023). Global water resources and the role of groundwater in a resilient water future. Nature Reviews Earth & Environment, 4, 87-101.

Senanayake, S., Pradhan, B., Huber, A., & Alamri, A. (2024). Deep learning framework with Bayesian data imputation for modelling and forecasting groundwater levels. Environmental Modelling & Software, 178, 106072.

Shepard, D., Jones, N. L., & Williams, G. P. (2025). Application of the Groundwater Data Mapper Tool to Assess Storage Changes in a Groundwater-Driven Basin in the Klamath Watershed, Oregon, USA. Hydrology, 12(6), 140. https://doi.org/10.3390/hydrology12060140

Stevens, M. D., Ramirez, S. G., Martin, E.-M. H., Jones, N. L., Williams, G. P., Adams, K. H., Ames, D. P., & Pulla, S. T. (2025). Groundwater Storage Loss in the Central Valley Analysis Using a Novel Method based on In Situ Data Compared to GRACE-Derived Data. Environmental Modelling & Software, 186, 106368. https://doi.org/10.1016/j.envsoft.2025.106368

Sun, Z., et al. (2025). Noisy matrix completion for longitudinal data with subject- and time-specific covariates. Canadian Journal of Statistics.

Tapley, B. D., Bettadpur, S., Ries, J. C., Thompson, P. F., & Watkins, M. M. (2004). GRACE measurements of mass variability in the Earth system. Science, 305(5683), 503-505.

Tripathi, V., Groh, A., Horwath, M., & Ramsankaran, R. (2022). Scaling methods of leakage correction in GRACE mass change estimates revisited for the complex hydro-climatic setting of the Indus Basin. Hydrology and Earth System Sciences, 26, 4515-4535.

Vishwakarma, B. D., Devaraju, B., & Sneeuw, N. (2018). What is the spatial resolution of GRACE satellite products for hydrology? Remote Sensing, 10(6), 852.

Watkins, M. M., Wiese, D. N., Yuan, D.-N., Boening, C., & Landerer, F. W. (2015). Improved methods for observing Earth's time variable mass distribution with GRACE using spherical cap mascons. Journal of Geophysical Research: Solid Earth, 120(4), 2648-2671.

Wine, M. L. (2019). Climatization -- Negligent Attribution of Great Salt Lake Desiccation: A Comment on Meng (2019). Climate, 7(5), 67.

Wu, H., et al. (2025). Forecasting Groundwater Level by Characterizing Multiple Spatial Dependencies of Environmental Factors Using Graph-Based Deep Learning. Journal of Geophysical Research: Machine Learning and Computation.

Wunsch, A., Liesch, T., & Broda, S. (2021). Groundwater level forecasting with artificial neural networks: a comparison of LSTM, CNN, and NARX. Hydrology and Earth System Sciences, 25, 1671-1687.

Young, Z., Kreemer, C., & Blewitt, G. (2021). GPS Constraints on Drought-Induced Groundwater Loss Around Great Salt Lake, Utah, With Implications for Seismicity Modulation. Journal of Geophysical Research: Solid Earth, 126, e2021JB022020.

Zamora, H. & Inkenbrandt, P. (2024). Estimate of groundwater flow and salinity contribution to the Great Salt Lake using groundwater levels and spatial analysis. Geosites, 51, 1-24.

Zowam, F. J. & Milewski, A. M. (2024). Groundwater Level Prediction Using Machine Learning and Geostatistical Interpolation Models. Water, 16(19), 2771.
