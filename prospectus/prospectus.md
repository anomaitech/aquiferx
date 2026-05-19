# PhD Prospectus

# Advancing Basin-Scale Groundwater Storage Estimation Through Hybrid Spatiotemporal Imputation, Empirical Orthogonal Function (EOF)-Based Interpolation, and Satellite-Derived Leakage Correction

**Henok Teklu**

Department of Civil and Construction Engineering, Brigham Young University

**Supervisor:** Professor Norm Jones

**Committee:** Professor Gustavious Williams, Professor Jim Nelson, Professor Dan Ames

---

## 1. Introduction

### 1.1 Overall Objective and Significance

Groundwater is the world's most extracted raw material, supplying approximately 50% of global irrigation water, 40% of industrial water, and 36% of drinking water (Jasechko et al., 2024). Approximately two billion people depend on groundwater as their primary water source, and in arid and semi-arid regions -- where surface reservoirs are increasingly stressed by warming-driven evaporation and growing demand -- groundwater is often the sole reliable supply. Despite its critical importance, groundwater remains the most poorly monitored and least understood component of the terrestrial water budget.

The consequences of inadequate groundwater monitoring are severe and accelerating. Jasechko et al. (2024) analyzed in situ groundwater-level trends from approximately 170,000 monitoring wells spanning 1,693 aquifer systems across 40 countries on six continents -- the most comprehensive global assessment to date. Their analysis revealed that rapid groundwater-level declines exceeding 0.5 meters per year are widespread, particularly in irrigated regions of South Asia, the Middle East, North Africa, and the western United States. More alarmingly, 30% of regional aquifer systems showed accelerating rates of decline over the past four decades, indicating that groundwater stress is intensifying rather than stabilizing. The resulting impacts -- land subsidence, saline intrusion, baseflow depletion, ecosystem degradation, and wells running dry -- pose existential threats to food security and economic stability in affected regions.

Reliable basin-scale quantification of groundwater storage change is therefore essential for water-resources planning, drought-response assessment, and long-term sustainability policy. Yet three fundamental methodological barriers prevent accurate storage estimation at the scales relevant to water management, and these barriers are interconnected in a way that demands an integrated solution.

### 1.2 The Three Barriers

#### 1.2.1 Barrier 1: Temporal Gaps in In Situ Well Records

The most direct evidence of groundwater storage change comes from monitoring well networks, where changes in water-table elevation can be converted to volumetric storage change via the specific yield. Public databases such as the United States Geological Survey (USGS) National Water Information System (NWIS) contain millions of individual water-level measurements spanning decades. However, these records are profoundly incomplete. Wells are installed by different agencies for different purposes, monitored intermittently over their lifetimes, and frequently abandoned when funding lapses or infrastructure decays. The result is a patchwork of observations: some wells have dense daily measurements for a few years followed by decade-long silences; others have quarterly measurements scattered irregularly across 30 years; many have fewer than a handful of observations total.

The World Meteorological Organization's State of Global Water Resources Report documented that only 47 countries had usable in situ groundwater monitoring data as of 2024, with a total of just 37,406 reporting wells worldwide -- a number that is strikingly small relative to the scale of the resource being monitored. In developing regions of Sub-Saharan Africa, South Asia, and South America, monitoring networks are particularly sparse, with many major aquifer systems having no systematic long-term observations at all.

For the wells that do exist, temporal gaps are the norm rather than the exception. Multi-year gaps are commonplace in public databases, and these gaps are not randomly distributed: they tend to cluster during periods of reduced funding, institutional transitions, or restricted field access -- precisely the periods when groundwater stress may be changing most rapidly. Standard interpolation methods (linear interpolation, cubic splines) can fill gaps of a few months where bounding observations constrain the estimate, but they fail catastrophically for gaps exceeding one to two years, where the trajectory of the water table between the last pre-gap observation and the first post-gap observation is fundamentally unconstrained.

This temporal incompleteness directly limits every downstream use of the data: trend analysis requires continuous records; volumetric storage estimation requires spatially complete snapshots at each timestep; and satellite calibration requires temporally dense comparison series. Without addressing the gap problem, the vast majority of the world's groundwater monitoring data is unusable for basin-scale storage estimation.

#### 1.2.2 Barrier 2: Spatial Interpolation from Point Data to Continuous Fields

Even if every well record were temporally complete, the wells themselves are irregularly distributed points in space. Converting these point observations into the continuous spatial fields needed for volumetric storage estimation requires spatial interpolation -- predicting water-table elevation at locations where no well exists.

Geostatistical methods, particularly ordinary kriging, have been the dominant spatial interpolation approach in hydrogeology for decades. Kriging provides optimal unbiased predictions under the assumption of second-order stationarity -- that the spatial covariance structure is constant across the domain. However, this assumption is routinely violated in heterogeneous basins where water-table elevation varies by hundreds to thousands of meters over distances of tens of kilometers, driven by topography, geology, and anthropogenic influences. In the Great Salt Lake Basin, for example, water-table elevation ranges from approximately 4,200 to 7,200 feet across the study area -- a 3,000-foot gradient that renders any single variogram model inadequate.

Furthermore, conventional spatial interpolation treats each timestep independently. If there are 288 monthly timesteps over a 24-year record, kriging solves 288 independent spatial interpolation problems, one per month. This approach ignores the temporal correlations between months -- the fact that a well that was anomalously high in January is likely still anomalously high in February -- and can produce spatially smooth but temporally incoherent fields, where adjacent months show unrealistic jumps at individual grid cells.

Methods that simultaneously exploit both the shared temporal patterns across wells and the spatial structure of the monitoring network for interpolation remain largely unexplored in groundwater hydrology. This is the methodological gap that the third paper of this dissertation addresses.

#### 1.2.3 Barrier 3: GRACE Signal Leakage and the Groundwater Partition

Satellite gravimetry from the Gravity Recovery and Climate Experiment (GRACE, 2002-2017) and its successor GRACE Follow-On (GRACE-FO, 2018-present) has revolutionized the study of large-scale water storage change by providing the first global, spatially continuous measurements of total water storage anomalies (TWSa) at monthly temporal resolution (Tapley et al., 2004). The missions measure changes in the Earth's gravity field caused by mass redistribution -- primarily the movement of water between the atmosphere, oceans, ice sheets, surface water, soil moisture, and groundwater.

However, extracting the groundwater component from GRACE requires solving a partition equation:

GWSa = Lf * TWSa - SMa - SWEa - SWSa - CANa

where SMa is the soil moisture anomaly, SWEa is the snow water equivalent anomaly, SWSa is the surface water storage anomaly, CANa is the canopy storage anomaly, and Lf is the leakage correction factor. The auxiliary storage components are typically obtained from land-surface models such as the Global Land Data Assimilation System (GLDAS; Rodell et al., 2004), which inherit their own assumptions about subsurface representation, irrigation withdrawals, and terminal-lake hydrology.

The leakage correction factor Lf addresses a fundamental limitation of GRACE: its native spatial resolution of approximately 300 km (Vishwakarma et al., 2018). At this scale, the mass change signal from one basin "leaks" into adjacent basins, and the signal within the basin is spatially smoothed, attenuating the true amplitude of storage change. The conventional approach applies Lf as a single basin-uniform scalar -- the same multiplier everywhere within the basin. Long et al. (2014) established the forward modeling approach and showed that leakage correction can improve amplitude estimates by 37% and trend estimates by 36%. However, Tripathi et al. (2022) demonstrated that basin-average corrections can be misleading when different parts of the basin experience opposite trends -- a situation common in basins with spatially concentrated pumping. Ma et al. (2024) showed through Coordinated Forward Modeling that sub-regional groundwater storage trends can diverge substantially from basin averages.

The uniform leakage factor is a known oversimplification: it treats a basin where all groundwater loss is concentrated along a narrow urban corridor (e.g., the Wasatch Front in the GSLB) identically to a basin where loss is distributed uniformly. Li et al. (2024) estimated pixel-scale correction factors using in situ data, establishing precedent for spatially distributed correction. However, all prior pixel-scale approaches have been constrained by the incompleteness of the underlying in situ well records -- the very limitation that this dissertation's first two papers address.

### 1.3 The Chain: How the Three Barriers Are Connected

The three barriers form a sequential chain. Temporal gaps in well records (Barrier 1) prevent the construction of spatially complete snapshots for any given month, which in turn prevents reliable spatial interpolation to continuous fields (Barrier 2), which in turn prevents pixel-wise calibration of GRACE leakage factors against spatially complete in situ data (Barrier 3). Breaking this chain requires addressing all three barriers in sequence, with each solution building on the outputs of the previous.

This dissertation addresses all three barriers through three papers:

1. **Paper 1** (submitted to journal) establishes baseline groundwater storage estimates for the Great Salt Lake Basin using multiple independent methods -- GRACE-derived, model-derived, and in situ -- revealing the sensitivity of the results to leakage handling and imputation quality, and motivating the advances of Papers 2 and 3.

2. **Paper 2** (in development) develops a novel hybrid imputation framework coupling Matrix Completion (MC) with Liquid Neural Networks (LNN) that jointly exploits spatial cross-well correlations and continuous-time temporal dynamics to reconstruct groundwater records with multi-year gaps -- a coupling that remains largely unexplored in groundwater imputation literature.

3. **Paper 3** applies the temporally complete records from Paper 2 to produce spatially continuous groundwater-level fields via a spatial trend decomposition and Empirical Orthogonal Function (EOF) analysis framework, and uses those fields to derive pixel-wise GRACE leakage correction factors calibrated against spatially complete in situ data.

### 1.4 Study Area and Planned Validation

The Great Salt Lake Basin (GSLB), a closed hydrologic system covering approximately 93,000 km-squared in the Intermountain West of the United States, serves as the initial validation site. The GSLB was selected for several reasons: its dense USGS monitoring network provides 592 eligible wells with observations spanning the 2000-2023 study period; its heterogeneous hydrogeology, ranging from alluvial valley-fill aquifers along the Wasatch Front to fractured volcanic aquifers in the West Desert, provides a challenging testbed for interpolation methods; and its concentrated anthropogenic pumping along the Wasatch Front creates the kind of spatially heterogeneous storage change signal that motivates pixel-wise leakage correction.

However, the methods developed in this dissertation are designed to be general-purpose, not basin-specific. The imputation framework requires only well locations, irregular observations, and globally available GLDAS auxiliary data, with all hyperparameters automatically optimized per well. No basin-specific training or calibration is required. Validation on additional basins across diverse hydrogeological and climatic settings -- including sites in Sub-Saharan Africa and South America where monitoring networks are sparser and the need for imputation is most acute -- is planned to demonstrate the framework's global transferability.

### 1.5 Specific Objectives

**Trends in Groundwater Storage in the Great Salt Lake Basin, 2002-2024 (Paper 1).** Quantify multi-decadal groundwater storage change in the GSLB by integrating GRACE-derived, GLDAS-derived, and in situ estimates within a unified framework. Identify the methodological sensitivity to surface-water inclusion and GRACE leakage correction that motivates the spatially distributed approach of Paper 3. *(Submitted to journal.)*

**Hybrid Spatiotemporal Imputation via Matrix Completion and Liquid Neural Networks (Paper 2).** Develop and validate a general-purpose hybrid imputation framework for sparse, irregular groundwater-level time series that couples PCHIP for short gaps with Matrix Completion + Liquid Neural Networks (MC+LNN) for long gaps, using globally available auxiliary climatic forcings as continuous-time inputs.

**EOF-Based Spatial Interpolation and Pixel-Wise Leakage Correction (Paper 3).** Apply the imputation framework from Paper 2 to produce spatially continuous groundwater-level fields via spatial trend decomposition and Empirical Orthogonal Function analysis, and use those fields to derive a pixel-wise GRACE leakage correction grid calibrated against spatially complete in situ data.

---

## 2. Trends in Groundwater Storage in the Great Salt Lake Basin, 2002-2024 (Paper 1)

### 2.1 Objective

To quantify groundwater storage change in the Great Salt Lake Basin from 2002 through 2024 using multiple independent estimation methods, evaluate methodological sensitivities related to surface-water inclusion and GRACE leakage correction, and establish the baseline that motivates the imputation and leakage-correction advances of Papers 2 and 3.

### 2.2 Background

#### 2.2.1 The Great Salt Lake Basin

The GSLB is a terminal, closed-basin hydrologic system spanning portions of Utah, Idaho, Wyoming, and Nevada. The Great Salt Lake itself is the largest saline lake in the Western Hemisphere and a critical ecological resource supporting brine shrimp, migratory birds, and a mineral extraction industry. The basin's water budget is dominated by snowmelt-driven streamflow from the Wasatch Range, with groundwater serving as both a storage buffer and a direct contributor to lake inflows.

Consumptive water uses have depleted inflows to the Great Salt Lake by approximately 39%, lowering the lake by 3.4 meters and reducing its volume by an estimated 64% (Null and Wurtsbaugh, 2020). Wine (2019) argued that consumptive use, not climate change, is the dominant driver of lake decline, while Bigalke et al. (2025) attributed the 2022 record-low lake volume to a combination of reduced streamflow (approximately two-thirds) and increased evaporation from warming (approximately one-third). Hall et al. (2024) used GRACE/GRACE-FO data to document 68.7 km-cubed of groundwater loss from 2002-2023 across the broader Great Basin, finding that even record snow years fail to reverse the long-term downward trend due to warming-driven evaporation and continued diversions. Zamora and Inkenbrandt (2024) revised the groundwater contribution to the Great Salt Lake substantially upward, from the historical estimate of 3% to approximately 10% of total inflows -- a finding that fundamentally changes the lake's water budget and underscores the importance of accurate groundwater accounting.

#### 2.2.2 GRACE/GRACE-FO Satellite Gravimetry

The GRACE twin-satellite mission (2002-2017) and its successor GRACE-FO (2018-present) measure changes in the Earth's gravity field by precisely tracking the distance between two co-orbiting satellites approximately 220 km apart using K-Band microwave ranging (Tapley et al., 2004). GRACE-FO additionally carries a Laser Ranging Interferometer for enhanced precision. Monthly gravity field solutions are processed into spherical harmonic coefficients (e.g., by JPL, CSR, or GFZ) or into mass concentration (mascon) solutions (Watkins et al., 2015) that can be interpreted as total water storage anomalies on a gridded basis.

The native spatial resolution of GRACE products is approximately 300 km (3 degrees), determined by the maximum recoverable spherical harmonic degree and the truncation/filtering applied to suppress noise at short wavelengths. Vishwakarma et al. (2018) characterized the effective resolution and its implications for hydrology, showing that signal leakage between adjacent basins is a fundamental limitation that must be corrected before basin-scale groundwater storage can be isolated.

Rateb and Herring (2020) compared GRACE-derived groundwater storage changes with approximately 23,000 monitoring wells across 14 major US aquifer systems and found correlations ranging from R=0.52 to R=0.95, providing confidence in GRACE's ability to detect groundwater trends at the basin scale but also highlighting the importance of the leakage correction approach used.

#### 2.2.3 The GWDM Imputation Lineage

The in situ groundwater storage estimates in Paper 1 are produced using the Groundwater Data Mapper (GWDM) workflow, which represents a progression of imputation methods developed in the BYU Hydroinformatics group. Evans et al. (2020a) introduced the approach of coupling Extreme Learning Machines (ELM; Huang et al., 2006) with Earth observation data from GRACE and GLDAS as auxiliary inputs for gap-filling, demonstrating that satellite-derived variables can substantially improve imputation in data-sparse regions. Evans et al. (2020b) packaged this approach as an open-source web application for assessing groundwater sustainability.

Ramirez et al. (2022) extended the framework by incorporating inductive bias from remote sensing into the machine learning imputation, demonstrating that GRACE-derived total water storage anomalies and GLDAS soil moisture products improve predictive performance when used as auxiliary features. Ramirez et al. (2023) further advanced the methodology through iterative refinement that exploits both spatial correlations from neighboring wells and temporal correlations from auxiliary variables, showing that sequential incorporation of in situ spatial context improves accuracy over single-pass approaches. This line of work has been applied to basin-scale storage assessment in California's Central Valley (Stevens et al., 2025) and the Klamath watershed in Oregon (Shepard et al., 2025).

### 2.3 Methods

Five independent GWSa estimates were computed over the 2002-2024 study period:

1. **GRACE-raw**: JPL GRACE TWSa minus GLDAS v2.1 soil moisture, snow water equivalent, and canopy storage anomalies.
2. **GRACE-sw**: Identical to GRACE-raw but with surface-water storage from 19 reservoirs plus the Great Salt Lake subtracted explicitly, removing the substantial surface-water contribution to total storage variability.
3. **GRACE-Lf**: The surface-water-adjusted estimate with an empirical leakage factor Lf=2 applied multiplicatively to TWSa, with Lf calibrated to maximize agreement with the in situ estimate.
4. **GLDAS-2.2**: The GRACE-assimilated Catchment Land Surface Model (CLSM) groundwater product, which provides an independent model-derived estimate.
5. **GWDM**: An in situ estimate built from approximately 1,200 USGS wells, using PCHIP for short temporal gaps, ELM with Earth observation inputs for longer discontinuities, ordinary kriging for spatial interpolation, and a basin-representative specific yield of 0.15 to convert water-level change to volumetric storage change.

The five estimates were compared via Pearson correlation and coefficient of determination, and directly benchmarked against the independent GPS-based estimate of Young, Kreemer, and Blewitt (2021), which infers subsurface mass loss from observed ground deformation.

### 2.4 Results

All four independent methods identified two major drawdown intervals (2012-2016 and 2019-2022) with only partial recovery in between, consistent with the prolonged drought conditions affecting the Intermountain West during this period. The leakage-corrected GRACE estimate (GRACE-Lf) yielded a 2011-2016 drought-period loss of approximately 10.1 km-cubed, consistent with the independent GPS-based estimate of 10.9 +/- 2.8 km-cubed reported by Young, Kreemer, and Blewitt (2021).

Including surface-water storage in the GRACE partition substantially altered the derived GWSa, with approximately 31% of basin total storage change attributable to surface-water variability -- primarily fluctuations in the Great Salt Lake. This finding highlights the importance of explicitly accounting for surface water when estimating groundwater storage in terminal-lake basins.

The basin-uniform leakage factor improved the Pearson correlation between GRACE-derived and in situ GWSa from 0.17 (uncorrected) to 0.77 (Lf=2), demonstrating the critical importance of leakage handling. However, this uniform treatment masks known spatial heterogeneity in pumping and recharge within the basin -- pumping is concentrated along the narrow Wasatch Front urban corridor while recharge is distributed across the mountain fronts -- motivating the spatially distributed (pixel-wise) leakage correction approach developed in Paper 3.

Annual precipitation correlated most strongly with in situ GWSa at a two-year lag and with three-year cumulative rainfall (r=0.67), consistent with multi-year recharge memory in the basin's thick vadose zones.

The in situ estimate itself was limited by two factors: well-record gaps that prevented continuous temporal coverage at many wells, and reliance on a single imputation method (ELM) whose performance degrades for gaps exceeding approximately two years. These limitations directly motivate the improved hybrid imputation framework developed in Paper 2.

---

## 3. Hybrid Spatiotemporal Imputation via Matrix Completion and Liquid Neural Networks (Paper 2)

### 3.1 Objective

To develop, validate, and benchmark a general-purpose hybrid imputation framework that jointly exploits spatial cross-well correlations (via matrix completion) and continuous-time temporal dynamics (via liquid neural networks) for reconstructing groundwater-level records with multi-year gaps.

### 3.2 Background

#### 3.2.1 Existing Temporal Imputation Approaches

The ELM-based approaches described in Paper 1, while effective for gaps of up to approximately two years, operate primarily in the temporal domain: each well is imputed independently using auxiliary time series (GRACE TWSa, GLDAS soil moisture) as features, with spatial information entering only through basin-average satellite-derived covariates rather than the actual cross-well correlation structure of the monitoring network. Each well's imputation is thus an isolated regression problem, blind to the fact that neighboring wells may be experiencing the same seasonal cycles, drought responses, and recharge events.

Deep learning architectures face similar limitations. Jeong et al. (2020) demonstrated that Long Short-Term Memory (LSTM) networks can reconstruct missing groundwater levels with high accuracy for wells with spatially correlated neighbors, but performance degrades sharply when gaps exceed one to two years and when neighboring wells are themselves sparse. Wunsch et al. (2021) systematically compared LSTM, convolutional neural networks (CNN), and nonlinear autoregressive networks with exogenous inputs (NARX) for groundwater-level forecasting, finding that multivariate temporal architectures consistently outperform univariate models but remain fundamentally limited by the availability of training data during gap periods -- when the gap itself eliminates the training signal that the network relies upon. Gharehbaghi et al. (2022) showed that Gated Recurrent Unit (GRU) networks perform comparably to LSTM with lower computational cost, and Lin et al. (2022) achieved R-squared of 0.86 with a double-GRU architecture for monthly groundwater predictions.

A common limitation across all these approaches -- ELM, LSTM, GRU, and CNN alike -- is that they treat each well's temporal imputation as essentially independent. Spatial information enters only indirectly through satellite-derived covariates that provide basin-average (not well-specific) spatial context. The cross-well correlation structure -- the fact that nearby wells in the same aquifer respond to the same recharge events, pumping stresses, and climate forcings -- is not exploited.

#### 3.2.2 Regional Correlation Approaches

Regional correlation-based approaches represent an intermediate strategy that begins to exploit spatial structure. Levy et al. (2025) developed ARCHI (Automated Regional Correlation Analysis for Hydrologic Record Imputation), a USGS R package that imputes missing data in "target" records by linear regression using more complete "reference" records as predictors. ARCHI's iterative algorithm allows each site to serve as both target and reference, progressively growing the pool of complete records until viable gap-filling ceases. This approach effectively leverages the correlation between nearby wells -- if Well A and Well B are strongly correlated, the observed values at Well A during Well B's gap can predict what Well B would have measured.

However, ARCHI operates within a purely linear regression framework and does not incorporate auxiliary climatic forcings (soil moisture, precipitation, temperature) or nonlinear temporal dynamics. It also processes well pairs independently rather than simultaneously exploiting the structure across the entire monitoring network.

#### 3.2.3 Matrix Completion

Matrix completion offers a fundamentally different perspective by treating the imputation problem not as a collection of independent per-well regressions, but as a single matrix-level optimization. All well records are arranged as columns in a partially observed matrix (wells x time), and the low-rank structure of the matrix is exploited to infer missing entries simultaneously from the cross-well correlations inherent in the data.

The theoretical foundations were established by Candes and Recht (2009), who proved that low-rank matrices can be recovered exactly from a surprisingly small number of observed entries under mild incoherence conditions -- a result with implications across recommender systems, signal processing, and scientific data recovery. The SoftImpute algorithm of Mazumder, Hastie, and Tibshirani (2010) provides an efficient iterative implementation: at each step, the current matrix is decomposed via Singular Value Decomposition (SVD), the singular values are soft-thresholded to enforce low-rank structure, the matrix is reconstructed, and observed entries are re-inserted. This cycle repeats until convergence, yielding predictions for all missing entries that are consistent with the low-rank structure shared across all wells.

The application of matrix completion to groundwater data remains extremely limited. Sharma, Kim, and Tayerani Charmchi (2024) evaluated SoftImpute alongside four other methods (KNN, MICE, MLP, Random Forest) for monthly groundwater levels in the Chao-Phraya River Basin in Thailand, finding that SoftImpute excels in sparse networks with Pearson R above 0.80 -- precisely the setting where deep learning methods, which need dense training data, struggle most. However, their study treats SoftImpute as one of several standalone methods, without coupling it to temporal models or incorporating auxiliary climatic forcings.

#### 3.2.4 Liquid Neural Networks and Continuous-Time Dynamics

On the temporal modeling side, a recent class of neural network architectures offers particular promise for the irregular sampling cadence of groundwater monitoring data. Conventional recurrent networks (LSTM, GRU) operate in discrete time: they update their hidden state at fixed time steps, requiring regular sampling intervals or artificial resampling of irregularly sampled data. For groundwater wells measured at intervals ranging from days to years, this mismatch is problematic.

Reservoir computing (Jaeger, 2001) provides an alternative: a randomly initialized recurrent network (the "reservoir") transforms the input signal into a high-dimensional representation, and only the output layer (the "readout") is trained. This approach is computationally efficient and surprisingly effective for time-series tasks, but it still operates in discrete time.

Hasani et al. (2021) introduced Liquid Time-Constant Networks (LTCs), in which the hidden state evolves according to an ordinary differential equation (ODE) with time-varying coefficients. Time enters as a structural property of the dynamical model rather than as an engineered input feature, allowing the network to naturally accommodate irregular sampling without resampling. The subsequent Closed-form Continuous-depth (CfC) architecture (Hasani et al., 2022) achieves one order of magnitude faster training and inference than LTCs by eliminating the need for numerical ODE solvers, instead providing an exact analytical solution:

x(t+dt) = x * exp(-lambda*dt) + (b/lambda) * (1 - exp(-lambda*dt))

where x is the reservoir state, lambda is the leak rate controlling memory decay, and b = tanh(W_in * input + W_res * x) is the nonlinear activation driven by the current input and recurrent connections. This closed-form update means that arbitrarily large or small time steps incur no additional computational cost and no discretization error -- a property uniquely suited to groundwater monitoring data where the interval between successive measurements varies from days to years.

#### 3.2.5 The Gap This Dissertation Addresses

To the authors' knowledge, matrix completion has not previously been coupled with continuous-time neural networks for groundwater imputation. The ELM-based approaches (Evans et al., 2020a; Ramirez et al., 2023) incorporate temporal auxiliary features but treat spatial context only through basin-scale remote sensing products. ARCHI (Levy et al., 2025) exploits regional cross-well correlation through iterative donor regression but operates within a linear framework without auxiliary forcings or nonlinear temporal modeling. The sole published application of matrix completion to groundwater (Sharma et al., 2024) uses SoftImpute as a standalone method without temporal coupling or auxiliary data integration.

The MC+LNN framework proposed in this dissertation addresses this gap by coupling two complementary components: matrix completion provides spatially informed initial estimates by exploiting the correlation structure across the full well network -- constructing a composite matrix of the target well, correlated donor wells, GLDAS auxiliary variables, and seasonal encoding -- while the Liquid Neural Network refines those estimates using closed-form continuous-time dynamics conditioned on auxiliary climatic forcings. The MC predictions serve as reservoir input (placeholders) for the LNN during gap periods, but the LNN readout is trained exclusively on real observations, ensuring the temporal model learns from ground truth while benefiting from MC's spatial context.

### 3.3 Methods

The framework operates on monthly-aggregated well observations via a two-stage pipeline.

**Stage 1: PCHIP Small-Gap Fill.** Gaps of 24 months or shorter are filled using Piecewise Cubic Hermite Interpolating Polynomials (PCHIP), based on the Fritsch and Carlson (1980) algorithm for monotone piecewise cubic interpolation. PCHIP preserves the local monotonicity and shape of the observed data without introducing the spurious oscillations characteristic of standard cubic splines -- a property particularly important for groundwater records where overshooting can produce physically implausible water-table elevations.

The purpose of Stage 1 is twofold: first, it fills short interruptions where the bounding observations provide sufficient constraint for deterministic interpolation; second, and critically, it densifies the monitoring network so that the subsequent matrix completion stage has more overlapping observations between wells for computing donor correlations. Cross-validation confirms that PCHIP densification improves the overall pipeline performance by 10-19% in KGE relative to using neural network-based small-gap filling, because denser observations produce more reliable Pearson correlations for donor selection.

**Stage 2: MC+LNN Large-Gap Fill.** Gaps exceeding 24 months are filled via a coupled Matrix Completion and Liquid Neural Network approach with three phases:

*Phase 2a -- Donor Regression.* For each target well with a large gap, the top 15 most correlated donor wells are identified from the PCHIP-densified network based on Pearson correlation over common observation periods. For each donor, an ordinary least squares (OLS) regression (target = a * donor + b) is fitted on the overlapping observations, and a weighted average of donor predictions (weight = r-squared) provides a trend-aware initialization for the gap period. This follows the donor-correlation concept of ARCHI (Levy et al., 2025) but extends it into the matrix-completion framework described next.

*Phase 2b -- Matrix Completion.* A composite matrix is constructed with rows representing:
- The target well (observed months + NaN at gaps)
- The 15 donor wells (weighted by |Pearson r|)
- 5 GLDAS auxiliary variables (soil moisture at monthly, 1-year, 3-year, 5-year, and 10-year temporal averaging scales)
- 2 seasonal encoding rows (sin and cos at 12-month period)

The matrix is normalized per-row using z-score standardization. The GLDAS and seasonal rows are fully observed at all timesteps, anchoring the SVD decomposition even when multiple wells have simultaneous gaps. SoftImpute (Mazumder et al., 2010) is applied with adaptive rank selection: the algorithm tests ranks k in {3, 5, 8, 10, 12}, selects the rank that minimizes reconstruction error on observed target entries, then runs the final iterative SVD to convergence. A MOVE.1 variance-preserving bias correction ensures that the predictions match the observed mean and standard deviation of the target well.

*Phase 2c -- LNN Temporal Refinement.* A Liquid Neural Network with Closed-form Continuous-time cells (Hasani et al., 2022) refines the MC predictions. The input vector at each timestep concatenates: (1) the observed value (or MC placeholder during gaps); (2) the five GLDAS auxiliary variables; and (3) sin/cos seasonal encoding -- a total of 8 inputs per timestep. The reservoir state evolves via the CfC equation, with the MC predictions serving as reservoir input during gap periods to maintain a physically plausible trajectory.

Critically, the readout weights are trained via ridge regression exclusively on the timesteps where real observations exist. This design ensures that the LNN learns the true input-to-output mapping from ground truth, rather than fitting to potentially erroneous MC predictions. Hyperparameters (reservoir size 10-80 neurons, leak rate 0.05-0.95, input scaling 0.01-0.40) are auto-optimized per well via grid search over 8 random trials, with an ensemble of 3 models (different random seeds) selecting the best by Kling-Gupta Efficiency (Gupta et al., 2009) on the observed data.

**Validation.** The framework is validated on the GSLB well network (592 wells, 288 months, 2000-2023) under two cross-validation scenarios designed to test different aspects of imputation quality:
- *Random missing data*: 5%, 10%, 20%, 30%, 40%, and 50% of observed months are removed at random (50 trials each), testing the pipeline's robustness to progressively sparser data.
- *Consecutive year gaps*: 1, 2, 3, 4, and 5 years of continuous data are removed (20 trials each), testing the pipeline's ability to reconstruct long uninterrupted gaps -- the scenario where conventional methods fail.

Performance metrics include the Kling-Gupta Efficiency (KGE), R-squared, RMSE, and MAE. Comparison baselines include pure PCHIP, the ELM-based approach from Paper 1, and isolated MC and LNN components to quantify the contribution of each pipeline stage. Additional basins across diverse hydrogeological and climatic settings are planned for transferability assessment.

### 3.4 Anticipated Results

Cross-validation on the GSLB demonstrates robust performance across all scenarios:

| Scenario | KGE | R-squared | RMSE (ft) |
|---|---|---|---|
| 5% random missing | 0.837 +/- 0.085 | 0.771 | 2.53 |
| 20% random missing | 0.853 +/- 0.051 | 0.787 | 2.64 |
| 50% random missing | 0.847 +/- 0.035 | 0.788 | 2.74 |
| 1-year consecutive gap | 0.783 +/- 0.116 | 0.703 | 2.98 |
| 3-year consecutive gap | 0.802 +/- 0.076 | 0.730 | 3.02 |
| 5-year consecutive gap | 0.815 +/- 0.063 | 0.744 | 2.91 |

Several features of these results merit emphasis. First, KGE remains above 0.84 across all random-missing rates from 5% to 50%, with the standard deviation actually decreasing at higher missing-data fractions (from 0.085 at 5% to 0.035 at 50%), indicating that performance is remarkably consistent regardless of how much data is removed. Second, KGE remains above 0.78 even for five-year consecutive gaps -- the most challenging imputation scenario, where conventional methods typically degrade to KGE below 0.3. The stability at longer gaps is attributed to the ARCHI donor regression providing reliable trend information from correlated wells even when the target well has no observations for years.

The PCHIP small-gap fill stage is critical to the overall pipeline: replacing it with LNN-based small-gap filling degrades mean KGE by approximately 10% for random missing data and 19% for consecutive gaps. This improvement cascades through the pipeline because PCHIP densification provides better donor correlations for the MC stage.

Cross-validation across wells spanning the full variance spectrum -- from low-variance wells (standard deviation < 0.2 ft) to high-variance wells (standard deviation > 20 ft) -- confirms that the framework performs robustly across diverse well characteristics, with 13 of 15 tested wells achieving KGE above 0.65 regardless of their temporal variability. The two exceptions were wells with either sub-foot resolution (where any small error destroys KGE) or highly erratic pumping-influenced records (where the data itself lacks a predictable pattern).

The framework successfully imputes all 592 GSLB wells to temporal completeness (288 months each, zero remaining gaps), producing the spatially continuous dataset required for Paper 3.

---

## 4. EOF-Based Spatial Interpolation and Pixel-Wise GRACE Leakage Correction (Paper 3)

### 4.1 Objective

To produce spatially continuous groundwater-level fields from the imputed records of Paper 2 via EOF-based interpolation, and to use those fields to derive a pixel-wise GRACE leakage correction grid calibrated against spatially complete in situ data.

### 4.2 Background

#### 4.2.1 Limitations of Conventional Spatial Interpolation

Converting imputed point-well records to continuous spatial fields requires interpolation -- predicting water-table elevation at every grid cell in the basin from the 592 well observations. The conventional approach is to interpolate each timestep independently: for each of the 288 monthly snapshots, apply kriging or Inverse Distance Weighting (IDW) to the well values for that month to produce a spatial grid. This treats every month as a separate spatial interpolation problem.

This per-timestep approach has two fundamental limitations. First, it ignores temporal correlations between months: a well that is anomalously high in March is almost certainly still anomalously high in April, but per-timestep kriging does not exploit this information. The resulting fields may be spatially smooth within each timestep but temporally incoherent -- adjacent months can show unrealistic jumps at individual grid cells due to independent interpolation errors.

Second, kriging assumes spatial stationarity -- that the spatial covariance structure is constant across the domain. In heterogeneous basins where water-table elevation spans hundreds to thousands of meters over distances of tens of kilometers, this assumption is routinely violated (Ahmadi et al., 2024; Li et al., 2025). The variogram estimated from wells across such a gradient does not meaningfully describe the spatial correlation at any particular location.

#### 4.2.2 Empirical Orthogonal Function (EOF) Analysis

EOF analysis offers a fundamentally different strategy. Rather than treating each timestep as an independent spatial problem, EOF simultaneously analyzes the temporal and spatial structure of the entire dataset. SVD applied to the wells-by-time matrix reveals that the temporal variations across hundreds of wells can be explained by a small number of shared patterns (modes): a long-term trend, a seasonal cycle, a multi-year drought response, and so on. Each well's time series is then characterized not by its 288 individual monthly values but by a handful of spatial loadings -- scalar weights indicating how strongly that well follows each shared mode.

Since these loadings vary smoothly across space (nearby wells in the same hydrogeological setting tend to have similar loadings), they can be reliably interpolated to unobserved grid cells using standard IDW. The grid cell's full time series is then reconstructed by combining the interpolated loadings with the shared temporal modes. This approach reduces the interpolation problem from 288 independent spatial problems to k scalar interpolation problems (typically k=20), while guaranteeing temporal coherence because all timesteps share the same modal structure. The dominant modes typically explain more than 95% of the residual variance after spatial trend removal.

#### 4.2.3 GRACE Leakage Correction

The GRACE partition equation for groundwater storage anomalies requires a leakage correction factor Lf that is conventionally applied as a basin-uniform scalar. Long et al. (2014) established the forward modeling approach for leakage correction, showing improvements of 37% in annual amplitudes and 36% in trends. Ma et al. (2024) demonstrated through Coordinated Forward Modeling at 0.5-degree scale that sub-regional groundwater storage trends can diverge substantially from basin-average behavior. Tripathi et al. (2022) showed that basin-average grid-scaled GRACE can be misleading due to compensating over-scaled and under-scaled pixels, recommending thorough grid-level assessment before downstream applications. Li et al. (2024) used in situ groundwater observations and aquifer storage coefficients as a priori information to estimate pixel-scale leakage correction factors via forward modeling, establishing precedent for the spatially distributed correction proposed here.

However, all prior pixel-scale approaches have been constrained by the incompleteness of the underlying well records -- a limitation that Paper 2 directly addresses by producing temporally complete records at all 592 wells.

### 4.3 Methods

**Spatial Interpolation.** The complete imputed dataset from Paper 2 (592 wells, 288 months) is interpolated to a regular spatial grid via three stages:

*Stage 1: Spatial Trend Surface.* A degree-2 polynomial trend surface is fitted via ridge regression to the temporal mean WTE of all 592 wells as a function of latitude, longitude, and ground surface elevation. This polynomial captures the large-scale spatial gradient driven by topography and regional geology, achieving R-squared = 0.957 on the GSLB dataset -- meaning that 95.7% of the variation in mean WTE across the 592 wells is explained by location and elevation alone. This single step reduces the 3,000-foot WTE range to residuals of approximately +/-30 feet. Grid cell elevations are obtained from the Copernicus DEM via the Open-Meteo Elevation API.

*Stage 2: EOF Decomposition.* The detrended residual matrix (288 months x 592 wells) is decomposed via Singular Value Decomposition into k temporal modes (U), singular values (S), and spatial loadings (V^T). The number of retained modes k is typically set to 20, capturing approximately 95% of the residual variance. The temporal modes represent shared hydrological patterns across the basin -- long-term trends, seasonal cycles, multi-year drought responses -- while the spatial loadings quantify how strongly each well participates in each pattern.

*Stage 3: Loading Interpolation.* For each grid cell, the k spatial loadings are estimated from nearby wells via Inverse Distance Weighting (exponent=2, 30 nearest neighbors). The grid cell's reconstructed WTE at each timestep is then the sum of the trend prediction and the EOF reconstruction. This guarantees temporal coherence because all timesteps share the same modal structure.

**Leakage Correction.** The interpolated groundwater-level fields are converted to volumetric storage anomalies using spatially varying specific yield estimates and aggregated to the 0.5-degree JPL TWSa grid. For each grid cell with sufficient in situ coverage, a pixel-wise leakage factor Lf(phi, lambda) is calibrated as the multiplicative scalar minimizing the mismatch between the GRACE-derived GWSa partition and the in situ-derived GWSa over the 2002-2024 study period. For grid cells lacking sufficient well coverage, Lf is propagated via a covariate-aware model using land cover, irrigation fraction, elevation, depth to groundwater, and aquifer-type indicators as predictors.

**Validation.** The resulting pixel-wise GWSa is validated against the in situ reconstruction at sub-basin scales and compared with the basin-uniform Lf approach of Paper 1 to quantify the improvement from spatially distributed correction.

### 4.4 Anticipated Results

Leave-one-out cross-validation on 30 wells demonstrates that the EOF interpolation framework achieves RMSE of 32.3 ft, representing a 49% reduction compared to plain IDW (RMSE = 62.9 ft) and a 73% reduction compared to per-timestep ordinary kriging (RMSE = 121.9 ft). The improvement is attributable to two factors: the trend surface absorbs the dominant spatial gradient (R-squared = 0.957), and EOF decomposition ensures temporal coherence by interpolating spatial loadings -- smooth scalar fields -- rather than raw values with large spatial gradients.

The pixel-wise Lf grid is anticipated to depart substantially from the basin-uniform value of 2 used in Paper 1. The largest leakage factors are expected along the Wasatch Front, where pumping-driven mass loss is most concentrated and the GRACE signal is most severely attenuated by spatial smoothing. Substantially smaller factors (closer to unity) are expected in the West Desert and Bear River sub-basins, where anthropogenic stress is diffuse and leakage effects are minimal. This spatial pattern should improve agreement between the corrected GWSa and the in situ reconstruction at sub-basin scales, where the uniform scalar of Paper 1 systematically over-corrects some regions and under-corrects others.

---

## 5. Timeline

| Period | Activity |
|---|---|
| May 2026 | Paper 1 submitted to journal; Paper 2 method development in progress, validation ongoing |
| Jun-Aug 2026 | Paper 2 manuscript drafting; begin cross-basin validation (Africa, South America sites) |
| Sep 2026 | Paper 2 submission |
| Sep-Nov 2026 | Paper 3 gridded Lf calibration, covariate propagation, sensitivity analyses |
| Dec 2026 | Paper 3 results, leakage correction validation, cross-basin interpolation testing |
| Jan-Feb 2027 | Paper 3 manuscript drafting and submission |
| Mar 2027 | Dissertation compilation |
| Apr 2027 | Dissertation defense and graduation |

---

## References

Abbott, B. W., et al. (2023). Emergency measures needed to rescue Great Salt Lake from ongoing collapse. Brigham Young University.

Ahmadi, A., et al. (2024). Integrating an interpolation technique and AI models using Bayesian model averaging to enhance groundwater level monitoring. Earth Science Informatics, 17, 4963-4984.

Bigalke, S., Loikith, P. C., & Siler, N. (2025). Explaining the 2022 Record Low Great Salt Lake Volume. Geophysical Research Letters, 52, e2024GL112154.

Candes, E. J. & Recht, B. (2009). Exact matrix completion via convex optimization. Foundations of Computational Mathematics, 9(6), 717-772.

Evans, S., Williams, G. P., Jones, N. L., Ames, D. P., & Nelson, E. J. (2020a). Exploiting Earth Observation Data to Impute Groundwater Level Measurements with an Extreme Learning Machine. Remote Sensing, 12, 2044.

Evans, S. W., Jones, N. L., Williams, G. P., Ames, D. P., & Nelson, E. J. (2020b). Groundwater Data Mapper: An open source web application for assessing groundwater sustainability. Environmental Modelling & Software, 131, 104782.

Fritsch, F. N. & Carlson, R. E. (1980). Monotone Piecewise Cubic Interpolation. SIAM Journal on Numerical Analysis, 17(2), 238-246.

Gharehbaghi, A., Ghasemlounia, R., Ahmadi, F., & Albaji, M. (2022). Groundwater level prediction with meteorologically sensitive GRU neural networks. Journal of Hydrology, 612, 128262.

Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling. Journal of Hydrology, 377(1-2), 80-91.

Hall, D. K., et al. (2024). Snowfall Replenishes Groundwater Loss in the Great Basin of the Western United States, but Cannot Compensate for Increasing Aridification. Geophysical Research Letters, 51, e2023GL107913.

Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R. (2021). Liquid Time-constant Networks. Proceedings of the AAAI Conference on Artificial Intelligence, 35(9), 7657-7666.

Hasani, R., Lechner, M., Amini, A., Liebenwein, L., Ray, A., Tschaikowski, M., Teschl, G., & Rus, D. (2022). Closed-form continuous-time neural networks. Nature Machine Intelligence, 4, 992-1003.

Huang, G.-B., Zhu, Q.-Y., & Siew, C.-K. (2006). Extreme learning machine: Theory and applications. Neurocomputing, 70(1-3), 489-501.

Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks. GMD Technical Report 148, German National Research Center for Information Technology.

Jasechko, S., et al. (2024). Rapid groundwater decline and some cases of recovery in aquifers globally. Nature, 625(7996), 715-720.

Jeong, J., Park, E., Chen, H., Kim, K.-Y., Han, W. S., & Suk, H. (2020). Estimation of groundwater level based on the robust training of recurrent neural networks using corrupted data. Journal of Hydrology, 582, 124512.

Levy, Z., Glas, R. L., Stagnitta, T. J., & Terry, N. (2025). ARCHI: A new R package for automated imputation of regionally correlated hydrologic records. Groundwater.

Li, B., et al. (2024). A New GRACE Downscaling Approach for Deriving High-Resolution Groundwater Storage Changes Using Ground-Based Scaling Factors. Water Resources Research, 60, e2023WR035210.

Li, Y., et al. (2025). Predicting regional-scale groundwater levels at high spatial resolution using spatial Random Forest models. International Journal of Applied Earth Observation and Geoinformation.

Lin, H., et al. (2022). Time series-based groundwater level forecasting using gated recurrent unit deep neural networks. Engineering Applications of Computational Fluid Mechanics, 16(1), 1655-1672.

Long, D., et al. (2014). Drought and flood monitoring for a large karst plateau in Southwest China using extended GRACE data. Remote Sensing of Environment, 155, 145-160.

Ma, G., et al. (2024). Improved Estimates of Sub-Regional Groundwater Storage Anomaly Using Coordinated Forward Modeling. Water Resources Research, 60(7), e2023WR036105.

Mazumder, R., Hastie, T., & Tibshirani, R. (2010). Spectral Regularization Algorithms for Learning Large Incomplete Matrices. Journal of Machine Learning Research, 11, 2287-2322.

Null, S. E. & Wurtsbaugh, W. A. (2020). Water Development, Consumptive Water Uses, and Great Salt Lake. In Baxter, B. K. & Butler, J. K. (Eds.), Great Salt Lake Biology, Springer.

Ramirez, S. G., Williams, G. P., & Jones, N. L. (2022). Groundwater Level Data Imputation Using Machine Learning and Remote Earth Observations Using Inductive Bias. Remote Sensing, 14, 5509.

Ramirez, S. G., Williams, G. P., Jones, N. L., Ames, D. P., & Radebaugh, J. (2023). Improving Groundwater Imputation through Iterative Refinement Using Spatial and Temporal Correlations from In Situ Data with Machine Learning. Water, 15, 1236.

Rateb, A. & Herring, T. A. (2020). Comparison of Groundwater Storage Changes From GRACE Satellites With Monitoring and Modeling of Major U.S. Aquifers. Water Resources Research, 56(12), e2020WR027556.

Rodell, M., et al. (2004). The global land data assimilation system. Bulletin of the American Meteorological Society, 85(3), 381-394.

Sharma, Y. K., Kim, S., & Tayerani Charmchi, A. S. (2024). Strategic imputation of groundwater data using machine learning. Groundwater for Sustainable Development, 27, 101300.

Shepard, D., Jones, N. L., & Williams, G. P. (2025). Application of the Groundwater Data Mapper Tool to Assess Storage Changes in a Groundwater-Driven Basin in the Klamath Watershed. Hydrology, 12(6), 140.

Stevens, M. D., et al. (2025). Groundwater Storage Loss in the Central Valley Analysis Using a Novel Method based on In Situ Data Compared to GRACE-Derived Data. Environmental Modelling & Software, 186, 106368.

Tapley, B. D., Bettadpur, S., Ries, J. C., Thompson, P. F., & Watkins, M. M. (2004). GRACE measurements of mass variability in the Earth system. Science, 305(5683), 503-505.

Tripathi, V., Groh, A., Horwath, M., & Ramsankaran, R. (2022). Scaling methods of leakage correction in GRACE mass change estimates revisited for the complex hydro-climatic setting of the Indus Basin. Hydrology and Earth System Sciences, 26, 4515-4535.

Vishwakarma, B. D., Devaraju, B., & Sneeuw, N. (2018). What is the spatial resolution of GRACE satellite products for hydrology? Remote Sensing, 10(6), 852.

Watkins, M. M., Wiese, D. N., Yuan, D.-N., Boening, C., & Landerer, F. W. (2015). Improved methods for observing Earth's time variable mass distribution with GRACE using spherical cap mascons. Journal of Geophysical Research: Solid Earth, 120(4), 2648-2671.

Wine, M. L. (2019). Climatization -- Negligent Attribution of Great Salt Lake Desiccation. Climate, 7(5), 67.

Wunsch, A., Liesch, T., & Broda, S. (2021). Groundwater level forecasting with artificial neural networks: a comparison of LSTM, CNN, and NARX. Hydrology and Earth System Sciences, 25, 1671-1687.

Young, Z., Kreemer, C., & Blewitt, G. (2021). GPS Constraints on Drought-Induced Groundwater Loss Around Great Salt Lake. Journal of Geophysical Research: Solid Earth, 126, e2021JB022020.

Zamora, H. & Inkenbrandt, P. (2024). Estimate of groundwater flow and salinity contribution to the Great Salt Lake. Geosites, 51, 1-24.
