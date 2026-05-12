# PhD Prospectus

# Advancing Basin-Scale Groundwater Storage Estimation Through Hybrid Spatiotemporal Imputation, EOF-Based Interpolation, and Satellite-Derived Leakage Correction

---

## 1. Introduction

### 1.1 Overall Objective and Significance

Groundwater sustains roughly half of global irrigation and a third of municipal water supply, yet it remains the most poorly monitored component of the terrestrial water budget (Scanlon et al., 2023). Jasechko et al. (2024) analyzed 170,000 monitoring wells across 1,693 aquifer systems on six continents and found that 30% exhibit accelerating decline, with rates exceeding 0.5 m/yr widespread in irrigated regions of South Asia, the Middle East, North Africa, and the western United States. The consequences are severe: land subsidence, baseflow reduction, ecosystem degradation, and diminished drought resilience. Reliable basin-scale quantification of groundwater storage change is therefore essential for water-resources planning, yet three fundamental methodological barriers stand in the way.

First, the in situ monitoring records that underpin any ground-truth storage estimate are riddled with temporal gaps. Public databases such as the USGS National Water Information System contain millions of individual water-level measurements, but the median well has fewer than 50 observations scattered across decades, with gaps of multiple years commonplace. Standard interpolation methods (linear, spline) fail for gaps longer than one to two years, and the machine-learning approaches currently used to fill these gaps treat each well independently, ignoring the spatial correlation structure across the monitoring network.

Second, converting imputed point-well records into the continuous spatial fields needed for volumetric storage estimation requires spatial interpolation. Kriging, the dominant geostatistical method, assumes stationarity that is routinely violated in heterogeneous basins where water-table elevation spans hundreds to thousands of meters. No existing method simultaneously exploits both the shared temporal patterns across wells and the spatial structure of the monitoring network for interpolation.

Third, satellite gravimetry from GRACE/GRACE-FO (Tapley et al., 2004) provides the only global, spatially continuous measure of total water storage change, but its coarse resolution (~300 km) introduces signal leakage that must be corrected before basin-scale groundwater storage can be isolated. The leakage correction factor is conventionally applied as a single basin-uniform scalar, obscuring the spatial heterogeneity of storage change within the basin.

These three barriers form a chain: incomplete well records limit the quality of spatial fields, which in turn limit the fidelity of satellite-derived corrections. This dissertation addresses all three barriers in sequence through three papers, each building on the outputs of the previous:

1. **Paper 1** (completed) establishes baseline groundwater storage estimates for the Great Salt Lake Basin using multiple independent methods, revealing the sensitivity of GRACE-derived estimates to leakage handling and motivating the need for improved imputation.
2. **Paper 2** develops a novel hybrid imputation framework coupling Matrix Completion with Liquid Neural Networks (MC+LNN) that jointly exploits spatial cross-well correlations and continuous-time temporal dynamics -- the first such coupling in the groundwater literature.
3. **Paper 3** applies the imputed records from Paper 2 to produce spatially continuous groundwater fields via EOF-based interpolation, and uses those fields to derive the first pixel-wise GRACE leakage correction grid calibrated against spatially complete in situ data.

The Great Salt Lake Basin (GSLB) serves as the initial validation site, chosen for its dense USGS monitoring network (592 eligible wells), heterogeneous hydrogeology spanning four states, and concentrated anthropogenic pumping along the Wasatch Front. However, the methods are designed to be general-purpose: the imputation framework requires only well locations, irregular observations, and globally available GLDAS auxiliary data, with all hyperparameters auto-optimized per well. Validation on additional basins across diverse hydrogeological and climatic settings -- including sites in Sub-Saharan Africa and South America where monitoring networks are sparser -- is planned to demonstrate transferability.

### 1.2 Background

#### 1.2.1 The Gap Problem in Groundwater Monitoring

The utility of in situ groundwater-level records depends critically on temporal completeness. In practice, wells are installed, monitored intermittently, and often abandoned, producing records with gaps ranging from months to years. A progression of imputation methods has been developed to address this challenge.

Evans et al. (2020a) introduced the Groundwater Level Mapping Tool (GWLMT), coupling Extreme Learning Machines (ELM; Huang et al., 2006) with satellite-derived auxiliary variables from GRACE and GLDAS. Evans et al. (2020b) demonstrated that ELM with Earth observation inputs substantially outperforms conventional interpolation. Ramirez et al. (2022) incorporated inductive bias from remote sensing, and Ramirez et al. (2023) added iterative refinement using both spatial correlations from neighboring wells and temporal correlations from auxiliary variables. This line of work has been applied to basin-scale storage assessment in California's Central Valley (Stevens et al., 2025) and the Klamath watershed (Shepard et al., 2025).

Deep learning approaches have also been explored. Jeong et al. (2020) demonstrated that recurrent neural networks can reconstruct missing groundwater levels with high accuracy but degrade sharply beyond one- to two-year gaps. Wunsch et al. (2021) compared LSTM, CNN, and NARX architectures, finding that multivariate models consistently outperform univariate ones. Gharehbaghi et al. (2022) and Lin et al. (2022) showed GRU networks perform comparably to LSTM.

A common limitation across these approaches is that they treat each well's imputation as essentially independent. Spatial information enters only indirectly through basin-average satellite-derived covariates, not through the actual cross-well correlation structure. The recently released ARCHI package (Levy et al., 2025) addresses this by using iterative donor regression -- each well imputed via linear regression from more complete reference wells -- but operates within a purely linear framework without auxiliary climatic forcings or nonlinear temporal modeling.

Matrix completion offers a fundamentally different perspective by arranging all well records as a partially observed matrix and exploiting low-rank structure to infer missing entries from cross-well correlations (Candes and Recht, 2009). Sharma, Kim, and Tayerani Charmchi (2024) evaluated SoftImpute for groundwater levels in the Chao-Phraya River Basin, finding it excels in sparse networks -- but treated it as a standalone method without temporal coupling. On the temporal side, Liquid Neural Networks with Closed-form Continuous-depth cells (Hasani et al., 2022) model dynamics as continuous-time ODEs, naturally accommodating irregular sampling without resampling -- a property uniquely suited to groundwater monitoring data.

**No prior work has combined matrix completion with continuous-time neural networks for groundwater imputation.** This dissertation proposes a coupled MC+LNN framework in which matrix completion provides spatially informed initial estimates by exploiting cross-well correlations, and the LNN refines those estimates using continuous-time dynamics conditioned on auxiliary climatic forcings. The coupling is complementary: MC handles the spatial dimension, LNN handles the temporal dimension.

#### 1.2.2 Spatial Interpolation of Groundwater Levels

Converting imputed point-well records to continuous fields requires spatial interpolation. Geostatistical methods, particularly kriging, have been dominant but their stationarity assumptions are frequently violated in heterogeneous basins (Ahmadi et al., 2024; Li et al., 2025). Empirical Orthogonal Function (EOF) analysis decomposes spatiotemporal fields into temporal modes and spatial loadings, enabling interpolation by estimating a few smooth spatial scalars at unobserved locations rather than hundreds of raw time values. This approach has been widely used in climate science but has not been systematically applied to groundwater spatial interpolation in the context of satellite-derived storage estimation.

#### 1.2.3 GRACE Leakage Correction

The GRACE partition equation for groundwater storage anomalies requires a leakage correction factor Lf that is conventionally applied as a basin-uniform scalar (Vishwakarma et al., 2018). Long et al. (2014) established forward modeling for leakage correction. Ma et al. (2024) demonstrated that sub-regional trends can diverge substantially from basin averages. Tripathi et al. (2022) showed that basin-average corrections can be misleading due to compensating errors across pixels. Li et al. (2024) estimated pixel-scale correction factors using in situ data, establishing precedent for the spatially distributed correction proposed here. However, all prior pixel-scale approaches have been constrained by the incompleteness of the underlying in situ records -- the very limitation that Papers 1 and 2 of this dissertation address.

#### 1.2.4 The Great Salt Lake Basin as Initial Validation Site

The GSLB is a closed hydrologic system covering approximately 93,000 km-squared. Consumptive water uses have depleted inflows to the Great Salt Lake by 39% (Null and Wurtsbaugh, 2020), while Bigalke et al. (2025) attributed the 2022 record-low lake volume to a combination of reduced streamflow and climate warming. Hall et al. (2024) documented 68.7 km-cubed of groundwater loss from 2002-2023 across the broader Great Basin. Zamora and Inkenbrandt (2024) revised the groundwater contribution to the lake upward to 10% of total inflows. The basin's dense USGS monitoring network (592 eligible wells over 2000-2023), heterogeneous hydrogeology, and concentrated anthropogenic stress make it an ideal initial testbed. Subsequent validation on basins with sparser networks and different climatic regimes will test the framework's global transferability.

### 1.3 Specific Objectives

The three objectives are sequential -- each builds on the outputs of the previous, forming an integrated pipeline from raw observations to satellite-corrected storage estimates:

**Objective 1 (Paper 1).** Quantify multi-decadal groundwater storage change in the GSLB (2002-2024) by integrating GRACE-derived, GLDAS-derived, and in situ estimates within a unified framework. Identify the methodological sensitivity to surface-water inclusion and GRACE leakage correction that motivates the spatially distributed approach of Paper 3. *(Completed; submitted to journal.)*

**Objective 2 (Paper 2).** Develop and validate a general-purpose hybrid imputation framework for sparse, irregular groundwater-level time series that couples PCHIP for short gaps with Matrix Completion + Liquid Neural Networks (MC+LNN) for long gaps, using globally available auxiliary climatic forcings. Validate initially on the GSLB, with planned extension to basins in Africa and South America.

**Objective 3 (Paper 3).** Apply the imputation framework from Paper 2 to produce spatially continuous groundwater-level fields via spatial trend decomposition and EOF analysis, and use those fields to derive the first pixel-wise GRACE leakage correction grid calibrated against spatially complete in situ data. Benchmark against independent mascon-based estimates.

---

## 2. Objective 1 (Paper 1)

### 2.1 Objective

To quantify groundwater storage change in the Great Salt Lake Basin from 2002 through 2024 using multiple independent estimation methods, evaluate methodological sensitivities, and establish the baseline that motivates the imputation and leakage-correction advances of Papers 2 and 3.

### 2.2 Background

The GSLB's basin-scale groundwater contribution has been characterized only at sub-basin scale or in steady-state analyses (Zamora and Inkenbrandt, 2024). Prior GRACE studies addressed the broader Great Basin (Hall et al., 2024) or localized loss via GPS (Young, Kreemer, and Blewitt, 2021), but no long-term, multi-method record specific to the full GSLB existed.

### 2.3 Methods

Five GWSa estimates were computed: (1) GRACE-raw (TWSa minus GLDAS stores); (2) GRACE-sw (surface-water adjusted); (3) GRACE-Lf (leakage-corrected, Lf=2 calibrated against in situ data); (4) GLDAS-2.2 (GRACE-assimilated CLSM); and (5) GWDM (in situ via the GWLMT workflow of Evans et al., 2020a, using PCHIP, ELM, and kriging with specific yield 0.15).

### 2.4 Results

All methods identified two major drawdowns (2012-2016, 2019-2022). The leakage-corrected GRACE estimate yielded 10.1 km-cubed loss for 2011-2016, consistent with the independent GPS estimate of 10.9 +/- 2.8 km-cubed (Young, Kreemer, and Blewitt, 2021). Surface water accounted for 31% of basin storage change. The basin-uniform leakage factor improved GRACE-in situ correlation from 0.17 to 0.77, but this uniform treatment masks known spatial heterogeneity in pumping and recharge -- motivating the pixel-wise approach of Paper 3. The in situ estimate itself was limited by well-record gaps and reliance on a single imputation method (ELM) -- motivating the improved imputation of Paper 2.

---

## 3. Objective 2 (Paper 2)

### 3.1 Objective

To develop, validate, and benchmark a general-purpose hybrid imputation framework that, for the first time, jointly exploits spatial cross-well correlations (via matrix completion) and continuous-time temporal dynamics (via liquid neural networks) for reconstructing groundwater-level records with multi-year gaps.

### 3.2 Methods

The framework operates on monthly-aggregated well observations via a two-stage pipeline:

**Stage 1: PCHIP Small-Gap Fill.** Gaps of 24 months or shorter are filled using Piecewise Cubic Hermite Interpolating Polynomials, which preserve monotonicity and local shape without oscillation. This deterministic step densifies the monitoring network, providing the spatial coverage needed for reliable donor correlation in Stage 2.

**Stage 2: MC+LNN Large-Gap Fill.** Gaps exceeding 24 months are filled via three coupled phases:

*Phase 2a -- Donor Regression.* For each target well, the top 15 most correlated donor wells are selected from the PCHIP-densified network. Per-donor OLS regression provides a trend-aware initialization for the gap period, following the donor-correlation concept of ARCHI (Levy et al., 2025) but extending it into a matrix-completion framework.

*Phase 2b -- Matrix Completion.* A composite matrix is constructed with rows representing the target well, weighted donor wells, GLDAS auxiliary variables (soil moisture at five temporal scales), and seasonal encoding. SoftImpute (iterative truncated SVD with adaptive rank selection) fills the target row's gaps by exploiting the low-rank structure shared across all rows. Variance-preserving bias correction ensures the predictions match the observed mean and variance.

*Phase 2c -- LNN Temporal Refinement.* A Liquid Neural Network with Closed-form Continuous-time cells (Hasani et al., 2022) refines the MC predictions. The MC output serves as reservoir input during gap periods, while the LNN readout is trained exclusively on real observations via ridge regression. This design ensures the temporal model learns from ground truth while benefiting from MC's spatial context. Hyperparameters are auto-optimized per well via grid search over 8 trials with 3-model ensemble selection by Kling-Gupta Efficiency.

**Validation.** The framework is validated on the GSLB well network (592 wells, 288 months, 2000-2023) under two cross-validation scenarios: random missing data (5-50% removed, 50 trials each) and consecutive year gaps (1-5 years, 20 trials each). Performance is assessed via KGE, R-squared, and RMSE, with baselines including ELM, pure MC, and pure LNN to isolate each component's contribution. Additional basins across diverse hydrogeological and climatic settings are planned for transferability assessment.

### 3.3 Anticipated Results

Cross-validation on the GSLB demonstrates:

| Scenario | KGE | R-squared | RMSE (ft) |
|---|---|---|---|
| 5% random missing | 0.837 +/- 0.085 | 0.771 | 2.53 |
| 20% random missing | 0.853 +/- 0.051 | 0.787 | 2.64 |
| 50% random missing | 0.847 +/- 0.035 | 0.788 | 2.74 |
| 1-year consecutive gap | 0.783 +/- 0.116 | 0.703 | 2.98 |
| 3-year consecutive gap | 0.802 +/- 0.076 | 0.730 | 3.02 |
| 5-year consecutive gap | 0.815 +/- 0.063 | 0.744 | 2.91 |

KGE remains above 0.84 across all random-missing rates and above 0.78 for all consecutive-gap lengths, with RMSE consistently below 3 ft. The PCHIP small-gap stage is critical: replacing it with LNN-based filling degrades KGE by 10-19%, because PCHIP densification improves donor correlations for the MC stage. Cross-validation across low-variance (std < 0.2 ft), medium-variance, and high-variance (std > 20 ft) wells confirms robust performance across the full variance spectrum, with 13 of 15 tested wells achieving KGE above 0.65. The framework imputes all 592 GSLB wells to temporal completeness, producing the input dataset for Paper 3.

---

## 4. Objective 3 (Paper 3)

### 4.1 Objective

To produce spatially continuous groundwater-level fields from the imputed records of Paper 2 via EOF-based interpolation, and to use those fields to derive the first pixel-wise GRACE leakage correction grid calibrated against spatially complete in situ data.

### 4.2 Methods

**Spatial Interpolation.** The complete imputed dataset (592 wells, 288 months) is interpolated to a regular grid via three stages: (1) a degree-2 polynomial trend surface of temporal-mean WTE as a function of latitude, longitude, and elevation (R-squared = 0.957); (2) EOF decomposition of the detrended residuals via SVD into k temporal modes and spatial loadings; (3) IDW interpolation of the spatial loadings -- not raw values -- to grid cells. This approach guarantees temporal coherence because all timesteps share the same modal structure, and reduces the interpolation problem from 288 values to k small scalars per grid cell.

**Leakage Correction.** The interpolated fields are converted to volumetric storage anomalies using spatially varying specific yield and aggregated to the 0.5-degree GRACE grid. A pixel-wise leakage factor Lf(phi, lambda) is calibrated against the in situ-derived GWSa. For grid cells lacking sufficient well coverage, Lf is propagated via a covariate-aware model.

**Benchmarking.** The resulting GWSa is compared against mascon-based GWSa (Watkins et al., 2015), which handles leakage structurally rather than empirically. Because all auxiliary terms are identical, any divergence is attributable to leakage handling alone.

### 4.3 Anticipated Results

Leave-one-out cross-validation on 30 wells yields interpolation RMSE of 32.3 ft, a 49% reduction versus IDW (62.9 ft) and 73% versus per-timestep kriging (121.9 ft). The pixel-wise Lf grid is expected to show large factors along the Wasatch Front (concentrated pumping) and near-unity values in the West Desert (diffuse stress), departing substantially from the basin-uniform Lf = 2 of Paper 1 and improving sub-basin agreement with the in situ reconstruction.

---

## 5. Timeline

| Period | Activity |
|---|---|
| May 2026 | Paper 1 submitted to journal; Paper 2 method development in progress, validation ongoing |
| Jun-Aug 2026 | Paper 2 manuscript drafting; begin cross-basin validation (Africa, South America sites) |
| Sep 2026 | Paper 2 submission |
| Sep-Dec 2026 | Paper 3 gridded Lf calibration, covariate propagation, sensitivity analyses |
| Jan-Feb 2027 | Paper 3 results, mascon benchmarking, cross-basin interpolation testing |
| Mar 2027 | Paper 3 manuscript drafting |
| Apr 2027 | Paper 3 submission; dissertation compilation |
| May-Jun 2027 | Dissertation defense and graduation |

---

## References

Abbott, B. W., et al. (2023). Emergency measures needed to rescue Great Salt Lake from ongoing collapse. Brigham Young University. https://pws.byu.edu/great-salt-lake

Ahmadi, A., et al. (2024). Integrating an interpolation technique and AI models using Bayesian model averaging to enhance groundwater level monitoring. Earth Science Informatics, 17, 4963-4984.

Bigalke, S., Loikith, P. C., & Siler, N. (2025). Explaining the 2022 Record Low Great Salt Lake Volume. Geophysical Research Letters, 52, e2024GL112154.

Candes, E. J. & Recht, B. (2009). Exact matrix completion via convex optimization. Foundations of Computational Mathematics, 9(6), 717-772.

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

Levy, Z., Glas, R. L., Stagnitta, T. J., & Terry, N. (2025). ARCHI: A new R package for automated imputation of regionally correlated hydrologic records. Groundwater. https://doi.org/10.1111/gwat.13474

Li, B., et al. (2024). A New GRACE Downscaling Approach for Deriving High-Resolution Groundwater Storage Changes Using Ground-Based Scaling Factors. Water Resources Research, 60, e2023WR035210.

Li, Y., et al. (2025). Predicting regional-scale groundwater levels at high spatial resolution using spatial Random Forest models. International Journal of Applied Earth Observation and Geoinformation.

Lin, H., et al. (2022). Time series-based groundwater level forecasting using gated recurrent unit deep neural networks. Engineering Applications of Computational Fluid Mechanics, 16(1), 1655-1672.

Long, D., et al. (2014). Drought and flood monitoring for a large karst plateau in Southwest China using extended GRACE data. Remote Sensing of Environment, 155, 145-160.

Ma, G., et al. (2024). Improved Estimates of Sub-Regional Groundwater Storage Anomaly Using Coordinated Forward Modeling. Water Resources Research, 60(7), e2023WR036105.

Null, S. E. & Wurtsbaugh, W. A. (2020). Water Development, Consumptive Water Uses, and Great Salt Lake. In Baxter, B. K. & Butler, J. K. (Eds.), Great Salt Lake Biology, Springer, pp. 1-30.

Ramirez, S. G., Williams, G. P., & Jones, N. L. (2022). Groundwater Level Data Imputation Using Machine Learning and Remote Earth Observations Using Inductive Bias. Remote Sensing, 14, 5509. https://doi.org/10.3390/rs14215509

Ramirez, S. G., Williams, G. P., Jones, N. L., Ames, D. P., & Radebaugh, J. (2023). Improving Groundwater Imputation through Iterative Refinement Using Spatial and Temporal Correlations from In Situ Data with Machine Learning. Water, 15, 1236. https://doi.org/10.3390/w15061236

Rateb, A. & Herring, T. A. (2020). Comparison of Groundwater Storage Changes From GRACE Satellites With Monitoring and Modeling of Major U.S. Aquifers. Water Resources Research, 56(12), e2020WR027556.

Rodell, M., et al. (2004). The global land data assimilation system. Bulletin of the American Meteorological Society, 85(3), 381-394.

Scanlon, B. R., et al. (2023). Global water resources and the role of groundwater in a resilient water future. Nature Reviews Earth & Environment, 4, 87-101.

Sharma, Y. K., Kim, S., & Tayerani Charmchi, A. S. (2024). Strategic imputation of groundwater data using machine learning: Insights from diverse aquifers in the Chao-Phraya River Basin. Groundwater for Sustainable Development, 27, 101300.

Shepard, D., Jones, N. L., & Williams, G. P. (2025). Application of the Groundwater Data Mapper Tool to Assess Storage Changes in a Groundwater-Driven Basin in the Klamath Watershed, Oregon, USA. Hydrology, 12(6), 140. https://doi.org/10.3390/hydrology12060140

Stevens, M. D., Ramirez, S. G., Martin, E.-M. H., Jones, N. L., Williams, G. P., Adams, K. H., Ames, D. P., & Pulla, S. T. (2025). Groundwater Storage Loss in the Central Valley Analysis Using a Novel Method based on In Situ Data Compared to GRACE-Derived Data. Environmental Modelling & Software, 186, 106368. https://doi.org/10.1016/j.envsoft.2025.106368

Tapley, B. D., Bettadpur, S., Ries, J. C., Thompson, P. F., & Watkins, M. M. (2004). GRACE measurements of mass variability in the Earth system. Science, 305(5683), 503-505.

Tripathi, V., Groh, A., Horwath, M., & Ramsankaran, R. (2022). Scaling methods of leakage correction in GRACE mass change estimates revisited for the complex hydro-climatic setting of the Indus Basin. Hydrology and Earth System Sciences, 26, 4515-4535.

Vishwakarma, B. D., Devaraju, B., & Sneeuw, N. (2018). What is the spatial resolution of GRACE satellite products for hydrology? Remote Sensing, 10(6), 852.

Watkins, M. M., Wiese, D. N., Yuan, D.-N., Boening, C., & Landerer, F. W. (2015). Improved methods for observing Earth's time variable mass distribution with GRACE using spherical cap mascons. Journal of Geophysical Research: Solid Earth, 120(4), 2648-2671.

Young, Z., Kreemer, C., & Blewitt, G. (2021). GPS Constraints on Drought-Induced Groundwater Loss Around Great Salt Lake, Utah, With Implications for Seismicity Modulation. Journal of Geophysical Research: Solid Earth, 126, e2021JB022020.

Zamora, H. & Inkenbrandt, P. (2024). Estimate of groundwater flow and salinity contribution to the Great Salt Lake using groundwater levels and spatial analysis. Geosites, 51, 1-24.
