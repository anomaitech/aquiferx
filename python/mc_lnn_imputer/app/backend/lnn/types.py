"""
Data types and constants aligned with the JS frontend (types.ts, constants.ts).
"""

from dataclasses import dataclass, field
from typing import List, Optional

# Time step for reservoir ODE (matches JS constants.ts DT = 0.1)
DT = 0.1


@dataclass
class DataPoint:
    """Single time-series point. Mirrors JS DataPoint interface."""
    time: float
    observed: Optional[float]
    instance_id: str = ""
    ground_truth: Optional[float] = None
    auxiliaries: Optional[List[float]] = None
    imputed: Optional[float] = None
    imputed_std: Optional[float] = None  # Std across ensemble runs (uncertainty)
    is_masked: bool = False
    is_outlier: bool = False
    original_observed: Optional[float] = None
    date_label: Optional[str] = None
    timestamp: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


@dataclass
class SimulationParams:
    """LNN simulation parameters. Mirrors JS SimulationParams and DEFAULT_PARAMS."""
    reservoir_size: int = 20
    spectral_radius: float = 0.9
    leak_rate: float = 0.2
    sparsity: float = 0.4
    input_scaling: float = 0.5
    learning_rate: float = 0.1
    iterations: int = 50
    use_liquid_time_constant: bool = False
    transfer_threshold: float = 0.3
    transfer_neighbor_count: int = 5
    max_gap_threshold: int = 10
    large_gap_threshold: int = 200
    kge_threshold: float = 0.5
    small_gap_kge_threshold: float = 0.3
    large_gap_kge_threshold: float = 0.4
    small_gap_use_auxiliary: bool = True
    small_gap_auxiliary_weight: float = 0.5
    small_gap_use_neighbors: bool = False
    small_gap_max_iterations: int = 10
    small_gap_optimize_trials: int = 10  # LNNcfc/GLNN autotune trials per instance
    small_gap_gan_trials: int = 3  # GAIN/CondGAN autotune trials per instance (default 3 for speed)
    ridge_alpha: float = 1e-4
    # Geostat (Kriging/Cokriging) settings
    geostat_variogram: str = "spherical"  # linear, spherical, gaussian, exponential
    geostat_seasonal_period: int = 12  # 0 = off, 12 = monthly seasonality
    geostat_use_patched: bool = False  # True = patched kriging (temporal windows), Meggiorin et al. Water 2023
    geostat_range_adjustment: str = "none"  # none | series | interval (interval = patched)
    geostat_patch_half_frac: float = 0.25  # half-window for patched kriging as fraction of time span
    geostat_use_spatial_neighbors: bool = True  # use 2D (lon,lat) kriging from neighbor instances when coords present
    geostat_max_spatial_neighbors: Optional[int] = 10  # max neighbors by distance (None = use all)
    geostat_min_spatial_neighbors: int = 3  # min neighbors required for kriging at a time step
    # Outlier handling: keep = do not remove; remove_iterative = detect and re-impute (default)
    small_gap_outlier_handling: str = "remove_iterative"
    large_gap_outlier_handling: str = "remove_iterative"
    # LNN Advanced (A, B, C, D)
    lnn_ensemble_size: int = 1          # Number of ensemble runs (1 = single, >1 = average + std)
    lnn_bidirectional: bool = False     # Run reservoir forward + backward, concatenate states
    lnn_washout: int = 10               # Skip first N reservoir states before ridge regression
    lnn_time_aware_leak: bool = False   # Scale leak rate by actual dt / DT for irregular time steps
    # GAN Advanced (F, H, I)
    gan_mode: str = "gain"              # 'gain' | 'wgan_gp' | 'conditional'
    gan_gp_lambda: float = 10.0         # Gradient penalty lambda for WGAN-GP
    gan_multiple_imputations: int = 1   # Number of imputation samples (1 = single, >1 = mean + std)
    gan_use_auxiliary: bool = True       # Use auxiliary variables as conditioning input in GAN
    # Kriging Advanced (J, L, M, N)
    kriging_spatiotemporal: bool = True  # Use 3D spatiotemporal kriging when coords available
    kriging_auto_variogram: bool = False # Auto-select variogram model via cross-validation
    kriging_non_negative: bool = False   # Clamp kriging predictions to >= 0
    kriging_autotune_long_gap: bool = False  # Auto-tune LNN+Kriging params for long gaps
    kriging_donor_selection: str = "distance"  # "distance" = closest by (lon,lat); "correlation" = ARCHI (useful when many instances)
    # ARCHI-style large-gap donor selection (when kriging_donor_selection == "correlation")
    archi_use_regional_correlation: bool = True
    archi_min_correlation: float = 0.3
    archi_max_donors: int = 10
    archi_correlation_weight_power: float = 2.0
    archi_algorithm_selection: str = "weighted_average"  # weighted_average | best_donor | regression | auto
    archi_donor_pool: str = "all"  # "all" = consider all instances; "proximity" = only within radius
    archi_donor_radius: float = 100.0  # Radius in km for proximity-based donor selection
    # LNN Aux Placeholder readout: "ridge" (default) or "gp" (Gaussian Process for uncertainty)
    lnn_aux_placeholder_readout: str = "ridge"
    # Spike correction: blend with local estimate in sharp-change regions (sudden peak/dip)
    lnn_aux_placeholder_spike_correction: bool = False
    lnn_aux_placeholder_spike_blend: float = 0.5   # 0=no correction, 1=full local in spike regions
    lnn_aux_placeholder_spike_window: int = 5      # window size for detecting sharp aux change
    lnn_aux_placeholder_spike_sharp_frac: float = 0.15  # gap is "sharp" if |next_obs-last_obs| >= this * range
    # LNN-DBE (Deep Bidirectional Ensemble)
    lnn_dbe_ensemble_size: int = 10      # Number of ensemble members (averaged for final prediction)
    lnn_dbe_seasonal_period: int = 12    # Seasonal encoding period (12 = monthly/annual, 0 = off)
    lnn_dbe_bidirectional: bool = True   # Run reservoir forward + backward, concatenate states
    # LNN CFC Enhanced (Bidirectional Multi-Scale + Polynomial Placeholder + Anchor Injection + Ensemble)
    lnn_cfc_enhanced_seasonal_period: int = 12       # Seasonal encoding period (12 = monthly/annual, 0 = off)
    lnn_cfc_enhanced_bidirectional: bool = True       # Forward + backward reservoir, concatenated states
    lnn_cfc_enhanced_ensemble_size: int = 1           # Number of ensemble members (1 = single, >1 = average + std)
    lnn_cfc_enhanced_anchor_injection: bool = True    # Inject placeholder into reservoir state during large gaps
    lnn_cfc_enhanced_anchor_interval: int = 5         # Inject every N steps during large gaps
    lnn_cfc_enhanced_anchor_blend: float = 0.3        # Blend: x = (1-blend)*drifted + blend*anchor
    lnn_cfc_enhanced_polynomial_placeholder: bool = True  # Polynomial + interaction features in placeholder ridge
    lnn_cfc_enhanced_edge_blend_width: int = 3        # Linearly blend reservoir+placeholder at gap edges
    lnn_cfc_enhanced_pseudo_weight: float = 1.0       # Weight for placeholder pseudo-labels in readout training (0-1; 1.0 = equal to observed)
    # GP Aux Enhanced (Gaussian Process with Auxiliary Placeholder Training)
    gp_aux_enhanced_polynomial: bool = True        # Polynomial + interaction features (default on)
    gp_aux_enhanced_pseudo_weight: float = 1.0     # Placeholder training weight (1.0 = full trust)
    # Auxiliary selection/filtering in auxiliary-only mode (frontend-driven, backend stores for reference)
    aux_selection_mode: str = "all"       # "all" | "top_n" | "positive_only" | "above_threshold"
    aux_top_n: int = 1                    # When mode="top_n": keep top N by |r|
    aux_min_correlation: float = 0.3      # When mode="above_threshold": min r
    aux_min_count: int = 1               # Min qualifying auxiliaries required (below = skip instance)
    aux_require_positive: bool = False    # Additional filter: discard negative correlations
    aux_correlation_weighted: bool = True  # Weight selected auxiliaries by |correlation| (default: on)
    aux_fallback_archi: str = "none"      # "none" | "classic" | "combined" — fallback for skipped instances
    # MC + LNN CFC parameters
    mc_max_donors: int = 15              # Max donor wells for Matrix Completion
    mc_min_correlation: float = 0.3      # Min Pearson R for MC donor selection
