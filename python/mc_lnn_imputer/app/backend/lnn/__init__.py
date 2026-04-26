"""
Liquid Neural Network (LNN) backend for time-series imputation.

Converts the small-gap batch LNN algorithm from the JS frontend (utils/math.ts)
into a Python backend for server-side or batch processing.
"""

from .types import DataPoint, SimulationParams, DT
from .math_utils import (
    ridge_regression,
    calculate_mean,
    calculate_std_dev,
    calculate_pearson_correlation,
    calculate_kge,
    calculate_r2,
    normalize_value,
    denormalize_value,
    get_scaler,
)
from .lnn_core import prepare_data, get_current_observation_value, run_lnn_simulation  # noqa: F401
from .gaps import identify_gaps, identify_outliers
from .small_gap_batch import batch_impute_small_gap_instances

__all__ = [
    "DataPoint",
    "SimulationParams",
    "DT",
    "ridge_regression",
    "calculate_mean",
    "calculate_std_dev",
    "calculate_pearson_correlation",
    "calculate_kge",
    "calculate_r2",
    "normalize_value",
    "denormalize_value",
    "get_scaler",
    "prepare_data",
    "get_current_observation_value",
    "run_lnn_simulation",
    "identify_gaps",
    "identify_outliers",
    "batch_impute_small_gap_instances",
]
