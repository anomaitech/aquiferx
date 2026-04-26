"""
Linear algebra, statistics, and normalization.
Mirrors utils/math.ts: matMul, transpose, ridgeRegression, mean, std, Pearson, KGE, R2, normalize/denormalize.
"""

import math
from typing import List, Tuple

import numpy as np


# --- Normalization (JS: normalizeValue maps to [-0.8, 0.8], denormalizeValue reverses) ---

def get_scaler(values: List[float]) -> Tuple[float, float]:
    """Return (min, max) for a list of values. JS: getScaler."""
    if not values:
        return 0.0, 1.0
    return min(values), max(values)


def normalize_value(val: float, min_val: float, max_val: float) -> float:
    """Normalize to [-0.8, 0.8]. JS: normalizeValue."""
    if max_val == min_val:
        return 0.0
    return ((val - min_val) / (max_val - min_val)) * 1.6 - 0.8


def denormalize_value(norm_val: float, min_val: float, max_val: float) -> float:
    """Denormalize from [-0.8, 0.8]. JS: denormalizeValue."""
    if max_val == min_val:
        return min_val
    return ((norm_val + 0.8) / 1.6) * (max_val - min_val) + min_val


# --- Statistics ---

def calculate_mean(data: List[float]) -> float:
    """JS: calculateMean."""
    if not data:
        return 0.0
    return sum(data) / len(data)


def calculate_std_dev(data: List[float], mean_val: float = None) -> float:
    """Sample standard deviation. JS: calculateStdDev."""
    if len(data) < 2:
        return 0.0
    m = mean_val if mean_val is not None else calculate_mean(data)
    variance = sum((x - m) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(variance)


def calculate_pearson_correlation(x: List[float], y: List[float]) -> float:
    """JS: calculatePearsonCorrelation."""
    n = len(x)
    if n != len(y) or n == 0:
        return 0.0
    mean_x = calculate_mean(x)
    mean_y = calculate_mean(y)
    num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    den_x = sum((x[i] - mean_x) ** 2 for i in range(n))
    den_y = sum((y[i] - mean_y) ** 2 for i in range(n))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / math.sqrt(den_x * den_y)


def calculate_r2(observed: List[float], predicted: List[float]) -> float:
    """R² = 1 - SS_res/SS_tot. JS: calculateR2."""
    if len(observed) != len(predicted) or not observed:
        return 0.0
    mean_obs = calculate_mean(observed)
    ss_res = sum((observed[i] - predicted[i]) ** 2 for i in range(len(observed)))
    ss_tot = sum((observed[i] - mean_obs) ** 2 for i in range(len(observed)))
    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def calculate_kge(observed: List[float], predicted: List[float]) -> float:
    """Kling-Gupta Efficiency. JS: calculateKGE. Returns finite value when possible so API does not serialize -inf as null."""
    if len(observed) != len(predicted) or not observed:
        return float("-inf")
    mean_obs = calculate_mean(observed)
    mean_pred = calculate_mean(predicted)
    std_obs = calculate_std_dev(observed, mean_obs)
    std_pred = calculate_std_dev(predicted, mean_pred)
    # Constant observed series: return 0.0 if means match else -1.0 so UI gets a number, not null
    if std_obs == 0:
        return 0.0 if (mean_pred == mean_obs or abs(mean_pred - mean_obs) < 1e-9) else -1.0
    if mean_obs == 0:
        return float("-inf")
    r = calculate_pearson_correlation(observed, predicted)
    alpha = std_pred / std_obs
    beta = mean_pred / mean_obs
    kge = 1.0 - math.sqrt(
        (r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2
    )
    return kge


# --- Ridge regression (JS: ridgeRegression with Gaussian elimination) ---

def ridge_regression(X: np.ndarray, y: np.ndarray, alpha: float = 1e-4) -> np.ndarray:
    """
    Solve (X'X + alpha*I) w = X'y.
    JS uses manual Gaussian elimination; we use numpy for stability.
    """
    if X.size == 0 or len(y) == 0 or X.shape[0] != len(y):
        return np.array([])
    Xt = X.T
    XtX = Xt @ X
    n = XtX.shape[0]
    XtX = XtX + alpha * np.eye(n)
    XtY = Xt @ np.asarray(y, dtype=float)
    try:
        w = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(XtX, XtY, rcond=None)[0]
    return w
