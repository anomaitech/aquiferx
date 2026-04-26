"""
GP Aux Enhanced: Gaussian Process with Auxiliary Placeholder Training.

Extends the existing GP+Aux (kriging_aux_impute.py) with two key ideas:

1. **Placeholder pseudo-label training** — Build ridge placeholder from auxiliary
   data (same polynomial features as LNN CFC Enhanced), then train the GP on ALL
   points: observed (weight 1.0) + placeholder predictions at gaps (configurable
   weight). This gives the GP many more training points.

2. **Polynomial features** — Upgrade from [time, sin, cos, aux] to include
   [t^2, aux*t, aux_i*aux_j] for nonlinear relationships.

Supports both small-gap (batch) and large-gap (auxiliary-only, single instance)
modes. GP naturally provides posterior uncertainty (std).
"""

from dataclasses import replace
from typing import List, Dict, Any, Optional, Tuple, Callable
from collections import defaultdict
import warnings

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

from .types import DataPoint, SimulationParams
from .math_utils import calculate_kge
from .gaps import identify_gaps

warnings.simplefilter("ignore", category=ConvergenceWarning)

_TAG = "[GP+Aux-Enh]"


# ---------------------------------------------------------------------------
# Seasonal helpers (same as kriging_aux_impute)
# ---------------------------------------------------------------------------

def _seasonal_features(times: np.ndarray, period: float) -> np.ndarray:
    angle = 2.0 * np.pi * times / period
    return np.column_stack([np.sin(angle), np.cos(angle)])


def _noise_kernel():
    return C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=1e-2)


def _gp_optimizer_restarts(n_train: int, n_features: int, quality_mode: str = "adaptive") -> int:
    """
    Keep the same GP architecture while limiting optimizer restarts on large fits.
    """
    if quality_mode == "max":
        return 10
    if quality_mode == "fast":
        return 1
    if n_train >= 400 or n_features >= 20:
        return 2
    if n_train >= 200 or n_features >= 14:
        return 4
    return 6


def _learn_seasonal_noise(
    all_known_t: np.ndarray,
    all_known_z: np.ndarray,
    period: float,
) -> Callable[[np.ndarray], np.ndarray]:
    n = len(all_known_t)
    print(f"{_TAG} Stage 1: learning seasonal noise from {n} pooled known points (period={period})", flush=True)
    if n < 10:
        return lambda t: np.ones(len(t))
    sort_idx = np.argsort(all_known_t)
    sorted_t = all_known_t[sort_idx]
    sorted_z = all_known_z[sort_idx]
    half_win = period / 2.0
    local_residuals = np.zeros(n, dtype=np.float64)
    for i in range(n):
        mask = np.abs(sorted_t - sorted_t[i]) <= half_win
        local_mean = sorted_z[mask].mean()
        local_residuals[i] = sorted_z[i] - local_mean
    residuals_sq = np.maximum(local_residuals ** 2, 1e-12)
    log_res_sq = np.log(residuals_sq)
    max_pool = 2000
    if n > max_pool:
        idx = np.random.default_rng(42).choice(n, max_pool, replace=False)
        idx.sort()
        fit_t, fit_log_res = sorted_t[idx], log_res_sq[idx]
    else:
        fit_t, fit_log_res = sorted_t, log_res_sq
    X_season = _seasonal_features(fit_t, period)
    try:
        gpr_noise = GaussianProcessRegressor(kernel=_noise_kernel(), n_restarts_optimizer=3, normalize_y=True)
        gpr_noise.fit(X_season, fit_log_res)
        test_t = np.linspace(0, period, 50)
        test_var = np.exp(gpr_noise.predict(_seasonal_features(test_t, period)))
        ratio = test_var.max() / max(test_var.min(), 1e-12)
        print(f"{_TAG}   seasonal noise ratio: {ratio:.2f}x", flush=True)
    except Exception as e:
        print(f"{_TAG}   noise GP failed ({e}), uniform weight", flush=True)
        return lambda t: np.ones(len(t))

    def noise_model(t_arr: np.ndarray) -> np.ndarray:
        feat = _seasonal_features(t_arr, period)
        var = np.exp(gpr_noise.predict(feat))
        mean_var = var.mean()
        if mean_var > 0:
            var = var / mean_var
        return np.clip(var, 0.1, 10.0)

    return noise_model


# ---------------------------------------------------------------------------
# ARD kernel
# ---------------------------------------------------------------------------

def _gpr_ard_kernel(n_features: int):
    length_scale = np.ones(n_features)
    length_scale_bounds = [(1e-2, 1e3)] * n_features
    return (
        C(1.0, (1e-3, 1e3))
        * RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds)
    )


# ---------------------------------------------------------------------------
# Placeholder builder (ridge from auxiliary data)
# ---------------------------------------------------------------------------

def _build_placeholder_from_aux(
    points: List[DataPoint],
    n_aux: int,
    period: float,
    use_polynomial: bool = True,
) -> np.ndarray:
    """
    Build ridge placeholder predictions at ALL points using auxiliary data.
    Design matrix: [1, t, (t^2), sin, cos, aux1, ..., auxK, (aux*t), (aux_i*aux_j)]
    Train on observed points only, predict everywhere.
    Returns array of length len(points).
    """
    n = len(points)
    if n == 0:
        return np.array([])

    X, _ = _build_design_matrix(points, n_aux, period, use_polynomial)

    obs_mask = np.array([p.observed is not None for p in points])
    y_all = np.array([
        float(p.observed) if p.observed is not None else 0.0
        for p in points
    ], dtype=np.float64)

    n_obs = int(np.sum(obs_mask))
    min_samples = X.shape[1] + 2
    if n_obs < min_samples:
        if use_polynomial:
            return _build_placeholder_from_aux(points, n_aux, period, use_polynomial=False)
        # Linear fallback: average
        avg = float(y_all[obs_mask].mean()) if n_obs > 0 else 0.0
        result = y_all.copy()
        result[~obs_mask] = avg
        return result

    X_train = X[obs_mask]
    y_train = y_all[obs_mask]

    # Ridge regression
    alpha = 1e-2 * (5.0 if use_polynomial else 1.0)
    XtX = X_train.T @ X_train + alpha * np.eye(X_train.shape[1])
    Xty = X_train.T @ y_train
    try:
        w = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        avg = float(y_train.mean())
        result = y_all.copy()
        result[~obs_mask] = avg
        return result

    predictions = X @ w
    # Use observed values where available, placeholder elsewhere
    result = np.where(obs_mask, y_all, predictions)
    return result


def _build_design_matrix(
    points: List[DataPoint],
    n_aux: int,
    period: float,
    use_polynomial: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shared placeholder design matrix builder.
    Returns (X, times).
    """
    n = len(points)
    if n == 0:
        return np.empty((0, 0), dtype=np.float64), np.empty(0, dtype=np.float64)

    times = np.array([p.time for p in points], dtype=np.float64)
    t_min, t_max = times.min(), times.max()
    t_scale = max(t_max - t_min, 1e-10)
    time_norm = ((times - t_min) / t_scale) * 1.6 - 0.8

    features: List[np.ndarray] = [np.ones(n, dtype=np.float64), time_norm]
    if use_polynomial:
        features.append(time_norm ** 2)

    if period > 0:
        angle = 2.0 * np.pi * times / period
        features.append(np.sin(angle))
        features.append(np.cos(angle))

    aux_cols: List[np.ndarray] = []
    for j in range(n_aux):
        col = np.fromiter(
            (
                float(p.auxiliaries[j])
                if p.auxiliaries and j < len(p.auxiliaries)
                else 0.0
                for p in points
            ),
            dtype=np.float64,
            count=n,
        )
        features.append(col)
        aux_cols.append(col)

    if use_polynomial and aux_cols:
        for col in aux_cols:
            features.append(col * time_norm)
        if len(aux_cols) >= 2:
            count = 0
            for a in range(len(aux_cols)):
                for b in range(a + 1, len(aux_cols)):
                    features.append(aux_cols[a] * aux_cols[b])
                    count += 1
                    if count >= 10:
                        break
                if count >= 10:
                    break

    return np.column_stack(features), times


# ---------------------------------------------------------------------------
# Locally-weighted placeholder for large gaps
# ---------------------------------------------------------------------------

def _build_local_placeholder_from_aux(
    points: List[DataPoint],
    n_aux: int,
    period: float,
    large_gap_threshold: int = 10,
    use_polynomial: bool = True,
) -> Tuple[np.ndarray, List[bool]]:
    """
    Build placeholder with locally-weighted ridge for large-gap regions.

    For each contiguous large gap (> large_gap_threshold consecutive missing),
    fit a weighted ridge where observed points closer to the gap center get
    higher weight.  Small-gap points keep the global placeholder.

    Returns (placeholder_array, in_large_gap_mask).
    """
    n = len(points)
    if n == 0:
        return np.array([]), []

    # Global placeholder as baseline
    global_ph = _build_placeholder_from_aux(points, n_aux, period, use_polynomial)

    # Identify contiguous large gaps
    in_large_gap: List[bool] = [False] * n
    obs_mask = [p.observed is not None or p.imputed is not None for p in points]

    gap_start: Optional[int] = None
    for i in range(n):
        if not obs_mask[i]:
            if gap_start is None:
                gap_start = i
        else:
            if gap_start is not None:
                if i - gap_start > large_gap_threshold:
                    for j in range(gap_start, i):
                        in_large_gap[j] = True
                gap_start = None
    if gap_start is not None and n - gap_start > large_gap_threshold:
        for j in range(gap_start, n):
            in_large_gap[j] = True

    if not any(in_large_gap):
        return global_ph, in_large_gap

    X, times = _build_design_matrix(points, n_aux, period, use_polynomial)
    t_min, t_max = times.min(), times.max()
    t_scale = max(t_max - t_min, 1e-10)
    obs_indices = np.array([i for i in range(n) if obs_mask[i]])
    y_obs = np.array([
        float(points[i].observed) if points[i].observed is not None
        else float(points[i].imputed)
        for i in obs_indices
    ], dtype=np.float64)

    if len(obs_indices) < X.shape[1] + 2:
        return global_ph, in_large_gap

    X_obs = X[obs_indices]
    obs_times = times[obs_indices]
    result = global_ph.copy()

    # Identify contiguous large-gap blocks
    gap_blocks: List[Tuple[int, int]] = []
    block_start: Optional[int] = None
    for i in range(n):
        if in_large_gap[i]:
            if block_start is None:
                block_start = i
        else:
            if block_start is not None:
                gap_blocks.append((block_start, i))
                block_start = None
    if block_start is not None:
        gap_blocks.append((block_start, n))

    for gap_s, gap_e in gap_blocks:
        gap_center_time = (times[gap_s] + times[gap_e - 1]) / 2.0
        distances = np.abs(obs_times - gap_center_time)
        dist_scale = t_scale * 0.2  # 20% of total range
        weights = 1.0 / (1.0 + distances / max(dist_scale, 1e-10))

        alpha_reg = 1e-2 * (5.0 if use_polynomial else 1.0)
        sqrt_w = np.sqrt(weights)
        Xw = X_obs * sqrt_w[:, None]
        yw = y_obs * sqrt_w
        XtWX = Xw.T @ Xw + alpha_reg * np.eye(X.shape[1])
        XtWy = Xw.T @ yw
        try:
            w = np.linalg.solve(XtWX, XtWy)
            result[gap_s:gap_e] = X[gap_s:gap_e] @ w
        except np.linalg.LinAlgError:
            pass  # keep global placeholder

    n_large = sum(in_large_gap)
    print(f"{_TAG}   local placeholder: {len(gap_blocks)} large-gap blocks, {n_large} pts upgraded", flush=True)
    return result, in_large_gap


def _compute_distance_to_nearest_obs(
    points: List[DataPoint],
) -> np.ndarray:
    """For each point, compute temporal distance to nearest observed point."""
    n = len(points)
    times = np.array([p.time for p in points], dtype=np.float64)
    obs_times = np.array([
        p.time for p in points
        if p.observed is not None or p.imputed is not None
    ], dtype=np.float64)

    if len(obs_times) == 0:
        return np.ones(n)

    obs_times = np.sort(obs_times)
    insert_pos = np.searchsorted(obs_times, times, side="left")
    left_idx = np.clip(insert_pos - 1, 0, len(obs_times) - 1)
    right_idx = np.clip(insert_pos, 0, len(obs_times) - 1)
    left_dist = np.abs(times - obs_times[left_idx])
    right_dist = np.abs(obs_times[right_idx] - times)
    return np.minimum(left_dist, right_dist)


# ---------------------------------------------------------------------------
# Enhanced feature builder for GP
# ---------------------------------------------------------------------------

def _build_enhanced_features(
    time_val: float,
    aux: List[float],
    n_aux: int,
    period: float,
    use_polynomial: bool,
) -> List[float]:
    """Build feature vector for GP.
    Basic: [time, sin, cos, aux1, ..., auxK]
    Polynomial: + [t^2, aux1*t, ..., auxK*t, aux1*aux2, ...]
    """
    features: List[float] = [time_val]
    if period > 0:
        angle = 2.0 * np.pi * time_val / period
        features.extend([np.sin(angle), np.cos(angle)])

    aux_vals = aux[:n_aux] if len(aux) >= n_aux else aux + [0.0] * (n_aux - len(aux))
    features.extend(aux_vals)

    if use_polynomial:
        features.append(time_val * time_val)  # t^2
        for a in aux_vals:
            features.append(a * time_val)  # aux_j * t
        if len(aux_vals) >= 2:
            count = 0
            for i in range(len(aux_vals)):
                for j in range(i + 1, len(aux_vals)):
                    features.append(aux_vals[i] * aux_vals[j])
                    count += 1
                    if count >= 10:
                        break
                if count >= 10:
                    break

    return features


def _feature_labels(n_aux: int, period: float, use_polynomial: bool) -> List[str]:
    labels = ['time']
    if period > 0:
        labels.extend(['sin_season', 'cos_season'])
    labels.extend([f'aux{j+1}' for j in range(n_aux)])
    if use_polynomial:
        labels.append('t^2')
        labels.extend([f'aux{j+1}*t' for j in range(n_aux)])
        if n_aux >= 2:
            count = 0
            for i in range(n_aux):
                for j in range(i + 1, n_aux):
                    labels.append(f'aux{i+1}*aux{j+1}')
                    count += 1
                    if count >= 10:
                        break
                if count >= 10:
                    break
    return labels


def _build_enhanced_feature_matrix(
    points: List[DataPoint],
    n_aux: int,
    period: float,
    use_polynomial: bool,
) -> np.ndarray:
    """Vectorized GP feature matrix for all points in an instance."""
    n = len(points)
    if n == 0:
        return np.empty((0, 0), dtype=np.float64)

    times = np.array([p.time for p in points], dtype=np.float64)
    features: List[np.ndarray] = [times]

    if period > 0:
        angle = 2.0 * np.pi * times / period
        features.extend([np.sin(angle), np.cos(angle)])

    aux_cols: List[np.ndarray] = []
    for j in range(n_aux):
        col = np.fromiter(
            (
                float(p.auxiliaries[j])
                if p.auxiliaries and j < len(p.auxiliaries)
                else 0.0
                for p in points
            ),
            dtype=np.float64,
            count=n,
        )
        features.append(col)
        aux_cols.append(col)

    if use_polynomial:
        features.append(times * times)
        for col in aux_cols:
            features.append(col * times)
        if len(aux_cols) >= 2:
            count = 0
            for i in range(len(aux_cols)):
                for j in range(i + 1, len(aux_cols)):
                    features.append(aux_cols[i] * aux_cols[j])
                    count += 1
                    if count >= 10:
                        break
                if count >= 10:
                    break

    return np.column_stack(features)


# ---------------------------------------------------------------------------
# Linear interpolation fallback
# ---------------------------------------------------------------------------

def _local_interp(t: float, t_known: np.ndarray, z_known: np.ndarray) -> float:
    idx = np.searchsorted(t_known, t)
    if idx == 0:
        return float(z_known[0])
    if idx >= len(t_known):
        return float(z_known[-1])
    t0, t1 = t_known[idx - 1], t_known[idx]
    z0, z1 = z_known[idx - 1], z_known[idx]
    if abs(t1 - t0) < 1e-12:
        return float((z0 + z1) / 2)
    frac = (t - t0) / (t1 - t0)
    return float(z0 + frac * (z1 - z0))


# ---------------------------------------------------------------------------
# Per-instance GP imputation with placeholder training
# ---------------------------------------------------------------------------

def _gp_aux_enhanced_impute_instance(
    points: List[DataPoint],
    n_aux: int,
    period: float = 0,
    noise_model: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    use_polynomial: bool = True,
    pseudo_weight: float = 1.0,
    large_gap_threshold: int = 10,
) -> List[DataPoint]:
    """
    GPR with auxiliary placeholder training — improved for large gaps.

    1. Build locally-weighted placeholder for large-gap points (global for small gaps)
    2. Train GP on observed + pseudo-labels with distance-aware noise: points deep
       in large gaps get much higher noise (less trust) than near-boundary points
    3. At gap edges, smoothly blend GP prediction toward observed neighbors
    """
    inst_id = points[0].instance_id if points else "?"
    use_seasonal = period > 0 and noise_model is not None

    # Step 1: Build placeholder — locally-weighted for large gaps
    placeholder, in_large_gap = _build_local_placeholder_from_aux(
        points, n_aux, period, large_gap_threshold, use_polynomial,
    )
    if len(placeholder) == 0:
        placeholder = np.zeros(len(points))

    # Compute distance-to-nearest-observed for noise scaling
    dist_to_obs = _compute_distance_to_nearest_obs(points)
    # Compute a sensible distance scale: median inter-observation spacing
    obs_times_sorted = np.sort([
        p.time for p in points if p.observed is not None or p.imputed is not None
    ])
    if len(obs_times_sorted) >= 2:
        spacing = np.diff(obs_times_sorted)
        dist_scale = float(np.median(spacing)) * 2.0
    else:
        dist_scale = 1.0
    dist_scale = max(dist_scale, 1e-10)

    # Step 2: Collect features and training data (observed + placeholder)
    all_features = _build_enhanced_feature_matrix(points, n_aux, period, use_polynomial)
    train_indices: List[int] = []
    train_y: List[float] = []
    train_is_observed: List[bool] = []
    train_times: List[float] = []
    train_dist_to_obs: List[float] = []
    train_in_large_gap: List[bool] = []
    missing_indices: List[int] = []

    for i, p in enumerate(points):
        if p.observed is not None:
            train_indices.append(i)
            train_y.append(float(p.observed))
            train_is_observed.append(True)
            train_times.append(p.time)
            train_dist_to_obs.append(0.0)
            train_in_large_gap.append(False)
        elif p.imputed is not None:
            train_indices.append(i)
            train_y.append(float(p.imputed))
            train_is_observed.append(True)
            train_times.append(p.time)
            train_dist_to_obs.append(0.0)
            train_in_large_gap.append(False)
        else:
            if pseudo_weight > 0:
                train_indices.append(i)
                train_y.append(float(placeholder[i]))
                train_is_observed.append(False)
                train_times.append(p.time)
                train_dist_to_obs.append(float(dist_to_obs[i]))
                train_in_large_gap.append(in_large_gap[i])
            missing_indices.append(i)

    n_train = len(train_indices)
    n_miss = len(missing_indices)
    n_obs_train = sum(1 for x in train_is_observed if x)
    n_pseudo = n_train - n_obs_train
    n_feat = all_features.shape[1] if all_features.size else 0
    n_large_gap_pseudo = sum(1 for lg in train_in_large_gap if lg)

    poly_str = "poly" if use_polynomial else "basic"
    mode_str = f"seasonal(P={period})+{poly_str}" if use_seasonal else poly_str
    print(f"{_TAG}   '{inst_id}': total={len(points)}, train={n_train} (obs={n_obs_train}, "
          f"pseudo={n_pseudo}, large_gap_pseudo={n_large_gap_pseudo}), "
          f"missing={n_miss}, features={n_feat}, mode={mode_str}", flush=True)

    if n_miss == 0:
        return [replace(p, imputed=p.observed if p.observed is not None else p.imputed, imputed_std=None) for p in points]

    if n_obs_train < 2:
        avg_val = float(np.mean([y for y, is_obs in zip(train_y, train_is_observed) if is_obs])) if n_obs_train > 0 else 0.0
        results = []
        for i, p in enumerate(points):
            if p.observed is not None:
                results.append(replace(p, imputed=p.observed, imputed_std=None))
            elif p.imputed is not None:
                results.append(replace(p, imputed=p.imputed, imputed_std=None))
            else:
                results.append(replace(p, imputed=float(placeholder[i]), imputed_std=None))
        return results

    X_train = all_features[train_indices]
    y_train = np.array(train_y, dtype=np.float64)
    X_predict = all_features[missing_indices]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_predict_scaled = scaler.transform(X_predict)

    # Step 3: Distance-aware alpha (noise) for GP training
    # Observed: base_noise (high trust)
    # Pseudo-labels near observed data: moderate noise
    # Pseudo-labels deep in large gaps: moderate noise (placeholder is our best estimate)
    base_noise = 1e-2
    is_obs_arr = np.array(train_is_observed, dtype=bool)
    dist_arr = np.array(train_dist_to_obs, dtype=np.float64)
    large_gap_arr = np.array(train_in_large_gap, dtype=bool)

    if use_seasonal:
        seasonal_weights = noise_model(np.array(train_times, dtype=np.float64))
        alpha_train = base_noise * seasonal_weights
    else:
        alpha_train = np.full(n_train, base_noise)

    # For pseudo-labels: scale noise by distance to nearest observed point
    # Use a CAPPED distance factor so the GP still trusts the placeholder in large gaps
    # (prevents wild extrapolation/divergence away from the observed range)
    max_dist_factor = 5.0  # cap: even deepest pseudo-labels get at most 5x noise
    for k in range(n_train):
        if not is_obs_arr[k]:
            dist_factor = 1.0 + dist_arr[k] / dist_scale
            if large_gap_arr[k]:
                # Moderate penalty for large-gap pseudo-labels (reduced from 3.0)
                dist_factor *= 1.5
            dist_factor = min(dist_factor, max_dist_factor)
            alpha_train[k] = alpha_train[k] * dist_factor / max(pseudo_weight, 1e-6)

    alpha_train = np.maximum(alpha_train, 1e-10)

    alpha_obs = alpha_train[is_obs_arr]
    alpha_pseudo = alpha_train[~is_obs_arr]
    print(f"{_TAG}   '{inst_id}': alpha obs=[{alpha_obs.min():.6f}, {alpha_obs.max():.6f}], "
          f"pseudo=[{alpha_pseudo.min():.6f}, {alpha_pseudo.max():.6f}]" if len(alpha_pseudo) > 0
          else f"{_TAG}   '{inst_id}': alpha obs=[{alpha_obs.min():.6f}, {alpha_obs.max():.6f}], no pseudo",
          flush=True)

    krig_pred = np.zeros(n_miss, dtype=np.float64)
    imputed_std_arr = np.zeros(n_miss, dtype=np.float64)

    try:
        n_restarts = _gp_optimizer_restarts(n_train, n_feat)
        print(f"{_TAG}   '{inst_id}': GP optimizer restarts={n_restarts}", flush=True)
        gpr = GaussianProcessRegressor(
            kernel=_gpr_ard_kernel(n_feat),
            alpha=alpha_train,
            n_restarts_optimizer=n_restarts,
            normalize_y=True,
        )
        gpr.fit(X_train_scaled, y_train)
        krig_pred, imputed_std_arr = gpr.predict(X_predict_scaled, return_std=True)
        krig_pred = np.asarray(krig_pred, dtype=np.float64)
        imputed_std_arr = np.maximum(np.asarray(imputed_std_arr, dtype=np.float64), 0.0)

        # Report ARD length scales
        learned_kernel = gpr.kernel_
        for k in learned_kernel.get_params().values():
            if hasattr(k, 'length_scale') and np.ndim(k.length_scale) > 0:
                ls = k.length_scale
                labels = _feature_labels(n_aux, period, use_polynomial)
                ls_str = ', '.join(f'{labels[j]}={ls[j]:.3f}' for j in range(min(len(ls), len(labels))))
                print(f"{_TAG}   '{inst_id}': ARD length scales: {ls_str}", flush=True)
                break

        print(f"{_TAG}   '{inst_id}': predicted {len(krig_pred)} pts, range=[{krig_pred.min():.4f}, {krig_pred.max():.4f}]", flush=True)
        print(f"{_TAG}   '{inst_id}': uncertainty avg_std={imputed_std_arr.mean():.4f}, max_std={imputed_std_arr.max():.4f}", flush=True)
    except Exception as e:
        print(f"{_TAG}   '{inst_id}': GPR failed ({e}), placeholder fallback", flush=True)
        for j, idx in enumerate(missing_indices):
            krig_pred[j] = float(placeholder[idx])
        imputed_std_arr[:] = 0.0

    # Step 4: Compute observed-range bounds for deviation detection
    obs_values = [float(p.observed) for p in points if p.observed is not None]
    if not obs_values:
        obs_values = [float(p.imputed) for p in points if p.imputed is not None]
    if obs_values:
        obs_min = min(obs_values)
        obs_max = max(obs_values)
        obs_range = max(obs_max - obs_min, 1e-10)
        obs_mean = np.mean(obs_values)
    else:
        obs_min, obs_max, obs_range, obs_mean = 0.0, 0.0, 1.0, 0.0

    # Step 5: Build result with uncertainty-aware GP↔placeholder blending
    # Strategy: In large gaps, blend GP toward placeholder proportional to how
    # far the GP deviates from the observed range. This preserves the placeholder's
    # aux-driven shape instead of producing flat clamped lines.
    # At edges, additionally blend toward the nearest observed neighbor.
    missing_set = set(missing_indices)
    results: List[DataPoint] = []
    miss_ptr = 0
    n_blended = 0

    # Pre-compute per-gap-block parameters
    gap_blocks: List[Tuple[int, int]] = []
    block_start_idx: Optional[int] = None
    for i in range(len(in_large_gap)):
        if in_large_gap[i]:
            if block_start_idx is None:
                block_start_idx = i
        else:
            if block_start_idx is not None:
                gap_blocks.append((block_start_idx, i))
                block_start_idx = None
    if block_start_idx is not None:
        gap_blocks.append((block_start_idx, len(in_large_gap)))

    # Map each large-gap index to its block info (edge_width, gap_size)
    gap_info_map: Dict[int, Tuple[int, int]] = {}  # idx -> (edge_width, gap_size)
    for gs, ge in gap_blocks:
        gap_size = ge - gs
        ew = max(3, min(gap_size // 5, 15))
        for idx in range(gs, ge):
            gap_info_map[idx] = (ew, gap_size)

    for i, p in enumerate(points):
        if p.observed is not None:
            results.append(replace(p, imputed=p.observed, imputed_std=None))
        elif i in missing_set:
            gp_val = float(krig_pred[miss_ptr])
            ph_val = float(placeholder[i])
            std_val = float(imputed_std_arr[miss_ptr]) if imputed_std_arr[miss_ptr] > 0 else None

            if in_large_gap[i]:
                edge_width, gap_size = gap_info_map.get(i, (3, 10))

                # 1. Edge blending: near gap boundaries, blend toward placeholder
                dist_to_edge = min(
                    abs(i - next((j for j in range(i, -1, -1) if not in_large_gap[j]), i)),
                    abs(i - next((j for j in range(i, len(in_large_gap)) if not in_large_gap[j]), i)),
                )
                if dist_to_edge <= edge_width and dist_to_edge > 0:
                    blend = dist_to_edge / edge_width  # 0=boundary, 1=deep
                    gp_val = blend * gp_val + (1.0 - blend) * ph_val

                # 2. Deviation-aware blending: if GP deviates far from observed range,
                #    blend toward placeholder to preserve aux-driven shape
                deviation = 0.0
                if gp_val < obs_min:
                    deviation = (obs_min - gp_val) / obs_range
                elif gp_val > obs_max:
                    deviation = (gp_val - obs_max) / obs_range

                if deviation > 0.0:
                    # Sigmoid-like blend: small deviation -> mostly GP,
                    # large deviation -> mostly placeholder
                    # At deviation=0.5 (50% of obs range), blend is ~0.5
                    ph_weight = min(deviation / 0.5, 1.0)
                    # For very large gaps, be more aggressive with blending
                    if gap_size > 40:
                        ph_weight = min(deviation / 0.3, 1.0)
                    gp_val = (1.0 - ph_weight) * gp_val + ph_weight * ph_val
                    n_blended += 1

                    # Final safety clamp: don't allow more than 100% beyond observed range
                    hard_margin = max(obs_range * 1.0, 2.0)
                    gp_val = float(np.clip(gp_val, obs_min - hard_margin, obs_max + hard_margin))

            results.append(replace(p, imputed=gp_val, imputed_std=std_val))
            miss_ptr += 1
        else:
            results.append(replace(p, imputed=p.imputed, imputed_std=None))

    if n_blended > 0:
        print(f"{_TAG}   '{inst_id}': blended {n_blended} predictions toward placeholder (deviation from obs range)", flush=True)

    return results


# ---------------------------------------------------------------------------
# LOO KGE for quality check
# ---------------------------------------------------------------------------

def _gp_aux_enhanced_loo_obs_pred(
    points: List[DataPoint],
    n_aux: int,
    period: float = 0,
    noise_model: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    use_polynomial: bool = True,
    pseudo_weight: float = 1.0,
) -> Tuple[List[float], List[float]]:
    """In-sample KGE approximation using the full enhanced GP with distance-aware noise."""
    placeholder, in_large_gap = _build_local_placeholder_from_aux(
        points, n_aux, period, large_gap_threshold=10, use_polynomial=use_polynomial,
    )
    if len(placeholder) == 0:
        placeholder = np.zeros(len(points))

    dist_to_obs = _compute_distance_to_nearest_obs(points)
    obs_times_sorted = np.sort([
        p.time for p in points if p.observed is not None or p.imputed is not None
    ])
    if len(obs_times_sorted) >= 2:
        dist_scale = float(np.median(np.diff(obs_times_sorted))) * 2.0
    else:
        dist_scale = 1.0
    dist_scale = max(dist_scale, 1e-10)

    all_features = _build_enhanced_feature_matrix(points, n_aux, period, use_polynomial)
    train_y: List[float] = []
    train_is_observed: List[bool] = []
    train_times: List[float] = []
    train_dist: List[float] = []
    train_in_large_gap: List[bool] = []
    obs_indices: List[int] = []

    for i, p in enumerate(points):
        if p.observed is not None:
            train_y.append(float(p.observed))
            train_is_observed.append(True)
            train_times.append(p.time)
            train_dist.append(0.0)
            train_in_large_gap.append(False)
            obs_indices.append(i)
        elif p.imputed is not None:
            train_y.append(float(p.imputed))
            train_is_observed.append(True)
            train_times.append(p.time)
            train_dist.append(0.0)
            train_in_large_gap.append(False)
        else:
            if pseudo_weight > 0:
                train_y.append(float(placeholder[i]))
                train_is_observed.append(False)
                train_times.append(p.time)
                train_dist.append(float(dist_to_obs[i]))
                train_in_large_gap.append(in_large_gap[i])

    if len(train_y) < 2 or len(obs_indices) < 2:
        return [], []

    train_idx = []
    for i, p in enumerate(points):
        if p.observed is not None or p.imputed is not None:
            train_idx.append(i)
        elif pseudo_weight > 0:
            train_idx.append(i)

    X_train = all_features[train_idx]
    y_train_arr = np.array(train_y, dtype=np.float64)
    X_obs = all_features[obs_indices]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_obs_scaled = scaler.transform(X_obs)
    n_feat = X_train.shape[1]
    n_train = len(train_y)

    base_noise = 1e-2
    is_obs_arr = np.array(train_is_observed, dtype=bool)
    dist_arr = np.array(train_dist, dtype=np.float64)
    lg_arr = np.array(train_in_large_gap, dtype=bool)

    alpha_train = np.full(n_train, base_noise)
    for k in range(n_train):
        if not is_obs_arr[k]:
            dist_factor = 1.0 + dist_arr[k] / dist_scale
            if lg_arr[k]:
                dist_factor *= 3.0
            alpha_train[k] = alpha_train[k] * dist_factor / max(pseudo_weight, 1e-6)
    alpha_train = np.maximum(alpha_train, 1e-10)

    try:
        n_restarts = min(2, _gp_optimizer_restarts(n_train, n_feat))
        gpr = GaussianProcessRegressor(
            kernel=_gpr_ard_kernel(n_feat),
            alpha=alpha_train,
            n_restarts_optimizer=n_restarts,
            normalize_y=True,
        )
        gpr.fit(X_train_scaled, y_train_arr)
        pred = gpr.predict(X_obs_scaled)
    except Exception:
        return [], []

    obs_y = [float(points[i].observed) for i in obs_indices]
    return obs_y, pred.tolist()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_gp_aux_enhanced_simulation(
    data: List[DataPoint],
    params: Dict[str, Any],
) -> List[DataPoint]:
    """GP Aux Enhanced: train on observed + placeholder pseudo-labels."""
    period = params.get("seasonal_period", 0)
    use_polynomial = params.get("use_polynomial", True)
    pseudo_weight = params.get("pseudo_weight", 1.0)

    instance_map: Dict[str, List[DataPoint]] = defaultdict(list)
    for d in data:
        key = d.instance_id or "__default__"
        instance_map[key].append(d)

    n_aux = 0
    for d in data:
        if d.auxiliaries and len(d.auxiliaries) > 0:
            n_aux = len(d.auxiliaries)
            break

    if n_aux == 0:
        print(f"{_TAG} WARNING: no auxiliary data found -- falling back to time-only GP", flush=True)

    print(f"{_TAG} run_gp_aux_enhanced_simulation: n_aux={n_aux}, instances={len(instance_map)}, "
          f"period={period}, polynomial={use_polynomial}, pseudo_weight={pseudo_weight}", flush=True)

    # Stage 1: seasonal noise model
    noise_model: Optional[Callable[[np.ndarray], np.ndarray]] = None
    if period > 0:
        all_known_t, all_known_z = [], []
        for pts in instance_map.values():
            for p in pts:
                if p.observed is not None:
                    all_known_t.append(p.time)
                    all_known_z.append(float(p.observed))
                elif p.imputed is not None:
                    all_known_t.append(p.time)
                    all_known_z.append(float(p.imputed))
        noise_model = _learn_seasonal_noise(
            np.array(all_known_t, dtype=np.float64),
            np.array(all_known_z, dtype=np.float64),
            period,
        )

    def _known_count(pts):
        return sum(1 for p in pts if p.observed is not None or p.imputed is not None)

    instance_order = sorted(instance_map.keys(), key=lambda iid: _known_count(instance_map[iid]), reverse=True)

    final_results: List[DataPoint] = []
    instances_ok = 0
    instances_fallback = 0

    for inst_id in instance_order:
        points = sorted(instance_map[inst_id], key=lambda p: p.time)
        n_miss = sum(1 for p in points if p.observed is None and p.imputed is None)

        has_aux = any(p.auxiliaries and len(p.auxiliaries) >= n_aux and n_aux > 0 for p in points)
        if not has_aux and n_aux > 0:
            print(f"{_TAG}   '{inst_id}': skipped (no auxiliary data)", flush=True)
            for p in points:
                val = p.observed if p.observed is not None else p.imputed
                final_results.append(replace(p, imputed=val, imputed_std=None))
            continue

        if n_miss == 0:
            for p in points:
                val = p.observed if p.observed is not None else p.imputed
                final_results.append(replace(p, imputed=val, imputed_std=None))
            instances_ok += 1
            continue

        try:
            imputed_points = _gp_aux_enhanced_impute_instance(
                points, n_aux, period=period, noise_model=noise_model,
                use_polynomial=use_polynomial, pseudo_weight=pseudo_weight,
            )
            final_results.extend(imputed_points)
            instances_ok += 1
        except Exception as e:
            instances_fallback += 1
            print(f"{_TAG}   '{inst_id}': FAILED ({type(e).__name__}: {e}), linear fallback", flush=True)
            knowns_t, knowns_z = [], []
            for p in points:
                if p.observed is not None:
                    knowns_t.append(p.time)
                    knowns_z.append(float(p.observed))
                elif p.imputed is not None:
                    knowns_t.append(p.time)
                    knowns_z.append(float(p.imputed))
            t_known = np.array(knowns_t, dtype=np.float64) if knowns_t else np.array([])
            z_known = np.array(knowns_z, dtype=np.float64) if knowns_z else np.array([])
            avg_val = float(z_known.mean()) if len(z_known) > 0 else 0.0
            for p in points:
                if p.observed is not None:
                    final_results.append(replace(p, imputed=p.observed, imputed_std=None))
                elif p.imputed is not None:
                    final_results.append(replace(p, imputed=p.imputed, imputed_std=None))
                elif len(t_known) >= 2:
                    val = _local_interp(p.time, t_known, z_known)
                    final_results.append(replace(p, imputed=val, imputed_std=None))
                else:
                    final_results.append(replace(p, imputed=avg_val, imputed_std=None))

    n_filled = sum(1 for p in final_results if p.observed is None and p.imputed is not None)
    print(f"{_TAG} done. ok={instances_ok}, fallback={instances_fallback}, filled={n_filled}", flush=True)
    return final_results


# ---------------------------------------------------------------------------
# Batch entry point (small-gap only)
# ---------------------------------------------------------------------------

def batch_impute_small_gap_gp_aux_enhanced(
    flat_data: List[DataPoint],
    params: SimulationParams,
    on_progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """Batch GP+Aux Enhanced for small gaps only. Requires auxiliary data."""
    print(f"{_TAG} batch start (GP+Aux Enhanced, small-gap only)", flush=True)
    max_gap_threshold = getattr(params, "max_gap_threshold", 10) or 10
    period = getattr(params, "geostat_seasonal_period", 0) or 0
    use_polynomial = getattr(params, "gp_aux_enhanced_polynomial", True)
    pseudo_weight = getattr(params, "gp_aux_enhanced_pseudo_weight", 1.0)

    n_aux = 0
    for d in flat_data:
        if d.auxiliaries and len(d.auxiliaries) > 0:
            n_aux = len(d.auxiliaries)
            break

    print(f"{_TAG}   flat_data={len(flat_data)}, max_gap={max_gap_threshold}, n_aux={n_aux}, "
          f"period={period}, polynomial={use_polynomial}, pseudo_weight={pseudo_weight}", flush=True)

    if n_aux == 0:
        print(f"{_TAG}   WARNING: no auxiliary data found", flush=True)

    gp_params: Dict[str, Any] = {
        "seasonal_period": period,
        "use_polynomial": use_polynomial,
        "pseudo_weight": pseudo_weight,
    }

    if on_progress:
        on_progress(1, 1, "gp-aux-enhanced")
    filled_list = run_gp_aux_enhanced_simulation(flat_data, gp_params)

    # Restrict to small-gap only
    input_groups_by_inst: Dict[str, List[DataPoint]] = defaultdict(list)
    for d in flat_data:
        key = d.instance_id or "__default__"
        input_groups_by_inst[key].append(d)
    filled_groups: Dict[str, List[DataPoint]] = defaultdict(list)
    for d in filled_list:
        key = d.instance_id or "__default__"
        filled_groups[key].append(d)

    def _known_count(pts):
        return sum(1 for p in pts if p.observed is not None or p.imputed is not None)

    instance_order = sorted(input_groups_by_inst.keys(), key=lambda iid: _known_count(input_groups_by_inst[iid]), reverse=True)
    for inst_id in instance_order:
        input_pts = input_groups_by_inst[inst_id]
        points_sorted = sorted(input_pts, key=lambda p: p.time)
        filled_pts = filled_groups.get(inst_id, [])
        if len(filled_pts) != len(points_sorted):
            continue
        small_gap_indices = _small_gap_only_indices(points_sorted, max_gap_threshold)
        for i, inp in enumerate(points_sorted):
            if inp.observed is None and inp.imputed is None:
                if i not in small_gap_indices and i < len(filled_pts):
                    filled_pts[i] = replace(filled_pts[i], imputed=None, imputed_std=None)
    filled_list = []
    for inst_id in instance_order:
        filled_list.extend(filled_groups.get(inst_id, []))

    # Group filled results by instance
    groups: Dict[str, List[DataPoint]] = defaultdict(list)
    for d in filled_list:
        if d.instance_id:
            groups[d.instance_id].append(d)

    input_groups: Dict[str, List[DataPoint]] = defaultdict(list)
    for d in flat_data:
        if d.instance_id:
            input_groups[d.instance_id].append(d)

    # Rebuild noise model for KGE evaluation
    noise_model: Optional[Callable[[np.ndarray], np.ndarray]] = None
    if period > 0:
        all_known_t, all_known_z = [], []
        for pts in input_groups.values():
            for p in pts:
                if p.observed is not None:
                    all_known_t.append(p.time)
                    all_known_z.append(float(p.observed))
                elif p.imputed is not None:
                    all_known_t.append(p.time)
                    all_known_z.append(float(p.imputed))
        noise_model = _learn_seasonal_noise(
            np.array(all_known_t, dtype=np.float64) if all_known_t else np.array([]),
            np.array(all_known_z, dtype=np.float64) if all_known_z else np.array([]),
            period,
        )

    kge_threshold = getattr(params, "small_gap_kge_threshold", None) or params.kge_threshold
    results: Dict[str, Dict[str, Any]] = {}
    saved_count = 0
    skipped_count = 0

    print(f"{_TAG} per-instance KGE (threshold={kge_threshold:.4f})", flush=True)
    for instance_id, pts in groups.items():
        input_pts = sorted(input_groups.get(instance_id, []), key=lambda p: p.time)
        obs_list, pred_list = [], []
        if len(input_pts) >= 2:
            obs_list, pred_list = _gp_aux_enhanced_loo_obs_pred(
                input_pts, n_aux, period=period, noise_model=noise_model,
                use_polynomial=use_polynomial, pseudo_weight=pseudo_weight,
            )
        kge = float("-inf")
        if obs_list and len(obs_list) >= 2:
            kge = calculate_kge(obs_list, pred_list)
        saved = kge >= kge_threshold
        results[instance_id] = {"kge": kge, "saved": saved}
        n_filled = sum(1 for d in pts if d.observed is None and d.imputed is not None)
        print(f"{_TAG}   '{instance_id}': LOO pairs={len(obs_list)}, filled={n_filled}, KGE={kge:.4f}, saved={saved}", flush=True)
        if saved:
            saved_count += 1
        else:
            skipped_count += 1

    print(f"{_TAG} batch done: instances={len(groups)}, saved={saved_count}, skipped={skipped_count}", flush=True)
    return {
        "imputed": len(groups),
        "saved": saved_count,
        "skipped": skipped_count,
        "results": results,
        "filled_data": filled_list,
    }


def _small_gap_only_indices(points_sorted: List[DataPoint], max_gap_threshold: int) -> set:
    gaps, _ = identify_gaps(points_sorted, max_gap_threshold)
    small = set()
    for g in gaps:
        if g["size"] <= max_gap_threshold:
            for i in range(g["startIdx"], g["endIdx"] + 1):
                small.add(i)
    return small


# ---------------------------------------------------------------------------
# Single-instance entry point (for large-gap / auxiliary-only mode)
# ---------------------------------------------------------------------------

def single_impute_gp_aux_enhanced(
    flat_data: List[DataPoint],
    params: SimulationParams,
) -> List[DataPoint]:
    """GP Aux Enhanced on a single instance — impute every missing point (no large-gap masking).
    Used by the large-gap / auxiliary-only pipeline."""
    period = getattr(params, "geostat_seasonal_period", 0) or 0
    use_polynomial = getattr(params, "gp_aux_enhanced_polynomial", True)
    pseudo_weight = getattr(params, "gp_aux_enhanced_pseudo_weight", 1.0)

    gp_params: Dict[str, Any] = {
        "seasonal_period": period,
        "use_polynomial": use_polynomial,
        "pseudo_weight": pseudo_weight,
    }
    return run_gp_aux_enhanced_simulation(flat_data, gp_params)
