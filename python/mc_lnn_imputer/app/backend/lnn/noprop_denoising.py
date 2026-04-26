"""
NoProp Denoising — standalone iterative denoising imputation algorithm.
Inspired by the NoProp paper (arxiv:2503.24322): block-wise denoising without backpropagation.

Completely independent — no LNN reservoir. Builds features from time + auxiliary data,
then iteratively denoises through decreasing noise levels with ridge regression.
Multiple realizations provide uncertainty estimates.
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict
from math import sin, cos, pi

import numpy as np

from .types import DataPoint, SimulationParams
from .math_utils import (
    ridge_regression,
    normalize_value,
    denormalize_value,
    get_scaler,
    calculate_kge,
)
from .gaps import identify_gaps


def run_noprop_denoising_simulation(
    data: List[DataPoint],
    params: SimulationParams,
    rng: Optional[np.random.Generator] = None,
) -> List[DataPoint]:
    """
    Standalone NoProp Denoising for a single instance.

    Features: [bias, time_norm, sin(2*pi*t), cos(2*pi*t), aux_1_norm, ..., aux_k_norm]
    Algorithm:
      1. Ridge baseline on observed → predict at all points
      2. For each noise level sigma (high→low):
         - Augment features: [base_features, current_pred]
         - Train denoising ridge: augmented → noisy_observed
         - Predict at all points
      3. Average N realizations → mean + std (uncertainty)

    Returns list of DataPoint with imputed and imputed_std filled.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(data)
    if n == 0:
        return []

    # --- 1. Extract observed data ---
    obs_indices = [i for i in range(n) if data[i].observed is not None]
    if len(obs_indices) < 2:
        # Not enough observed data — return as-is
        return data

    obs_values = [data[i].observed for i in obs_indices]
    obs_min, obs_max = min(obs_values), max(obs_values)
    if obs_max == obs_min:
        obs_max = obs_min + 1e-6

    # Normalize observed to [-0.8, 0.8] (same as rest of codebase)
    y = np.array([normalize_value(v, obs_min, obs_max) for v in obs_values])
    y_std = max(float(np.std(y)), 1e-6)

    # --- 2. Build feature matrix ---
    times = [d.time for d in data]
    t_min, t_max = min(times), max(times)
    t_range = t_max - t_min
    if t_range == 0:
        t_range = 1.0

    # Detect auxiliary dimensions
    num_aux = 0
    for d in data:
        if d.auxiliaries and len(d.auxiliaries) > 0:
            num_aux = len(d.auxiliaries)
            break

    # Collect raw auxiliary columns for normalization
    aux_columns = []
    if num_aux > 0:
        for j in range(num_aux):
            col = []
            for d in data:
                if d.auxiliaries and len(d.auxiliaries) > j:
                    col.append(d.auxiliaries[j])
                else:
                    col.append(0.0)
            aux_columns.append(col)

    # Normalize auxiliary columns
    aux_scalers = []
    for j in range(num_aux):
        col_arr = np.array(aux_columns[j])
        c_min, c_max = float(col_arr.min()), float(col_arr.max())
        if c_max - c_min < 1e-12:
            c_max = c_min + 1.0
        aux_scalers.append((c_min, c_max))

    # Build feature rows: [bias, time_norm, sin, cos, aux_1_norm, ..., aux_k_norm]
    features = np.zeros((n, 4 + num_aux))
    for i in range(n):
        t_norm = (times[i] - t_min) / t_range
        features[i, 0] = 1.0           # bias
        features[i, 1] = t_norm         # normalized time
        features[i, 2] = sin(2.0 * pi * t_norm)  # seasonal sin
        features[i, 3] = cos(2.0 * pi * t_norm)  # seasonal cos
        for j in range(num_aux):
            c_min, c_max = aux_scalers[j]
            raw = aux_columns[j][i]
            features[i, 4 + j] = (raw - c_min) / (c_max - c_min)

    X_all = features                           # (n, D)
    X_obs = X_all[obs_indices]                 # (n_obs, D)

    # --- 3. NoProp denoising loop ---
    alpha = getattr(params, "ridge_alpha", 1e-4)
    T_steps = 5
    n_realizations = 5

    # Baseline ridge
    W_base = ridge_regression(X_obs, y, alpha=alpha)
    if W_base.size == 0:
        return data
    base_preds = (X_all @ W_base).astype(float)

    sigmas = np.linspace(1.0, 0.05, T_steps)
    all_realization_preds = []

    for _r in range(n_realizations):
        current_preds = base_preds.copy()
        for sigma in sigmas:
            # Augmented features at observed points: [base_features, current_pred]
            obs_preds = current_preds[obs_indices]
            X_aug_obs = np.column_stack([X_obs, obs_preds])

            # Noisy targets
            noisy_y = y + rng.normal(0, sigma * y_std, size=y.shape)

            # Denoising ridge with adaptive regularization
            W_denoise = ridge_regression(X_aug_obs, noisy_y, alpha=alpha * (1.0 + sigma))
            if W_denoise.size == 0:
                break

            # Predict at ALL points
            X_aug_all = np.column_stack([X_all, current_preds])
            current_preds = (X_aug_all @ W_denoise).astype(float)

        all_realization_preds.append(current_preds)

    # --- 4. Aggregate realizations ---
    stacked = np.stack(all_realization_preds)
    mean_preds = np.mean(stacked, axis=0)
    std_preds = np.std(stacked, axis=0) if n_realizations > 1 else np.zeros(n)

    # --- 5. Build output DataPoints ---
    result = []
    for i, d in enumerate(data):
        pred_raw = denormalize_value(float(mean_preds[i]), obs_min, obs_max)
        std_raw = float(std_preds[i]) * (obs_max - obs_min) / 1.6  # scale from [-0.8,0.8] range

        if d.observed is not None:
            result.append(DataPoint(
                time=d.time, observed=d.observed, instance_id=d.instance_id,
                auxiliaries=d.auxiliaries,
                imputed=pred_raw,
                imputed_std=max(std_raw, 0.0),
                is_masked=d.is_masked,
                latitude=d.latitude, longitude=d.longitude,
            ))
        else:
            result.append(DataPoint(
                time=d.time, observed=None, instance_id=d.instance_id,
                auxiliaries=d.auxiliaries,
                imputed=pred_raw,
                imputed_std=max(std_raw, 0.0),
                is_masked=d.is_masked,
                latitude=d.latitude, longitude=d.longitude,
            ))
    return result


def batch_impute_small_gap_noprop_denoising(
    flat_data: List[DataPoint],
    params: SimulationParams,
    on_progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """Batch small-gap imputation using standalone NoProp Denoising.
    Works with or without auxiliary data."""

    def run_per_instance(_id, sorted_data, large_gap_indices, p, neighbor_series_list=None):
        # Prepare data: mask large gaps
        data_for_imputation = []
        for idx, d in enumerate(sorted_data):
            obs = d.observed if idx not in large_gap_indices else None
            aux = list(d.auxiliaries) if d.auxiliaries else None
            data_for_imputation.append(DataPoint(
                time=d.time, observed=obs, instance_id=d.instance_id,
                auxiliaries=aux, is_masked=idx in large_gap_indices,
                latitude=d.latitude, longitude=d.longitude,
            ))

        rng = np.random.default_rng(0)
        result = run_noprop_denoising_simulation(data_for_imputation, p, rng=rng)

        # Reconstruct output
        out = []
        for idx, d in enumerate(sorted_data):
            r = result[idx] if idx < len(result) else None
            imputed_val = r.imputed if r else None
            imputed_std_val = getattr(r, "imputed_std", None) if r else None
            if idx in large_gap_indices:
                out.append(DataPoint(
                    time=d.time, observed=None, instance_id=d.instance_id,
                    auxiliaries=d.auxiliaries, imputed=None, imputed_std=None,
                    is_masked=True, latitude=d.latitude, longitude=d.longitude,
                ))
            elif d.observed is not None:
                out.append(DataPoint(
                    time=d.time, observed=d.observed, instance_id=d.instance_id,
                    auxiliaries=d.auxiliaries, imputed=imputed_val, imputed_std=imputed_std_val,
                    is_masked=False, latitude=d.latitude, longitude=d.longitude,
                ))
            else:
                out.append(DataPoint(
                    time=d.time, observed=None, instance_id=d.instance_id,
                    auxiliaries=d.auxiliaries, imputed=imputed_val, imputed_std=imputed_std_val,
                    is_masked=False, latitude=d.latitude, longitude=d.longitude,
                ))
        return out

    # Use the shared batch template
    from .small_gap_algos import _batch_small_gap_template
    return _batch_small_gap_template(flat_data, params, on_progress, run_per_instance,
                                      algo_name="NoProp Denoising")
