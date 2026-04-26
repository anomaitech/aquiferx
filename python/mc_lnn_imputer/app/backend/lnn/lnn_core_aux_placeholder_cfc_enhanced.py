"""
LNN CFC Enhanced: Bidirectional Multi-Scale Ensemble with Polynomial Placeholder and Anchor Injection.

Builds on lnn_core_aux_placeholder_cfc.py with six enhancements for improved large-gap accuracy:
  1. Enhanced Placeholder: polynomial features [1, t, t^2, aux1, ..., auxK, aux1*t, aux2*t, aux1*aux2]
  2. Multi-Scale CFC: per-neuron leak rates (fast=0.1, medium=0.3, slow=0.7) instead of uniform
  3. Bidirectional CFC: forward + backward pass with separate Win/Wres matrices
  4. Seasonal Encoding: sin/cos annual cycle appended to input vector
  5. Anchor Injection: periodically reset reservoir state blend during large gaps
  6. Ensemble Averaging: N random seeds averaged for final prediction + std for uncertainty

Does NOT use ARCHI donors -- all improvements are self-contained ("more memories").
"""

from dataclasses import replace
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

from .types import DataPoint, SimulationParams, DT
from .math_utils import (
    get_scaler,
    normalize_value,
    denormalize_value,
    ridge_regression,
    calculate_kge,
)
from .gaps import identify_outliers
from .lnn_core import prepare_data, get_current_observation_value
from .lnn_core_aux_placeholder import _apply_spike_correction
from .lnn_advanced import _seasonal_input, _build_multiscale_leak_rates

_TAG = "[CFC-Enhanced]"


# ---------------------------------------------------------------------------
# Enhanced polynomial placeholder
# ---------------------------------------------------------------------------

def _compute_enhanced_aux_placeholders(
    data: List[DataPoint],
    norm_data: List[Dict],
    has_aux: bool,
    num_aux: int,
    obs_scaler: Dict[str, float],
    alpha: float = 1e-2,
    use_polynomial: bool = True,
) -> List[float]:
    """
    Enhanced placeholder: ridge regression with polynomial and interaction features.
    Design matrix: [1, t, t^2, aux1, ..., auxK, aux1*t, ..., aux1*aux2, ...]
    Falls back to linear if polynomial creates numerical issues or insufficient samples.
    """
    n = len(data)
    if n == 0:
        return []

    times = np.array([d.time for d in data], dtype=float)
    t_min, t_max = float(times.min()), float(times.max())
    t_scale = max(t_max - t_min, 1e-10)
    time_norm = ((times - t_min) / t_scale) * 1.6 - 0.8

    # Base features: [1, t]
    features = [np.ones(n), time_norm]

    if use_polynomial:
        features.append(time_norm ** 2)  # t^2

    # Auxiliary features
    aux_cols: List[np.ndarray] = []
    if has_aux and num_aux > 0:
        for j in range(num_aux):
            col = np.array([
                norm_data[i]["auxiliaries"][j]
                if norm_data[i]["auxiliaries"] and j < len(norm_data[i]["auxiliaries"])
                else 0.0
                for i in range(n)
            ])
            features.append(col)
            aux_cols.append(col)

    # Interaction terms (only if polynomial enabled and there are auxiliaries)
    if use_polynomial and aux_cols:
        for col in aux_cols:
            features.append(col * time_norm)  # aux_j * t
        # Pairwise aux interactions (capped at 10 terms)
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

    X = np.column_stack(features)

    obs_mask = np.array([norm_data[i]["observed"] is not None for i in range(n)])
    y_all = np.array([
        norm_data[i]["observed"] if norm_data[i]["observed"] is not None else np.nan
        for i in range(n)
    ], dtype=float)

    n_obs = int(np.sum(obs_mask))
    min_samples = X.shape[1] + 2
    if n_obs < min_samples:
        if use_polynomial:
            return _compute_enhanced_aux_placeholders(
                data, norm_data, has_aux, num_aux, obs_scaler,
                alpha=alpha, use_polynomial=False,
            )
        return []

    X_train = X[obs_mask]
    y_train = y_all[obs_mask]

    # Higher regularization for polynomial to prevent overfitting
    effective_alpha = alpha * (5.0 if use_polynomial else 1.0)
    w = ridge_regression(X_train, y_train.tolist(), alpha=effective_alpha)
    if w.size == 0:
        return []

    placeholders_norm = (X @ w).astype(float)
    out = np.where(obs_mask, y_all, placeholders_norm)
    return out.tolist()


# ---------------------------------------------------------------------------
# Gap-edge blending alpha
# ---------------------------------------------------------------------------

def _compute_gap_edge_alpha(
    data: List[DataPoint],
    in_large_gap: List[bool],
    edge_width: int,
) -> List[float]:
    """
    Compute blending alpha for each point:
    - 0.0 = deep in gap (use placeholder)
    - >0.0 at gap edges (blend reservoir + placeholder)
    - 1.0 = not in large gap (use reservoir)
    """
    T = len(data)
    alpha = [0.0] * T

    for i in range(T):
        if not in_large_gap[i]:
            alpha[i] = 1.0
            continue
        # Distance to nearest non-gap boundary
        dist_left = T
        dist_right = T
        for j in range(i - 1, -1, -1):
            if not in_large_gap[j] or data[j].observed is not None:
                dist_left = i - j
                break
        for j in range(i + 1, T):
            if not in_large_gap[j] or data[j].observed is not None:
                dist_right = j - i
                break
        min_dist = min(dist_left, dist_right)
        if min_dist <= edge_width:
            alpha[i] = min_dist / (edge_width + 1)
        else:
            alpha[i] = 0.0

    return alpha


# ---------------------------------------------------------------------------
# Single-seed bidirectional multi-scale CFC with anchor injection
# ---------------------------------------------------------------------------

def _run_single_seed_cfc_enhanced(
    norm_data: List[Dict],
    data: List[DataPoint],
    has_aux: bool,
    num_aux: int,
    input_dim: int,
    obs_scaler: Dict,
    params: SimulationParams,
    placeholder_norm: List[float],
    in_large_gap: List[bool],
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """
    Single-seed bidirectional multi-scale CFC with anchor injection.
    Returns normalized predictions array (T,) or None on failure.
    """
    reservoir_size = params.reservoir_size
    spectral_radius = params.spectral_radius
    input_scaling = params.input_scaling
    ridge_alpha = getattr(params, "ridge_alpha", 1e-4)
    seasonal_period = getattr(params, "lnn_cfc_enhanced_seasonal_period", 12)
    bidirectional = getattr(params, "lnn_cfc_enhanced_bidirectional", True)
    anchor_injection = getattr(params, "lnn_cfc_enhanced_anchor_injection", True)
    anchor_interval = max(1, getattr(params, "lnn_cfc_enhanced_anchor_interval", 5))
    anchor_blend = getattr(params, "lnn_cfc_enhanced_anchor_blend", 0.3)

    T = len(norm_data)

    # Multi-scale leak rates (vector)
    leak_rates = _build_multiscale_leak_rates(reservoir_size)
    leak_rates = np.clip(leak_rates, 1e-6, None)

    def _build_weights(rng_local):
        Win = (rng_local.random((reservoir_size, input_dim)) * 2 - 1) * input_scaling
        mask = rng_local.random((reservoir_size, reservoir_size)) < 0.2
        Wres = (rng_local.random((reservoir_size, reservoir_size)) * 2 - 1) * mask
        eigvals = np.linalg.eigvals(Wres)
        actual_sr = np.max(np.abs(eigvals)) if eigvals.size > 0 else 1.0
        if actual_sr > 1e-10:
            Wres *= spectral_radius / actual_sr
        else:
            Wres *= spectral_radius
        return Win, Wres

    Win_fwd, Wres_fwd = _build_weights(rng)

    def _make_input(i, d):
        current_obs = float(placeholder_norm[i]) if placeholder_norm and i < len(placeholder_norm) else 0.0
        vec = [current_obs]
        if seasonal_period > 0:
            vec.extend(_seasonal_input(data[i].time, seasonal_period))
        if has_aux and d["auxiliaries"]:
            vec.extend(d["auxiliaries"])
        while len(vec) < input_dim:
            vec.append(0.0)
        return np.array(vec[:input_dim], dtype=np.float64)

    # --- Forward CFC pass ---
    x_fwd = np.zeros(reservoir_size, dtype=np.float64)
    fwd_states = []
    gap_step_counter = 0

    for i in range(T):
        d = norm_data[i]
        u = _make_input(i, d)

        dt_step = DT
        if getattr(params, "lnn_time_aware_leak", False) and i > 0:
            dt_step = max(data[i].time - data[i - 1].time, 1e-6)

        step_exp = np.exp(-leak_rates * dt_step)
        step_coef = (1.0 - step_exp) / leak_rates

        b = np.tanh(Win_fwd @ u + Wres_fwd @ x_fwd)
        x_fwd = x_fwd * step_exp + step_coef * b

        # Anchor injection during large gaps
        if anchor_injection and in_large_gap[i] and d["observed"] is None:
            gap_step_counter += 1
            if gap_step_counter % anchor_interval == 0:
                b_anchor = np.tanh(Win_fwd @ u)  # No recurrent term
                x_fwd = (1.0 - anchor_blend) * x_fwd + anchor_blend * (step_coef * b_anchor)
        else:
            gap_step_counter = 0

        fwd_states.append(x_fwd.copy())

    if not bidirectional:
        all_states = [np.concatenate([[1.0], fwd_states[i]]) for i in range(T)]
    else:
        # --- Backward CFC pass (separate weights) ---
        Win_bwd, Wres_bwd = _build_weights(rng)
        leak_rates_bwd = _build_multiscale_leak_rates(reservoir_size)
        leak_rates_bwd = np.clip(leak_rates_bwd, 1e-6, None)

        x_bwd = np.zeros(reservoir_size, dtype=np.float64)
        bwd_states = [None] * T
        gap_step_counter_bwd = 0

        for i in range(T - 1, -1, -1):
            d = norm_data[i]
            u = _make_input(i, d)

            dt_step = DT
            if getattr(params, "lnn_time_aware_leak", False) and i < T - 1:
                dt_step = max(data[i + 1].time - data[i].time, 1e-6)

            step_exp = np.exp(-leak_rates_bwd * dt_step)
            step_coef = (1.0 - step_exp) / leak_rates_bwd

            b = np.tanh(Win_bwd @ u + Wres_bwd @ x_bwd)
            x_bwd = x_bwd * step_exp + step_coef * b

            if anchor_injection and in_large_gap[i] and d["observed"] is None:
                gap_step_counter_bwd += 1
                if gap_step_counter_bwd % anchor_interval == 0:
                    b_anchor = np.tanh(Win_bwd @ u)
                    x_bwd = (1.0 - anchor_blend) * x_bwd + anchor_blend * (step_coef * b_anchor)
            else:
                gap_step_counter_bwd = 0

            bwd_states[i] = x_bwd.copy()

        # Concatenate: [1, fwd, bwd]
        all_states = [
            np.concatenate([[1.0], fwd_states[i], bwd_states[i]])
            for i in range(T)
        ]

    # --- Train on ALL points: observed + placeholder pseudo-labels ---
    # Key insight: train readout on the FULL time series so the reservoir
    # learns what to output even for drifted gap states.
    # Observed points use real values; missing points use placeholder as
    # pseudo-labels with lower weight (via sample weighting in ridge).
    train_states = []
    train_targets = []
    sample_weights = []
    pseudo_weight = getattr(params, "lnn_cfc_enhanced_pseudo_weight", 1.0)

    n_obs_train = 0
    for i in range(T):
        obs_val = norm_data[i]["observed"]
        ph_val = float(placeholder_norm[i]) if placeholder_norm and i < len(placeholder_norm) else None

        if obs_val is not None:
            train_states.append(all_states[i])
            train_targets.append(obs_val)
            sample_weights.append(1.0)
            n_obs_train += 1
        elif ph_val is not None:
            train_states.append(all_states[i])
            train_targets.append(ph_val)
            sample_weights.append(pseudo_weight)

    if n_obs_train < 2:
        return None

    X_train = np.array(train_states, dtype=np.float64)
    y_train = np.array(train_targets, dtype=np.float64)
    W_diag = np.diag(np.array(sample_weights, dtype=np.float64))

    # Weighted ridge regression: (X'WX + αI)w = X'Wy
    XtW = X_train.T @ W_diag
    XtWX = XtW @ X_train + ridge_alpha * np.eye(X_train.shape[1])
    XtWy = XtW @ y_train
    try:
        Wout = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        Wout = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]
    if Wout.size == 0:
        return None

    preds = np.array([float(np.dot(Wout, all_states[i])) for i in range(T)], dtype=np.float64)
    return preds


# ---------------------------------------------------------------------------
# Main simulation (ensemble orchestrator)
# ---------------------------------------------------------------------------

def run_lnn_simulation(
    data: List[DataPoint],
    params: SimulationParams,
    rng: Optional[np.random.Generator] = None,
) -> List[DataPoint]:
    """
    Run Enhanced LNN CFC with aux-based placeholders.
    Ensemble of bidirectional multi-scale CFC reservoirs with polynomial placeholder.
    """
    if rng is None:
        rng = np.random.default_rng()

    ensemble_size = max(1, getattr(params, "lnn_cfc_enhanced_ensemble_size", 1))
    use_polynomial = getattr(params, "lnn_cfc_enhanced_polynomial_placeholder", True)
    edge_blend_width = getattr(params, "lnn_cfc_enhanced_edge_blend_width", 3)
    seasonal_period = getattr(params, "lnn_cfc_enhanced_seasonal_period", 12)

    prep = prepare_data(data)
    has_aux = prep["has_aux"]
    num_aux = prep["num_aux"]
    obs_scaler = prep["obs_scaler"]
    aux_scalers = prep["aux_scalers"]

    # Input dimension: obs + seasonal(2) + aux
    seasonal_dims = 2 if seasonal_period > 0 else 0
    input_dim = 1 + seasonal_dims + num_aux

    # Normalize data
    norm_data: List[Dict] = []
    for d in data:
        norm_aux = []
        if has_aux and d.auxiliaries:
            for i in range(num_aux):
                val = d.auxiliaries[i] if i < len(d.auxiliaries) else 0.0
                s = aux_scalers[i]
                norm_aux.append(normalize_value(val, s["min"], s["max"]))
        norm_data.append({
            "observed": normalize_value(d.observed, obs_scaler["min"], obs_scaler["max"])
            if d.observed is not None else None,
            "auxiliaries": norm_aux,
        })

    # Enhanced polynomial placeholder
    alpha_placeholder = getattr(params, "ridge_alpha", 1e-4) * 10.0
    placeholder_norm = _compute_enhanced_aux_placeholders(
        data, norm_data, has_aux, num_aux, obs_scaler,
        alpha=alpha_placeholder, use_polynomial=use_polynomial,
    )

    # Denormalize placeholder for all points
    obs_min, obs_max = obs_scaler["min"], obs_scaler["max"]
    scale = max(obs_max - obs_min, 1e-10)
    placeholder_raw: List[Optional[float]] = []
    for i in range(len(data)):
        if placeholder_norm and i < len(placeholder_norm):
            placeholder_raw.append(denormalize_value(float(placeholder_norm[i]), obs_min, obs_max))
        else:
            placeholder_raw.append(None)

    # Detect large gaps
    max_gap_th = getattr(params, "max_gap_threshold", 10)
    in_large_gap: List[bool] = [False] * len(data)
    gap_start = -1
    gap_len = 0
    for i, d in enumerate(data):
        if d.observed is None:
            if gap_len == 0:
                gap_start = i
            gap_len += 1
        else:
            if gap_len > max_gap_th:
                for j in range(gap_start, gap_start + gap_len):
                    in_large_gap[j] = True
            gap_len = 0
            gap_start = -1
    if gap_len > max_gap_th and gap_start >= 0:
        for j in range(gap_start, gap_start + gap_len):
            in_large_gap[j] = True

    n_large_gap = sum(in_large_gap)
    bidir_str = "bidir" if getattr(params, "lnn_cfc_enhanced_bidirectional", True) else "fwd-only"
    poly_str = "poly" if use_polynomial else "linear"
    anchor_str = "anchor" if getattr(params, "lnn_cfc_enhanced_anchor_injection", True) else "no-anchor"
    pseudo_w = getattr(params, "lnn_cfc_enhanced_pseudo_weight", 1.0)
    print(f"{_TAG} {bidir_str} | {poly_str} placeholder | {anchor_str} | ensemble={ensemble_size} | pseudo_weight={pseudo_w} | large_gap={n_large_gap} pts", flush=True)

    # --- Ensemble loop ---
    T = len(data)
    all_preds: List[np.ndarray] = []
    for seed_idx in range(ensemble_size):
        seed_rng = np.random.default_rng(seed_idx * 1000 + 42)
        preds = _run_single_seed_cfc_enhanced(
            norm_data, data, has_aux, num_aux, input_dim,
            obs_scaler, params, placeholder_norm, in_large_gap, seed_rng,
        )
        if preds is not None:
            all_preds.append(preds)

    if not all_preds:
        # Fallback: return placeholder predictions
        return [
            DataPoint(
                time=d.time, observed=d.observed, instance_id=d.instance_id,
                imputed=placeholder_raw[i] if placeholder_raw[i] is not None else d.observed,
                auxiliaries=d.auxiliaries, is_masked=d.is_masked,
                latitude=d.latitude, longitude=d.longitude,
            )
            for i, d in enumerate(data)
        ]

    # Average ensemble
    pred_stack = np.stack(all_preds, axis=0)
    mean_preds = np.mean(pred_stack, axis=0)
    std_preds = np.std(pred_stack, axis=0) if len(all_preds) > 1 else np.zeros(T)

    # Gap-edge blending alpha
    gap_edge_alpha = _compute_gap_edge_alpha(data, in_large_gap, edge_blend_width)

    # --- Build results with hybrid selection ---
    result: List[DataPoint] = []
    for i, d in enumerate(data):
        reservoir_pred = denormalize_value(float(mean_preds[i]), obs_min, obs_max)
        reservoir_std = float(std_preds[i]) * scale if std_preds[i] > 0 else None
        ph_val = placeholder_raw[i]

        if d.observed is not None:
            final_pred = reservoir_pred
            final_std = reservoir_std
        elif in_large_gap[i] and ph_val is not None:
            alpha_edge = gap_edge_alpha[i]
            if alpha_edge > 0:
                final_pred = alpha_edge * reservoir_pred + (1.0 - alpha_edge) * ph_val
            else:
                final_pred = ph_val
            final_std = reservoir_std
        else:
            final_pred = reservoir_pred
            final_std = reservoir_std

        result.append(DataPoint(
            time=d.time, observed=d.observed, instance_id=d.instance_id,
            imputed=final_pred, imputed_std=final_std,
            auxiliaries=d.auxiliaries, is_masked=d.is_masked,
            latitude=d.latitude, longitude=d.longitude,
            ground_truth=ph_val if d.observed is None else None,
        ))

    # Optional spike correction
    if getattr(params, "lnn_aux_placeholder_spike_correction", False):
        blend = float(getattr(params, "lnn_aux_placeholder_spike_blend", 0.5))
        blend = max(0.01, min(0.99, blend))
        window = int(getattr(params, "lnn_aux_placeholder_spike_window", 5))
        window = max(2, min(15, window))
        sharp_frac = float(getattr(params, "lnn_aux_placeholder_spike_sharp_frac", 0.15))
        sharp_frac = max(0.05, min(0.5, sharp_frac))
        result = _apply_spike_correction(
            result, data, placeholder_norm or [],
            obs_scaler, has_aux, num_aux,
            blend=blend, window=window,
            use_target_interp=True,
            sharp_threshold_frac=sharp_frac,
        )

    return result


# ---------------------------------------------------------------------------
# Helpers for evaluation and optimization
# ---------------------------------------------------------------------------

def extract_observed(result: List[DataPoint]) -> Tuple[List[float], List[float]]:
    """Collect observed and imputed for KGE."""
    obs: List[float] = []
    pred: List[float] = []
    for d in result:
        if d.observed is not None and d.imputed is not None:
            obs.append(d.observed)
            pred.append(d.imputed)
    return obs, pred


def optimize_lnn_params(
    data: List[DataPoint],
    base_params: SimulationParams,
    mode: str = "projection",
    rng: Optional[np.random.Generator] = None,
) -> SimulationParams:
    """Auto-tune LNN hyperparameters for enhanced CFC. Uses ensemble_size=1 during search."""
    if rng is None:
        rng = np.random.default_rng()

    best_params = base_params
    best_score = float("-inf")
    trials = 10 if mode == "projection" else 8

    for trial_i in range(trials):
        size = int(rng.integers(30, 121))
        sr = round(0.7 + rng.random() * 0.5, 3)
        inp_sc = round(0.2 + rng.random() * 0.8, 3)
        alpha = 10.0 ** (rng.random() * 4 - 5)

        current_params = replace(
            base_params,
            reservoir_size=size,
            spectral_radius=sr,
            input_scaling=inp_sc,
            ridge_alpha=alpha,
            lnn_cfc_enhanced_ensemble_size=1,  # Single seed during search
        )
        result = run_lnn_simulation(data, current_params, rng=rng)
        obs, pred = extract_observed(result)
        kge_score = float("-inf")
        if obs and pred:
            kge_score = calculate_kge(obs, pred)
        outlier_indices = identify_outliers(result)
        final_score = kge_score - (len(outlier_indices) * 0.05)
        if final_score > best_score:
            best_score = final_score
            best_params = current_params
            print(f"  {_TAG} trial {trial_i+1}/{trials}: KGE={kge_score:.4f} (new best)", flush=True)
        if best_score >= 0.9:
            print(f"  {_TAG} early stop at trial {trial_i+1}/{trials}: KGE={best_score:.4f}", flush=True)
            break

    # Restore full ensemble size for final prediction
    best_params = replace(
        best_params,
        lnn_cfc_enhanced_ensemble_size=getattr(base_params, "lnn_cfc_enhanced_ensemble_size", 3),
    )
    return best_params
