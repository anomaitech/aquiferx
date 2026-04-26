"""
LNN simulation core with auxiliary-based placeholders for missing data.

Placeholder: always ridge regression (aux/time -> observed). Readout: ridge or GP (uncertainty).
Optional spike correction: in regions where aux shows sudden peak/dip, blend LNN imputation
with the aux-based estimate so sharp transitions are better followed. Used for short- and long-gap.
"""

from dataclasses import replace
from typing import List, Dict, Any, Tuple, Optional
import warnings

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

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
    from sklearn.exceptions import ConvergenceWarning
except ImportError:
    GaussianProcessRegressor = None  # type: ignore
    RBF = C = WhiteKernel = None  # type: ignore
    ConvergenceWarning = None  # type: ignore  # unused when sklearn missing


def _compute_aux_placeholders(
    data: List[DataPoint],
    norm_data: List[Dict],
    has_aux: bool,
    num_aux: int,
    obs_scaler: Dict[str, float],
    alpha: float = 1e-2,
) -> List[float]:
    """
    Pre-impute missing observed values using auxiliary data so the reservoir
    sees a smoother input. Fit: (time_norm, aux_1, aux_2, ...) -> observed_norm
    on observed points; predict at all points. Return normalized placeholder
    series (one per time step). Where observed is not None, placeholder = observed_norm.
    """
    n = len(data)
    if n == 0:
        return []

    # Build design matrix: [1, time_norm, aux_1, aux_2, ...] for all points
    times = np.array([d.time for d in data], dtype=float)
    t_min, t_max = float(times.min()), float(times.max())
    t_scale = max(t_max - t_min, 1e-10)
    time_norm = ((times - t_min) / t_scale) * 1.6 - 0.8  # same scale as obs

    if has_aux and num_aux > 0:
        # X: (n, 1 + 1 + num_aux) = [1, time_norm, aux_1, ..., aux_k]
        X = np.ones((n, 2 + num_aux))
        X[:, 1] = time_norm
        for j in range(num_aux):
            X[:, 2 + j] = [norm_data[i]["auxiliaries"][j] if norm_data[i]["auxiliaries"] else 0.0 for i in range(n)]
    else:
        # No aux: use only [1, time_norm]
        X = np.ones((n, 2))
        X[:, 1] = time_norm

    # Observed indices and values (normalized)
    obs_mask = np.array([norm_data[i]["observed"] is not None for i in range(n)])
    y_all = np.array([
        norm_data[i]["observed"] if norm_data[i]["observed"] is not None else np.nan
        for i in range(n)
    ], dtype=float)

    n_obs = int(np.sum(obs_mask))
    if n_obs < 2 or (has_aux and num_aux > 0 and n_obs < num_aux + 1):
        # Fallback: no reliable fit, return None so caller uses last-known
        return []

    X_train = X[obs_mask]
    y_train = y_all[obs_mask]
    w = ridge_regression(X_train, y_train.tolist(), alpha=alpha)
    if w.size == 0:
        return []

    # Predict at all points (normalized space)
    placeholders_norm = (X @ w).astype(float)
    # Where we have observed, use observed; where missing, use prediction
    out = np.where(obs_mask, y_all, placeholders_norm)
    return out.tolist()


def _detect_spike_regions(
    data: List[DataPoint],
    has_aux: bool,
    num_aux: int,
    window: int = 5,
    threshold_quantile: float = 0.85,
) -> np.ndarray:
    """
    Mark indices where auxiliaries show sharp change over a short window (sudden peak/dip).
    Returns boolean array of length len(data): True = spike region.
    """
    n = len(data)
    if n < 3 or not has_aux or num_aux == 0:
        return np.zeros(n, dtype=bool)
    half = max(1, window // 2)
    # Per-step "sharpness": max absolute change of aux sum (or first aux) in a small window
    aux_vals = np.array([
        (sum(d.auxiliaries) / len(d.auxiliaries)) if d.auxiliaries else 0.0
        for d in data
    ], dtype=float)
    grad = np.abs(np.diff(aux_vals, prepend=aux_vals[0]))
    local_max_grad = np.zeros(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        local_max_grad[i] = float(np.max(grad[lo:hi])) if hi > lo else 0.0
    thresh = float(np.quantile(local_max_grad, threshold_quantile)) if local_max_grad.size else 0.0
    if thresh <= 0:
        return np.zeros(n, dtype=bool)
    return local_max_grad >= thresh


def _linear_interp_between_observed(
    data: List[DataPoint],
    idx: int,
    obs_range: float,
    sharp_threshold_frac: float = 0.15,
) -> Optional[float]:
    """
    For a missing point at idx, find last and next observed; if the gap shows a sharp
    change (|next_obs - last_obs| > sharp_threshold_frac * obs_range), return linear
    interpolation at this point's time so we can follow the target's peak/dip.
    """
    n = len(data)
    if idx < 0 or idx >= n or data[idx].observed is not None:
        return None
    t = data[idx].time
    last_obs_idx, last_obs_val, last_t = None, None, None
    for i in range(idx - 1, -1, -1):
        if data[i].observed is not None:
            last_obs_idx, last_obs_val, last_t = i, data[i].observed, data[i].time
            break
    next_obs_idx, next_obs_val, next_t = None, None, None
    for i in range(idx + 1, n):
        if data[i].observed is not None:
            next_obs_idx, next_obs_val, next_t = i, data[i].observed, data[i].time
            break
    if last_obs_val is None or next_obs_val is None or last_t is None or next_t is None:
        return None
    if abs(next_t - last_t) < 1e-10:
        return float(last_obs_val)
    jump = abs(next_obs_val - last_obs_val)
    if obs_range <= 0 or jump < sharp_threshold_frac * obs_range:
        return None  # not a sharp gap
    frac = (t - last_t) / (next_t - last_t)
    return float(last_obs_val + frac * (next_obs_val - last_obs_val))


def _compute_observed_brackets(data: List[DataPoint]) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest observed index to the left and right for each position."""
    n = len(data)
    prev_obs = np.full(n, -1, dtype=int)
    next_obs = np.full(n, -1, dtype=int)

    last = -1
    for i in range(n):
        if data[i].observed is not None:
            last = i
        prev_obs[i] = last

    last = -1
    for i in range(n - 1, -1, -1):
        if data[i].observed is not None:
            last = i
        next_obs[i] = last

    return prev_obs, next_obs


def _linear_interp_between_observed_fast(
    data: List[DataPoint],
    idx: int,
    prev_obs_idx: np.ndarray,
    next_obs_idx: np.ndarray,
    obs_range: float,
    sharp_threshold_frac: float = 0.15,
) -> Optional[float]:
    """Fast version of target interpolation using precomputed observed brackets."""
    if idx < 0 or idx >= len(data) or data[idx].observed is not None:
        return None

    left = int(prev_obs_idx[idx])
    right = int(next_obs_idx[idx])
    if left < 0 or right < 0 or left == right:
        return None

    last_obs_val = data[left].observed
    next_obs_val = data[right].observed
    last_t = data[left].time
    next_t = data[right].time
    if last_obs_val is None or next_obs_val is None:
        return None
    if abs(next_t - last_t) < 1e-10:
        return float(last_obs_val)

    jump = abs(next_obs_val - last_obs_val)
    if obs_range <= 0 or jump < sharp_threshold_frac * obs_range:
        return None

    frac = (data[idx].time - last_t) / (next_t - last_t)
    return float(last_obs_val + frac * (next_obs_val - last_obs_val))


def _apply_spike_correction(
    result: List[DataPoint],
    data: List[DataPoint],
    placeholder_norm: List[float],
    obs_scaler: Dict[str, float],
    has_aux: bool,
    num_aux: int,
    blend: float,
    window: int,
    use_target_interp: bool = True,
    sharp_threshold_frac: float = 0.15,
) -> List[DataPoint]:
    """
    In spike regions, blend LNN imputation with a local estimate so sudden peaks/dips
    are followed. Prefer target-based: linear interp between bracketing observed values
    when the gap shows a sharp change. Else use aux-based (placeholder) in aux-spike regions.
    """
    n = len(result)
    if blend <= 0 or blend >= 1:
        return result
    obs_min, obs_max = obs_scaler["min"], obs_scaler["max"]
    obs_range = max(obs_max - obs_min, 1e-10)
    spike_aux = _detect_spike_regions(data, has_aux, num_aux, window=window) if (placeholder_norm and len(placeholder_norm) == n) else np.zeros(n, dtype=bool)
    prev_obs_idx, next_obs_idx = _compute_observed_brackets(data)
    out: List[DataPoint] = []
    for i, dp in enumerate(result):
        if dp.observed is not None or dp.imputed is None:
            out.append(dp)
            continue
        local_val: Optional[float] = None
        # 1) Target-aware: linear interp between observed endpoints when gap has sharp change
        if use_target_interp:
            local_val = _linear_interp_between_observed_fast(
                data, i, prev_obs_idx, next_obs_idx, obs_range, sharp_threshold_frac
            )
        # 2) Fallback: aux-based (placeholder) in aux-spike regions
        if local_val is None and spike_aux[i] and placeholder_norm and i < len(placeholder_norm):
            local_val = denormalize_value(float(placeholder_norm[i]), obs_min, obs_max)
        if local_val is not None:
            blended = (1.0 - blend) * dp.imputed + blend * local_val
            out.append(replace(dp, imputed=blended))
        else:
            out.append(dp)
    return out


def _noprop_readout(
    X: np.ndarray,
    y: np.ndarray,
    all_states: List[np.ndarray],
    obs_indices: List[int],
    n_total: int,
    alpha: float = 1e-4,
    T_steps: int = 5,
    n_realizations: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    NoProp-inspired iterative denoising readout.

    Adapts the NoProp paper (block-wise denoising without backprop) to reservoir
    readout: start from a noisy ridge baseline, iteratively refine predictions by
    training augmented ridge regressors [state, current_pred] -> clean_target at
    decreasing noise levels.

    Returns (mean_predictions, std_predictions) each shape (n_total,) in normalized space.
    std is None when n_realizations == 1.
    """
    if rng is None:
        rng = np.random.default_rng()

    # All states with bias prepended
    all_X = np.array([np.concatenate([[1.0], s]) for s in all_states])  # (n_total, D)

    # Initial ridge baseline
    Wout_base = ridge_regression(X, y, alpha=alpha)
    if Wout_base.size == 0:
        return np.zeros(n_total), None
    base_preds = (all_X @ Wout_base).astype(float)

    # Noise schedule: linear decrease from 1.0 to near-zero
    sigmas = np.linspace(1.0, 0.05, T_steps)

    # Scale noise by target std so it's meaningful regardless of data range
    y_std = max(float(np.std(y)), 1e-6)

    all_realization_preds: List[np.ndarray] = []

    for _r in range(n_realizations):
        current_preds = base_preds.copy()

        for sigma in sigmas:
            # Predictions at observed indices
            obs_preds = current_preds[obs_indices]  # (n_obs,)

            # Augmented training features: [reservoir_state_with_bias, current_pred]
            X_aug = np.column_stack([X, obs_preds])  # (n_obs, D+1)

            # Noisy target: add noise scaled by sigma * y_std
            noisy_y = y + rng.normal(0, sigma * y_std, size=y.shape)

            # Train denoising ridge with slightly higher regularization at high noise
            W_denoise = ridge_regression(X_aug, noisy_y, alpha=alpha * (1.0 + sigma))
            if W_denoise.size == 0:
                break

            # Predict at ALL points
            all_X_aug = np.column_stack([all_X, current_preds])  # (n_total, D+1)
            current_preds = (all_X_aug @ W_denoise).astype(float)

        all_realization_preds.append(current_preds)

    stacked = np.stack(all_realization_preds)  # (n_realizations, n_total)
    mean_preds = np.mean(stacked, axis=0)
    std_preds = np.std(stacked, axis=0) if n_realizations > 1 else None

    return mean_preds, std_preds


def run_lnn_simulation(
    data: List[DataPoint],
    params: SimulationParams,
    rng: Optional[np.random.Generator] = None,
) -> List[DataPoint]:
    """
    Run LNN with aux-based placeholders. Placeholder: always ridge (aux/time -> observed).
    Readout: ridge (default), Gaussian Process (uncertainty), or NoProp denoising (iterative).
    """
    if rng is None:
        rng = np.random.default_rng()

    prep = prepare_data(data)
    has_aux = prep["has_aux"]
    num_aux = prep["num_aux"]
    input_dim = prep["input_dim"]
    obs_scaler = prep["obs_scaler"]
    aux_scalers = prep["aux_scalers"]

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

    # Placeholder: always ridge (aux/time -> observed)
    alpha_placeholder = getattr(params, "ridge_alpha", 1e-4) * 10.0
    placeholder_norm = _compute_aux_placeholders(
        data, norm_data, has_aux, num_aux, obs_scaler, alpha=alpha_placeholder
    )

    reservoir_size = params.reservoir_size
    Win = (rng.random((reservoir_size, input_dim)) * 2 - 1) * params.input_scaling
    # Vectorized sparse reservoir: 20% connectivity
    mask = rng.random((reservoir_size, reservoir_size)) < 0.2
    Wres = (rng.random((reservoir_size, reservoir_size)) * 2 - 1) * mask
    eigvals = np.linalg.eigvals(Wres)
    actual_sr = np.max(np.abs(eigvals)) if eigvals.size > 0 else 1.0
    if actual_sr > 1e-10:
        Wres *= params.spectral_radius / actual_sr
    else:
        Wres *= params.spectral_radius

    states: List[np.ndarray] = []
    targets: List[float] = []
    all_states: List[np.ndarray] = []
    x = np.zeros(reservoir_size)
    last_valid_observed = 0.0

    for i, d in enumerate(norm_data):
        # Use aux-based placeholder if available, else fall back to last/aux average
        if placeholder_norm and i < len(placeholder_norm):
            current_obs = float(placeholder_norm[i])
        else:
            current_obs = get_current_observation_value(
                DataPoint(
                    time=data[i].time,
                    observed=d["observed"],
                    instance_id=data[i].instance_id,
                    auxiliaries=d["auxiliaries"] if d["auxiliaries"] else None,
                ),
                last_valid_observed,
                has_aux,
            )
        if d["observed"] is not None:
            last_valid_observed = d["observed"]

        input_vector = [current_obs]
        if has_aux and d["auxiliaries"]:
            input_vector.extend(d["auxiliaries"])
        while len(input_vector) < input_dim:
            input_vector.append(0.0)
        input_vector = np.array(input_vector, dtype=float)

        effective_leak = params.leak_rate
        if params.use_liquid_time_constant and has_aux and d["auxiliaries"]:
            aux_avg = sum(d["auxiliaries"]) / len(d["auxiliaries"])
            effective_leak = params.leak_rate * (1.0 + 0.5 * np.tanh(aux_avg))

        activation = np.tanh(Win @ input_vector + Wres @ x)
        dxdt = -effective_leak * x + activation
        x = x + dxdt * DT
        all_states.append(x.copy())

        if d["observed"] is not None:
            states.append(np.concatenate([[1.0], x]))
            targets.append(d["observed"])

    if len(states) == 0 or len(targets) == 0:
        return [
            DataPoint(
                time=d.time,
                observed=d.observed,
                instance_id=d.instance_id,
                imputed=d.observed,
                auxiliaries=d.auxiliaries,
                is_masked=d.is_masked,
                latitude=d.latitude,
                longitude=d.longitude,
            )
            for d in data
        ]

    X = np.array(states)
    y = np.array(targets)
    readout_mode = (getattr(params, "lnn_aux_placeholder_readout", None) or "ridge").lower()
    use_gp_readout = readout_mode == "gp" and GaussianProcessRegressor is not None and C is not None and len(states) >= 2
    use_noprop_readout = readout_mode == "noprop"

    Wout = np.array([])
    gpr = None
    obs_min, obs_max = obs_scaler["min"], obs_scaler["max"]
    scale = max(obs_max - obs_min, 1e-10)

    # --- NoProp denoising readout ---
    if use_noprop_readout:
        alpha = getattr(params, "ridge_alpha", 1e-4)
        obs_indices = [i for i in range(len(norm_data)) if norm_data[i]["observed"] is not None]
        preds_norm, std_norm_arr = _noprop_readout(
            X, y, all_states, obs_indices, len(data),
            alpha=alpha, T_steps=5, n_realizations=5, rng=rng,
        )
        print(f"  [NoProp] Denoising readout: {len(obs_indices)} training pts, "
              f"5 steps x 5 realizations", flush=True)
        result: List[DataPoint] = []
        for i, d in enumerate(data):
            pred_raw = denormalize_value(float(preds_norm[i]), obs_min, obs_max)
            std_raw = max(0.0, float(std_norm_arr[i]) * scale) if std_norm_arr is not None else None
            result.append(DataPoint(
                time=d.time,
                observed=d.observed,
                instance_id=d.instance_id,
                imputed=pred_raw,
                imputed_std=std_raw,
                auxiliaries=d.auxiliaries,
                is_masked=d.is_masked,
                latitude=d.latitude,
                longitude=d.longitude,
            ))
    # --- GP readout ---
    elif use_gp_readout:
        alpha = getattr(params, "ridge_alpha", 1e-4)
        kernel = C(1.0, (1e-3, 1e5)) * RBF(1.0, (1e-2, 1e3)) + WhiteKernel(noise_level=1e-5)
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=alpha)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                gpr.fit(X.astype(np.float64), y.astype(np.float64))
        except Exception:
            use_gp_readout = False
            gpr = None

    if not use_noprop_readout and not use_gp_readout:
        alpha = getattr(params, "ridge_alpha", 1e-4)
        Wout = ridge_regression(X, y, alpha=alpha)
        if Wout.size == 0:
            return [
                DataPoint(
                    time=d.time,
                    observed=d.observed,
                    instance_id=d.instance_id,
                    imputed=d.observed,
                    auxiliaries=d.auxiliaries,
                    is_masked=d.is_masked,
                    latitude=d.latitude,
                    longitude=d.longitude,
                )
                for d in data
            ]

    # Predict at all points (ridge or GP); NoProp already handled above
    if not use_noprop_readout:
        result: List[DataPoint] = []
        for i, d in enumerate(data):
            state_with_bias = np.concatenate([[1.0], all_states[i]])
            if use_gp_readout and gpr is not None:
                state_2d = state_with_bias.reshape(1, -1).astype(np.float64)
                pred_norm, std_norm = gpr.predict(state_2d, return_std=True)
                pred_norm = float(pred_norm[0])
                std_norm = float(std_norm[0])
                pred_raw = denormalize_value(pred_norm, obs_min, obs_max)
                std_raw = max(0.0, std_norm * scale)
                result.append(DataPoint(
                    time=d.time,
                    observed=d.observed,
                    instance_id=d.instance_id,
                    imputed=pred_raw,
                    imputed_std=std_raw,
                    auxiliaries=d.auxiliaries,
                    is_masked=d.is_masked,
                    latitude=d.latitude,
                    longitude=d.longitude,
                ))
            else:
                pred_norm = float(np.dot(Wout, state_with_bias))
                pred_raw = denormalize_value(pred_norm, obs_min, obs_max)
                result.append(DataPoint(
                    time=d.time,
                    observed=d.observed,
                    instance_id=d.instance_id,
                    imputed=pred_raw,
                    auxiliaries=d.auxiliaries,
                    is_masked=d.is_masked,
                    latitude=d.latitude,
                    longitude=d.longitude,
                ))
    # Optional: in sudden-peak/dip regions, blend with local estimate so spikes are followed
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


def extract_observed(result: List[DataPoint]) -> Tuple[List[float], List[float]]:
    """Collect observed and imputed for KGE (same as lnn_core)."""
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
    """Auto-tune LNN hyperparameters using run_lnn_simulation with aux placeholders."""
    if rng is None:
        rng = np.random.default_rng()

    best_params = base_params
    best_score = float("-inf")
    trials = 8 if mode == "projection" else 6
    iterations_for_search = 1 if mode == "projection" else 25

    for trial_i in range(trials):
        size = int(rng.integers(10, 81))
        leak = 0.05 + rng.random() * 0.90
        lr = 0.01 + rng.random() * 0.39
        current_params = replace(
            base_params,
            reservoir_size=size,
            leak_rate=leak,
            learning_rate=lr,
            iterations=iterations_for_search,
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
            print(f"  [Aux Placeholder optimize] trial {trial_i+1}/{trials}: KGE={kge_score:.4f} (new best)", flush=True)
        # Early stop if KGE is already very good
        if best_score >= 0.9:
            print(f"  [Aux Placeholder optimize] early stop at trial {trial_i+1}/{trials}: KGE={best_score:.4f}", flush=True)
            break
    return best_params
