"""
LNN CFC simulation core with auxiliary-based placeholders for missing data.

Same as lnn_core_aux_placeholder but uses Closed-form Continuous-time (CFC) dynamics
instead of discrete ODE (Euler) integration:
  dx/dt = -leak*x + b(t)  =>  x(t+dt) = x*exp(-leak*dt) + (b/leak)*(1 - exp(-leak*dt))

Placeholder: always ridge regression (aux/time -> observed). Readout: ridge or GP (uncertainty).
Optional spike correction inherited from the ODE variant.

Large-gap improvements:
  1. Locally-weighted linear ridge placeholder — fits aux→target with [1, t, aux_1, ..., aux_k]
     weighted by inverse temporal distance to gap center so nearby observations dominate.
  2. Anchor injection — periodically re-grounds reservoir state to aux-driven activation during large
     gaps to prevent drift.
  3. Smooth edge blending — linearly transitions between reservoir and placeholder at gap boundaries
     instead of a hard binary switch.
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
    calculate_pearson_correlation,
)
from .gaps import identify_outliers
from .lnn_core import prepare_data, get_current_observation_value

# Reuse placeholder, spike-correction, and NoProp helpers from the ODE variant
from .lnn_core_aux_placeholder import (
    _compute_aux_placeholders,
    _apply_spike_correction,
    _noprop_readout,
)

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
    from sklearn.exceptions import ConvergenceWarning
except ImportError:
    GaussianProcessRegressor = None  # type: ignore
    RBF = C = WhiteKernel = None  # type: ignore
    ConvergenceWarning = None  # type: ignore


# ---------------------------------------------------------------------------
# Per-instance auxiliary variable weighting
# ---------------------------------------------------------------------------

def _compute_aux_weights(
    data: List[DataPoint],
    norm_data: List[Dict],
    num_aux: int,
    exponent: float = 2.0,
    corr_threshold: float = 0.15,
    soft_floor: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute two sets of per-auxiliary weights from observed-only Pearson correlations.

    Returns (placeholder_weights, reservoir_weights):
      - placeholder_weights: hard threshold — columns with |r| < corr_threshold
        get weight=0 (excluded from placeholder feature matrix entirely).
        Reduces feature count for better-conditioned ridge regression.
      - reservoir_weights: soft floor — all columns kept with w_j >= soft_floor.
        Preserves information for the reservoir/readout which benefits from
        more features.

    Both arrays have shape (num_aux,) and are normalized so max weight = 1.0.
    """
    if num_aux == 0:
        return np.ones(0), np.ones(0)

    # Collect observed target values
    obs_target = []
    obs_aux = [[] for _ in range(num_aux)]
    for i, d in enumerate(data):
        if d.observed is not None and d.auxiliaries and len(d.auxiliaries) >= num_aux:
            obs_target.append(norm_data[i]["observed"] if norm_data[i]["observed"] is not None else 0.0)
            for j in range(num_aux):
                obs_aux[j].append(
                    norm_data[i]["auxiliaries"][j]
                    if norm_data[i]["auxiliaries"] and j < len(norm_data[i]["auxiliaries"])
                    else 0.0
                )

    if len(obs_target) < 3:
        return np.ones(num_aux), np.ones(num_aux)

    correlations = np.zeros(num_aux)
    for j in range(num_aux):
        correlations[j] = calculate_pearson_correlation(obs_aux[j], obs_target)

    # --- Placeholder weights: hard threshold (drop low-correlation columns) ---
    ph_weights = np.zeros(num_aux)
    for j in range(num_aux):
        abs_r = abs(correlations[j])
        if abs_r >= corr_threshold:
            ph_weights[j] = abs_r ** exponent

    # Safety: if ALL columns were excluded, keep the best one
    if ph_weights.max() < 1e-10:
        best_j = int(np.argmax(np.abs(correlations)))
        ph_weights[best_j] = abs(correlations[best_j]) ** exponent if abs(correlations[best_j]) > 0 else 1.0

    w_max = ph_weights.max()
    if w_max > 1e-10:
        ph_weights /= w_max

    # --- Reservoir weights: soft floor (keep all columns) ---
    res_weights = np.array([max(soft_floor, abs(r) ** exponent) for r in correlations])
    w_max = res_weights.max()
    if w_max > 1e-10:
        res_weights /= w_max

    return ph_weights, res_weights


# ---------------------------------------------------------------------------
# Locally-weighted linear ridge placeholder for large gaps
# ---------------------------------------------------------------------------

def _compute_local_gap_placeholders(
    data: List[DataPoint],
    norm_data: List[Dict],
    has_aux: bool,
    num_aux: int,
    obs_scaler: Dict[str, float],
    in_large_gap: List[bool],
    alpha: float = 1e-2,
    aux_weights: Optional[np.ndarray] = None,
) -> List[Optional[float]]:
    """
    For each large gap, fit a locally-weighted linear ridge regression
    using auxiliary data to predict the target. Observed points are weighted
    by inverse temporal distance to the gap center so nearby observations
    dominate the fit.

    Feature matrix: [1, t, aux_1, ..., aux_k]  (simple linear — no interactions)

    Returns a list of length len(data) with denormalized predictions for
    large-gap indices and None elsewhere.
    """
    n = len(data)
    if n == 0:
        return [None] * n

    times = np.array([d.time for d in data], dtype=float)
    t_min, t_max = float(times.min()), float(times.max())
    t_scale = max(t_max - t_min, 1e-10)
    time_norm = ((times - t_min) / t_scale) * 1.6 - 0.8

    obs_mask = np.array([norm_data[i]["observed"] is not None for i in range(n)])
    y_all = np.array([
        norm_data[i]["observed"] if norm_data[i]["observed"] is not None else np.nan
        for i in range(n)
    ], dtype=float)
    obs_min, obs_max = obs_scaler["min"], obs_scaler["max"]

    # Build simple linear features: [1, t, aux_1, ..., aux_k]
    # No t², no aux*t interactions, no aux_i*aux_j — prevents overfitting
    features = [np.ones(n, dtype=float), time_norm]
    n_aux_kept = 0
    if has_aux and num_aux > 0:
        for j in range(num_aux):
            if aux_weights is not None and j < len(aux_weights) and aux_weights[j] < 1e-10:
                continue
            col = np.fromiter(
                (
                    norm_data[i]["auxiliaries"][j]
                    if norm_data[i]["auxiliaries"] and j < len(norm_data[i]["auxiliaries"])
                    else 0.0
                    for i in range(n)
                ),
                dtype=float,
                count=n,
            )
            if aux_weights is not None and j < len(aux_weights):
                col = col * aux_weights[j]
            features.append(col)
            n_aux_kept += 1

    X_full = np.column_stack(features)
    n_features = X_full.shape[1]
    print(f"  [CFC] Placeholder features: [1, t, {n_aux_kept} aux] = {n_features} features (linear ridge)", flush=True)

    # Identify individual large gaps (contiguous blocks)
    gap_blocks: List[Tuple[int, int]] = []  # (start, end_exclusive)
    i = 0
    while i < n:
        if in_large_gap[i]:
            start = i
            while i < n and in_large_gap[i]:
                i += 1
            gap_blocks.append((start, i))
        else:
            i += 1

    if not gap_blocks:
        return [None] * n

    result: List[Optional[float]] = [None] * n

    obs_indices = np.where(obs_mask)[0]
    if len(obs_indices) < n_features + 2:
        return result

    obs_times = times[obs_indices]
    X_train = X_full[obs_indices]
    y_train = y_all[obs_indices]
    eye = np.eye(n_features)
    effective_alpha = alpha * 10.0

    for gap_start, gap_end in gap_blocks:
        gap_center_time = float(np.mean(times[gap_start:gap_end]))
        distances = np.abs(obs_times - gap_center_time)
        norm_distances = distances / t_scale
        weights = 1.0 / (1.0 + norm_distances)

        # Weighted ridge: (X'WX + αI)w = X'Wy — stronger regularization for stability
        sqrt_w = np.sqrt(weights)
        X_weighted = X_train * sqrt_w[:, None]
        y_weighted = y_train * sqrt_w
        XtWX = X_weighted.T @ X_weighted + effective_alpha * eye
        XtWy = X_weighted.T @ y_weighted
        try:
            w = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]

        if w.size == 0:
            continue

        X_gap = X_full[gap_start:gap_end]
        preds_norm = (X_gap @ w).astype(float)
        for k, idx in enumerate(range(gap_start, gap_end)):
            result[idx] = denormalize_value(float(preds_norm[k]), obs_min, obs_max)

    return result


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
    - 1.0 = not in large gap (use reservoir)
    - >0.0 at gap edges (blend reservoir + placeholder)
    - 0.0 = deep in gap (use placeholder)
    """
    T = len(data)
    alpha = np.ones(T, dtype=float)
    large_gap_mask = np.asarray(in_large_gap, dtype=bool)
    if T == 0 or not np.any(large_gap_mask):
        return alpha.tolist()

    valid_boundary = np.array(
        [(not in_large_gap[i]) or (data[i].observed is not None) for i in range(T)],
        dtype=bool,
    )

    left_boundary = np.full(T, -1, dtype=int)
    last = -1
    for i in range(T):
        if valid_boundary[i]:
            last = i
        left_boundary[i] = last

    right_boundary = np.full(T, T, dtype=int)
    last = T
    for i in range(T - 1, -1, -1):
        if valid_boundary[i]:
            last = i
        right_boundary[i] = last

    large_idx = np.where(large_gap_mask)[0]
    left_dist = np.where(left_boundary[large_idx] >= 0, large_idx - left_boundary[large_idx], T)
    right_dist = np.where(right_boundary[large_idx] < T, right_boundary[large_idx] - large_idx, T)
    min_dist = np.minimum(left_dist, right_dist)
    alpha[large_idx] = np.where(min_dist <= edge_width, min_dist / (edge_width + 1), 0.0)
    return alpha.tolist()


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


def _target_linear_interp_at_idx(
    data: List[DataPoint],
    idx: int,
    prev_obs_idx: np.ndarray,
    next_obs_idx: np.ndarray,
    obs_range: float,
    sharp_threshold_frac: float = 0.15,
) -> Optional[float]:
    """Fast target-aware interpolation using precomputed observed brackets."""
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
    if last_obs_val is None or next_obs_val is None or abs(next_t - last_t) < 1e-10:
        return float(last_obs_val) if last_obs_val is not None else None

    jump = abs(next_obs_val - last_obs_val)
    if obs_range <= 0 or jump < sharp_threshold_frac * obs_range:
        return None

    frac = (data[idx].time - last_t) / (next_t - last_t)
    return float(last_obs_val + frac * (next_obs_val - last_obs_val))


def _precompute_cfc_aux_context(
    data: List[DataPoint],
    params: SimulationParams,
) -> Dict[str, Any]:
    """
    Precompute static context for the CFC auxiliary-only architecture.
    This work is identical across autotune trials when only reservoir
    hyperparameters change.
    """
    prep = prepare_data(data)
    has_aux = prep["has_aux"]
    num_aux = prep["num_aux"]
    input_dim = prep["input_dim"]
    obs_scaler = prep["obs_scaler"]
    aux_scalers = prep["aux_scalers"]

    norm_data: List[Dict[str, Any]] = []
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
            # MC placeholder: if observed is None but imputed is set,
            # store normalized imputed value for reservoir input (not training)
            "mc_placeholder": normalize_value(d.imputed, obs_scaler["min"], obs_scaler["max"])
            if d.observed is None and d.imputed is not None else None,
            "auxiliaries": norm_aux,
        })

    if has_aux and num_aux > 0:
        ph_weights, res_weights = _compute_aux_weights(data, norm_data, num_aux)
    else:
        ph_weights = np.ones(max(num_aux, 1))
        res_weights = np.ones(max(num_aux, 1))

    alpha_placeholder = getattr(params, "ridge_alpha", 1e-4) * 10.0
    placeholder_norm = _compute_aux_placeholders(
        data, norm_data, has_aux, num_aux, obs_scaler, alpha=alpha_placeholder
    )

    max_gap_th = getattr(params, "max_gap_threshold", 10)
    in_large_gap: List[bool] = [False] * len(data)
    anchor_params_per_idx: Dict[int, Tuple[int, float]] = {}
    gap_start = -1
    gap_len = 0
    for i, d in enumerate(data):
        if d.observed is None:
            if gap_len == 0:
                gap_start = i
            gap_len += 1
        else:
            if gap_len > max_gap_th:
                if gap_len > 50:
                    g_interval, g_blend = 3, 0.45
                else:
                    g_interval, g_blend = 5, 0.3
                for j in range(gap_start, gap_start + gap_len):
                    in_large_gap[j] = True
                    anchor_params_per_idx[j] = (g_interval, g_blend)
            gap_len = 0
            gap_start = -1
    if gap_len > max_gap_th and gap_start >= 0:
        if gap_len > 50:
            g_interval, g_blend = 3, 0.45
        else:
            g_interval, g_blend = 5, 0.3
        for j in range(gap_start, gap_start + gap_len):
            in_large_gap[j] = True
            anchor_params_per_idx[j] = (g_interval, g_blend)

    n_large_gap = sum(in_large_gap)
    obs_min, obs_max = obs_scaler["min"], obs_scaler["max"]
    placeholder_raw: List[Optional[float]] = []
    for i in range(len(data)):
        if placeholder_norm and i < len(placeholder_norm):
            placeholder_raw.append(denormalize_value(float(placeholder_norm[i]), obs_min, obs_max))
        else:
            placeholder_raw.append(None)

    local_gap_preds: List[Optional[float]] = [None] * len(data)
    if n_large_gap > 0 and has_aux and num_aux > 0:
        local_gap_preds = _compute_local_gap_placeholders(
            data, norm_data, has_aux, num_aux, obs_scaler,
            in_large_gap, alpha=alpha_placeholder,
            aux_weights=ph_weights,
        )

    edge_blend_width = 3
    gap_edge_alpha = _compute_gap_edge_alpha(data, in_large_gap, edge_blend_width)

    return {
        "has_aux": has_aux,
        "num_aux": num_aux,
        "input_dim": input_dim,
        "obs_scaler": obs_scaler,
        "aux_scalers": aux_scalers,
        "norm_data": norm_data,
        "ph_weights": ph_weights,
        "res_weights": res_weights,
        "placeholder_norm": placeholder_norm,
        "placeholder_raw": placeholder_raw,
        "in_large_gap": in_large_gap,
        "anchor_params_per_idx": anchor_params_per_idx,
        "n_large_gap": n_large_gap,
        "local_gap_preds": local_gap_preds,
        "gap_edge_alpha": gap_edge_alpha,
    }


def run_lnn_simulation(
    data: List[DataPoint],
    params: SimulationParams,
    rng: Optional[np.random.Generator] = None,
    precomputed: Optional[Dict[str, Any]] = None,
) -> List[DataPoint]:
    """
    Run LNN with aux-based placeholders and CFC (Closed-form Continuous-time) dynamics.
    Placeholder: always ridge (aux/time -> observed).
    Readout: ridge (default) or Gaussian Process; GP returns imputed_std for uncertainty.

    Large-gap handling:
      - Locally-weighted linear ridge placeholder (aux→target fit weighted by proximity to gap)
      - Anchor injection (re-ground reservoir to aux signal every N steps during gap)
      - Smooth edge blending (linear transition at gap boundaries instead of hard switch)
    """
    if rng is None:
        rng = np.random.default_rng()

    context = precomputed if precomputed is not None else _precompute_cfc_aux_context(data, params)
    has_aux = context["has_aux"]
    num_aux = context["num_aux"]
    input_dim = context["input_dim"]
    obs_scaler = context["obs_scaler"]
    norm_data = context["norm_data"]
    ph_weights = context["ph_weights"]
    res_weights = context["res_weights"]
    if has_aux and num_aux > 0:
        n_kept = int(np.sum(ph_weights > 1e-10))
        n_dropped = num_aux - n_kept
        ph_str = ", ".join(f"{w:.3f}" for w in ph_weights)
        res_str = ", ".join(f"{w:.3f}" for w in res_weights)
        print(f"  [CFC] Placeholder weights: [{ph_str}] ({n_kept}/{num_aux} kept, {n_dropped} dropped)", flush=True)
        print(f"  [CFC] Reservoir weights:   [{res_str}]", flush=True)

    placeholder_norm = context["placeholder_norm"]
    in_large_gap = context["in_large_gap"]
    anchor_params_per_idx = context["anchor_params_per_idx"]
    n_large_gap = context["n_large_gap"]

    reservoir_size = params.reservoir_size
    leak = max(params.leak_rate, 1e-6)  # CFC requires non-zero leak for division
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
    gap_step_counter = 0

    for i, d in enumerate(norm_data):
        # Priority for reservoir input:
        # 1. Real observation (always best)
        # 2. MC placeholder (from upstream matrix completion — better than aux-only)
        # 3. Aux-based placeholder (locally-weighted ridge from auxiliary data)
        # 4. Last valid observed (fallback)
        if d["observed"] is not None:
            current_obs = d["observed"]
        elif d.get("mc_placeholder") is not None:
            current_obs = float(d["mc_placeholder"])
        elif placeholder_norm and i < len(placeholder_norm):
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
            for j_aux, a_val in enumerate(d["auxiliaries"]):
                w = res_weights[j_aux] if j_aux < len(res_weights) else 1.0
                input_vector.append(a_val * w)
        while len(input_vector) < input_dim:
            input_vector.append(0.0)
        input_vector = np.array(input_vector, dtype=float)

        effective_leak = leak
        if params.use_liquid_time_constant and has_aux and d["auxiliaries"]:
            # Multi-scale leak modulation:
            # - Fast aux (soilw, index 0): modulates short-term dynamics
            # - Slow aux (rolling averages, indices 1+): modulates memory retention
            # Aux values are already z-score normalized, so tanh operates in consistent range
            d_aux = np.array(d["auxiliaries"], dtype=float)
            n_aux_avail = len(d_aux)
            w_slice = res_weights[:n_aux_avail]

            if n_aux_avail >= 2:
                # Fast component: instantaneous soil moisture (index 0)
                fast_signal = d_aux[0] * w_slice[0]
                # Slow component: weighted average of rolling means (indices 1+)
                slow_vals = d_aux[1:]
                slow_w = w_slice[1:]
                slow_denom = max(np.sum(slow_w), 1e-10)
                slow_signal = float(np.dot(slow_vals, slow_w) / slow_denom)

                # Modulation: fast drives responsiveness, slow drives memory
                # Scale by correlation strength so weakly-correlated aux has less influence
                max_corr_weight = float(np.max(w_slice)) if len(w_slice) > 0 else 0.5
                modulation_strength = 0.4 * max_corr_weight  # adaptive, not hardcoded

                fast_mod = modulation_strength * np.tanh(fast_signal)
                slow_mod = 0.5 * modulation_strength * np.tanh(slow_signal)
                effective_leak = leak * (1.0 + fast_mod) * (1.0 + slow_mod)
            else:
                # Single aux: simple modulation
                aux_avg = float(np.dot(d_aux, w_slice) / max(np.sum(w_slice), 1e-10))
                effective_leak = leak * (1.0 + 0.4 * np.tanh(aux_avg))

        effective_leak = max(effective_leak, 1e-6)

        # ---- CFC: Closed-form Continuous-time update ----
        dt_step = DT
        if params.lnn_time_aware_leak and i > 0:
            dt_step = max(data[i].time - data[i - 1].time, 1e-6)

        step_exp = np.exp(-effective_leak * dt_step)
        step_coef = (1.0 - step_exp) / effective_leak

        b = np.tanh(Win @ input_vector + Wres @ x)
        x = x * step_exp + step_coef * b

        # ---- Anchor injection during large gaps ----
        # Periodically re-ground reservoir state to aux-driven activation
        # (feedforward only, no recurrent memory) to prevent drift
        if in_large_gap[i] and d["observed"] is None:
            gap_step_counter += 1
            a_interval, a_blend = anchor_params_per_idx.get(i, (5, 0.3))
            if gap_step_counter % a_interval == 0:
                b_anchor = np.tanh(Win @ input_vector)  # No recurrent term
                x = (1.0 - a_blend) * x + a_blend * (step_coef * b_anchor)
        else:
            gap_step_counter = 0
        # ---- End CFC update ----

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
                date_label=d.date_label,
                timestamp=d.timestamp,
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
    noprop_preds_norm: Optional[np.ndarray] = None
    noprop_std_norm: Optional[np.ndarray] = None
    obs_min, obs_max = obs_scaler["min"], obs_scaler["max"]
    scale = max(obs_max - obs_min, 1e-10)

    # --- NoProp denoising readout ---
    if use_noprop_readout:
        alpha = getattr(params, "ridge_alpha", 1e-4)
        obs_indices = [i for i in range(len(norm_data)) if norm_data[i]["observed"] is not None]
        noprop_preds_norm, noprop_std_norm = _noprop_readout(
            X, y, all_states, obs_indices, len(data),
            alpha=alpha, T_steps=5, n_realizations=5, rng=rng,
        )
        print(f"  [CFC NoProp] Denoising readout: {len(obs_indices)} training pts, "
              f"5 steps x 5 realizations", flush=True)
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

    use_elm_readout = readout_mode == "elm"
    use_bptt_readout = readout_mode == "bptt"

    # --- BPTT (Backpropagation Through Time) — trainable LNN ---
    if use_bptt_readout:
        alpha = getattr(params, "ridge_alpha", 1e-4)
        n_epochs = 10
        lr = 0.001
        n_obs = len(targets)

        # Start with ridge solution as initialization (warm start)
        Wout_init = ridge_regression(X, y, alpha=alpha)
        if Wout_init.size == 0:
            use_bptt_readout = False
        else:
            # Also make Win trainable (copy from current)
            Win_train = Win.copy()
            Wout_train = Wout_init.copy()

            best_loss = float('inf')
            best_Win = Win_train.copy()
            best_Wout = Wout_train.copy()

            for epoch in range(n_epochs):
                # Forward pass: re-run reservoir with current Win
                x_state = np.zeros(reservoir_size)
                last_valid = 0.0
                epoch_states = []
                epoch_obs_indices = []
                epoch_targets = []

                for t_idx, d in enumerate(norm_data):
                    if placeholder_norm and t_idx < len(placeholder_norm):
                        current_obs = float(placeholder_norm[t_idx])
                    else:
                        current_obs = last_valid
                    if d["observed"] is not None:
                        last_valid = d["observed"]

                    input_vec = [current_obs]
                    if has_aux and d["auxiliaries"]:
                        for j_aux, a_val in enumerate(d["auxiliaries"]):
                            w = res_weights[j_aux] if j_aux < len(res_weights) else 1.0
                            input_vec.append(a_val * w)
                    while len(input_vec) < input_dim:
                        input_vec.append(0.0)
                    input_vec = np.array(input_vec, dtype=float)

                    dt_step = 0.1
                    step_exp = np.exp(-leak * dt_step)
                    step_coef = (1.0 - step_exp) / leak
                    pre_act = Win_train @ input_vec + Wres @ x_state
                    b = np.tanh(pre_act)
                    x_state = x_state * step_exp + step_coef * b

                    if d["observed"] is not None:
                        epoch_states.append(np.concatenate([[1.0], x_state.copy()]))
                        epoch_obs_indices.append(t_idx)
                        epoch_targets.append(d["observed"])

                # Compute loss and gradients
                X_epoch = np.array(epoch_states)
                y_epoch = np.array(epoch_targets)
                preds = X_epoch @ Wout_train
                errors = preds - y_epoch
                loss = np.mean(errors ** 2) + alpha * np.sum(Wout_train ** 2)

                if loss < best_loss:
                    best_loss = loss
                    best_Win = Win_train.copy()
                    best_Wout = Wout_train.copy()

                # Gradient for Wout: dL/dWout = (2/n) * X' * errors + 2*alpha*Wout
                grad_Wout = (2.0 / len(errors)) * X_epoch.T @ errors + 2 * alpha * Wout_train
                Wout_train -= lr * grad_Wout

                # Gradient for Win (approximate — only through readout, not full BPTT)
                # dL/dWin ≈ dL/dpred * dpred/dstate * dstate/dWin
                # Simplified: use finite differences for a few random directions
                if epoch < 5:  # only tune Win in early epochs
                    n_perturb = 3
                    for _ in range(n_perturb):
                        direction = np.random.randn(*Win_train.shape) * 0.001
                        Win_plus = Win_train + direction

                        # Re-run forward with perturbed Win (fast approx: use last few states)
                        x_p = np.zeros(reservoir_size)
                        lv = 0.0
                        states_p = []
                        for t_idx, d in enumerate(norm_data):
                            if placeholder_norm and t_idx < len(placeholder_norm):
                                co = float(placeholder_norm[t_idx])
                            else:
                                co = lv
                            if d["observed"] is not None:
                                lv = d["observed"]
                            iv = [co]
                            if has_aux and d["auxiliaries"]:
                                for j_a, a_v in enumerate(d["auxiliaries"]):
                                    w = res_weights[j_a] if j_a < len(res_weights) else 1.0
                                    iv.append(a_v * w)
                            while len(iv) < input_dim:
                                iv.append(0.0)
                            iv = np.array(iv, dtype=float)
                            se = np.exp(-leak * 0.1)
                            sc = (1.0 - se) / leak
                            b_p = np.tanh(Win_plus @ iv + Wres @ x_p)
                            x_p = x_p * se + sc * b_p
                            if d["observed"] is not None:
                                states_p.append(np.concatenate([[1.0], x_p.copy()]))

                        if len(states_p) == len(epoch_states):
                            Xp = np.array(states_p)
                            preds_p = Xp @ Wout_train
                            loss_p = np.mean((preds_p - y_epoch) ** 2)
                            # Finite difference gradient
                            grad_approx = (loss_p - loss) / 0.001
                            Win_train -= lr * 0.1 * grad_approx * direction

            # Use best weights
            Win[:] = best_Win
            Wout = best_Wout

            # Re-run reservoir with trained Win to get final all_states
            x_state = np.zeros(reservoir_size)
            last_valid = 0.0
            all_states_trained = []
            for t_idx, d in enumerate(norm_data):
                if placeholder_norm and t_idx < len(placeholder_norm):
                    current_obs = float(placeholder_norm[t_idx])
                else:
                    current_obs = last_valid
                if d["observed"] is not None:
                    last_valid = d["observed"]
                input_vec = [current_obs]
                if has_aux and d["auxiliaries"]:
                    for j_aux, a_val in enumerate(d["auxiliaries"]):
                        w = res_weights[j_aux] if j_aux < len(res_weights) else 1.0
                        input_vec.append(a_val * w)
                while len(input_vec) < input_dim:
                    input_vec.append(0.0)
                input_vec = np.array(input_vec, dtype=float)
                se = np.exp(-leak * 0.1)
                sc = (1.0 - se) / leak
                b = np.tanh(Win @ input_vec + Wres @ x_state)
                x_state = x_state * se + sc * b
                all_states_trained.append(x_state.copy())

            all_states = all_states_trained
            print(f"  [CFC BPTT] Trained: {n_epochs} epochs, loss={best_loss:.6f}, "
                  f"{n_obs} obs, lr={lr}", flush=True)

    # --- ELM (Extreme Learning Machine) readout ---
    elm_W_hidden = None
    elm_b_hidden = None
    elm_Wout = None
    if use_elm_readout:
        alpha = getattr(params, "ridge_alpha", 1e-4)
        n_hidden = max(reservoir_size * 2, 50)  # expand feature space
        n_input = X.shape[1]

        # Random hidden layer (fixed, not trained)
        elm_rng = np.random.default_rng(42)
        elm_W_hidden = (elm_rng.random((n_input, n_hidden)) * 2 - 1) * 0.5
        elm_b_hidden = elm_rng.random(n_hidden) * 2 - 1

        # Project training data through hidden layer
        H = np.tanh(X @ elm_W_hidden + elm_b_hidden)
        # Add bias
        H_bias = np.column_stack([np.ones(H.shape[0]), H])

        # Ridge regression on expanded features
        elm_Wout = ridge_regression(H_bias, y, alpha=alpha)
        if elm_Wout.size > 0:
            print(f"  [CFC ELM] Readout: {n_input} → {n_hidden} hidden → 1 output "
                  f"({X.shape[0]} training pts)", flush=True)

    if not use_noprop_readout and not use_gp_readout and not use_elm_readout and not use_bptt_readout:
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
                    date_label=d.date_label,
                    timestamp=d.timestamp,
                )
                for d in data
            ]

    placeholder_raw = context["placeholder_raw"]
    local_gap_preds = context["local_gap_preds"]
    if n_large_gap > 0 and has_aux and num_aux > 0:
        n_local = sum(1 for v in local_gap_preds if v is not None)
        # Summarize adaptive anchoring
        unique_params = set(anchor_params_per_idx.values())
        anchor_info = ", ".join(f"interval={iv} blend={bl}" for iv, bl in sorted(unique_params))
        print(f"  [CFC] Large gap: {n_large_gap} pts | linear ridge placeholder for {n_local} pts | anchor: {anchor_info}", flush=True)
    elif n_large_gap > 0:
        print(f"  [CFC] Large gap fallback: {n_large_gap} pts (no aux for linear ridge, using global placeholder)", flush=True)

    # ─── Compute gap-edge blending alpha ───────────────────────────────
    gap_edge_alpha = context["gap_edge_alpha"]

    # ─── Build results with smooth blending ────────────────────────────
    result: List[DataPoint] = []
    for i, d in enumerate(data):
        state_with_bias = np.concatenate([[1.0], all_states[i]])

        # Reservoir (LNN) prediction
        if use_noprop_readout and noprop_preds_norm is not None:
            lnn_pred = denormalize_value(float(noprop_preds_norm[i]), obs_min, obs_max)
            lnn_std = max(0.0, float(noprop_std_norm[i]) * scale) if noprop_std_norm is not None else None
        elif use_gp_readout and gpr is not None:
            state_2d = state_with_bias.reshape(1, -1).astype(np.float64)
            pred_norm, std_norm = gpr.predict(state_2d, return_std=True)
            lnn_pred = denormalize_value(float(pred_norm[0]), obs_min, obs_max)
            lnn_std = max(0.0, float(std_norm[0]) * scale)
        elif use_elm_readout and elm_Wout is not None and elm_Wout.size > 0:
            h = np.tanh(state_with_bias @ elm_W_hidden + elm_b_hidden)
            h_bias = np.concatenate([[1.0], h])
            lnn_pred = denormalize_value(float(np.dot(elm_Wout, h_bias)), obs_min, obs_max)
            lnn_std = None
        else:
            lnn_pred = denormalize_value(float(np.dot(Wout, state_with_bias)), obs_min, obs_max)
            lnn_std = None

        # Choose best placeholder: local linear ridge (preferred) or global linear (fallback)
        local_val = local_gap_preds[i] if local_gap_preds[i] is not None else None
        global_val = placeholder_raw[i]
        ph_val = local_val if local_val is not None else global_val

        # Smooth blending for large gaps instead of hard switch
        if in_large_gap[i] and ph_val is not None and d.observed is None:
            alpha_edge = gap_edge_alpha[i]
            if alpha_edge > 0:
                # Near gap edge: blend reservoir + placeholder
                final_pred = alpha_edge * lnn_pred + (1.0 - alpha_edge) * ph_val
            else:
                # Deep in gap: use placeholder (locally-weighted linear ridge)
                final_pred = ph_val
        else:
            final_pred = lnn_pred

        result.append(DataPoint(
            time=d.time,
            observed=d.observed,
            instance_id=d.instance_id,
            imputed=final_pred,
            imputed_std=lnn_std,
            auxiliaries=d.auxiliaries,
            is_masked=d.is_masked,
            latitude=d.latitude,
            longitude=d.longitude,
            ground_truth=ph_val if d.observed is None else None,
            date_label=d.date_label,
            timestamp=d.timestamp,
        ))

    # Optional spike correction (inherited from ODE variant)
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


def compute_aux_target_correlations(
    result: List[DataPoint],
) -> Dict[str, Any]:
    """
    Compute Pearson correlation between each auxiliary variable and the target,
    returning three sets:
      1. obs_only:         aux vs observed (only where observed is not None)
      2. obs_plus_filled:  aux vs the filled series (observed where present, imputed where missing)
      3. gap_imputed_only: aux vs imputed in gap regions only (observed is None, imputed exists)

    Each set includes per-aux correlations and an "avg" entry.
    """
    if not result:
        return {"obs_only": {}, "obs_plus_filled": {}, "gap_imputed_only": {}}

    # Determine number of auxiliaries
    num_aux = 0
    for d in result:
        if d.auxiliaries:
            num_aux = max(num_aux, len(d.auxiliaries))
            break

    if num_aux == 0:
        return {"obs_only": {}, "obs_plus_filled": {}, "gap_imputed_only": {}}

    obs_only_corrs: Dict[str, float] = {}
    filled_corrs: Dict[str, float] = {}
    gap_imputed_corrs: Dict[str, float] = {}

    for j in range(num_aux):
        aux_obs = []
        target_obs = []
        aux_filled = []
        target_filled = []
        aux_gap = []
        target_gap = []

        for d in result:
            if not d.auxiliaries or j >= len(d.auxiliaries):
                continue
            aux_val = d.auxiliaries[j]

            if d.observed is not None:
                aux_obs.append(aux_val)
                target_obs.append(d.observed)

            filled_val = d.observed if d.observed is not None else d.imputed
            if filled_val is not None:
                aux_filled.append(aux_val)
                target_filled.append(filled_val)

            # Gap-imputed only: observed is missing but imputed exists
            if d.observed is None and d.imputed is not None:
                aux_gap.append(aux_val)
                target_gap.append(d.imputed)

        key = f"aux_{j}"
        if len(aux_obs) >= 3:
            obs_only_corrs[key] = round(calculate_pearson_correlation(aux_obs, target_obs), 4)
        if len(aux_filled) >= 3:
            filled_corrs[key] = round(calculate_pearson_correlation(aux_filled, target_filled), 4)
        if len(aux_gap) >= 3:
            gap_imputed_corrs[key] = round(calculate_pearson_correlation(aux_gap, target_gap), 4)

    # Compute averages
    if obs_only_corrs:
        obs_only_corrs["avg"] = round(
            sum(obs_only_corrs.values()) / len(obs_only_corrs), 4
        )
    if filled_corrs:
        filled_corrs["avg"] = round(
            sum(filled_corrs.values()) / len(filled_corrs), 4
        )
    if gap_imputed_corrs:
        gap_imputed_corrs["avg"] = round(
            sum(gap_imputed_corrs.values()) / len(gap_imputed_corrs), 4
        )

    # Log to server console
    all_keys = sorted(set(
        list(obs_only_corrs.keys()) + list(filled_corrs.keys()) + list(gap_imputed_corrs.keys())
    ))
    if all_keys:
        print(f"  [CFC] Aux-Target Correlations:", flush=True)
        for key in all_keys:
            r_obs = obs_only_corrs.get(key, None)
            r_filled = filled_corrs.get(key, None)
            r_gap = gap_imputed_corrs.get(key, None)
            print(f"    {key}: obs_only={r_obs}, obs+filled={r_filled}, gap_imputed={r_gap}", flush=True)

    return {"obs_only": obs_only_corrs, "obs_plus_filled": filled_corrs, "gap_imputed_only": gap_imputed_corrs}


def optimize_lnn_params(
    data: List[DataPoint],
    base_params: SimulationParams,
    mode: str = "projection",
    rng: Optional[np.random.Generator] = None,
    precomputed: Optional[Dict[str, Any]] = None,
    optimize_metric: str = "blend",  # "kge" | "rmse" | "blend"
) -> SimulationParams:
    """Auto-tune LNN hyperparameters using run_lnn_simulation with aux placeholders + CFC."""
    if rng is None:
        rng = np.random.default_rng()

    best_params = base_params
    best_score = float("-inf")
    default_trials = 8 if mode == "projection" else 6
    trials = max(1, int(getattr(base_params, "small_gap_optimize_trials", None) or default_trials))
    iterations_for_search = 1 if mode == "projection" else 25

    shared_context = precomputed if precomputed is not None else _precompute_cfc_aux_context(data, base_params)

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
        result = run_lnn_simulation(data, current_params, rng=rng, precomputed=shared_context)
        obs, pred = extract_observed(result)

        if obs and pred:
            kge_score = calculate_kge(obs, pred)
            obs_arr = np.array(obs, dtype=float)
            pred_arr = np.array(pred, dtype=float)
            rmse = float(np.sqrt(np.mean((obs_arr - pred_arr) ** 2)))
            obs_std = max(float(np.std(obs_arr, ddof=1)), 1e-10)
            nrmse = rmse / obs_std

            if optimize_metric == "rmse":
                # Minimize NRMSE: score = -NRMSE (higher is better)
                final_score = -nrmse
            elif optimize_metric == "blend":
                # Blend: 50% KGE + 50% (1 - NRMSE)
                final_score = 0.5 * kge_score + 0.5 * (1.0 - nrmse)
            else:
                # Default: KGE
                final_score = kge_score
        else:
            kge_score = float("-inf")
            nrmse = float("inf")
            final_score = float("-inf")

        outlier_indices = identify_outliers(result)
        final_score -= len(outlier_indices) * 0.05

        if final_score > best_score:
            best_score = final_score
            best_params = current_params
            metric_str = f"KGE={kge_score:.4f}" if optimize_metric == "kge" else f"NRMSE={nrmse:.4f} KGE={kge_score:.4f}"
            print(f"  [CFC optimize] trial {trial_i+1}/{trials}: {metric_str} (new best)", flush=True)
        # Early stop if score is already very good
        if optimize_metric == "kge" and best_score >= 0.9:
            print(f"  [CFC optimize] early stop at trial {trial_i+1}/{trials}: score={best_score:.4f}", flush=True)
            break
    return best_params
