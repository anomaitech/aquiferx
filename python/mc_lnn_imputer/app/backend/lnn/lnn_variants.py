"""
LNN variants for small-gap imputation: LNNcfc (Closed-form Continuous-time) and GLNN (Gated LNN).
Both use the same data format as lnn_core and support autotuning for high KGE.
"""

from dataclasses import replace
from typing import List, Dict, Any, Tuple

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
from .lnn_core import prepare_data, get_current_observation_value, extract_observed


def run_lnncfc_simulation(
    data: List[DataPoint],
    params: SimulationParams,
    rng: np.random.Generator = None,
) -> List[DataPoint]:
    """
    LNNcfc: Closed-form Continuous-time reservoir.
    Dynamics dx/dt = -leak*x + b; closed-form step:
    x(t+dt) = x*exp(-leak*dt) + (b/leak)*(1 - exp(-leak*dt)), with b = tanh(Win@u + Wres@x).
    """
    if rng is None:
        rng = np.random.default_rng()

    prep = prepare_data(data)
    has_aux = prep["has_aux"]
    num_aux = prep["num_aux"]
    input_dim = prep["input_dim"]
    obs_scaler = prep["obs_scaler"]
    aux_scalers = prep["aux_scalers"]

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

    reservoir_size = params.reservoir_size
    leak = max(params.leak_rate, 1e-6)
    # Input weights — data is already normalized to [-0.8, 0.8], no per-feature re-scaling needed
    Win = (rng.random((reservoir_size, input_dim)) * 2 - 1) * params.input_scaling
    Wres = np.zeros((reservoir_size, reservoir_size))
    for i in range(reservoir_size):
        for j in range(reservoir_size):
            if rng.random() < 0.2:
                Wres[i, j] = rng.random() * 2 - 1
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
        u = np.array(input_vector, dtype=float)

        # Liquid time constant: adapt leak based on auxiliary signals (same as LNN)
        effective_leak = leak
        if params.use_liquid_time_constant and has_aux and d["auxiliaries"]:
            aux_avg = sum(d["auxiliaries"]) / len(d["auxiliaries"])
            effective_leak = leak * (1.0 + 0.5 * np.tanh(aux_avg))
        effective_leak = max(effective_leak, 1e-6)

        # Closed-form step with actual time delta
        dt_step = DT
        if params.lnn_time_aware_leak and i > 0:
            dt_step = max(data[i].time - data[i - 1].time, 1e-6)
        step_exp = np.exp(-effective_leak * dt_step)
        step_coef = (1.0 - step_exp) / effective_leak

        b = np.tanh(Win @ u + Wres @ x)
        x = x * step_exp + step_coef * b
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

    result: List[DataPoint] = []
    for i, d in enumerate(data):
        state_with_bias = np.concatenate([[1.0], all_states[i]])
        pred_norm = float(np.dot(Wout, state_with_bias))
        pred_raw = denormalize_value(pred_norm, obs_scaler["min"], obs_scaler["max"])
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
    return result


def run_glnn_simulation(
    data: List[DataPoint],
    params: SimulationParams,
    rng: np.random.Generator = None,
) -> List[DataPoint]:
    """
    GLNN: Gated LNN. Gate g = sigmoid(Wg @ [u; x]); h = tanh(Win@u + Wres@x);
    x_new = g*x + (1-g)*h.
    """
    if rng is None:
        rng = np.random.default_rng()

    prep = prepare_data(data)
    has_aux = prep["has_aux"]
    num_aux = prep["num_aux"]
    input_dim = prep["input_dim"]
    obs_scaler = prep["obs_scaler"]
    aux_scalers = prep["aux_scalers"]

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

    reservoir_size = params.reservoir_size
    gate_dim = input_dim + reservoir_size
    # Input weights — data is already normalized to [-0.8, 0.8], no per-feature re-scaling needed
    Win = (rng.random((reservoir_size, input_dim)) * 2 - 1) * params.input_scaling
    Wres = np.zeros((reservoir_size, reservoir_size))
    for i in range(reservoir_size):
        for j in range(reservoir_size):
            if rng.random() < 0.2:
                Wres[i, j] = rng.random() * 2 - 1
    eigvals = np.linalg.eigvals(Wres)
    actual_sr = np.max(np.abs(eigvals)) if eigvals.size > 0 else 1.0
    if actual_sr > 1e-10:
        Wres *= params.spectral_radius / actual_sr
    else:
        Wres *= params.spectral_radius
    Wgate = (rng.random((reservoir_size, gate_dim)) * 2 - 1) * 0.5

    states: List[np.ndarray] = []
    targets: List[float] = []
    all_states: List[np.ndarray] = []
    x = np.zeros(reservoir_size)
    last_valid_observed = 0.0

    for i, d in enumerate(norm_data):
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
        u = np.array(input_vector, dtype=float)

        h = np.tanh(Win @ u + Wres @ x)
        ux = np.concatenate([u, x])
        g = 1.0 / (1.0 + np.exp(-np.clip(Wgate @ ux, -20, 20)))
        x = g * x + (1.0 - g) * h
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

    result: List[DataPoint] = []
    for i, d in enumerate(data):
        state_with_bias = np.concatenate([[1.0], all_states[i]])
        pred_norm = float(np.dot(Wout, state_with_bias))
        pred_raw = denormalize_value(pred_norm, obs_scaler["min"], obs_scaler["max"])
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
    return result


def optimize_lnncfc_params(
    data: List[DataPoint],
    base_params: SimulationParams,
    mode: str = "projection",
    rng: np.random.Generator = None,
) -> SimulationParams:
    """Autotune LNNcfc hyperparameters (reservoir_size, leak_rate, input_scaling) to maximize KGE - 0.05*outliers."""
    if rng is None:
        rng = np.random.default_rng()

    best_params = base_params
    best_score = float("-inf")
    trials = getattr(base_params, "small_gap_optimize_trials", None) or (30 if mode == "projection" else 10)

    for _ in range(trials):
        size = int(rng.integers(10, 81))
        leak = 0.05 + rng.random() * 0.90
        scale = 0.2 + rng.random() * 0.6
        spectral = 0.5 + rng.random() * 0.5  # 0.5 to 1.0
        current_params = replace(
            base_params,
            reservoir_size=size,
            leak_rate=leak,
            input_scaling=scale,
            spectral_radius=spectral,
        )
        result = run_lnncfc_simulation(data, current_params, rng=rng)
        obs, pred = extract_observed(result)
        kge_score = float("-inf")
        if obs and pred:
            kge_score = calculate_kge(obs, pred)
        outlier_indices = identify_outliers(result)
        final_score = kge_score - (len(outlier_indices) * 0.05)
        if final_score > best_score:
            best_score = final_score
            best_params = current_params

    return best_params


def optimize_glnn_params(
    data: List[DataPoint],
    base_params: SimulationParams,
    mode: str = "projection",
    rng: np.random.Generator = None,
) -> SimulationParams:
    """Autotune GLNN hyperparameters to maximize KGE - 0.05*outliers."""
    if rng is None:
        rng = np.random.default_rng()

    best_params = base_params
    best_score = float("-inf")
    trials = getattr(base_params, "small_gap_optimize_trials", None) or (30 if mode == "projection" else 10)

    for _ in range(trials):
        size = int(rng.integers(10, 81))
        scale = 0.2 + rng.random() * 0.6
        spectral = 0.5 + rng.random() * 0.5
        current_params = replace(
            base_params,
            reservoir_size=size,
            input_scaling=scale,
            spectral_radius=spectral,
        )
        result = run_glnn_simulation(data, current_params, rng=rng)
        obs, pred = extract_observed(result)
        kge_score = float("-inf")
        if obs and pred:
            kge_score = calculate_kge(obs, pred)
        outlier_indices = identify_outliers(result)
        final_score = kge_score - (len(outlier_indices) * 0.05)
        if final_score > best_score:
            best_score = final_score
            best_params = current_params

    return best_params
