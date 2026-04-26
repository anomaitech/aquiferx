"""
LNN simulation core: reservoir dynamics + ridge readout.
Mirrors utils/math.ts: prepareData, getCurrentObservationValue, runLNNSimulation, optimizeLNNParams.
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


def prepare_data(data: List[DataPoint]) -> Dict[str, Any]:
    """
    Compute scalers and input dimension from data.
    JS: prepareData (hasAux, numAux, inputDim, obsScaler, auxScalers).
    """
    first_with_aux = next((d for d in data if d.auxiliaries and len(d.auxiliaries) > 0), None)
    num_aux = len(first_with_aux.auxiliaries) if first_with_aux else 0
    has_aux = num_aux > 0
    input_dim = 1 + num_aux

    obs_values = [d.observed for d in data if d.observed is not None]
    obs_scaler = get_scaler(obs_values)  # (min, max)

    aux_scalers: List[tuple] = []
    if has_aux:
        for i in range(num_aux):
            vals = [d.auxiliaries[i] if d.auxiliaries and i < len(d.auxiliaries) else 0.0 for d in data]
            aux_scalers.append(get_scaler(vals))

    return {
        "has_aux": has_aux,
        "num_aux": num_aux,
        "input_dim": input_dim,
        "obs_scaler": {"min": obs_scaler[0], "max": obs_scaler[1]},
        "aux_scalers": [{"min": s[0], "max": s[1]} for s in aux_scalers],
    }


def get_current_observation_value(
    d: DataPoint,
    last_valid_observed: float,
    has_aux: bool,
) -> float:
    """
    For missing values: use auxiliary average if available, else last valid observed.
    JS: getCurrentObservationValue.
    """
    if d.observed is not None:
        return d.observed
    if has_aux and d.auxiliaries and len(d.auxiliaries) > 0:
        valid_aux = [v for v in d.auxiliaries if v != 0 and not (v != v)]  # exclude NaN
        if valid_aux:
            return sum(valid_aux) / len(valid_aux)
    return last_valid_observed


def run_lnn_simulation(
    data: List[DataPoint],
    params: SimulationParams,
    rng: np.random.Generator = None,
) -> List[DataPoint]:
    """
    Run LNN (reservoir + ridge readout) and return data with imputed values.
    JS: runLNNSimulation.
    """
    if rng is None:
        rng = np.random.default_rng()

    prep = prepare_data(data)
    has_aux = prep["has_aux"]
    num_aux = prep["num_aux"]
    input_dim = prep["input_dim"]
    obs_scaler = prep["obs_scaler"]
    aux_scalers = prep["aux_scalers"]

    # Normalize data (JS: normData)
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
    # Win: (reservoir_size, input_dim)
    Win = (rng.random((reservoir_size, input_dim)) * 2 - 1) * params.input_scaling
    # Wres: sparse ~20%, normalized to desired spectral radius
    Wres = np.zeros((reservoir_size, reservoir_size))
    for i in range(reservoir_size):
        for j in range(reservoir_size):
            if rng.random() < 0.2:
                Wres[i, j] = rng.random() * 2 - 1
    # Normalize: scale so actual spectral radius matches params.spectral_radius
    eigvals = np.linalg.eigvals(Wres)
    actual_sr = np.max(np.abs(eigvals)) if eigvals.size > 0 else 1.0
    if actual_sr > 1e-10:
        Wres *= params.spectral_radius / actual_sr
    else:
        Wres *= params.spectral_radius

    # Run reservoir
    states: List[np.ndarray] = []   # each (1 + reservoir_size,) for ridge
    targets: List[float] = []
    all_states: List[np.ndarray] = []
    x = np.zeros(reservoir_size)
    last_valid_observed = 0.0

    for i, d in enumerate(norm_data):
        # Current observation value (handles missing via aux/last)
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

    # Predict all points
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


def extract_observed(result: List[DataPoint]) -> Tuple[List[float], List[float]]:
    """
    Collect observed and imputed values for points that have both (for KGE scoring).
    JS: extractObserved in optimizeLNNParams.
    """
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
    rng: np.random.Generator = None,
) -> SimulationParams:
    """
    Auto-tune LNN hyperparameters per instance (mirror JS optimizeLNNParams).
    PROJECTION mode: 30 trials, random reservoir_size 10–80, leak 0.05–0.95, lr 0.01–0.40,
    1 iteration per trial; score = KGE - 0.05 * outlier_count.
    Returns best_params to use for run_lnn_simulation.
    """
    if rng is None:
        rng = np.random.default_rng()

    best_params = base_params
    best_score = float("-inf")

    trials = 30 if mode == "projection" else 10
    iterations_for_search = 1 if mode == "projection" else 25

    for _ in range(trials):
        size = int(rng.integers(10, 81))  # 10 to 80 inclusive
        leak = 0.05 + rng.random() * 0.90  # 0.05 to 0.95
        lr = 0.01 + rng.random() * 0.39   # 0.01 to 0.40

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
        outlier_count = len(outlier_indices)
        final_score = kge_score - (outlier_count * 0.05)

        if final_score > best_score:
            best_score = final_score
            best_params = current_params

    return best_params
