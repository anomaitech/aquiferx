"""
LNN core + Kriging (GPR) readout: same reservoir dynamics as lnn_core, but readout is a
Gaussian Process (kriging) on reservoir state -> target instead of ridge regression.
Provides predictive mean and standard deviation -> confidence intervals (imputed_std).
"""

from typing import List, Dict, Optional

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

from .types import DataPoint, SimulationParams, DT
from .lnn_core import (
    prepare_data,
    get_current_observation_value,
)
from .math_utils import (
    normalize_value,
    denormalize_value,
)


def run_lnn_kriging_readout_simulation(
    data: List[DataPoint],
    params: SimulationParams,
    rng: Optional[np.random.Generator] = None,
) -> List[DataPoint]:
    """
    Run LNN with same reservoir as lnn_core, but use a GPR (kriging) readout instead of
    ridge regression. Fits GPR on (state_with_bias) -> target at observed points;
    predicts mean and std at all points. Returns imputed = mean, imputed_std = std
    for confidence intervals.
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
        input_vector = np.array(input_vector, dtype=float)

        effective_leak = params.leak_rate
        if params.use_liquid_time_constant and has_aux and d["auxiliaries"]:
            aux_avg = sum(d["auxiliaries"]) / len(d["auxiliaries"])
            effective_leak = params.leak_rate * (1.0 + 0.5 * np.tanh(aux_avg))
        # Time step: use actual time delta when time-aware leak is enabled
        dt_step = DT
        if params.lnn_time_aware_leak and i > 0:
            dt_step = max(data[i].time - data[i - 1].time, 1e-6)

        activation = np.tanh(Win @ input_vector + Wres @ x)
        dxdt = -effective_leak * x + activation
        x = x + dxdt * dt_step
        all_states.append(x.copy())

        if d["observed"] is not None:
            states.append(np.concatenate([[1.0], x]))
            targets.append(d["observed"])

    if len(states) < 2:
        # GPR needs at least 2 points; fall back to observed-only for imputed
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
    obs_min, obs_max = obs_scaler["min"], obs_scaler["max"]
    scale = max(obs_max - obs_min, 1e-10)

    alpha = getattr(params, "ridge_alpha", 1e-4)
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=1e-5)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=alpha)
    try:
        gpr.fit(X, y)
    except Exception:
        # Fallback: use last observed for all
        return [
            DataPoint(
                time=d.time,
                observed=d.observed,
                instance_id=d.instance_id,
                imputed=d.observed if d.observed is not None else None,
                auxiliaries=d.auxiliaries,
                is_masked=d.is_masked,
                latitude=d.latitude,
                longitude=d.longitude,
            )
            for d in data
        ]

    result: List[DataPoint] = []
    for i, d in enumerate(data):
        state_with_bias = np.concatenate([[1.0], all_states[i]]).reshape(1, -1)
        pred_norm, std_norm = gpr.predict(state_with_bias, return_std=True)
        pred_norm = float(pred_norm[0])
        std_norm = float(std_norm[0])
        pred_raw = denormalize_value(pred_norm, obs_min, obs_max)
        std_raw = std_norm * scale
        result.append(DataPoint(
            time=d.time,
            observed=d.observed,
            instance_id=d.instance_id,
            imputed=pred_raw,
            imputed_std=max(0.0, std_raw),
            auxiliaries=d.auxiliaries,
            is_masked=d.is_masked,
            latitude=d.latitude,
            longitude=d.longitude,
        ))
    return result
