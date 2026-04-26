"""
SIREN (Sinusoidal Implicit Neural Representation) + LNN CFC Aux Placeholder.

Pipeline:
1. Fit SIREN to observed points: t → value (learns continuous signal)
2. Generate SIREN predictions at ALL time steps
3. Append SIREN prediction as additional auxiliary feature
4. Run LNN CFC Aux Placeholder with combined auxiliaries
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any

from .types import DataPoint, SimulationParams
from .lnn_core_aux_placeholder_cfc import (
    run_lnn_simulation as run_lnn_cfc_simulation,
    optimize_lnn_params as optimize_lnn_cfc_params,
)


# ═══════════════════════════════════════════════════════════════
# SIREN Network
# ═══════════════════════════════════════════════════════════════

class _SineLayer(nn.Module):
    def __init__(self, in_f: int, out_f: int, is_first: bool = False, omega: float = 15.0):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_f, out_f)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_f, 1 / in_f)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / in_f) / omega,
                    np.sqrt(6 / in_f) / omega,
                )

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class _SIREN(nn.Module):
    def __init__(self, hidden: int = 32, n_layers: int = 2, omega: float = 15.0):
        super().__init__()
        layers = [_SineLayer(1, hidden, is_first=True, omega=omega)]
        for _ in range(n_layers):
            layers.append(_SineLayer(hidden, hidden, omega=omega))
        self.net = nn.Sequential(*layers)
        self.final = nn.Linear(hidden, 1)
        with torch.no_grad():
            self.final.weight.uniform_(
                -np.sqrt(6 / hidden) / omega,
                np.sqrt(6 / hidden) / omega,
            )

    def forward(self, x):
        return self.final(self.net(x))


def fit_siren_and_predict(
    obs_times: np.ndarray,
    obs_values: np.ndarray,
    all_times: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
    hidden: int = 32,
    n_layers: int = 2,
    omega: float = 15.0,
) -> np.ndarray:
    """Fit SIREN on observed data, return predictions at all time steps."""
    t_min, t_max = all_times.min(), all_times.max()
    t_range = t_max - t_min
    if t_range < 1e-10:
        return np.full(len(all_times), obs_values.mean())

    # Normalize time to [-1, 1]
    tn_obs = 2.0 * (obs_times - t_min) / t_range - 1.0
    tn_all = 2.0 * (all_times - t_min) / t_range - 1.0

    # Normalize values
    v_mean = obs_values.mean()
    v_std = max(obs_values.std(), 1e-10)
    v_norm = (obs_values - v_mean) / v_std

    tt = torch.FloatTensor(tn_obs).unsqueeze(1)
    vt = torch.FloatTensor(v_norm).unsqueeze(1)
    tp = torch.FloatTensor(tn_all).unsqueeze(1)

    model = _SIREN(hidden=hidden, n_layers=n_layers, omega=omega)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for _ in range(epochs):
        pred = model(tt)
        # MSE on observed + smoothness regularization to prevent wild oscillations
        loss_data = nn.MSELoss()(pred, vt)
        # Gradient penalty: penalize large changes between consecutive predictions
        pred_all = model(tp)
        if pred_all.shape[0] > 1:
            diffs = pred_all[1:] - pred_all[:-1]
            loss_smooth = 0.01 * torch.mean(diffs ** 2)
        else:
            loss_smooth = torch.tensor(0.0)
        loss = loss_data + loss_smooth
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(tp).squeeze().numpy()

    return preds * v_std + v_mean


# ═══════════════════════════════════════════════════════════════
# SIREN + LNN CFC Pipeline
# ═══════════════════════════════════════════════════════════════

def run_siren_lnn_cfc_simulation(
    data: List[DataPoint],
    params: SimulationParams,
    rng: Optional[np.random.Generator] = None,
    precomputed: Optional[Dict] = None,
) -> List[DataPoint]:
    """
    1. Fit SIREN to observed points
    2. Append SIREN predictions as extra auxiliary
    3. Run LNN CFC with augmented auxiliaries
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Extract observed data for SIREN
    obs_times = []
    obs_values = []
    all_times = []
    for d in data:
        all_times.append(d.time)
        if d.observed is not None:
            obs_times.append(d.time)
            obs_values.append(d.observed)

    all_times_arr = np.array(all_times, dtype=np.float32)
    obs_times_arr = np.array(obs_times, dtype=np.float32)
    obs_values_arr = np.array(obs_values, dtype=np.float32)

    # Fit SIREN with low omega to avoid oscillations in gaps
    if len(obs_times_arr) >= 5:
        siren_preds = fit_siren_and_predict(
            obs_times_arr, obs_values_arr, all_times_arr,
            epochs=100, lr=5e-4, hidden=32, n_layers=2, omega=5.0,
        )
        print(f"  [SIREN] Fit on {len(obs_times_arr)} obs → {len(all_times_arr)} predictions", flush=True)
    else:
        siren_preds = np.zeros(len(all_times_arr))
        print(f"  [SIREN] Too few obs ({len(obs_times_arr)}), using zeros", flush=True)

    # Augment auxiliaries with SIREN prediction
    augmented_data = []
    for i, d in enumerate(data):
        aux = list(d.auxiliaries) if d.auxiliaries else []
        aux.append(float(siren_preds[i]))
        augmented_data.append(DataPoint(
            time=d.time,
            observed=d.observed,
            auxiliaries=aux,
            instance_id=d.instance_id,
            is_masked=d.is_masked,
            latitude=d.latitude,
            longitude=d.longitude,
            date_label=d.date_label,
            timestamp=d.timestamp,
        ))

    # Run LNN CFC on augmented data
    return run_lnn_cfc_simulation(augmented_data, params, rng=rng, precomputed=precomputed)


def optimize_siren_lnn_cfc_params(
    data: List[DataPoint],
    params: SimulationParams,
    mode: str = "projection",
) -> SimulationParams:
    """Optimize hyperparameters for SIREN + LNN CFC.
    First fits SIREN, augments data, then optimizes LNN CFC params."""

    # Extract observed data for SIREN
    obs_times = np.array([d.time for d in data if d.observed is not None], dtype=np.float32)
    obs_values = np.array([d.observed for d in data if d.observed is not None], dtype=np.float32)
    all_times = np.array([d.time for d in data], dtype=np.float32)

    if len(obs_times) >= 5:
        siren_preds = fit_siren_and_predict(
            obs_times, obs_values, all_times,
            epochs=100, lr=5e-4, hidden=32, n_layers=2, omega=5.0,
        )
    else:
        siren_preds = np.zeros(len(all_times))

    # Augment
    augmented_data = []
    for i, d in enumerate(data):
        aux = list(d.auxiliaries) if d.auxiliaries else []
        aux.append(float(siren_preds[i]))
        augmented_data.append(DataPoint(
            time=d.time, observed=d.observed, auxiliaries=aux,
            instance_id=d.instance_id, is_masked=d.is_masked,
            latitude=d.latitude, longitude=d.longitude,
        ))

    return optimize_lnn_cfc_params(augmented_data, params, mode=mode)
