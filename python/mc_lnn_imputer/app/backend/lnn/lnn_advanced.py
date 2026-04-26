"""
LNN-DBE: Deep Bidirectional Ensemble Liquid Neural Network.

An advanced LNN variant that significantly improves accuracy over the base LNN
through five key enhancements:

  1. Bidirectional Reservoir: forward + backward passes through time, concatenated
     states give each prediction access to both past and future context.
  2. Ensemble Averaging: N independent random initializations averaged together,
     dramatically reducing variance from random weight matrices.
  3. Seasonal Input Encoding: sin/cos features of the annual cycle appended to
     the input vector so the reservoir can leverage periodic patterns.
  4. Multi-Scale Reservoir: neurons partitioned into sub-groups with different
     leak rates (fast=0.1, medium=0.3, slow=0.7) to capture dynamics at
     multiple timescales simultaneously.
  5. Larger Default Reservoir: 150 neurons (vs 10-80 in base LNN).
"""

from dataclasses import replace
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
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
from .gaps import identify_gaps, identify_outliers
from .lnn_core import prepare_data, get_current_observation_value, extract_observed

warnings.filterwarnings("ignore")

_TAG = "[LNN-DBE]"

# ---------------------------------------------------------------------------
# Seasonal encoding
# ---------------------------------------------------------------------------

def _seasonal_input(time_val: float, period: float) -> List[float]:
    """Return [sin(2*pi*step/P), cos(2*pi*step/P)] seasonal encoding.
    Converts time_val back to step index (time_val = step * DT) so that
    period is in time steps (e.g. 12 = annual cycle for monthly data)."""
    if period <= 0:
        return []
    step = time_val / DT  # convert to step index
    angle = 2.0 * np.pi * step / period
    return [np.sin(angle), np.cos(angle)]


# ---------------------------------------------------------------------------
# Multi-scale reservoir construction
# ---------------------------------------------------------------------------

def _build_multiscale_leak_rates(reservoir_size: int) -> np.ndarray:
    """
    Assign per-neuron leak rates in 3 bands:
      - Fast (first 1/3):   leak = 0.1  (short-term memory)
      - Medium (middle 1/3): leak = 0.3  (medium-term)
      - Slow (last 1/3):    leak = 0.7  (long-term trend)
    """
    leaks = np.zeros(reservoir_size, dtype=np.float64)
    third = reservoir_size // 3
    leaks[:third] = 0.1
    leaks[third:2*third] = 0.3
    leaks[2*third:] = 0.7
    return leaks


def _build_reservoir_weights(
    reservoir_size: int,
    input_dim: int,
    spectral_radius: float,
    input_scaling: float,
    rng: np.random.Generator,
    sparsity: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build input weight matrix Win and sparse recurrent matrix Wres."""
    Win = (rng.random((reservoir_size, input_dim)) * 2 - 1) * input_scaling

    Wres = np.zeros((reservoir_size, reservoir_size), dtype=np.float64)
    for i in range(reservoir_size):
        for j in range(reservoir_size):
            if rng.random() < sparsity:
                Wres[i, j] = rng.random() * 2 - 1

    eigvals = np.linalg.eigvals(Wres)
    actual_sr = np.max(np.abs(eigvals)) if eigvals.size > 0 else 1.0
    if actual_sr > 1e-10:
        Wres *= spectral_radius / actual_sr
    else:
        Wres *= spectral_radius

    return Win, Wres


# ---------------------------------------------------------------------------
# Single-seed bidirectional reservoir run
# ---------------------------------------------------------------------------

def _run_single_seed(
    norm_data: List[Dict],
    data: List[DataPoint],
    has_aux: bool,
    num_aux: int,
    input_dim: int,
    obs_scaler: Dict,
    reservoir_size: int,
    spectral_radius: float,
    input_scaling: float,
    ridge_alpha: float,
    seasonal_period: float,
    bidirectional: bool,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """
    Run one reservoir seed: forward pass, optional backward pass,
    ridge readout, return raw predictions array or None on failure.
    """
    # Build reservoir
    Win, Wres = _build_reservoir_weights(
        reservoir_size, input_dim, spectral_radius, input_scaling, rng
    )
    leak_rates = _build_multiscale_leak_rates(reservoir_size)

    T = len(norm_data)
    last_valid = 0.0

    # --- Forward pass ---
    x_fwd = np.zeros(reservoir_size, dtype=np.float64)
    fwd_states = []

    for i in range(T):
        d = norm_data[i]
        current_obs = get_current_observation_value(
            DataPoint(
                time=data[i].time,
                observed=d["observed"],
                instance_id=data[i].instance_id,
                auxiliaries=d["auxiliaries"] if d["auxiliaries"] else None,
            ),
            last_valid,
            has_aux,
        )
        if d["observed"] is not None:
            last_valid = d["observed"]

        input_vec = [current_obs]
        # Add seasonal features
        if seasonal_period > 0:
            input_vec.extend(_seasonal_input(data[i].time, seasonal_period))
        # Add auxiliary features
        if has_aux and d["auxiliaries"]:
            input_vec.extend(d["auxiliaries"])
        while len(input_vec) < input_dim:
            input_vec.append(0.0)
        u = np.array(input_vec[:input_dim], dtype=np.float64)

        activation = np.tanh(Win @ u + Wres @ x_fwd)
        dxdt = -leak_rates * x_fwd + activation
        x_fwd = x_fwd + dxdt * DT
        fwd_states.append(x_fwd.copy())

    if not bidirectional:
        # Unidirectional: state = [1, fwd]
        all_states = [np.concatenate([[1.0], fwd_states[i]]) for i in range(T)]
    else:
        # --- Backward pass ---
        # Use separate random weights for backward (more diverse representations)
        Win_bwd, Wres_bwd = _build_reservoir_weights(
            reservoir_size, input_dim, spectral_radius, input_scaling, rng
        )
        leak_rates_bwd = _build_multiscale_leak_rates(reservoir_size)

        x_bwd = np.zeros(reservoir_size, dtype=np.float64)
        bwd_states = [None] * T
        last_valid_bwd = 0.0

        for i in range(T - 1, -1, -1):
            d = norm_data[i]
            current_obs = get_current_observation_value(
                DataPoint(
                    time=data[i].time,
                    observed=d["observed"],
                    instance_id=data[i].instance_id,
                    auxiliaries=d["auxiliaries"] if d["auxiliaries"] else None,
                ),
                last_valid_bwd,
                has_aux,
            )
            if d["observed"] is not None:
                last_valid_bwd = d["observed"]

            input_vec = [current_obs]
            if seasonal_period > 0:
                input_vec.extend(_seasonal_input(data[i].time, seasonal_period))
            if has_aux and d["auxiliaries"]:
                input_vec.extend(d["auxiliaries"])
            while len(input_vec) < input_dim:
                input_vec.append(0.0)
            u = np.array(input_vec[:input_dim], dtype=np.float64)

            activation = np.tanh(Win_bwd @ u + Wres_bwd @ x_bwd)
            dxdt = -leak_rates_bwd * x_bwd + activation
            x_bwd = x_bwd + dxdt * DT
            bwd_states[i] = x_bwd.copy()

        # Concatenate: [1, fwd, bwd]
        all_states = [
            np.concatenate([[1.0], fwd_states[i], bwd_states[i]])
            for i in range(T)
        ]

    # --- Collect training data (observed points only) ---
    train_states = []
    train_targets = []
    for i in range(T):
        if norm_data[i]["observed"] is not None:
            train_states.append(all_states[i])
            train_targets.append(norm_data[i]["observed"])

    if len(train_states) < 2:
        return None

    X_train = np.array(train_states, dtype=np.float64)
    y_train = np.array(train_targets, dtype=np.float64)

    # --- Ridge readout ---
    Wout = ridge_regression(X_train, y_train, alpha=ridge_alpha)
    if Wout.size == 0:
        return None

    # --- Predict all points ---
    preds = np.array([float(np.dot(Wout, all_states[i])) for i in range(T)], dtype=np.float64)
    return preds


# ---------------------------------------------------------------------------
# Ensemble orchestrator
# ---------------------------------------------------------------------------

def run_lnn_dbe_simulation(
    data: List[DataPoint],
    params: SimulationParams,
) -> List[DataPoint]:
    """
    Run LNN-DBE: ensemble of bidirectional multi-scale reservoirs with
    seasonal encoding. Returns DataPoints with imputed values.
    """
    ensemble_size = getattr(params, "lnn_dbe_ensemble_size", 10) or 10
    seasonal_period = getattr(params, "lnn_dbe_seasonal_period", 12) or 12
    bidirectional = getattr(params, "lnn_dbe_bidirectional", True)
    reservoir_size = params.reservoir_size
    spectral_radius = params.spectral_radius
    input_scaling = params.input_scaling
    ridge_alpha = getattr(params, "ridge_alpha", 1e-4)

    inst_id = data[0].instance_id if data else "?"
    T = len(data)

    # Prepare and normalize
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

    # Run ensemble
    all_preds: List[np.ndarray] = []
    for seed_idx in range(ensemble_size):
        rng = np.random.default_rng(seed_idx * 1000 + 42)
        preds = _run_single_seed(
            norm_data, data, has_aux, num_aux, input_dim,
            obs_scaler, reservoir_size, spectral_radius,
            input_scaling, ridge_alpha, seasonal_period,
            bidirectional, rng,
        )
        if preds is not None:
            all_preds.append(preds)

    if not all_preds:
        # Fallback: return observed values or mean
        obs_vals = [d.observed for d in data if d.observed is not None]
        fallback = float(np.mean(obs_vals)) if obs_vals else 0.0
        return [
            replace(d, imputed=d.observed if d.observed is not None else fallback, imputed_std=None)
            for d in data
        ]

    # Average ensemble predictions
    pred_stack = np.stack(all_preds, axis=0)  # (n_ensemble, T)
    mean_preds = np.mean(pred_stack, axis=0)  # (T,)
    std_preds = np.std(pred_stack, axis=0) if len(all_preds) > 1 else np.zeros(T)

    # Denormalize
    result: List[DataPoint] = []
    for i, d in enumerate(data):
        pred_raw = denormalize_value(float(mean_preds[i]), obs_scaler["min"], obs_scaler["max"])
        std_raw = float(std_preds[i]) * (obs_scaler["max"] - obs_scaler["min"]) / 1.6 if std_preds[i] > 0 else None
        result.append(replace(d, imputed=pred_raw, imputed_std=std_raw))

    return result


# ---------------------------------------------------------------------------
# Hyperparameter optimization
# ---------------------------------------------------------------------------

def optimize_lnn_dbe_params(
    data: List[DataPoint],
    base_params: SimulationParams,
    rng: np.random.Generator = None,
) -> SimulationParams:
    """
    Auto-tune LNN-DBE hyperparameters per instance (mirrors base LNN optimize_lnn_params).
    30 trials, each with ensemble=1 for speed. Random search over reservoir_size,
    spectral_radius, input_scaling, ridge_alpha. Picks the combination that gives
    the highest KGE. Final run uses full ensemble size.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    best_params = base_params
    best_score = float("-inf")

    trials = 30

    for trial_idx in range(trials):
        size = int(rng.integers(80, 201))              # 80 to 200
        sr = 0.7 + rng.random() * 0.5                  # 0.7 to 1.2
        inp_sc = 0.2 + rng.random() * 0.8              # 0.2 to 1.0
        alpha = 10.0 ** (rng.random() * 4 - 5)         # 1e-5 to 0.1

        candidate = replace(
            base_params,
            reservoir_size=size,
            spectral_radius=sr,
            input_scaling=inp_sc,
            ridge_alpha=alpha,
            lnn_dbe_ensemble_size=1,  # single seed for speed during search
        )

        result = run_lnn_dbe_simulation(data, candidate)
        obs, pred = extract_observed(result)
        if not obs or len(obs) < 2:
            continue

        kge = calculate_kge(obs, pred)
        outlier_count = len(identify_outliers(result))
        score = kge - outlier_count * 0.05

        if score > best_score:
            best_score = score
            best_params = candidate
            print(f"{_TAG}     trial {trial_idx}: KGE={kge:.4f}, score={score:.4f}, "
                  f"reservoir={size}, sr={sr:.3f}, input_sc={inp_sc:.3f}, "
                  f"ridge_alpha={alpha:.6f} (new best)", flush=True)

    print(f"{_TAG}   best score={best_score:.4f}", flush=True)

    # Restore full ensemble size for the final production run
    best_params = replace(best_params, lnn_dbe_ensemble_size=getattr(base_params, "lnn_dbe_ensemble_size", 10))
    return best_params


# ---------------------------------------------------------------------------
# Batch entry point (small-gap only)
# ---------------------------------------------------------------------------

def batch_impute_small_gap_lnn_dbe(
    flat_data: List[DataPoint],
    params: SimulationParams,
    on_progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Batch LNN-DBE for small gaps only.
    Per-instance: auto-tune → run full ensemble → restrict to small gaps → KGE.
    """
    print(f"{_TAG} batch start (Deep Bidirectional Ensemble LNN)", flush=True)
    max_gap_threshold = getattr(params, "max_gap_threshold", 10) or 10
    ensemble_size = getattr(params, "lnn_dbe_ensemble_size", 10) or 10
    seasonal_period = getattr(params, "lnn_dbe_seasonal_period", 12) or 12
    bidirectional = getattr(params, "lnn_dbe_bidirectional", True)

    print(f"{_TAG}   flat_data={len(flat_data)}, max_gap={max_gap_threshold}, "
          f"ensemble={ensemble_size}, seasonal_period={seasonal_period}, "
          f"bidirectional={bidirectional}", flush=True)

    # Group by instance
    instance_map: Dict[str, List[DataPoint]] = defaultdict(list)
    for d in flat_data:
        key = d.instance_id or "__default__"
        instance_map[key].append(d)

    def _known_count(pts: List[DataPoint]) -> int:
        return sum(1 for p in pts if p.observed is not None or p.imputed is not None)

    instance_order = sorted(
        instance_map.keys(),
        key=lambda iid: _known_count(instance_map[iid]),
        reverse=True,
    )

    kge_threshold = getattr(params, "small_gap_kge_threshold", None) or params.kge_threshold
    results: Dict[str, Dict[str, Any]] = {}
    saved_count = 0
    skipped_count = 0
    all_filled: List[DataPoint] = []

    total_instances = len(instance_order)
    for idx, inst_id in enumerate(instance_order):
        points = instance_map[inst_id]
        points_sorted = sorted(points, key=lambda p: p.time)

        n_obs = sum(1 for p in points_sorted if p.observed is not None)
        n_miss = sum(1 for p in points_sorted if p.observed is None and p.imputed is None)

        if on_progress:
            on_progress(idx + 1, total_instances, f"LNN-DBE: {inst_id}")

        if n_miss == 0:
            for p in points_sorted:
                val = p.observed if p.observed is not None else p.imputed
                all_filled.append(replace(p, imputed=val, imputed_std=None))
            results[inst_id] = {"kge": 1.0, "saved": True}
            saved_count += 1
            continue

        print(f"{_TAG}   '{inst_id}': total={len(points_sorted)}, observed={n_obs}, missing={n_miss}", flush=True)

        # Auto-tune
        print(f"{_TAG}   '{inst_id}': auto-tuning (50 trials)...", flush=True)
        best_params = optimize_lnn_dbe_params(points_sorted, params)
        print(f"{_TAG}   '{inst_id}': best params: reservoir={best_params.reservoir_size}, "
              f"sr={best_params.spectral_radius:.3f}, "
              f"input_scaling={best_params.input_scaling:.3f}, "
              f"ridge_alpha={best_params.ridge_alpha:.6f}", flush=True)

        # Run with full ensemble
        try:
            filled_pts = run_lnn_dbe_simulation(points_sorted, best_params)
        except Exception as e:
            print(f"{_TAG}   '{inst_id}': FAILED ({type(e).__name__}: {e})", flush=True)
            for p in points_sorted:
                val = p.observed if p.observed is not None else p.imputed
                all_filled.append(replace(p, imputed=val, imputed_std=None))
            results[inst_id] = {"kge": float("-inf"), "saved": False}
            skipped_count += 1
            continue

        # Restrict to small gaps only
        small_gap_indices = _small_gap_only_indices(points_sorted, max_gap_threshold)
        for i, inp in enumerate(points_sorted):
            if inp.observed is None and inp.imputed is None:
                if i not in small_gap_indices and i < len(filled_pts):
                    filled_pts[i] = replace(filled_pts[i], imputed=None, imputed_std=None)

        # Compute KGE
        obs_list, pred_list = extract_observed(filled_pts)
        kge = float("-inf")
        if obs_list and len(obs_list) >= 2:
            kge = calculate_kge(obs_list, pred_list)

        saved = kge >= kge_threshold
        results[inst_id] = {"kge": kge, "saved": saved}
        n_filled = sum(1 for p in filled_pts if p.observed is None and p.imputed is not None)
        print(f"{_TAG}   '{inst_id}': KGE={kge:.4f}, filled={n_filled}, saved={saved}", flush=True)

        all_filled.extend(filled_pts)
        if saved:
            saved_count += 1
        else:
            skipped_count += 1

    print(f"{_TAG} batch done: instances={len(instance_order)}, saved={saved_count}, skipped={skipped_count}", flush=True)

    return {
        "imputed": len(instance_order),
        "saved": saved_count,
        "skipped": skipped_count,
        "results": results,
        "filled_data": all_filled,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_gap_only_indices(points_sorted: List[DataPoint], max_gap_threshold: int) -> set:
    """Indices that belong to small gaps (gap size <= max_gap_threshold)."""
    gaps, _ = identify_gaps(points_sorted, max_gap_threshold)
    small = set()
    for g in gaps:
        if g["size"] <= max_gap_threshold:
            for i in range(g["startIdx"], g["endIdx"] + 1):
                small.add(i)
    return small
