#!/usr/bin/env python3
"""
MC+LNN Spatial Interpolation Module (Advanced)

Drop-in replacement for kriging using Spatial Detrending + EOF Interpolation.

Pipeline:
  1. Spatial Trend Surface: WTE_mean = f(lat, lon, elev) via polynomial regression
     - Captures the large-scale gradient (4200-7200 ft) driven by topography
     - Uses all 592 wells as training points
  2. Detrend: residuals = WTE - trend → much smaller, spatially smoother
  3. EOF Decomposition (SVD): Extract shared temporal modes + spatial loadings
     - Temporal modes U: seasonal, trend, multi-year patterns
     - Spatial loadings V: how strongly each well follows each mode
  4. Interpolate Spatial Loadings to grid cells via IDW
     - Loadings are smooth scalars → IDW works well on them
  5. Reconstruct: grid_value = trend(lat, lon, elev) + U × S × interpolated_V

Key insight: We interpolate k small spatial coefficients (loadings) instead
of 288 raw time values. The trend surface handles the large elevation
gradient that killed MC-based approaches.

Usage (drop-in for kriging):
    nc_file = generate_nc_file_mc_lnn(file_name, grid_x, grid_y, years_df,
                                       x_coords, y_coords, bbox, raster_extent,
                                       well_elevations=elevs)
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import time as _time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

# LNN imports for temporal refinement
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

try:
    from backend.lnn.lnn_core_aux_placeholder_cfc import optimize_lnn_params, run_lnn_simulation
    from backend.lnn.math_utils import calculate_kge
    from backend.lnn.types import DataPoint, SimulationParams
    HAS_LNN = True
except ImportError:
    HAS_LNN = False

try:
    import netCDF4
except ImportError:
    netCDF4 = None

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import requests
except ImportError:
    requests = None


# ─── Grid creation (same as kriging pipeline) ────────────────────────────────

def create_grid_coords(x_c, y_c, x_steps, bbox, raster_extent):
    """Create grid coordinates — identical to kriging pipeline."""
    if raster_extent == "aquifer":
        min_x, min_y, max_x, max_y = bbox
    elif raster_extent == "wells":
        min_x, max_x = min(x_c), max(x_c)
        min_y, max_y = min(y_c), max(y_c)
    else:
        min_x, min_y, max_x, max_y = bbox

    n_bin = np.absolute((max_x - min_x) / x_steps)
    grid_x = np.arange(min_x - 5 * n_bin, max_x + 5 * n_bin, n_bin)
    grid_y = np.arange(min_y - 5 * n_bin, max_y + 5 * n_bin, n_bin)
    return grid_x, grid_y


# ─── Distance ────────────────────────────────────────────────────────────────

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2)**2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _idw_weights(target_lat, target_lon, src_lats, src_lons,
                 exponent=2.0, max_neighbors=30):
    dists = np.array([_haversine(target_lat, target_lon, src_lats[i], src_lons[i])
                      for i in range(len(src_lats))])
    order = np.argsort(dists)
    sel = order[:max_neighbors]
    if dists[sel[0]] < 1.0:
        w = np.zeros(len(sel))
        w[0] = 1.0
        return sel, w
    w = 1.0 / (dists[sel] ** exponent + 1e-6)
    w /= w.sum()
    return sel, w


# ─── Elevation lookup ────────────────────────────────────────────────────────

def fetch_elevations(lats, lons, cache=None):
    """Fetch elevations from Open-Meteo API. Returns array of elevations in meters."""
    if cache is None:
        cache = {}
    elevs = np.zeros(len(lats))
    to_fetch = []

    for i, (lat, lon) in enumerate(zip(lats, lons)):
        key = f"{lat:.5f},{lon:.5f}"
        if key in cache:
            elevs[i] = cache[key]
        else:
            to_fetch.append(i)

    if to_fetch and requests is not None:
        for batch_start in range(0, len(to_fetch), 100):
            batch = to_fetch[batch_start:batch_start + 100]
            lat_str = ",".join(f"{lats[i]:.6f}" for i in batch)
            lon_str = ",".join(f"{lons[i]:.6f}" for i in batch)
            try:
                url = f"https://api.open-meteo.com/v1/elevation?latitude={lat_str}&longitude={lon_str}"
                resp = requests.get(url, timeout=30)
                if resp.ok:
                    data = resp.json()
                    el = data.get("elevation", [])
                    for j, idx in enumerate(batch):
                        if j < len(el) and el[j] is not None:
                            elevs[idx] = float(el[j])
                            cache[f"{lats[idx]:.5f},{lons[idx]:.5f}"] = float(el[j])
            except Exception:
                pass

    return elevs


# ─── Step 1: Spatial Trend Surface ────────────────────────────────────────────

def fit_spatial_trend(well_lats, well_lons, well_elevs, well_means, degree=2):
    """Fit polynomial trend surface: WTE_mean = f(lat, lon, elev).

    Returns coefficients for the design matrix.
    Degree 2: [1, lat, lon, elev, lat², lon², elev², lat*lon, lat*elev, lon*elev]
    """
    n = len(well_lats)

    # Normalize coordinates for numerical stability
    lat_mean, lat_std = np.mean(well_lats), max(np.std(well_lats), 1e-10)
    lon_mean, lon_std = np.mean(well_lons), max(np.std(well_lons), 1e-10)
    elev_mean, elev_std = np.mean(well_elevs), max(np.std(well_elevs), 1e-10)

    lat_n = (well_lats - lat_mean) / lat_std
    lon_n = (well_lons - lon_mean) / lon_std
    elev_n = (well_elevs - elev_mean) / elev_std

    # Build design matrix
    if degree == 1:
        X = np.column_stack([np.ones(n), lat_n, lon_n, elev_n])
    else:  # degree 2
        X = np.column_stack([
            np.ones(n), lat_n, lon_n, elev_n,
            lat_n**2, lon_n**2, elev_n**2,
            lat_n * lon_n, lat_n * elev_n, lon_n * elev_n,
        ])

    # Ridge regression (alpha=1.0 for stability)
    alpha = 1.0
    XtX = X.T @ X + alpha * np.eye(X.shape[1])
    Xty = X.T @ well_means
    coeffs = np.linalg.solve(XtX, Xty)

    # Training R²
    pred = X @ coeffs
    ss_res = np.sum((well_means - pred)**2)
    ss_tot = np.sum((well_means - np.mean(well_means))**2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    rmse = np.sqrt(np.mean((well_means - pred)**2))

    norm_params = {
        "lat_mean": lat_mean, "lat_std": lat_std,
        "lon_mean": lon_mean, "lon_std": lon_std,
        "elev_mean": elev_mean, "elev_std": elev_std,
        "degree": degree,
    }

    return coeffs, norm_params, r2, rmse


def predict_trend(lats, lons, elevs, coeffs, norm_params):
    """Predict spatial trend at given coordinates."""
    lat_n = (lats - norm_params["lat_mean"]) / norm_params["lat_std"]
    lon_n = (lons - norm_params["lon_mean"]) / norm_params["lon_std"]
    elev_n = (elevs - norm_params["elev_mean"]) / norm_params["elev_std"]

    n = len(lats)
    if norm_params["degree"] == 1:
        X = np.column_stack([np.ones(n), lat_n, lon_n, elev_n])
    else:
        X = np.column_stack([
            np.ones(n), lat_n, lon_n, elev_n,
            lat_n**2, lon_n**2, elev_n**2,
            lat_n * lon_n, lat_n * elev_n, lon_n * elev_n,
        ])

    return X @ coeffs


# ─── Step 1b: XGBoost Trend Surface ───────────────────────────────────────────

def fit_spatial_trend_xgb(well_lats, well_lons, well_elevs, well_means,
                          exclude_idx=None):
    """XGBoost trend surface: captures nonlinear lat/lon/elev → WTE relationship.

    Features: lat, lon, elev, lat², lon², elev², lat*lon, lat*elev, lon*elev,
              dist_to_centroid
    """
    if not HAS_XGB:
        return fit_spatial_trend(well_lats, well_lons, well_elevs, well_means)

    n = len(well_lats)

    # Build feature matrix
    lat_c, lon_c = np.mean(well_lats), np.mean(well_lons)
    dist_to_center = np.sqrt((well_lats - lat_c)**2 +
                              ((well_lons - lon_c) * np.cos(np.radians(lat_c)))**2)

    X = np.column_stack([
        well_lats, well_lons, well_elevs,
        well_lats**2, well_lons**2, well_elevs**2,
        well_lats * well_lons, well_lats * well_elevs, well_lons * well_elevs,
        dist_to_center,
    ])
    y = well_means

    # Exclude target well if doing LOO
    if exclude_idx is not None:
        mask = np.ones(n, dtype=bool)
        mask[exclude_idx] = False
        X_train, y_train = X[mask], y[mask]
    else:
        X_train, y_train = X, y

    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        min_child_weight=3,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X)
    ss_res = np.sum((well_means - pred)**2)
    ss_tot = np.sum((well_means - np.mean(well_means))**2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    rmse = np.sqrt(np.mean((well_means - pred)**2))

    return model, r2, rmse


def predict_trend_xgb(model, lats, lons, elevs, all_lats_mean=None, all_lons_mean=None):
    """Predict XGBoost trend at given coordinates."""
    lat_c = all_lats_mean if all_lats_mean is not None else np.mean(lats)
    lon_c = all_lons_mean if all_lons_mean is not None else np.mean(lons)
    dist_to_center = np.sqrt((lats - lat_c)**2 +
                              ((lons - lon_c) * np.cos(np.radians(lat_c)))**2)
    X = np.column_stack([
        lats, lons, elevs,
        lats**2, lons**2, elevs**2,
        lats * lons, lats * elevs, lons * elevs,
        dist_to_center,
    ])
    return model.predict(X)


# ─── Step 3: EOF Decomposition ────────────────────────────────────────────────

def eof_decomposition(residual_matrix, n_modes=10):
    """SVD of detrended residual matrix.

    residual_matrix: nTimes × nWells
    Returns: U (temporal modes), S (singular values), Vt (spatial loadings)
    """
    U, S, Vt = np.linalg.svd(residual_matrix, full_matrices=False)
    k = min(n_modes, len(S))
    return U[:, :k], S[:k], Vt[:k, :]


# ─── Step 4: Interpolate Loadings ─────────────────────────────────────────────

def interpolate_loadings(target_lat, target_lon, well_lats, well_lons,
                         well_loadings, max_neighbors=30):
    """Interpolate EOF spatial loadings to a target point via IDW.

    well_loadings: (k, nWells) — each row is one mode's spatial loadings
    Returns: (k,) — interpolated loadings at target
    """
    sel, weights = _idw_weights(target_lat, target_lon, well_lats, well_lons,
                                exponent=2.0, max_neighbors=max_neighbors)
    return well_loadings[:, sel] @ weights


# ─── Step 4b: Graph Laplacian Interpolation ───────────────────────────────────

def _build_graph_weights(lats, lons, elevs, k_neighbors=20, bandwidth_km=50.0,
                         elev_weight=0.3):
    """Build weighted adjacency matrix for spatial graph.

    Edge weight = exp(-dist²/(2*bw²)) × (1 + elev_weight * elev_similarity)
    where elev_similarity = exp(-|elev_diff|/elev_scale)
    """
    n = len(lats)
    W = np.zeros((n, n))
    bandwidth_m = bandwidth_km * 1000.0
    elev_range = max(elevs.max() - elevs.min(), 1.0)
    elev_scale = elev_range / 4.0  # characteristic elevation scale

    for i in range(n):
        dists = np.array([_haversine(lats[i], lons[i], lats[j], lons[j])
                          for j in range(n)])
        dists[i] = np.inf  # exclude self

        # Select k-nearest neighbors
        neighbors = np.argsort(dists)[:k_neighbors]

        for j in neighbors:
            dist_w = np.exp(-dists[j]**2 / (2 * bandwidth_m**2))
            elev_sim = np.exp(-abs(elevs[i] - elevs[j]) / elev_scale)
            edge_w = dist_w * (1.0 + elev_weight * elev_sim)
            W[i, j] = edge_w
            W[j, i] = edge_w  # symmetric

    return W


def graph_laplacian_interpolate(
    well_lats, well_lons, well_elevs,
    target_lats, target_lons, target_elevs,
    well_values,  # (k, nWells) — loading values per mode
    k_neighbors=20,
    bandwidth_km=50.0,
    elev_weight=0.3,
):
    """Interpolate EOF loadings using graph Laplacian harmonic extension.

    Builds a graph with wells + targets as nodes. Solves:
        f_targets = -L_tt⁻¹ @ L_tw @ f_wells

    This gives the smoothest function on the graph matching well values.
    Multi-hop: a target is influenced by distant wells through chains of
    intermediate wells, not just direct distance.
    """
    n_wells = len(well_lats)
    n_targets = len(target_lats)
    n_modes = well_values.shape[0]
    n_total = n_wells + n_targets

    # Combined coordinates
    all_lats = np.concatenate([well_lats, target_lats])
    all_lons = np.concatenate([well_lons, target_lons])
    all_elevs = np.concatenate([well_elevs, target_elevs])

    # Build adjacency matrix for full graph
    W = _build_graph_weights(all_lats, all_lons, all_elevs,
                             k_neighbors=k_neighbors,
                             bandwidth_km=bandwidth_km,
                             elev_weight=elev_weight)

    # Graph Laplacian: L = D - W
    D = np.diag(W.sum(axis=1))
    L = D - W

    # Partition: wells = [0:n_wells], targets = [n_wells:]
    L_tt = L[n_wells:, n_wells:]  # target-target
    L_tw = L[n_wells:, :n_wells]  # target-well

    # Regularize L_tt for numerical stability
    L_tt += 1e-8 * np.eye(n_targets)

    # Solve for each mode: f_target = -L_tt⁻¹ @ L_tw @ f_well
    try:
        L_tt_inv = np.linalg.inv(L_tt)
    except np.linalg.LinAlgError:
        # Fallback to IDW if graph solve fails
        result = np.zeros((n_modes, n_targets))
        for g in range(n_targets):
            sel, weights = _idw_weights(target_lats[g], target_lons[g],
                                        well_lats, well_lons, max_neighbors=20)
            result[:, g] = well_values[:, sel] @ weights
        return result

    result = np.zeros((n_modes, n_targets))
    for m in range(n_modes):
        result[m, :] = -L_tt_inv @ L_tw @ well_values[m, :]

    return result


# ─── Step 5: LNN Temporal Refinement ──────────────────────────────────────────

AUX_COLS = ["soilw", "soilw_yr01", "soilw_yr03", "soilw_yr05", "soilw_yr10"]

def _load_gldas_aux(n_times):
    """Load GLDAS auxiliary data for LNN input."""
    aux_csv = os.path.join(os.path.dirname(APP_DIR), "datas",
        "lnn_imputation_gslb_gldas_df_excercise.csv")
    if not os.path.exists(aux_csv):
        return None, {}
    aux_df = pd.read_csv(aux_csv)
    aux_df["time"] = pd.to_datetime(aux_df["time"])
    aux_df = aux_df[(aux_df["time"] >= "2000-01-01") & (aux_df["time"] <= "2023-12-31")]
    aux_df = aux_df.sort_values("time").set_index("time")
    months = pd.date_range("2000-01-01", "2023-12-01", freq="MS")
    aux_lk = {}
    for dt, row in aux_df.iterrows():
        aux_lk[(dt.year, dt.month)] = [
            float(row[c]) if c in row.index and pd.notna(row[c]) else 0.0
            for c in AUX_COLS
        ]
    return months, aux_lk


def _build_lnn_params():
    """Default LNN parameters for spatial refinement."""
    return SimulationParams(
        max_gap_threshold=20, large_gap_threshold=200,
        kge_threshold=0.5, small_gap_kge_threshold=0.30,
        ridge_alpha=1e-4, lnn_aux_placeholder_readout="ridge",
        small_gap_optimize_trials=5,
    )


def lnn_refine_prediction(
    eof_prediction: np.ndarray,      # nTimes — EOF prediction at grid cell
    nearby_well_series: np.ndarray,  # nTimes — nearest well's observed series
    nearby_well_lat: float,
    nearby_well_lon: float,
    aux_lk: dict,
    months,
    seed: int = 42,
) -> np.ndarray:
    """Refine EOF prediction using LNN transferred from nearest well.

    1. Train LNN on the nearest well (has real observations from imputation)
    2. Build target timeline: EOF predictions as pseudo-observations + GLDAS aux
    3. Run the trained LNN on target timeline → temporally refined prediction

    The LNN captures nonlinear temporal dynamics (drought lags, recharge
    response) that EOF's linear modes miss.
    """
    if not HAS_LNN:
        return eof_prediction

    n_times = len(eof_prediction)
    params = _build_lnn_params()

    # Build reference well timeline (train LNN on this)
    ref_tl = []
    for mi in range(min(n_times, len(months))):
        m = months[mi]
        aux = list(aux_lk.get((m.year, m.month), [0.0] * len(AUX_COLS)))
        aux.extend([np.sin(2 * np.pi * mi / 12.0), np.cos(2 * np.pi * mi / 12.0)])
        ref_tl.append(DataPoint(
            time=float(mi),
            observed=float(nearby_well_series[mi]) if mi < len(nearby_well_series) else None,
            instance_id="ref", auxiliaries=aux,
            date_label=m.strftime("%Y-%m"),
            latitude=nearby_well_lat, longitude=nearby_well_lon,
        ))

    # Train LNN on reference well
    with redirect_stdout(io.StringIO()):
        try:
            best_params = optimize_lnn_params(
                ref_tl, params, mode="projection",
                rng=np.random.default_rng(seed),
            )
        except Exception:
            return eof_prediction

    # Build target timeline: EOF prediction as pseudo-observations
    target_tl = []
    for mi in range(min(n_times, len(months))):
        m = months[mi]
        aux = list(aux_lk.get((m.year, m.month), [0.0] * len(AUX_COLS)))
        aux.extend([np.sin(2 * np.pi * mi / 12.0), np.cos(2 * np.pi * mi / 12.0)])
        eof_val = float(eof_prediction[mi]) if np.isfinite(eof_prediction[mi]) else None
        target_tl.append(DataPoint(
            time=float(mi),
            observed=eof_val,  # EOF prediction as pseudo-observation
            instance_id="target", auxiliaries=aux,
            date_label=m.strftime("%Y-%m"),
            latitude=nearby_well_lat, longitude=nearby_well_lon,
        ))

    # Run LNN with transferred params — 3 ensemble runs, pick best
    with redirect_stdout(io.StringIO()):
        best_result = None
        best_kge = float("-inf")
        for it in range(3):
            try:
                res = run_lnn_simulation(
                    target_tl, best_params,
                    rng=np.random.default_rng(seed + it),
                )
                obs = [d.observed for d in target_tl if d.observed is not None]
                pred = [res[i].imputed for i, d in enumerate(target_tl)
                        if d.observed is not None and i < len(res)
                        and res[i].imputed is not None]
                if obs and pred and len(obs) == len(pred):
                    k = float(calculate_kge(obs, pred))
                    if np.isfinite(k) and k > best_kge:
                        best_kge = k
                        best_result = res
            except Exception:
                pass

        if best_result is None:
            return eof_prediction

    # Extract LNN predictions
    lnn_pred = np.array([
        r.imputed if r.imputed is not None else eof_prediction[i]
        for i, r in enumerate(best_result[:n_times])
    ])

    return lnn_pred


# ─── Full Pipeline ────────────────────────────────────────────────────────────

def mc_lnn_eof_interpolation(
    well_lats: np.ndarray,
    well_lons: np.ndarray,
    well_elevs: np.ndarray,
    well_matrix: np.ndarray,       # nTimes × nWells
    target_lats: np.ndarray,       # grid cell latitudes
    target_lons: np.ndarray,       # grid cell longitudes
    target_elevs: np.ndarray,      # grid cell elevations
    n_modes: int = 10,
    max_neighbors: int = 30,
    trend_degree: int = 2,
    trend_method: str = "poly",    # "poly" or "xgb"
    use_lnn: bool = False,
    lnn_max_targets: int = 500,
    interpolation_method: str = "idw",  # "idw" or "graph"
    graph_k_neighbors: int = 20,
    graph_bandwidth_km: float = 50.0,
    graph_elev_weight: float = 0.3,
    exclude_well_idx: int = None,  # for LOO: exclude this well from XGB training
) -> np.ndarray:
    """Full pipeline: Spatial Trend + EOF + Graph/IDW loading interpolation.

    trend_method: "poly" (polynomial) or "xgb" (XGBoost)
    interpolation_method: "idw" or "graph"

    Returns: (nTimes × nTargets predictions, info dict)
    """
    n_times, n_wells = well_matrix.shape
    n_targets = len(target_lats)

    # Step 1: Fit spatial trend
    well_means = np.mean(well_matrix, axis=0)

    if trend_method == "xgb" and HAS_XGB:
        xgb_model, trend_r2, trend_rmse = fit_spatial_trend_xgb(
            well_lats, well_lons, well_elevs, well_means,
            exclude_idx=exclude_well_idx,
        )
        well_trend = predict_trend_xgb(xgb_model, well_lats, well_lons, well_elevs,
                                        np.mean(well_lats), np.mean(well_lons))
        target_trends = predict_trend_xgb(xgb_model, target_lats, target_lons, target_elevs,
                                           np.mean(well_lats), np.mean(well_lons))
    else:
        coeffs, norm_params, trend_r2, trend_rmse = fit_spatial_trend(
            well_lats, well_lons, well_elevs, well_means, degree=trend_degree
        )
        well_trend = predict_trend(well_lats, well_lons, well_elevs, coeffs, norm_params)
        target_trends = predict_trend(target_lats, target_lons, target_elevs, coeffs, norm_params)

    # Step 2: Detrend
    residuals = well_matrix - well_trend[np.newaxis, :]

    # Step 3: EOF decomposition
    k = min(n_modes, n_wells, n_times)
    U, S, Vt = eof_decomposition(residuals, n_modes=k)

    total_var = np.sum(S**2)
    var_explained = np.cumsum(S**2) / total_var

    # Step 4: Interpolate loadings to targets
    if interpolation_method == "graph" and n_targets <= 5000:
        target_loadings_all = graph_laplacian_interpolate(
            well_lats, well_lons, well_elevs,
            target_lats, target_lons, target_elevs,
            Vt, k_neighbors=graph_k_neighbors,
            bandwidth_km=graph_bandwidth_km, elev_weight=graph_elev_weight,
        )
        predictions = target_trends[np.newaxis, :] + \
                      (U * S[np.newaxis, :]) @ target_loadings_all
    else:
        predictions = np.zeros((n_times, n_targets))
        for g in range(n_targets):
            target_loadings = interpolate_loadings(
                target_lats[g], target_lons[g],
                well_lats, well_lons, Vt, max_neighbors=max_neighbors,
            )
            target_residual = U @ (S * target_loadings)
            predictions[:, g] = target_trends[g] + target_residual

    # Step 5: LNN temporal refinement (optional)
    lnn_applied = 0
    if use_lnn and HAS_LNN:
        months, aux_lk = _load_gldas_aux(n_times)
        if months is not None and aux_lk:
            n_refine = min(n_targets, lnn_max_targets)
            for g in range(n_refine):
                sel, _ = _idw_weights(target_lats[g], target_lons[g],
                                      well_lats, well_lons, max_neighbors=1)
                refined = lnn_refine_prediction(
                    predictions[:, g], well_matrix[:, sel[0]],
                    well_lats[sel[0]], well_lons[sel[0]],
                    aux_lk, months, seed=42 + g,
                )
                predictions[:, g] = refined
                lnn_applied += 1

    return predictions, {
        "trend_r2": trend_r2,
        "trend_rmse": trend_rmse,
        "n_modes": k,
        "var_explained": var_explained.tolist(),
        "lnn_refined": lnn_applied,
        "interpolation_method": interpolation_method,
    }


# ─── NetCDF generation (drop-in for kriging) ─────────────────────────────────

def mc_lnn_field_generate(
    x_coords, y_coords, years_df, grid_x, grid_y,
    well_elevations=None, grid_elevations=None,
    n_modes=10, max_neighbors=30, trend_degree=2,
    elev_cache=None,
):
    """Generate interpolated fields for all timesteps.

    Args:
        x_coords: Well longitudes
        y_coords: Well latitudes
        years_df: DataFrame (rows=wells, cols=dates)
        grid_x, grid_y: 1D grid axes
        well_elevations: Optional pre-fetched well elevations
        grid_elevations: Optional pre-fetched grid elevations
        n_modes: Number of EOF modes to retain
        elev_cache: Elevation cache dict

    Returns: dict with 'fields' (list of 2D arrays) and 'dates' and 'info'
    """
    n_wells = len(x_coords)
    dates = list(years_df.columns)
    n_times = len(dates)
    nx, ny = len(grid_x), len(grid_y)

    # Build well matrix
    well_matrix = np.full((n_times, n_wells), np.nan)
    for t, date in enumerate(dates):
        well_matrix[t, :] = years_df[date].values

    # Get elevations
    if elev_cache is None:
        elev_cache = {}

    if well_elevations is None:
        print("  [MC+LNN] Fetching well elevations...", flush=True)
        well_elevations = fetch_elevations(y_coords, x_coords, elev_cache)

    # Grid cell coordinates and elevations
    grid_lats = np.repeat(grid_y, nx)
    grid_lons = np.tile(grid_x, ny)

    if grid_elevations is None:
        print(f"  [MC+LNN] Fetching grid elevations ({len(grid_lats)} points)...", flush=True)
        grid_elevations = fetch_elevations(grid_lats, grid_lons, elev_cache)

    # Run pipeline
    print(f"  [MC+LNN] Running EOF interpolation: {n_wells} wells → "
          f"{nx}×{ny} grid, {n_times} timesteps, {n_modes} modes...", flush=True)

    predictions, info = mc_lnn_eof_interpolation(
        well_lats=y_coords, well_lons=x_coords, well_elevs=well_elevations,
        well_matrix=well_matrix,
        target_lats=grid_lats, target_lons=grid_lons, target_elevs=grid_elevations,
        n_modes=n_modes, max_neighbors=max_neighbors, trend_degree=trend_degree,
    )

    print(f"  [MC+LNN] Trend R²={info['trend_r2']:.4f}, RMSE={info['trend_rmse']:.2f}", flush=True)
    print(f"  [MC+LNN] EOF variance explained: "
          f"{info['var_explained'][:5]}" if len(info['var_explained']) >= 5
          else f"{info['var_explained']}", flush=True)

    # Reshape to 2D fields
    fields = []
    for t in range(n_times):
        field = predictions[t, :].reshape(ny, nx).T  # (nx, ny) to match kriging output
        fields.append(field)

    return {"fields": fields, "dates": dates, "info": info}


def generate_nc_file_mc_lnn(
    file_name, grid_x, grid_y, years_df,
    x_coords, y_coords, bbox, raster_extent,
    well_elevations=None, n_modes=10, max_neighbors=30,
):
    """Generate NetCDF using MC+LNN EOF interpolation.
    Drop-in replacement for generate_nc_file().
    """
    if netCDF4 is None:
        raise ImportError("netCDF4 required: pip install netCDF4")

    t0 = _time.time()
    print(f"[MC+LNN] Interpolating {len(years_df.columns)} timesteps "
          f"from {len(x_coords)} wells...", flush=True)

    result = mc_lnn_field_generate(
        x_coords, y_coords, years_df, grid_x, grid_y,
        well_elevations=well_elevations, n_modes=n_modes,
        max_neighbors=max_neighbors,
    )

    elapsed = _time.time() - t0
    print(f"[MC+LNN] Done in {elapsed:.1f}s", flush=True)

    # Write NetCDF
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file_name)
    h = netCDF4.Dataset(file_path, "w", format="NETCDF4")

    h.createDimension("time", 0)
    h.createDimension("lat", len(grid_y))
    h.createDimension("lon", len(grid_x))

    latitude = h.createVariable("lat", np.float64, ("lat"))
    longitude = h.createVariable("lon", np.float64, ("lon"))
    time_dim = h.createVariable("time", np.float64, ("time"), fill_value="NaN")
    ts_value = h.createVariable("tsvalue", np.float64, ("time", "lon", "lat"), fill_value=-9999)

    latitude.long_name = "Latitude"
    latitude.units = "degrees_north"
    latitude.axis = "Y"
    longitude.long_name = "Longitude"
    longitude.units = "degrees_east"
    longitude.axis = "X"
    time_dim.axis = "T"
    time_dim.units = "days since 0001-01-01 00:00:00 UTC"

    latitude[:] = grid_y[:]
    longitude[:] = grid_x[:]

    for t, date in enumerate(result["dates"]):
        time_dim[t] = date.toordinal() if hasattr(date, 'toordinal') else pd.Timestamp(date).toordinal()
        ts_value[t, :, :] = result["fields"][t]

    h.close()
    return Path(file_path)


# ─── CV ───────────────────────────────────────────────────────────────────────

def eof_interpolation_cv(well_lats, well_lons, well_elevs, well_matrix,
                         target_indices, n_modes=10, max_neighbors=30,
                         use_lnn=False, interpolation_method="idw",
                         graph_bandwidth_km=50.0, graph_elev_weight=0.3,
                         trend_method="poly"):
    """Leave-one-out CV using EOF interpolation, optionally with LNN refinement."""
    n_times, n_wells = well_matrix.shape

    # Load GLDAS aux if LNN is enabled
    months, aux_lk = None, {}
    if use_lnn and HAS_LNN:
        months, aux_lk = _load_gldas_aux(n_times)

    results = []

    for wi, target_idx in enumerate(target_indices):
        truth = well_matrix[:, target_idx].copy()

        other = np.ones(n_wells, dtype=bool)
        other[target_idx] = False
        ow_lats, ow_lons = well_lats[other], well_lons[other]
        ow_elevs = well_elevs[other]
        ow_matrix = well_matrix[:, other]

        # EOF pipeline
        pred, info = mc_lnn_eof_interpolation(
            ow_lats, ow_lons, ow_elevs, ow_matrix,
            np.array([well_lats[target_idx]]),
            np.array([well_lons[target_idx]]),
            np.array([well_elevs[target_idx]]),
            n_modes=n_modes, max_neighbors=max_neighbors,
            use_lnn=False,
            interpolation_method=interpolation_method,
            graph_bandwidth_km=graph_bandwidth_km,
            graph_elev_weight=graph_elev_weight,
            trend_method=trend_method,
            exclude_well_idx=None,  # already excluded via ow_* arrays
        )
        eof_pred = pred[:, 0]

        # LNN refinement of EOF prediction
        lnn_pred = eof_pred.copy()
        if use_lnn and HAS_LNN and months is not None and aux_lk:
            # Find nearest well for transfer
            sel_nn, _ = _idw_weights(well_lats[target_idx], well_lons[target_idx],
                                     ow_lats, ow_lons, max_neighbors=1)
            nearest_idx = sel_nn[0]
            lnn_pred = lnn_refine_prediction(
                eof_pred, ow_matrix[:, nearest_idx],
                ow_lats[nearest_idx], ow_lons[nearest_idx],
                aux_lk, months, seed=42 + wi,
            )

        # IDW baseline
        sel, weights = _idw_weights(well_lats[target_idx], well_lons[target_idx],
                                    ow_lats, ow_lons, max_neighbors=max_neighbors)
        idw_pred = ow_matrix[:, sel] @ weights

        def _m(obs, p):
            err = p - obs
            rmse = float(np.sqrt(np.mean(err**2)))
            mae = float(np.mean(np.abs(err)))
            mo, mp = np.mean(obs), np.mean(p)
            so, sp = np.std(obs, ddof=1), np.std(p, ddof=1)
            r = float(np.corrcoef(obs, p)[0, 1]) if so > 1e-10 and sp > 1e-10 else 0.0
            ss_tot = float(np.sum((obs - mo)**2))
            r2 = 1.0 - float(np.sum(err**2)) / (ss_tot + 1e-12) if ss_tot > 1e-12 else float("nan")
            alpha = sp / so if so > 1e-10 else 1.0
            beta = mp / mo if abs(mo) > 1e-10 else 1.0
            kge = 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
            return {"kge": float(kge), "r2": float(r2), "rmse": rmse, "mae": mae}

        row = {
            "target_idx": target_idx,
            "trend_r2": info["trend_r2"],
            **{f"eof_{k}": v for k, v in _m(truth, eof_pred).items()},
            **{f"idw_{k}": v for k, v in _m(truth, idw_pred).items()},
        }
        if use_lnn:
            row.update({f"lnn_{k}": v for k, v in _m(truth, lnn_pred).items()})
        results.append(row)

    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))

    IMPUTED_CSV = os.path.join(PROJECT_ROOT, "output",
        "full_imputed_series_alleligible_iterative_softimpute_standalone.csv")
    RAW_CSV = os.path.join(PROJECT_ROOT, "datas",
        "measurements_till_2023_to_lnn_imputation.csv")
    ELEV_CACHE_FILE = os.path.join(APP_DIR, "elevation_cache.json")

    print("=" * 70)
    print("Spatial Detrending + EOF Interpolation — LOO CV")
    print("=" * 70)

    imputed_df = pd.read_csv(IMPUTED_CSV)
    imputed_df["Well_ID"] = imputed_df["Well_ID"].astype(str)
    imputed_df["Date"] = pd.to_datetime(imputed_df["Date"])
    imputed_df = imputed_df.sort_values(["Well_ID", "Date"])

    raw_df = pd.read_csv(RAW_CSV)
    raw_df["Date"] = pd.to_datetime(raw_df["Date"])
    raw_df = raw_df[(raw_df["Date"].dt.year >= 2000) & (raw_df["Date"].dt.year <= 2023)]
    raw_df["Well_ID"] = raw_df["Well_ID"].astype(str)

    well_ids = sorted(imputed_df["Well_ID"].unique())
    dates = sorted(imputed_df["Date"].unique())
    n_wells, n_times = len(well_ids), len(dates)

    well_idx_map = {wid: i for i, wid in enumerate(well_ids)}
    date_idx_map = {d: i for i, d in enumerate(dates)}

    well_matrix = np.full((n_times, n_wells), np.nan)
    for _, row in imputed_df.iterrows():
        well_matrix[date_idx_map[row["Date"]], well_idx_map[row["Well_ID"]]] = row["final_wte"]

    coords = raw_df.groupby("Well_ID")[["lat_dec", "long_dec"]].first()
    well_lats = np.array([coords.loc[wid, "lat_dec"] for wid in well_ids])
    well_lons = np.array([coords.loc[wid, "long_dec"] for wid in well_ids])

    # Elevations
    elev_cache = {}
    if os.path.exists(ELEV_CACHE_FILE):
        with open(ELEV_CACHE_FILE) as f:
            elev_cache = json.load(f)

    print("Fetching elevations...")
    well_elevs = fetch_elevations(well_lats, well_lons, elev_cache)
    with open(ELEV_CACHE_FILE, "w") as f:
        json.dump(elev_cache, f)

    print(f"Wells: {n_wells}, Times: {n_times}")
    print(f"Elevation range: {well_elevs.min():.0f} to {well_elevs.max():.0f} m")

    # Target wells
    raw_counts = raw_df.groupby("Well_ID")["Date"].nunique().reset_index(name="n_obs")
    raw_counts = raw_counts[raw_counts["Well_ID"].isin(well_ids)]
    raw_counts = raw_counts.sort_values("n_obs", ascending=False)
    target_well_ids = raw_counts.head(30)["Well_ID"].tolist()
    target_indices = [well_idx_map[wid] for wid in target_well_ids]

    print(f"Targets: {len(target_indices)}")
    print()

    # Test Polynomial vs XGBoost trend
    print("--- Polynomial vs XGBoost Trend (IDW loading interp, 20 modes) ---")
    for trend in ["poly", "xgb"]:
        t0 = _time.time()
        results = eof_interpolation_cv(well_lats, well_lons, well_elevs, well_matrix,
                                        target_indices, n_modes=20, max_neighbors=30,
                                        trend_method=trend)
        elapsed = _time.time() - t0
        eof_kges = [r["eof_kge"] for r in results]
        eof_rmses = [r["eof_rmse"] for r in results]
        idw_kges = [r["idw_kge"] for r in results]
        eof_wins = sum(1 for e, i in zip(eof_kges, idw_kges) if e > i)
        trend_r2 = results[0]["trend_r2"]
        label = "XGBoost" if trend == "xgb" else "Poly   "
        print(f"  {label}: KGE med={np.median(eof_kges):.4f} mean={np.mean(eof_kges):.4f} "
              f"RMSE={np.mean(eof_rmses):.2f} | trend_R²={trend_r2:.4f} | "
              f">IDW: {eof_wins}/30 | {elapsed:.1f}s")

    # XGBoost with different mode counts
    print()
    print("--- XGBoost Trend + EOF at different mode counts ---")
    for n_modes in [5, 10, 20, 50]:
        t0 = _time.time()
        results = eof_interpolation_cv(well_lats, well_lons, well_elevs, well_matrix,
                                        target_indices, n_modes=n_modes, max_neighbors=30,
                                        trend_method="xgb")
        elapsed = _time.time() - t0
        eof_kges = [r["eof_kge"] for r in results]
        eof_rmses = [r["eof_rmse"] for r in results]
        idw_kges = [r["idw_kge"] for r in results]
        eof_wins = sum(1 for e, i in zip(eof_kges, idw_kges) if e > i)
        print(f"  {n_modes:>2} modes: KGE med={np.median(eof_kges):.4f} "
              f"RMSE={np.mean(eof_rmses):.2f} | >IDW: {eof_wins}/30 | {elapsed:.1f}s")

    # Detailed comparison: XGBoost vs Poly
    print()
    print("--- Detailed: XGBoost vs Polynomial (20 modes) ---")
    results_xgb = eof_interpolation_cv(well_lats, well_lons, well_elevs, well_matrix,
                                        target_indices, n_modes=20, max_neighbors=30,
                                        trend_method="xgb")
    results_poly = eof_interpolation_cv(well_lats, well_lons, well_elevs, well_matrix,
                                         target_indices, n_modes=20, max_neighbors=30,
                                         trend_method="poly")

    print(f"{'Well_ID':>20} | {'XGB_KGE':>8} {'XGB_RMSE':>8} | {'Poly_KGE':>8} {'Poly_RMSE':>9} | {'IDW_KGE':>8}")
    print("-" * 90)
    for (rx, rp, wid) in sorted(zip(results_xgb, results_poly, target_well_ids),
                                 key=lambda x: x[0]["eof_kge"], reverse=True):
        print(f"{wid:>20} | {rx['eof_kge']:>8.4f} {rx['eof_rmse']:>8.2f} | "
              f"{rp['eof_kge']:>8.4f} {rp['eof_rmse']:>9.2f} | "
              f"{rp['idw_kge']:>8.4f}")

    xgb_kges = [r["eof_kge"] for r in results_xgb]
    poly_kges = [r["eof_kge"] for r in results_poly]
    xgb_rmses = [r["eof_rmse"] for r in results_xgb]
    poly_rmses = [r["eof_rmse"] for r in results_poly]

    print()
    print(f"XGBoost EOF: KGE med={np.median(xgb_kges):.4f} mean={np.mean(xgb_kges):.4f} RMSE={np.mean(xgb_rmses):.2f}")
    print(f"Poly EOF:    KGE med={np.median(poly_kges):.4f} mean={np.mean(poly_kges):.4f} RMSE={np.mean(poly_rmses):.2f}")
    xgb_wins = sum(1 for x, p in zip(xgb_kges, poly_kges) if x > p)
    print(f"XGBoost > Poly: {xgb_wins}/30")
