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

import os
import tempfile
import time as _time
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

try:
    import netCDF4
except ImportError:
    netCDF4 = None

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
    # Weighted average of loadings
    return well_loadings[:, sel] @ weights


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
) -> np.ndarray:
    """Full pipeline: Spatial Trend + EOF + IDW loading interpolation.

    Returns: nTimes × nTargets array of predicted WTE values.
    """
    n_times, n_wells = well_matrix.shape
    n_targets = len(target_lats)

    # Step 1: Fit spatial trend
    well_means = np.mean(well_matrix, axis=0)
    coeffs, norm_params, trend_r2, trend_rmse = fit_spatial_trend(
        well_lats, well_lons, well_elevs, well_means, degree=trend_degree
    )

    # Step 2: Detrend — subtract both temporal mean AND spatial trend
    well_trend = predict_trend(well_lats, well_lons, well_elevs, coeffs, norm_params)
    residuals = well_matrix - well_trend[np.newaxis, :]  # remove spatial trend from each column

    # Step 3: EOF decomposition of residuals
    k = min(n_modes, n_wells, n_times)
    U, S, Vt = eof_decomposition(residuals, n_modes=k)

    # Variance explained
    total_var = np.sum(S**2)
    var_explained = np.cumsum(S**2) / total_var

    # Step 4: For each target, interpolate loadings and reconstruct
    predictions = np.zeros((n_times, n_targets))
    target_trends = predict_trend(target_lats, target_lons, target_elevs, coeffs, norm_params)

    for g in range(n_targets):
        # Interpolate EOF loadings
        target_loadings = interpolate_loadings(
            target_lats[g], target_lons[g],
            well_lats, well_lons, Vt, max_neighbors=max_neighbors,
        )

        # Reconstruct: residual = U × S × loadings
        target_residual = U @ (S * target_loadings)

        # Final prediction: trend + residual
        predictions[:, g] = target_trends[g] + target_residual

    return predictions, {
        "trend_r2": trend_r2,
        "trend_rmse": trend_rmse,
        "n_modes": k,
        "var_explained": var_explained.tolist(),
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
                         target_indices, n_modes=10, max_neighbors=30):
    """Leave-one-out CV using EOF interpolation."""
    n_times, n_wells = well_matrix.shape
    results = []

    for target_idx in target_indices:
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
        )
        eof_pred = pred[:, 0]

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

        results.append({
            "target_idx": target_idx,
            "trend_r2": info["trend_r2"],
            **{f"eof_{k}": v for k, v in _m(truth, eof_pred).items()},
            **{f"idw_{k}": v for k, v in _m(truth, idw_pred).items()},
        })

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

    # Test different mode counts
    for n_modes in [3, 5, 10, 20, 50]:
        t0 = _time.time()
        results = eof_interpolation_cv(well_lats, well_lons, well_elevs, well_matrix,
                                        target_indices, n_modes=n_modes, max_neighbors=30)
        elapsed = _time.time() - t0

        eof_kges = [r["eof_kge"] for r in results]
        idw_kges = [r["idw_kge"] for r in results]
        eof_rmses = [r["eof_rmse"] for r in results]
        idw_rmses = [r["idw_rmse"] for r in results]
        eof_wins = sum(1 for e, i in zip(eof_kges, idw_kges) if e > i)
        trend_r2 = results[0]["trend_r2"]

        print(f"Modes {n_modes:>3}: EOF KGE med={np.median(eof_kges):.4f} mean={np.mean(eof_kges):.4f} "
              f"RMSE={np.mean(eof_rmses):.2f} | "
              f"IDW KGE med={np.median(idw_kges):.4f} RMSE={np.mean(idw_rmses):.2f} | "
              f"EOF>IDW: {eof_wins}/30 | trend_R²={trend_r2:.4f} | {elapsed:.1f}s")

    # Detailed for best
    print()
    print("Per-well detail (10 modes):")
    results = eof_interpolation_cv(well_lats, well_lons, well_elevs, well_matrix,
                                    target_indices, n_modes=10, max_neighbors=30)
    print(f"{'Well_ID':>20} | {'EOF_KGE':>8} {'EOF_RMSE':>8} | {'IDW_KGE':>8} {'IDW_RMSE':>8} | {'Win':>6}")
    print("-" * 80)
    for r, wid in sorted(zip(results, target_well_ids),
                          key=lambda x: x[0]["eof_kge"], reverse=True):
        win = "EOF" if r["eof_kge"] > r["idw_kge"] else "IDW"
        print(f"{wid:>20} | {r['eof_kge']:>8.4f} {r['eof_rmse']:>8.2f} | "
              f"{r['idw_kge']:>8.4f} {r['idw_rmse']:>8.2f} | {win:>6}")
