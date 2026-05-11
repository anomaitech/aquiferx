#!/usr/bin/env python3
"""
Pure MC Spatial Interpolation with Auxiliary Rows (GSLB)

Approach: Build a matrix where BOTH wells and grid cells are columns,
and spatial auxiliary rows (lat, lon, elevation) provide the spatial
context MC needs to predict at unknown grid locations.

Matrix structure:
  Rows:
    [0..nTimes-1]    temporal values (well cols have data, grid cols NaN)
    [nTimes]         latitude  (known for all columns)
    [nTimes+1]       longitude (known for all columns)
    [nTimes+2]       elevation (known for all columns, from DEM)
    [nTimes+3]       sin(2π·month/12) — seasonal encoding (per timestep)
    [nTimes+4]       cos(2π·month/12)

  Columns:
    [0..nWells-1]    wells (temporal rows observed, aux rows observed)
    [nWells..]       grid cells (temporal rows NaN, aux rows observed)

The aux rows are FULLY OBSERVED for both wells and grid cells. They
anchor the low-rank structure — telling MC "this grid cell at this
lat/lon/elevation should behave like nearby wells at similar elevations."

Normalization: per-row z-score (same as MC+LNN imputation pipeline).

CV: Leave-one-well-out — remove a well, add its location as a grid cell,
    predict, compare to truth.

Grid resolutions: 0.1°, 0.2°, 0.5°
"""

from __future__ import annotations

import json
import os
import time
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
OUTPUT_DIR = APP_DIR

IMPUTED_CSV = os.path.join(PROJECT_ROOT, "output",
    "full_imputed_series_alleligible_iterative_softimpute_standalone.csv")
RAW_CSV = os.path.join(PROJECT_ROOT, "datas",
    "measurements_till_2023_to_lnn_imputation.csv")

ELEVATION_CACHE = os.path.join(OUTPUT_DIR, "elevation_cache.json")

MC_MAX_ITERATIONS = 100
MC_TOLERANCE = 1e-6
RANK_OPTIONS = [5, 8, 10, 15, 20, 30, 40]

GRID_RESOLUTIONS = [0.1, 0.2, 0.5]
N_TARGET_WELLS = 30


# ─── Elevation lookup (Open-Meteo Copernicus DEM) ────────────────────────────

def fetch_elevations_batch(lats: List[float], lons: List[float],
                           cache: Dict[str, float]) -> List[float]:
    """Fetch elevations via Open-Meteo API, with caching."""
    results = [0.0] * len(lats)
    to_fetch = []
    to_fetch_idx = []

    for i, (lat, lon) in enumerate(zip(lats, lons)):
        key = f"{lat:.6f},{lon:.6f}"
        if key in cache:
            results[i] = cache[key]
        else:
            to_fetch.append((i, lat, lon))
            to_fetch_idx.append(i)

    if to_fetch:
        # Open-Meteo: max 100 per request
        for batch_start in range(0, len(to_fetch), 100):
            batch = to_fetch[batch_start:batch_start + 100]
            lat_str = ",".join(f"{lat:.6f}" for _, lat, _ in batch)
            lon_str = ",".join(f"{lon:.6f}" for _, _, lon in batch)
            try:
                url = f"https://api.open-meteo.com/v1/elevation?latitude={lat_str}&longitude={lon_str}"
                resp = requests.get(url, timeout=30)
                if resp.ok:
                    data = resp.json()
                    elevs = data.get("elevation", [])
                    for j, (idx, lat, lon) in enumerate(batch):
                        if j < len(elevs) and elevs[j] is not None:
                            val = float(elevs[j])
                            results[idx] = val
                            cache[f"{lat:.6f},{lon:.6f}"] = val
            except Exception as e:
                print(f"  Elevation fetch error: {e}")

        print(f"  Fetched {len(to_fetch)} elevations from Open-Meteo")

    return results


# ─── SoftImpute with z-score normalization ────────────────────────────────────

def soft_impute_normalized(matrix: np.ndarray, obs_mask: np.ndarray,
                           rank: int, max_iterations: int = 100,
                           tolerance: float = 1e-6) -> np.ndarray:
    """SoftImpute with per-row z-score normalization (matches MC+LNN pipeline)."""
    n_rows, n_cols = matrix.shape

    # Per-row normalization
    row_means = np.zeros(n_rows)
    row_stds = np.ones(n_rows)
    M_norm = matrix.copy()

    for r in range(n_rows):
        obs_vals = M_norm[r, obs_mask[r, :]]
        if len(obs_vals) >= 2:
            row_means[r] = np.mean(obs_vals)
            row_stds[r] = max(np.std(obs_vals), 1e-10)
        elif len(obs_vals) == 1:
            row_means[r] = obs_vals[0]
        M_norm[r, :] = (M_norm[r, :] - row_means[r]) / row_stds[r]

    # Fill NaN with 0 (after normalization, 0 = row mean)
    M = M_norm.copy()
    M[np.isnan(M)] = 0.0

    # Normalized observed values for mask restoration
    norm_obs = M_norm.copy()

    for _ in range(max_iterations):
        M_prev = M.copy()
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        S_trunc = S.copy()
        S_trunc[min(rank, len(S)):] = 0
        M_new = U @ np.diag(S_trunc) @ Vt
        M_new[obs_mask] = norm_obs[obs_mask]
        M = M_new

        diff = np.linalg.norm(M - M_prev)
        denom = max(np.linalg.norm(M_prev), 1e-12)
        if diff / denom <= tolerance:
            break

    # Denormalize
    result = M * row_stds[:, np.newaxis] + row_means[:, np.newaxis]
    return result


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(obs, pred):
    obs, pred = np.asarray(obs), np.asarray(pred)
    if len(obs) < 2:
        return {"kge": float("nan"), "r2": float("nan"),
                "rmse": float("nan"), "mae": float("nan")}
    err = pred - obs
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    mo, mp = np.mean(obs), np.mean(pred)
    so, sp = np.std(obs, ddof=1), np.std(pred, ddof=1)
    r = float(np.corrcoef(obs, pred)[0, 1]) if so > 1e-10 and sp > 1e-10 else 0.0
    ss_tot = float(np.sum((obs - mo)**2))
    r2 = 1.0 - float(np.sum(err**2)) / (ss_tot + 1e-12) if ss_tot > 1e-12 else float("nan")
    alpha = sp / so if so > 1e-10 else 1.0
    beta = mp / mo if abs(mo) > 1e-10 else 1.0
    kge = 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return {"kge": float(kge), "r2": float(r2), "rmse": rmse, "mae": mae, "r": float(r)}


def idw_series(target_lat, target_lon, src_lats, src_lons, matrix,
               exponent=2.0, max_neighbors=20):
    dlat = src_lats - target_lat
    dlon = (src_lons - target_lon) * np.cos(np.radians(target_lat))
    dists = np.sqrt(dlat**2 + dlon**2)
    order = np.argsort(dists)
    sel = order[:max_neighbors]
    if dists[sel[0]] < 1e-8:
        return matrix[:, sel[0]].copy()
    w = 1.0 / (dists[sel] ** exponent + 1e-12)
    w /= w.sum()
    return matrix[:, sel] @ w


# ─── Grid ─────────────────────────────────────────────────────────────────────

def build_grid(lat_min, lat_max, lon_min, lon_max, resolution):
    ny = max(1, int(np.ceil((lat_max - lat_min) / resolution)))
    nx = max(1, int(np.ceil((lon_max - lon_min) / resolution)))
    lats, lons = [], []
    for row in range(ny):
        for col in range(nx):
            lats.append(lat_min + (row + 0.5) * resolution)
            lons.append(lon_min + (col + 0.5) * resolution)
    return np.array(lats), np.array(lons), ny, nx


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 70)
    print("Pure MC Spatial Interpolation with Aux Rows — LOO CV (GSLB)")
    print("=" * 70)

    # Load data
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
    n_wells = len(well_ids)
    n_times = len(dates)

    well_idx_map = {wid: i for i, wid in enumerate(well_ids)}
    date_idx_map = {d: i for i, d in enumerate(dates)}

    well_matrix = np.full((n_times, n_wells), np.nan)
    for _, row in imputed_df.iterrows():
        well_matrix[date_idx_map[row["Date"]], well_idx_map[row["Well_ID"]]] = row["final_wte"]

    coords = raw_df.groupby("Well_ID")[["lat_dec", "long_dec"]].first()
    well_lats = np.array([coords.loc[wid, "lat_dec"] for wid in well_ids])
    well_lons = np.array([coords.loc[wid, "long_dec"] for wid in well_ids])

    lat_min, lat_max = well_lats.min() - 0.1, well_lats.max() + 0.1
    lon_min, lon_max = well_lons.min() - 0.1, well_lons.max() + 0.1

    # Target wells
    raw_counts = raw_df.groupby("Well_ID")["Date"].nunique().reset_index(name="n_obs")
    raw_counts = raw_counts[raw_counts["Well_ID"].isin(well_ids)]
    raw_counts = raw_counts.sort_values("n_obs", ascending=False)
    target_wells = raw_counts.head(N_TARGET_WELLS)["Well_ID"].tolist()

    print(f"Wells: {n_wells}, Times: {n_times}")
    print(f"Target wells: {len(target_wells)}")
    print()

    # ── Fetch elevations ──────────────────────────────────────────────────
    print("Fetching well elevations...")
    elev_cache: Dict[str, float] = {}
    if os.path.exists(ELEVATION_CACHE):
        with open(ELEVATION_CACHE) as f:
            elev_cache = json.load(f)

    well_elevs = fetch_elevations_batch(well_lats.tolist(), well_lons.tolist(), elev_cache)
    well_elevs = np.array(well_elevs)
    print(f"  Well elevations: {well_elevs.min():.0f} to {well_elevs.max():.0f} m")

    all_rows = []

    for res in GRID_RESOLUTIONS:
        grid_lats, grid_lons, ny, nx = build_grid(lat_min, lat_max, lon_min, lon_max, res)
        n_grid = len(grid_lats)
        print(f"\nGrid {res}°: {ny}×{nx} = {n_grid} cells")

        # Fetch grid elevations
        print("  Fetching grid elevations...")
        grid_elevs = fetch_elevations_batch(grid_lats.tolist(), grid_lons.tolist(), elev_cache)
        grid_elevs = np.array(grid_elevs)

        # Save cache after each resolution
        with open(ELEVATION_CACHE, "w") as f:
            json.dump(elev_cache, f)

        # ── Adaptive rank selection ───────────────────────────────────
        # Build a small test: hold out 5 wells, build full matrix, find best rank
        rng = np.random.default_rng(42)
        val_wells = rng.choice(n_wells, size=min(5, n_wells // 10), replace=False)

        print("  Selecting best rank...")
        best_rank = 10
        best_err = float("inf")

        for k in RANK_OPTIONS:
            # Build matrix with held-out wells treated as grid cells
            # For speed, use wells-only matrix (no grid cells) for rank selection
            n_aux = 5  # lat, lon, elev, sin, cos
            n_rows_total = n_times + n_aux
            n_cols = n_wells

            M = np.full((n_rows_total, n_cols), np.nan)
            obs = np.zeros((n_rows_total, n_cols), dtype=bool)

            # Temporal rows
            M[:n_times, :] = well_matrix
            obs[:n_times, :] = True

            # Aux rows (observed for all)
            M[n_times, :] = well_lats
            M[n_times + 1, :] = well_lons
            M[n_times + 2, :] = well_elevs
            for t in range(n_times):
                M[n_times + 3, :] = np.sin(2 * np.pi * (t % 12) / 12.0)
                M[n_times + 4, :] = np.cos(2 * np.pi * (t % 12) / 12.0)
            # Use mean seasonal encoding for aux
            M[n_times + 3, :] = 0.0  # sin mean = 0
            M[n_times + 4, :] = 0.0  # cos mean = 0
            obs[n_times:, :] = True

            # Hold out validation wells' temporal data
            for vw in val_wells:
                M[:n_times, vw] = np.nan
                obs[:n_times, vw] = False

            if k >= min(M.shape):
                continue

            completed = soft_impute_normalized(M, obs, rank=k,
                                               max_iterations=30, tolerance=MC_TOLERANCE)
            err = 0.0
            cnt = 0
            for vw in val_wells:
                err += np.sum((completed[:n_times, vw] - well_matrix[:, vw])**2)
                cnt += n_times
            mse = err / cnt if cnt else float("inf")
            if mse < best_err:
                best_err = mse
                best_rank = k

        print(f"  Best rank: {best_rank} (MSE={best_err:.4f})")

        # ── LOO CV ────────────────────────────────────────────────────
        t_res = time.time()

        for wi, target_wid in enumerate(target_wells):
            w_col = well_idx_map[target_wid]
            truth = well_matrix[:, w_col].copy()
            t_lat, t_lon = well_lats[w_col], well_lons[w_col]
            t_elev = well_elevs[w_col]

            # Find nearest grid cell
            dlat = grid_lats - t_lat
            dlon = (grid_lons - t_lon) * np.cos(np.radians(t_lat))
            target_grid_idx = int(np.argmin(dlat**2 + dlon**2))

            # Other wells
            other = np.ones(n_wells, dtype=bool)
            other[w_col] = False
            n_other = int(other.sum())

            # ── Build MC matrix: (nTimes + nAux) × (nOtherWells + 1 target grid cell) ──
            n_aux = 3  # lat, lon, elev
            n_rows_total = n_times + n_aux
            n_cols = n_other + 1  # other wells + target grid cell

            M = np.full((n_rows_total, n_cols), np.nan)
            obs_mask = np.zeros((n_rows_total, n_cols), dtype=bool)

            # Temporal rows: wells observed, target grid cell NaN
            M[:n_times, :n_other] = well_matrix[:, other]
            obs_mask[:n_times, :n_other] = True

            # Target grid cell temporal rows: NaN (to be filled by MC)
            # M[:n_times, -1] stays NaN
            obs_mask[:n_times, -1] = False

            # Aux rows: OBSERVED for ALL columns (wells + grid cell)
            ow_lats = well_lats[other]
            ow_lons = well_lons[other]
            ow_elevs = well_elevs[other]

            # Lat row
            M[n_times, :n_other] = ow_lats
            M[n_times, -1] = grid_lats[target_grid_idx]
            obs_mask[n_times, :] = True

            # Lon row
            M[n_times + 1, :n_other] = ow_lons
            M[n_times + 1, -1] = grid_lons[target_grid_idx]
            obs_mask[n_times + 1, :] = True

            # Elevation row
            M[n_times + 2, :n_other] = ow_elevs
            M[n_times + 2, -1] = grid_elevs[target_grid_idx]
            obs_mask[n_times + 2, :] = True

            # Run MC
            completed = soft_impute_normalized(M, obs_mask, rank=best_rank,
                                               max_iterations=MC_MAX_ITERATIONS,
                                               tolerance=MC_TOLERANCE)

            mc_pred = completed[:n_times, -1]
            mc_metrics = compute_metrics(truth, mc_pred)

            # ── IDW baseline ──────────────────────────────────────────
            gc_lat = grid_lats[target_grid_idx]
            gc_lon = grid_lons[target_grid_idx]
            idw_pred = idw_series(gc_lat, gc_lon, ow_lats, ow_lons,
                                  well_matrix[:, other], exponent=2.0, max_neighbors=20)
            idw_metrics = compute_metrics(truth, idw_pred)

            all_rows.append({
                "resolution": res,
                "Well_ID": target_wid,
                "rank": best_rank,
                "true_mean": float(np.mean(truth)),
                "true_elev": t_elev,
                "grid_elev": grid_elevs[target_grid_idx],
                # Pure MC with aux
                "mc_kge": mc_metrics["kge"], "mc_r2": mc_metrics["r2"],
                "mc_rmse": mc_metrics["rmse"], "mc_mae": mc_metrics["mae"],
                "mc_r": mc_metrics.get("r", float("nan")),
                # IDW baseline
                "idw_kge": idw_metrics["kge"], "idw_r2": idw_metrics["r2"],
                "idw_rmse": idw_metrics["rmse"], "idw_mae": idw_metrics["mae"],
            })

            if (wi + 1) % 10 == 0 or wi + 1 == len(target_wells):
                elapsed = time.time() - t_res
                print(f"  [{wi+1}/{len(target_wells)}] {elapsed:.0f}s "
                      f"MC={mc_metrics['kge']:.4f} IDW={idw_metrics['kge']:.4f}",
                      flush=True)

    # ─── Results ──────────────────────────────────────────────────────────
    total_time = time.time() - t0
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "cv_mc_spatial_aux_detailed.csv"), index=False)

    lines = []
    lines.append("=" * 70)
    lines.append("PURE MC SPATIAL INTERPOLATION WITH AUX ROWS — LOO CV")
    lines.append("=" * 70)
    lines.append(f"Wells: {n_wells}, Times: {n_times}")
    lines.append(f"Target wells: {len(target_wells)}")
    lines.append(f"Aux rows: latitude, longitude, elevation")
    lines.append(f"Normalization: per-row z-score")
    lines.append(f"Total time: {total_time:.1f}s")
    lines.append("")

    for res in GRID_RESOLUTIONS:
        sub = df[df["resolution"] == res]
        if len(sub) == 0:
            continue

        lines.append(f"GRID: {res}° resolution (rank={sub['rank'].iloc[0]})")
        lines.append("-" * 70)
        methods = [("MC + aux rows", "mc"), ("IDW only", "idw")]
        lines.append(f"  {'Method':>16} | {'KGE_mean':>9} {'KGE_std':>8} {'KGE_med':>8} | "
                     f"{'R²_mean':>8} | {'RMSE_mean':>10} {'MAE_mean':>9}")
        lines.append("  " + "-" * 75)
        for name, pfx in methods:
            kge = sub[f"{pfx}_kge"]
            lines.append(
                f"  {name:>16} | {kge.mean():>9.4f} {kge.std():>8.4f} {kge.median():>8.4f} | "
                f"{sub[f'{pfx}_r2'].mean():>8.4f} | {sub[f'{pfx}_rmse'].mean():>10.3f} "
                f"{sub[f'{pfx}_mae'].mean():>9.3f}"
            )
        mc_wins = (sub["mc_kge"] > sub["idw_kge"]).sum()
        lines.append(f"  MC > IDW: {mc_wins}/{len(sub)} wells")
        lines.append("")

    # Per-well detail for best resolution
    if len(df):
        best_res = df.groupby("resolution")["mc_kge"].mean().idxmax()
        sub = df[df["resolution"] == best_res].sort_values("mc_kge", ascending=False)
        lines.append(f"PER-WELL DETAIL (grid={best_res}°):")
        lines.append(f"{'Well_ID':>20} {'MC_KGE':>8} {'MC_R²':>8} {'MC_RMSE':>8} | "
                     f"{'IDW_KGE':>8} {'IDW_RMSE':>8} | {'Elev':>6}")
        lines.append("-" * 85)
        for _, row in sub.iterrows():
            lines.append(
                f"{row['Well_ID']:>20} {row['mc_kge']:>8.4f} {row['mc_r2']:>8.4f} "
                f"{row['mc_rmse']:>8.2f} | {row['idw_kge']:>8.4f} {row['idw_rmse']:>8.2f} | "
                f"{row['true_elev']:>6.0f}"
            )

    summary_text = "\n".join(lines)
    print()
    print(summary_text)

    with open(os.path.join(OUTPUT_DIR, "cv_mc_spatial_aux_summary.txt"), "w") as f:
        f.write(summary_text + "\n")

    meta = {
        "n_wells": n_wells, "n_times": n_times, "n_targets": len(target_wells),
        "grid_resolutions": GRID_RESOLUTIONS, "rank_options": RANK_OPTIONS,
        "aux_rows": ["latitude", "longitude", "elevation"],
        "normalization": "per-row z-score",
        "elapsed_seconds": total_time,
    }
    with open(os.path.join(OUTPUT_DIR, "cv_mc_spatial_aux_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nOutput: cv_mc_spatial_aux_detailed.csv, cv_mc_spatial_aux_summary.txt")


if __name__ == "__main__":
    main()
