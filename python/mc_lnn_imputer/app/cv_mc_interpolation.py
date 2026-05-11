#!/usr/bin/env python3
"""
MC Spatial Interpolation CV with Grids (GSLB)

Tests MC (SoftImpute) for predicting WTE at unknown grid locations using
the fully imputed dataset (592 wells, 288 months).

Anomaly-based approach:
  1. Compute per-well temporal mean → spatial trend
  2. Build anomaly matrix: (WTE - mean) for wells, NaN for grid cells
  3. Initialize grid cell anomalies via IDW from wells
  4. Initialize grid cell means via IDW from wells
  5. MC refines the anomalies using low-rank temporal-spatial structure
  6. Final grid value = IDW_mean + MC_anomaly

Matrix: nTimes × (nWells + nGridCells)
  - Well columns: observed anomalies (fixed during MC)
  - Grid columns: initialized by IDW, refined by MC

CV: Leave-one-well-out — remove a well from the well columns,
    predict at the grid cell where it was, compare to truth.

Grid resolutions: 0.1°, 0.2°, 0.5°
Compares: Anomaly MC, IDW only, Raw MC
"""

from __future__ import annotations

import json
import os
import time
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
OUTPUT_DIR = APP_DIR

IMPUTED_CSV = os.path.join(PROJECT_ROOT, "output",
    "full_imputed_series_alleligible_iterative_softimpute_standalone.csv")
RAW_CSV = os.path.join(PROJECT_ROOT, "datas",
    "measurements_till_2023_to_lnn_imputation.csv")

MC_MAX_ITERATIONS = 100
MC_TOLERANCE = 1e-5
RANK_OPTIONS = [5, 8, 10, 15, 20, 30]

IDW_EXPONENT = 2.0
IDW_MAX_NEIGHBORS = 20

GRID_RESOLUTIONS = [0.1, 0.2, 0.5]
N_TARGET_WELLS = 30


# ─── IDW ──────────────────────────────────────────────────────────────────────

def idw_weights(target_lat, target_lon, src_lats, src_lons,
                exponent=2.0, max_neighbors=20):
    """Return (indices, weights) for IDW."""
    dlat = src_lats - target_lat
    dlon = (src_lons - target_lon) * np.cos(np.radians(target_lat))
    dists = np.sqrt(dlat**2 + dlon**2)
    order = np.argsort(dists)
    sel = order[:max_neighbors]
    if dists[sel[0]] < 1e-8:
        w = np.zeros(len(sel))
        w[0] = 1.0
        return sel, w
    w = 1.0 / (dists[sel] ** exponent + 1e-12)
    w /= w.sum()
    return sel, w


def idw_scalar(target_lat, target_lon, src_lats, src_lons, values,
               exponent=2.0, max_neighbors=20):
    sel, w = idw_weights(target_lat, target_lon, src_lats, src_lons,
                         exponent, max_neighbors)
    return float(np.sum(w * values[sel]))


def idw_series(target_lat, target_lon, src_lats, src_lons, matrix,
               exponent=2.0, max_neighbors=20):
    """matrix: nTimes × nSources"""
    sel, w = idw_weights(target_lat, target_lon, src_lats, src_lons,
                         exponent, max_neighbors)
    return matrix[:, sel] @ w


# ─── SoftImpute ───────────────────────────────────────────────────────────────

def soft_impute(matrix, obs_mask, rank, max_iterations=100, tolerance=1e-5):
    M = matrix.copy()
    for c in range(M.shape[1]):
        nans = np.isnan(M[:, c])
        if nans.any():
            vals = M[~nans, c]
            M[nans, c] = np.mean(vals) if len(vals) else 0.0

    for _ in range(max_iterations):
        M_prev = M.copy()
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        S_trunc = S.copy()
        S_trunc[min(rank, len(S)):] = 0
        M_new = U @ np.diag(S_trunc) @ Vt
        M_new[obs_mask] = matrix[obs_mask]
        M = M_new
        if np.linalg.norm(M - M_prev) / max(np.linalg.norm(M_prev), 1e-12) <= tolerance:
            break
    return M


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
    print("MC Spatial Interpolation — Grid-Based Anomaly LOO CV (GSLB)")
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

    well_means = np.mean(well_matrix, axis=0)
    anomaly_matrix = well_matrix - well_means[np.newaxis, :]

    lat_min, lat_max = well_lats.min() - 0.1, well_lats.max() + 0.1
    lon_min, lon_max = well_lons.min() - 0.1, well_lons.max() + 0.1

    raw_counts = raw_df.groupby("Well_ID")["Date"].nunique().reset_index(name="n_obs")
    raw_counts = raw_counts[raw_counts["Well_ID"].isin(well_ids)]
    raw_counts = raw_counts.sort_values("n_obs", ascending=False)
    target_wells = raw_counts.head(N_TARGET_WELLS)["Well_ID"].tolist()

    print(f"Wells: {n_wells}, Times: {n_times}")
    print(f"WTE range: {well_matrix.min():.1f} — {well_matrix.max():.1f}")
    print(f"Anomaly range: {anomaly_matrix.min():.2f} — {anomaly_matrix.max():.2f}")
    print(f"Grid resolutions: {GRID_RESOLUTIONS}°")
    print(f"Target wells: {len(target_wells)}")
    print()

    all_rows = []

    for res in GRID_RESOLUTIONS:
        grid_lats, grid_lons, ny, nx = build_grid(lat_min, lat_max, lon_min, lon_max, res)
        n_grid = len(grid_lats)
        print(f"Grid {res}°: {ny}×{nx} = {n_grid} cells")

        # ── Adaptive rank selection ───────────────────────────────────
        # Build full matrix: nTimes × (nWells + nGrid)
        # For rank selection, use wells only (hold out 10% of wells)
        rng = np.random.default_rng(42)
        val_wells = rng.choice(n_wells, size=max(3, n_wells // 10), replace=False)

        best_rank = 10
        best_err = float("inf")
        for k in RANK_OPTIONS:
            # Quick test on anomaly matrix with held-out wells
            A_test = anomaly_matrix.copy()
            for vw in val_wells:
                A_test[:, vw] = np.nan
                # IDW init
                other = np.ones(n_wells, dtype=bool)
                other[vw] = False
                A_test[:, vw] = idw_series(
                    well_lats[vw], well_lons[vw],
                    well_lats[other], well_lons[other],
                    anomaly_matrix[:, other]
                )
            obs_test = np.ones_like(A_test, dtype=bool)
            for vw in val_wells:
                obs_test[:, vw] = False
            A_test_orig = A_test.copy()
            A_test_orig[obs_test] = anomaly_matrix[obs_test]

            completed = soft_impute(A_test, obs_test, rank=k,
                                    max_iterations=30, tolerance=MC_TOLERANCE)
            err = 0.0
            cnt = 0
            for vw in val_wells:
                err += np.sum((completed[:, vw] - anomaly_matrix[:, vw])**2)
                cnt += n_times
            mse = err / cnt if cnt > 0 else float("inf")
            if mse < best_err:
                best_err = mse
                best_rank = k

        print(f"  Best rank: {best_rank} (anomaly MSE={best_err:.4f})")

        # ── LOO CV ────────────────────────────────────────────────────
        t_res = time.time()
        for wi, target_wid in enumerate(target_wells):
            w_col = well_idx_map[target_wid]
            truth = well_matrix[:, w_col].copy()
            truth_anom = anomaly_matrix[:, w_col].copy()
            t_lat, t_lon = well_lats[w_col], well_lons[w_col]

            # Find nearest grid cell to target well
            dlat = grid_lats - t_lat
            dlon = (grid_lons - t_lon) * np.cos(np.radians(t_lat))
            target_grid = int(np.argmin(dlat**2 + dlon**2))
            grid_dist = np.sqrt(dlat[target_grid]**2 + dlon[target_grid]**2)

            # Other wells (exclude target)
            other = np.ones(n_wells, dtype=bool)
            other[w_col] = False
            ow_lats, ow_lons = well_lats[other], well_lons[other]
            ow_matrix = well_matrix[:, other]
            ow_anomaly = anomaly_matrix[:, other]
            ow_means = well_means[other]

            # ── IDW baseline ──────────────────────────────────────────
            # Predict at grid cell location (not well location)
            gc_lat, gc_lon = grid_lats[target_grid], grid_lons[target_grid]
            idw_pred = idw_series(gc_lat, gc_lon, ow_lats, ow_lons,
                                  ow_matrix, IDW_EXPONENT, IDW_MAX_NEIGHBORS)
            idw_metrics = compute_metrics(truth, idw_pred)

            # ── Anomaly MC ────────────────────────────────────────────
            # IDW estimate of mean and anomaly at grid cell
            gc_mean = idw_scalar(gc_lat, gc_lon, ow_lats, ow_lons,
                                 ow_means, IDW_EXPONENT, IDW_MAX_NEIGHBORS)
            gc_anom_init = idw_series(gc_lat, gc_lon, ow_lats, ow_lons,
                                      ow_anomaly, IDW_EXPONENT, IDW_MAX_NEIGHBORS)

            # Build combined matrix: nTimes × (nOtherWells + 1 grid cell)
            n_other = ow_anomaly.shape[1]
            combined = np.column_stack([ow_anomaly, gc_anom_init])

            # Observed mask: wells observed, grid cell free
            obs_mask = np.ones((n_times, n_other + 1), dtype=bool)
            obs_mask[:, -1] = False

            # Store original for mask restoration
            combined_orig = combined.copy()
            combined_orig[:, :n_other] = ow_anomaly

            completed = soft_impute(combined, obs_mask, rank=best_rank,
                                    max_iterations=MC_MAX_ITERATIONS,
                                    tolerance=MC_TOLERANCE)
            mc_anom = completed[:, -1]
            amc_pred = gc_mean + mc_anom
            amc_metrics = compute_metrics(truth, amc_pred)

            # ── Raw MC (no anomaly) ───────────────────────────────────
            raw_init = idw_pred.copy()
            combined_raw = np.column_stack([ow_matrix, raw_init])
            obs_raw = np.ones((n_times, n_other + 1), dtype=bool)
            obs_raw[:, -1] = False
            combined_raw_orig = combined_raw.copy()
            combined_raw_orig[:, :n_other] = ow_matrix

            completed_raw = soft_impute(combined_raw, obs_raw, rank=best_rank,
                                        max_iterations=MC_MAX_ITERATIONS,
                                        tolerance=MC_TOLERANCE)
            rmc_pred = completed_raw[:, -1]
            rmc_metrics = compute_metrics(truth, rmc_pred)

            all_rows.append({
                "resolution": res,
                "Well_ID": target_wid,
                "grid_cell": target_grid,
                "grid_dist_deg": grid_dist,
                "true_mean": float(np.mean(truth)),
                "idw_mean_est": gc_mean,
                "mean_error": abs(gc_mean - np.mean(truth)),
                "rank": best_rank,
                # Anomaly MC
                "amc_kge": amc_metrics["kge"], "amc_r2": amc_metrics["r2"],
                "amc_rmse": amc_metrics["rmse"], "amc_mae": amc_metrics["mae"],
                # IDW
                "idw_kge": idw_metrics["kge"], "idw_r2": idw_metrics["r2"],
                "idw_rmse": idw_metrics["rmse"], "idw_mae": idw_metrics["mae"],
                # Raw MC
                "rmc_kge": rmc_metrics["kge"], "rmc_r2": rmc_metrics["r2"],
                "rmc_rmse": rmc_metrics["rmse"], "rmc_mae": rmc_metrics["mae"],
            })

            if (wi + 1) % 10 == 0 or wi + 1 == len(target_wells):
                elapsed = time.time() - t_res
                print(f"  [{wi+1}/{len(target_wells)}] {elapsed:.0f}s "
                      f"AMC={amc_metrics['kge']:.4f} IDW={idw_metrics['kge']:.4f} "
                      f"RMC={rmc_metrics['kge']:.4f}", flush=True)
        print()

    # ─── Results ──────────────────────────────────────────────────────────
    total_time = time.time() - t0
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "cv_mc_interpolation_detailed.csv"), index=False)

    lines = []
    lines.append("=" * 70)
    lines.append("MC SPATIAL INTERPOLATION — GRID-BASED ANOMALY LOO CV")
    lines.append("=" * 70)
    lines.append(f"Wells: {n_wells}, Times: {n_times}")
    lines.append(f"Target wells: {len(target_wells)}")
    lines.append(f"Total time: {total_time:.1f}s")
    lines.append("")

    for res in GRID_RESOLUTIONS:
        sub = df[df["resolution"] == res]
        if len(sub) == 0:
            continue

        lines.append(f"GRID: {res}° resolution")
        lines.append("-" * 70)
        methods = [("Anomaly MC+IDW", "amc"), ("IDW only", "idw"), ("Raw MC+IDW", "rmc")]
        lines.append(f"  {'Method':>18} | {'KGE_mean':>9} {'KGE_std':>8} {'KGE_med':>8} | "
                     f"{'R²_mean':>8} | {'RMSE_mean':>10} {'MAE_mean':>9}")
        lines.append("  " + "-" * 80)
        for name, pfx in methods:
            kge = sub[f"{pfx}_kge"]
            lines.append(
                f"  {name:>18} | {kge.mean():>9.4f} {kge.std():>8.4f} {kge.median():>8.4f} | "
                f"{sub[f'{pfx}_r2'].mean():>8.4f} | {sub[f'{pfx}_rmse'].mean():>10.3f} "
                f"{sub[f'{pfx}_mae'].mean():>9.3f}"
            )
        amc_wins = (sub["amc_kge"] > sub["idw_kge"]).sum()
        lines.append(f"  Anomaly MC > IDW: {amc_wins}/{len(sub)} wells")
        lines.append(f"  Mean error (IDW mean est): {sub['mean_error'].mean():.2f} ft")
        lines.append("")

    summary_text = "\n".join(lines)
    print(summary_text)

    with open(os.path.join(OUTPUT_DIR, "cv_mc_interpolation_summary.txt"), "w") as f:
        f.write(summary_text + "\n")

    meta = {
        "n_wells": n_wells, "n_times": n_times, "n_targets": len(target_wells),
        "grid_resolutions": GRID_RESOLUTIONS,
        "rank_options": RANK_OPTIONS,
        "elapsed_seconds": total_time,
        "results_by_resolution": {},
    }
    for res in GRID_RESOLUTIONS:
        sub = df[df["resolution"] == res]
        if len(sub):
            meta["results_by_resolution"][str(res)] = {
                "amc_kge_mean": float(sub["amc_kge"].mean()),
                "idw_kge_mean": float(sub["idw_kge"].mean()),
                "rmc_kge_mean": float(sub["rmc_kge"].mean()),
            }
    with open(os.path.join(OUTPUT_DIR, "cv_mc_interpolation_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nOutput: cv_mc_interpolation_detailed.csv, cv_mc_interpolation_summary.txt")


if __name__ == "__main__":
    main()
