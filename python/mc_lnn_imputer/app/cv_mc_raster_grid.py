#!/usr/bin/env python3
"""
MC Raster Grid Interpolation CV (GSLB)

Tests pure MC (SoftImpute) for spatial gridding using the space-time matrix
approach: matrix = nTimes × nGridCells, where well-mapped cells have values
and empty cells are filled by MC.

IDW initialization: instead of filling NaN columns with column means (which
converges to global average), empty grid cells are initialized using IDW
from nearby well-mapped cells. SoftImpute then refines.

CV: Leave-one-well-out — remove a well's contribution to its grid cell,
run MC, compare the prediction at that cell to truth.

Grid resolutions tested: 0.1°, 0.2°, 0.5°
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
OUTPUT_DIR = APP_DIR

IMPUTED_CSV = os.path.join(PROJECT_ROOT, "output",
    "full_imputed_series_alleligible_iterative_softimpute_standalone.csv")
RAW_CSV = os.path.join(PROJECT_ROOT, "datas",
    "measurements_till_2023_to_lnn_imputation.csv")

# Grid resolutions to test
GRID_RESOLUTIONS = [0.1, 0.2, 0.5]  # degrees

# SoftImpute parameters
MC_MAX_ITERATIONS = 100
MC_TOLERANCE = 1e-5
MC_SHRINKAGE = 0
RANK_OPTIONS = [3, 5, 8, 10, 15, 20]

# IDW initialization parameters
IDW_EXPONENT = 2.0
IDW_MAX_NEIGHBORS = 15

# CV: LOO on top-N most observed wells
N_TARGET_WELLS = 30


# ─── IDW initialization ──────────────────────────────────────────────────────

def idw_initialize_column(
    target_lat: float, target_lon: float,
    well_lats: np.ndarray, well_lons: np.ndarray,
    well_series: np.ndarray,  # nTimes × nWellsWithData
    exponent: float = 2.0,
    max_neighbors: int = 15,
) -> np.ndarray:
    """Initialize a grid cell's time series using IDW from nearby wells."""
    n_times = well_series.shape[0]

    # Compute distances (degrees, approximate)
    dlat = well_lats - target_lat
    dlon = (well_lons - target_lon) * np.cos(np.radians(target_lat))
    dists = np.sqrt(dlat**2 + dlon**2)

    # Select nearest neighbors
    order = np.argsort(dists)
    sel = order[:max_neighbors]
    sel_dists = dists[sel]

    # Handle case where a well is exactly at the grid cell
    if sel_dists[0] < 1e-8:
        return well_series[:, sel[0]].copy()

    weights = 1.0 / (sel_dists ** exponent + 1e-12)
    weights /= weights.sum()

    # Weighted average across time
    result = np.zeros(n_times)
    for i, s in enumerate(sel):
        result += weights[i] * well_series[:, s]

    return result


# ─── SoftImpute ───────────────────────────────────────────────────────────────

def soft_impute(matrix: np.ndarray, rank: int, max_iterations: int = 100,
                tolerance: float = 1e-5) -> np.ndarray:
    """SoftImpute with observed-value preservation."""
    M = matrix.copy()
    mask = ~np.isnan(M)

    # Fill remaining NaN with column means (shouldn't be many after IDW init)
    for c in range(M.shape[1]):
        col_nans = np.isnan(M[:, c])
        if col_nans.any():
            col_vals = M[~col_nans, c]
            M[col_nans, c] = np.mean(col_vals) if len(col_vals) else 0.0

    for iteration in range(max_iterations):
        M_prev = M.copy()

        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        use_rank = max(1, min(rank, len(S)))
        S_trunc = S.copy()
        S_trunc[use_rank:] = 0

        M_new = U @ np.diag(S_trunc) @ Vt
        M_new[mask] = matrix[mask]
        M = M_new

        diff = np.linalg.norm(M - M_prev)
        denom = max(np.linalg.norm(M_prev), 1e-12)
        if diff / denom <= tolerance:
            break

    return M


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(obs: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    if len(obs) < 2:
        return {"n": len(obs), "kge": float("nan"), "r2": float("nan"),
                "rmse": float("nan"), "mae": float("nan"), "nse": float("nan")}

    err = pred - obs
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))

    mean_o, mean_p = float(np.mean(obs)), float(np.mean(pred))
    std_o = float(np.std(obs, ddof=1))
    std_p = float(np.std(pred, ddof=1))

    r = float(np.corrcoef(obs, pred)[0, 1]) if std_o > 1e-10 and std_p > 1e-10 else 0.0

    ss_tot = float(np.sum((obs - mean_o)**2))
    ss_res = float(np.sum(err**2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12) if ss_tot > 1e-12 else float("nan")

    alpha = std_p / std_o if std_o > 1e-10 else 1.0
    beta = mean_p / mean_o if abs(mean_o) > 1e-10 else 1.0
    kge = 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    return {"n": len(obs), "kge": float(kge), "r2": float(r2),
            "rmse": rmse, "mae": mae, "nse": float(r2), "r": float(r)}


# ─── Grid building ────────────────────────────────────────────────────────────

def build_grid(lat_min, lat_max, lon_min, lon_max, resolution):
    """Build grid cell centers."""
    ny = max(1, int(np.ceil((lat_max - lat_min) / resolution)))
    nx = max(1, int(np.ceil((lon_max - lon_min) / resolution)))
    lats = []
    lons = []
    for row in range(ny):
        for col in range(nx):
            lats.append(lat_min + (row + 0.5) * resolution)
            lons.append(lon_min + (col + 0.5) * resolution)
    return np.array(lats), np.array(lons), ny, nx


def map_wells_to_grid(well_lats, well_lons, grid_lats, grid_lons):
    """Map each well to its nearest grid cell. Returns dict: grid_idx -> [well_indices]."""
    mapping = {}
    well_to_grid = {}
    for wi in range(len(well_lats)):
        dlat = grid_lats - well_lats[wi]
        dlon = (grid_lons - well_lons[wi]) * np.cos(np.radians(well_lats[wi]))
        dists = dlat**2 + dlon**2
        nearest = int(np.argmin(dists))
        if nearest not in mapping:
            mapping[nearest] = []
        mapping[nearest].append(wi)
        well_to_grid[wi] = nearest
    return mapping, well_to_grid


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 70)
    print("MC Raster Grid — Space-Time Matrix CV (GSLB)")
    print("=" * 70)

    # Load data
    print("Loading data...")
    imputed_df = pd.read_csv(IMPUTED_CSV)
    imputed_df["Well_ID"] = imputed_df["Well_ID"].astype(str)
    imputed_df["Date"] = pd.to_datetime(imputed_df["Date"])
    imputed_df = imputed_df.sort_values(["Well_ID", "Date"])

    raw_df = pd.read_csv(RAW_CSV)
    raw_df["Date"] = pd.to_datetime(raw_df["Date"])
    raw_df = raw_df[(raw_df["Date"].dt.year >= 2000) & (raw_df["Date"].dt.year <= 2023)]
    raw_df["Well_ID"] = raw_df["Well_ID"].astype(str)

    # Build well data
    well_ids = sorted(imputed_df["Well_ID"].unique())
    dates = sorted(imputed_df["Date"].unique())
    n_wells = len(well_ids)
    n_times = len(dates)

    well_idx_map = {wid: i for i, wid in enumerate(well_ids)}
    date_idx_map = {d: i for i, d in enumerate(dates)}

    # Well matrix: nTimes × nWells
    well_matrix = np.full((n_times, n_wells), np.nan)
    for _, row in imputed_df.iterrows():
        t = date_idx_map[row["Date"]]
        w = well_idx_map[row["Well_ID"]]
        well_matrix[t, w] = row["final_wte"]

    # Well coordinates
    coords = raw_df.groupby("Well_ID")[["lat_dec", "long_dec"]].first()
    well_lats = np.array([coords.loc[wid, "lat_dec"] for wid in well_ids])
    well_lons = np.array([coords.loc[wid, "long_dec"] for wid in well_ids])

    lat_min, lat_max = well_lats.min() - 0.1, well_lats.max() + 0.1
    lon_min, lon_max = well_lons.min() - 0.1, well_lons.max() + 0.1

    # Target wells for LOO
    raw_counts = raw_df.groupby("Well_ID")["Date"].nunique().reset_index(name="n_obs")
    raw_counts = raw_counts[raw_counts["Well_ID"].isin(well_ids)]
    raw_counts = raw_counts.sort_values("n_obs", ascending=False)
    target_wells = raw_counts.head(N_TARGET_WELLS)["Well_ID"].tolist()

    print(f"Wells: {n_wells}, Times: {n_times}")
    print(f"Target wells (LOO): {len(target_wells)}")
    print(f"Grid resolutions: {GRID_RESOLUTIONS}°")
    print()

    all_rows = []

    for res in GRID_RESOLUTIONS:
        grid_lats, grid_lons, ny, nx = build_grid(lat_min, lat_max, lon_min, lon_max, res)
        n_cells = len(grid_lats)
        mapping, well_to_grid = map_wells_to_grid(well_lats, well_lons, grid_lats, grid_lons)
        n_occupied = len(mapping)

        print(f"Grid {res}°: {ny}×{nx} = {n_cells} cells, {n_occupied} occupied by wells")

        # Build grid matrix: nTimes × nCells
        # Occupied cells get the average of mapped wells
        grid_matrix = np.full((n_times, n_cells), np.nan)
        for cell_idx, well_indices in mapping.items():
            # Average all wells mapping to this cell
            grid_matrix[:, cell_idx] = np.mean(well_matrix[:, well_indices], axis=1)

        # IDW-initialize unoccupied cells
        occupied_lats = np.array([grid_lats[c] for c in mapping])
        occupied_lons = np.array([grid_lons[c] for c in mapping])
        occupied_series = np.column_stack([grid_matrix[:, c] for c in mapping])  # nTimes × nOccupied

        unoccupied = [c for c in range(n_cells) if c not in mapping]
        for c in unoccupied:
            grid_matrix[:, c] = idw_initialize_column(
                grid_lats[c], grid_lons[c],
                occupied_lats, occupied_lons,
                occupied_series,
                exponent=IDW_EXPONENT,
                max_neighbors=IDW_MAX_NEIGHBORS,
            )

        print(f"  IDW initialized {len(unoccupied)} empty cells")

        # Adaptive rank selection on full grid (no holdout)
        best_rank = 8
        best_err = float("inf")
        for k in RANK_OPTIONS:
            if k >= min(grid_matrix.shape):
                continue
            # Quick test: hold out 10% of occupied cells' values
            rng = np.random.default_rng(42)
            test_mask = np.zeros_like(grid_matrix, dtype=bool)
            occ_list = list(mapping.keys())
            for c in rng.choice(occ_list, size=max(1, len(occ_list) // 5), replace=False):
                hold_t = rng.choice(n_times, size=n_times // 10, replace=False)
                test_mask[hold_t, c] = True

            M_test = grid_matrix.copy()
            M_test[test_mask] = np.nan
            # Re-mark observed
            obs_mask = ~np.isnan(grid_matrix)
            obs_mask[test_mask] = False

            completed = soft_impute(M_test, rank=k, max_iterations=50, tolerance=MC_TOLERANCE)
            err = float(np.mean((completed[test_mask] - grid_matrix[test_mask])**2))
            if err < best_err:
                best_err = err
                best_rank = k

        print(f"  Best rank: {best_rank} (MSE={best_err:.4f})")

        # LOO CV per target well
        t_res = time.time()
        for wi_idx, target_wid in enumerate(target_wells):
            w_idx = well_idx_map[target_wid]
            target_cell = well_to_grid[w_idx]
            truth = well_matrix[:, w_idx].copy()

            # Wells sharing this cell
            cell_wells = mapping[target_cell]
            other_wells_in_cell = [w for w in cell_wells if w != w_idx]

            # Build modified grid matrix
            M_loo = grid_matrix.copy()

            if other_wells_in_cell:
                # Other wells remain in the cell — recalculate cell value without target
                M_loo[:, target_cell] = np.mean(well_matrix[:, other_wells_in_cell], axis=1)
            else:
                # Only well in this cell — mark as NaN, IDW initialize from neighbors
                M_loo[:, target_cell] = np.nan
                other_occ = {c: ws for c, ws in mapping.items() if c != target_cell}
                if other_occ:
                    occ_lats = np.array([grid_lats[c] for c in other_occ])
                    occ_lons = np.array([grid_lons[c] for c in other_occ])
                    occ_series = np.column_stack([grid_matrix[:, c] for c in other_occ])
                    M_loo[:, target_cell] = idw_initialize_column(
                        grid_lats[target_cell], grid_lons[target_cell],
                        occ_lats, occ_lons, occ_series,
                        exponent=IDW_EXPONENT, max_neighbors=IDW_MAX_NEIGHBORS,
                    )

            # Mark observed cells (everything except the target cell is "observed")
            # But we treat the IDW-initialized target cell as NOT observed
            # so SoftImpute can refine it
            obs_mask_loo = ~np.isnan(grid_matrix)
            obs_mask_loo[:, target_cell] = False  # target cell is "missing"

            # Run SoftImpute — preserve all observed, let target cell float
            M_run = M_loo.copy()
            for iteration in range(MC_MAX_ITERATIONS):
                M_prev = M_run.copy()
                U, S, Vt = np.linalg.svd(M_run, full_matrices=False)
                S_trunc = S.copy()
                S_trunc[best_rank:] = 0
                M_new = U @ np.diag(S_trunc) @ Vt
                M_new[obs_mask_loo] = grid_matrix[obs_mask_loo]
                # For unoccupied cells (not target), keep IDW init as "soft anchor"
                M_run = M_new
                diff = np.linalg.norm(M_run - M_prev)
                denom = max(np.linalg.norm(M_prev), 1e-12)
                if diff / denom <= MC_TOLERANCE:
                    break

            pred = M_run[:, target_cell]
            metrics = compute_metrics(truth, pred)

            # Also compute IDW-only baseline (no MC refinement)
            idw_pred = M_loo[:, target_cell]
            idw_metrics = compute_metrics(truth, idw_pred)

            all_rows.append({
                "resolution": res,
                "Well_ID": target_wid,
                "grid_cell": target_cell,
                "n_wells_in_cell": len(cell_wells),
                "rank": best_rank,
                "mc_kge": metrics["kge"], "mc_r2": metrics["r2"],
                "mc_rmse": metrics["rmse"], "mc_mae": metrics["mae"],
                "mc_nse": metrics["nse"], "mc_r": metrics["r"],
                "idw_kge": idw_metrics["kge"], "idw_r2": idw_metrics["r2"],
                "idw_rmse": idw_metrics["rmse"], "idw_mae": idw_metrics["mae"],
            })

            if (wi_idx + 1) % 10 == 0 or wi_idx + 1 == len(target_wells):
                elapsed = time.time() - t_res
                print(f"  [{wi_idx+1}/{len(target_wells)}] {elapsed:.0f}s — "
                      f"last: MC KGE={metrics['kge']:.4f}, IDW KGE={idw_metrics['kge']:.4f}",
                      flush=True)

        print()

    # ─── Results ──────────────────────────────────────────────────────────
    total_time = time.time() - t0
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "cv_mc_raster_grid_detailed.csv"), index=False)

    lines = []
    lines.append("=" * 70)
    lines.append("MC RASTER GRID — SPACE-TIME MATRIX LOO CV RESULTS")
    lines.append("=" * 70)
    lines.append(f"Wells: {n_wells}, Times: {n_times}")
    lines.append(f"Target wells: {len(target_wells)} (LOO)")
    lines.append(f"Total time: {total_time:.1f}s")
    lines.append("")

    for res in GRID_RESOLUTIONS:
        sub = df[df["resolution"] == res]
        if len(sub) == 0:
            continue
        lines.append(f"GRID RESOLUTION: {res}°")
        lines.append("-" * 70)

        # MC results
        lines.append(f"  MC+IDW init:")
        lines.append(f"    KGE:  {sub['mc_kge'].mean():.4f} ± {sub['mc_kge'].std():.4f}  "
                     f"(median {sub['mc_kge'].median():.4f})")
        lines.append(f"    R²:   {sub['mc_r2'].mean():.4f} ± {sub['mc_r2'].std():.4f}")
        lines.append(f"    RMSE: {sub['mc_rmse'].mean():.3f} ± {sub['mc_rmse'].std():.3f}")
        lines.append(f"    MAE:  {sub['mc_mae'].mean():.3f} ± {sub['mc_mae'].std():.3f}")

        # IDW baseline
        lines.append(f"  IDW only (baseline):")
        lines.append(f"    KGE:  {sub['idw_kge'].mean():.4f} ± {sub['idw_kge'].std():.4f}  "
                     f"(median {sub['idw_kge'].median():.4f})")
        lines.append(f"    R²:   {sub['idw_r2'].mean():.4f} ± {sub['idw_r2'].std():.4f}")
        lines.append(f"    RMSE: {sub['idw_rmse'].mean():.3f} ± {sub['idw_rmse'].std():.3f}")
        lines.append(f"    MAE:  {sub['idw_mae'].mean():.3f} ± {sub['idw_mae'].std():.3f}")

        # Improvement
        mc_better = (sub['mc_kge'] > sub['idw_kge']).sum()
        lines.append(f"  MC > IDW: {mc_better}/{len(sub)} wells")
        lines.append("")

    # Per-well detail for best resolution
    lines.append("PER-WELL DETAIL (best resolution):")
    best_res = df.groupby("resolution")["mc_kge"].mean().idxmax()
    sub = df[df["resolution"] == best_res].sort_values("mc_kge", ascending=False)
    lines.append(f"{'Well_ID':>20} {'MC_KGE':>8} {'MC_R²':>8} {'MC_RMSE':>8} | "
                 f"{'IDW_KGE':>8} {'IDW_R²':>8} {'IDW_RMSE':>8}")
    lines.append("-" * 85)
    for _, row in sub.iterrows():
        lines.append(
            f"{row['Well_ID']:>20} {row['mc_kge']:>8.4f} {row['mc_r2']:>8.4f} "
            f"{row['mc_rmse']:>8.2f} | {row['idw_kge']:>8.4f} {row['idw_r2']:>8.4f} "
            f"{row['idw_rmse']:>8.2f}"
        )

    summary_text = "\n".join(lines)
    print()
    print(summary_text)

    with open(os.path.join(OUTPUT_DIR, "cv_mc_raster_grid_summary.txt"), "w") as f:
        f.write(summary_text + "\n")

    meta = {
        "n_wells": n_wells,
        "n_times": n_times,
        "n_targets": len(target_wells),
        "grid_resolutions": GRID_RESOLUTIONS,
        "rank_options": RANK_OPTIONS,
        "idw_exponent": IDW_EXPONENT,
        "idw_max_neighbors": IDW_MAX_NEIGHBORS,
        "elapsed_seconds": total_time,
    }
    with open(os.path.join(OUTPUT_DIR, "cv_mc_raster_grid_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print()
    print("Output files:")
    print(f"  cv_mc_raster_grid_detailed.csv")
    print(f"  cv_mc_raster_grid_summary.txt")
    print(f"  cv_mc_raster_grid_metadata.json")


if __name__ == "__main__":
    main()
