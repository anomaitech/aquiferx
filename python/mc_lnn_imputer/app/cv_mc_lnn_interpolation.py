#!/usr/bin/env python3
"""
MC+LNN Spatial Interpolation CV (GSLB)

Tests spatial interpolation using the fully imputed MC+LNN dataset.
Each well's imputed series IS its MC+LNN model output — we transfer
these models spatially via distance-weighted donor regression.

Methods compared:

  A) MC+LNN Transfer (distance-weighted):
     - Select top-k nearest wells by distance
     - Each well's MC+LNN imputed series = its model output
     - For each donor: OLS regression learned from cross-donor relationships
       (target = a * donor + b) to account for level/scale differences
     - Distance-weighted blend of donor predictions
     - This transfers the MC+LNN temporal dynamics spatially

  B) Kriging (per timestep):
     - Ordinary kriging with Gaussian variogram
     - Matches the browser pipeline (kriging.ts)

  C) IDW baseline

CV: Leave-one-well-out on top-30 wells.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from typing import Dict, List, Tuple

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

MAX_DONORS = 15
N_TARGET_WELLS = 30
VARIOGRAM_MODEL = "gaussian"


# ─── Distance ────────────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ─── IDW ──────────────────────────────────────────────────────────────────────

def idw_series(t_lat, t_lon, src_lats, src_lons, matrix, exp=2.0, max_n=20):
    dlat = src_lats - t_lat
    dlon = (src_lons - t_lon) * np.cos(np.radians(t_lat))
    dists = np.sqrt(dlat**2 + dlon**2)
    order = np.argsort(dists)
    sel = order[:max_n]
    if dists[sel[0]] < 1e-8:
        return matrix[:, sel[0]].copy()
    w = 1.0 / (dists[sel] ** exp + 1e-12)
    w /= w.sum()
    return matrix[:, sel] @ w


# ─── MC+LNN Transfer (distance-weighted ARCHI from imputed series) ────────────

def mc_lnn_transfer(
    target_lat: float, target_lon: float,
    donor_lats: np.ndarray, donor_lons: np.ndarray,
    donor_matrix: np.ndarray,  # nTimes × nDonors (MC+LNN imputed series)
    max_donors: int = 15,
) -> np.ndarray:
    """Transfer MC+LNN models via distance-weighted donor OLS regression.

    For each nearby donor well:
      1. Its MC+LNN imputed series IS the model output
      2. Learn OLS from its neighbors: how this donor relates to other donors
      3. Use that relationship to predict what the target's series should be
      4. Weight by 1/distance²

    The OLS step is crucial — it accounts for level/scale differences between
    wells at different elevations. Without it, we'd just be doing IDW.
    """
    n_times, n_donors = donor_matrix.shape

    # Compute distances from target to all donors
    dists = np.array([haversine(target_lat, target_lon,
                                donor_lats[i], donor_lons[i])
                      for i in range(n_donors)])
    order = np.argsort(dists)
    sel = order[:max_donors]

    if dists[sel[0]] < 1.0:  # < 1 meter = same location
        return donor_matrix[:, sel[0]].copy()

    predictions = []
    weights = []

    for di in sel:
        # For this donor, find ITS nearest neighbors (excluding itself)
        # and learn OLS from their relationship
        di_dists = np.array([haversine(donor_lats[di], donor_lons[di],
                                       donor_lats[j], donor_lons[j])
                             for j in range(n_donors)])
        di_neighbors = np.argsort(di_dists)
        # Skip self (dist=0), take next k neighbors
        di_neighbors = [n for n in di_neighbors if n != di][:max_donors]

        if len(di_neighbors) < 2:
            # Not enough neighbors for regression — use raw series
            predictions.append(donor_matrix[:, di].copy())
            weights.append(1.0 / (dists[di]**2 + 1e-6))
            continue

        # IDW predictor of this donor from its neighbors
        nb_lats = donor_lats[di_neighbors]
        nb_lons = donor_lons[di_neighbors]
        nb_dists = di_dists[di_neighbors]
        nb_w = 1.0 / (nb_dists**2 + 1e-6)
        nb_w /= nb_w.sum()
        nb_pred_of_di = donor_matrix[:, di_neighbors] @ nb_w

        # OLS: donor_i = a * nb_pred + b
        di_series = donor_matrix[:, di]
        pm = np.mean(nb_pred_of_di)
        dm = np.mean(di_series)
        ss = np.sum((nb_pred_of_di - pm)**2)
        if ss < 1e-10:
            predictions.append(di_series.copy())
            weights.append(1.0 / (dists[di]**2 + 1e-6))
            continue

        a = np.sum((nb_pred_of_di - pm) * (di_series - dm)) / ss
        b = dm - a * pm

        # Now predict target: IDW of donors from TARGET location
        # then apply OLS transform learned for this donor
        target_nb_pred = idw_series(target_lat, target_lon,
                                     donor_lats, donor_lons, donor_matrix,
                                     exp=2.0, max_n=max_donors)
        target_pred_via_donor = a * target_nb_pred + b

        predictions.append(target_pred_via_donor)
        weights.append(1.0 / (dists[di]**2 + 1e-6))

    weights = np.array(weights)
    weights /= weights.sum()

    result = np.zeros(n_times)
    for i, pred in enumerate(predictions):
        result += weights[i] * pred

    return result


# ─── Kriging ──────────────────────────────────────────────────────────────────

def covariance_fn(dist, sill, range_, nugget, model="gaussian"):
    spatial_var = sill - nugget
    if dist <= 0:
        return spatial_var
    ratio = dist / range_
    if model == "exponential":
        return spatial_var * np.exp(-ratio)
    elif model == "spherical":
        return 0.0 if dist >= range_ else spatial_var * (1 - 1.5 * ratio + 0.5 * ratio**3)
    else:
        return spatial_var * np.exp(-(ratio**2))


def kriging_predict(t_lat, t_lon, w_lats, w_lons, w_vals, sill, range_, nugget, model):
    n = len(w_lats)
    if n <= 1:
        return w_vals[0] if n == 1 else float("nan")

    K = np.zeros((n + 1, n + 1))
    for i in range(n):
        K[i, i] = sill
        for j in range(i + 1, n):
            d = haversine(w_lats[i], w_lons[i], w_lats[j], w_lons[j])
            c = covariance_fn(d, sill, range_, nugget, model)
            K[i, j] = c
            K[j, i] = c
        K[i, n] = 1.0
        K[n, i] = 1.0

    rhs = np.zeros(n + 1)
    for i in range(n):
        d = haversine(t_lat, t_lon, w_lats[i], w_lons[i])
        rhs[i] = covariance_fn(d, sill, range_, nugget, model)
    rhs[n] = 1.0

    try:
        weights = np.linalg.solve(K, rhs)
        val = float(np.sum(weights[:n] * w_vals))
        return val if np.isfinite(val) else float("nan")
    except np.linalg.LinAlgError:
        return float("nan")


def estimate_variogram(lats, lons, values):
    variance = np.var(values)
    diagonal = haversine(lats.min(), lons.min(), lats.max(), lons.max())
    return {
        "sill": max(variance, 0.01),
        "range": max(diagonal / 3.0, 100),
        "nugget": max(variance * 0.05, 0.001),
    }


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(obs, pred):
    obs, pred = np.asarray(obs, dtype=float), np.asarray(pred, dtype=float)
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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 70)
    print("MC+LNN Spatial Interpolation — LOO CV (GSLB)")
    print("=" * 70)
    print("A) MC+LNN Transfer (distance-weighted multi-well OLS)")
    print("B) Kriging (ordinary, Gaussian variogram)")
    print("C) IDW baseline")
    print()

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

    # Target wells
    raw_counts = raw_df.groupby("Well_ID")["Date"].nunique().reset_index(name="n_obs")
    raw_counts = raw_counts[raw_counts["Well_ID"].isin(well_ids)]
    raw_counts = raw_counts.sort_values("n_obs", ascending=False)
    target_wells = raw_counts.head(N_TARGET_WELLS)["Well_ID"].tolist()

    print(f"Wells: {n_wells}, Times: {n_times}")
    print(f"Target wells: {len(target_wells)}")
    print()

    rows = []

    for wi, target_wid in enumerate(target_wells):
        w_col = well_idx_map[target_wid]
        truth = well_matrix[:, w_col].copy()
        t_lat, t_lon = well_lats[w_col], well_lons[w_col]

        # Other wells
        other = np.ones(n_wells, dtype=bool)
        other[w_col] = False
        ow_lats = well_lats[other]
        ow_lons = well_lons[other]
        ow_matrix = well_matrix[:, other]

        # ── A) MC+LNN Transfer ────────────────────────────────────────
        transfer_pred = mc_lnn_transfer(
            t_lat, t_lon, ow_lats, ow_lons, ow_matrix, max_donors=MAX_DONORS
        )
        transfer_metrics = compute_metrics(truth, transfer_pred)

        # ── B) IDW baseline ───────────────────────────────────────────
        idw_pred = idw_series(t_lat, t_lon, ow_lats, ow_lons, ow_matrix)
        idw_metrics = compute_metrics(truth, idw_pred)

        rows.append({
            "Well_ID": target_wid,
            "tf_kge": transfer_metrics["kge"], "tf_r2": transfer_metrics["r2"],
            "tf_rmse": transfer_metrics["rmse"], "tf_mae": transfer_metrics["mae"],
            "idw_kge": idw_metrics["kge"], "idw_r2": idw_metrics["r2"],
            "idw_rmse": idw_metrics["rmse"], "idw_mae": idw_metrics["mae"],
        })

        elapsed = time.time() - t0
        print(f"  [{wi+1}/{len(target_wells)}] {target_wid}: "
              f"Transfer={transfer_metrics['kge']:.4f}  "
              f"IDW={idw_metrics['kge']:.4f}  ({elapsed:.0f}s)", flush=True)

    # ─── Results ──────────────────────────────────────────────────────────
    total_time = time.time() - t0
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "cv_mc_lnn_interpolation_detailed.csv"), index=False)

    lines = []
    lines.append("=" * 70)
    lines.append("MC+LNN SPATIAL INTERPOLATION — LOO CV RESULTS")
    lines.append("=" * 70)
    lines.append(f"Wells: {n_wells}, Times: {n_times}")
    lines.append(f"Target wells: {len(target_wells)}")
    lines.append(f"Total time: {total_time:.1f}s")
    lines.append("")

    methods = [("MC+LNN Transfer", "tf"), ("IDW", "idw")]
    lines.append(f"{'Method':>18} | {'KGE_mean':>9} {'KGE_std':>8} {'KGE_med':>8} | "
                 f"{'R²_mean':>8} | {'RMSE_mean':>10} {'MAE_mean':>9}")
    lines.append("-" * 85)
    for name, pfx in methods:
        kge = df[f"{pfx}_kge"]
        lines.append(
            f"{name:>18} | {kge.mean():>9.4f} {kge.std():>8.4f} {kge.median():>8.4f} | "
            f"{df[f'{pfx}_r2'].mean():>8.4f} | {df[f'{pfx}_rmse'].mean():>10.3f} "
            f"{df[f'{pfx}_mae'].mean():>9.3f}"
        )

    # Win counts
    lines.append("")
    tf_wins = (df["tf_kge"] > df["idw_kge"]).sum()
    lines.append(f"MC+LNN Transfer > IDW: {tf_wins}/{len(df)} wells")

    lines.append("")
    lines.append("PER-WELL DETAIL (sorted by MC+LNN Transfer KGE):")
    lines.append(f"{'Well_ID':>20} | {'TF_KGE':>8} {'TF_R²':>8} {'TF_RMSE':>8} | "
                 f"{'IDW_KGE':>8} {'IDW_RMSE':>8} | {'Winner':>10}")
    lines.append("-" * 90)
    for _, row in df.sort_values("tf_kge", ascending=False).iterrows():
        winner = "Transfer" if row["tf_kge"] > row["idw_kge"] else "IDW"
        lines.append(
            f"{row['Well_ID']:>20} | {row['tf_kge']:>8.4f} {row['tf_r2']:>8.4f} "
            f"{row['tf_rmse']:>8.2f} | {row['idw_kge']:>8.4f} {row['idw_rmse']:>8.2f} | "
            f"{winner:>10}"
        )

    summary_text = "\n".join(lines)
    print()
    print(summary_text)

    with open(os.path.join(OUTPUT_DIR, "cv_mc_lnn_interpolation_summary.txt"), "w") as f:
        f.write(summary_text + "\n")

    meta = {
        "n_wells": n_wells, "n_times": n_times, "n_targets": len(target_wells),
        "max_donors": MAX_DONORS,
        "elapsed_seconds": total_time,
        "tf_kge_mean": float(df["tf_kge"].mean()),
        "idw_kge_mean": float(df["idw_kge"].mean()),
    }
    with open(os.path.join(OUTPUT_DIR, "cv_mc_lnn_interpolation_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nOutput: cv_mc_lnn_interpolation_detailed.csv, cv_mc_lnn_interpolation_summary.txt")


if __name__ == "__main__":
    main()
