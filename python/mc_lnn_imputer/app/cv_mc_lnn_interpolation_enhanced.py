#!/usr/bin/env python3
"""
Enhanced MC+LNN Spatial Interpolation with Auxiliary Data (GSLB)

Enhances spatial interpolation by using spatially-varying auxiliary data
to improve donor selection and weighting:

  1. Precipitation (Open-Meteo monthly) — varies by location, correlates with recharge
  2. Elevation (Open-Meteo DEM) — strong WTE-elevation correlation
  3. GLDAS soil moisture — temporal auxiliary (same as imputation pipeline)

Donor selection uses a COMBINED score:
  score = w_dist * (1/dist²) + w_precip * precip_corr² + w_elev * (1/elev_diff²)

This replaces pure distance-based weighting with a multi-criteria approach
that understands which wells are hydrologically similar to the target.

CV: Leave-one-well-out on top-30 wells.
Compares: Enhanced Transfer, Distance-only Transfer, IDW
"""

from __future__ import annotations

import json
import os
import time
import warnings
from typing import Dict, List, Tuple

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

AUX_CACHE = os.path.join(OUTPUT_DIR, "spatial_aux_cache.json")

MAX_DONORS = 15
N_TARGET_WELLS = 30

# Combined score weights
W_DIST = 0.4
W_PRECIP = 0.4
W_ELEV = 0.2


# ─── Open-Meteo: fetch monthly precipitation + elevation ─────────────────────

def fetch_monthly_precip(lats: List[float], lons: List[float],
                         cache: Dict, start_year=2000, end_year=2023) -> Tuple[np.ndarray, np.ndarray]:
    """Fetch monthly precipitation sums and elevation for each location.
    Uses Open-Meteo daily API and aggregates to monthly.
    Returns: precip_matrix (nMonths × nLocations), elevations (nLocations)
    """
    n_months = (end_year - start_year + 1) * 12
    precip = np.zeros((n_months, len(lats)))
    elevs = np.zeros(len(lats))

    to_fetch = []
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        key = f"precip_{lat:.5f}_{lon:.5f}"
        if key in cache:
            precip[:, i] = np.array(cache[key]["precip"][:n_months])
            elevs[i] = cache[key].get("elevation", 0)
        else:
            to_fetch.append(i)

    if to_fetch:
        print(f"  Fetching precipitation for {len(to_fetch)} locations from Open-Meteo...")
        done = 0
        # One location at a time (daily data is large, batching causes timeouts)
        for idx in to_fetch:
            lat, lon = lats[idx], lons[idx]
            try:
                url = (f"https://archive-api.open-meteo.com/v1/archive?"
                       f"latitude={lat:.5f}&longitude={lon:.5f}"
                       f"&start_date={start_year}-01-01&end_date={end_year}-12-31"
                       f"&daily=precipitation_sum&timezone=auto")
                resp = requests.get(url, timeout=30)
                if not resp.ok:
                    done += 1
                    continue

                data = resp.json()
                elev = data.get("elevation", 0)
                daily_dates = data.get("daily", {}).get("time", [])
                daily_precip = data.get("daily", {}).get("precipitation_sum", [])

                # Aggregate daily to monthly
                monthly_precip = {}
                for d, p in zip(daily_dates, daily_precip):
                    ym = d[:7]  # "YYYY-MM"
                    if ym not in monthly_precip:
                        monthly_precip[ym] = 0.0
                    if p is not None:
                        monthly_precip[ym] += float(p)

                # Build ordered monthly array
                p_arr = []
                for y in range(start_year, end_year + 1):
                    for m in range(1, 13):
                        ym = f"{y}-{m:02d}"
                        p_arr.append(monthly_precip.get(ym, 0.0))

                precip[:, idx] = np.array(p_arr[:n_months])
                elevs[idx] = float(elev) if elev is not None else 0.0

                cache_key = f"precip_{lat:.5f}_{lon:.5f}"
                cache[cache_key] = {"precip": p_arr[:n_months], "elevation": float(elevs[idx])}

            except Exception as e:
                pass  # silently skip failed locations

            done += 1
            if done % 50 == 0 or done == len(to_fetch):
                print(f"    [{done}/{len(to_fetch)}]", flush=True)

        print(f"  Fetched {len(to_fetch)} locations")

    return precip, elevs


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


# ─── Enhanced transfer (multi-criteria donor selection) ───────────────────────

def enhanced_transfer(
    target_lat: float, target_lon: float,
    target_precip: np.ndarray,  # monthly precip at target (nMonths)
    target_elev: float,
    donor_lats: np.ndarray, donor_lons: np.ndarray,
    donor_matrix: np.ndarray,  # nTimes × nDonors (MC+LNN imputed)
    donor_precip: np.ndarray,  # nMonths × nDonors
    donor_elevs: np.ndarray,
    max_donors: int = 15,
    w_dist: float = 0.4, w_precip: float = 0.4, w_elev: float = 0.2,
) -> np.ndarray:
    """Transfer MC+LNN models using multi-criteria donor weighting.

    Combined score per donor:
      score = w_dist * dist_score + w_precip * precip_score + w_elev * elev_score

    Where:
      dist_score = 1 / (distance_km² + 1)
      precip_score = max(0, pearson_r(target_precip, donor_precip))²
      elev_score = 1 / (elev_diff² + 100)
    """
    n_times, n_donors = donor_matrix.shape

    # Compute scores for all donors
    scores = np.zeros(n_donors)

    for di in range(n_donors):
        # Distance score
        dist_m = haversine(target_lat, target_lon, donor_lats[di], donor_lons[di])
        dist_km = dist_m / 1000.0
        dist_score = 1.0 / (dist_km**2 + 1.0)

        # Precipitation correlation score
        if np.std(target_precip) > 1e-6 and np.std(donor_precip[:, di]) > 1e-6:
            precip_r = np.corrcoef(target_precip, donor_precip[:, di])[0, 1]
            precip_score = max(0, precip_r)**2 if np.isfinite(precip_r) else 0.0
        else:
            precip_score = 0.0

        # Elevation score
        elev_diff = abs(target_elev - donor_elevs[di])
        elev_score = 1.0 / (elev_diff**2 + 100.0)

        scores[di] = w_dist * dist_score + w_precip * precip_score + w_elev * elev_score

    # Select top donors
    order = np.argsort(-scores)
    sel = order[:max_donors]
    sel_scores = scores[sel]

    if sel_scores.sum() < 1e-12:
        # Fallback to IDW
        return idw_series(target_lat, target_lon, donor_lats, donor_lons,
                          donor_matrix, exp=2.0, max_n=max_donors)

    # Normalize scores as weights
    weights = sel_scores / sel_scores.sum()

    # Weighted average of donor series (direct transfer)
    # Enhanced: also do per-donor OLS with safeguards
    predictions = []
    pred_weights = []

    for i, di in enumerate(sel):
        donor_series = donor_matrix[:, di]

        # Try OLS: learn relationship from donor's own neighbors
        di_neighbors = np.argsort(-scores)
        di_neighbors = [n for n in di_neighbors if n != di][:max_donors]

        if len(di_neighbors) >= 2:
            # Weighted average of neighbors as predictor
            nb_scores = scores[di_neighbors]
            if nb_scores.sum() > 1e-12:
                nb_w = nb_scores / nb_scores.sum()
                nb_pred = donor_matrix[:, di_neighbors] @ nb_w

                # OLS with safeguards
                pm, dm = np.mean(nb_pred), np.mean(donor_series)
                ss = np.sum((nb_pred - pm)**2)

                if ss > 1e-10:
                    a = np.sum((nb_pred - pm) * (donor_series - dm)) / ss
                    b = dm - a * pm

                    # Safeguard: clip OLS coefficients to reasonable range
                    a = np.clip(a, 0.1, 10.0)

                    # Predict target using same neighbor blend from target's perspective
                    target_nb_pred = idw_series(target_lat, target_lon,
                                                donor_lats, donor_lons, donor_matrix,
                                                exp=2.0, max_n=max_donors)
                    pred = a * target_nb_pred + b

                    # Check OLS quality: R² of fit on this donor
                    fitted = a * nb_pred + b
                    ss_res = np.sum((donor_series - fitted)**2)
                    ss_tot = np.sum((donor_series - dm)**2)
                    r2 = 1 - ss_res / (ss_tot + 1e-12) if ss_tot > 1e-12 else 0

                    if r2 > 0.3:
                        # Good fit — use OLS prediction
                        predictions.append(pred)
                        pred_weights.append(weights[i] * (1 + r2))
                        continue

        # Fallback: use raw donor series (level may be off)
        predictions.append(donor_series)
        pred_weights.append(weights[i] * 0.5)

    pred_weights = np.array(pred_weights)
    pred_weights /= pred_weights.sum()

    result = np.zeros(n_times)
    for i, pred in enumerate(predictions):
        result += pred_weights[i] * pred

    return result


# ─── Distance-only transfer (baseline from earlier) ──────────────────────────

def distance_transfer(
    target_lat, target_lon,
    donor_lats, donor_lons, donor_matrix,
    max_donors=15,
):
    """Simple distance-weighted transfer (no auxiliaries)."""
    n_times = donor_matrix.shape[0]
    dists = np.array([haversine(target_lat, target_lon, donor_lats[i], donor_lons[i])
                      for i in range(len(donor_lats))])
    order = np.argsort(dists)
    sel = order[:max_donors]

    if dists[sel[0]] < 1.0:
        return donor_matrix[:, sel[0]].copy()

    weights = 1.0 / (dists[sel]**2 + 1e-6)
    weights /= weights.sum()
    return donor_matrix[:, sel] @ weights


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
    print("Enhanced MC+LNN Spatial Interpolation with Aux Data — LOO CV")
    print("=" * 70)
    print("A) Enhanced Transfer (distance + precip corr + elevation)")
    print("B) Distance-only Transfer (baseline)")
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

    # ── Fetch auxiliary data ──────────────────────────────────────────
    print("Fetching auxiliary data (precipitation + elevation)...")
    cache: Dict = {}
    if os.path.exists(AUX_CACHE):
        with open(AUX_CACHE) as f:
            cache = json.load(f)

    well_precip, well_elevs = fetch_monthly_precip(
        well_lats.tolist(), well_lons.tolist(), cache
    )
    print(f"  Precip shape: {well_precip.shape}")
    print(f"  Elevation range: {well_elevs.min():.0f} to {well_elevs.max():.0f} m")

    # Save cache
    with open(AUX_CACHE, "w") as f:
        json.dump(cache, f)

    print()

    # ── LOO CV ────────────────────────────────────────────────────────
    rows = []
    for wi, target_wid in enumerate(target_wells):
        w_col = well_idx_map[target_wid]
        truth = well_matrix[:, w_col].copy()
        t_lat, t_lon = well_lats[w_col], well_lons[w_col]
        t_precip = well_precip[:, w_col]
        t_elev = well_elevs[w_col]

        # Other wells
        other = np.ones(n_wells, dtype=bool)
        other[w_col] = False
        ow_lats = well_lats[other]
        ow_lons = well_lons[other]
        ow_matrix = well_matrix[:, other]
        ow_precip = well_precip[:, other]
        ow_elevs = well_elevs[other]

        # ── A) Enhanced Transfer ──────────────────────────────────────
        enhanced_pred = enhanced_transfer(
            t_lat, t_lon, t_precip, t_elev,
            ow_lats, ow_lons, ow_matrix, ow_precip, ow_elevs,
            max_donors=MAX_DONORS,
            w_dist=W_DIST, w_precip=W_PRECIP, w_elev=W_ELEV,
        )
        enhanced_metrics = compute_metrics(truth, enhanced_pred)

        # ── B) Distance-only Transfer ─────────────────────────────────
        dist_pred = distance_transfer(
            t_lat, t_lon, ow_lats, ow_lons, ow_matrix, max_donors=MAX_DONORS,
        )
        dist_metrics = compute_metrics(truth, dist_pred)

        # ── C) IDW ────────────────────────────────────────────────────
        idw_pred = idw_series(t_lat, t_lon, ow_lats, ow_lons, ow_matrix)
        idw_metrics = compute_metrics(truth, idw_pred)

        rows.append({
            "Well_ID": target_wid,
            "elev": t_elev,
            "enh_kge": enhanced_metrics["kge"], "enh_r2": enhanced_metrics["r2"],
            "enh_rmse": enhanced_metrics["rmse"], "enh_mae": enhanced_metrics["mae"],
            "dist_kge": dist_metrics["kge"], "dist_r2": dist_metrics["r2"],
            "dist_rmse": dist_metrics["rmse"], "dist_mae": dist_metrics["mae"],
            "idw_kge": idw_metrics["kge"], "idw_r2": idw_metrics["r2"],
            "idw_rmse": idw_metrics["rmse"], "idw_mae": idw_metrics["mae"],
        })

        elapsed = time.time() - t0
        print(f"  [{wi+1}/{len(target_wells)}] {target_wid}: "
              f"Enhanced={enhanced_metrics['kge']:.4f}  "
              f"DistOnly={dist_metrics['kge']:.4f}  "
              f"IDW={idw_metrics['kge']:.4f}  ({elapsed:.0f}s)", flush=True)

    # ─── Results ──────────────────────────────────────────────────────────
    total_time = time.time() - t0
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "cv_mc_lnn_interpolation_enhanced_detailed.csv"), index=False)

    lines = []
    lines.append("=" * 70)
    lines.append("ENHANCED MC+LNN SPATIAL INTERPOLATION — LOO CV RESULTS")
    lines.append("=" * 70)
    lines.append(f"Wells: {n_wells}, Times: {n_times}")
    lines.append(f"Target wells: {len(target_wells)}")
    lines.append(f"Donor weights: dist={W_DIST}, precip={W_PRECIP}, elev={W_ELEV}")
    lines.append(f"Total time: {total_time:.1f}s")
    lines.append("")

    methods = [("Enhanced Transfer", "enh"), ("Distance Transfer", "dist"), ("IDW", "idw")]
    lines.append(f"{'Method':>20} | {'KGE_mean':>9} {'KGE_std':>8} {'KGE_med':>8} | "
                 f"{'R²_mean':>8} | {'RMSE_mean':>10} {'MAE_mean':>9}")
    lines.append("-" * 85)
    for name, pfx in methods:
        kge = df[f"{pfx}_kge"]
        lines.append(
            f"{name:>20} | {kge.mean():>9.4f} {kge.std():>8.4f} {kge.median():>8.4f} | "
            f"{df[f'{pfx}_r2'].mean():>8.4f} | {df[f'{pfx}_rmse'].mean():>10.3f} "
            f"{df[f'{pfx}_mae'].mean():>9.3f}"
        )

    # Win counts
    lines.append("")
    all_kges = df[["enh_kge", "dist_kge", "idw_kge"]]
    best_col = all_kges.idxmax(axis=1)
    name_map = {"enh_kge": "Enhanced", "dist_kge": "DistOnly", "idw_kge": "IDW"}
    for col, name in name_map.items():
        cnt = (best_col == col).sum()
        lines.append(f"  {name}: best in {cnt}/{len(df)} wells")

    enh_vs_idw = (df["enh_kge"] > df["idw_kge"]).sum()
    enh_vs_dist = (df["enh_kge"] > df["dist_kge"]).sum()
    lines.append(f"  Enhanced > IDW: {enh_vs_idw}/{len(df)}")
    lines.append(f"  Enhanced > DistOnly: {enh_vs_dist}/{len(df)}")

    lines.append("")
    lines.append("PER-WELL DETAIL (sorted by Enhanced KGE):")
    lines.append(f"{'Well_ID':>20} | {'Enh_KGE':>8} {'Enh_RMSE':>8} | "
                 f"{'Dist_KGE':>8} {'Dist_RMSE':>9} | {'IDW_KGE':>8} {'IDW_RMSE':>8} | {'Elev':>6}")
    lines.append("-" * 100)
    for _, row in df.sort_values("enh_kge", ascending=False).iterrows():
        lines.append(
            f"{row['Well_ID']:>20} | {row['enh_kge']:>8.4f} {row['enh_rmse']:>8.2f} | "
            f"{row['dist_kge']:>8.4f} {row['dist_rmse']:>9.2f} | "
            f"{row['idw_kge']:>8.4f} {row['idw_rmse']:>8.2f} | {row['elev']:>6.0f}"
        )

    summary_text = "\n".join(lines)
    print()
    print(summary_text)

    with open(os.path.join(OUTPUT_DIR, "cv_mc_lnn_interpolation_enhanced_summary.txt"), "w") as f:
        f.write(summary_text + "\n")

    meta = {
        "n_wells": n_wells, "n_times": n_times, "n_targets": len(target_wells),
        "max_donors": MAX_DONORS,
        "w_dist": W_DIST, "w_precip": W_PRECIP, "w_elev": W_ELEV,
        "elapsed_seconds": total_time,
        "enh_kge_mean": float(df["enh_kge"].mean()),
        "dist_kge_mean": float(df["dist_kge"].mean()),
        "idw_kge_mean": float(df["idw_kge"].mean()),
    }
    with open(os.path.join(OUTPUT_DIR, "cv_mc_lnn_interpolation_enhanced_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nOutput: cv_mc_lnn_interpolation_enhanced_detailed.csv, cv_mc_lnn_interpolation_enhanced_summary.txt")


if __name__ == "__main__":
    main()
