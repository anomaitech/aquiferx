#!/usr/bin/env python3
"""
MC Raster Interpolation Cross-Validation (GSLB)

Tests the SoftImpute matrix completion for spatial interpolation using
the fully imputed GSLB dataset (592 wells, 288 months, 2000-2023).

Approach: For each target well, hold out temporal blocks (random months
or consecutive years) and let MC recover them from the spatial correlation
with the other 591 wells at those time steps.

This mirrors how MC raster interpolation works in the browser
(rasterAnalysis.ts → buildCompletedMcWellValues): temporal gaps in
individual wells are filled using the low-rank structure across all wells.

Experiments:
  A) Random holdout: 10%, 20%, 30%, 40%, 50% of months removed (20 trials)
  B) Consecutive gaps: 1, 2, 3, 4, 5 years removed (20 trials)

Tests on top-10 most observed wells, using all 592 wells as the matrix.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from hashlib import sha256
from typing import Any, Dict, List, Optional, Tuple

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

# SoftImpute parameters (matching browser rasterAnalysis.ts)
MC_MAX_ITERATIONS = 80
MC_TOLERANCE = 1e-5
MC_SHRINKAGE = 0
RANK_OPTIONS = [3, 5, 8, 10, 12]

# CV config
N_TARGET_WELLS = 10
RANDOM_PERCENTAGES = [10, 20, 30, 40, 50]
RANDOM_REPEATS = 20
CONSECUTIVE_YEARS = [1, 2, 3, 4, 5]
CONSECUTIVE_REPEATS = 20
BASE_SEED = 42


def _rng_seed(*parts: Any) -> int:
    h = sha256(b"|".join(str(p).encode() for p in parts)).digest()
    return int.from_bytes(h[:8], "little") % (2**31)


# ─── SoftImpute (matches mcSoftImpute.ts) ─────────────────────────────────────

def soft_impute(matrix: np.ndarray, rank: int, max_iterations: int = 80,
                tolerance: float = 1e-5, shrinkage: float = 0) -> np.ndarray:
    """SoftImpute matrix completion. Matches browser mcSoftImpute.ts."""
    M = matrix.copy()
    mask = ~np.isnan(M)

    # Fill missing with column means
    col_means = np.nanmean(M, axis=0)
    col_means = np.where(np.isnan(col_means), 0, col_means)
    for c in range(M.shape[1]):
        M[np.isnan(M[:, c]), c] = col_means[c]

    for iteration in range(max_iterations):
        M_prev = M.copy()

        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        use_rank = max(1, min(rank, len(S)))
        S_trunc = np.zeros_like(S)
        for i in range(use_rank):
            S_trunc[i] = max(0, S[i] - shrinkage)

        M_new = U @ np.diag(S_trunc) @ Vt
        M_new[mask] = matrix[mask]
        M = M_new

        diff = np.linalg.norm(M - M_prev)
        denom = max(np.linalg.norm(M_prev), 1e-12)
        if diff / denom <= tolerance:
            break

    return M


def soft_impute_adaptive(matrix: np.ndarray, target_col: int,
                         holdout_rows: np.ndarray) -> np.ndarray:
    """Run SoftImpute with adaptive rank selection."""
    # For rank selection, use a small validation set from OTHER columns
    rng = np.random.default_rng(42)
    n_times, n_wells = matrix.shape
    other_cols = [c for c in range(n_wells) if c != target_col]

    best_rank = 5
    best_err = float("inf")

    for k in RANK_OPTIONS:
        if k >= min(matrix.shape):
            continue
        completed = soft_impute(matrix, rank=k, max_iterations=50,
                                tolerance=MC_TOLERANCE, shrinkage=MC_SHRINKAGE)
        # Evaluate on the holdout positions of the target well
        # But we can't use truth here for rank selection (that's cheating)
        # Instead use internal convergence quality: reconstruction error on
        # observed positions of the target well
        obs_rows = np.array([t for t in range(n_times) if t not in holdout_rows
                             and not np.isnan(matrix[t, target_col])])
        if len(obs_rows) > 3:
            err = float(np.mean((completed[obs_rows, target_col] -
                                  matrix[obs_rows, target_col])**2))
            if err < best_err:
                best_err = err
                best_rank = k

    # Final run with best rank
    return soft_impute(matrix, rank=best_rank, max_iterations=MC_MAX_ITERATIONS,
                       tolerance=MC_TOLERANCE, shrinkage=MC_SHRINKAGE), best_rank


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(obs: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    if len(obs) < 2:
        return {"n": len(obs), "kge": float("nan"), "r2": float("nan"),
                "rmse": float("nan"), "mae": float("nan"), "nse": float("nan")}

    err = pred - obs
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    mean_o, mean_p = float(np.mean(obs)), float(np.mean(pred))
    std_o = float(np.std(obs, ddof=1))
    std_p = float(np.std(pred, ddof=1))

    r = float(np.corrcoef(obs, pred)[0, 1]) if std_o > 1e-10 and std_p > 1e-10 else 0.0

    ss_tot = float(np.sum((obs - mean_o)**2))
    ss_res = float(np.sum(err**2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12) if ss_tot > 1e-12 else float("nan")
    nse = r2

    alpha = std_p / std_o if std_o > 1e-10 else 1.0
    beta = mean_p / mean_o if abs(mean_o) > 1e-10 else 1.0
    kge = 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    return {"n": len(obs), "kge": float(kge), "r2": float(r2),
            "rmse": rmse, "mae": mae, "nse": float(nse), "r": float(r)}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 70)
    print("MC Raster Interpolation — Temporal Holdout CV (GSLB)")
    print("=" * 70)

    # Load fully imputed data
    print("Loading fully imputed dataset...")
    imputed_df = pd.read_csv(IMPUTED_CSV)
    imputed_df["Well_ID"] = imputed_df["Well_ID"].astype(str)
    imputed_df["Date"] = pd.to_datetime(imputed_df["Date"])
    imputed_df = imputed_df.sort_values(["Well_ID", "Date"])

    # Load raw data for observation counts
    raw_df = pd.read_csv(RAW_CSV)
    raw_df["Date"] = pd.to_datetime(raw_df["Date"])
    raw_df = raw_df[(raw_df["Date"].dt.year >= 2000) & (raw_df["Date"].dt.year <= 2023)]
    raw_df["Well_ID"] = raw_df["Well_ID"].astype(str)

    # Build complete matrix: nTimes x nWells
    well_ids = sorted(imputed_df["Well_ID"].unique())
    dates = sorted(imputed_df["Date"].unique())
    n_wells = len(well_ids)
    n_times = len(dates)
    print(f"Matrix: {n_times} times x {n_wells} wells")

    well_idx = {wid: i for i, wid in enumerate(well_ids)}
    date_idx = {d: i for i, d in enumerate(dates)}

    full_matrix = np.full((n_times, n_wells), np.nan)
    for _, row in imputed_df.iterrows():
        t = date_idx[row["Date"]]
        w = well_idx[row["Well_ID"]]
        full_matrix[t, w] = row["final_wte"]

    assert not np.any(np.isnan(full_matrix)), "Matrix should be complete"
    print(f"Complete: no NaN")

    # Select top-N target wells by raw obs count
    raw_counts = raw_df.groupby("Well_ID")["Date"].nunique().reset_index(name="n_obs")
    raw_counts = raw_counts[raw_counts["Well_ID"].isin(well_ids)]
    raw_counts = raw_counts.sort_values("n_obs", ascending=False)
    target_wells = raw_counts.head(N_TARGET_WELLS)["Well_ID"].tolist()
    print(f"Target wells: {len(target_wells)}")
    print()

    rows_random = []
    rows_consec = []

    # ═══ Experiment A: Random temporal holdout ═════════════════════════════
    total_a = len(target_wells) * len(RANDOM_PERCENTAGES) * RANDOM_REPEATS
    done = 0
    print(f"Experiment A: Random temporal holdout ({total_a} folds)")
    print("-" * 50)

    for wid in target_wells:
        col = well_idx[wid]
        for pct in RANDOM_PERCENTAGES:
            for rep in range(RANDOM_REPEATS):
                rng = np.random.default_rng(_rng_seed(BASE_SEED, "random", wid, pct, rep))
                n_hold = max(1, int(n_times * pct / 100.0))
                holdout_idx = rng.choice(n_times, size=n_hold, replace=False)
                truth = full_matrix[holdout_idx, col].copy()

                # Create matrix with holdout
                M = full_matrix.copy()
                M[holdout_idx, col] = np.nan

                completed, rank = soft_impute_adaptive(M, col, holdout_idx)
                pred = completed[holdout_idx, col]
                metrics = compute_metrics(truth, pred)

                rows_random.append({
                    "Well_ID": wid, "pct_missing": pct, "rep": rep,
                    "n_held": n_hold, "rank": rank, **metrics,
                })

                done += 1
                if done % 50 == 0 or done == total_a:
                    elapsed = time.time() - t0
                    print(f"  [{done}/{total_a}] {elapsed:.0f}s", flush=True)

    # ═══ Experiment B: Consecutive year gaps ═══════════════════════════════
    total_b = len(target_wells) * len(CONSECUTIVE_YEARS) * CONSECUTIVE_REPEATS
    done = 0
    t1 = time.time()
    print()
    print(f"Experiment B: Consecutive year gaps ({total_b} folds)")
    print("-" * 50)

    for wid in target_wells:
        col = well_idx[wid]
        for n_years in CONSECUTIVE_YEARS:
            n_months = n_years * 12
            for rep in range(CONSECUTIVE_REPEATS):
                rng = np.random.default_rng(_rng_seed(BASE_SEED, "consec", wid, n_years, rep))
                max_start = n_times - n_months
                if max_start < 0:
                    rows_consec.append({
                        "Well_ID": wid, "n_years": n_years, "rep": rep,
                        "status": "skip",
                    })
                    done += 1
                    continue

                start = int(rng.integers(0, max_start + 1))
                holdout_idx = np.arange(start, start + n_months)
                truth = full_matrix[holdout_idx, col].copy()

                M = full_matrix.copy()
                M[holdout_idx, col] = np.nan

                completed, rank = soft_impute_adaptive(M, col, holdout_idx)
                pred = completed[holdout_idx, col]
                metrics = compute_metrics(truth, pred)

                window_start = pd.Timestamp(dates[start]).strftime("%Y-%m")
                window_end = pd.Timestamp(dates[min(start + n_months - 1, n_times - 1)]).strftime("%Y-%m")
                rows_consec.append({
                    "Well_ID": wid, "n_years": n_years, "rep": rep,
                    "n_held": n_months, "rank": rank,
                    "window_start": window_start, "window_end": window_end,
                    **metrics,
                })

                done += 1
                if done % 20 == 0 or done == total_b:
                    elapsed = time.time() - t1
                    print(f"  [{done}/{total_b}] {elapsed:.0f}s", flush=True)

    # ═══ Results ══════════════════════════════════════════════════════════
    total_time = time.time() - t0
    print()
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    df_random = pd.DataFrame(rows_random)
    df_consec = pd.DataFrame([r for r in rows_consec if "kge" in r])

    df_random.to_csv(os.path.join(OUTPUT_DIR, "cv_mc_raster_random_detailed.csv"), index=False)
    pd.DataFrame(rows_consec).to_csv(os.path.join(OUTPUT_DIR, "cv_mc_raster_consecutive_detailed.csv"), index=False)

    metric_cols = ["kge", "r2", "rmse", "mae", "nse"]
    lines = []
    lines.append("=" * 70)
    lines.append("MC RASTER INTERPOLATION — TEMPORAL HOLDOUT CV RESULTS")
    lines.append("=" * 70)
    lines.append(f"Dataset: {n_wells} wells, {n_times} months (2000-2023)")
    lines.append(f"Target wells: {len(target_wells)}")
    lines.append(f"Rank options: {RANK_OPTIONS}")
    lines.append(f"Total time: {total_time:.1f}s")
    lines.append("")

    # Random summary
    lines.append("EXPERIMENT A: RANDOM TEMPORAL HOLDOUT")
    lines.append("-" * 70)
    if len(df_random):
        lines.append(f"{'%Miss':>6} {'N':>5} | {'KGE_mean':>9} {'KGE_std':>8} {'KGE_med':>8} | "
                     f"{'R²_mean':>8} {'R²_std':>7} | {'RMSE_mean':>10} {'RMSE_std':>9} | "
                     f"{'MAE_mean':>9} {'MAE_std':>8}")
        lines.append("-" * 110)
        for pct in RANDOM_PERCENTAGES:
            sub = df_random[df_random["pct_missing"] == pct]
            if len(sub) == 0:
                continue
            n = len(sub)
            lines.append(
                f"{pct:>5}% {n:>5} | "
                f"{sub['kge'].mean():>9.4f} {sub['kge'].std():>8.4f} {sub['kge'].median():>8.4f} | "
                f"{sub['r2'].mean():>8.4f} {sub['r2'].std():>7.4f} | "
                f"{sub['rmse'].mean():>10.3f} {sub['rmse'].std():>9.3f} | "
                f"{sub['mae'].mean():>9.3f} {sub['mae'].std():>8.3f}"
            )

    lines.append("")
    lines.append("EXPERIMENT B: CONSECUTIVE YEAR GAPS")
    lines.append("-" * 70)
    if len(df_consec):
        lines.append(f"{'Years':>6} {'N':>5} | {'KGE_mean':>9} {'KGE_std':>8} {'KGE_med':>8} | "
                     f"{'R²_mean':>8} {'R²_std':>7} | {'RMSE_mean':>10} {'RMSE_std':>9} | "
                     f"{'MAE_mean':>9} {'MAE_std':>8}")
        lines.append("-" * 110)
        for ny in CONSECUTIVE_YEARS:
            sub = df_consec[df_consec["n_years"] == ny]
            if len(sub) == 0:
                continue
            n = len(sub)
            lines.append(
                f"{ny:>5}y {n:>5} | "
                f"{sub['kge'].mean():>9.4f} {sub['kge'].std():>8.4f} {sub['kge'].median():>8.4f} | "
                f"{sub['r2'].mean():>8.4f} {sub['r2'].std():>7.4f} | "
                f"{sub['rmse'].mean():>10.3f} {sub['rmse'].std():>9.3f} | "
                f"{sub['mae'].mean():>9.3f} {sub['mae'].std():>8.3f}"
            )

    # Per-well summary
    lines.append("")
    lines.append("PER-WELL AVERAGE (across all experiments):")
    lines.append(f"{'Well_ID':>20} {'KGE':>8} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
    lines.append("-" * 60)
    all_results = pd.concat([df_random, df_consec], ignore_index=True)
    for wid in target_wells:
        sub = all_results[all_results["Well_ID"] == wid]
        if len(sub) == 0:
            continue
        lines.append(
            f"{wid:>20} {sub['kge'].mean():>8.4f} {sub['r2'].mean():>8.4f} "
            f"{sub['rmse'].mean():>8.2f} {sub['mae'].mean():>8.2f}"
        )

    summary_text = "\n".join(lines)
    print()
    print(summary_text)

    with open(os.path.join(OUTPUT_DIR, "cv_mc_raster_summary.txt"), "w") as f:
        f.write(summary_text + "\n")

    meta = {
        "n_wells_total": n_wells,
        "n_times": n_times,
        "n_target_wells": len(target_wells),
        "target_wells": target_wells,
        "rank_options": RANK_OPTIONS,
        "random_percentages": RANDOM_PERCENTAGES,
        "random_repeats": RANDOM_REPEATS,
        "consecutive_years": CONSECUTIVE_YEARS,
        "consecutive_repeats": CONSECUTIVE_REPEATS,
        "elapsed_seconds": total_time,
    }
    with open(os.path.join(OUTPUT_DIR, "cv_mc_raster_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print()
    print("Output files:")
    print(f"  cv_mc_raster_random_detailed.csv")
    print(f"  cv_mc_raster_consecutive_detailed.csv")
    print(f"  cv_mc_raster_summary.txt")
    print(f"  cv_mc_raster_metadata.json")


if __name__ == "__main__":
    main()
