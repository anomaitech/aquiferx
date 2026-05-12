#!/usr/bin/env python3
"""
MC+LNN Robustness Cross-Validation

Tests the MC+LNN imputation pipeline under two missing-data scenarios:
  A) Random missing: 5%, 10%, 20%, 30%, 40%, 50% of observed months removed (50 trials each)
  B) Consecutive year gaps: 1, 2, 3, 4, 5 years removed (20 trials each)

Uses the most complete well (415703112514501, 202/288 months in 2000-2023).
Reports: KGE, R², RMSE, MAE, NSE per scenario with mean/std/median.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import warnings
from contextlib import redirect_stdout
from dataclasses import asdict
from hashlib import sha256
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
sys.path.insert(0, APP_DIR)

from backend.lnn.lnn_core_aux_placeholder_cfc import optimize_lnn_params, run_lnn_simulation
from backend.lnn.math_utils import calculate_kge, calculate_pearson_correlation, calculate_r2
from backend.lnn.types import DataPoint, SimulationParams
from scipy.interpolate import PchipInterpolator

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────

OUTPUT_DIR = APP_DIR
TARGET_CSV = os.path.join(PROJECT_ROOT, "datas", "measurements_till_2023_to_lnn_imputation.csv")
AUX_CSV = os.path.join(PROJECT_ROOT, "datas", "lnn_imputation_gslb_gldas_df_excercise.csv")

DATE_START = "2000-01-01"
DATE_END = "2023-12-31"
AUX_COLS = ["soilw", "soilw_yr01", "soilw_yr03", "soilw_yr05", "soilw_yr10"]

# Target well (most complete: 202/288 months)
TARGET_WELL = "415703112514501"

# Donor pool: top 50 wells by coverage
DONOR_POOL_SIZE = 50
MAX_DONORS = 15
MIN_DONOR_CORR = 0.3

# Random missing experiments
RANDOM_PERCENTAGES = [5, 10, 20, 30, 40, 50]
RANDOM_REPEATS = 50

# Consecutive gap experiments
CONSECUTIVE_YEARS = [1, 2, 3, 4, 5]
CONSECUTIVE_REPEATS = 20

BASE_SEED = 42
MIN_POINTS_PER_INSTANCE = 10


def _rng_seed(*parts: Any) -> int:
    h = sha256(b"|".join(str(p).encode() for p in parts)).digest()
    return int.from_bytes(h[:8], "little") % (2**31)


def build_params() -> SimulationParams:
    return SimulationParams(
        max_gap_threshold=20,
        large_gap_threshold=200,
        kge_threshold=0.5,
        small_gap_kge_threshold=0.30,
        large_gap_kge_threshold=0.4,
        small_gap_use_auxiliary=True,
        small_gap_auxiliary_weight=0.5,
        small_gap_use_neighbors=False,
        small_gap_max_iterations=3,
        small_gap_optimize_trials=8,
        ridge_alpha=1e-4,
        lnn_aux_placeholder_readout="ridge",
        lnn_aux_placeholder_spike_correction=False,
        archi_use_regional_correlation=True,
        archi_min_correlation=0.3,
        archi_max_donors=MAX_DONORS,
        archi_correlation_weight_power=2.0,
        mc_max_donors=MAX_DONORS,
        mc_min_correlation=MIN_DONOR_CORR,
    )


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data():
    main_df = pd.read_csv(TARGET_CSV)
    aux_df = pd.read_csv(AUX_CSV)
    main_df["Date"] = pd.to_datetime(main_df["Date"])
    aux_df["time"] = pd.to_datetime(aux_df["time"])
    main_df = main_df[(main_df["Date"] >= DATE_START) & (main_df["Date"] <= DATE_END)].copy()
    aux_df = aux_df[(aux_df["time"] >= DATE_START) & (aux_df["time"] <= DATE_END)].copy()
    main_df["Well_ID"] = main_df["Well_ID"].astype(str)
    main_df["month"] = main_df["Date"].dt.to_period("M")
    aux_df = aux_df.sort_values("time").set_index("time")

    months = pd.date_range(DATE_START, DATE_END, freq="MS")

    # Find eligible wells (>= MIN_POINTS)
    counts = main_df.groupby("Well_ID")["month"].nunique()
    eligible = counts[counts >= MIN_POINTS_PER_INSTANCE]
    stats = []
    for wid in eligible.index.tolist():
        seen = set(main_df.loc[main_df["Well_ID"] == wid, "month"].unique())
        cur = mx = 0
        for p in pd.period_range("2000-01", "2023-12", freq="M"):
            if p in seen:
                cur += 1
                mx = max(mx, cur)
            else:
                cur = 0
        stats.append((wid, mx, int(eligible[wid])))
    stats.sort(key=lambda x: (-x[1], -x[2], x[0]))
    donor_pool_wells = [wid for wid, _, _ in stats[:DONOR_POOL_SIZE]]

    latlon: Dict[str, Tuple[float, float]] = {}
    for wid, _, _ in stats:
        r0 = main_df[main_df["Well_ID"] == wid].iloc[0]
        latlon[wid] = (float(r0["lat_dec"]), float(r0["long_dec"]))

    return main_df, aux_df, months, donor_pool_wells, latlon


def aux_lookup(aux_df: pd.DataFrame) -> Dict[Tuple[int, int], List[float]]:
    out = {}
    for dt, row in aux_df.iterrows():
        out[(dt.year, dt.month)] = [float(row[c]) if c in row.index and pd.notna(row[c]) else 0.0 for c in AUX_COLS]
    return out


def monthly_series(main_df: pd.DataFrame, well_id: str, months: pd.DatetimeIndex) -> List[Optional[float]]:
    monthly = main_df.loc[main_df["Well_ID"] == well_id].groupby("month")["WTE"].mean()
    return [float(monthly.get(m.to_period("M"))) if pd.notna(monthly.get(m.to_period("M"))) else None for m in months]


def build_aux_timeline(obs_list, well_id, months, aux_lk, lat, lon, add_seasonal=False):
    tl = []
    for mi, m in enumerate(months):
        aux = list(aux_lk.get((m.year, m.month), [0.0] * len(AUX_COLS)))
        if add_seasonal:
            aux.extend([np.sin(2 * np.pi * mi / 12.0), np.cos(2 * np.pi * mi / 12.0)])
        tl.append(DataPoint(
            time=float(mi), observed=obs_list[mi], instance_id=well_id,
            auxiliaries=aux, date_label=m.strftime("%Y-%m"),
            latitude=lat, longitude=lon,
        ))
    return tl


# ─── PCHIP small-gap fill (matches browser mcLnnPureBrowser.ts) ──────────────

def pchip_fill(series: List[Optional[float]], max_gap: int = 24) -> List[Optional[float]]:
    """Fill small gaps (≤ max_gap months) using PCHIP interpolation.
    Mirrors the browser pchipFill() in mcLnnPureBrowser.ts."""
    out = list(series)
    obs_idx = [i for i, v in enumerate(series) if v is not None]
    obs_vals = [series[i] for i in obs_idx]
    if len(obs_idx) < 3:
        return out

    # Identify small gaps
    gaps = []
    i = 0
    while i < len(series):
        if series[i] is None:
            start = i
            while i < len(series) and series[i] is None:
                i += 1
            if i - start <= max_gap:
                gaps.append((start, i - 1))
        else:
            i += 1
    if not gaps:
        return out

    # PCHIP interpolation
    pchip = PchipInterpolator(obs_idx, obs_vals)
    for g_start, g_end in gaps:
        for t in range(g_start, g_end + 1):
            val = float(pchip(t))
            if np.isfinite(val):
                out[t] = val
    return out


def prefill_small_gaps_all(raw_obs, gap_size=24, pad_size=6):
    """Phase 1: PCHIP small-gap fill + large-gap blanking for all wells.
    Matches browser mcLnnPureBrowser.ts Phase 1."""
    filled = {}
    for wid, raw in raw_obs.items():
        # PCHIP fill all gaps first
        out = pchip_fill(raw, max_gap=gap_size * 2)

        # Blank interior of large gaps (keep pad_size months at edges)
        obs_idx = [i for i, v in enumerate(raw) if v is not None]
        for g in range(len(obs_idx) - 1):
            gap_len = obs_idx[g + 1] - obs_idx[g] - 1
            if gap_len > gap_size:
                blank_start = obs_idx[g] + pad_size + 1
                blank_end = obs_idx[g + 1] - pad_size - 1
                for t in range(max(blank_start, 0), min(blank_end, len(out) - 1) + 1):
                    out[t] = None
        filled[wid] = out
    return filled


# ─── Imputation helpers ──────────────────────────────────────────────────────

def impute_timeline(timeline, params, seed):
    # Compute observed range for output clamping
    obs_vals = [d.observed for d in timeline if d.observed is not None]
    if len(obs_vals) >= 2:
        o_min, o_max = min(obs_vals), max(obs_vals)
        o_range = o_max - o_min
        clamp_margin = max(o_range * 0.3, 2.0)
    else:
        o_min, o_max, clamp_margin = -1e18, 1e18, 1e18

    with redirect_stdout(io.StringIO()):
        best = optimize_lnn_params(timeline, params, mode="projection", rng=np.random.default_rng(seed))
        best_result = None
        best_kge = float("-inf")
        for it in range(5):
            res = run_lnn_simulation(timeline, best, rng=np.random.default_rng(seed + it))
            # Clamp LNN output to observed range
            for r in res:
                if r.imputed is not None:
                    r.imputed = max(o_min - clamp_margin, min(o_max + clamp_margin, r.imputed))
            obs = [d.observed for d in timeline if d.observed is not None]
            pred = [res[i].imputed for i, d in enumerate(timeline) if d.observed is not None and res[i].imputed is not None]
            if obs and pred and len(obs) == len(pred):
                k = float(calculate_kge(obs, pred))
                if np.isfinite(k) and k > best_kge:
                    best_kge = k
                    best_result = res
        result = best_result if best_result is not None else run_lnn_simulation(timeline, best, rng=np.random.default_rng(seed))
        # Final clamp
        for r in result:
            if r.imputed is not None:
                r.imputed = max(o_min - clamp_margin, min(o_max + clamp_margin, r.imputed))
        return result


# ─── ARCHI + MC ───────────────────────────────────────────────────────────────

def _archi_regression(target_obs, donor_obs, target_id):
    if len(target_obs) < 5:
        return {}, []
    gap_times = sorted(set(range(288)) - set(target_obs.keys()))
    donors = []
    for wid, series in donor_obs.items():
        if wid == target_id:
            continue
        dobs = {i: v for i, v in enumerate(series) if v is not None}
        common = sorted(set(target_obs.keys()) & set(dobs.keys()))
        if len(common) < 8:
            continue
        tv = np.array([target_obs[t] for t in common])
        dv = np.array([dobs[t] for t in common])
        if np.std(tv) < 1e-10 or np.std(dv) < 1e-10:
            continue
        r = float(np.corrcoef(tv, dv)[0, 1])
        if abs(r) < MIN_DONOR_CORR:
            continue
        donors.append({"wid": wid, "r": r, "dobs": dobs})
    donors.sort(key=lambda x: abs(x["r"]), reverse=True)
    donors = donors[:MAX_DONORS]
    if not donors:
        return {}, []

    preds_by_donor = []
    weights = []
    for di in donors:
        common = sorted(set(target_obs.keys()) & set(di["dobs"].keys()))
        if len(common) < 5:
            continue
        tv = np.array([target_obs[t] for t in common])
        dv = np.array([di["dobs"][t] for t in common])
        dm, tm = np.mean(dv), np.mean(tv)
        ss = np.sum((dv - dm) ** 2)
        if ss < 1e-10:
            continue
        a = float(np.sum((dv - dm) * (tv - tm)) / ss)
        b = float(tm - a * dm)
        preds = {t: a * di["dobs"][t] + b for t in gap_times if t in di["dobs"]}
        if preds:
            preds_by_donor.append(preds)
            weights.append(di["r"] ** 2)

    combined = {}
    for t in gap_times:
        ws = wc = 0.0
        for i, preds in enumerate(preds_by_donor):
            if t in preds:
                ws += preds[t] * weights[i]
                wc += weights[i]
        if wc > 0:
            combined[t] = ws / wc
    return combined, donors


def _matrix_completion_archi_init(target_series, target_aux, donor_obs, donors, archi_preds):
    n_times = len(target_series)
    target_obs_idx = [i for i, v in enumerate(target_series) if v is not None]
    sel_wids = [None] + [d["wid"] for d in donors]
    nw = len(sel_wids)
    n_aux = len(target_aux[0]) if target_aux else 0
    M_raw = np.full((nw, n_times), np.nan)
    for t, v in enumerate(target_series):
        if v is not None:
            M_raw[0, t] = v
    for wi, wid in enumerate(sel_wids[1:], start=1):
        series = donor_obs[wid]
        for t, v in enumerate(series):
            if v is not None:
                M_raw[wi, t] = v

    M = np.full((nw + n_aux + 2, n_times), np.nan)
    rmeans = np.zeros(nw)
    rstds = np.ones(nw)
    for wi in range(nw):
        vals = M_raw[wi, ~np.isnan(M_raw[wi, :])]
        if len(vals) >= 3:
            rmeans[wi] = np.mean(vals)
            rstds[wi] = max(np.std(vals), 1e-10)
        elif len(vals) > 0:
            rmeans[wi] = np.mean(vals)
        M[wi, :] = (M_raw[wi, :] - rmeans[wi]) / rstds[wi]

    for di, donor in enumerate(donors, start=1):
        M[di, :] *= abs(donor["r"])

    for j in range(n_aux):
        ac = np.array([target_aux[t][j] for t in range(n_times)], dtype=float)
        am, ast = np.mean(ac), max(np.std(ac), 1e-10)
        M[nw + j, :] = (ac - am) / ast

    for t in range(n_times):
        M[nw + n_aux, t] = np.sin(2 * np.pi * t / 12.0)
        M[nw + n_aux + 1, t] = np.cos(2 * np.pi * t / 12.0)

    obs_mask = ~np.isnan(M)
    X = M.copy()
    for t in range(n_times):
        if np.isnan(X[0, t]):
            X[0, t] = (archi_preds[t] - rmeans[0]) / rstds[0] if t in archi_preds else 0.0
    for wi in range(1, nw):
        row_vals = M[wi, ~np.isnan(M[wi, :])]
        row_mean = np.mean(row_vals) if len(row_vals) else 0.0
        X[wi, np.isnan(X[wi, :])] = row_mean

    best_k = 5
    best_err = float("inf")
    for k_try in [3, 5, 8, 10, 12]:
        if k_try >= min(M.shape):
            continue
        Xt = X.copy()
        for _ in range(50):
            U, S, Vt = np.linalg.svd(Xt, full_matrices=False)
            St = np.zeros_like(S)
            St[:k_try] = S[:k_try]
            Xn = U @ np.diag(St) @ Vt
            Xn[obs_mask] = M[obs_mask]
            if np.max(np.abs(Xn - Xt)) < 1e-6:
                break
            Xt = Xn
        if len(target_obs_idx) > 3:
            err = float(np.mean((Xt[0, target_obs_idx] - M[0, target_obs_idx]) ** 2))
            if err < best_err:
                best_err = err
                best_k = k_try

    for _ in range(100):
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        St = np.zeros_like(S)
        St[:best_k] = S[:best_k]
        Xn = U @ np.diag(St) @ Vt
        Xn[obs_mask] = M[obs_mask]
        if np.max(np.abs(Xn - X)) < 1e-6:
            break
        X = Xn

    for di, donor in enumerate(donors, start=1):
        if abs(donor["r"]) > 1e-10:
            X[di, :] /= abs(donor["r"])

    pred = X[0, :] * rstds[0] + rmeans[0]

    # MOVE.1 with safeguard for low/high variance wells
    if len(target_obs_idx) >= 3:
        ov = M_raw[0, target_obs_idx]
        mv = pred[target_obs_idx]
        om, os_ = np.mean(ov), max(np.std(ov), 1e-10)
        mm, ms = np.mean(mv), max(np.std(mv), 1e-10)
        var_ratio = os_ / ms
        if 0.1 < var_ratio < 10:
            pred = (pred - mm) / ms * os_ + om

    # Clamp to observed range + margin
    if len(target_obs_idx) >= 2:
        obs_vals = M_raw[0, target_obs_idx]
        obs_min, obs_max = float(np.min(obs_vals)), float(np.max(obs_vals))
        obs_range = obs_max - obs_min
        margin = max(obs_range * 0.2, 1.0)
        pred = np.clip(pred, obs_min - margin, obs_max + margin)

    # Quality check: if MC RMSE on observed is worse than 1.5x std, blend with ARCHI
    if len(target_obs_idx) > 0:
        mc_rmse = float(np.sqrt(np.mean((pred[target_obs_idx] - M_raw[0, target_obs_idx])**2)))
        if mc_rmse > rstds[0] * 1.5 and archi_preds:
            for t in range(n_times):
                if t in archi_preds and np.isfinite(archi_preds[t]):
                    pred[t] = 0.3 * pred[t] + 0.7 * archi_preds[t]

    return {t: float(pred[t]) for t in range(n_times)}


def run_mc_lnn_fold(target_id, mod_target, donor_obs_filled, months, aux_lk, latlon, params, seed):
    lat, lon = latlon[target_id]
    target_obs = {i: v for i, v in enumerate(mod_target) if v is not None}
    archi_preds, donors = _archi_regression(target_obs, donor_obs_filled, target_id)
    target_aux = [list(aux_lk.get((m.year, m.month), [0.0] * len(AUX_COLS))) for m in months]

    if not donors:
        tl = build_aux_timeline(mod_target, target_id, months, aux_lk, lat, lon, add_seasonal=True)
        return impute_timeline(tl, params, seed)

    mc_preds = _matrix_completion_archi_init(mod_target, target_aux, donor_obs_filled, donors, archi_preds)

    tl = build_aux_timeline(mod_target, target_id, months, aux_lk, lat, lon, add_seasonal=True)

    # Clamp MC placeholders to observed range
    all_obs = [v for v in mod_target if v is not None]
    if all_obs:
        obs_min, obs_max = min(all_obs), max(all_obs)
        obs_range = obs_max - obs_min
        clamp_margin = max(obs_range * 0.3, 2.0)
        clamp_lo, clamp_hi = obs_min - clamp_margin, obs_max + clamp_margin
    else:
        clamp_lo, clamp_hi = -np.inf, np.inf

    for i in range(len(tl)):
        if mod_target[i] is None and i in mc_preds and np.isfinite(mc_preds[i]):
            val = max(clamp_lo, min(clamp_hi, float(mc_preds[i])))
            tl[i] = DataPoint(
                time=tl[i].time, observed=None, imputed=val,
                instance_id=tl[i].instance_id, auxiliaries=tl[i].auxiliaries,
                date_label=tl[i].date_label, latitude=tl[i].latitude, longitude=tl[i].longitude,
            )
    return impute_timeline(tl, params, seed)


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(obs, pred):
    o = np.asarray(obs, dtype=float)
    p = np.asarray(pred, dtype=float)
    if o.size < 1 or p.shape != o.shape:
        return None
    err = p - o
    mse = float(np.mean(err**2))
    out = {
        "n_scored": float(o.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(mse)),
    }
    if o.size < 2:
        out.update({"r2": float("nan"), "kge": float("nan"), "nse": float("nan")})
        return out
    std_o = float(np.std(o, ddof=1))
    mean_o = float(np.mean(o))
    r = float(calculate_pearson_correlation(list(o), list(p)))
    kge = float(calculate_kge(list(o), list(p)))
    ss_tot = float(np.sum((o - mean_o) ** 2))
    nse = 1.0 - float(np.sum(err**2)) / (ss_tot + 1e-12) if ss_tot > 1e-12 else float("nan")
    out.update({
        "r": r if np.isfinite(r) else float("nan"),
        "r2": float(calculate_r2(list(o), list(p))),
        "kge": kge if np.isfinite(kge) else float("nan"),
        "nse": float(nse),
    })
    return out


def metrics_on_truth(result, truth):
    obs, pred = [], []
    for idx, tv in truth.items():
        if idx < len(result) and result[idx].imputed is not None:
            obs.append(tv)
            pred.append(float(result[idx].imputed))
    return compute_metrics(obs, pred)


# ─── Experiment A: Random missing ─────────────────────────────────────────────

def random_holdout(raw, pct, rng):
    """Remove pct% of observed months at random. Return {index: truth_value}."""
    obs_idx = [i for i, v in enumerate(raw) if v is not None]
    n_remove = max(1, int(len(obs_idx) * pct / 100.0))
    if n_remove >= len(obs_idx) - 5:
        n_remove = len(obs_idx) - 5  # keep at least 5 observations
    chosen = rng.choice(obs_idx, size=n_remove, replace=False)
    return {int(i): float(raw[i]) for i in chosen}


# ─── Experiment B: Consecutive gap ────────────────────────────────────────────

def consecutive_holdout(raw, n_years, rng, min_truth=1):
    """Remove a consecutive window of n_years. Return (start, {index: truth})."""
    n_months = n_years * 12
    if len(raw) < n_months:
        return None, {}
    valid_starts = []
    for s in range(0, len(raw) - n_months + 1):
        n_obs = sum(1 for i in range(s, s + n_months) if raw[i] is not None)
        if n_obs >= min_truth:
            valid_starts.append(s)
    if not valid_starts:
        return None, {}
    start = int(rng.choice(valid_starts))
    truth = {i: float(raw[i]) for i in range(start, start + n_months) if raw[i] is not None}
    return start, truth


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 70)
    print("MC+LNN Robustness Cross-Validation")
    print("=" * 70)
    print(f"Target well: {TARGET_WELL}")
    print(f"Date range: {DATE_START} to {DATE_END}")
    print(f"Random missing: {RANDOM_PERCENTAGES}% x {RANDOM_REPEATS} trials each")
    print(f"Consecutive gaps: {CONSECUTIVE_YEARS} years x {CONSECUTIVE_REPEATS} trials each")
    print()

    params = build_params()
    main_df, aux_df, months, donor_pool_wells, latlon = load_data()
    aux_lk = aux_lookup(aux_df)

    # Ensure target well is in pool
    if TARGET_WELL not in donor_pool_wells:
        donor_pool_wells.append(TARGET_WELL)

    # Load raw observations for all pool wells
    raw_obs = {wid: monthly_series(main_df, wid, months) for wid in donor_pool_wells}
    target_raw = raw_obs[TARGET_WELL]
    n_obs_target = sum(1 for v in target_raw if v is not None)
    print(f"Target well observations: {n_obs_target}/{len(months)} months")
    print(f"Donor pool: {len(donor_pool_wells)} wells")
    print()

    # Phase 1: PCHIP small-gap fill + large-gap blanking (matches browser pipeline)
    print("Phase 1: PCHIP small-gap fill for all wells...")
    filled_obs = prefill_small_gaps_all(raw_obs, gap_size=24, pad_size=6)
    n_prefilled = sum(v is not None for v in filled_obs[TARGET_WELL]) - n_obs_target
    print(f"  Target well: {n_prefilled} small-gap points added (PCHIP)")
    print()

    rows_random = []
    rows_consec = []

    # ═══ Experiment A: Random missing ═══════════════════════════════════════
    total_random = len(RANDOM_PERCENTAGES) * RANDOM_REPEATS
    done = 0
    print(f"Experiment A: Random missing ({total_random} total folds)")
    print("-" * 50)

    for pct in RANDOM_PERCENTAGES:
        for rep in range(RANDOM_REPEATS):
            rng = np.random.default_rng(_rng_seed(BASE_SEED, "random", pct, rep))
            truth = random_holdout(raw_obs[TARGET_WELL], pct, rng)
            if not truth or len(truth) < 1:
                rows_random.append({"pct_missing": pct, "rep": rep, "status": "skip"})
                done += 1
                continue

            # Create modified donor observations
            donor_obs_fold = {k: list(v) for k, v in filled_obs.items()}
            mod_target = list(donor_obs_fold[TARGET_WELL])
            for idx in truth:
                mod_target[idx] = None
            donor_obs_fold[TARGET_WELL] = mod_target

            try:
                res = run_mc_lnn_fold(
                    target_id=TARGET_WELL,
                    mod_target=mod_target,
                    donor_obs_filled=donor_obs_fold,
                    months=months,
                    aux_lk=aux_lk,
                    latlon=latlon,
                    params=params,
                    seed=_rng_seed(BASE_SEED, "mc_lnn_random", pct, rep),
                )
                metrics = metrics_on_truth(res, truth)
                row = {
                    "pct_missing": pct,
                    "rep": rep,
                    "status": "ok" if metrics else "metrics_nan",
                    "n_removed": len(truth),
                    "n_remaining": n_obs_target - len(truth),
                }
                if metrics:
                    row.update(metrics)
                rows_random.append(row)
            except Exception as exc:
                rows_random.append({"pct_missing": pct, "rep": rep, "status": f"error:{exc}", "n_removed": len(truth)})

            done += 1
            if done % 10 == 0 or done == total_random:
                elapsed = time.time() - t0
                print(f"  [{done}/{total_random}] {elapsed:.0f}s elapsed", flush=True)

    # ═══ Experiment B: Consecutive gaps ═══════════════════════════════════
    total_consec = len(CONSECUTIVE_YEARS) * CONSECUTIVE_REPEATS
    done = 0
    t1 = time.time()
    print()
    print(f"Experiment B: Consecutive year gaps ({total_consec} total folds)")
    print("-" * 50)

    for n_years in CONSECUTIVE_YEARS:
        for rep in range(CONSECUTIVE_REPEATS):
            rng = np.random.default_rng(_rng_seed(BASE_SEED, "consec", n_years, rep))
            start_idx, truth = consecutive_holdout(raw_obs[TARGET_WELL], n_years, rng, min_truth=1)
            if start_idx is None or not truth:
                rows_consec.append({"n_years": n_years, "rep": rep, "status": "skip"})
                done += 1
                continue

            donor_obs_fold = {k: list(v) for k, v in filled_obs.items()}
            mod_target = list(donor_obs_fold[TARGET_WELL])
            for idx in truth:
                mod_target[idx] = None
            donor_obs_fold[TARGET_WELL] = mod_target

            try:
                res = run_mc_lnn_fold(
                    target_id=TARGET_WELL,
                    mod_target=mod_target,
                    donor_obs_filled=donor_obs_fold,
                    months=months,
                    aux_lk=aux_lk,
                    latlon=latlon,
                    params=params,
                    seed=_rng_seed(BASE_SEED, "mc_lnn_consec", n_years, rep),
                )
                metrics = metrics_on_truth(res, truth)
                window_start = months[start_idx].strftime("%Y-%m")
                window_end = months[min(len(months) - 1, start_idx + n_years * 12 - 1)].strftime("%Y-%m")
                row = {
                    "n_years": n_years,
                    "rep": rep,
                    "status": "ok" if metrics else "metrics_nan",
                    "n_removed": len(truth),
                    "window_start": window_start,
                    "window_end": window_end,
                }
                if metrics:
                    row.update(metrics)
                rows_consec.append(row)
            except Exception as exc:
                rows_consec.append({"n_years": n_years, "rep": rep, "status": f"error:{exc}", "n_removed": len(truth)})

            done += 1
            if done % 5 == 0 or done == total_consec:
                elapsed = time.time() - t1
                print(f"  [{done}/{total_consec}] {elapsed:.0f}s elapsed", flush=True)

    # ═══ Save results ═══════════════════════════════════════════════════════
    total_time = time.time() - t0
    print()
    print("=" * 70)
    print(f"Total elapsed: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 70)

    df_random = pd.DataFrame(rows_random)
    df_consec = pd.DataFrame(rows_consec)

    df_random.to_csv(os.path.join(OUTPUT_DIR, "cv_robustness_random_detailed.csv"), index=False)
    df_consec.to_csv(os.path.join(OUTPUT_DIR, "cv_robustness_consecutive_detailed.csv"), index=False)

    # ─── Summary tables ───────────────────────────────────────────────────
    metric_cols = ["kge", "r2", "rmse", "mae", "nse", "r"]
    lines = []
    lines.append("=" * 70)
    lines.append("MC+LNN ROBUSTNESS CROSS-VALIDATION RESULTS")
    lines.append("=" * 70)
    lines.append(f"Target well: {TARGET_WELL}")
    lines.append(f"Observations: {n_obs_target}/{len(months)} months (2000-2023)")
    lines.append(f"Total time: {total_time:.1f}s")
    lines.append("")

    # Random missing summary
    ok_random = df_random[df_random["status"] == "ok"].copy()
    lines.append("EXPERIMENT A: RANDOM MISSING DATA")
    lines.append("-" * 70)
    if len(ok_random):
        avail = [c for c in metric_cols if c in ok_random.columns]
        summary_r = ok_random.groupby("pct_missing")[avail].agg(["mean", "std", "median"])
        summary_r.columns = ["_".join(c) for c in summary_r.columns]
        count_r = ok_random.groupby("pct_missing").size()

        lines.append(f"{'%Miss':>6} {'N':>4} | {'KGE_mean':>9} {'KGE_std':>8} {'KGE_med':>8} | {'R²_mean':>8} {'R²_std':>7} | {'RMSE_mean':>10} {'RMSE_std':>9} | {'MAE_mean':>9} {'MAE_std':>8} | {'NSE_mean':>9} {'NSE_std':>8}")
        lines.append("-" * 130)
        for pct in RANDOM_PERCENTAGES:
            if pct not in count_r.index:
                continue
            n = count_r[pct]
            row_data = summary_r.loc[pct]
            kge_m = row_data.get("kge_mean", float("nan"))
            kge_s = row_data.get("kge_std", float("nan"))
            kge_med = row_data.get("kge_median", float("nan"))
            r2_m = row_data.get("r2_mean", float("nan"))
            r2_s = row_data.get("r2_std", float("nan"))
            rmse_m = row_data.get("rmse_mean", float("nan"))
            rmse_s = row_data.get("rmse_std", float("nan"))
            mae_m = row_data.get("mae_mean", float("nan"))
            mae_s = row_data.get("mae_std", float("nan"))
            nse_m = row_data.get("nse_mean", float("nan"))
            nse_s = row_data.get("nse_std", float("nan"))
            lines.append(f"{pct:>5}% {n:>4} | {kge_m:>9.4f} {kge_s:>8.4f} {kge_med:>8.4f} | {r2_m:>8.4f} {r2_s:>7.4f} | {rmse_m:>10.3f} {rmse_s:>9.3f} | {mae_m:>9.3f} {mae_s:>8.3f} | {nse_m:>9.4f} {nse_s:>8.4f}")
    else:
        lines.append("No successful random folds.")

    lines.append("")
    lines.append("EXPERIMENT B: CONSECUTIVE YEAR GAPS")
    lines.append("-" * 70)
    ok_consec = df_consec[df_consec["status"] == "ok"].copy()
    if len(ok_consec):
        avail = [c for c in metric_cols if c in ok_consec.columns]
        summary_c = ok_consec.groupby("n_years")[avail].agg(["mean", "std", "median"])
        summary_c.columns = ["_".join(c) for c in summary_c.columns]
        count_c = ok_consec.groupby("n_years").size()

        lines.append(f"{'Years':>6} {'N':>4} | {'KGE_mean':>9} {'KGE_std':>8} {'KGE_med':>8} | {'R²_mean':>8} {'R²_std':>7} | {'RMSE_mean':>10} {'RMSE_std':>9} | {'MAE_mean':>9} {'MAE_std':>8} | {'NSE_mean':>9} {'NSE_std':>8}")
        lines.append("-" * 130)
        for ny in CONSECUTIVE_YEARS:
            if ny not in count_c.index:
                continue
            n = count_c[ny]
            row_data = summary_c.loc[ny]
            kge_m = row_data.get("kge_mean", float("nan"))
            kge_s = row_data.get("kge_std", float("nan"))
            kge_med = row_data.get("kge_median", float("nan"))
            r2_m = row_data.get("r2_mean", float("nan"))
            r2_s = row_data.get("r2_std", float("nan"))
            rmse_m = row_data.get("rmse_mean", float("nan"))
            rmse_s = row_data.get("rmse_std", float("nan"))
            mae_m = row_data.get("mae_mean", float("nan"))
            mae_s = row_data.get("mae_std", float("nan"))
            nse_m = row_data.get("nse_mean", float("nan"))
            nse_s = row_data.get("nse_std", float("nan"))
            lines.append(f"{ny:>5}y {n:>4} | {kge_m:>9.4f} {kge_s:>8.4f} {kge_med:>8.4f} | {r2_m:>8.4f} {r2_s:>7.4f} | {rmse_m:>10.3f} {rmse_s:>9.3f} | {mae_m:>9.3f} {mae_s:>8.3f} | {nse_m:>9.4f} {nse_s:>8.4f}")
    else:
        lines.append("No successful consecutive folds.")

    summary_text = "\n".join(lines)
    print()
    print(summary_text)

    with open(os.path.join(OUTPUT_DIR, "cv_robustness_summary.txt"), "w") as f:
        f.write(summary_text + "\n")

    # Save metadata
    meta = {
        "target_well": TARGET_WELL,
        "target_obs_count": n_obs_target,
        "total_months": len(months),
        "date_range": f"{DATE_START} to {DATE_END}",
        "donor_pool_size": len(donor_pool_wells),
        "prefilled_points": n_prefilled,
        "random_percentages": RANDOM_PERCENTAGES,
        "random_repeats": RANDOM_REPEATS,
        "consecutive_years": CONSECUTIVE_YEARS,
        "consecutive_repeats": CONSECUTIVE_REPEATS,
        "random_successful": int(len(ok_random)),
        "consecutive_successful": int(len(ok_consec)),
        "elapsed_seconds": total_time,
        "params": asdict(params),
    }
    with open(os.path.join(OUTPUT_DIR, "cv_robustness_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print()
    print("Output files:")
    print(f"  cv_robustness_random_detailed.csv     ({len(df_random)} rows)")
    print(f"  cv_robustness_consecutive_detailed.csv ({len(df_consec)} rows)")
    print(f"  cv_robustness_summary.txt")
    print(f"  cv_robustness_metadata.json")


if __name__ == "__main__":
    main()
