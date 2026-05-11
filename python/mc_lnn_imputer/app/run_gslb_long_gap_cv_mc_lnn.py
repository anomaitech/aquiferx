#!/usr/bin/env python3
"""
GSLB long-gap CV using:
  1) small-gap fill for all eligible wells
  2) donor-initialized matrix completion
  3) LNN CFC refinement for long gaps

This is the stronger long-gap path intended to raise KGE relative to the
baseline aux-only long-gap runner.
"""

from __future__ import annotations

import argparse
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

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_IMPUTER_ROOT = os.path.abspath(os.path.join(_APP_DIR, ".."))
sys.path.insert(0, _APP_DIR)

from backend.lnn.lnn_core_aux_placeholder_cfc import optimize_lnn_params, run_lnn_simulation
from backend.lnn.math_utils import calculate_kge, calculate_pearson_correlation, calculate_r2
from backend.lnn.types import DataPoint, SimulationParams
from scipy.interpolate import PchipInterpolator

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_CSV = os.path.join(_IMPUTER_ROOT, "datas", "measurements_till_2023_to_lnn_imputation.csv")
AUX_CSV = os.path.join(_IMPUTER_ROOT, "datas", "lnn_imputation_gslb_gldas_df_excercise.csv")

DATE_START = "2000-01-01"
DATE_END = "2023-12-21"
MIN_POINTS_PER_INSTANCE = 10
N_TOP_INSTANCES = 10
CONSECUTIVE_YEARS = [1, 2, 3, 4, 5]
AUX_COLS = ["soilw", "soilw_yr01", "soilw_yr03", "soilw_yr05", "soilw_yr10"]

BASE_SEED = 42
DEFAULT_REPEATS = 5
DEFAULT_MIN_TRUTH_IN_WINDOW = 1
DONOR_POOL_SIZE = 50
MAX_DONORS = 15
MIN_DONOR_CORR = 0.3


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
        small_gap_max_iterations=5,
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


def nse(obs: Sequence[float], pred: Sequence[float]) -> float:
    o = np.asarray(obs, dtype=float)
    p = np.asarray(pred, dtype=float)
    if len(o) < 2:
        return float("nan")
    denom = np.sum((o - o.mean()) ** 2)
    if denom < 1e-12:
        return float("nan")
    return float(1.0 - np.sum((o - p) ** 2) / denom)


def compute_metrics(obs: Sequence[float], pred: Sequence[float]) -> Optional[Dict[str, float]]:
    o = np.asarray(obs, dtype=float)
    p = np.asarray(pred, dtype=float)
    if o.size < 1 or p.shape != o.shape:
        return None
    err = p - o
    mse = float(np.mean(err**2))
    out: Dict[str, float] = {
        "n_scored": float(o.size),
        "mae": float(np.mean(np.abs(err))),
        "me": float(np.mean(err)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "max_ae": float(np.max(np.abs(err))),
        "pbias": float(100.0 * np.sum(err) / (np.sum(np.abs(o)) + 1e-12)),
    }
    if o.size < 2:
        out.update({"rrmse": float("nan"), "ia": float("nan"), "r": float("nan"), "r2": float("nan"), "kge": float("nan"), "nse": float("nan")})
        return out
    std_o = float(np.std(o, ddof=1))
    mean_o = float(np.mean(o))
    den_ia = float(np.sum((np.abs(p - mean_o) + np.abs(o - mean_o)) ** 2))
    r = float(calculate_pearson_correlation(list(o), list(p)))
    kge = float(calculate_kge(list(o), list(p)))
    out.update(
        {
            "rrmse": float(out["rmse"] / (std_o + 1e-12)),
            "ia": float(1.0 - np.sum(err**2) / (den_ia + 1e-12)) if den_ia > 1e-12 else float("nan"),
            "r": r if np.isfinite(r) else float("nan"),
            "r2": float(calculate_r2(list(o), list(p))),
            "kge": kge if np.isfinite(kge) else float("nan"),
            "nse": float(nse(o, p)),
        }
    )
    return out


def summarize_by_year(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["n_scored", "mae", "me", "mse", "rmse", "rrmse", "pbias", "ia", "max_ae", "r", "r2", "kge", "nse"]
    metric_cols = [c for c in metric_cols if c in df.columns]
    grouped = df.groupby("n_years")[metric_cols].agg(["mean", "std", "median", "count"])
    grouped.columns = ["_".join(c).strip("_") for c in grouped.columns.to_flat_index()]
    return grouped.reset_index()


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex, List[str], List[str], Dict[str, Tuple[float, float]]]:
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
    counts = main_df.groupby("Well_ID")["month"].nunique()
    eligible = counts[counts >= MIN_POINTS_PER_INSTANCE]
    stats = []
    for wid in eligible.index.tolist():
        seen = set(main_df.loc[main_df["Well_ID"] == wid, "month"].unique())
        cur = 0
        mx = 0
        for p in pd.period_range("2000-01", "2023-12", freq="M"):
            if p in seen:
                cur += 1
                mx = max(mx, cur)
            else:
                cur = 0
        stats.append((wid, mx, int(eligible[wid])))
    stats.sort(key=lambda x: (-x[1], -x[2], x[0]))
    top_wells = [wid for wid, _, _ in stats[:N_TOP_INSTANCES]]
    donor_pool_wells = [wid for wid, _, _ in stats[:DONOR_POOL_SIZE]]

    latlon: Dict[str, Tuple[float, float]] = {}
    for wid, _, _ in stats:
        r0 = main_df[main_df["Well_ID"] == wid].iloc[0]
        latlon[wid] = (float(r0["lat_dec"]), float(r0["long_dec"]))
    return main_df, aux_df, months, top_wells, donor_pool_wells, latlon


def aux_lookup(aux_df: pd.DataFrame) -> Dict[Tuple[int, int], List[float]]:
    out: Dict[Tuple[int, int], List[float]] = {}
    for dt, row in aux_df.iterrows():
        out[(dt.year, dt.month)] = [float(row[c]) if c in row.index and pd.notna(row[c]) else 0.0 for c in AUX_COLS]
    return out


def monthly_series(main_df: pd.DataFrame, well_id: str, months: pd.DatetimeIndex) -> List[Optional[float]]:
    monthly = main_df.loc[main_df["Well_ID"] == well_id].groupby("month")["WTE"].mean()
    out: List[Optional[float]] = []
    for m in months:
        v = monthly.get(m.to_period("M"))
        out.append(float(v) if pd.notna(v) else None)
    return out


def build_aux_timeline(
    obs_list: List[Optional[float]],
    well_id: str,
    months: pd.DatetimeIndex,
    aux_lk: Dict[Tuple[int, int], List[float]],
    lat: float,
    lon: float,
    add_seasonal: bool = False,
) -> List[DataPoint]:
    tl: List[DataPoint] = []
    for mi, m in enumerate(months):
        aux = list(aux_lk.get((m.year, m.month), [0.0] * len(AUX_COLS)))
        if add_seasonal:
            aux.extend([np.sin(2 * np.pi * mi / 12.0), np.cos(2 * np.pi * mi / 12.0)])
        tl.append(
            DataPoint(
                time=float(mi),
                observed=obs_list[mi],
                instance_id=well_id,
                auxiliaries=aux,
                date_label=m.strftime("%Y-%m"),
                latitude=lat,
                longitude=lon,
            )
        )
    return tl


def impute_timeline(timeline: List[DataPoint], params: SimulationParams, seed: int) -> List[DataPoint]:
    with redirect_stdout(io.StringIO()):
        best = optimize_lnn_params(timeline, params, mode="projection", rng=np.random.default_rng(seed))
        best_result = None
        best_kge = float("-inf")
        for it in range(5):
            res = run_lnn_simulation(timeline, best, rng=np.random.default_rng(seed + it))
            obs = [d.observed for d in timeline if d.observed is not None]
            pred = [res[i].imputed for i, d in enumerate(timeline) if d.observed is not None and res[i].imputed is not None]
            if obs and pred and len(obs) == len(pred):
                k = float(calculate_kge(obs, pred))
                if np.isfinite(k) and k > best_kge:
                    best_kge = k
                    best_result = res
        return best_result if best_result is not None else run_lnn_simulation(timeline, best, rng=np.random.default_rng(seed))


def pchip_fill_series(series: List[Optional[float]], max_gap: int = 24) -> List[Optional[float]]:
    """Fill small gaps (≤ max_gap months) using PCHIP interpolation.
    Matches browser mcLnnPureBrowser.ts pchipFill()."""
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

    pchip = PchipInterpolator(obs_idx, obs_vals)
    for g_start, g_end in gaps:
        for t in range(g_start, g_end + 1):
            val = float(pchip(t))
            if np.isfinite(val):
                out[t] = val
    return out


def prefill_small_gaps_all(
    raw_obs: Dict[str, List[Optional[float]]],
    gap_size: int = 24,
    pad_size: int = 6,
) -> Dict[str, List[Optional[float]]]:
    """Phase 1: PCHIP small-gap fill + large-gap blanking for all wells.
    Matches browser mcLnnPureBrowser.ts Phase 1."""
    filled: Dict[str, List[Optional[float]]] = {}
    n_total = 0
    for wid, raw in raw_obs.items():
        # PCHIP fill all gaps first
        out = pchip_fill_series(raw, max_gap=gap_size * 2)

        # Blank interior of large gaps (keep pad_size months at edges)
        obs_idx = [i for i, v in enumerate(raw) if v is not None]
        for g in range(len(obs_idx) - 1):
            gap_len = obs_idx[g + 1] - obs_idx[g] - 1
            if gap_len > gap_size:
                blank_start = obs_idx[g] + pad_size + 1
                blank_end = obs_idx[g + 1] - pad_size - 1
                for t in range(max(blank_start, 0), min(blank_end, len(out) - 1) + 1):
                    out[t] = None

        added = sum(1 for a, b in zip(out, raw) if a is not None and b is None)
        n_total += added
        filled[wid] = out

    print(f"  [Phase 1] PCHIP small gaps filled: {n_total} points across "
          f"{len(filled)} wells", flush=True)
    return filled


def relaxed_holdout_indices(
    raw: List[Optional[float]],
    n_years: int,
    rng: np.random.Generator,
    min_truth_in_window: int,
) -> Tuple[Optional[int], Dict[int, float]]:
    n_months = n_years * 12
    if len(raw) < n_months:
        return None, {}
    valid_starts: List[int] = []
    for s in range(0, len(raw) - n_months + 1):
        n_obs = sum(1 for i in range(s, s + n_months) if raw[i] is not None)
        if n_obs >= min_truth_in_window:
            valid_starts.append(s)
    if not valid_starts:
        return None, {}
    start = int(rng.choice(valid_starts))
    truth = {i: float(raw[i]) for i in range(start, start + n_months) if raw[i] is not None}
    return start, truth


def _archi_regression(
    target_obs: Dict[int, float],
    donor_obs: Dict[str, List[Optional[float]]],
    target_id: str,
) -> Tuple[Dict[int, float], List[Dict[str, Any]]]:
    if len(target_obs) < 5:
        return {}, []
    gap_times = sorted(set(range(288)) - set(target_obs.keys()))
    donors: List[Dict[str, Any]] = []
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

    preds_by_donor: List[Dict[int, float]] = []
    weights: List[float] = []
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

    combined: Dict[int, float] = {}
    for t in gap_times:
        ws = 0.0
        wc = 0.0
        for i, preds in enumerate(preds_by_donor):
            if t in preds:
                ws += preds[t] * weights[i]
                wc += weights[i]
        if wc > 0:
            combined[t] = ws / wc
    return combined, donors


def _matrix_completion_archi_init(
    target_series: List[Optional[float]],
    target_aux: List[List[float]],
    donor_obs: Dict[str, List[Optional[float]]],
    donors: List[Dict[str, Any]],
    archi_preds: Dict[int, float],
) -> Dict[int, float]:
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
    if len(target_obs_idx) >= 3:
        ov = M_raw[0, target_obs_idx]
        mv = pred[target_obs_idx]
        om, os = np.mean(ov), max(np.std(ov), 1e-10)
        mm, ms = np.mean(mv), max(np.std(mv), 1e-10)
        pred = (pred - mm) / ms * os + om
    return {t: float(pred[t]) for t in range(n_times)}


def run_mc_lnn_fold(
    target_id: str,
    mod_target: List[Optional[float]],
    donor_obs_filled: Dict[str, List[Optional[float]]],
    months: pd.DatetimeIndex,
    aux_lk: Dict[Tuple[int, int], List[float]],
    latlon: Dict[str, Tuple[float, float]],
    params: SimulationParams,
    seed: int,
    mc_only_placeholder: bool = False,
) -> List[DataPoint]:
    lat, lon = latlon[target_id]
    target_obs = {i: v for i, v in enumerate(mod_target) if v is not None}
    archi_preds, donors = _archi_regression(target_obs, donor_obs_filled, target_id)
    target_aux = [list(aux_lk.get((m.year, m.month), [0.0] * len(AUX_COLS))) for m in months]
    if not donors:
        tl = build_aux_timeline(mod_target, target_id, months, aux_lk, lat, lon, add_seasonal=True)
        return impute_timeline(tl, params, seed)

    mc_preds = _matrix_completion_archi_init(mod_target, target_aux, donor_obs_filled, donors, archi_preds)

    # Build timeline with REAL observations only
    tl = build_aux_timeline(mod_target, target_id, months, aux_lk, lat, lon, add_seasonal=True)

    # Inject MC predictions as placeholders (reservoir input, NOT training targets)
    for i in range(len(tl)):
        if mod_target[i] is None and i in mc_preds and np.isfinite(mc_preds[i]):
            tl[i] = DataPoint(
                time=tl[i].time,
                observed=None,  # readout won't train on this
                imputed=float(mc_preds[i]),  # MC value for reservoir input
                instance_id=tl[i].instance_id,
                auxiliaries=tl[i].auxiliaries,
                date_label=tl[i].date_label,
                latitude=tl[i].latitude,
                longitude=tl[i].longitude,
            )

    return impute_timeline(tl, params, seed)


def metrics_on_truth(result: List[DataPoint], truth: Dict[int, float]) -> Optional[Dict[str, float]]:
    obs, pred = [], []
    for idx, tv in truth.items():
        if idx < len(result) and result[idx].imputed is not None:
            obs.append(tv)
            pred.append(float(result[idx].imputed))
    return compute_metrics(obs, pred)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--seed", type=int, default=BASE_SEED)
    ap.add_argument("--max-wells", type=int, default=None)
    ap.add_argument("--min-truth-in-window", type=int, default=DEFAULT_MIN_TRUTH_IN_WINDOW)
    ap.add_argument("--mc-only-placeholder", action="store_true")
    ap.add_argument(
        "--well-ids",
        type=str,
        default="",
        help="Comma-separated explicit Well_ID list to use as CV targets.",
    )
    ap.add_argument(
        "--output-tag",
        type=str,
        default="",
        help="Optional suffix tag for output filenames, e.g. screenshot10.",
    )
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    params = build_params()
    main_df, aux_df, months, top_wells, donor_pool_wells, latlon = load_data()
    explicit_wells = [w.strip() for w in args.well_ids.split(",") if w.strip()]
    if explicit_wells:
        available = set(donor_pool_wells)
        top_wells = [w for w in explicit_wells if w in available]
        missing = [w for w in explicit_wells if w not in available]
        if missing:
            print(f"Skipping unavailable wells: {missing}", flush=True)
        if not top_wells:
            raise ValueError("No requested well IDs were available after filtering.")
    elif args.max_wells is not None:
        top_wells = top_wells[: max(1, args.max_wells)]
    aux_lk = aux_lookup(aux_df)

    raw_obs: Dict[str, List[Optional[float]]] = {wid: monthly_series(main_df, wid, months) for wid in donor_pool_wells}
    t0 = time.time()
    filled_obs = prefill_small_gaps_all(raw_obs, gap_size=24, pad_size=6)

    rows: List[Dict[str, Any]] = []
    total = len(top_wells) * len(CONSECUTIVE_YEARS) * args.repeats
    done = 0

    for wid in top_wells:
        base = raw_obs[wid]
        n_prefilled = sum(v is not None for v in filled_obs[wid]) - sum(v is not None for v in base)
        for n_years in CONSECUTIVE_YEARS:
            for rep in range(args.repeats):
                rng = np.random.default_rng(_rng_seed(args.seed, "cv", wid, n_years, rep))
                start_idx, truth = relaxed_holdout_indices(base, n_years, rng, args.min_truth_in_window)
                if start_idx is None or not truth:
                    rows.append({"Well_ID": wid, "n_years": n_years, "rep": rep, "status": "skip_no_valid_window", "prefill_added_points": n_prefilled})
                    done += 1
                    continue

                donor_obs_fold = {k: list(v) for k, v in filled_obs.items()}
                mod_target = list(donor_obs_fold[wid])
                for idx in truth:
                    mod_target[idx] = None
                donor_obs_fold[wid] = mod_target

                try:
                    res = run_mc_lnn_fold(
                        target_id=wid,
                        mod_target=mod_target,
                        donor_obs_filled=donor_obs_fold,
                        months=months,
                        aux_lk=aux_lk,
                        latlon=latlon,
                        params=params,
                        seed=_rng_seed(args.seed, "mc_lnn", wid, n_years, rep),
                        mc_only_placeholder=args.mc_only_placeholder,
                    )
                    metrics = metrics_on_truth(res, truth)
                    row = {
                        "Well_ID": wid,
                        "n_years": n_years,
                        "rep": rep,
                        "status": "ok" if metrics else "metrics_nan",
                        "prefill_added_points": n_prefilled,
                        "window_start": months[start_idx].strftime("%Y-%m"),
                        "window_end": months[min(len(months) - 1, start_idx + n_years * 12 - 1)].strftime("%Y-%m"),
                        "n_removed": len(truth),
                        "method": "mc_lnn",
                    }
                    if metrics:
                        row.update(metrics)
                    rows.append(row)
                except Exception as exc:
                    rows.append({"Well_ID": wid, "n_years": n_years, "rep": rep, "status": f"error:{exc}", "prefill_added_points": n_prefilled, "method": "mc_lnn"})
                done += 1
                if done % 25 == 0 or done == total:
                    print(f"[MC+LNN CV] {done}/{total} folds complete in {time.time() - t0:.0f}s", flush=True)

    detailed = pd.DataFrame(rows)
    ok_df = detailed[detailed["status"] == "ok"].copy()
    summary = summarize_by_year(ok_df) if len(ok_df) else pd.DataFrame()

    suffix = "_mc_only_placeholder" if args.mc_only_placeholder else "_mc_lnn"
    if args.output_tag:
        safe_tag = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in args.output_tag)
        suffix = f"{suffix}_{safe_tag}"
    detailed_name = f"consecutive_long_gap_cv{suffix}_detailed.csv"
    summary_name = f"consecutive_long_gap_cv{suffix}_summary.csv"
    per_well_name = f"consecutive_long_gap_cv{suffix}_by_well.csv"
    meta_name = f"run_metadata{suffix}.json"
    summary_txt_name = f"SUMMARY{suffix}.txt"

    detailed.to_csv(os.path.join(OUTPUT_DIR, detailed_name), index=False)
    summary.to_csv(os.path.join(OUTPUT_DIR, summary_name), index=False)

    per_well = pd.DataFrame()
    if len(ok_df):
        metric_cols = ["n_scored", "mae", "me", "mse", "rmse", "rrmse", "pbias", "ia", "max_ae", "r", "r2", "kge", "nse"]
        per_well = ok_df.groupby(["Well_ID", "n_years"])[metric_cols].agg(["mean", "std", "count"])
        per_well.columns = ["_".join(c).strip("_") for c in per_well.columns.to_flat_index()]
        per_well = per_well.reset_index()
    per_well.to_csv(os.path.join(OUTPUT_DIR, per_well_name), index=False)

    run_meta = {
        "target_csv": TARGET_CSV,
        "aux_csv": AUX_CSV,
        "date_start": DATE_START,
        "date_end": DATE_END,
        "top_wells": top_wells,
        "eligible_wells": len(donor_pool_wells),
        "donor_pool_size": DONOR_POOL_SIZE,
        "consecutive_years": CONSECUTIVE_YEARS,
        "consecutive_repeats": args.repeats,
        "min_truth_in_window": args.min_truth_in_window,
        "requested_mode": "small-gap-fill-all + donor-initialized mc + lnn refinement",
        "mc_only_placeholder": args.mc_only_placeholder,
        "params": asdict(params),
        "successful_folds_by_year": ok_df.groupby("n_years").size().astype(int).to_dict() if len(ok_df) else {},
        "elapsed_seconds": time.time() - t0,
    }
    with open(os.path.join(OUTPUT_DIR, meta_name), "w") as f:
        json.dump(run_meta, f, indent=2)

    lines = [
        "GSLB long-gap CV summary (MC as only placeholder + CFC refinement)" if args.mc_only_placeholder else "GSLB long-gap CV summary (MC + LNN)",
        f"Elapsed seconds: {run_meta['elapsed_seconds']:.1f}",
        f"Target CSV: {TARGET_CSV}",
        f"Aux CSV: {AUX_CSV}",
        f"Date filter: {DATE_START} .. {DATE_END}",
        f"Top wells: {json.dumps(top_wells)}",
        f"Eligible donor wells used: {len(donor_pool_wells)}",
        f"Consecutive repeats: {args.repeats}",
        f"Relaxed minimum truth in window: {args.min_truth_in_window}",
        f"Successful folds by year: {json.dumps(run_meta['successful_folds_by_year'])}",
        "",
        "Outputs:",
        f"  {detailed_name}",
        f"  {summary_name}",
        f"  {per_well_name}",
        f"  {meta_name}",
    ]
    if len(summary):
        lines.extend(["", summary.to_string(index=False)])
    with open(os.path.join(OUTPUT_DIR, summary_txt_name), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
