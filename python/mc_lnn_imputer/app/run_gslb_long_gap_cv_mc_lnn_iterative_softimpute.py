#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fancyimpute import SoftImpute

OUT_DIR = Path(__file__).resolve().parent
DEFAULT_WELLS = [
    "415703112514501",
    "414236112101201",
    "414411112543701",
    "411544111461001",
    "411348112013601",
    "401818112014501",
    "402333111513401",
    "401312112442301",
    "403916111575901",
]


def load_base():
    runner_path = OUT_DIR / "run_gslb_long_gap_cv_mc_lnn.py"
    spec = importlib.util.spec_from_file_location("mc_lnn_base", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _matrix_completion_archi_init_iter(
    base,
    target_series: List[Optional[float]],
    target_aux: List[List[float]],
    donor_obs: Dict[str, List[Optional[float]]],
    donors: List[Dict[str, Any]],
    archi_preds: Dict[int, float],
    target_init_preds: Optional[Dict[int, float]] = None,
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

    best_x = None
    best_k = 5
    best_err = float("inf")
    for k_try in [3, 5, 8, 10, 12]:
        if k_try >= min(M.shape):
            continue
        solver = SoftImpute(
            shrinkage_value=None,
            convergence_threshold=1e-6,
            max_iters=50,
            max_rank=k_try,
            init_fill_method="zero",
            verbose=False,
        )
        Xt = solver.fit_transform(M)
        if len(target_obs_idx) > 3:
            err = float(np.mean((Xt[0, target_obs_idx] - M[0, target_obs_idx]) ** 2))
            if err < best_err:
                best_err = err
                best_k = k_try
                best_x = Xt

    if best_x is None:
        solver = SoftImpute(
            shrinkage_value=None,
            convergence_threshold=1e-6,
            max_iters=100,
            max_rank=best_k,
            init_fill_method="zero",
            verbose=False,
        )
        X = solver.fit_transform(M)
    else:
        solver = SoftImpute(
            shrinkage_value=None,
            convergence_threshold=1e-6,
            max_iters=100,
            max_rank=best_k,
            init_fill_method="zero",
            verbose=False,
        )
        X = solver.fit_transform(M)

    for di, donor in enumerate(donors, start=1):
        if abs(donor["r"]) > 1e-10:
            X[di, :] /= abs(donor["r"])

    pred = X[0, :] * rstds[0] + rmeans[0]
    if target_init_preds:
        for t, init_v in target_init_preds.items():
            if 0 <= t < n_times and np.isfinite(init_v) and np.isfinite(pred[t]) and target_series[t] is None:
                pred[t] = 0.35 * float(init_v) + 0.65 * float(pred[t])
    if len(target_obs_idx) >= 3:
        ov = M_raw[0, target_obs_idx]
        mv = pred[target_obs_idx]
        om, os = np.mean(ov), max(np.std(ov), 1e-10)
        mm, ms = np.mean(mv), max(np.std(mv), 1e-10)
        pred = (pred - mm) / ms * os + om
    return {t: float(pred[t]) for t in range(n_times)}


def run_mc_lnn_single_pass(
    base,
    target_id: str,
    mod_target: List[Optional[float]],
    donor_obs_filled: Dict[str, List[Optional[float]]],
    months: pd.DatetimeIndex,
    aux_lk: Dict[Tuple[int, int], List[float]],
    latlon: Dict[str, Tuple[float, float]],
    params,
    seed: int,
    target_init_preds: Optional[Dict[int, float]] = None,
):
    lat, lon = latlon[target_id]
    target_obs = {i: v for i, v in enumerate(mod_target) if v is not None}
    archi_preds, donors = base._archi_regression(target_obs, donor_obs_filled, target_id)
    target_aux = [list(aux_lk.get((m.year, m.month), [0.0] * len(base.AUX_COLS))) for m in months]
    if not donors:
        tl = base.build_aux_timeline(mod_target, target_id, months, aux_lk, lat, lon, add_seasonal=True)
        return base.impute_timeline(tl, params, seed)

    mc_preds = _matrix_completion_archi_init_iter(
        base,
        mod_target,
        target_aux,
        donor_obs_filled,
        donors,
        archi_preds,
        target_init_preds=target_init_preds,
    )
    enriched: List[Optional[float]] = []
    for i, v in enumerate(mod_target):
        if v is not None:
            enriched.append(v)
        elif i in mc_preds and np.isfinite(mc_preds[i]):
            enriched.append(float(mc_preds[i]))
        else:
            enriched.append(None)
    tl = base.build_aux_timeline(enriched, target_id, months, aux_lk, lat, lon, add_seasonal=True)
    return base.impute_timeline(tl, params, seed)


def select_support_truth(
    raw_target: List[Optional[float]],
    truth: Dict[int, float],
    rng: np.random.Generator,
    support_frac: float,
    min_support: int,
    max_support: int,
) -> Dict[int, float]:
    candidates = [i for i, v in enumerate(raw_target) if v is not None and i not in truth]
    if len(candidates) < min_support:
        return {}
    n_pick = int(round(len(candidates) * support_frac))
    n_pick = max(min_support, min(max_support, n_pick, len(candidates)))
    picked = rng.choice(candidates, size=n_pick, replace=False)
    return {int(i): float(raw_target[int(i)]) for i in picked}


def score_tuple(metrics: Optional[Dict[str, float]]) -> Tuple[float, float]:
    if not metrics:
        return (float("-inf"), float("-inf"))
    kge = float(metrics.get("kge", float("-inf")))
    rmse = float(metrics.get("rmse", float("inf")))
    if not np.isfinite(kge):
        kge = float("-inf")
    if not np.isfinite(rmse):
        rmse = float("inf")
    return (kge, -rmse)


def blend_predictions(
    prev_init: Optional[Dict[int, float]],
    result,
    missing_idx: List[int],
    prev_weight: float,
) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for i in missing_idx:
        new_v = result[i].imputed if i < len(result) and result[i].imputed is not None else None
        if new_v is None or not np.isfinite(float(new_v)):
            continue
        if prev_init and i in prev_init and np.isfinite(prev_init[i]):
            out[i] = float(prev_weight * prev_init[i] + (1.0 - prev_weight) * float(new_v))
        else:
            out[i] = float(new_v)
    return out


def run_iterative_fold(
    base,
    target_id: str,
    raw_target: List[Optional[float]],
    mod_target: List[Optional[float]],
    donor_obs_filled: Dict[str, List[Optional[float]]],
    months: pd.DatetimeIndex,
    aux_lk: Dict[Tuple[int, int], List[float]],
    latlon: Dict[str, Tuple[float, float]],
    params,
    seed: int,
    outer_iterations: int,
    feedback_prev_weight: float,
    support_frac: float,
    min_support: int,
    max_support: int,
):
    truth_missing_idx = [i for i, v in enumerate(mod_target) if v is None]
    support_rng = np.random.default_rng(base._rng_seed(seed, "support"))
    support_truth = select_support_truth(raw_target, {i: raw_target[i] for i in truth_missing_idx if raw_target[i] is not None}, support_rng, support_frac, min_support, max_support)
    if not support_truth:
        return base.run_mc_lnn_fold(target_id, mod_target, donor_obs_filled, months, aux_lk, latlon, params, seed), {
            "outer_iterations_used": 0,
            "support_points": 0,
            "best_support_kge": float("nan"),
            "best_support_rmse": float("nan"),
        }

    work_target = list(mod_target)
    for i in support_truth:
        work_target[i] = None
    missing_for_loop = sorted(set(truth_missing_idx) | set(support_truth.keys()))

    best_support_metrics: Optional[Dict[str, float]] = None
    best_init: Optional[Dict[int, float]] = None
    accepted_iters = 0

    for outer in range(outer_iterations):
        donor_fold = {k: list(v) for k, v in donor_obs_filled.items()}
        donor_fold[target_id] = list(work_target)
        result = run_mc_lnn_single_pass(
            base,
            target_id,
            work_target,
            donor_fold,
            months,
            aux_lk,
            latlon,
            params,
            seed=base._rng_seed(seed, "outer", outer),
            target_init_preds=best_init,
        )
        support_metrics = base.metrics_on_truth(result, support_truth)
        candidate_init = blend_predictions(best_init, result, missing_for_loop, feedback_prev_weight)
        if score_tuple(support_metrics) > score_tuple(best_support_metrics):
            best_support_metrics = support_metrics
            best_init = candidate_init
            accepted_iters = outer + 1
        else:
            break

    if best_init:
        final_init = {i: v for i, v in best_init.items() if i in truth_missing_idx}
        final_donor_fold = {k: list(v) for k, v in donor_obs_filled.items()}
        final_donor_fold[target_id] = list(mod_target)
        final_result = run_mc_lnn_single_pass(
            base,
            target_id,
            mod_target,
            final_donor_fold,
            months,
            aux_lk,
            latlon,
            params,
            seed=base._rng_seed(seed, "final"),
            target_init_preds=final_init,
        )
    else:
        final_result = base.run_mc_lnn_fold(target_id, mod_target, donor_obs_filled, months, aux_lk, latlon, params, seed)

    meta = {
        "outer_iterations_used": accepted_iters,
        "support_points": len(support_truth),
        "best_support_kge": float(best_support_metrics["kge"]) if best_support_metrics and np.isfinite(best_support_metrics.get("kge", np.nan)) else float("nan"),
        "best_support_rmse": float(best_support_metrics["rmse"]) if best_support_metrics and np.isfinite(best_support_metrics.get("rmse", np.nan)) else float("nan"),
    }
    return final_result, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-truth-in-window", type=int, default=1)
    ap.add_argument("--well-ids", type=str, default=",".join(DEFAULT_WELLS))
    ap.add_argument("--output-tag", type=str, default="screenshot10_iterative_softimpute")
    ap.add_argument("--outer-iterations", type=int, default=2)
    ap.add_argument("--feedback-prev-weight", type=float, default=0.35)
    ap.add_argument("--support-frac", type=float, default=0.12)
    ap.add_argument("--min-support", type=int, default=6)
    ap.add_argument("--max-support", type=int, default=24)
    ap.add_argument("--baseline-summary", type=str, default=str(OUT_DIR / "consecutive_long_gap_cv_mc_lnn_screenshot10_summary.csv"))
    args = ap.parse_args()

    base = load_base()
    os.makedirs(OUT_DIR, exist_ok=True)
    params = base.build_params()
    main_df, aux_df, months, _top_wells, donor_pool_wells, latlon = base.load_data()
    requested = [w.strip() for w in args.well_ids.split(",") if w.strip()]
    available = set(donor_pool_wells)
    top_wells = [w for w in requested if w in available]
    missing = [w for w in requested if w not in available]
    if missing:
        print(f"Skipping unavailable wells: {missing}", flush=True)
    if not top_wells:
        raise ValueError("No requested wells were available after filtering.")

    aux_lk = base.aux_lookup(aux_df)
    raw_obs = {wid: base.monthly_series(main_df, wid, months) for wid in donor_pool_wells}
    t0 = time.time()
    filled_obs = base.prefill_small_gaps_all(raw_obs, months, aux_lk, latlon, params, base._rng_seed(args.seed, "prefill_all_iterative"))

    rows: List[Dict[str, Any]] = []
    total = len(top_wells) * len(base.CONSECUTIVE_YEARS) * args.repeats
    done = 0

    for wid in top_wells:
        raw_target = raw_obs[wid]
        base_target = filled_obs[wid]
        n_prefilled = sum(v is not None for v in base_target) - sum(v is not None for v in raw_target)
        for n_years in base.CONSECUTIVE_YEARS:
            for rep in range(args.repeats):
                rng = np.random.default_rng(base._rng_seed(args.seed, "cv", wid, n_years, rep))
                start_idx, truth = base.relaxed_holdout_indices(raw_target, n_years, rng, args.min_truth_in_window)
                if start_idx is None or not truth:
                    rows.append({
                        "Well_ID": wid,
                        "n_years": n_years,
                        "rep": rep,
                        "status": "skip_no_valid_window",
                        "prefill_added_points": n_prefilled,
                        "method": "mc_lnn_iterative",
                    })
                    done += 1
                    continue

                donor_obs_fold = {k: list(v) for k, v in filled_obs.items()}
                mod_target = list(donor_obs_fold[wid])
                for idx in truth:
                    mod_target[idx] = None
                donor_obs_fold[wid] = mod_target

                try:
                    res, iter_meta = run_iterative_fold(
                        base,
                        wid,
                        raw_target,
                        mod_target,
                        donor_obs_fold,
                        months,
                        aux_lk,
                        latlon,
                        params,
                        seed=base._rng_seed(args.seed, "mc_lnn_iter", wid, n_years, rep),
                        outer_iterations=args.outer_iterations,
                        feedback_prev_weight=args.feedback_prev_weight,
                        support_frac=args.support_frac,
                        min_support=args.min_support,
                        max_support=args.max_support,
                    )
                    metrics = base.metrics_on_truth(res, truth)
                    row = {
                        "Well_ID": wid,
                        "n_years": n_years,
                        "rep": rep,
                        "status": "ok" if metrics else "metrics_nan",
                        "prefill_added_points": n_prefilled,
                        "window_start": months[start_idx].strftime("%Y-%m"),
                        "window_end": months[min(len(months) - 1, start_idx + n_years * 12 - 1)].strftime("%Y-%m"),
                        "n_removed": len(truth),
                        "method": "mc_lnn_iterative",
                    }
                    row.update(iter_meta)
                    if metrics:
                        row.update(metrics)
                    rows.append(row)
                except Exception as exc:
                    rows.append({
                        "Well_ID": wid,
                        "n_years": n_years,
                        "rep": rep,
                        "status": f"error:{exc}",
                        "prefill_added_points": n_prefilled,
                        "method": "mc_lnn_iterative",
                    })
                done += 1
                if done % 25 == 0 or done == total:
                    print(f"[MC+LNN Iter CV] {done}/{total} folds complete in {time.time() - t0:.0f}s", flush=True)

    detailed = pd.DataFrame(rows)
    ok_df = detailed[detailed["status"] == "ok"].copy()
    summary = base.summarize_by_year(ok_df) if len(ok_df) else pd.DataFrame()

    suffix = f"_mc_lnn_iterative_{args.output_tag}"
    detailed_name = f"consecutive_long_gap_cv{suffix}_detailed.csv"
    summary_name = f"consecutive_long_gap_cv{suffix}_summary.csv"
    per_well_name = f"consecutive_long_gap_cv{suffix}_by_well.csv"
    meta_name = f"run_metadata{suffix}.json"
    compare_name = f"comparison_vs_baseline{suffix}.csv"
    summary_txt_name = f"SUMMARY{suffix}.txt"

    detailed.to_csv(OUT_DIR / detailed_name, index=False)
    summary.to_csv(OUT_DIR / summary_name, index=False)

    per_well = pd.DataFrame()
    if len(ok_df):
        metric_cols = ["n_scored", "mae", "me", "mse", "rmse", "rrmse", "pbias", "ia", "max_ae", "r", "r2", "kge", "nse", "outer_iterations_used", "support_points", "best_support_kge", "best_support_rmse"]
        metric_cols = [c for c in metric_cols if c in ok_df.columns]
        per_well = ok_df.groupby(["Well_ID", "n_years"])[metric_cols].agg(["mean", "std", "count"])
        per_well.columns = ["_".join(c).strip("_") for c in per_well.columns.to_flat_index()]
        per_well = per_well.reset_index()
    per_well.to_csv(OUT_DIR / per_well_name, index=False)

    compare_df = pd.DataFrame()
    baseline_path = Path(args.baseline_summary)
    if baseline_path.exists() and len(summary):
        baseline = pd.read_csv(baseline_path)
        keep_cols = ["n_years", "mae_mean", "mse_mean", "rmse_mean", "r_mean", "kge_mean"]
        baseline = baseline[[c for c in keep_cols if c in baseline.columns]].rename(columns={
            "mae_mean": "baseline_mae_mean",
            "mse_mean": "baseline_mse_mean",
            "rmse_mean": "baseline_rmse_mean",
            "r_mean": "baseline_r_mean",
            "kge_mean": "baseline_kge_mean",
        })
        current = summary[[c for c in ["n_years", "mae_mean", "mse_mean", "rmse_mean", "r_mean", "kge_mean"] if c in summary.columns]].rename(columns={
            "mae_mean": "iter_mae_mean",
            "mse_mean": "iter_mse_mean",
            "rmse_mean": "iter_rmse_mean",
            "r_mean": "iter_r_mean",
            "kge_mean": "iter_kge_mean",
        })
        compare_df = baseline.merge(current, on="n_years", how="outer")
        for metric in ["mae", "mse", "rmse", "r", "kge"]:
            bcol = f"baseline_{metric}_mean"
            icol = f"iter_{metric}_mean"
            if bcol in compare_df.columns and icol in compare_df.columns:
                compare_df[f"delta_{metric}"] = compare_df[icol] - compare_df[bcol]
        compare_df.to_csv(OUT_DIR / compare_name, index=False)

    run_meta = {
        "top_wells": top_wells,
        "missing_requested_wells": missing,
        "consecutive_years": base.CONSECUTIVE_YEARS,
        "consecutive_repeats": args.repeats,
        "min_truth_in_window": args.min_truth_in_window,
        "outer_iterations": args.outer_iterations,
        "feedback_prev_weight": args.feedback_prev_weight,
        "support_frac": args.support_frac,
        "min_support": args.min_support,
        "max_support": args.max_support,
        "requested_mode": "iterative fancyimpute SoftImpute mc-init -> lnn refinement -> soft feedback",
        "baseline_summary": str(baseline_path),
        "params": asdict(params),
        "successful_folds_by_year": ok_df.groupby("n_years").size().astype(int).to_dict() if len(ok_df) else {},
        "elapsed_seconds": time.time() - t0,
    }
    with open(OUT_DIR / meta_name, "w") as f:
        json.dump(run_meta, f, indent=2)

    lines = [
        "GSLB long-gap CV summary (Iterative SoftImpute MC-init + LNN)",
        f"Elapsed seconds: {run_meta['elapsed_seconds']:.1f}",
        f"Top wells: {json.dumps(top_wells)}",
        f"Consecutive repeats: {args.repeats}",
        f"Relaxed minimum truth in window: {args.min_truth_in_window}",
        f"Outer iterations: {args.outer_iterations}",
        f"Feedback previous weight: {args.feedback_prev_weight}",
        f"Support fraction: {args.support_frac}",
        f"Successful folds by year: {json.dumps(run_meta['successful_folds_by_year'])}",
        "",
        "Outputs:",
        f"  {detailed_name}",
        f"  {summary_name}",
        f"  {per_well_name}",
        f"  {meta_name}",
    ]
    if len(compare_df):
        lines.extend([f"  {compare_name}", "", "Comparison vs baseline:", compare_df.to_string(index=False)])
    if len(summary):
        lines.extend(["", "Iterative summary:", summary.to_string(index=False)])
    with open(OUT_DIR / summary_txt_name, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
