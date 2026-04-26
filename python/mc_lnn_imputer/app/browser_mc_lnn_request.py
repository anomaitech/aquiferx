#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import math
import sys
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fancyimpute import SoftImpute

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from backend.lnn.gaps import identify_gaps
from backend.lnn.lnn_core_aux_placeholder_cfc import optimize_lnn_params, run_lnn_simulation
from backend.lnn.math_utils import calculate_kge
from backend.lnn.types import DataPoint, SimulationParams

MAX_DONORS = 15
MIN_DONOR_CORR = 0.3


def emit(event: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


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
        small_gap_optimize_trials=3,
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


def _rng_seed(*parts: Any) -> int:
    import hashlib
    h = hashlib.sha256(b"|".join(str(p).encode() for p in parts)).digest()
    return int.from_bytes(h[:8], "little") % (2**31)


def mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return float(math.sqrt(sum((v - m) ** 2 for v in values) / len(values)))


def build_timeline(points: List[Dict[str, Any]], well_id: str, add_seasonal: bool) -> List[DataPoint]:
    timeline: List[DataPoint] = []
    for i, p in enumerate(points):
        aux = list(p.get("auxiliaries") or [])
        if add_seasonal:
            aux = aux + [math.sin(2 * math.pi * i / 12.0), math.cos(2 * math.pi * i / 12.0)]
        timeline.append(
            DataPoint(
                time=float(p["time"]),
                observed=p.get("observed"),
                instance_id=well_id,
                auxiliaries=aux,
                date_label=p["date"],
                latitude=0.0,
                longitude=0.0,
            )
        )
    return timeline


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


def impute_timeline_fast(timeline: List[DataPoint], params: SimulationParams, seed: int) -> List[DataPoint]:
    with redirect_stdout(io.StringIO()):
        best = optimize_lnn_params(timeline, params, mode="projection", rng=np.random.default_rng(seed))
        return run_lnn_simulation(timeline, best, rng=np.random.default_rng(seed))


def small_gap_index_set(timeline: List[DataPoint], max_gap_threshold: int) -> set[int]:
    gaps, _ = identify_gaps(timeline, max_gap_threshold)
    out: set[int] = set()
    for g in gaps:
        if g["size"] <= max_gap_threshold:
            for i in range(g["startIdx"], g["endIdx"] + 1):
                out.add(i)
    return out


def prefill_small_gaps_all(raw_obs: Dict[str, List[Optional[float]]], point_template: List[Dict[str, Any]], params: SimulationParams, seed: int) -> Dict[str, List[Optional[float]]]:
    filled: Dict[str, List[Optional[float]]] = {}
    sg_params = SimulationParams(
        max_gap_threshold=min(params.max_gap_threshold, 20),
        large_gap_threshold=params.large_gap_threshold,
        kge_threshold=params.kge_threshold,
        small_gap_kge_threshold=params.small_gap_kge_threshold,
        small_gap_use_auxiliary=True,
        small_gap_auxiliary_weight=params.small_gap_auxiliary_weight,
        small_gap_use_neighbors=False,
        small_gap_max_iterations=3,
        small_gap_optimize_trials=3,
        ridge_alpha=params.ridge_alpha,
        lnn_aux_placeholder_readout=params.lnn_aux_placeholder_readout,
    )
    for idx, (wid, raw) in enumerate(raw_obs.items()):
        pts = []
        for i, base_point in enumerate(point_template):
            pts.append({
                "date": base_point["date"],
                "time": base_point["time"],
                "observed": raw[i],
                "auxiliaries": list(base_point.get("auxiliaries") or []),
            })
        tl = build_timeline(pts, wid, add_seasonal=False)
        small_ix = small_gap_index_set(tl, sg_params.max_gap_threshold)
        if not small_ix:
            filled[wid] = list(raw)
            continue
        res = impute_timeline_fast(tl, sg_params, seed + idx)
        out = list(raw)
        for i in small_ix:
            if raw[i] is None and i < len(res) and res[i].imputed is not None:
                out[i] = float(res[i].imputed)
        filled[wid] = out
    return filled


def _archi_regression(target_obs: Dict[int, float], donor_obs: Dict[str, List[Optional[float]]], target_id: str, n_times: int) -> Tuple[Dict[int, float], List[Dict[str, Any]]]:
    if len(target_obs) < 5:
        return {}, []
    gap_times = sorted(set(range(n_times)) - set(target_obs.keys()))
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


def _matrix_completion_archi_init_iter(
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
        solver = SoftImpute(shrinkage_value=None, convergence_threshold=1e-6, max_iters=50, max_rank=k_try, init_fill_method="zero", verbose=False)
        Xt = solver.fit_transform(M)
        if len(target_obs_idx) > 3:
            err = float(np.mean((Xt[0, target_obs_idx] - M[0, target_obs_idx]) ** 2))
            if err < best_err:
                best_err = err
                best_k = k_try
                best_x = Xt

    solver = SoftImpute(shrinkage_value=None, convergence_threshold=1e-6, max_iters=100, max_rank=best_k, init_fill_method="zero", verbose=False)
    X = solver.fit_transform(M) if best_x is None else solver.fit_transform(M)

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


def run_mc_lnn_single_pass(target_id: str, mod_target: List[Optional[float]], donor_obs_filled: Dict[str, List[Optional[float]]], point_template: List[Dict[str, Any]], params: SimulationParams, seed: int, target_init_preds: Optional[Dict[int, float]] = None) -> List[DataPoint]:
    target_obs = {i: v for i, v in enumerate(mod_target) if v is not None}
    n_times = len(mod_target)
    archi_preds, donors = _archi_regression(target_obs, donor_obs_filled, target_id, n_times)
    target_aux = [list((point_template[i].get("auxiliaries") or [])[:5]) for i in range(n_times)]
    if not donors:
        pts = []
        for i, base_point in enumerate(point_template):
            pts.append({
                "date": base_point["date"],
                "time": base_point["time"],
                "observed": mod_target[i],
                "auxiliaries": list(base_point.get("auxiliaries") or []),
            })
        return impute_timeline(build_timeline(pts, target_id, add_seasonal=True), params, seed)

    mc_preds = _matrix_completion_archi_init_iter(target_series=mod_target, target_aux=target_aux, donor_obs=donor_obs_filled, donors=donors, archi_preds=archi_preds, target_init_preds=target_init_preds)
    enriched: List[Optional[float]] = []
    for i, v in enumerate(mod_target):
        if v is not None:
            enriched.append(v)
        elif i in mc_preds and np.isfinite(mc_preds[i]):
            enriched.append(float(mc_preds[i]))
        else:
            enriched.append(None)
    pts = []
    for i, base_point in enumerate(point_template):
        pts.append({
            "date": base_point["date"],
            "time": base_point["time"],
            "observed": enriched[i],
            "auxiliaries": list(base_point.get("auxiliaries") or []),
        })
    return impute_timeline(build_timeline(pts, target_id, add_seasonal=True), params, seed)


def compute_metrics(obs: List[float], pred: List[float]) -> Dict[str, float]:
    o = np.asarray(obs, dtype=float)
    p = np.asarray(pred, dtype=float)
    err = p - o
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    kge = float(calculate_kge(list(o), list(p))) if len(o) >= 2 else float("nan")
    return {"rmse": rmse, "kge": kge}


def select_support_truth(raw_target: List[Optional[float]], truth_missing_idx: List[int], rng: np.random.Generator, support_frac: float, min_support: int, max_support: int) -> Dict[int, float]:
    truth_set = set(truth_missing_idx)
    candidates = [i for i, v in enumerate(raw_target) if v is not None and i not in truth_set]
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


def blend_predictions(prev_init: Optional[Dict[int, float]], result: List[DataPoint], missing_idx: List[int], prev_weight: float) -> Dict[int, float]:
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


def run_iterative_fold(target_id: str, raw_target: List[Optional[float]], mod_target: List[Optional[float]], donor_obs_filled: Dict[str, List[Optional[float]]], point_template: List[Dict[str, Any]], params: SimulationParams, seed: int, outer_iterations: int, feedback_prev_weight: float, support_frac: float, min_support: int, max_support: int) -> Tuple[List[DataPoint], Dict[str, Any]]:
    truth_missing_idx = [i for i, v in enumerate(mod_target) if v is None]
    support_rng = np.random.default_rng(_rng_seed(seed, "support"))
    support_truth = select_support_truth(raw_target, [i for i in truth_missing_idx if raw_target[i] is not None], support_rng, support_frac, min_support, max_support)
    if not support_truth:
        return run_mc_lnn_single_pass(target_id, mod_target, donor_obs_filled, point_template, params, seed), {
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
        result = run_mc_lnn_single_pass(target_id, work_target, donor_fold, point_template, params, _rng_seed(seed, "outer", outer), target_init_preds=best_init)
        support_obs: List[float] = []
        support_pred: List[float] = []
        for idx, truth_v in support_truth.items():
            pred_v = result[idx].imputed if idx < len(result) else None
            if pred_v is not None:
                support_obs.append(truth_v)
                support_pred.append(float(pred_v))
        support_metrics = compute_metrics(support_obs, support_pred) if support_obs else None
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
        final_result = run_mc_lnn_single_pass(target_id, mod_target, final_donor_fold, point_template, params, _rng_seed(seed, "final"), target_init_preds=final_init)
    else:
        final_result = run_mc_lnn_single_pass(target_id, mod_target, donor_obs_filled, point_template, params, seed)

    meta = {
        "outer_iterations_used": accepted_iters,
        "support_points": len(support_truth),
        "best_support_kge": float(best_support_metrics["kge"]) if best_support_metrics and np.isfinite(best_support_metrics.get("kge", np.nan)) else float("nan"),
        "best_support_rmse": float(best_support_metrics["rmse"]) if best_support_metrics and np.isfinite(best_support_metrics.get("rmse", np.nan)) else float("nan"),
    }
    return final_result, meta


def main() -> None:
    payload = json.load(sys.stdin)
    wells_in = payload["wells"]
    config = payload.get("config", {})
    params = build_params()
    seed = int(config.get("seed", 42))
    outer_iterations = int(config.get("outerIterations", 2))
    feedback_prev_weight = float(config.get("feedbackPrevWeight", 0.35))
    support_frac = float(config.get("supportFrac", 0.12))
    min_support = int(config.get("minSupport", 6))
    max_support = int(config.get("maxSupport", 24))

    raw_obs: Dict[str, List[Optional[float]]] = {w["wellId"]: [p.get("observed") for p in w["points"]] for w in wells_in}
    point_templates: Dict[str, List[Dict[str, Any]]] = {w["wellId"]: w["points"] for w in wells_in}
    first_points = wells_in[0]["points"] if wells_in else []
    emit({"type": "progress", "label": "Python MC+LNN small-gap prefill...", "pct": 28})
    filled_obs = prefill_small_gaps_all(raw_obs, first_points, params, _rng_seed(seed, "prefill_all_iterative_browser"))

    notes: List[str] = []
    out_wells: List[Dict[str, Any]] = []
    total_wells = max(1, len(wells_in))
    for idx, well in enumerate(wells_in):
        wid = well["wellId"]
        raw_target = list(raw_obs[wid])
        target_series = list(filled_obs[wid])
        donor_obs_filled = {k: list(v) for k, v in filled_obs.items()}
        result, iter_meta = run_iterative_fold(
            wid,
            raw_target,
            target_series,
            donor_obs_filled,
            point_templates[wid],
            params,
            seed=_rng_seed(seed, "iter_full", wid),
            outer_iterations=outer_iterations,
            feedback_prev_weight=feedback_prev_weight,
            support_frac=support_frac,
            min_support=min_support,
            max_support=max_support,
        )

        rows: List[Dict[str, Any]] = []
        raw_missing = sum(v is None for v in raw_target)
        remaining_after_small = sum(target_series[i] is None for i in range(len(target_series)))
        small_note = f"[{wid}] small-gap complete | rawMissing={raw_missing} | remainingAfterSmall={remaining_after_small}"
        large_note = (
            f"[{wid}] large-gap support: outer={iter_meta.get('outer_iterations_used', 0)}, "
            f"support={iter_meta.get('support_points', 0)}, "
            f"supportKGE={iter_meta.get('best_support_kge') if np.isfinite(iter_meta.get('best_support_kge', np.nan)) else 'NA'}, "
            f"supportRMSE={iter_meta.get('best_support_rmse') if np.isfinite(iter_meta.get('best_support_rmse', np.nan)) else 'NA'}"
        )
        notes.append(small_note)
        notes.append(large_note)
        emit({"type": "note", "message": small_note})
        emit({"type": "note", "message": large_note})
        emit({
            "type": "well_metrics",
            "wellId": wid,
            "rawMissing": raw_missing,
            "remainingAfterSmall": remaining_after_small,
            "outerIterationsUsed": int(iter_meta.get("outer_iterations_used", 0)),
            "supportPoints": int(iter_meta.get("support_points", 0)),
            "supportKGE": None if not np.isfinite(iter_meta.get("best_support_kge", np.nan)) else float(iter_meta.get("best_support_kge")),
            "supportRMSE": None if not np.isfinite(iter_meta.get("best_support_rmse", np.nan)) else float(iter_meta.get("best_support_rmse")),
        })
        for i, dp in enumerate(result):
            raw_v = raw_target[i]
            sg_v = target_series[i]
            final_v = dp.imputed
            if raw_v is not None:
                stage = "observed"
            elif sg_v is not None:
                stage = "small_gap_lnn_cfc_aux"
            elif final_v is not None:
                stage = "large_gap_mc_lnn_iterative"
            else:
                stage = "unfilled"
            rows.append({
                "date": dp.date_label,
                "raw": raw_v,
                "smallGap": sg_v,
                "final": float(final_v) if final_v is not None else None,
                "fillStage": stage,
            })
        out_wells.append({"wellId": wid, "rows": rows})
        emit({
            "type": "progress",
            "label": f"Validated Python MC + LNN... ({idx + 1}/{total_wells} wells)",
            "pct": 28 + ((idx + 1) / total_wells) * 54,
        })

    emit({"type": "progress", "label": "Validated Python MC + LNN complete", "pct": 84})
    emit({
        "type": "result",
        "payload": {
            "wells": out_wells,
            "notes": notes,
            "config": {
                "outerIterations": outer_iterations,
                "feedbackPrevWeight": feedback_prev_weight,
                "supportFrac": support_frac,
                "minSupport": min_support,
                "maxSupport": max_support,
                "params": asdict(params),
            },
        },
    })


if __name__ == "__main__":
    main()
