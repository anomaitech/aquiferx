"""
MC + LNN CFC: Full pipeline for large gap imputation.

Pipeline:
  Phase 1: Small gap fill (≤20 steps) ALL wells using LNN CFC Aux
           → densified data for all instances

  Phase 2: Donor-initialized MC → LNN CFC
           a) ARCHI-style donor selection (top correlated wells from full pool)
           b) Donor regression (OLS per donor) to compute trend-aware
              initialization for the MC target row
           c) MC: z-score normalized SVD with weighted donor rows +
              aux anchor rows + sin/cos temporal rows.
              Target row initialized with donor regression (not zero-mean).
              Adaptive rank via cross-validation.
           d) MC output → pseudo-observations → LNN CFC refines

  The key insight: donor regression provides the TREND during the gap,
  MC provides the SPATIAL STRUCTURE, and LNN provides NONLINEAR REFINEMENT.
  ARCHI's concept (donor correlation) is embedded in the MC initialization,
  not blended as a separate step.
"""

from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import numpy as np

from .types import DataPoint, SimulationParams
from .lnn_core_aux_placeholder_cfc import (
    run_lnn_simulation,
    optimize_lnn_params,
    _precompute_cfc_aux_context,
    compute_aux_target_correlations,
)
from .math_utils import calculate_kge, calculate_pearson_correlation
from .gaps import identify_gaps

try:
    from scipy.interpolate import PchipInterpolator
except ImportError:
    PchipInterpolator = None


# ─── Phase 1: Small gap fill (PCHIP — matches browser pipeline) ───────────

def _pchip_fill_single(well_data: List[DataPoint], max_gap: int = 24) -> List[DataPoint]:
    """Fill gaps ≤ max_gap using PCHIP interpolation.
    Matches the browser pchipFill() in mcLnnPureBrowser.ts."""
    obs_idx = [i for i, d in enumerate(well_data) if d.observed is not None]
    obs_vals = [well_data[i].observed for i in obs_idx]

    if len(obs_idx) < 3:
        return list(well_data)

    # Identify small gaps
    gaps = []
    i = 0
    while i < len(well_data):
        if well_data[i].observed is None:
            start = i
            while i < len(well_data) and well_data[i].observed is None:
                i += 1
            if i - start <= max_gap:
                gaps.append((start, i - 1))
        else:
            i += 1

    if not gaps:
        return list(well_data)

    pchip = PchipInterpolator(obs_idx, obs_vals)
    filled = list(well_data)
    for g_start, g_end in gaps:
        for t in range(g_start, g_end + 1):
            val = float(pchip(t))
            if np.isfinite(val):
                filled[t] = DataPoint(
                    time=well_data[t].time,
                    observed=val,
                    instance_id=well_data[t].instance_id,
                    auxiliaries=well_data[t].auxiliaries,
                    latitude=well_data[t].latitude,
                    longitude=well_data[t].longitude,
                    date_label=well_data[t].date_label,
                    timestamp=well_data[t].timestamp,
                )
    return filled


def fill_small_gaps_all(
    all_instance_data: Dict[str, List[DataPoint]],
    params: SimulationParams,
) -> Dict[str, List[DataPoint]]:
    """Phase 1: PCHIP small-gap fill for ALL wells.
    Matches the browser mcLnnPureBrowser.ts Phase 1 pipeline."""
    max_gap = min(params.max_gap_threshold, 24)

    if PchipInterpolator is None:
        print("  [MC+LNN Phase 1] WARNING: scipy not available, skipping PCHIP fill", flush=True)
        return dict(all_instance_data)

    filled = {}
    n_total = 0
    for wid, data in all_instance_data.items():
        before = sum(1 for d in data if d.observed is not None)
        filled[wid] = _pchip_fill_single(data, max_gap=max_gap)
        after = sum(1 for d in filled[wid] if d.observed is not None)
        n_total += (after - before)

    n_denser = sum(1 for wid in filled
                   if sum(1 for d in filled[wid] if d.observed is not None) >
                      sum(1 for d in all_instance_data[wid] if d.observed is not None))
    print(f"  [MC+LNN Phase 1] PCHIP small gaps filled: {n_total} points across "
          f"{n_denser}/{len(all_instance_data)} wells", flush=True)
    return filled


# ─── Phase 2: ARCHI + MC → LNN ─────────────────────────────────────────────

def _archi_regression(
    target_data: List[DataPoint],
    all_data_filled: Dict[str, List[DataPoint]],
    target_id: str,
    max_donors: int = 15,
    min_correlation: float = 0.3,
) -> Tuple[Dict[float, float], list]:
    """ARCHI-style donor regression for trend capture."""
    target_obs = {d.time: d.observed for d in target_data if d.observed is not None}
    gap_times = {d.time for d in target_data if d.observed is None}

    if len(target_obs) < 5:
        return {}, []

    donors = []
    for wid, wdata in all_data_filled.items():
        if wid == target_id:
            continue
        dobs = {d.time: d.observed for d in wdata if d.observed is not None}
        common = sorted(set(target_obs.keys()) & set(dobs.keys()))
        if len(common) < 8:
            continue
        tv = np.array([target_obs[t] for t in common])
        dv = np.array([dobs[t] for t in common])
        if np.std(tv) < 1e-10 or np.std(dv) < 1e-10:
            continue
        r = float(np.corrcoef(tv, dv)[0, 1])
        if abs(r) < min_correlation:
            continue
        donors.append({'wid': wid, 'r': r, 'dobs': dobs})

    donors.sort(key=lambda x: abs(x['r']), reverse=True)
    donors = donors[:max_donors]

    if not donors:
        return {}, []

    # OLS regression per donor, weighted average
    all_preds = []
    weights = []
    for di in donors:
        common = sorted(set(target_obs.keys()) & set(di['dobs'].keys()))
        if len(common) < 5:
            continue
        tv = np.array([target_obs[t] for t in common])
        dv = np.array([di['dobs'][t] for t in common])
        dm, tm = np.mean(dv), np.mean(tv)
        ss = np.sum((dv - dm) ** 2)
        if ss < 1e-10:
            continue
        a = np.sum((dv - dm) * (tv - tm)) / ss
        b = tm - a * dm
        preds = {t: a * di['dobs'][t] + b for t in gap_times if t in di['dobs']}
        if preds:
            all_preds.append(preds)
            weights.append(di['r'] ** 2)

    combined = {}
    for t in gap_times:
        ws, wc = 0.0, 0.0
        for pi, p in enumerate(all_preds):
            if t in p:
                ws += p[t] * weights[pi]
                wc += weights[pi]
        if wc > 0:
            combined[t] = ws / wc

    print(f"  [MC+LNN Phase 2] ARCHI: {len(donors)} donors, "
          f"filled {len(combined)}/{len(gap_times)} large gap positions", flush=True)
    return combined, donors


def _matrix_completion_archi_init(
    target_data: List[DataPoint],
    all_data_filled: Dict[str, List[DataPoint]],
    target_id: str,
    donors: list,
    archi_preds: Dict[float, float],
    params: SimulationParams,
) -> Dict[float, float]:
    """MC with ARCHI-initialized target row + temporal/aux anchor rows."""
    n_times = len(target_data)
    target_obs = {d.time: d.observed for d in target_data if d.observed is not None}

    sel_wids = [target_id] + [d['wid'] for d in donors]
    nw = len(sel_wids)

    n_aux = 0
    for d in target_data:
        if d.auxiliaries:
            n_aux = len(d.auxiliaries)
            break
    n_temp = 2  # sin, cos
    n_extra = n_aux + n_temp

    # Build well matrix from FILLED data
    M_raw = np.full((nw, n_times), np.nan)
    for wi, wid in enumerate(sel_wids):
        if wid == target_id:
            for d in target_data:
                if d.observed is not None:
                    idx = int(d.time)
                    if 0 <= idx < n_times:
                        M_raw[wi, idx] = d.observed
        else:
            wdata = all_data_filled.get(wid, [])
            for d in wdata:
                if d.observed is not None:
                    idx = int(d.time)
                    if 0 <= idx < n_times:
                        M_raw[wi, idx] = d.observed

    # Z-score normalize
    rmeans, rstds = np.zeros(nw), np.ones(nw)
    M = np.full((nw + n_extra, n_times), np.nan)
    for wi in range(nw):
        v = M_raw[wi, ~np.isnan(M_raw[wi, :])]
        if len(v) >= 3:
            rmeans[wi] = np.mean(v)
            rstds[wi] = max(np.std(v), 1e-10)
        elif len(v) > 0:
            rmeans[wi] = np.mean(v)
        M[wi, :] = (M_raw[wi, :] - rmeans[wi]) / rstds[wi]

    # Weight donor rows by correlation
    for di_idx, di in enumerate(donors):
        M[di_idx + 1, :] *= abs(di['r'])

    # Aux rows
    for j in range(n_aux):
        ac = np.array([
            target_data[t].auxiliaries[j]
            if target_data[t].auxiliaries and j < len(target_data[t].auxiliaries) else 0.0
            for t in range(n_times)
        ])
        am, ast = np.mean(ac), max(np.std(ac), 1e-10)
        M[nw + j, :] = (ac - am) / ast

    # Temporal rows
    for t in range(n_times):
        M[nw + n_aux, t] = np.sin(2 * np.pi * t / 12.0)
        M[nw + n_aux + 1, t] = np.cos(2 * np.pi * t / 12.0)

    obs_mask = ~np.isnan(M)
    X = M.copy()

    # Initialize target row with ARCHI predictions (trend-aware)
    for t in range(n_times):
        if np.isnan(X[0, t]):
            ft = float(t)
            if ft in archi_preds:
                X[0, t] = (archi_preds[ft] - rmeans[0]) / rstds[0]
            else:
                X[0, t] = 0.0
    for wi in range(1, nw):
        rv = M[wi, ~np.isnan(M[wi, :])]
        rm = np.mean(rv) if len(rv) > 0 else 0.0
        X[wi, np.isnan(X[wi, :])] = rm

    # Adaptive rank
    t_obs_idx = np.where(~np.isnan(M[0, :]))[0]
    best_k, best_err = 5, float('inf')
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
        if len(t_obs_idx) > 3:
            err = np.mean((Xt[0, t_obs_idx] - M[0, t_obs_idx]) ** 2)
            if err < best_err:
                best_err = err
                best_k = k_try

    # Final SVD
    for it in range(100):
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        St = np.zeros_like(S)
        St[:best_k] = S[:best_k]
        Xn = U @ np.diag(St) @ Vt
        Xn[obs_mask] = M[obs_mask]
        if np.max(np.abs(Xn - X)) < 1e-6:
            break
        X = Xn

    # Un-weight donors
    for di_idx, di in enumerate(donors):
        if abs(di['r']) > 1e-10:
            X[di_idx + 1, :] /= abs(di['r'])

    # De-normalize + bias correct (with safeguards)
    pred = X[0, :] * rstds[0] + rmeans[0]
    if len(t_obs_idx) >= 3:
        ov = M_raw[0, t_obs_idx]
        mv = pred[t_obs_idx]
        om, ost = np.mean(ov), max(np.std(ov), 1e-10)
        mm, mst = np.mean(mv), max(np.std(mv), 1e-10)
        # Safeguard: skip MOVE.1 if variance ratio is extreme
        var_ratio = ost / mst
        if 0.1 < var_ratio < 10:
            pred = (pred - mm) / mst * ost + om

    # Clamp to observed range (with margin)
    if len(t_obs_idx) >= 2:
        obs_vals = M_raw[0, t_obs_idx]
        obs_min, obs_max = float(np.min(obs_vals)), float(np.max(obs_vals))
        obs_range = obs_max - obs_min
        margin = max(obs_range * 0.2, 1.0)
        pred = np.clip(pred, obs_min - margin, obs_max + margin)

    # Quality check: if MC reconstruction error on observed points is poor,
    # blend with ARCHI predictions
    if len(t_obs_idx) > 0:
        mc_rmse_obs = float(np.sqrt(np.mean((pred[t_obs_idx] - M_raw[0, t_obs_idx]) ** 2)))
        if mc_rmse_obs > rstds[0] * 1.5 and archi_preds:
            # MC is unreliable — blend with ARCHI
            for t in range(n_times):
                ft = float(t)
                if ft in archi_preds and np.isfinite(archi_preds[ft]):
                    pred[t] = 0.3 * pred[t] + 0.7 * archi_preds[ft]

    print(f"  [MC+LNN Phase 2] MC: rank={best_k}, "
          f"{sum(~np.isnan(M_raw[0,:]))} target obs + "
          f"{sum(sum(~np.isnan(M_raw[wi,:])) for wi in range(1,nw))} donor obs", flush=True)

    return {float(t): float(pred[t]) for t in range(n_times)}


def run_mc_lnn_cfc_simulation(
    target_data: List[DataPoint],
    all_instance_data: Dict[str, List[DataPoint]],
    target_instance_id: str,
    params: SimulationParams,
    rng: Optional[np.random.Generator] = None,
) -> List[DataPoint]:
    """
    Full MC+LNN CFC pipeline:
      Phase 1: Small gap fill ALL wells
      Phase 2: ARCHI + MC → LNN for large gaps
    """
    if rng is None:
        rng = np.random.default_rng()

    max_donors = getattr(params, 'mc_max_donors', 15) or 15
    min_corr = getattr(params, 'mc_min_correlation', 0.3) or 0.3

    # Phase 1: Small gap fill all wells
    print(f"  [MC+LNN] Phase 1: Small gap fill ({len(all_instance_data)} wells)...", flush=True)
    filled_data = fill_small_gaps_all(all_instance_data, params)

    # Use filled target data for Phase 2
    target_filled = filled_data.get(target_instance_id, target_data)
    target_obs_original = {d.time: d.observed for d in target_data if d.observed is not None}

    # Phase 2: Donor-initialized MC → LNN
    print(f"  [MC+LNN] Phase 2: Donor-initialized MC → LNN...", flush=True)

    # ARCHI regression on filled data
    archi_preds, donors = _archi_regression(
        target_filled, filled_data, target_instance_id,
        max_donors=max_donors, min_correlation=min_corr,
    )

    if not donors:
        # Fallback: standard LNN
        print(f"  [MC+LNN] No donors found, falling back to standard LNN CFC", flush=True)
        ctx = _precompute_cfc_aux_context(target_filled, params)
        bp = optimize_lnn_params(target_filled, params, mode="projection", rng=rng, precomputed=ctx)
        return run_lnn_simulation(target_filled, bp, rng=rng, precomputed=ctx)

    # MC with ARCHI initialization
    mc_preds = _matrix_completion_archi_init(
        target_filled, filled_data, target_instance_id,
        donors, archi_preds, params,
    )

    # MC output goes directly to LNN as pseudo-observations
    # (ARCHI's concept is already embedded in the MC via initialization —
    #  no separate blending needed)
    enriched = []
    for d in target_data:
        if d.time in target_obs_original:
            enriched.append(d)  # original observation
        elif d.time in mc_preds and mc_preds[d.time] is not None and np.isfinite(mc_preds[d.time]):
            enriched.append(DataPoint(
                time=d.time, observed=mc_preds[d.time],
                instance_id=d.instance_id, auxiliaries=d.auxiliaries,
                latitude=d.latitude, longitude=d.longitude,
                date_label=d.date_label, timestamp=d.timestamp,
            ))
        elif d.observed is not None:  # small-gap-filled
            enriched.append(d)
        else:
            enriched.append(d)

    n_enriched = sum(1 for d in enriched if d.observed is not None)
    print(f"  [MC+LNN] Enriched: {n_enriched}/{len(target_data)} observed", flush=True)

    # LNN CFC refinement
    print(f"  [MC+LNN] Running LNN CFC refinement...", flush=True)
    lnn_params = SimulationParams(
        reservoir_size=params.reservoir_size,
        spectral_radius=params.spectral_radius,
        leak_rate=params.leak_rate,
        input_scaling=params.input_scaling,
        max_gap_threshold=9999,
        kge_threshold=params.kge_threshold,
        small_gap_kge_threshold=params.small_gap_kge_threshold,
        ridge_alpha=params.ridge_alpha,
        lnn_aux_placeholder_readout=params.lnn_aux_placeholder_readout,
        small_gap_optimize_trials=max(params.small_gap_optimize_trials or 8, 8),
    )

    ctx = _precompute_cfc_aux_context(enriched, lnn_params)
    bp = optimize_lnn_params(enriched, lnn_params, mode="projection", rng=rng, precomputed=ctx)

    best_result = None
    best_kge = float('-inf')
    for it in range(5):
        rng_it = np.random.default_rng(rng.integers(0, 2**31) + it)
        result = run_lnn_simulation(enriched, bp, rng=rng_it, precomputed=ctx)
        obs_v = [d.observed for d in enriched if d.observed is not None]
        pred_v = [result[i].imputed for i, d in enumerate(enriched)
                  if d.observed is not None and result[i].imputed is not None]
        if obs_v and pred_v and len(obs_v) == len(pred_v):
            kge = calculate_kge(obs_v, pred_v)
            if kge > best_kge:
                best_kge = kge
                best_result = result

    if best_result is None:
        best_result = run_lnn_simulation(enriched, bp, rng=rng, precomputed=ctx)

    print(f"  [MC+LNN] Done. Training KGE={best_kge:.4f}", flush=True)

    # Map back with original observations
    final = []
    for i, d in enumerate(target_data):
        final.append(DataPoint(
            time=d.time,
            observed=d.observed,
            instance_id=d.instance_id,
            auxiliaries=d.auxiliaries,
            imputed=best_result[i].imputed if i < len(best_result) else None,
            imputed_std=getattr(best_result[i], 'imputed_std', None) if i < len(best_result) else None,
            latitude=d.latitude,
            longitude=d.longitude,
            date_label=d.date_label,
            timestamp=d.timestamp,
        ))

    return final
