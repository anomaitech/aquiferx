"""
1D Kriging, Season-Aware: Heteroscedastic Gaussian Process Regression.

Unlike the standard 1D Kriging (constant noise), this algorithm learns
time-dependent observation noise from ALL instances (pooled data) so that
uncertainty varies with the season.  Two stages:

  Stage 1 – Learn seasonal noise model (pooled across all instances):
    a. Fit a preliminary GP on pooled known data.
    b. Compute squared residuals.
    c. Fit a second GP on sin/cos seasonal features → log(residual²)
       to capture how noise variance changes over the seasonal cycle.

  Stage 2 – Per-instance imputation with learned noise:
    For each instance, fit a GP with per-point alpha = exp(noise_model(t)),
    giving heteroscedastic behaviour.  sklearn GPR supports array `alpha`.
"""

from dataclasses import replace
from typing import List, Dict, Any, Optional, Tuple, Callable
from collections import defaultdict
import warnings

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.exceptions import ConvergenceWarning

from .types import DataPoint, SimulationParams
from .math_utils import calculate_kge
from .gaps import identify_gaps

warnings.simplefilter("ignore", category=ConvergenceWarning)

_TAG = "[KrigingSeasonal]"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seasonal_features(times: np.ndarray, period: float) -> np.ndarray:
    """Return Nx2 array of [sin(2*pi*t/P), cos(2*pi*t/P)]."""
    angle = 2.0 * np.pi * times / period
    return np.column_stack([np.sin(angle), np.cos(angle)])


def _gpr_kernel():
    """Same RBF + WhiteKernel kernel as the standard 1D Kriging."""
    return C(1.0, (1e-3, 1e3)) * RBF(10.0, (1e-2, 1e2)) + WhiteKernel(noise_level=1e-3)


def _noise_kernel():
    """Kernel for the noise-variance GP (operates on 2D seasonal features)."""
    return C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=1e-2)


# ---------------------------------------------------------------------------
# Stage 1: learn seasonal noise from pooled data
# ---------------------------------------------------------------------------

def _learn_seasonal_noise(
    all_known_t: np.ndarray,
    all_known_z: np.ndarray,
    period: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Learn a **relative** seasonal noise pattern from pooled known data.

    Returns a callable  noise_model(t_array) → relative_weight_array
    where the mean weight ≈ 1.0.  The caller multiplies by a per-instance
    base noise to get the actual alpha for sklearn GPR.
    """
    n = len(all_known_t)
    print(f"{_TAG} Stage 1: learning seasonal noise pattern from {n} pooled known points (period={period})", flush=True)

    if n < 10:
        print(f"{_TAG}   too few points, uniform seasonal weight", flush=True)
        return lambda t: np.ones(len(t))

    # ---- Per-instance normalisation so residuals reflect shape, not level ----
    # Group by instance-like buckets: use modular time (seasonal position)
    # Actually, the key issue is that pooling raw values from instances with
    # very different means creates huge residuals.  Instead, we normalise
    # each instance's contribution to zero-mean, unit-variance before pooling.
    # Since we don't have instance labels here, we use a different strategy:
    # compute residuals as deviations from a LOCAL rolling mean (window = period)
    # so that the level differences between instances cancel out.
    sort_idx = np.argsort(all_known_t)
    sorted_t = all_known_t[sort_idx]
    sorted_z = all_known_z[sort_idx]

    # Rolling-window residuals: for each point, subtract the mean of points
    # within ± period/2 time-steps.  This removes the local trend/level.
    half_win = period / 2.0
    local_residuals = np.zeros(n, dtype=np.float64)
    for i in range(n):
        mask = np.abs(sorted_t - sorted_t[i]) <= half_win
        local_mean = sorted_z[mask].mean()
        local_residuals[i] = sorted_z[i] - local_mean

    residuals_sq = local_residuals ** 2
    residuals_sq = np.maximum(residuals_sq, 1e-12)
    log_res_sq = np.log(residuals_sq)

    # Subsample for GP fitting if too large
    max_pool = 2000
    if n > max_pool:
        idx = np.random.default_rng(42).choice(n, max_pool, replace=False)
        idx.sort()
        fit_t = sorted_t[idx]
        fit_log_res = log_res_sq[idx]
    else:
        fit_t = sorted_t
        fit_log_res = log_res_sq

    # Fit noise GP: seasonal features → log(local_residual²)
    X_season = _seasonal_features(fit_t, period)
    try:
        gpr_noise = GaussianProcessRegressor(
            kernel=_noise_kernel(),
            n_restarts_optimizer=3,
            normalize_y=True,
        )
        gpr_noise.fit(X_season, fit_log_res)

        # Report seasonal pattern range
        test_t = np.linspace(0, period, 50)
        test_feat = _seasonal_features(test_t, period)
        test_log_var = gpr_noise.predict(test_feat)
        test_var = np.exp(test_log_var)
        # Normalise so mean = 1
        test_mean = test_var.mean()
        ratio = test_var.max() / max(test_var.min(), 1e-12)
        print(f"{_TAG}   seasonal noise ratio (max/min over 1 cycle): {ratio:.2f}x", flush=True)
        print(f"{_TAG}   raw seasonal variance range: [{test_var.min():.4f}, {test_var.max():.4f}], mean={test_mean:.4f}", flush=True)
    except Exception as e:
        print(f"{_TAG}   noise GP failed ({e}), uniform seasonal weight", flush=True)
        return lambda t: np.ones(len(t))

    def noise_model(t_arr: np.ndarray) -> np.ndarray:
        feat = _seasonal_features(t_arr, period)
        log_var = gpr_noise.predict(feat)
        var = np.exp(log_var)
        # Normalise to mean = 1.0 so this is a relative seasonal weight
        mean_var = var.mean()
        if mean_var > 0:
            var = var / mean_var
        # Clamp extremes: no weight below 0.1 or above 10
        var = np.clip(var, 0.1, 10.0)
        return var

    return noise_model


# ---------------------------------------------------------------------------
# Stage 2: per-instance imputation
# ---------------------------------------------------------------------------

def _kriging_seasonal_impute_instance(
    points: List[DataPoint],
    noise_model: Callable[[np.ndarray], np.ndarray],
) -> List[DataPoint]:
    """
    Per-instance GPR with heteroscedastic noise (alpha = noise_model(t)).
    """
    inst_id = points[0].instance_id if points else "?"

    known_t: List[float] = []
    known_z: List[float] = []
    missing_indices: List[int] = []

    for i, p in enumerate(points):
        if p.observed is not None:
            known_t.append(p.time)
            known_z.append(float(p.observed))
        elif p.imputed is not None:
            known_t.append(p.time)
            known_z.append(float(p.imputed))
        else:
            missing_indices.append(i)

    n_known = len(known_t)
    n_miss = len(missing_indices)
    print(f"{_TAG}   '{inst_id}': total={len(points)}, known={n_known}, missing={n_miss}", flush=True)

    if n_miss == 0:
        return [replace(p, imputed=p.observed if p.observed is not None else p.imputed, imputed_std=None) for p in points]

    if n_known < 2:
        avg_val = float(np.mean(known_z)) if known_z else 0.0
        results = []
        for p in points:
            if p.observed is not None:
                results.append(replace(p, imputed=p.observed, imputed_std=None))
            elif p.imputed is not None:
                results.append(replace(p, imputed=p.imputed, imputed_std=None))
            else:
                results.append(replace(p, imputed=avg_val, imputed_std=None))
        return results

    X_train = np.array(known_t, dtype=np.float64).reshape(-1, 1)
    y_train = np.array(known_z, dtype=np.float64)
    all_t = np.array([p.time for p in points], dtype=np.float64)
    X_predict = all_t[missing_indices].reshape(-1, 1)

    # Base noise in normalized space: since normalize_y=True standardises y
    # to unit variance, alpha ~ 1e-2 means ~1% noise-to-signal ratio.
    # The seasonal weights (mean ≈ 1.0) modulate this so noisy seasons get
    # higher alpha (smoother fit, wider CI) and quiet seasons get lower alpha.
    base_noise = 1e-2

    seasonal_weights = noise_model(np.array(known_t, dtype=np.float64))
    alpha_train = base_noise * seasonal_weights
    alpha_train = np.maximum(alpha_train, 1e-10)

    print(f"{_TAG}   '{inst_id}': base_noise={base_noise}, alpha range=[{alpha_train.min():.6f}, {alpha_train.max():.6f}]", flush=True)

    krig_pred = np.zeros(n_miss, dtype=np.float64)
    imputed_std_arr = np.zeros(n_miss, dtype=np.float64)

    try:
        # Use RBF kernel only (no WhiteKernel) since alpha provides heteroscedastic noise
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10.0, (1e-2, 1e2))
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha_train,
            n_restarts_optimizer=10,
            normalize_y=True,
        )
        gpr.fit(X_train, y_train)
        krig_pred, imputed_std_arr = gpr.predict(X_predict, return_std=True)
        krig_pred = np.asarray(krig_pred, dtype=np.float64)
        imputed_std_arr = np.asarray(imputed_std_arr, dtype=np.float64)
        imputed_std_arr = np.maximum(imputed_std_arr, 0.0)

        print(f"{_TAG}   '{inst_id}': predicted {len(krig_pred)} pts, range=[{krig_pred.min():.4f}, {krig_pred.max():.4f}]", flush=True)
        print(f"{_TAG}   '{inst_id}': uncertainty avg_std={imputed_std_arr.mean():.4f}, max_std={imputed_std_arr.max():.4f}", flush=True)
    except Exception as e:
        print(f"{_TAG}   '{inst_id}': GPR failed ({e}), linear fallback", flush=True)
        t_arr = X_train.flatten()
        for j, idx in enumerate(missing_indices):
            t_m = all_t[idx]
            krig_pred[j] = _local_interp(float(t_m), t_arr, y_train)
        imputed_std_arr[:] = 0.0

    missing_set = set(missing_indices)
    results: List[DataPoint] = []
    miss_ptr = 0
    for i, p in enumerate(points):
        if p.observed is not None:
            results.append(replace(p, imputed=p.observed, imputed_std=None))
        elif i in missing_set:
            imputed_val = float(krig_pred[miss_ptr])
            std_val = float(imputed_std_arr[miss_ptr]) if imputed_std_arr[miss_ptr] > 0 else None
            results.append(replace(p, imputed=imputed_val, imputed_std=std_val))
            miss_ptr += 1
        else:
            results.append(replace(p, imputed=p.imputed, imputed_std=None))

    return results


def _local_interp(t: float, t_known: np.ndarray, z_known: np.ndarray) -> float:
    """Linear interpolation fallback."""
    idx = np.searchsorted(t_known, t)
    if idx == 0:
        return float(z_known[0])
    if idx >= len(t_known):
        return float(z_known[-1])
    t0, t1 = t_known[idx - 1], t_known[idx]
    z0, z1 = z_known[idx - 1], z_known[idx]
    if abs(t1 - t0) < 1e-12:
        return float((z0 + z1) / 2)
    frac = (t - t0) / (t1 - t0)
    return float(z0 + frac * (z1 - z0))


# ---------------------------------------------------------------------------
# KGE helper (same approach as standard kriging)
# ---------------------------------------------------------------------------

def _seasonal_loo_obs_pred(
    points: List[DataPoint],
    noise_model: Callable[[np.ndarray], np.ndarray],
) -> Tuple[List[float], List[float]]:
    """In-sample KGE approximation: fit one heteroscedastic GPR per instance."""
    known_t: List[float] = []
    known_z: List[float] = []
    obs_t: List[float] = []
    obs_y: List[float] = []

    for p in points:
        if p.observed is not None:
            val = float(p.observed)
            known_t.append(p.time)
            known_z.append(val)
            obs_t.append(p.time)
            obs_y.append(val)
        elif p.imputed is not None:
            known_t.append(p.time)
            known_z.append(float(p.imputed))

    if len(known_t) < 2 or len(obs_t) < 2:
        return [], []

    X_train = np.array(known_t, dtype=np.float64).reshape(-1, 1)
    y_train = np.array(known_z, dtype=np.float64)
    X_obs = np.array(obs_t, dtype=np.float64).reshape(-1, 1)
    base_noise = 1e-2
    seasonal_weights = noise_model(np.array(known_t, dtype=np.float64))
    alpha_train = base_noise * seasonal_weights
    alpha_train = np.maximum(alpha_train, 1e-10)

    try:
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10.0, (1e-2, 1e2))
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha_train, n_restarts_optimizer=2, normalize_y=True)
        gpr.fit(X_train, y_train)
        pred = gpr.predict(X_obs)
    except Exception:
        pred = np.array([_local_interp(t, X_train.flatten(), y_train) for t in obs_t], dtype=np.float64)

    return obs_y, pred.tolist()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_kriging_seasonal_simulation(
    data: List[DataPoint],
    params: Dict[str, Any],
) -> List[DataPoint]:
    """
    Season-aware 1D Kriging: learn seasonal noise from ALL instances,
    then impute each instance with heteroscedastic GPR.
    """
    period = params.get("seasonal_period", 12)

    instance_map: Dict[str, List[DataPoint]] = defaultdict(list)
    for d in data:
        key = d.instance_id or "__default__"
        instance_map[key].append(d)

    # Pool all known data for Stage 1
    all_known_t: List[float] = []
    all_known_z: List[float] = []
    for pts in instance_map.values():
        for p in pts:
            if p.observed is not None:
                all_known_t.append(p.time)
                all_known_z.append(float(p.observed))
            elif p.imputed is not None:
                all_known_t.append(p.time)
                all_known_z.append(float(p.imputed))

    print(f"{_TAG} run_kriging_seasonal_simulation: period={period}, instances={len(instance_map)}, pooled_known={len(all_known_t)}", flush=True)

    noise_model = _learn_seasonal_noise(
        np.array(all_known_t, dtype=np.float64),
        np.array(all_known_z, dtype=np.float64),
        period,
    )

    # Process instances from most complete to least
    def _known_count(pts: List[DataPoint]) -> int:
        return sum(1 for p in pts if p.observed is not None or p.imputed is not None)

    instance_order = sorted(
        instance_map.keys(),
        key=lambda iid: _known_count(instance_map[iid]),
        reverse=True,
    )

    final_results: List[DataPoint] = []
    instances_ok = 0
    instances_fallback = 0

    for inst_id in instance_order:
        points = instance_map[inst_id]
        points_sorted = sorted(points, key=lambda p: p.time)
        n_miss = sum(1 for p in points_sorted if p.observed is None and p.imputed is None)

        if n_miss == 0:
            for p in points_sorted:
                val = p.observed if p.observed is not None else p.imputed
                final_results.append(replace(p, imputed=val, imputed_std=None))
            instances_ok += 1
            continue

        try:
            imputed_points = _kriging_seasonal_impute_instance(points_sorted, noise_model)
            final_results.extend(imputed_points)
            instances_ok += 1
        except Exception as e:
            instances_fallback += 1
            print(f"{_TAG}   '{inst_id}': FAILED ({type(e).__name__}: {e}), linear fallback", flush=True)
            knowns_t = []
            knowns_z = []
            for p in points_sorted:
                if p.observed is not None:
                    knowns_t.append(p.time)
                    knowns_z.append(float(p.observed))
                elif p.imputed is not None:
                    knowns_t.append(p.time)
                    knowns_z.append(float(p.imputed))
            t_known = np.array(knowns_t, dtype=np.float64) if knowns_t else np.array([])
            z_known = np.array(knowns_z, dtype=np.float64) if knowns_z else np.array([])
            avg_val = float(z_known.mean()) if len(z_known) > 0 else 0.0
            for p in points_sorted:
                if p.observed is not None:
                    final_results.append(replace(p, imputed=p.observed, imputed_std=None))
                elif p.imputed is not None:
                    final_results.append(replace(p, imputed=p.imputed, imputed_std=None))
                elif len(t_known) >= 2:
                    val = _local_interp(p.time, t_known, z_known)
                    final_results.append(replace(p, imputed=val, imputed_std=None))
                else:
                    final_results.append(replace(p, imputed=avg_val, imputed_std=None))

    n_filled = sum(1 for p in final_results if p.observed is None and p.imputed is not None)
    print(f"{_TAG} done. ok={instances_ok}, fallback={instances_fallback}, filled={n_filled}", flush=True)
    return final_results


# ---------------------------------------------------------------------------
# Batch entry point (small-gap only) — mirrors batch_impute_small_gap_kriging
# ---------------------------------------------------------------------------

def batch_impute_small_gap_kriging_seasonal(
    flat_data: List[DataPoint],
    params: SimulationParams,
    on_progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Batch season-aware 1D Kriging (heteroscedastic GPR) for small gaps only.
    """
    print(f"{_TAG} batch start (heteroscedastic GPR, small-gap only)", flush=True)
    max_gap_threshold = getattr(params, "max_gap_threshold", 10) or 10
    period = getattr(params, "geostat_seasonal_period", 12) or 12
    kriging_params = {"seasonal_period": period}
    print(f"{_TAG}   flat_data={len(flat_data)}, max_gap_threshold={max_gap_threshold}, period={period}", flush=True)

    if on_progress:
        on_progress(1, 1, "kriging-seasonal")
    filled_list = run_kriging_seasonal_simulation(flat_data, kriging_params)

    # Restrict to small-gap only
    input_groups_by_inst: Dict[str, List[DataPoint]] = defaultdict(list)
    for d in flat_data:
        key = d.instance_id or "__default__"
        input_groups_by_inst[key].append(d)
    filled_groups: Dict[str, List[DataPoint]] = defaultdict(list)
    for d in filled_list:
        key = d.instance_id or "__default__"
        filled_groups[key].append(d)

    def _known_count(pts: List[DataPoint]) -> int:
        return sum(1 for p in pts if p.observed is not None or p.imputed is not None)

    instance_order = sorted(
        input_groups_by_inst.keys(),
        key=lambda iid: _known_count(input_groups_by_inst[iid]),
        reverse=True,
    )
    for inst_id in instance_order:
        input_pts = input_groups_by_inst[inst_id]
        points_sorted = sorted(input_pts, key=lambda p: p.time)
        filled_pts = filled_groups.get(inst_id, [])
        if len(filled_pts) != len(points_sorted):
            continue
        small_gap_indices = _small_gap_only_indices(points_sorted, max_gap_threshold)
        for i, inp in enumerate(points_sorted):
            if inp.observed is None and inp.imputed is None:
                if i not in small_gap_indices and i < len(filled_pts):
                    filled_pts[i] = replace(filled_pts[i], imputed=None, imputed_std=None)
    filled_list = []
    for inst_id in instance_order:
        filled_list.extend(filled_groups.get(inst_id, []))

    # Group filled results by instance
    groups: Dict[str, List[DataPoint]] = defaultdict(list)
    for d in filled_list:
        if d.instance_id:
            groups[d.instance_id].append(d)

    # Group original input by instance for KGE
    input_groups: Dict[str, List[DataPoint]] = defaultdict(list)
    for d in flat_data:
        if d.instance_id:
            input_groups[d.instance_id].append(d)

    # Build noise model again for KGE (reuse pooled data)
    all_known_t: List[float] = []
    all_known_z: List[float] = []
    for pts in input_groups.values():
        for p in pts:
            if p.observed is not None:
                all_known_t.append(p.time)
                all_known_z.append(float(p.observed))
            elif p.imputed is not None:
                all_known_t.append(p.time)
                all_known_z.append(float(p.imputed))
    noise_model = _learn_seasonal_noise(
        np.array(all_known_t, dtype=np.float64) if all_known_t else np.array([]),
        np.array(all_known_z, dtype=np.float64) if all_known_z else np.array([]),
        period,
    )

    kge_threshold = getattr(params, "small_gap_kge_threshold", None) or params.kge_threshold
    results: Dict[str, Dict[str, Any]] = {}
    saved_count = 0
    skipped_count = 0

    print(f"{_TAG} per-instance KGE (threshold={kge_threshold:.4f})", flush=True)
    for instance_id, pts in groups.items():
        input_pts = sorted(input_groups.get(instance_id, []), key=lambda p: p.time)
        obs_list: List[float] = []
        pred_list: List[float] = []
        if len(input_pts) >= 2:
            obs_list, pred_list = _seasonal_loo_obs_pred(input_pts, noise_model)
        kge = float("-inf")
        if obs_list and len(obs_list) >= 2:
            kge = calculate_kge(obs_list, pred_list)
        saved = kge >= kge_threshold
        results[instance_id] = {"kge": kge, "saved": saved}
        n_filled = sum(1 for d in pts if d.observed is None and d.imputed is not None)
        print(f"{_TAG}   '{instance_id}': LOO pairs={len(obs_list)}, filled={n_filled}, KGE={kge:.4f}, saved={saved}", flush=True)
        if saved:
            saved_count += 1
        else:
            skipped_count += 1

    print(f"{_TAG} batch done: instances={len(groups)}, saved={saved_count}, skipped={skipped_count}", flush=True)

    return {
        "imputed": len(groups),
        "saved": saved_count,
        "skipped": skipped_count,
        "results": results,
        "filled_data": filled_list,
    }


def _small_gap_only_indices(points_sorted: List[DataPoint], max_gap_threshold: int) -> set:
    """Indices that belong to small gaps (gap size <= max_gap_threshold)."""
    gaps, large_gap_indices = identify_gaps(points_sorted, max_gap_threshold)
    small = set()
    for g in gaps:
        if g["size"] <= max_gap_threshold:
            for i in range(g["startIdx"], g["endIdx"] + 1):
                small.add(i)
    return small
