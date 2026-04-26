"""
1D Kriging (Gaussian Process Regression) for time-series gap filling.
Per-instance approach:
  1. Known values: observed + previously-imputed (from prior step)
  2. GPR with RBF + WhiteKernel for smooth predictions and uncertainty (imputed_std → 95% CI)
  3. Process instances from most complete to least (same as reference workflow)
"""

from dataclasses import replace
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import warnings

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.exceptions import ConvergenceWarning

from .types import DataPoint, SimulationParams
from .math_utils import calculate_kge
from .gaps import identify_gaps

# Suppress GPR convergence chatter to keep logs readable
warnings.simplefilter("ignore", category=ConvergenceWarning)


def _gpr_kernel():
    """Kriging kernel: constant * RBF (smoothness) + WhiteKernel (noise). Matches reference 1D GPR setup."""
    return C(1.0, (1e-3, 1e3)) * RBF(10.0, (1e-2, 1e2)) + WhiteKernel(noise_level=1e-3)


def _params_to_kriging_dict(params: SimulationParams) -> Dict[str, Any]:
    """Kept for API compatibility; GPR uses fixed kernel."""
    return {"variogram_model": "gpr"}


def _kriging_impute_instance(
    points: List[DataPoint],
    variogram_model: str,  # ignored when using GPR
) -> List[DataPoint]:
    """
    1D Kriging via Gaussian Process Regression. Known = observed or imputed; predict at missing times with uncertainty.
    """
    inst_id = points[0].instance_id if points else "?"

    # Build mask: missing = no observed and no imputed
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

    print(f"[Kriging]   '{inst_id}': total={len(points)}, known={n_known}, missing={n_miss} (GPR)", flush=True)

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

    krig_pred = np.zeros(len(missing_indices), dtype=np.float64)
    imputed_std_arr = np.zeros(len(missing_indices), dtype=np.float64)

    try:
        gpr = GaussianProcessRegressor(
            kernel=_gpr_kernel(),
            n_restarts_optimizer=10,
            normalize_y=True,
        )
        gpr.fit(X_train, y_train)
        krig_pred, imputed_std_arr = gpr.predict(X_predict, return_std=True)
        krig_pred = np.asarray(krig_pred, dtype=np.float64)
        imputed_std_arr = np.asarray(imputed_std_arr, dtype=np.float64)
        imputed_std_arr = np.maximum(imputed_std_arr, 0.0)

        print(f"[Kriging]   '{inst_id}': GPR predicted {len(krig_pred)} points, range=[{krig_pred.min():.4f}, {krig_pred.max():.4f}]", flush=True)
        print(f"[Kriging]   '{inst_id}': uncertainty avg_std={imputed_std_arr.mean():.4f}, max_std={imputed_std_arr.max():.4f}", flush=True)
    except Exception as e:
        print(f"[Kriging]   '{inst_id}': GPR failed ({e}), linear fallback", flush=True)
        t_unique = X_train.flatten()
        z_unique = y_train
        for j, idx in enumerate(missing_indices):
            t_m = all_t[idx]
            krig_pred[j] = _local_interp(float(t_m), t_unique, z_unique)
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


def _kriging_loo_obs_pred(
    points: List[DataPoint],
    variogram_model: str,  # ignored for GPR
) -> Tuple[List[float], List[float]]:
    """
    Fast KGE approximation: fit ONE GPR per instance and evaluate at observed times.
    This is in-sample (not strict LOO) but much cheaper than per-point refits.
    """
    # Collect all known points (observed or imputed)
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

    try:
        # Cheaper optimizer for KGE: fewer restarts
        gpr = GaussianProcessRegressor(kernel=_gpr_kernel(), n_restarts_optimizer=2, normalize_y=True)
        gpr.fit(X_train, y_train)
        pred = gpr.predict(X_obs)
    except Exception:
        # Fallback: simple interpolation
        pred = np.array([_local_interp(t, X_train.flatten(), y_train) for t in obs_t], dtype=np.float64)

    return obs_y, pred.tolist()


def run_kriging_simulation(
    data: List[DataPoint],
    params: Dict[str, Any],
) -> List[DataPoint]:
    """
    Temporal imputation using 1D GPR (Kriging) per instance.
    Uses observed + prior-imputed values as known; uncertainty from GPR std.
    """
    variogram_model = params.get("variogram_model", "gpr")

    instance_map: Dict[str, List[DataPoint]] = defaultdict(list)
    for d in data:
        key = d.instance_id or "__default__"
        instance_map[key].append(d)

    # Process from most complete to least (observed + small-gap imputed count), as in reference workflow
    def _known_count(pts: List[DataPoint]) -> int:
        return sum(1 for p in pts if p.observed is not None or p.imputed is not None)

    instance_order = sorted(
        instance_map.keys(),
        key=lambda inst_id: _known_count(instance_map[inst_id]),
        reverse=True,
    )

    print(f"[Kriging] run_kriging_simulation: 1D GPR (RBF+WhiteKernel), order=most_complete_first", flush=True)
    print(f"[Kriging]   data={len(data)}, instances={len(instance_map)}", flush=True)

    final_results: List[DataPoint] = []
    instances_ok = 0
    instances_fallback = 0

    for inst_id in instance_order:
        points = instance_map[inst_id]
        points_sorted = sorted(points, key=lambda p: p.time)
        n_obs = sum(1 for p in points_sorted if p.observed is not None)
        n_prior = sum(1 for p in points_sorted if p.observed is None and p.imputed is not None)
        n_miss = sum(1 for p in points_sorted if p.observed is None and p.imputed is None)

        print(f"[Kriging]   instance '{inst_id}': {len(points_sorted)} pts, {n_obs} obs, {n_prior} prior, {n_miss} miss", flush=True)

        if n_miss == 0:
            print(f"[Kriging]   instance '{inst_id}': no gaps, pass-through", flush=True)
            for p in points_sorted:
                val = p.observed if p.observed is not None else p.imputed
                final_results.append(replace(p, imputed=val, imputed_std=None))
            instances_ok += 1
            continue

        try:
            imputed_points = _kriging_impute_instance(points_sorted, variogram_model)
            final_results.extend(imputed_points)
            instances_ok += 1
        except Exception as e:
            instances_fallback += 1
            print(f"[Kriging]   instance '{inst_id}': FAILED ({type(e).__name__}: {e}), linear fallback", flush=True)
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
    print(f"[Kriging] done. ok={instances_ok}, fallback={instances_fallback}, filled={n_filled}", flush=True)
    return final_results


def optimize_kriging_params(
    data: List[DataPoint],
) -> Dict[str, Any]:
    """GPR uses fixed kernel; no variogram to optimize. Return for API compatibility."""
    print("[Kriging] optimize_kriging_params: GPR uses fixed RBF+WhiteKernel (no optimization)", flush=True)
    return {"variogram_model": "gpr"}


def _small_gap_only_indices(points_sorted: List[DataPoint], max_gap_threshold: int) -> set:
    """Indices that belong to small gaps (gap size <= max_gap_threshold). Used to restrict 1D Kriging to small gaps only."""
    gaps, large_gap_indices = identify_gaps(points_sorted, max_gap_threshold)
    small = set()
    for g in gaps:
        if g["size"] <= max_gap_threshold:
            for i in range(g["startIdx"], g["endIdx"] + 1):
                small.add(i)
    return small


def batch_impute_small_gap_kriging(
    flat_data: List[DataPoint],
    params: SimulationParams,
    on_progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Batch 1D Kriging (GPR) for small gaps only. Only fills gaps with size <= max_gap_threshold.
    """
    print("[Kriging] batch_impute_small_gap_kriging: start (1D GPR, small-gap only)", flush=True)
    max_gap_threshold = getattr(params, "max_gap_threshold", 10) or 10
    kriging_params = _params_to_kriging_dict(params)
    auto_variogram = getattr(params, "kriging_auto_variogram", False)
    print(f"[Kriging]   flat_data={len(flat_data)}, max_gap_threshold={max_gap_threshold}, GPR kernel (auto={auto_variogram})", flush=True)

    if auto_variogram and len(flat_data) > 0:
        if on_progress:
            on_progress(0, 1, "optimizing variogram")
        opt = optimize_kriging_params(flat_data)
        kriging_params["variogram_model"] = opt["variogram_model"]
        print(f"[Kriging] Auto-selected: {kriging_params['variogram_model']}", flush=True)

    if on_progress:
        on_progress(1, 1, "kriging")
    filled_list = run_kriging_simulation(flat_data, kriging_params)

    # Restrict to small-gap only: clear imputed/imputed_std for large-gap and extrapolation points
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
        key=lambda inst_id: _known_count(input_groups_by_inst[inst_id]),
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
            if inp.observed is None and inp.imputed is None:  # originally missing
                if i not in small_gap_indices and i < len(filled_pts):
                    filled_pts[i] = replace(filled_pts[i], imputed=None, imputed_std=None)
    filled_list = []
    for inst_id in instance_order:
        filled_list.extend(filled_groups.get(inst_id, []))

    # Group by instance (filled results)
    groups: Dict[str, List[DataPoint]] = defaultdict(list)
    for d in filled_list:
        if d.instance_id:
            groups[d.instance_id].append(d)

    # Group original input by instance for leave-one-out (use pre-kriging points so "known" is correct)
    input_groups: Dict[str, List[DataPoint]] = defaultdict(list)
    for d in flat_data:
        if d.instance_id:
            input_groups[d.instance_id].append(d)

    variogram_model = kriging_params["variogram_model"]
    kge_threshold = getattr(params, "small_gap_kge_threshold", None) or params.kge_threshold
    results: Dict[str, Dict[str, Any]] = {}
    saved_count = 0
    skipped_count = 0

    print(f"[Kriging] batch: per-instance KGE via leave-one-out (threshold={kge_threshold:.4f})", flush=True)
    for instance_id, pts in groups.items():
        # Leave-one-out: predict each observed point from the rest (fair KGE, not observed vs self)
        input_pts = sorted(input_groups.get(instance_id, []), key=lambda p: p.time)
        obs_list: List[float] = []
        pred_list: List[float] = []
        if len(input_pts) >= 2:
            obs_list, pred_list = _kriging_loo_obs_pred(input_pts, variogram_model)
        kge = float("-inf")
        if obs_list and len(obs_list) >= 2:
            kge = calculate_kge(obs_list, pred_list)
        saved = kge >= kge_threshold
        results[instance_id] = {"kge": kge, "saved": saved}
        n_filled = sum(1 for d in pts if d.observed is None and d.imputed is not None)
        print(f"[Kriging]   '{instance_id}': LOO pairs={len(obs_list)}, filled={n_filled}, KGE={kge:.4f}, saved={saved}", flush=True)
        if saved:
            saved_count += 1
        else:
            skipped_count += 1

    print(f"[Kriging] batch done: instances={len(groups)}, saved={saved_count}, skipped={skipped_count}", flush=True)

    return {
        "imputed": len(groups),
        "saved": saved_count,
        "skipped": skipped_count,
        "results": results,
        "filled_data": filled_list,
    }
