"""
Batch small-gap imputation for LNNcfc, GLNN, GAIN, and Conditional GAN.
Same return shape as small_gap_batch / geostat_impute.
"""

from typing import List, Dict, Any, Optional, Callable
from collections import defaultdict
from dataclasses import replace as dc_replace

import numpy as np

from .types import DataPoint, SimulationParams
from .gaps import identify_gaps
from .math_utils import calculate_kge
from .lnn_variants import (
    run_lnncfc_simulation,
    run_glnn_simulation,
    optimize_lnncfc_params,
    optimize_glnn_params,
)
from .gan_impute import (
    run_gain_imputation,
    run_condgan_imputation,
    optimize_gain_hyperparams,
    optimize_condgan_hyperparams,
)
from .LNN_kriging import run_lnn_kriging_simulation
from .lnn_kriging_readout import run_lnn_kriging_readout_simulation
from .graph_lnn import build_spatial_graph, run_graph_lnn_simulation
from .lnn_core_aux_placeholder import run_lnn_simulation as run_lnn_aux_placeholder_simulation
from .lnn_core_aux_placeholder import optimize_lnn_params as optimize_lnn_aux_placeholder_params
from .lnn_core_aux_placeholder_cfc import run_lnn_simulation as run_lnn_aux_placeholder_cfc_simulation
from .lnn_core_aux_placeholder_cfc import optimize_lnn_params as optimize_lnn_aux_placeholder_cfc_params
from .lnn_core_aux_placeholder_cfc_enhanced import run_lnn_simulation as run_lnn_aux_placeholder_cfc_enhanced_simulation
from .lnn_core_aux_placeholder_cfc_enhanced import optimize_lnn_params as optimize_lnn_aux_placeholder_cfc_enhanced_params
from .siren_lnn_cfc import run_siren_lnn_cfc_simulation, optimize_siren_lnn_cfc_params


def _protected_missing_indices(sorted_data: List[DataPoint], max_gap_threshold: int) -> set[int]:
    """
    Missing points that must NOT be filled during the small-gap phase:
    - extrapolation points before first observed
    - extrapolation points after last observed
    - interior gaps larger than max_gap_threshold
    """
    protected: set[int] = set()
    observed_indices = [i for i, d in enumerate(sorted_data) if d.observed is not None]
    if not observed_indices:
        return set(range(len(sorted_data)))

    first_obs = observed_indices[0]
    last_obs = observed_indices[-1]

    for i, d in enumerate(sorted_data):
        if d.observed is None and (i < first_obs or i > last_obs):
            protected.add(i)

    gaps, large_gap_indices = identify_gaps(sorted_data, max_gap_threshold)
    protected.update(large_gap_indices)
    return protected


def _batch_small_gap_template(
    flat_data: List[DataPoint],
    params: SimulationParams,
    on_progress: Optional[Any],
    run_per_instance,
    algo_name: str = "",
    get_neighbor_weights: Optional[Callable[[str], List[float]]] = None,
) -> Dict[str, Any]:
    """Shared loop: group by instance, identify gaps, run_per_instance(...) -> final_result list, then KGE and counts. Logs per instance to console.
    When params.small_gap_use_neighbors is True, passes neighbor series (other instances' sorted data) to run_per_instance for algorithms that use them.
    When get_neighbor_weights is provided, passes weights (same order as neighbor_series_list) as 6th argument to run_per_instance."""
    groups: Dict[str, List[DataPoint]] = defaultdict(list)
    for d in flat_data:
        if d.instance_id:
            groups[d.instance_id].append(d)
    instance_ids = list(groups.keys())
    total = len(instance_ids)
    prefix = f"[{algo_name}] " if algo_name else ""
    use_neighbors = getattr(params, "small_gap_use_neighbors", False)
    if use_neighbors:
        print(f"{prefix}use_neighbors=True: merging other instances' series into auxiliaries", flush=True)
    sorted_by_instance: Dict[str, List[DataPoint]] = {
        iid: sorted(groups[iid], key=lambda d: d.time) for iid in instance_ids
    }
    results: Dict[str, Dict[str, Any]] = {}
    imputed_count = 0
    saved_count = 0
    skipped_count = 0
    filled_data: List[DataPoint] = []
    max_gap_threshold = params.max_gap_threshold
    kge_threshold = getattr(params, "small_gap_kge_threshold", None) or params.kge_threshold

    for i, instance_id in enumerate(instance_ids):
        if on_progress:
            on_progress(i + 1, total, instance_id)
        print(f"{prefix}Instance {i + 1}/{total}: {instance_id}", end="", flush=True)
        sorted_data = sorted_by_instance[instance_id]
        gaps, _ = identify_gaps(sorted_data, max_gap_threshold)
        protected_indices = _protected_missing_indices(sorted_data, max_gap_threshold)
        has_small_gaps = any(g["size"] <= max_gap_threshold for g in gaps)
        if not has_small_gaps:
            skipped_count += 1
            results[instance_id] = {"kge": float("-inf"), "saved": False}
            filled_data.extend(sorted_data)
            print(f" -> skipped (no small gaps)", flush=True)
            continue

        neighbor_series_list = [sorted_by_instance[nid] for nid in instance_ids if nid != instance_id] if use_neighbors else []
        if get_neighbor_weights is not None:
            neighbor_weights = get_neighbor_weights(instance_id)
            final_result = run_per_instance(instance_id, sorted_data, protected_indices, params, neighbor_series_list, neighbor_weights)
        else:
            final_result = run_per_instance(instance_id, sorted_data, protected_indices, params, neighbor_series_list)
        if final_result is None:
            filled_data.extend(sorted_data)
            results[instance_id] = {"kge": float("-inf"), "saved": False}
            print(f" -> failed (no result)", flush=True)
            continue

        # GAIN/CondGAN may return (result_list, holdout_kge) so we use gap-filling quality, not reconstruction
        override_kge = None
        if isinstance(final_result, (list, tuple)) and len(final_result) == 2:
            a, b = final_result
            if isinstance(a, list) and isinstance(b, (int, float)):
                final_result = a
                override_kge = float(b)

        training_true = []
        training_pred = []
        for idx, r in enumerate(final_result):
            if idx not in protected_indices and r.observed is not None and r.imputed is not None:
                training_true.append(r.observed)
                training_pred.append(r.imputed)
        kge = float("-inf")
        if override_kge is not None:
            kge = override_kge
        elif training_true:
            kge = calculate_kge(training_true, training_pred)
        saved = kge >= kge_threshold
        results[instance_id] = {"kge": kge, "saved": saved}
        imputed_count += 1
        if saved:
            saved_count += 1
        filled_data.extend(final_result)
        kge_str = f"{kge:.4f}" if kge > float("-inf") else "N/A"
        print(f" -> KGE={kge_str}, saved={saved}", flush=True)

    return {
        "imputed": imputed_count,
        "saved": saved_count,
        "skipped": skipped_count,
        "results": results,
        "filled_data": filled_data,
    }


def batch_impute_small_gap_lnncfc(
    flat_data: List[DataPoint],
    params: SimulationParams,
    on_progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """Batch small-gap imputation using LNNcfc (Closed-form Continuous-time) with autotuned hyperparameters."""
    def run_per_instance(_id, sorted_data, large_gap_indices, p, neighbor_series_list=None):
        use_auxiliary = getattr(p, "small_gap_use_auxiliary", True)
        data_for_imputation = []
        for idx, d in enumerate(sorted_data):
            obs = d.observed if idx not in large_gap_indices else None
            aux = list(d.auxiliaries) if (d.auxiliaries and use_auxiliary) else []
            data_for_imputation.append(DataPoint(
                time=d.time, observed=obs, instance_id=d.instance_id,
                auxiliaries=aux if aux else None, is_masked=idx in large_gap_indices,
            ))
        best_params = optimize_lnncfc_params(data_for_imputation, p, mode="projection")
        rng = np.random.default_rng(0)
        result = run_lnncfc_simulation(data_for_imputation, best_params, rng=rng)
        out = []
        for idx, d in enumerate(sorted_data):
            if idx in large_gap_indices:
                out.append(DataPoint(
                    time=d.time, observed=None, instance_id=d.instance_id,
                    auxiliaries=d.auxiliaries, imputed=None, is_masked=True,
                    latitude=d.latitude, longitude=d.longitude,
                    date_label=d.date_label, timestamp=d.timestamp,
                ))
            elif d.observed is not None:
                # Set imputed from model prediction so KGE (observed vs imputed) can be computed
                pred_val = result[idx].imputed if idx < len(result) else None
                out.append(DataPoint(
                    time=d.time, observed=d.observed, instance_id=d.instance_id,
                    auxiliaries=d.auxiliaries, imputed=pred_val, is_masked=False,
                    latitude=d.latitude, longitude=d.longitude,
                    date_label=d.date_label, timestamp=d.timestamp,
                ))
            else:
                imputed_val = result[idx].imputed if idx < len(result) else None
                out.append(DataPoint(
                    time=d.time, observed=None, instance_id=d.instance_id,
                    auxiliaries=d.auxiliaries, imputed=imputed_val, is_masked=False,
                    latitude=d.latitude, longitude=d.longitude,
                    date_label=d.date_label, timestamp=d.timestamp,
                ))
        return out

    return _batch_small_gap_template(flat_data, params, on_progress, run_per_instance, algo_name="LNNcfc")


def batch_impute_small_gap_lnn_aux_placeholder(
    flat_data: List[DataPoint],
    params: SimulationParams,
    on_progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """Batch small-gap imputation using LNN with aux-based placeholders. Auxiliary data is required per instance; instances without aux are skipped."""
    def run_per_instance(_id, sorted_data, large_gap_indices, p, neighbor_series_list=None):
        use_auxiliary = getattr(p, "small_gap_use_auxiliary", True)
        has_aux = any(
            d.auxiliaries and len(d.auxiliaries) > 0
            for d in sorted_data
        )
        if not has_aux or not use_auxiliary:
            print(f"  [LNN aux placeholder] Instance {_id} -> skipped (auxiliary data required)", flush=True)
            return None
        data_for_imputation = []
        for idx, d in enumerate(sorted_data):
            obs = d.observed if idx not in large_gap_indices else None
            aux = list(d.auxiliaries) if (d.auxiliaries and use_auxiliary) else []
            data_for_imputation.append(DataPoint(
                time=d.time, observed=obs, instance_id=d.instance_id,
                auxiliaries=aux if aux else None, is_masked=idx in large_gap_indices,
                latitude=d.latitude, longitude=d.longitude,
            ))
        best_params = optimize_lnn_aux_placeholder_params(data_for_imputation, p, mode="projection")
        rng = np.random.default_rng(0)
        result = run_lnn_aux_placeholder_simulation(data_for_imputation, best_params, rng=rng)
        out = []
        for idx, d in enumerate(sorted_data):
            r = result[idx] if idx < len(result) else None
            imputed_val = r.imputed if r else None
            imputed_std_val = getattr(r, "imputed_std", None) if r else None
            if idx in large_gap_indices:
                out.append(DataPoint(
                    time=d.time, observed=None, instance_id=d.instance_id,
                    auxiliaries=d.auxiliaries, imputed=None, imputed_std=None, is_masked=True,
                    latitude=d.latitude, longitude=d.longitude,
                    date_label=d.date_label, timestamp=d.timestamp,
                ))
            elif d.observed is not None:
                out.append(DataPoint(
                    time=d.time, observed=d.observed, instance_id=d.instance_id,
                    auxiliaries=d.auxiliaries, imputed=imputed_val, imputed_std=imputed_std_val, is_masked=False,
                    latitude=d.latitude, longitude=d.longitude,
                    date_label=d.date_label, timestamp=d.timestamp,
                ))
            else:
                out.append(DataPoint(
                    time=d.time, observed=None, instance_id=d.instance_id,
                    auxiliaries=d.auxiliaries, imputed=imputed_val, imputed_std=imputed_std_val, is_masked=False,
                    latitude=d.latitude, longitude=d.longitude,
                    date_label=d.date_label, timestamp=d.timestamp,
                ))
        return out

    return _batch_small_gap_template(flat_data, params, on_progress, run_per_instance, algo_name="LNN (aux placeholders)")


def batch_impute_small_gap_lnn_aux_placeholder_cfc(
    flat_data: List[DataPoint],
    params: SimulationParams,
    on_progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """Batch small-gap imputation using LNN CFC with aux-based placeholders. Uses closed-form continuous-time dynamics instead of ODE. Auxiliary data is required."""
    def run_per_instance(_id, sorted_data, large_gap_indices, p, neighbor_series_list=None):
        use_auxiliary = getattr(p, "small_gap_use_auxiliary", True)
        use_neighbors = getattr(p, "small_gap_use_neighbors", False)
        has_aux = any(
            d.auxiliaries and len(d.auxiliaries) > 0
            for d in sorted_data
        )
        if not has_aux or not use_auxiliary:
            print(f"  [LNN CFC aux placeholder] Instance {_id} -> skipped (auxiliary data required)", flush=True)
            return None
        neighbor_series_list = neighbor_series_list or []
        # Pre-index neighbor series by time for fast lookup
        neighbor_time_maps = []
        if use_neighbors and neighbor_series_list:
            for series in neighbor_series_list:
                tmap = {}
                for pt in series:
                    tmap[pt.time] = pt.imputed if pt.imputed is not None else (pt.observed if pt.observed is not None else 0.0)
                neighbor_time_maps.append(tmap)
            n_neighbors_used = len(neighbor_time_maps)
            print(f"  [LNN CFC aux placeholder] Merging {n_neighbors_used} neighbor series into auxiliaries", flush=True)
        data_for_imputation = []
        for idx, d in enumerate(sorted_data):
            obs = d.observed if idx not in large_gap_indices else None
            aux = list(d.auxiliaries) if (d.auxiliaries and use_auxiliary) else []
            # Append neighbor values as additional auxiliary features
            for tmap in neighbor_time_maps:
                aux.append(tmap.get(d.time, 0.0))
            data_for_imputation.append(DataPoint(
                time=d.time, observed=obs, instance_id=d.instance_id,
                auxiliaries=aux if aux else None, is_masked=idx in large_gap_indices,
                latitude=d.latitude, longitude=d.longitude,
            ))
        best_params = optimize_lnn_aux_placeholder_cfc_params(data_for_imputation, p, mode="projection")
        rng = np.random.default_rng(0)
        result = run_lnn_aux_placeholder_cfc_simulation(data_for_imputation, best_params, rng=rng)
        out = []
        for idx, d in enumerate(sorted_data):
            r = result[idx] if idx < len(result) else None
            imputed_val = r.imputed if r else None
            imputed_std_val = getattr(r, "imputed_std", None) if r else None
            if idx in large_gap_indices:
                out.append(DataPoint(
                    time=d.time, observed=None, instance_id=d.instance_id,
                    auxiliaries=d.auxiliaries, imputed=None, imputed_std=None, is_masked=True,
                    latitude=d.latitude, longitude=d.longitude,
                    date_label=d.date_label, timestamp=d.timestamp,
                ))
            elif d.observed is not None:
                out.append(DataPoint(
                    time=d.time, observed=d.observed, instance_id=d.instance_id,
                    auxiliaries=d.auxiliaries, imputed=imputed_val, imputed_std=imputed_std_val, is_masked=False,
                    latitude=d.latitude, longitude=d.longitude,
                    date_label=d.date_label, timestamp=d.timestamp,
                ))
            else:
                out.append(DataPoint(
                    time=d.time, observed=None, instance_id=d.instance_id,
                    auxiliaries=d.auxiliaries, imputed=imputed_val, imputed_std=imputed_std_val, is_masked=False,
                    latitude=d.latitude, longitude=d.longitude,
                    date_label=d.date_label, timestamp=d.timestamp,
                ))
        return out

    return _batch_small_gap_template(flat_data, params, on_progress, run_per_instance, algo_name="LNN CFC (aux placeholders)")


def batch_impute_small_gap_lnn_aux_placeholder_cfc_enhanced(
    flat_data: List[DataPoint],
    params: SimulationParams,
    on_progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """Batch small-gap imputation using Enhanced LNN CFC: bidirectional multi-scale + polynomial placeholder + anchor injection + ensemble."""
    def run_per_instance(_id, sorted_data, large_gap_indices, p, neighbor_series_list=None):
        use_auxiliary = getattr(p, "small_gap_use_auxiliary", True)
        has_aux = any(
            d.auxiliaries and len(d.auxiliaries) > 0
            for d in sorted_data
        )
        if not has_aux or not use_auxiliary:
            print(f"  [CFC-Enhanced] Instance {_id} -> skipped (auxiliary data required)", flush=True)
            return None
        data_for_imputation = []
        for idx, d in enumerate(sorted_data):
            obs = d.observed if idx not in large_gap_indices else None
            aux = list(d.auxiliaries) if (d.auxiliaries and use_auxiliary) else []
            data_for_imputation.append(DataPoint(
                time=d.time, observed=obs, instance_id=d.instance_id,
                auxiliaries=aux if aux else None, is_masked=idx in large_gap_indices,
                latitude=d.latitude, longitude=d.longitude,
            ))
        best_params = optimize_lnn_aux_placeholder_cfc_enhanced_params(data_for_imputation, p, mode="projection")
        rng = np.random.default_rng(0)
        result = run_lnn_aux_placeholder_cfc_enhanced_simulation(data_for_imputation, best_params, rng=rng)
        out = []
        for idx, d in enumerate(sorted_data):
            r = result[idx] if idx < len(result) else None
            imputed_val = r.imputed if r else None
            imputed_std_val = getattr(r, "imputed_std", None) if r else None
            if idx in large_gap_indices:
                out.append(DataPoint(
                    time=d.time, observed=None, instance_id=d.instance_id,
                    auxiliaries=d.auxiliaries, imputed=None, imputed_std=None, is_masked=True,
                    latitude=d.latitude, longitude=d.longitude,
                    date_label=d.date_label, timestamp=d.timestamp,
                ))
            elif d.observed is not None:
                out.append(DataPoint(
                    time=d.time, observed=d.observed, instance_id=d.instance_id,
                    auxiliaries=d.auxiliaries, imputed=imputed_val, imputed_std=imputed_std_val, is_masked=False,
                    latitude=d.latitude, longitude=d.longitude,
                    date_label=d.date_label, timestamp=d.timestamp,
                ))
            else:
                out.append(DataPoint(
                    time=d.time, observed=None, instance_id=d.instance_id,
                    auxiliaries=d.auxiliaries, imputed=imputed_val, imputed_std=imputed_std_val, is_masked=False,
                    latitude=d.latitude, longitude=d.longitude,
                    date_label=d.date_label, timestamp=d.timestamp,
                ))
        return out

    return _batch_small_gap_template(flat_data, params, on_progress, run_per_instance, algo_name="LNN CFC Enhanced (aux placeholders)")


def batch_impute_small_gap_siren_lnn_cfc(
    flat_data: List[DataPoint],
    params: SimulationParams,
    on_progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """Batch small-gap imputation using SIREN (INR) + LNN CFC with aux-based placeholders.
    SIREN fits a continuous sinusoidal representation per instance, predictions are appended as auxiliary.
    Auxiliary data is required."""
    def run_per_instance(_id, sorted_data, large_gap_indices, p, neighbor_series_list=None):
        use_auxiliary = getattr(p, "small_gap_use_auxiliary", True)
        has_aux = any(
            d.auxiliaries and len(d.auxiliaries) > 0
            for d in sorted_data
        )
        if not has_aux or not use_auxiliary:
            print(f"  [SIREN+CFC] Instance {_id} -> skipped (auxiliary data required)", flush=True)
            return None
        data_for_imputation = []
        for idx, d in enumerate(sorted_data):
            obs = d.observed if idx not in large_gap_indices else None
            aux = list(d.auxiliaries) if (d.auxiliaries and use_auxiliary) else []
            data_for_imputation.append(DataPoint(
                time=d.time, observed=obs, instance_id=d.instance_id,
                auxiliaries=aux if aux else None, is_masked=idx in large_gap_indices,
                latitude=d.latitude, longitude=d.longitude,
            ))
        best_params = optimize_siren_lnn_cfc_params(data_for_imputation, p, mode="projection")
        rng = np.random.default_rng(0)
        result = run_siren_lnn_cfc_simulation(data_for_imputation, best_params, rng=rng)
        out = []
        for idx, d in enumerate(sorted_data):
            r = result[idx] if idx < len(result) else None
            imputed_val = r.imputed if r else None
            imputed_std_val = getattr(r, "imputed_std", None) if r else None
            if idx in large_gap_indices:
                out.append(DataPoint(
                    time=d.time, observed=None, instance_id=d.instance_id,
                    auxiliaries=d.auxiliaries, imputed=None, imputed_std=None, is_masked=True,
                    latitude=d.latitude, longitude=d.longitude,
                    date_label=d.date_label, timestamp=d.timestamp,
                ))
            elif d.observed is not None:
                out.append(DataPoint(
                    time=d.time, observed=d.observed, instance_id=d.instance_id,
                    auxiliaries=d.auxiliaries, imputed=imputed_val, imputed_std=imputed_std_val, is_masked=False,
                    latitude=d.latitude, longitude=d.longitude,
                    date_label=d.date_label, timestamp=d.timestamp,
                ))
            else:
                out.append(DataPoint(
                    time=d.time, observed=None, instance_id=d.instance_id,
                    auxiliaries=d.auxiliaries, imputed=imputed_val, imputed_std=imputed_std_val, is_masked=False,
                    latitude=d.latitude, longitude=d.longitude,
                    date_label=d.date_label, timestamp=d.timestamp,
                ))
        return out

    return _batch_small_gap_template(flat_data, params, on_progress, run_per_instance, algo_name="SIREN + LNN CFC (aux placeholders)")


def batch_impute_small_gap_lnn_kriging(
    flat_data: List[DataPoint],
    params: SimulationParams,
    on_progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """Batch small-gap imputation using hybrid LNN + Residual Kriging (GPR on lat/lon). Respects small_gap_use_neighbors."""
    def run_per_instance(_id, sorted_data, large_gap_indices, p, neighbor_series_list=None):
        use_auxiliary = getattr(p, "small_gap_use_auxiliary", True)
        neighbor_series_list = neighbor_series_list or []
        data_for_imputation = []
        for idx, d in enumerate(sorted_data):
            obs = d.observed if idx not in large_gap_indices else None
            aux = list(d.auxiliaries) if (d.auxiliaries and use_auxiliary) else []
            for series in neighbor_series_list:
                match = next((pt for pt in series if abs(pt.time - d.time) < 1e-9), None)
                val = 0.0
                if match:
                    val = match.imputed if match.imputed is not None else (match.observed if match.observed is not None else 0.0)
                aux.append(val)
            data_for_imputation.append(DataPoint(
                time=d.time, observed=obs, instance_id=d.instance_id,
                auxiliaries=aux if aux else None, is_masked=idx in large_gap_indices,
                latitude=d.latitude, longitude=d.longitude,
            ))
        rng = np.random.default_rng(0)
        result = run_lnn_kriging_simulation(data_for_imputation, p, rng=rng)
        out = []
        for idx, d in enumerate(sorted_data):
            if idx in large_gap_indices:
                out.append(DataPoint(
                    time=d.time, observed=None, instance_id=d.instance_id,
                    auxiliaries=d.auxiliaries, imputed=None, is_masked=True,
                    latitude=d.latitude, longitude=d.longitude,
                ))
            elif d.observed is not None:
                pred_val = result[idx].imputed if idx < len(result) else None
                out.append(DataPoint(
                    time=d.time, observed=d.observed, instance_id=d.instance_id,
                    auxiliaries=d.auxiliaries, imputed=pred_val, is_masked=False,
                    latitude=d.latitude, longitude=d.longitude,
                ))
            else:
                imputed_val = result[idx].imputed if idx < len(result) else None
                out.append(DataPoint(
                    time=d.time, observed=None, instance_id=d.instance_id,
                    auxiliaries=d.auxiliaries, imputed=imputed_val, is_masked=False,
                    latitude=d.latitude, longitude=d.longitude,
                ))
        return out

    return _batch_small_gap_template(flat_data, params, on_progress, run_per_instance, algo_name="LNN+Kriging")


def batch_impute_small_gap_lnn_kriging_readout(
    flat_data: List[DataPoint],
    params: SimulationParams,
    on_progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """Batch small-gap imputation using LNN core + Kriging (GPR) readout. Provides confidence intervals (imputed_std)."""
    def run_per_instance(_id, sorted_data, large_gap_indices, p, neighbor_series_list=None):
        use_auxiliary = getattr(p, "small_gap_use_auxiliary", True)
        neighbor_series_list = neighbor_series_list or []
        data_for_imputation = []
        for idx, d in enumerate(sorted_data):
            obs = d.observed if idx not in large_gap_indices else None
            aux = list(d.auxiliaries) if (d.auxiliaries and use_auxiliary) else []
            for series in neighbor_series_list:
                match = next((pt for pt in series if abs(pt.time - d.time) < 1e-9), None)
                val = 0.0
                if match:
                    val = match.imputed if match.imputed is not None else (match.observed if match.observed is not None else 0.0)
                aux.append(val)
            data_for_imputation.append(DataPoint(
                time=d.time, observed=obs, instance_id=d.instance_id,
                auxiliaries=aux if aux else None, is_masked=idx in large_gap_indices,
                latitude=d.latitude, longitude=d.longitude,
            ))
        rng = np.random.default_rng(0)
        result = run_lnn_kriging_readout_simulation(data_for_imputation, p, rng=rng)
        out = []
        for idx, d in enumerate(sorted_data):
            r = result[idx] if idx < len(result) else None
            imputed_val = r.imputed if r else None
            imputed_std_val = r.imputed_std if r else None
            if idx in large_gap_indices:
                out.append(DataPoint(
                    time=d.time, observed=None, instance_id=d.instance_id,
                    auxiliaries=d.auxiliaries, imputed=imputed_val, imputed_std=imputed_std_val, is_masked=True,
                    latitude=d.latitude, longitude=d.longitude,
                ))
            elif d.observed is not None:
                out.append(DataPoint(
                    time=d.time, observed=d.observed, instance_id=d.instance_id,
                    auxiliaries=d.auxiliaries, imputed=imputed_val, imputed_std=imputed_std_val, is_masked=False,
                    latitude=d.latitude, longitude=d.longitude,
                ))
            else:
                out.append(DataPoint(
                    time=d.time, observed=None, instance_id=d.instance_id,
                    auxiliaries=d.auxiliaries, imputed=imputed_val, imputed_std=imputed_std_val, is_masked=False,
                    latitude=d.latitude, longitude=d.longitude,
                ))
        return out

    return _batch_small_gap_template(flat_data, params, on_progress, run_per_instance, algo_name="LNN+Kriging Readout")


def batch_impute_small_gap_graph_lnn(
    flat_data: List[DataPoint],
    params: SimulationParams,
    on_progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """Batch small-gap imputation using Graph-LNN (spatial multi-well coupling).
    Builds k-NN spatial graph from (lat, lon); LNN input includes weighted aggregate of neighbor values at each time."""
    groups: Dict[str, List[DataPoint]] = defaultdict(list)
    for d in flat_data:
        if d.instance_id:
            groups[d.instance_id].append(d)
    instance_ids = list(groups.keys())
    coords_by_instance: Dict[str, tuple] = {}
    for iid in instance_ids:
        pts = groups[iid]
        first = next((p for p in pts if getattr(p, "latitude", None) is not None and getattr(p, "longitude", None) is not None), None)
        if first is not None:
            coords_by_instance[iid] = (float(first.latitude), float(first.longitude))
    k = getattr(params, "transfer_neighbor_count", 5) or 5
    graph = build_spatial_graph(instance_ids, coords_by_instance, k=k, use_inverse_distance_weights=True)
    n_with_coords = len(coords_by_instance)
    print(f"[Graph-LNN] Batch: {len(instance_ids)} instances, {n_with_coords} with coords; k-NN k={k}", flush=True)

    def get_neighbor_weights(instance_id: str) -> List[float]:
        other_ids = [nid for nid in instance_ids if nid != instance_id]
        graph_neighbors = dict(graph.get(instance_id, []))
        return [graph_neighbors.get(nid, 0.0) for nid in other_ids]

    params_use_neighbors = dc_replace(params, small_gap_use_neighbors=True)

    def run_per_instance(_id, sorted_data, large_gap_indices, p, neighbor_series_list=None, neighbor_weights=None):
        neighbor_series_list = neighbor_series_list or []
        neighbor_weights = neighbor_weights or [1.0] * len(neighbor_series_list)
        other_ids = [nid for nid in instance_ids if nid != _id]
        if len(neighbor_series_list) != len(other_ids):
            neighbor_series_by_id = {}
        else:
            neighbor_series_by_id = dict(zip(other_ids, neighbor_series_list))
        if len(neighbor_weights) != len(other_ids):
            neighbor_weights = [1.0] * len(neighbor_series_list)
        data_for_imputation = []
        for idx, d in enumerate(sorted_data):
            obs = d.observed if idx not in large_gap_indices else None
            data_for_imputation.append(DataPoint(
                time=d.time, observed=obs, instance_id=d.instance_id,
                auxiliaries=None, is_masked=idx in large_gap_indices,
                latitude=d.latitude, longitude=d.longitude,
            ))
        rng = np.random.default_rng(0)
        result = run_graph_lnn_simulation(
            data_for_imputation, p, neighbor_series_by_id, neighbor_weights, rng=rng
        )
        out = []
        for idx, d in enumerate(sorted_data):
            r = result[idx] if idx < len(result) else None
            imputed_val = r.imputed if r else None
            if idx in large_gap_indices:
                out.append(DataPoint(
                    time=d.time, observed=None, instance_id=d.instance_id,
                    auxiliaries=d.auxiliaries, imputed=imputed_val, is_masked=True,
                    latitude=d.latitude, longitude=d.longitude,
                ))
            elif d.observed is not None:
                out.append(DataPoint(
                    time=d.time, observed=d.observed, instance_id=d.instance_id,
                    auxiliaries=d.auxiliaries, imputed=imputed_val, is_masked=False,
                    latitude=d.latitude, longitude=d.longitude,
                ))
            else:
                out.append(DataPoint(
                    time=d.time, observed=None, instance_id=d.instance_id,
                    auxiliaries=d.auxiliaries, imputed=imputed_val, is_masked=False,
                    latitude=d.latitude, longitude=d.longitude,
                ))
        return out

    return _batch_small_gap_template(
        flat_data, params_use_neighbors, on_progress, run_per_instance,
        algo_name="Graph-LNN", get_neighbor_weights=get_neighbor_weights,
    )


def batch_impute_small_gap_glnn(
    flat_data: List[DataPoint],
    params: SimulationParams,
    on_progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """Batch small-gap imputation using GLNN (Gated LNN) with autotuned hyperparameters."""
    def run_per_instance(_id, sorted_data, large_gap_indices, p, neighbor_series_list=None):
        use_auxiliary = getattr(p, "small_gap_use_auxiliary", True)
        data_for_imputation = []
        for idx, d in enumerate(sorted_data):
            obs = d.observed if idx not in large_gap_indices else None
            aux = list(d.auxiliaries) if (d.auxiliaries and use_auxiliary) else []
            data_for_imputation.append(DataPoint(
                time=d.time, observed=obs, instance_id=d.instance_id,
                auxiliaries=aux if aux else None, is_masked=idx in large_gap_indices,
                latitude=d.latitude, longitude=d.longitude,
            ))
        best_params = optimize_glnn_params(data_for_imputation, p, mode="projection")
        rng = np.random.default_rng(0)
        result = run_glnn_simulation(data_for_imputation, best_params, rng=rng)
        out = []
        for idx, d in enumerate(sorted_data):
            if idx in large_gap_indices:
                out.append(DataPoint(
                    time=d.time, observed=None, instance_id=d.instance_id,
                    auxiliaries=d.auxiliaries, imputed=None, is_masked=True,
                    latitude=d.latitude, longitude=d.longitude,
                ))
            elif d.observed is not None:
                pred_val = result[idx].imputed if idx < len(result) else None
                out.append(DataPoint(
                    time=d.time, observed=d.observed, instance_id=d.instance_id,
                    auxiliaries=d.auxiliaries, imputed=pred_val, is_masked=False,
                    latitude=d.latitude, longitude=d.longitude,
                ))
            else:
                imputed_val = result[idx].imputed if idx < len(result) else None
                out.append(DataPoint(
                    time=d.time, observed=None, instance_id=d.instance_id,
                    auxiliaries=d.auxiliaries, imputed=imputed_val, is_masked=False,
                    latitude=d.latitude, longitude=d.longitude,
                ))
        return out

    return _batch_small_gap_template(flat_data, params, on_progress, run_per_instance, algo_name="GLNN")


def batch_impute_small_gap_gain(
    flat_data: List[DataPoint],
    params: SimulationParams,
    on_progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """Batch small-gap imputation using GAIN with autotuned epochs and hint_rate."""
    def run_per_instance(_id, sorted_data, large_gap_indices, p, neighbor_series_list=None):
        n_trials = getattr(p, "small_gap_gan_trials", 3) or 3
        print(f"  [GAIN] Autotuning ({n_trials} trials)...", flush=True)
        rng = np.random.default_rng(0)
        best_epochs, best_hint = optimize_gain_hyperparams(
            sorted_data, large_gap_indices, p, rng=rng,
        )
        result = run_gain_imputation(
            sorted_data, large_gap_indices, p, rng=rng,
            epochs_override=best_epochs, hint_rate_override=best_hint,
        )
        return result

    return _batch_small_gap_template(flat_data, params, on_progress, run_per_instance, algo_name="GAIN")


def batch_impute_small_gap_condgan(
    flat_data: List[DataPoint],
    params: SimulationParams,
    on_progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """Batch small-gap imputation using Conditional Imputation GAN with autotuned epochs."""
    def run_per_instance(_id, sorted_data, large_gap_indices, p, neighbor_series_list=None):
        n_trials = getattr(p, "small_gap_gan_trials", 3) or 3
        print(f"  [CondGAN] Autotuning ({n_trials} trials)...", flush=True)
        rng = np.random.default_rng(0)
        best_epochs = optimize_condgan_hyperparams(
            sorted_data, large_gap_indices, p, rng=rng,
        )
        result = run_condgan_imputation(
            sorted_data, large_gap_indices, p, rng=rng, epochs_override=best_epochs,
        )
        return result

    return _batch_small_gap_template(flat_data, params, on_progress, run_per_instance, algo_name="CondGAN")


# ---------------------------------------------------------------------------
# Single-instance functions for ARCHI large-gap / extrapolation pipeline.
# These bypass _batch_small_gap_template: they impute ALL missing values
# (no large-gap masking) so the frontend ARCHI pipeline can use them per-donor.
# ---------------------------------------------------------------------------

def single_impute_glnn(flat_data: List[DataPoint], params: SimulationParams) -> List[DataPoint]:
    """GLNN on a single donor series — impute every missing point (no large-gap masking)."""
    sorted_data = sorted(flat_data, key=lambda d: d.time)
    rng = np.random.default_rng(0)
    best_params = optimize_glnn_params(sorted_data, params, mode="projection")
    result = run_glnn_simulation(sorted_data, best_params, rng=rng)
    out: List[DataPoint] = []
    for idx, d in enumerate(sorted_data):
        imputed_val = result[idx].imputed if idx < len(result) else None
        out.append(DataPoint(
            time=d.time, observed=d.observed, instance_id=d.instance_id,
            auxiliaries=d.auxiliaries, imputed=imputed_val, is_masked=False,
            latitude=d.latitude, longitude=d.longitude,
        ))
    return out


def single_impute_gain(flat_data: List[DataPoint], params: SimulationParams) -> List[DataPoint]:
    """GAIN on a single donor series — impute every missing point (no large-gap masking)."""
    sorted_data = sorted(flat_data, key=lambda d: d.time)
    # Identify missing indices (all of them, not just small gaps)
    missing_indices = set(i for i, d in enumerate(sorted_data) if d.observed is None)
    rng = np.random.default_rng(0)
    best_epochs, best_hint = optimize_gain_hyperparams(sorted_data, missing_indices, params, rng=rng)
    result = run_gain_imputation(
        sorted_data, missing_indices, params, rng=rng,
        epochs_override=best_epochs, hint_rate_override=best_hint,
    )
    return result


def single_impute_condgan(flat_data: List[DataPoint], params: SimulationParams) -> List[DataPoint]:
    """Conditional GAN on a single donor series — impute every missing point (no large-gap masking)."""
    sorted_data = sorted(flat_data, key=lambda d: d.time)
    missing_indices = set(i for i, d in enumerate(sorted_data) if d.observed is None)
    rng = np.random.default_rng(0)
    best_epochs = optimize_condgan_hyperparams(sorted_data, missing_indices, params, rng=rng)
    result = run_condgan_imputation(
        sorted_data, missing_indices, params, rng=rng, epochs_override=best_epochs,
    )
    return result


def single_impute_lnncfc(flat_data: List[DataPoint], params: SimulationParams) -> List[DataPoint]:
    """LNNcfc on a single donor series — impute every missing point (no large-gap masking)."""
    sorted_data = sorted(flat_data, key=lambda d: d.time)
    rng = np.random.default_rng(0)
    best_params = optimize_lnncfc_params(sorted_data, params, mode="projection")
    result = run_lnncfc_simulation(sorted_data, best_params, rng=rng)
    out: List[DataPoint] = []
    for idx, d in enumerate(sorted_data):
        imputed_val = result[idx].imputed if idx < len(result) else None
        out.append(DataPoint(
            time=d.time, observed=d.observed, instance_id=d.instance_id,
            auxiliaries=d.auxiliaries, imputed=imputed_val, is_masked=False,
            latitude=d.latitude, longitude=d.longitude,
        ))
    return out


def single_impute_graph_lnn(flat_data: List[DataPoint], params: SimulationParams) -> List[DataPoint]:
    """Graph-LNN on a single donor series — impute every missing point (no large-gap masking).
    Since this is a single-instance call (not batch), spatial graph is trivial (no neighbors)."""
    sorted_data = sorted(flat_data, key=lambda d: d.time)
    rng = np.random.default_rng(0)
    # Single instance: no neighbor graph available, run with empty neighbors
    neighbor_series_by_id: Dict[str, List[DataPoint]] = {}
    neighbor_weights: List[float] = []
    result = run_graph_lnn_simulation(
        sorted_data, params, neighbor_series_by_id, neighbor_weights, rng=rng,
    )
    out: List[DataPoint] = []
    for idx, d in enumerate(sorted_data):
        imputed_val = result[idx].imputed if idx < len(result) else None
        out.append(DataPoint(
            time=d.time, observed=d.observed, instance_id=d.instance_id,
            auxiliaries=d.auxiliaries, imputed=imputed_val, is_masked=False,
            latitude=d.latitude, longitude=d.longitude,
        ))
    return out
