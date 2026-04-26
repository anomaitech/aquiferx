"""
Batch small-gap LNN imputation.
Mirrors utils/math.ts: batchImputeSmallGapInstances (without DB/neighbor loading).
"""

from typing import List, Dict, Optional, Callable, Any
from collections import defaultdict

import numpy as np

from .types import DataPoint, SimulationParams
from .lnn_core import run_lnn_simulation, optimize_lnn_params
from .gaps import identify_gaps, identify_outliers
from .math_utils import calculate_kge

# Core LNN: single run (no uncertainty). Use LNN+Kriging Readout for confidence intervals.


def batch_impute_small_gap_instances(
    flat_data: List[DataPoint],
    params: SimulationParams,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    neighbors_by_instance: Optional[Dict[str, List[DataPoint]]] = None,
) -> Dict[str, Any]:
    """
    Batch impute all instances that have at least one small gap using LNN.

    Process (aligned with JS batchImputeSmallGapInstances):
    1. Group data by instance_id
    2. For each instance with at least one small gap (size <= max_gap_threshold):
       - Mask large gaps (set observed=None for those indices)
       - Optionally merge neighbor series as auxiliary inputs (if neighbors_by_instance provided)
       - Run LNN simulation
       - Iterative outlier removal (z-score > 2.5) up to small_gap_max_iterations
       - Compute KGE on observed vs imputed (excluding large gaps and outliers)
    3. Return counts and per-instance results; optionally include full imputed series.

    Args:
        flat_data: All data points (multiple instances, each with instance_id).
        params: SimulationParams (max_gap_threshold, kge_threshold, small_gap_max_iterations, etc.).
        on_progress: Optional callback(current_index, total, instance_id).
        neighbors_by_instance: Optional dict instance_id -> list of DataPoint (neighbor series).
            If provided and params.small_gap_use_neighbors, neighbor values at same time are
            appended to auxiliaries for that instance.

    Returns:
        {
            "imputed": int,
            "saved": int,  # count with KGE >= kge_threshold
            "skipped": int,
            "results": { instance_id: { "kge": float, "saved": bool } },
            "filled_data": [ DataPoint, ... ]  # all points with imputed set where filled
        }
    """
    groups: Dict[str, List[DataPoint]] = defaultdict(list)
    for d in flat_data:
        if d.instance_id:
            groups[d.instance_id].append(d)

    instance_ids = list(groups.keys())
    results: Dict[str, Dict[str, Any]] = {}
    imputed_count = 0
    saved_count = 0
    skipped_count = 0
    filled_data: List[DataPoint] = []

    max_iterations = getattr(params, "small_gap_max_iterations", 10) or 10
    max_gap_threshold = params.max_gap_threshold
    kge_threshold = params.kge_threshold
    use_neighbors = getattr(params, "small_gap_use_neighbors", False)
    use_auxiliary = getattr(params, "small_gap_use_auxiliary", True)
    outlier_handling = getattr(params, "small_gap_outlier_handling", "remove_iterative") or "remove_iterative"
    neighbors = neighbors_by_instance or {}

    print(f"[LNN core] Batch: {len(instance_ids)} instances, KGE threshold={kge_threshold:.3f}", flush=True)

    for inst_idx, instance_id in enumerate(instance_ids):
        if on_progress:
            on_progress(inst_idx + 1, len(instance_ids), instance_id)

        instance_data = groups[instance_id]
        sorted_data = sorted(instance_data, key=lambda d: d.time)

        gaps, large_gap_indices = identify_gaps(sorted_data, max_gap_threshold)
        has_small_gaps = any(g["size"] <= max_gap_threshold for g in gaps)

        if not has_small_gaps:
            skipped_count += 1
            results[instance_id] = {"kge": float("-inf"), "saved": False}
            filled_data.extend(sorted_data)
            print(f"  [LNN core] Instance {inst_idx + 1}/{len(instance_ids)}: {instance_id} -> skipped (no small gaps)", flush=True)
            continue

        # Build data for imputation: mask large gaps; keep CSV auxiliaries; optionally add neighbor series as aux
        # neighbors_by_instance: instance_id -> list of neighbor series (each series = list of DataPoint sorted by time)
        data_for_imputation: List[DataPoint] = []
        neighbor_series_list = (neighbors.get(instance_id) or []) if use_neighbors else []
        if neighbor_series_list and not isinstance(neighbor_series_list[0], list):
            neighbor_series_list = [neighbor_series_list]
        for idx, d in enumerate(sorted_data):
            obs = d.observed
            aux = list(d.auxiliaries) if (d.auxiliaries and use_auxiliary) else []
            if idx in large_gap_indices:
                obs = None
            for series in neighbor_series_list:
                match = next((p for p in series if abs(p.time - d.time) < 1e-9), None)
                val = 0.0
                if match:
                    val = match.imputed if match.imputed is not None else (match.observed if match.observed is not None else 0.0)
                aux.append(val)
            data_for_imputation.append(DataPoint(
                time=d.time,
                observed=obs,
                instance_id=d.instance_id,
                auxiliaries=aux if aux else None,
                is_masked=idx in large_gap_indices,
                latitude=d.latitude,
                longitude=d.longitude,
            ))

        # Auto-tune LNN params per instance (same as math.js optimizeLNNParams)
        best_params = optimize_lnn_params(data_for_imputation, params, mode="projection")

        # Outlier removal loop (use best_params so observed vs imputed plot matches math.js)
        # When outlier_handling == "keep", run LNN once and skip iterative removal.
        current_result: List[DataPoint] = []
        iteration_count = 0
        all_outlier_indices: set = set()
        # Use a fresh RNG seeded once per instance (not per iteration) so each
        # outlier-removal iteration gets a different reservoir when re-tuning.
        rng = np.random.default_rng(inst_idx)

        while iteration_count < max_iterations:
            iteration_count += 1
            current_result = run_lnn_simulation(data_for_imputation, best_params, rng=rng)
            if outlier_handling == "keep":
                break
            imputed_subset = [r for idx, r in enumerate(current_result) if idx not in large_gap_indices]
            outlier_indices = identify_outliers(imputed_subset)
            full_outlier_indices = []
            imputed_idx = 0
            for full_idx, r in enumerate(current_result):
                if full_idx not in large_gap_indices:
                    if imputed_idx in outlier_indices:
                        full_outlier_indices.append(full_idx)
                        all_outlier_indices.add(full_idx)
                    imputed_idx += 1
            if full_outlier_indices and iteration_count < max_iterations:
                for idx in full_outlier_indices:
                    old = data_for_imputation[idx]
                    data_for_imputation[idx] = DataPoint(
                        time=old.time,
                        observed=None,
                        instance_id=old.instance_id,
                        auxiliaries=old.auxiliaries,
                        is_masked=True,
                        is_outlier=True,
                        latitude=old.latitude,
                        longitude=old.longitude,
                    )
            else:
                break

        # Optional ensemble for uncertainty (imputed_std) like Kriging: run LNN multiple times, mean + std
        lnn_ensemble_size = getattr(params, "lnn_ensemble_size", 1) or 1
        final_result: List[DataPoint] = []
        if lnn_ensemble_size > 1:
            runs: List[List[DataPoint]] = [current_result]
            for run_idx in range(1, lnn_ensemble_size):
                rng_ens = np.random.default_rng(inst_idx * 10000 + run_idx)
                run_result = run_lnn_simulation(data_for_imputation, best_params, rng=rng_ens)
                runs.append(run_result)
            # Per-index mean and std (only for non-large-gap indices)
            n_pts = len(current_result)
            for idx in range(n_pts):
                r = current_result[idx]
                if idx in large_gap_indices:
                    final_result.append(DataPoint(
                        time=r.time,
                        observed=None,
                        instance_id=r.instance_id,
                        imputed=None,
                        imputed_std=None,
                        auxiliaries=r.auxiliaries,
                        is_masked=True,
                        latitude=r.latitude,
                        longitude=r.longitude,
                    ))
                else:
                    imputed_vals = [run[idx].imputed for run in runs if run[idx].imputed is not None]
                    if not imputed_vals:
                        final_result.append(r)
                        continue
                    mean_val = float(np.mean(imputed_vals))
                    std_val = float(np.std(imputed_vals)) if len(imputed_vals) >= 2 else None
                    if std_val is not None and (np.isnan(std_val) or std_val <= 0):
                        std_val = None
                    final_result.append(DataPoint(
                        time=r.time,
                        observed=r.observed,
                        instance_id=r.instance_id,
                        imputed=mean_val,
                        imputed_std=std_val,
                        auxiliaries=r.auxiliaries,
                        is_masked=False,
                        latitude=r.latitude,
                        longitude=r.longitude,
                    ))
        else:
            # Single run: no imputed_std (original behavior)
            for idx, r in enumerate(current_result):
                if idx in large_gap_indices:
                    final_result.append(DataPoint(
                        time=r.time,
                        observed=None,
                        instance_id=r.instance_id,
                        imputed=None,
                        auxiliaries=r.auxiliaries,
                        is_masked=True,
                        latitude=r.latitude,
                        longitude=r.longitude,
                    ))
                else:
                    final_result.append(r)

        # KGE: only points that are observed and imputed (not large gap, not outlier)
        training_true: List[float] = []
        training_pred: List[float] = []
        for idx, d in enumerate(final_result):
            if idx not in large_gap_indices and idx not in all_outlier_indices:
                if d.observed is not None and d.imputed is not None:
                    training_true.append(d.observed)
                    training_pred.append(d.imputed)
        kge = float("-inf")
        if training_true:
            kge = calculate_kge(training_true, training_pred)

        saved = kge >= kge_threshold
        results[instance_id] = {"kge": kge, "saved": saved}
        imputed_count += 1
        if saved:
            saved_count += 1
        kge_str = f"{kge:.3f}" if kge > float("-inf") else "-inf"
        print(f"  [LNN core] Instance {inst_idx + 1}/{len(instance_ids)}: {instance_id} -> KGE={kge_str}, saved={saved}", flush=True)
        filled_data.extend(final_result)

    return {
        "imputed": imputed_count,
        "saved": saved_count,
        "skipped": skipped_count,
        "results": results,
        "filled_data": filled_data,
    }
