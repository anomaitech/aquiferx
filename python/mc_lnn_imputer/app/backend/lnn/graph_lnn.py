"""
Graph-LNN: instance-based imputation with spatial multi-well coupling.

Builds a spatial graph over instances (nodes = wells/sites, edges = k-NN by distance).
At each time step, each instance's LNN input includes a weighted aggregate of neighbor
values (spatial coupling). Uses standard LNN reservoir + ridge readout.
"""

from typing import List, Dict, Tuple, Optional

import numpy as np

from .types import DataPoint, SimulationParams
from .lnn_core import (
    prepare_data,
    get_current_observation_value,
    run_lnn_simulation,
)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate distance in km between two (lat, lon) points."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def build_spatial_graph(
    instance_ids: List[str],
    coords_by_instance: Dict[str, Tuple[float, float]],
    k: int = 5,
    use_inverse_distance_weights: bool = True,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Build k-NN spatial graph. For each instance, return list of (neighbor_id, weight).
    Weight = 1 / (1 + distance_km) if use_inverse_distance_weights else 1.0.
    """
    graph: Dict[str, List[Tuple[str, float]]] = {}
    for iid in instance_ids:
        coord = coords_by_instance.get(iid)
        if coord is None:
            graph[iid] = []
            continue
        lat, lon = coord
        neighbors_with_dist: List[Tuple[str, float]] = []
        for nid in instance_ids:
            if nid == iid:
                continue
            ncoord = coords_by_instance.get(nid)
            if ncoord is None:
                continue
            d = _haversine_km(lat, lon, ncoord[0], ncoord[1])
            neighbors_with_dist.append((nid, d))
        neighbors_with_dist.sort(key=lambda x: x[1])
        top = neighbors_with_dist[:k]
        if use_inverse_distance_weights:
            weights = [1.0 / (1.0 + d) for _, d in top]
        else:
            weights = [1.0] * len(top)
        graph[iid] = [(nid, w) for (nid, _), w in zip(top, weights)]
    return graph


def _value_at_time(series: List[DataPoint], t: float, use_imputed: bool = True) -> Optional[float]:
    """Value at time t from series (nearest time). observed or imputed."""
    if not series:
        return None
    times = np.array([d.time for d in series])
    idx = int(np.argmin(np.abs(times - t)))
    d = series[idx]
    v = d.observed if d.observed is not None else (d.imputed if use_imputed and d.imputed is not None else None)
    return float(v) if v is not None else None


def run_graph_lnn_simulation(
    data: List[DataPoint],
    params: SimulationParams,
    neighbor_series_by_id: Dict[str, List[DataPoint]],
    neighbor_weights: List[float],
    rng: Optional[np.random.Generator] = None,
) -> List[DataPoint]:
    """
    Run LNN with spatial coupling: at each time step, input = [current_obs, weighted_agg]
    where weighted_agg = sum(weight_j * value_j(t)) / sum(weights) over graph neighbors.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    # Build auxiliary at each time: single value = weighted mean of neighbors
    neighbor_ids = list(neighbor_series_by_id.keys())
    if len(neighbor_ids) != len(neighbor_weights):
        neighbor_weights = [1.0] * len(neighbor_ids)
    weight_sum = max(sum(neighbor_weights), 1e-10)
    data_with_aux: List[DataPoint] = []
    for d in data:
        t = d.time
        agg = 0.0
        for nid, w in zip(neighbor_ids, neighbor_weights):
            series = neighbor_series_by_id.get(nid, [])
            v = _value_at_time(series, t, use_imputed=True)
            if v is not None:
                agg += w * v
        agg /= weight_sum
        aux = [agg] if neighbor_ids else []
        data_with_aux.append(DataPoint(
            time=d.time,
            observed=d.observed,
            instance_id=d.instance_id,
            auxiliaries=aux if aux else None,
            is_masked=d.is_masked,
            latitude=d.latitude,
            longitude=d.longitude,
        ))
    return run_lnn_simulation(data_with_aux, params, rng=rng)
