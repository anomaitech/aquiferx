"""
Hybrid LNN + residual correction for instance-based imputation.

- Single spatial correction (this module): run_lnn_kriging_simulation fits one GPR on (lat, lon)
  to residuals at observed points; same correction at all times for a given location.

- Per-date residual correction (geostat_impute.batch_impute_long_gap_lnn_kriging): at each
  missing date t, map error from neighbors' residuals at t (observed_j(t) - LNN_j(t)) via 2D
  kriging at (target_lon, target_lat) → correction(t). Fill = LNN_target(t) + correction(t).
  So the correction is different for each missed date (error mapped per date).
"""

import numpy as np
from dataclasses import replace
from typing import List, Any, Optional

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

from .types import DataPoint, SimulationParams
from .lnn_core import run_lnn_simulation


def run_lnn_kriging_simulation(
    data: List[DataPoint],
    params: SimulationParams,
    rng: Optional[np.random.Generator] = None,
) -> List[DataPoint]:
    """
    Hybrid LNN + Residual Kriging (GPR) simulation.
    1. Run LNN to get base imputed values.
    2. Fit GPR on (lat, lon) to residuals (observed - imputed) at observed points.
    3. Add GPR-predicted correction to every point's imputed value.
    Uses DataPoint.latitude / .longitude; if fewer than 3 points have coords, returns LNN-only.
    """
    if rng is None:
        rng = np.random.default_rng()

    instance_id = data[0].instance_id if data else ""
    lnn_results = run_lnn_simulation(data, params, rng)
    n_obs = sum(1 for d in lnn_results if d.observed is not None)
    print(f"    [LNN+Kriging] Instance {instance_id}: LNN done, {n_obs} observed points", flush=True)

    coords_train = []
    residuals_train = []
    for d in lnn_results:
        if d.observed is not None and d.imputed is not None:
            lat = (d.latitude if d.latitude is not None else 0.0)
            lon = (d.longitude if d.longitude is not None else 0.0)
            res = float(d.observed) - float(d.imputed)
            coords_train.append([lat, lon])
            residuals_train.append(res)

    if len(coords_train) < 3:
        print(f"    [LNN+Kriging] Instance {instance_id}: skipping GPR (< 3 points with coords), using LNN only", flush=True)
        return lnn_results

    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=1e-5)
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        alpha=getattr(params, "ridge_alpha", 1e-4),
    )
    X_train = np.array(coords_train)
    y_train = np.array(residuals_train)
    try:
        gpr.fit(X_train, y_train)
    except Exception as e:
        print(f"    [LNN+Kriging] Instance {instance_id}: GPR fit failed ({e}), using LNN only", flush=True)
        return lnn_results

    print(f"    [LNN+Kriging] Instance {instance_id}: GPR fitted on {len(coords_train)} points, applying spatial correction", flush=True)
    final_results: List[DataPoint] = []
    for d in lnn_results:
        lat = d.latitude if d.latitude is not None else 0.0
        lon = d.longitude if d.longitude is not None else 0.0
        spatial_correction = float(gpr.predict(np.array([[lat, lon]]))[0])
        refined = (float(d.imputed) + spatial_correction) if d.imputed is not None else spatial_correction
        final_results.append(replace(d, imputed=refined))
    return final_results
