# MC+LNN Spatial Interpolation

## Overview

This module provides spatial interpolation of groundwater levels (WTE) from well point data to a regular grid, serving as a drop-in replacement for kriging in the raster pipeline.

The approach uses **Spatial Detrending + EOF (Empirical Orthogonal Function) Decomposition** to separate the problem into what's predictable spatially (elevation-driven gradient) vs temporally (shared hydrological patterns across wells).

## Pipeline

```
Input: 592 fully imputed wells (288 months, 2000-2023)
  |
  v
[1. Spatial Trend Surface]
  WTE_mean = f(lat, lon, elevation)
  Polynomial regression (degree 2)
  R² = 0.957 — captures 95.7% of spatial variance
  |
  v
[2. Detrend]
  residuals = WTE - trend(lat, lon, elev)
  Residuals are small, spatially smooth
  |
  v
[3. EOF Decomposition (SVD)]
  residuals (288 x 592) = U x S x V^T
  
  U = temporal modes (288 x k) — shared time patterns
      Mode 1: long-term trend
      Mode 2: seasonal cycle
      Mode 3: multi-year drought signal
      ...
  
  S = singular values (k) — mode importance
  
  V = spatial loadings (k x 592) — per-well mode weights
      How strongly each well follows each temporal mode
  |
  v
[4. Interpolate Loadings to Grid]
  For each grid cell:
    - IDW from nearby wells' loadings (k small scalars)
    - Loadings are smooth in space → IDW works well
  |
  v
[5. Reconstruct]
  grid_value = trend(grid_lat, grid_lon, grid_elev)
             + U x S x interpolated_loadings
  |
  v
Output: nTimes x nGridX x nGridY raster (NetCDF)
```

## Why This Works

The key insight: **don't interpolate raw WTE values**.

WTE ranges from 4200-7200 ft across the basin — a 3000 ft gradient driven by geology and topography. Directly interpolating these values (via IDW, kriging, or MC) produces large errors because:
- IDW averages values from wells at very different elevations
- Kriging's variogram can't capture non-stationary gradients
- MC's low-rank assumption fails on heterogeneous absolute values

By decomposing into trend + EOF:
1. The **trend surface** handles the large gradient (polynomial of lat/lon/elevation)
2. The **EOF modes** capture shared temporal patterns (seasonal, trends, drought)
3. We only interpolate **k spatial loadings** (small smooth scalars) — not 288 raw values

## Cross-Validation Results (LOO on 30 wells)

| Method | KGE median | RMSE mean (ft) | Notes |
|--------|:----------:|:--------------:|-------|
| **EOF (20 modes)** | **0.075** | **32.3** | Best RMSE |
| IDW | 0.051 | 62.9 | Baseline |
| Kriging (k=50) | -0.045 | 121.9 | Per-timestep, noisy variogram |
| MC SoftImpute (anomaly) | 0.047 | 58.1 | MC alone |
| MC + aux rows (lat/lon/elev) | -0.53 | 241.4 | Failed approach |
| Raw MC (no detrending) | -0.60 | 78.4 | Converges to mean |

**EOF reduces RMSE by 49% vs IDW and 73% vs kriging.**

## Usage

### Drop-in replacement for kriging

```python
from mc_lnn_spatial_interpolation import generate_nc_file_mc_lnn, create_grid_coords

# Same grid creation as kriging pipeline
grid_x, grid_y = create_grid_coords(x_coords, y_coords, x_steps, bbox, raster_extent)

# Replace generate_nc_file() with:
nc_file = generate_nc_file_mc_lnn(
    file_name, grid_x, grid_y, years_df,
    x_coords, y_coords, bbox, raster_extent,
    well_elevations=elevs,  # optional, fetched from Open-Meteo if not provided
    n_modes=20,             # EOF modes to retain
    max_neighbors=30,       # IDW neighbors for loading interpolation
)

# Same clipping workflow as kriging
interp_nc = xarray.open_dataset(nc_file)
interp_nc.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
interp_nc.rio.write_crs("epsg:4326", inplace=True)
clipped_nc = interp_nc.rio.clip(aquifer.geometry.apply(mapping), crs=4326, drop=True)
```

### Direct field generation

```python
from mc_lnn_spatial_interpolation import mc_lnn_eof_interpolation

predictions, info = mc_lnn_eof_interpolation(
    well_lats, well_lons, well_elevs,
    well_matrix,              # nTimes x nWells (fully imputed)
    target_lats, target_lons, target_elevs,
    n_modes=20,
    max_neighbors=30,
)
# predictions: nTimes x nTargets
# info: {'trend_r2': 0.957, 'var_explained': [...], ...}
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_modes` | 10-20 | EOF modes to retain. More modes = more detail but potential overfitting. 20 recommended. |
| `max_neighbors` | 30 | IDW neighbors for loading interpolation. Higher = smoother. |
| `trend_degree` | 2 | Polynomial degree for trend surface. 2 = quadratic (recommended). |
| `batch_size` | 100 | Grid cells per MC batch (memory vs speed tradeoff). |

## Relationship to MC+LNN Imputation

This spatial interpolation module builds on the MC+LNN imputation pipeline:

1. **Imputation** (MC+LNN, KGE ~0.85): Fills temporal gaps in wells that have some data
   - PCHIP for small gaps
   - ARCHI donor regression + SoftImpute MC + LNN CFC for large gaps

2. **Interpolation** (EOF, RMSE 32 ft): Predicts at locations with no wells
   - Spatial trend surface + EOF decomposition + IDW loading interpolation
   - Uses the fully imputed well data as input

The pipeline: **Raw observations → MC+LNN imputation → Complete well data → EOF interpolation → Raster grid**

## Files

| File | Description |
|------|-------------|
| `mc_lnn_spatial_interpolation.py` | Main module — EOF interpolation + NetCDF generation |
| `cv_mc_lnn_interpolation.py` | Cross-validation scripts (LOO on 30 wells) |
| `cv_mc_lnn_interpolation_enhanced.py` | Enhanced version with precipitation aux data |
| `cv_mc_raster_grid.py` | Earlier MC grid approach (for comparison) |
| `elevation_cache.json` | Cached DEM elevations from Open-Meteo |

## Approaches Tested and Discarded

During development, several MC-based spatial interpolation approaches were tested and found inferior:

1. **Raw MC (SoftImpute on well matrix)**: Converges to global mean due to 3000 ft WTE range
2. **Anomaly MC (subtract per-well mean)**: IDW mean estimate has ~58 ft error
3. **MC + spatial aux rows (lat/lon/elev)**: Aux rows too few to anchor spatial structure
4. **Grid-based MC (nTimes x nGridCells)**: Same convergence issue
5. **MC+LNN Transfer (OLS from nearest well)**: Unstable OLS coefficients
6. **Enhanced Transfer (precipitation correlation)**: Precip patterns too similar across basin
7. **Distance-weighted Transfer**: KGE ~0.12 but RMSE still ~58 ft

The EOF approach succeeded because it separates the spatial trend (handled by polynomial regression on elevation) from the temporal patterns (handled by SVD modes with smooth spatial loadings).
