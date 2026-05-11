# Spatial Interpolation: Trend + EOF + IDW

## Overview

This module interpolates groundwater water table elevation (WTE) from well point data to a regular spatial grid. It serves as a drop-in replacement for kriging in the raster pipeline.

The approach decomposes the problem into three steps that each handle what they're best at:

1. **Spatial Trend** — captures the large elevation-driven gradient
2. **EOF (SVD)** — extracts shared temporal patterns across wells
3. **IDW on Loadings** — transfers those patterns to grid cells

## Detailed Walkthrough with Examples

### Step 1: Spatial Trend Surface

We have 592 wells, each with a temporal mean WTE. These means range from ~4200 to ~7200 ft — a 3000 ft gradient driven by topography.

We fit a degree-2 polynomial:

```
WTE_mean = β0 + β1·lat + β2·lon + β3·elev
         + β4·lat² + β5·lon² + β6·elev²
         + β7·lat·lon + β8·lat·elev + β9·lon·elev
```

Example with 3 wells:

```
Well A:  lat=40.3, lon=-111.5, elev=1473m → mean WTE = 4812 ft
Well B:  lat=38.2, lon=-113.8, elev=1336m → mean WTE = 5397 ft
Well C:  lat=42.7, lon=-111.1, elev=1978m → mean WTE = 6891 ft
```

The polynomial learns: higher elevation → higher WTE, southern wells → different relationship than northern wells, etc.

```
poly(40.3, -111.5, 1473) = 4798 ft   (actual: 4812, error: 14 ft)
poly(38.2, -113.8, 1336) = 5410 ft   (actual: 5397, error: 13 ft)
poly(42.7, -111.1, 1978) = 6905 ft   (actual: 6891, error: 14 ft)
```

**R² = 0.957** — the polynomial explains 95.7% of why wells have different mean WTEs. The remaining 4.3% are local deviations the polynomial can't capture.

### Step 2: Compute Residuals

Subtract the trend from every well at every timestep:

```
                    Raw WTE    Trend     Residual
Well A, 2000-01:   4815.3  -  4798.0  =  +17.3
Well A, 2000-02:   4810.1  -  4798.0  =  +12.1
Well A, 2000-03:   4822.7  -  4798.0  =  +24.7
Well A, 2000-04:   4805.9  -  4798.0  =   +7.9
...
Well B, 2000-01:   5398.1  -  5410.0  =  -11.9
Well B, 2000-02:   5396.5  -  5410.0  =  -13.5
...
```

The residuals are much smaller (~±30 ft) compared to raw WTE (4200-7200 ft). This is the data EOF will work with.

The full residual matrix:

```
              Well_A   Well_B   Well_C  ...  Well_592
2000-01       +17.3    -11.9    +22.1  ...    -8.4
2000-02       +12.1    -13.5    +18.3  ...    -5.2
2000-03       +24.7     -7.8    +30.5  ...   +11.6
...
2023-12        -5.1     +3.2     -9.7  ...    +2.8

Shape: 288 rows (months) × 592 columns (wells)
```

### Step 3: EOF Decomposition (SVD)

SVD decomposes the residual matrix into three parts:

```
Residuals (288 × 592) = U (288 × k) × S (k) × V^T (k × 592)
```

With k=20 modes, this says: **every well's 288-month residual series can be approximated as a weighted combination of just 20 shared temporal patterns.**

**U columns — Temporal Modes** (what happens over time):

```
Mode 1: [+0.02, +0.01, +0.03, +0.04, ..., -0.01]   288 values
         A slow declining trend across all wells
         (e.g., regional water table decline over 24 years)

Mode 2: [+0.05, -0.03, -0.05, +0.04, ..., +0.03]   288 values
         The seasonal wet/dry cycle
         (e.g., spring recharge peak, summer drawdown)

Mode 3: [+0.01, +0.01, +0.02, +0.04, ..., -0.03]   288 values
         A multi-year drought signal
         (e.g., the 2012-2016 drought impact)

... up to Mode 20
```

**S — Singular Values** (how important each mode is):

```
S[0] = 450    Mode 1 (trend) is very strong
S[1] = 280    Mode 2 (seasonal) is moderate
S[2] = 120    Mode 3 (drought) is weaker
...
S[19] = 15    Mode 20 captures fine detail
```

**V^T rows — Spatial Loadings** (how much each well follows each pattern):

```
                   Well_A  Well_B  Well_C  ...  Well_592
Mode 1 loading:     0.80    0.20    0.70  ...     0.90
Mode 2 loading:     0.30    0.90    0.40  ...     0.10
Mode 3 loading:     0.50    0.10    0.60  ...     0.30
...
```

- Well A has loading 0.80 on Mode 1 → strong long-term trend
- Well B has loading 0.90 on Mode 2 → strong seasonality
- Well C has loading 0.60 on Mode 3 → affected by drought

**Reconstruct any well's residual:**

```
Well A residual at time t = S[0] × 0.80 × Mode1[t]
                          + S[1] × 0.30 × Mode2[t]
                          + S[2] × 0.50 × Mode3[t]
                          + ... (20 terms)
```

### Step 4: Interpolate Loadings to Grid Cells (IDW)

For a grid cell at (lat=39.5, lon=-112.0) with no well, we need its loadings. We estimate them from nearby wells using Inverse Distance Weighting:

```
Nearby wells:
  Well A: 12 km away → weight = 1/12² = 0.0069
  Well B: 5 km away  → weight = 1/5²  = 0.0400
  Well C: 20 km away → weight = 1/20² = 0.0025

Normalized weights: A=0.14, B=0.81, C=0.05

Grid cell Mode 1 loading = 0.14 × 0.80 + 0.81 × 0.20 + 0.05 × 0.70 = 0.31
Grid cell Mode 2 loading = 0.14 × 0.30 + 0.81 × 0.90 + 0.05 × 0.40 = 0.79
Grid cell Mode 3 loading = 0.14 × 0.50 + 0.81 × 0.10 + 0.05 × 0.60 = 0.18
```

We interpolated **3 small numbers** (the loadings), not 288 raw values. The nearest well (B, 5 km) dominates the prediction, so the grid cell will behave most like Well B.

### Step 5: Reconstruct at Grid Cell

```
Grid cell residual at time t = S[0] × 0.31 × Mode1[t]
                              + S[1] × 0.79 × Mode2[t]
                              + S[2] × 0.18 × Mode3[t]
                              + ...

Grid cell WTE at time t = poly_trend(39.5, -112.0, elev)  +  residual[t]
                        = 5234.7                           +  residual[t]
```

The final prediction has:
- The right **absolute level** (from the trend surface using elevation)
- The right **seasonal pattern** (from Mode 2, weighted by nearby wells)
- The right **long-term trend** (from Mode 1, weighted by nearby wells)
- Temporal coherence across all 288 months (from the shared modes)

## Why This Works Better Than Alternatives

### vs Plain IDW (RMSE 62.9 ft → 32.3 ft, 49% reduction)

IDW interpolates 288 raw WTE values independently — each month is a separate weighted average. Problems:
- Month 1: averages 4815, 5398, 6891 → gets a value far from truth
- No temporal coherence — January prediction is independent of February
- Dominated by the 3000 ft spatial gradient

EOF interpolates just **k small loadings**, then multiplies by shared modes:
- Temporal coherence guaranteed (same seasonal shape as nearby wells)
- The trend surface handles the 3000 ft gradient separately
- Loadings are smooth scalars that interpolate well via IDW

### vs Kriging (RMSE 121.9 ft)

Kriging interpolates per-timestep using a variogram. Problems with 592 wells:
- Variogram estimation is noisy with so many wells at varied distances
- Per-timestep: no temporal coherence between months
- The non-stationary WTE field violates kriging's stationarity assumption

### vs Pure MC/SoftImpute (RMSE 58-241 ft)

MC assumes the data matrix is low-rank. Problems:
- Raw WTE (4200-7200 ft) is NOT low-rank — each well has a distinct absolute level
- MC converges to the global mean for unknown locations
- Even with normalization, MC can't capture the spatial gradient

## Cross-Validation Results (LOO on 30 wells)

| Method | KGE median | RMSE mean (ft) | Notes |
|--------|:----------:|:--------------:|-------|
| **Poly Trend + EOF (20 modes) + IDW** | **0.075** | **32.3** | Best overall |
| Poly Trend + EOF + Kriging loadings | -0.303 | 31.6 | Lower RMSE but worse KGE |
| XGBoost Trend + EOF + IDW | 0.075 | 36.5 | Trend R²=0.997 but overfits under LOO |
| Clustered Trend + EOF + IDW | 0.075 | 52.4 | Blending artifacts at cluster boundaries |
| Graph Laplacian + EOF | -0.060 | 34.0 | Over-smooths loadings |
| Plain IDW | 0.051 | 62.9 | Baseline |
| Kriging (per-timestep, k=50) | -0.045 | 121.9 | Noisy variogram |
| MC SoftImpute (anomaly) | 0.047 | 58.1 | MC alone, no trend |
| MC + aux rows (lat/lon/elev) | -0.53 | 241.4 | Aux rows too few |
| Distance Transfer (MC+LNN series) | 0.117 | 56.6 | Transfer without EOF |

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
| `n_modes` | 20 | EOF modes to retain. 20 captures ~95% of residual variance. |
| `max_neighbors` | 30 | IDW neighbors for loading interpolation. |
| `trend_degree` | 2 | Polynomial degree for trend surface (2 = quadratic). |
| `trend_method` | "poly" | Trend type: "poly", "xgb" (XGBoost), or "clustered". |
| `interpolation_method` | "idw" | Loading interpolation: "idw", "kriging", or "graph". |

## Full Pipeline: Imputation → Interpolation

```
Raw well observations (sparse, gaps)
  |
  v
[MC+LNN Imputation]  ← KGE 0.85
  PCHIP small-gap fill
  ARCHI donor regression
  SoftImpute Matrix Completion
  LNN CFC temporal refinement
  |
  v
Complete well data (592 wells × 288 months, no gaps)
  |
  v
[EOF Spatial Interpolation]  ← RMSE 32.3 ft
  Polynomial trend surface (R²=0.957)
  SVD → temporal modes + spatial loadings
  IDW loading interpolation to grid
  Reconstruct: trend + modes × loadings
  |
  v
Raster grid (nTimes × nX × nY)  → NetCDF → clip to aquifer
```

## Files

| File | Description |
|------|-------------|
| `mc_lnn_spatial_interpolation.py` | Main module: EOF interpolation + NetCDF generation |
| `SPATIAL_INTERPOLATION.md` | This documentation |
| `cv_mc_lnn_interpolation.py` | LOO cross-validation (ARCHI+MC, transfer LNN, IDW) |
| `cv_mc_lnn_interpolation_enhanced.py` | Enhanced CV with precipitation auxiliary data |
| `cv_mc_raster_grid.py` | Grid-based MC approach (for comparison) |
| `cv_mc_raster_interpolation.py` | MC temporal holdout CV |
| `cv_mc_spatial_interpolation.py` | MC with spatial aux rows CV |
| `cv_mc_interpolation.py` | Anomaly MC spatial interpolation CV |
| `elevation_cache.json` | Cached DEM elevations from Open-Meteo |
