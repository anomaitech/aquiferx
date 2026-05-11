/**
 * EOF (Empirical Orthogonal Function) Spatial Interpolation
 *
 * Pipeline:
 *   1. Polynomial trend surface: WTE_mean = f(lat, lon, elevation)
 *   2. Detrend: residuals = WTE - trend
 *   3. SVD on residuals → temporal modes (U) + spatial loadings (V)
 *   4. IDW interpolation of loadings to grid cells
 *   5. Reconstruct: grid = trend + U × S × interpolated_loadings
 */

import { Matrix, SingularValueDecomposition } from 'ml-matrix';
import { haversineDistance } from '../utils/geo';

// ─── Spatial Trend Surface ─────────────────────────────────────────────────

interface TrendModel {
  coeffs: number[];
  latMean: number; latStd: number;
  lonMean: number; lonStd: number;
  elevMean: number; elevStd: number;
}

function fitSpatialTrend(lats: number[], lons: number[], elevs: number[], means: number[]): {
  model: TrendModel; r2: number;
} {
  const n = lats.length;
  const latMean = lats.reduce((a, b) => a + b, 0) / n;
  const lonMean = lons.reduce((a, b) => a + b, 0) / n;
  const elevMean = elevs.reduce((a, b) => a + b, 0) / n;
  const latStd = Math.max(Math.sqrt(lats.reduce((a, v) => a + (v - latMean) ** 2, 0) / n), 1e-10);
  const lonStd = Math.max(Math.sqrt(lons.reduce((a, v) => a + (v - lonMean) ** 2, 0) / n), 1e-10);
  const elevStd = Math.max(Math.sqrt(elevs.reduce((a, v) => a + (v - elevMean) ** 2, 0) / n), 1e-10);

  const latN = lats.map(v => (v - latMean) / latStd);
  const lonN = lons.map(v => (v - lonMean) / lonStd);
  const elevN = elevs.map(v => (v - elevMean) / elevStd);

  // Design matrix: [1, lat, lon, elev, lat², lon², elev², lat*lon, lat*elev, lon*elev]
  const p = 10;
  const X: number[][] = [];
  for (let i = 0; i < n; i++) {
    X.push([1, latN[i], lonN[i], elevN[i],
            latN[i] ** 2, lonN[i] ** 2, elevN[i] ** 2,
            latN[i] * lonN[i], latN[i] * elevN[i], lonN[i] * elevN[i]]);
  }

  // Ridge regression: coeffs = (X'X + αI)⁻¹ X'y
  const alpha = 1.0;
  const XtX: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
  const Xty: number[] = new Array(p).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) {
      Xty[j] += X[i][j] * means[i];
      for (let k = 0; k < p; k++) {
        XtX[j][k] += X[i][j] * X[i][k];
      }
    }
  }
  for (let j = 0; j < p; j++) XtX[j][j] += alpha;

  // Solve via Gaussian elimination
  const coeffs = solveLinear(XtX, Xty);

  // R²
  let ssTot = 0, ssRes = 0;
  const meanY = means.reduce((a, b) => a + b, 0) / n;
  for (let i = 0; i < n; i++) {
    let pred = 0;
    for (let j = 0; j < p; j++) pred += coeffs[j] * X[i][j];
    ssRes += (means[i] - pred) ** 2;
    ssTot += (means[i] - meanY) ** 2;
  }

  return {
    model: { coeffs, latMean, latStd, lonMean, lonStd, elevMean, elevStd },
    r2: 1 - ssRes / (ssTot + 1e-12),
  };
}

function predictTrend(model: TrendModel, lats: number[], lons: number[], elevs: number[]): number[] {
  const { coeffs, latMean, latStd, lonMean, lonStd, elevMean, elevStd } = model;
  return lats.map((lat, i) => {
    const ln = (lat - latMean) / latStd;
    const lo = (lons[i] - lonMean) / lonStd;
    const el = (elevs[i] - elevMean) / elevStd;
    const x = [1, ln, lo, el, ln * ln, lo * lo, el * el, ln * lo, ln * el, lo * el];
    return x.reduce((sum, v, j) => sum + v * coeffs[j], 0);
  });
}

function solveLinear(A: number[][], b: number[]): number[] {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let maxRow = col, maxVal = Math.abs(M[col][col]);
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(M[row][col]) > maxVal) { maxVal = Math.abs(M[row][col]); maxRow = row; }
    }
    [M[col], M[maxRow]] = [M[maxRow], M[col]];
    if (Math.abs(M[col][col]) < 1e-12) continue;
    for (let row = col + 1; row < n; row++) {
      const f = M[row][col] / M[col][col];
      for (let j = col; j <= n; j++) M[row][j] -= f * M[col][j];
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = M[i][n];
    for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j];
    x[i] = Math.abs(M[i][i]) > 1e-12 ? s / M[i][i] : 0;
  }
  return x;
}

// ─── IDW for loadings ──────────────────────────────────────────────────────

function idwInterpolateLoadings(
  targetLat: number, targetLng: number,
  wellLats: number[], wellLngs: number[],
  loadings: number[][], // [nModes][nWells]
  maxNeighbors: number = 30,
): number[] {
  const nWells = wellLats.length;
  const nModes = loadings.length;
  const dists: { idx: number; dist: number }[] = [];

  for (let i = 0; i < nWells; i++) {
    dists.push({ idx: i, dist: haversineDistance(targetLat, targetLng, wellLats[i], wellLngs[i]) });
  }
  dists.sort((a, b) => a.dist - b.dist);
  const sel = dists.slice(0, maxNeighbors);

  if (sel[0].dist < 1) {
    return loadings.map(row => row[sel[0].idx]);
  }

  const weights = sel.map(s => 1 / (s.dist ** 2 + 1e-6));
  const wSum = weights.reduce((a, b) => a + b, 0);

  return loadings.map(row => {
    let val = 0;
    for (let i = 0; i < sel.length; i++) {
      val += (weights[i] / wSum) * row[sel[i].idx];
    }
    return val;
  });
}

// ─── Elevation fetch (Open-Meteo) ──────────────────────────────────────────

async function fetchElevations(lats: number[], lons: number[]): Promise<number[]> {
  const elevs = new Array(lats.length).fill(0);
  const batchSize = 100;

  for (let i = 0; i < lats.length; i += batchSize) {
    const batch = lats.slice(i, i + batchSize);
    const batchLons = lons.slice(i, i + batchSize);
    const latStr = batch.map(l => l.toFixed(6)).join(',');
    const lonStr = batchLons.map(l => l.toFixed(6)).join(',');

    try {
      const resp = await fetch(`https://api.open-meteo.com/v1/elevation?latitude=${latStr}&longitude=${lonStr}`);
      if (resp.ok) {
        const data = await resp.json();
        const el = data.elevation || [];
        for (let j = 0; j < batch.length; j++) {
          if (j < el.length && el[j] != null) elevs[i + j] = el[j];
        }
      }
    } catch {
      // silently continue with 0 elevation
    }
  }

  return elevs;
}

// ─── Main EOF interpolation ────────────────────────────────────────────────

export interface EofInterpolationInput {
  wellLats: number[];
  wellLngs: number[];
  wellElevs: number[];
  wellValues: (number | null)[][]; // [nWells][nTimes] — temporal values per well
  gridLats: number[];
  gridLngs: number[];
  mask: (0 | 1)[];
  nModes: number;
  maxNeighbors: number;
  onProgress?: (msg: string, pct: number) => void;
}

export async function eofGridInterpolation(input: EofInterpolationInput): Promise<(number | null)[][]> {
  const { wellLats, wellLngs, wellElevs, wellValues, gridLats, gridLngs, mask,
          nModes, maxNeighbors, onProgress } = input;

  const nWells = wellLats.length;
  const nTimes = wellValues[0]?.length ?? 0;
  const nGrid = gridLats.length;

  if (nWells < 3 || nTimes < 2) {
    return Array.from({ length: nTimes }, () => mask.map(m => m === 1 ? null : null));
  }

  onProgress?.('EOF: Fitting trend surface...', 0);

  // Step 1: Compute well means
  const wellMeans: number[] = wellValues.map(vals => {
    const valid = vals.filter((v): v is number => v !== null);
    return valid.length > 0 ? valid.reduce((a, b) => a + b, 0) / valid.length : 0;
  });

  // Step 2: Fit spatial trend
  const { model: trendModel, r2 } = fitSpatialTrend(wellLats, wellLngs, wellElevs, wellMeans);
  const wellTrend = predictTrend(trendModel, wellLats, wellLngs, wellElevs);

  console.log(`[EOF] Trend R²=${r2.toFixed(4)}, ${nWells} wells, ${nTimes} timesteps`);

  // Step 3: Build residual matrix (nTimes × nWells)
  onProgress?.('EOF: Computing residuals...', 10);
  const residualMatrix: number[][] = Array.from({ length: nTimes }, (_, t) =>
    wellValues.map((vals, w) => {
      const v = vals[t];
      return (v !== null ? v : wellMeans[w]) - wellTrend[w];
    })
  );

  // Step 4: SVD
  onProgress?.('EOF: SVD decomposition...', 20);
  const mat = new Matrix(residualMatrix);
  const svd = new SingularValueDecomposition(mat, { autoTranspose: true });
  const U = svd.leftSingularVectors;
  const S = svd.diagonal;
  const V = svd.rightSingularVectors;
  const k = Math.min(nModes, S.length, nWells, nTimes);

  // Extract loadings: V^T rows (k × nWells)
  const loadings: number[][] = Array.from({ length: k }, (_, m) =>
    Array.from({ length: nWells }, (_, w) => V.get(w, m))
  );

  // Step 5: Fetch grid elevations
  onProgress?.('EOF: Fetching grid elevations...', 30);
  const gridElevs = await fetchElevations(gridLats, gridLngs);

  // Step 6: Predict trend at grid cells
  onProgress?.('EOF: Computing grid trends...', 40);
  const gridTrend = predictTrend(trendModel, gridLats, gridLngs, gridElevs);

  // Step 7: Interpolate loadings to each grid cell and reconstruct
  const frames: (number | null)[][] = Array.from({ length: nTimes }, () =>
    new Array(nGrid).fill(null)
  );

  for (let g = 0; g < nGrid; g++) {
    if (mask[g] === 0) continue;

    if (g % 100 === 0) {
      onProgress?.(`EOF: Interpolating grid cell ${g + 1}/${nGrid}...`,
                    40 + (g / nGrid) * 55);
    }

    // IDW interpolate loadings
    const targetLoadings = idwInterpolateLoadings(
      gridLats[g], gridLngs[g], wellLats, wellLngs, loadings, maxNeighbors
    );

    // Reconstruct: residual[t] = Σ U[t,m] * S[m] * loading[m]
    for (let t = 0; t < nTimes; t++) {
      let residual = 0;
      for (let m = 0; m < k; m++) {
        residual += U.get(t, m) * S[m] * targetLoadings[m];
      }
      frames[t][g] = gridTrend[g] + residual;
    }
  }

  onProgress?.('EOF: Complete', 100);
  return frames;
}
