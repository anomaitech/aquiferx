import {
  Aquifer, Region, Well, Measurement,
  RasterAnalysisParams, RasterAnalysisResult, RasterGrid, RasterFrame, RasterFrameStats,
  SpatialMethod, KrigingOptions, IdwOptions, GeneralInterpolationOptions, TemporalOptions, RasterOptions,
  ImputationModelResult,
} from '../types';
import { isPointInGeoJSON } from '../utils/geo';
import { interpolatePCHIP, interpolateLinear, kernelSmooth, smoothModelCombined } from '../utils/interpolation';
import { krigGrid, estimateVariogramParams } from './kriging';
import { idwGrid } from './idw';
import { softImpute } from './mcSoftImpute';
import { slugify } from '../utils/strings';

export interface RasterPipelineInput {
  temporal: TemporalOptions;
  spatial: {
    method: SpatialMethod;
    resolution: number;
    kriging: KrigingOptions;
    idw: IdwOptions;
  };
  general: GeneralInterpolationOptions;
  title: string;
}

function buildCompletedMcWellValues(
  wellInterp: Map<string, { well: Well; values: (number | null)[] }>
): Map<string, { well: Well; values: (number | null)[] }> {
  const entries = Array.from(wellInterp.entries());
  if (!entries.length) return new Map();
  const nWells = entries.length;
  const nTimes = entries[0][1].values.length;
  if (!nTimes) return new Map(entries);

  const matrix: number[][] = Array.from({ length: nTimes }, (_, t) =>
    entries.map(([, entry]) => {
      const v = entry.values[t];
      return v == null ? Number.NaN : v;
    })
  );

  const rank = Math.max(1, Math.min(8, nWells, nTimes));
  const completed = softImpute(matrix, {
    rank,
    shrinkage: 0,
    maxIterations: 80,
    tolerance: 1e-5,
  }).completed;

  const out = new Map<string, { well: Well; values: (number | null)[] }>();
  for (let w = 0; w < entries.length; w++) {
    const [wellId, entry] = entries[w];
    const values: (number | null)[] = new Array(nTimes);
    for (let t = 0; t < nTimes; t++) {
      const original = entry.values[t];
      values[t] = original != null ? original : completed[t]?.[w] ?? null;
    }
    out.set(wellId, { well: entry.well, values });
  }
  return out;
}

// Generate interval dates between start and end (inclusive) as ISO strings
function generateIntervalDates(startDate: string, endDate: string, interval: '3months' | '6months' | '1year'): string[] {
  const dates: string[] = [];
  const start = new Date(startDate);
  const end = new Date(endDate);

  let current = new Date(start);
  while (current <= end) {
    dates.push(current.toISOString().slice(0, 10));
    if (interval === '1year') {
      current.setFullYear(current.getFullYear() + 1);
    } else if (interval === '6months') {
      current.setMonth(current.getMonth() + 6);
    } else {
      current.setMonth(current.getMonth() + 3);
    }
  }
  return dates;
}


export async function runRasterAnalysis(
  input: RasterPipelineInput,
  dataType: string,
  aquifer: Aquifer,
  region: Region,
  wells: Well[],
  measurements: Measurement[],
  onProgress: (step: string, pct: number) => void
): Promise<RasterAnalysisResult> {
  const { temporal, spatial, general, title } = input;
  const { startDate, endDate, interval } = temporal;
  const resolution = spatial.resolution;

  // Step 1: Build grid
  onProgress('Building grid...', 0);
  await yieldToUI();

  const [minLat, minLng, maxLat, maxLng] = aquifer.bounds;
  const nx = resolution;
  const dx = (maxLng - minLng) / nx;
  // Adjust dy so cells are square in real-world distance (1° lng is shorter than 1° lat)
  const centerLat = (minLat + maxLat) / 2;
  const dy = dx * Math.cos(centerLat * Math.PI / 180);
  const ny = Math.max(1, Math.ceil((maxLat - minLat) / dy));

  // Build grid cell centers and mask
  const gridLats: number[] = [];
  const gridLngs: number[] = [];
  const mask: (0 | 1)[] = [];

  for (let row = 0; row < ny; row++) {
    for (let col = 0; col < nx; col++) {
      const cellLng = minLng + (col + 0.5) * dx;
      const cellLat = minLat + (row + 0.5) * dy;
      gridLngs.push(cellLng);
      gridLats.push(cellLat);
      mask.push(isPointInGeoJSON(cellLat, cellLng, aquifer.geojson) ? 1 : 0);
    }
  }

  // Step 2: Generate interval dates
  const intervalDates = generateIntervalDates(startDate, endDate, interval);
  const intervalTimestamps = intervalDates.map(d => new Date(d).getTime());

  // Step 3: Temporal interpolation per well
  // Group measurements by well for the target data type
  const wteMeasurements = measurements.filter(m => m.dataType === dataType);
  const byWell = new Map<string, Measurement[]>();
  for (const m of wteMeasurements) {
    if (!byWell.has(m.wellId)) byWell.set(m.wellId, []);
    byWell.get(m.wellId)!.push(m);
  }

  // For each well, interpolate to interval dates
  const wellInterp = new Map<string, { well: Well; values: (number | null)[] }>();

  if (temporal.method === 'model' && temporal.modelFilePath) {
    // Model-based temporal interpolation: fetch model JSON, use combined column
    onProgress('Loading imputation model...', 5);
    await yieldToUI();

    const modelResp = await fetch(`/data/${temporal.modelFilePath}`);
    if (!modelResp.ok) throw new Error(`Failed to load model: ${temporal.modelFilePath}`);
    const modelResult: ImputationModelResult = await modelResp.json();

    // Group model data by well
    const modelByWell = new Map<string, { dates: number[]; values: number[] }>();
    for (const row of modelResult.data) {
      if (!modelByWell.has(row.well_id)) modelByWell.set(row.well_id, { dates: [], values: [] });
      const entry = modelByWell.get(row.well_id)!;
      entry.dates.push(new Date(row.date).getTime());
      entry.values.push(row.combined);
    }

    const modelWellIds = wells.map(w => w.id).filter(id => modelByWell.has(id));

    for (let wi = 0; wi < modelWellIds.length; wi++) {
      const wellId = modelWellIds[wi];
      const well = wells.find(w => w.id === wellId)!;
      const { dates: modelDates, values: modelValues } = modelByWell.get(wellId)!;

      onProgress(`Model interpolation well ${wi + 1}/${modelWellIds.length}...`, 5 + (wi / modelWellIds.length) * 25);
      if (wi % 5 === 0) await yieldToUI();

      if (modelDates.length < 2) continue;

      const minT = modelDates[0];
      const maxT = modelDates[modelDates.length - 1];

      // PCHIP interpolate the monthly model data to the requested interval timestamps
      const validTargets: number[] = [];
      const validIndices: number[] = [];
      for (let i = 0; i < intervalTimestamps.length; i++) {
        const t = intervalTimestamps[i];
        if (t >= minT && t <= maxT) {
          validTargets.push(t);
          validIndices.push(i);
        }
      }

      const interpValues: (number | null)[] = intervalTimestamps.map(() => null);
      if (validTargets.length > 0) {
        const interpolated = interpolatePCHIP(modelDates, modelValues, validTargets);
        for (let i = 0; i < validIndices.length; i++) {
          interpValues[validIndices[i]] = interpolated[i];
        }
      }

      wellInterp.set(wellId, { well, values: interpValues });
    }
  } else if ((temporal.method === 'model-direct' || temporal.method === 'model-mavg') && temporal.modelFilePath) {
    // Model-based: direct lookup or smoothed moving average
    onProgress('Loading imputation model...', 5);
    await yieldToUI();

    const modelResp = await fetch(`/data/${temporal.modelFilePath}`);
    if (!modelResp.ok) throw new Error(`Failed to load model: ${temporal.modelFilePath}`);
    const modelResult: ImputationModelResult = await modelResp.json();

    // Group model data by well
    const modelByWell = new Map<string, { date: string; combined: number }[]>();
    for (const row of modelResult.data) {
      if (!modelByWell.has(row.well_id)) modelByWell.set(row.well_id, []);
      modelByWell.get(row.well_id)!.push({ date: row.date, combined: row.combined });
    }

    const modelWellIds = wells.map(w => w.id).filter(id => modelByWell.has(id));

    for (let wi = 0; wi < modelWellIds.length; wi++) {
      const wellId = modelWellIds[wi];
      const well = wells.find(w => w.id === wellId)!;
      const rows = modelByWell.get(wellId)!;

      onProgress(`Model ${temporal.method === 'model-direct' ? 'direct' : 'MA'} well ${wi + 1}/${modelWellIds.length}...`, 5 + (wi / modelWellIds.length) * 25);
      if (wi % 5 === 0) await yieldToUI();

      // Build timestamp→value map (direct or smoothed)
      let tsMap: Map<number, number>;

      if (temporal.method === 'model-mavg') {
        const { dates, values } = smoothModelCombined(rows, temporal.maWindow);
        tsMap = new Map();
        for (let i = 0; i < dates.length; i++) tsMap.set(dates[i], values[i]);
      } else {
        // model-direct: exact lookup by timestamp
        tsMap = new Map();
        for (const r of rows) tsMap.set(new Date(r.date).getTime(), r.combined);
      }

      // Map interval timestamps to values via exact lookup or nearest 1st-of-month
      const interpValues: (number | null)[] = intervalTimestamps.map(t => {
        const exact = tsMap.get(t);
        if (exact !== undefined) return exact;
        // Try snapping to 1st of month (intervals should already align)
        const d = new Date(t);
        const snap = new Date(d.getFullYear(), d.getMonth(), 1).getTime();
        const snapped = tsMap.get(snap);
        return snapped !== undefined ? snapped : null;
      });

      wellInterp.set(wellId, { well, values: interpValues });
    }
  } else {
    // Standard temporal interpolation (pchip, linear, moving-average)
    const wellIds = wells.map(w => w.id).filter(id => byWell.has(id));

    for (let wi = 0; wi < wellIds.length; wi++) {
      const wellId = wellIds[wi];
      const well = wells.find(w => w.id === wellId)!;
      const meas = byWell.get(wellId)!;

      onProgress(`Interpolating well ${wi + 1}/${wellIds.length}...`, (wi / wellIds.length) * 30);
      if (wi % 5 === 0) await yieldToUI();

      const sorted = [...meas]
        .filter(m => !isNaN(new Date(m.date).getTime()))
        .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());

      // Filter by min observations and min time span
      if (sorted.length < Math.max(2, temporal.minObservations)) continue;

      const xValues = sorted.map(m => new Date(m.date).getTime());
      const yValues = sorted.map(m => m.value);
      const minT = xValues[0];
      const maxT = xValues[xValues.length - 1];

      const timeSpanYears = (maxT - minT) / (365.25 * 24 * 60 * 60 * 1000);
      if (timeSpanYears < temporal.minTimeSpan) continue;

      // Interpolate within the well's data range (no extrapolation) using selected method
      let interpValues: (number | null)[];

      if (temporal.method === 'moving-average') {
        const smoothed = kernelSmooth(xValues, yValues, intervalTimestamps, temporal.maWindow);
        interpValues = intervalTimestamps.map((t, i) => {
          if (t < minT || t > maxT) return null;
          return smoothed[i];
        });
      } else {
        // PCHIP or Linear
        interpValues = intervalTimestamps.map(() => null);
        const validTargets: number[] = [];
        const validIndices: number[] = [];
        for (let i = 0; i < intervalTimestamps.length; i++) {
          const t = intervalTimestamps[i];
          if (t >= minT && t <= maxT) {
            validTargets.push(t);
            validIndices.push(i);
          }
        }
        if (validTargets.length > 0) {
          const interpolated = temporal.method === 'linear'
            ? interpolateLinear(xValues, yValues, validTargets)
            : interpolatePCHIP(xValues, yValues, validTargets);
          for (let i = 0; i < validIndices.length; i++) {
            interpValues[validIndices[i]] = interpolated[i];
          }
        }
      }

      wellInterp.set(wellId, { well, values: interpValues });
    }
  }

  console.log(`[RasterAnalysis] ${wellInterp.size}/${wells.length} wells qualified (method=${temporal.method}, minObs=${temporal.minObservations}, minSpan=${temporal.minTimeSpan}yr)`);

  // Apply log transform to well values before spatial interpolation
  if (general.logInterpolation) {
    for (const [, entry] of wellInterp) {
      entry.values = entry.values.map(v => v !== null ? Math.log(v) : null);
    }
  }

  const spatialWellValues = spatial.method === 'mc'
    ? buildCompletedMcWellValues(wellInterp)
    : wellInterp;

  // Step 4: Compute variogram from all wells' mean values (kriging only)
  let variogramParams: { sill: number; range: number; nugget: number } | undefined;

  if (spatial.method === 'kriging') {
    const allWellLats: number[] = [];
    const allWellLngs: number[] = [];
    const allWellMeanValues: number[] = [];

    for (const [, { well, values }] of spatialWellValues) {
      const validValues = values.filter((v): v is number => v !== null);
      if (validValues.length > 0) {
        allWellLats.push(well.lat);
        allWellLngs.push(well.lng);
        allWellMeanValues.push(validValues.reduce((a, b) => a + b, 0) / validValues.length);
      }
    }

    variogramParams = estimateVariogramParams(allWellLats, allWellLngs, allWellMeanValues, {
      nuggetEnabled: spatial.kriging.nugget,
      rangeMode: spatial.kriging.rangeMode,
      rangeValue: spatial.kriging.rangeValue,
    });
    console.log(`[RasterAnalysis] Using single variogram for all ${intervalDates.length} timesteps, ${allWellLats.length} wells total`);
  }

  // Step 5: Spatial interpolation per timestep
  const frames: RasterFrame[] = [];

  for (let ti = 0; ti < intervalDates.length; ti++) {
    const methodLabel = spatial.method === 'kriging' ? 'Kriging' : spatial.method === 'mc' ? 'MC' : 'IDW';
    onProgress(`${methodLabel} timestep ${ti + 1}/${intervalDates.length}...`, 30 + (ti / intervalDates.length) * 50);
    await yieldToUI();

    // Collect wells that have a value at this timestep
    const activeLats: number[] = [];
    const activeLngs: number[] = [];
    const activeValues: number[] = [];

    for (const [, { well, values }] of spatialWellValues) {
      const val = values[ti];
      if (val !== null) {
        activeLats.push(well.lat);
        activeLngs.push(well.lng);
        activeValues.push(val);
      }
    }

    console.log(`[RasterAnalysis] Timestep ${intervalDates[ti]}: ${activeValues.length} wells, values [${Math.min(...activeValues).toFixed(1)}, ${Math.max(...activeValues).toFixed(1)}]`);

    let gridValues: (number | null)[];
    if (activeValues.length >= 2) {
      try {
        if (spatial.method === 'kriging') {
          gridValues = krigGrid(
            activeLats, activeLngs, activeValues,
            gridLats, gridLngs, mask,
            variogramParams,
            spatial.kriging.variogramModel
          );
        } else {
          gridValues = idwGrid(
            activeLats, activeLngs, activeValues,
            gridLats, gridLngs, mask,
            {
              exponent: spatial.idw.exponent,
              nodalFunction: spatial.idw.nodalFunction,
              neighborMode: spatial.idw.neighborMode,
              neighborCount: spatial.idw.neighborCount,
            }
          );
        }
      } catch (err) {
        throw new Error(`Spatial interpolation failed at timestep ${intervalDates[ti]}: ${err instanceof Error ? err.message : String(err)}`);
      }
    } else if (activeValues.length === 1) {
      // Single well: constant value for all masked cells
      gridValues = mask.map(m => m === 1 ? activeValues[0] : null);
    } else {
      gridValues = mask.map(() => null);
    }

    // Post-processing: inverse log transform and truncation
    if (general.logInterpolation || general.truncateLow || general.truncateHigh) {
      for (let i = 0; i < gridValues.length; i++) {
        if (gridValues[i] === null) continue;
        let v = gridValues[i]!;
        if (general.logInterpolation) v = Math.exp(v);
        if (general.truncateLow && v < general.truncateLowValue) v = general.truncateLowValue;
        if (general.truncateHigh && v > general.truncateHighValue) v = general.truncateHighValue;
        gridValues[i] = v;
      }
    }

    frames.push({ date: intervalDates[ti], values: gridValues });
  }

  // Step 6: Compute per-frame statistics from well-level values
  onProgress('Computing statistics...', 82);
  await yieldToUI();

  const stats: RasterFrameStats[] = [];
  for (let ti = 0; ti < intervalDates.length; ti++) {
    const vals: number[] = [];
    for (const [, { values }] of spatialWellValues) {
      const v = values[ti];
      if (v !== null) vals.push(v);
    }
    if (vals.length === 0) {
      stats.push({ date: intervalDates[ti], count: 0, min: 0, max: 0, mean: 0, std: 0, median: 0, p25: 0, p75: 0 });
      continue;
    }
    vals.sort((a, b) => a - b);
    const n = vals.length;
    const sum = vals.reduce((s, v) => s + v, 0);
    const mean = sum / n;
    const variance = vals.reduce((s, v) => s + (v - mean) ** 2, 0) / n;
    const std = Math.sqrt(variance);
    const percentile = (p: number) => {
      const idx = (p / 100) * (n - 1);
      const lo = Math.floor(idx);
      const hi = Math.ceil(idx);
      return lo === hi ? vals[lo] : vals[lo] + (vals[hi] - vals[lo]) * (idx - lo);
    };
    stats.push({
      date: intervalDates[ti],
      count: n,
      min: Math.round(vals[0] * 100) / 100,
      max: Math.round(vals[n - 1] * 100) / 100,
      mean: Math.round(mean * 100) / 100,
      std: Math.round(std * 100) / 100,
      median: Math.round(percentile(50) * 100) / 100,
      p25: Math.round(percentile(25) * 100) / 100,
      p75: Math.round(percentile(75) * 100) / 100,
    });
  }

  onProgress('Saving results...', 85);
  await yieldToUI();

  // Step 7: Assemble result
  const code = slugify(title);

  // Build legacy params for backward compatibility
  const params: RasterAnalysisParams = {
    startDate,
    endDate,
    resolution,
    interval,
    title,
    minObservations: temporal.minObservations,
    minTimeSpanYears: temporal.minTimeSpan,
    smoothingMethod: (temporal.method === 'model' || temporal.method === 'model-direct' || temporal.method === 'model-mavg') ? 'pchip' : temporal.method,
    smoothingMonths: temporal.maWindow,
  };

  const options: RasterOptions = {
    temporal: { ...temporal },
    spatial: { ...spatial },
    general: { ...general },
  };

  const result: RasterAnalysisResult = {
    version: 1,
    title,
    code,
    aquiferId: aquifer.id,
    aquiferName: aquifer.name,
    regionId: region.id,
    dataType,
    params,
    grid: { minLng, minLat, dx, dy, nx, ny, mask },
    frames,
    createdAt: new Date().toISOString(),
    options,
    generatedAt: new Date().toISOString(),
    stats,
  };

  // Save to disk via API — per-aquifer subfolder with raster_ prefix
  const aquiferSlug = slugify(aquifer.name);
  await fetch('/api/save-data', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      files: [{
        path: `${region.id}/${aquiferSlug}/raster_${dataType}_${code}.json`,
        content: JSON.stringify(result),
      }]
    }),
  });

  onProgress('Complete!', 100);
  return result;
}

// Yield to UI thread for progress updates
function yieldToUI(): Promise<void> {
  return new Promise(r => setTimeout(r, 0));
}
