import {
  Aquifer, Region, Well, Measurement,
  ImputationParams, ImputationDataRow, ImputationModelResult, ImputationWellMetrics,
} from '../types';
import { fetchGldasFeatures, GldasFeatures } from './gldasFetch';
import { trainElm, predictElm } from './elm';
import { interpolatePCHIP } from '../utils/interpolation';
import { slugify } from '../utils/strings';

export interface ImputationPipelineInput {
  title: string;
  startDate: string;
  endDate: string;
  gldasStartDate: string;  // full GLDAS feature range start (from fetchGldasDateRange)
  gldasEndDate: string;    // full GLDAS feature range end
  minSamples: number;
  gapSize: number;    // days (Python default: 730)
  padSize: number;    // days (Python default: 180)
  hiddenUnits: number;
  lambda: number;
}

function yieldToUI(): Promise<void> {
  return new Promise(r => setTimeout(r, 0));
}

/**
 * Generate monthly date grid from startDate to endDate (inclusive), UTC-based.
 * Matches Python: pd.date_range(start, freq='1MS', end=end)
 */
function generateMonthlyDates(startDate: string, endDate: string): string[] {
  const dates: string[] = [];
  const start = new Date(startDate);
  const end = new Date(endDate);
  let year = start.getUTCFullYear();
  let month = start.getUTCMonth();
  while (true) {
    const d = new Date(Date.UTC(year, month, 1));
    if (d > end) break;
    dates.push(d.toISOString().slice(0, 10));
    month++;
    if (month > 11) { month = 0; year++; }
  }
  return dates;
}

function clampDateRange(
  startDate: string,
  endDate: string,
  gldasStartDate: string,
  gldasEndDate: string,
): { min: string; max: string } | null {
  const min = gldasStartDate > startDate ? gldasStartDate : startDate;
  const max = gldasEndDate < endDate ? gldasEndDate : endDate;
  return min <= max ? { min, max } : null;
}

const MS_PER_DAY = 24 * 60 * 60 * 1000;

/**
 * Run the full imputation pipeline.
 *
 * Matches the Python notebook flow:
 * 1. interp_well — PCHIP per well with gap/pad blanking
 * 2. Build combined dataframe (well PCHIP + GLDAS features)
 * 3. zscore_training_data — global z-score normalization
 * 4. Add year (min-max) and month (one-hot) features
 * 5. impute_data — per-well ELM training on ALL wells
 * 6. reverse_zscore_data — per-well denormalization
 */
export async function runImputationPipeline(
  input: ImputationPipelineInput,
  aquifer: Aquifer,
  region: Region,
  wells: Well[],
  measurements: Measurement[],
  onLog: (msg: string) => void,
  onProgress: (step: string, pct: number) => void,
): Promise<ImputationModelResult> {
  const { title, startDate, endDate, minSamples, gapSize, padSize, hiddenUnits, lambda } = input;
  const effectiveRange = clampDateRange(startDate, endDate, input.gldasStartDate, input.gldasEndDate);
  if (!effectiveRange) {
    throw new Error('Selected date range does not overlap the GLDAS feature range.');
  }

  // Gap/pad are already in days (matching Python's gap_size=730, pad=180)
  const gapSizeMs = gapSize * MS_PER_DAY;
  const padSizeMs = padSize * MS_PER_DAY;

  // ===== Step 1: Fetch GLDAS data (0-5%) =====
  onProgress('Fetching GLDAS data...', 0);
  onLog(`Fetching GLDAS soil moisture data for aquifer "${aquifer.name}"...`);
  await yieldToUI();

  let gldas: GldasFeatures;
  try {
    gldas = await fetchGldasFeatures(aquifer.id, aquifer.geojson, input.gldasStartDate, input.gldasEndDate);
    onLog(`GLDAS data loaded: ${gldas.dates.length} monthly records, range ${gldas.dates[0]} to ${gldas.dates[gldas.dates.length - 1]}`);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    onLog(`ERROR: Failed to fetch GLDAS data: ${msg}`);
    throw new Error(`GLDAS data fetch failed: ${msg}`);
  }
  onProgress('GLDAS data loaded', 5);

  // ===== Step 2: Prepare well data (5-10%) =====
  onProgress('Preparing well data...', 5);
  await yieldToUI();

  const wteMeasurements = measurements.filter(m => m.dataType === 'wte');
  const byWell = new Map<string, Measurement[]>();
  for (const m of wteMeasurements) {
    if (m.date < effectiveRange.min || m.date > effectiveRange.max) continue;
    if (!byWell.has(m.wellId)) byWell.set(m.wellId, []);
    byWell.get(m.wellId)!.push(m);
  }

  // Filter wells by minSamples (matches Python: wells_df.dropna(thresh=min_samples, axis=1))
  interface QualifiedWell {
    well: Well;
    sorted: { date: string; ts: number; value: number }[];
  }
  const qualifiedWells: QualifiedWell[] = [];
  let omitted = 0;

  for (const well of wells) {
    const meas = byWell.get(well.id);
    if (!meas || meas.length < minSamples) {
      omitted++;
      continue;
    }
    const sorted = [...meas]
      .filter(m => !isNaN(new Date(m.date).getTime()))
      .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
      .map(m => ({ date: m.date, ts: new Date(m.date).getTime(), value: m.value }));
    if (sorted.length < 2) {
      omitted++;
      continue;
    }
    qualifiedWells.push({ well, sorted });
  }

  onLog(`${qualifiedWells.length} wells qualified (>= ${minSamples} measurements in ${effectiveRange.min} to ${effectiveRange.max}), ${omitted} omitted`);

  if (qualifiedWells.length === 0) {
    throw new Error(`No wells have >= ${minSamples} measurements. Adjust min samples or date range.`);
  }

  onProgress('Well data prepared', 10);

  // ===== Step 3: Generate monthly date grid from actual GLDAS feature dates =====
  // Use the actual trimmed GLDAS dates (not WMS capabilities dates) to match Python
  const gldasFeatureStart = gldas.dates[0];
  const gldasFeatureEnd = gldas.dates[gldas.dates.length - 1];
  const monthlyDates = generateMonthlyDates(gldasFeatureStart, gldasFeatureEnd);
  const monthlyTimestamps = monthlyDates.map(d => new Date(d).getTime());
  onLog(`Training date grid: ${monthlyDates.length} months from ${gldasFeatureStart} to ${gldasFeatureEnd}`);
  onLog(`Output will be clipped to: ${startDate} to ${endDate}`);

  // Build GLDAS lookup by date
  const gldasByDate = new Map<string, number>();
  for (let i = 0; i < gldas.dates.length; i++) {
    gldasByDate.set(gldas.dates[i], i);
  }

  // ===== PHASE A: PCHIP interpolation for ALL wells =====
  // Matches Python interp_well(wells_df, gap_size, pad, spacing)
  onProgress('PCHIP interpolation...', 12);
  await yieldToUI();

  const wellPchip = new Map<string, (number | null)[]>();

  for (const { well, sorted } of qualifiedWells) {
    const measTimestamps = sorted.map(m => m.ts);
    const measValues = sorted.map(m => m.value);
    const firstMeasTs = measTimestamps[0];
    const lastMeasTs = measTimestamps[measTimestamps.length - 1];

    // PCHIP interpolate to monthly dates within [firstMeas, lastMeas)
    // Python blanks index >= end_meas_date, so last measurement date is excluded
    const pchipFull: (number | null)[] = new Array(monthlyDates.length).fill(null);
    const inRangeIndices: number[] = [];
    const inRangeTimestamps: number[] = [];
    for (let i = 0; i < monthlyDates.length; i++) {
      const ts = monthlyTimestamps[i];
      if (ts >= firstMeasTs && ts < lastMeasTs) {
        inRangeIndices.push(i);
        inRangeTimestamps.push(ts);
      }
    }

    if (inRangeIndices.length > 0) {
      const pchipValues = interpolatePCHIP(measTimestamps, measValues, inRangeTimestamps);
      for (let j = 0; j < inRangeIndices.length; j++) {
        pchipFull[inRangeIndices[j]] = pchipValues[j];
      }
    }

    // Blank out interiors of large gaps, keeping pad at boundaries
    // Matches Python: gap comparison in days, pad offsets from measurement dates
    // Python: x_diff > gap_size (timedelta), start = meas[g-1] + timedelta(days=pad)
    for (let i = 0; i < sorted.length - 1; i++) {
      const gapMs = measTimestamps[i + 1] - measTimestamps[i];
      if (gapMs > gapSizeMs) {
        const padStartTs = measTimestamps[i] + padSizeMs;
        const padEndTs = measTimestamps[i + 1] - padSizeMs;
        // Python: interp_df[start:end] = np.nan (inclusive on both ends)
        for (const idx of inRangeIndices) {
          const ts = monthlyTimestamps[idx];
          if (ts >= padStartTs && ts <= padEndTs) {
            pchipFull[idx] = null;
          }
        }
      }
    }

    wellPchip.set(well.id, pchipFull);
  }

  // Second filter: drop wells with < minSamples non-NaN PCHIP values
  // Matches Python: well_interp_df.dropna(thresh=min_samples, axis=1)
  const activeWells: QualifiedWell[] = [];
  for (const qw of qualifiedWells) {
    const pchip = wellPchip.get(qw.well.id);
    if (!pchip) continue;
    const nonNullCount = pchip.filter(v => v !== null).length;
    if (nonNullCount >= minSamples) {
      activeWells.push(qw);
    } else {
      wellPchip.delete(qw.well.id);
      onLog(`Well ${qw.well.name}: dropped after PCHIP (only ${nonNullCount} non-NaN values)`);
    }
  }

  onLog(`PCHIP interpolation complete: ${activeWells.length} wells retained`);

  // ===== PHASE B: Identify valid GLDAS rows =====
  // Matches Python: combined_df.dropna(subset=names) — keep rows where GLDAS features are present
  const validRowIndices: number[] = [];
  for (let i = 0; i < monthlyDates.length; i++) {
    if (gldasByDate.has(monthlyDates[i])) {
      validRowIndices.push(i);
    }
  }
  const nRows = validRowIndices.length;
  onLog(`${nRows} valid GLDAS rows in date range`);

  if (nRows === 0) {
    throw new Error('No valid GLDAS data in the selected date range');
  }

  // GLDAS feature arrays for valid rows
  const gldasSoilw = validRowIndices.map(i => gldas.soilw[gldasByDate.get(monthlyDates[i])!]);
  const gldasYr01 = validRowIndices.map(i => gldas.soilw_yr01[gldasByDate.get(monthlyDates[i])!]);
  const gldasYr03 = validRowIndices.map(i => gldas.soilw_yr03[gldasByDate.get(monthlyDates[i])!]);
  const gldasYr05 = validRowIndices.map(i => gldas.soilw_yr05[gldasByDate.get(monthlyDates[i])!]);
  const gldasYr10 = validRowIndices.map(i => gldas.soilw_yr10[gldasByDate.get(monthlyDates[i])!]);

  // ===== PHASE C: Compute z-score statistics =====
  // Matches Python: norm_df = zscore_training_data(combined_df, combined_df)
  // Global per-column mean/std using sample std (ddof=1, pandas default)
  onProgress('Computing normalization statistics...', 15);
  await yieldToUI();

  // Feature z-score stats (global, from ALL valid rows)
  const gArrays = [gldasSoilw, gldasYr01, gldasYr03, gldasYr05, gldasYr10];
  const featureMeans: number[] = [];
  const featureStds: number[] = [];

  for (let f = 0; f < 5; f++) {
    let sum = 0;
    for (let r = 0; r < nRows; r++) sum += gArrays[f][r];
    const mean = sum / nRows;
    featureMeans.push(mean);

    let sumSq = 0;
    for (let r = 0; r < nRows; r++) sumSq += (gArrays[f][r] - mean) ** 2;
    // Sample std (ddof=1) matching pandas .std() default
    featureStds.push(nRows > 1 ? Math.sqrt(sumSq / (nRows - 1)) || 1 : 1);
  }

  // Per-well target z-score stats (from non-NaN PCHIP values in valid GLDAS rows)
  // Matches Python: combined_df[well_names].mean() / .std() (per-column, skipna=True, ddof=1)
  const wellTargetStats = new Map<string, { mean: number; std: number }>();
  const wellPchipAtRows = new Map<string, (number | null)[]>();

  for (const { well } of activeWells) {
    const pchipFull = wellPchip.get(well.id);
    if (!pchipFull) continue;

    // Get PCHIP values at valid GLDAS rows
    const pchipAtRows = validRowIndices.map(i => pchipFull[i]);
    wellPchipAtRows.set(well.id, pchipAtRows);

    const nonNull = pchipAtRows.filter((v): v is number => v !== null);
    if (nonNull.length < 2) continue;

    const mean = nonNull.reduce((a, b) => a + b, 0) / nonNull.length;
    let sumSq = 0;
    for (const v of nonNull) sumSq += (v - mean) ** 2;
    // Sample std (ddof=1)
    const std = Math.sqrt(sumSq / (nonNull.length - 1)) || 1;

    wellTargetStats.set(well.id, { mean, std });
  }

  // ===== PHASE D: Build feature matrix for ALL valid rows =====
  // Columns: 5 z-scored GLDAS + 1 min-max year + 12 one-hot month + 1 bias = 19
  // Matches Python: names = ['soilw', ..., 'soilw_yr10', 'year', 'month_1', ..., 'month_12'] + bias
  onProgress('Building feature matrix...', 18);
  await yieldToUI();

  // Year min-max normalization (matches Python: (year - min) / (max - min))
  const years = validRowIndices.map(i => new Date(monthlyDates[i]).getUTCFullYear());
  const yearMin = Math.min(...years);
  const yearMax = Math.max(...years);
  const yearRange = yearMax - yearMin || 1;

  const allFeatures: number[][] = [];
  for (let r = 0; r < nRows; r++) {
    const d = new Date(monthlyDates[validRowIndices[r]]);
    const year = d.getUTCFullYear();
    const month = d.getUTCMonth(); // 0-11 (matches Python's 1-12 one-hot encoding order)

    // Z-score GLDAS features (using global stats)
    const zGldas = gArrays.map((arr, f) => (arr[r] - featureMeans[f]) / featureStds[f]);

    // Min-max normalized year
    const normYear = (year - yearMin) / yearRange;

    // One-hot month (12 columns)
    const monthOneHot = new Array(12).fill(0);
    monthOneHot[month] = 1;

    // Combine: 5 GLDAS + 1 year + 12 month + 1 bias = 19
    // Bias matches Python: np.hstack((tx, np.ones(n).T))
    allFeatures.push([...zGldas, normYear, ...monthOneHot, 1.0]);
  }

  // ===== PHASE E: Per-well ELM training and prediction =====
  // Matches Python: impute_data(norm_df, well_names, names) — trains ELM for EVERY well
  const allDataRows: ImputationDataRow[] = [];
  const wellMetrics: Record<string, ImputationWellMetrics> = {};

  for (let wi = 0; wi < activeWells.length; wi++) {
    const { well } = activeWells[wi];
    const pct = 20 + (wi / activeWells.length) * 70;
    onProgress(`ELM training well ${wi + 1}/${activeWells.length}...`, pct);

    const targetStats = wellTargetStats.get(well.id);
    const pchipAtRows = wellPchipAtRows.get(well.id);
    if (!targetStats || !pchipAtRows) {
      onLog(`Well ${well.name}: skipped (insufficient PCHIP overlap with GLDAS)`);
      continue;
    }

    // Z-score the PCHIP values for this well
    // Matches Python: norm_df[well] = (combined_df[well] - mean) / std
    const zPchip: (number | null)[] = pchipAtRows.map(v =>
      v !== null ? (v - targetStats.mean) / targetStats.std : null
    );

    // Training rows: where z-scored PCHIP is non-NaN
    // Matches Python: train_nona_df = comb_df.dropna(subset=[well])
    const trainIndices: number[] = [];
    const trainTargets: number[] = [];
    for (let r = 0; r < nRows; r++) {
      if (zPchip[r] !== null) {
        trainIndices.push(r);
        trainTargets.push(zPchip[r]!);
      }
    }

    if (trainIndices.length < 3) {
      onLog(`Well ${well.name}: too few training points (${trainIndices.length}), skipping`);
      continue;
    }

    // Training feature matrix (subset of allFeatures)
    // Matches Python: tx = train_nona_df[names].values
    const trainX = trainIndices.map(r => allFeatures[r]);

    try {
      // Train ELM on pre-normalized data
      // Matches Python: W_out = np.linalg.lstsq(X.T.dot(X) + lamb*I, X.T.dot(ty))
      const elmResult = trainElm(trainX, trainTargets, hiddenUnits, lambda);

      // Compute R²/RMSE on original (denormalized) scale
      const trainPredDenorm = elmResult.trainPredictions.map(v => v * targetStats.std + targetStats.mean);
      const trainTargetsDenorm = trainTargets.map(v => v * targetStats.std + targetStats.mean);
      const trainMean = trainTargetsDenorm.reduce((a, b) => a + b, 0) / trainTargetsDenorm.length;
      let ssTot = 0, ssRes = 0;
      for (let i = 0; i < trainTargetsDenorm.length; i++) {
        ssTot += (trainTargetsDenorm[i] - trainMean) ** 2;
        ssRes += (trainTargetsDenorm[i] - trainPredDenorm[i]) ** 2;
      }
      const r2 = ssTot === 0 ? 0 : 1 - ssRes / ssTot;
      const rmse = Math.sqrt(ssRes / trainTargetsDenorm.length);

      wellMetrics[well.id] = { r2, rmse };
      onLog(`Well ${well.name}: ELM R²=${r2.toFixed(4)}, RMSE=${rmse.toFixed(2)} (${trainIndices.length} training pts)`);

      // Predict for ALL valid rows (full extrapolation)
      // Matches Python: predict(all_tx_values, W_in, b, W_out)
      const allPredNorm = predictElm(elmResult.model, allFeatures);

      // Denormalize predictions
      // Matches Python: reverse_zscore_data(imputed_norm_df, ref_df)
      // = imputed * ref_df.std() + ref_df.mean()
      const allPredDenorm = allPredNorm.map(v => v * targetStats.std + targetStats.mean);

      // Assemble data rows: PCHIP where available, else ELM
      const pchipFull = wellPchip.get(well.id)!;
      for (let r = 0; r < nRows; r++) {
        const monthlyIdx = validRowIndices[r];
        const pchipVal = pchipFull[monthlyIdx];
        const modelVal = allPredDenorm[r];
        const combined = pchipVal !== null ? pchipVal : modelVal;

        allDataRows.push({
          well_id: well.id,
          date: monthlyDates[monthlyIdx],
          model: modelVal,
          pchip: pchipVal,
          combined,
        });
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      onLog(`Well ${well.name}: ELM training failed: ${msg}`);
    }

    if (wi % 3 === 0) await yieldToUI();
  }

  // ===== Step 5: Save result (90-100%) =====
  onProgress('Saving model...', 90);
  await yieldToUI();

  const code = slugify(title);
  const aquiferSlug = slugify(aquifer.name);
  const filePath = `${region.id}/${aquiferSlug}/model_wte_${code}.json`;

  // Filter output rows to the user's requested date range
  const outputStartTs = new Date(startDate).getTime();
  const outputEndTs = new Date(endDate).getTime();
  const outputRows = allDataRows.filter(row => {
    const ts = new Date(row.date).getTime();
    return ts >= outputStartTs && ts <= outputEndTs;
  });

  const params: ImputationParams = {
    startDate,
    endDate,
    gldasStartDate: gldasFeatureStart,
    gldasEndDate: gldasFeatureEnd,
    minSamples,
    gapSize,
    padSize,
    hiddenUnits,
    lambda,
  };

  const result: ImputationModelResult = {
    title,
    code,
    aquiferId: aquifer.id,
    aquiferName: aquifer.name,
    regionId: region.id,
    dataType: 'wte',
    filePath,
    createdAt: new Date().toISOString(),
    params,
    wellMetrics,
    data: outputRows,
    log: [], // Will be populated by caller from accumulated log messages
  };

  // Save to disk
  await fetch('/api/save-data', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      files: [{
        path: filePath,
        content: JSON.stringify(result),
      }],
    }),
  });

  onProgress('Complete!', 100);
  onLog('Imputation complete! Model saved.');

  return result;
}
