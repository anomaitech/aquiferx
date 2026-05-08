/**
 * Pure browser MC-LNN imputation pipeline.
 *
 * Replaces ELM with MC-LNN for large-gap imputation while keeping
 * the same PCHIP + GLDAS pipeline structure as the original.
 *
 * Pipeline: PCHIP (small gaps) → MC SoftImpute (spatial) → LNN CFC (temporal)
 */

import {
  Aquifer, Region, Well, Measurement,
  ImputationParams, ImputationDataRow, ImputationModelResult, ImputationWellMetrics,
} from '../types';
import { fetchGldasFeatures, GldasFeatures } from './gldasFetch';
import { interpolatePCHIP } from '../utils/interpolation';
import { computeKGE } from './mcLnnPureBrowser';
import type { McRunConfig } from './mcLnnPureBrowser';
import { slugify } from '../utils/strings';

export interface McLnnPurePipelineInput {
  title: string;
  startDate: string;
  endDate: string;
  gldasStartDate: string;
  gldasEndDate: string;
  minSamples: number;
  gapSize: number;    // days (default: 730)
  padSize: number;    // days (default: 180)
  hiddenUnits: number; // unused, kept for compatibility
  lambda: number;      // unused, kept for compatibility
}

function yieldToUI(): Promise<void> {
  return new Promise(r => setTimeout(r, 0));
}

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
  startDate: string, endDate: string, gldasStartDate: string, gldasEndDate: string,
): { min: string; max: string } | null {
  const min = gldasStartDate > startDate ? gldasStartDate : startDate;
  const max = gldasEndDate < endDate ? gldasEndDate : endDate;
  return min <= max ? { min, max } : null;
}

const MS_PER_DAY = 24 * 60 * 60 * 1000;

function mean(values: number[]): number | null {
  if (!values.length) return null;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

export async function runMcLnnPureImputationPipeline(
  input: McLnnPurePipelineInput,
  aquifer: Aquifer,
  region: Region,
  wells: Well[],
  measurements: Measurement[],
  onLog: (msg: string) => void,
  onProgress: (step: string, pct: number) => void,
): Promise<ImputationModelResult> {
  const { title, startDate, endDate, minSamples, gapSize, padSize } = input;
  const effectiveRange = clampDateRange(startDate, endDate, input.gldasStartDate, input.gldasEndDate);
  if (!effectiveRange) throw new Error('Selected date range does not overlap the GLDAS feature range.');

  const gapSizeMs = gapSize * MS_PER_DAY;
  const padSizeMs = padSize * MS_PER_DAY;

  // ===== Step 1: Fetch GLDAS =====
  onProgress('Fetching GLDAS data...', 0);
  onLog('Fetching GLDAS soil moisture data...');
  await yieldToUI();

  const gldas = await fetchGldasFeatures(aquifer.id, aquifer.geojson, input.gldasStartDate, input.gldasEndDate);
  onLog(`GLDAS data loaded: ${gldas.dates.length} monthly records`);
  onProgress('GLDAS data loaded', 5);

  // ===== Step 2: Prepare well data =====
  onProgress('Preparing well data...', 5);
  await yieldToUI();

  const wteMeasurements = measurements.filter(m => m.dataType === 'wte');
  const byWell = new Map<string, Measurement[]>();
  for (const m of wteMeasurements) {
    if (m.date < effectiveRange.min || m.date > effectiveRange.max) continue;
    if (!byWell.has(m.wellId)) byWell.set(m.wellId, []);
    byWell.get(m.wellId)!.push(m);
  }

  interface QualifiedWell {
    well: Well;
    sorted: { date: string; ts: number; value: number }[];
  }
  const qualifiedWells: QualifiedWell[] = [];

  for (const well of wells) {
    const meas = byWell.get(well.id);
    if (!meas || meas.length < minSamples) continue;
    const sorted = [...meas]
      .filter(m => !isNaN(new Date(m.date).getTime()))
      .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
      .map(m => ({ date: m.date, ts: new Date(m.date).getTime(), value: m.value }));
    if (sorted.length < 2) continue;
    qualifiedWells.push({ well, sorted });
  }

  onLog(`${qualifiedWells.length} wells qualified (>= ${minSamples} measurements)`);
  if (!qualifiedWells.length) throw new Error(`No wells have >= ${minSamples} measurements.`);
  onProgress('Well data prepared', 10);

  // ===== Step 3: Monthly date grid =====
  const gldasFeatureStart = gldas.dates[0];
  const gldasFeatureEnd = gldas.dates[gldas.dates.length - 1];
  const monthlyDates = generateMonthlyDates(gldasFeatureStart, gldasFeatureEnd);
  const monthlyTimestamps = monthlyDates.map(d => new Date(d).getTime());
  const nMonths = monthlyDates.length;

  // GLDAS lookup
  const gldasByDate = new Map<string, number>();
  for (let i = 0; i < gldas.dates.length; i++) gldasByDate.set(gldas.dates[i], i);

  // ===== Step 4: PCHIP interpolation (aquiferx-style) =====
  onProgress('PCHIP interpolation...', 12);
  await yieldToUI();

  const wellPchip = new Map<string, (number | null)[]>();

  for (const { well, sorted } of qualifiedWells) {
    const measTimestamps = sorted.map(m => m.ts);
    const measValues = sorted.map(m => m.value);
    const firstMeasTs = measTimestamps[0];
    const lastMeasTs = measTimestamps[measTimestamps.length - 1];

    // PCHIP to monthly grid within measurement range
    const pchipFull: (number | null)[] = new Array(nMonths).fill(null);
    const inRangeIndices: number[] = [];
    const inRangeTimestamps: number[] = [];
    for (let i = 0; i < nMonths; i++) {
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

    // Blank interior of large gaps (keeping pad)
    for (let i = 0; i < sorted.length - 1; i++) {
      const gapMs = measTimestamps[i + 1] - measTimestamps[i];
      if (gapMs > gapSizeMs) {
        const padStartTs = measTimestamps[i] + padSizeMs;
        const padEndTs = measTimestamps[i + 1] - padSizeMs;
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

  // Filter wells with enough PCHIP data
  const activeWells: QualifiedWell[] = [];
  for (const qw of qualifiedWells) {
    const pchip = wellPchip.get(qw.well.id);
    if (!pchip) continue;
    const nonNullCount = pchip.filter(v => v !== null).length;
    if (nonNullCount >= minSamples) {
      activeWells.push(qw);
    } else {
      wellPchip.delete(qw.well.id);
    }
  }

  onLog(`PCHIP interpolation complete: ${activeWells.length} wells retained`);

  // ===== Step 5: Build aux data =====
  const auxData: number[][] = monthlyDates.map(d => {
    const gi = gldasByDate.get(d);
    if (gi === undefined) return [0, 0, 0, 0, 0];
    return [
      gldas.soilw[gi],
      gldas.soilw_yr01[gi],
      gldas.soilw_yr03[gi],
      gldas.soilw_yr05[gi],
      gldas.soilw_yr10[gi],
    ];
  });

  // ===== Step 6: MC-LNN imputation =====
  onProgress('Running MC-LNN imputation...', 20);
  await yieldToUI();

  // Build well series map for MC-LNN
  const wellSeriesMap = new Map<string, (number | null)[]>();
  for (const { well } of activeWells) {
    wellSeriesMap.set(well.id, wellPchip.get(well.id)!);
  }

  // Find wells that have remaining gaps (need MC-LNN)
  const wellsWithGaps: string[] = [];
  for (const { well } of activeWells) {
    const pchip = wellPchip.get(well.id)!;
    if (pchip.some(v => v === null)) wellsWithGaps.push(well.id);
  }

  onLog(`${wellsWithGaps.length} wells have remaining gaps for MC-LNN`);

  // Run MC-LNN on wells with gaps via Web Worker (non-blocking)
  let mcLnnResults = new Map<string, number[]>();
  if (wellsWithGaps.length > 0) {
    mcLnnResults = await new Promise<Map<string, number[]>>((resolve, reject) => {
      const worker = new Worker(
        new URL('./mcLnnWorker.ts', import.meta.url),
        { type: 'module' }
      );

      worker.onmessage = (e: MessageEvent) => {
        const data = e.data;
        if (data.type === 'progress') {
          onProgress(`MC-LNN: ${data.message}`, 20 + (data.pct ?? 0) * 0.65);
          onLog(data.message);
        } else if (data.type === 'result') {
          worker.terminate();
          resolve(new Map(data.results));
        } else if (data.type === 'error') {
          worker.terminate();
          reject(new Error(data.error));
        }
      };

      worker.onerror = (err) => {
        worker.terminate();
        reject(new Error(err.message));
      };

      // Serialize Map to array for transfer
      const serializedSeries: [string, (number | null)[]][] = Array.from(wellSeriesMap.entries());
      worker.postMessage({
        type: 'run',
        wellSeries: serializedSeries,
        auxData,
        targetWellIds: wellsWithGaps,
        config: { seed: 42, maxDonors: 50 } as McRunConfig,
      });
    });
  }

  // ===== Step 7: Assemble output =====
  onProgress('Assembling output...', 85);
  await yieldToUI();

  const outputStartTs = new Date(startDate).getTime();
  const outputEndTs = new Date(endDate).getTime();
  const allDataRows: ImputationDataRow[] = [];
  const wellMetrics: Record<string, ImputationWellMetrics> = {};

  for (const { well } of activeWells) {
    const pchip = wellPchip.get(well.id)!;
    const mcLnn = mcLnnResults.get(well.id);

    const observed: number[] = [];
    const predicted: number[] = [];

    for (let i = 0; i < nMonths; i++) {
      const ts = monthlyTimestamps[i];
      if (ts < outputStartTs || ts > outputEndTs) continue;

      const pchipVal = pchip[i];
      const modelVal = mcLnn ? mcLnn[i] : null;
      const combined = pchipVal !== null ? pchipVal : (modelVal !== null && isFinite(modelVal!) ? modelVal : null);

      // Track metrics on observed points
      if (pchipVal !== null && modelVal !== null && isFinite(modelVal!)) {
        observed.push(pchipVal);
        predicted.push(modelVal!);
      }

      allDataRows.push({
        well_id: well.id,
        date: monthlyDates[i],
        pchip: pchipVal,
        model: modelVal !== null && isFinite(modelVal!) ? modelVal : null,
        combined: combined ?? 0,
      });
    }

    // Compute metrics
    if (observed.length >= 2) {
      const meanObs = observed.reduce((a, b) => a + b, 0) / observed.length;
      let ssTot = 0, ssRes = 0;
      for (let i = 0; i < observed.length; i++) {
        ssTot += (observed[i] - meanObs) ** 2;
        ssRes += (observed[i] - predicted[i]) ** 2;
      }
      const r2 = ssTot === 0 ? 0 : 1 - ssRes / ssTot;
      const rmse = Math.sqrt(ssRes / observed.length);
      wellMetrics[well.id] = { r2, rmse };
      onLog(`Well ${well.name}: MC-LNN R²=${r2.toFixed(4)}, RMSE=${rmse.toFixed(2)}`);
    }
  }

  // ===== Step 8: Save =====
  onProgress('Saving model...', 90);
  await yieldToUI();

  const code = slugify(title);
  const aquiferSlug = slugify(aquifer.name);
  const filePath = `${region.id}/${aquiferSlug}/model_wte_${code}.json`;

  const params: ImputationParams = {
    startDate,
    endDate,
    gldasStartDate: gldasFeatureStart,
    gldasEndDate: gldasFeatureEnd,
    minSamples,
    gapSize,
    padSize,
    hiddenUnits: input.hiddenUnits,
    lambda: input.lambda,
  };

  const result: ImputationModelResult = {
    title,
    code,
    method: 'browser-mc-lnn-pure',
    aquiferId: aquifer.id,
    aquiferName: aquifer.name,
    regionId: region.id,
    dataType: 'wte',
    filePath,
    createdAt: new Date().toISOString(),
    params,
    wellMetrics,
    data: allDataRows,
    log: [],
  };

  await fetch('/api/save-data', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      files: [{ path: filePath, content: JSON.stringify(result) }],
    }),
  });

  onProgress('Complete!', 100);
  onLog('MC-LNN (pure browser) imputation complete! Model saved.');

  return result;
}
