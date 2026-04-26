import fs from 'node:fs';
import path from 'node:path';
import { trainElm, predictElm } from '../services/elm';
import { interpolatePCHIP } from '../utils/interpolation';
import { runBrowserMcLnnExact, BrowserMcLnnInputWell } from '../services/mcLnnBrowser';

type GldasRow = {
  date: string;
  soilw: number;
  soilw_yr01: number;
  soilw_yr03: number;
  soilw_yr05: number;
  soilw_yr10: number;
};

type Metrics = {
  mae: number;
  mse: number;
  rmse: number;
  r2: number;
  n: number;
};

type WellSeries = {
  wellId: string;
  values: Array<number | null>;
  observedCount: number;
};

const ROOT = '/tmp/aquiferx_repo';
const START_DATE = '2000-01-01';
const END_DATE = '2023-12-01';
const TOP_N_WELLS = Number(process.env.TOP_N_WELLS ?? 10);
const REPEATS = Number(process.env.REPEATS ?? 3);
const HOLDOUT_FRAC = 0.2;
const MIN_REMAINING_OBS = 6;
const GAP_SIZE_DAYS = 730;
const PAD_SIZE_DAYS = 180;
const HIDDEN_UNITS = 500;
const LAMBDA = 100;

function seededRandom(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

function withSeededMathRandom<T>(seed: number, fn: () => T): T {
  const orig = Math.random;
  const rand = seededRandom(seed);
  (Math as unknown as { random: () => number }).random = rand;
  try {
    return fn();
  } finally {
    (Math as unknown as { random: () => number }).random = orig;
  }
}

function mean(values: number[]): number {
  return values.reduce((a, b) => a + b, 0) / values.length;
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
    if (month > 11) {
      month = 0;
      year++;
    }
  }
  return dates;
}

function parseSimpleCsv(filePath: string): string[][] {
  const text = fs.readFileSync(filePath, 'utf8').trim();
  const lines = text.split(/\r?\n/);
  return lines.map(line => {
    const out: string[] = [];
    let cur = '';
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '"') {
        if (inQuotes && line[i + 1] === '"') {
          cur += '"';
          i++;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (ch === ',' && !inQuotes) {
        out.push(cur);
        cur = '';
      } else {
        cur += ch;
      }
    }
    out.push(cur);
    return out;
  });
}

function loadGldasRows(): GldasRow[] {
  const filePath = path.join(ROOT, 'python/mc_lnn_imputer/datas/lnn_imputation_gslb_gldas_df_excercise.csv');
  const rows = parseSimpleCsv(filePath);
  const header = rows[0];
  const idx = Object.fromEntries(header.map((h, i) => [h, i]));
  return rows.slice(1)
    .map(cols => ({
      date: cols[idx.time],
      soilw: Number(cols[idx.soilw]),
      soilw_yr01: Number(cols[idx.soilw_yr01]),
      soilw_yr03: Number(cols[idx.soilw_yr03]),
      soilw_yr05: Number(cols[idx.soilw_yr05]),
      soilw_yr10: Number(cols[idx.soilw_yr10]),
    }))
    .filter(r => r.date >= START_DATE && r.date <= END_DATE);
}

function loadMonthlyWellSeries(monthDates: string[]): WellSeries[] {
  const filePath = path.join(ROOT, 'public/data/great-salt-lake-basin/data_wte.csv');
  const rows = parseSimpleCsv(filePath);
  const header = rows[0];
  const idx = Object.fromEntries(header.map((h, i) => [h, i]));
  const monthSet = new Set(monthDates);
  const byWell = new Map<string, Map<string, number[]>>();
  for (const cols of rows.slice(1)) {
    const date = cols[idx.date];
    const month = `${date.slice(0, 7)}-01`;
    if (!monthSet.has(month)) continue;
    const wellId = cols[idx.well_id];
    const value = Number(cols[idx.value]);
    if (!Number.isFinite(value)) continue;
    if (!byWell.has(wellId)) byWell.set(wellId, new Map());
    const perMonth = byWell.get(wellId)!;
    if (!perMonth.has(month)) perMonth.set(month, []);
    perMonth.get(month)!.push(value);
  }

  const all: WellSeries[] = [];
  for (const [wellId, perMonth] of byWell.entries()) {
    const values = monthDates.map(month => {
      const vals = perMonth.get(month);
      return vals?.length ? mean(vals) : null;
    });
    const observedCount = values.filter(v => v != null).length;
    all.push({ wellId, values, observedCount });
  }
  all.sort((a, b) => b.observedCount - a.observedCount || a.wellId.localeCompare(b.wellId));
  return all.slice(0, TOP_N_WELLS);
}

function computeFeatureStats(gldasRows: GldasRow[]) {
  const arrays = [
    gldasRows.map(r => r.soilw),
    gldasRows.map(r => r.soilw_yr01),
    gldasRows.map(r => r.soilw_yr03),
    gldasRows.map(r => r.soilw_yr05),
    gldasRows.map(r => r.soilw_yr10),
  ];
  const means = arrays.map(arr => mean(arr));
  const stds = arrays.map((arr, i) => {
    const m = means[i];
    const variance = arr.reduce((acc, v) => acc + (v - m) ** 2, 0) / Math.max(1, arr.length - 1);
    return Math.sqrt(variance) || 1;
  });
  return { arrays, means, stds };
}

function buildAllFeatures(monthDates: string[], gldasRows: GldasRow[], featureStats: ReturnType<typeof computeFeatureStats>): number[][] {
  const years = monthDates.map(d => new Date(d).getUTCFullYear());
  const yearMin = Math.min(...years);
  const yearMax = Math.max(...years);
  const yearRange = yearMax - yearMin || 1;

  return monthDates.map((date, r) => {
    const d = new Date(date);
    const month = d.getUTCMonth();
    const year = d.getUTCFullYear();
    const zGldas = featureStats.arrays.map((arr, f) => (arr[r] - featureStats.means[f]) / featureStats.stds[f]);
    const normYear = (year - yearMin) / yearRange;
    const monthOneHot = new Array(12).fill(0);
    monthOneHot[month] = 1;
    return [...zGldas, normYear, ...monthOneHot, 1.0];
  });
}

function runOriginalElmImputer(
  maskedValues: Array<number | null>,
  monthDates: string[],
  allFeatures: number[][],
): Array<number | null> {
  const monthlyTimestamps = monthDates.map(d => new Date(d).getTime());
  const observed = maskedValues
    .map((v, i) => (v != null ? { ts: monthlyTimestamps[i], value: v, i } : null))
    .filter((v): v is { ts: number; value: number; i: number } => v != null);

  if (observed.length < 2) {
    return [...maskedValues];
  }

  const measTimestamps = observed.map(o => o.ts);
  const measValues = observed.map(o => o.value);
  const firstMeasTs = measTimestamps[0];
  const lastMeasTs = measTimestamps[measTimestamps.length - 1];
  const pchipFull: Array<number | null> = new Array(monthDates.length).fill(null);

  const inRangeIndices: number[] = [];
  const inRangeTimestamps: number[] = [];
  for (let i = 0; i < monthDates.length; i++) {
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

  const gapSizeMs = GAP_SIZE_DAYS * 24 * 60 * 60 * 1000;
  const padSizeMs = PAD_SIZE_DAYS * 24 * 60 * 60 * 1000;
  for (let i = 0; i < observed.length - 1; i++) {
    const gapMs = observed[i + 1].ts - observed[i].ts;
    if (gapMs > gapSizeMs) {
      const padStartTs = observed[i].ts + padSizeMs;
      const padEndTs = observed[i + 1].ts - padSizeMs;
      for (const idx of inRangeIndices) {
        const ts = monthlyTimestamps[idx];
        if (ts >= padStartTs && ts <= padEndTs) pchipFull[idx] = null;
      }
    }
  }

  const nonNull = pchipFull.filter((v): v is number => v != null);
  if (nonNull.length < 3) return pchipFull;
  const targetMean = mean(nonNull);
  const variance = nonNull.reduce((acc, v) => acc + (v - targetMean) ** 2, 0) / Math.max(1, nonNull.length - 1);
  const targetStd = Math.sqrt(variance) || 1;
  const zPchip = pchipFull.map(v => (v != null ? (v - targetMean) / targetStd : null));

  const trainIndices: number[] = [];
  const trainTargets: number[] = [];
  for (let r = 0; r < zPchip.length; r++) {
    if (zPchip[r] != null) {
      trainIndices.push(r);
      trainTargets.push(zPchip[r] as number);
    }
  }
  if (trainIndices.length < 3) return pchipFull;

  const trainX = trainIndices.map(r => allFeatures[r]);
  let allPredDenorm: number[];
  try {
    const elmResult = withSeededMathRandom(20260426 + trainIndices.length, () =>
      trainElm(trainX, trainTargets, HIDDEN_UNITS, LAMBDA),
    );
    const allPredNorm = withSeededMathRandom(20260426 + trainIndices.length, () =>
      predictElm(elmResult.model, allFeatures),
    );
    allPredDenorm = allPredNorm.map(v => v * targetStd + targetMean);
  } catch {
    return pchipFull;
  }

  return pchipFull.map((v, i) => (v != null ? v : allPredDenorm[i]));
}

function computeMetrics(obs: number[], pred: number[]): Metrics {
  const n = obs.length;
  const mae = obs.reduce((acc, v, i) => acc + Math.abs(pred[i] - v), 0) / n;
  const mse = obs.reduce((acc, v, i) => acc + (pred[i] - v) ** 2, 0) / n;
  const rmse = Math.sqrt(mse);
  const obsMean = mean(obs);
  const ssTot = obs.reduce((acc, v) => acc + (v - obsMean) ** 2, 0);
  const ssRes = obs.reduce((acc, v, i) => acc + (v - pred[i]) ** 2, 0);
  const r2 = ssTot === 0 ? 0 : 1 - ssRes / ssTot;
  return { mae, mse, rmse, r2, n };
}

async function main() {
  const gldasRows = loadGldasRows();
  const monthDates = gldasRows.map(r => r.date);
  const topWells = loadMonthlyWellSeries(monthDates);
  const featureStats = computeFeatureStats(gldasRows);
  const allFeatures = buildAllFeatures(monthDates, gldasRows, featureStats);

  const originalObs: number[] = [];
  const originalPred: number[] = [];
  const oursObs: number[] = [];
  const oursPred: number[] = [];

  const summaryRows: string[] = ['repeat,well_id,withheld_n,original_r2,original_rmse,ours_r2,ours_rmse'];

  for (let repeat = 0; repeat < REPEATS; repeat++) {
    const maskedByWell = new Map<string, Array<number | null>>();
    const holdoutsByWell = new Map<string, number[]>();

    for (const well of topWells) {
      const observedIdx = well.values.map((v, i) => (v != null ? i : -1)).filter(i => i >= 0);
      const targetHoldout = Math.max(1, Math.round(observedIdx.length * HOLDOUT_FRAC));
      const maxHoldout = Math.max(1, observedIdx.length - MIN_REMAINING_OBS);
      const nHoldout = Math.min(targetHoldout, maxHoldout);
      const rng = seededRandom(1000 + repeat * 97 + observedIdx.length);
      const pool = [...observedIdx];
      const picked: number[] = [];
      for (let k = 0; k < nHoldout && pool.length; k++) {
        const idx = Math.floor(rng() * pool.length);
        picked.push(pool.splice(idx, 1)[0]);
      }
      picked.sort((a, b) => a - b);
      const masked = [...well.values];
      for (const idx of picked) masked[idx] = null;
      maskedByWell.set(well.wellId, masked);
      holdoutsByWell.set(well.wellId, picked);
    }

    const browserInput: BrowserMcLnnInputWell[] = topWells.map(well => {
      const masked = maskedByWell.get(well.wellId)!;
      return {
        wellId: well.wellId,
        points: monthDates.map((date, i) => ({
          date,
          time: i,
          observed: masked[i],
          auxiliaries: [
            gldasRows[i].soilw,
            gldasRows[i].soilw_yr01,
            gldasRows[i].soilw_yr03,
            gldasRows[i].soilw_yr05,
            gldasRows[i].soilw_yr10,
          ],
        })),
      };
    });

    const browserResult = await runBrowserMcLnnExact(browserInput, {
      smallGapThresholdMonths: Math.max(1, Math.round(GAP_SIZE_DAYS / 30)),
      outerIterations: 2,
      feedbackPrevWeight: 0.35,
      softImpute: {
        rank: 12,
        maxIterations: 100,
        shrinkage: 0,
        tolerance: 1e-5,
      },
    });
    const browserByWell = new Map(browserResult.wells.map(w => [w.wellId, w]));

    for (const well of topWells) {
      const masked = maskedByWell.get(well.wellId)!;
      const holdouts = holdoutsByWell.get(well.wellId)!;
      const originalFilled = runOriginalElmImputer(masked, monthDates, allFeatures);
      const browserFilled = browserByWell.get(well.wellId)!.rows.map(r => r.final);

      const obsVals: number[] = [];
      const origVals: number[] = [];
      const ourVals: number[] = [];
      for (const idx of holdouts) {
        const truth = well.values[idx];
        const origPred = originalFilled[idx];
        const ourPred = browserFilled[idx];
        if (truth != null && origPred != null && ourPred != null) {
          obsVals.push(truth);
          origVals.push(origPred);
          ourVals.push(ourPred);
          originalObs.push(truth);
          originalPred.push(origPred);
          oursObs.push(truth);
          oursPred.push(ourPred);
        }
      }
      if (obsVals.length) {
        const om = computeMetrics(obsVals, origVals);
        const mm = computeMetrics(obsVals, ourVals);
        summaryRows.push([
          repeat,
          well.wellId,
          obsVals.length,
          om.r2.toFixed(6),
          om.rmse.toFixed(6),
          mm.r2.toFixed(6),
          mm.rmse.toFixed(6),
        ].join(','));
      }
    }
  }

  const originalMetrics = computeMetrics(originalObs, originalPred);
  const ourMetrics = computeMetrics(oursObs, oursPred);

  const outDir = path.join(ROOT, 'benchmark_outputs');
  fs.mkdirSync(outDir, { recursive: true });
  const summaryPath = path.join(outDir, 'gslb_browser_imputer_benchmark_summary.txt');
  const detailPath = path.join(outDir, 'gslb_browser_imputer_benchmark_by_well_repeat.csv');

  const summary = [
    'GSLB Browser Imputer Benchmark',
    `Date overlap: ${START_DATE} to ${END_DATE} (2024 unavailable in bundled observed/GLDAS data)`,
    `Top wells by monthly coverage: ${topWells.map(w => `${w.wellId}(${w.observedCount})`).join(', ')}`,
    `Repeats: ${REPEATS}`,
    `Holdout fraction per well: ${HOLDOUT_FRAC}`,
    '',
    'Pooled held-out metrics',
    `Existing PCHIP + ELM: n=${originalMetrics.n}, R2=${originalMetrics.r2.toFixed(4)}, RMSE=${originalMetrics.rmse.toFixed(4)}, MAE=${originalMetrics.mae.toFixed(4)}, MSE=${originalMetrics.mse.toFixed(4)}`,
    `Browser MC + LNN:    n=${ourMetrics.n}, R2=${ourMetrics.r2.toFixed(4)}, RMSE=${ourMetrics.rmse.toFixed(4)}, MAE=${ourMetrics.mae.toFixed(4)}, MSE=${ourMetrics.mse.toFixed(4)}`,
  ].join('\n');

  fs.writeFileSync(summaryPath, `${summary}\n`, 'utf8');
  fs.writeFileSync(detailPath, `${summaryRows.join('\n')}\n`, 'utf8');

  console.log(summary);
  console.log(`\nWrote:\n- ${summaryPath}\n- ${detailPath}`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
