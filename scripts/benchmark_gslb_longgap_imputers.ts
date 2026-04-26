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
const TOP_N_WELLS = Number(process.env.TOP_N_WELLS ?? 5);
const REPEATS = Number(process.env.REPEATS ?? 2);
const DURATIONS_YEARS = (process.env.DURATIONS_YEARS ?? '1,2,3,4,5').split(',').map(s => Number(s.trim())).filter(Number.isFinite);
const MIN_REMAINING_OBS = 12;
const MIN_TRUTH_IN_WINDOW = 6;
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

function runOriginalElmImputer(maskedValues: Array<number | null>, monthDates: string[], allFeatures: number[][]): Array<number | null> {
  const monthlyTimestamps = monthDates.map(d => new Date(d).getTime());
  const observed = maskedValues
    .map((v, i) => (v != null ? { ts: monthlyTimestamps[i], value: v } : null))
    .filter((v): v is { ts: number; value: number } => v != null);
  if (observed.length < 2) return [...maskedValues];

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
    for (let j = 0; j < inRangeIndices.length; j++) pchipFull[inRangeIndices[j]] = pchipValues[j];
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
  try {
    const elmResult = withSeededMathRandom(20260426 + trainIndices.length, () =>
      trainElm(trainX, trainTargets, HIDDEN_UNITS, LAMBDA),
    );
    const allPredNorm = withSeededMathRandom(20260426 + trainIndices.length, () =>
      predictElm(elmResult.model, allFeatures),
    );
    const allPredDenorm = allPredNorm.map(v => v * targetStd + targetMean);
    return pchipFull.map((v, i) => (v != null ? v : allPredDenorm[i]));
  } catch {
    return pchipFull;
  }
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

function chooseLongGapStart(values: Array<number | null>, gapMonths: number, rng: () => number): number | null {
  const candidates: number[] = [];
  for (let start = 0; start + gapMonths <= values.length; start++) {
    const inWindow = values.slice(start, start + gapMonths);
    const truthCount = inWindow.filter(v => v != null).length;
    const remainingObs = values.filter((v, i) => v != null && (i < start || i >= start + gapMonths)).length;
    if (truthCount >= MIN_TRUTH_IN_WINDOW && remainingObs >= MIN_REMAINING_OBS) {
      candidates.push(start);
    }
  }
  if (!candidates.length) return null;
  return candidates[Math.floor(rng() * candidates.length)];
}

async function main() {
  const gldasRows = loadGldasRows();
  const monthDates = gldasRows.map(r => r.date);
  const topWells = loadMonthlyWellSeries(monthDates);
  const featureStats = computeFeatureStats(gldasRows);
  const allFeatures = buildAllFeatures(monthDates, gldasRows, featureStats);

  const outDir = path.join(ROOT, 'benchmark_outputs');
  fs.mkdirSync(outDir, { recursive: true });

  const detailRows: string[] = ['duration_years,repeat,well_id,start_idx,start_date,end_date,withheld_n,original_r2,original_rmse,ours_r2,ours_rmse'];
  const summaryLines: string[] = [
    'GSLB Long-Gap Browser Imputer Benchmark',
    `Date overlap: ${START_DATE} to ${END_DATE} (2024 unavailable in bundled observed/GLDAS data)`,
    `Top wells by monthly coverage: ${topWells.map(w => `${w.wellId}(${w.observedCount})`).join(', ')}`,
    `Repeats per duration: ${REPEATS}`,
    `Durations (years): ${DURATIONS_YEARS.join(', ')}`,
    '',
  ];

  for (const years of DURATIONS_YEARS) {
    const gapMonths = years * 12;
    const originalObs: number[] = [];
    const originalPred: number[] = [];
    const oursObs: number[] = [];
    const oursPred: number[] = [];

    for (let repeat = 0; repeat < REPEATS; repeat++) {
      const maskedByWell = new Map<string, Array<number | null>>();
      const holdoutsByWell = new Map<string, number[]>();

      const windowByWell = new Map<string, { start: number; startDate: string; endDate: string }>();
      for (const well of topWells) {
        const rng = seededRandom(5000 + years * 101 + repeat * 17 + well.observedCount);
        const start = chooseLongGapStart(well.values, gapMonths, rng);
        if (start == null) continue;
        const masked = [...well.values];
        const holdouts: number[] = [];
        for (let i = start; i < start + gapMonths; i++) {
          if (masked[i] != null) holdouts.push(i);
          masked[i] = null;
        }
        maskedByWell.set(well.wellId, masked);
        holdoutsByWell.set(well.wellId, holdouts);
        windowByWell.set(well.wellId, {
          start,
          startDate: monthDates[start],
          endDate: monthDates[Math.min(monthDates.length - 1, start + gapMonths - 1)],
        });
      }

      const browserInput: BrowserMcLnnInputWell[] = topWells
        .filter(well => maskedByWell.has(well.wellId))
        .map(well => ({
          wellId: well.wellId,
          points: monthDates.map((date, i) => ({
            date,
            time: i,
            observed: maskedByWell.get(well.wellId)![i],
            auxiliaries: [
              gldasRows[i].soilw,
              gldasRows[i].soilw_yr01,
              gldasRows[i].soilw_yr03,
              gldasRows[i].soilw_yr05,
              gldasRows[i].soilw_yr10,
            ],
          })),
        }));

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
        const masked = maskedByWell.get(well.wellId);
        const holdouts = holdoutsByWell.get(well.wellId);
        const window = windowByWell.get(well.wellId);
        if (!masked || !holdouts || !window) continue;
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
          detailRows.push([
            years,
            repeat,
            well.wellId,
            window.start,
            window.startDate,
            window.endDate,
            obsVals.length,
            om.r2.toFixed(6),
            om.rmse.toFixed(6),
            mm.r2.toFixed(6),
            mm.rmse.toFixed(6),
          ].join(','));
        }
      }
    }

    const om = computeMetrics(originalObs, originalPred);
    const mm = computeMetrics(oursObs, oursPred);
    summaryLines.push(`${years}y long gap`);
    summaryLines.push(`Existing PCHIP + ELM: n=${om.n}, R2=${om.r2.toFixed(4)}, RMSE=${om.rmse.toFixed(4)}, MAE=${om.mae.toFixed(4)}, MSE=${om.mse.toFixed(4)}`);
    summaryLines.push(`Browser MC + LNN:    n=${mm.n}, R2=${mm.r2.toFixed(4)}, RMSE=${mm.rmse.toFixed(4)}, MAE=${mm.mae.toFixed(4)}, MSE=${mm.mse.toFixed(4)}`);
    summaryLines.push('');
  }

  const summaryPath = path.join(outDir, 'gslb_longgap_browser_imputer_benchmark_summary.txt');
  const detailPath = path.join(outDir, 'gslb_longgap_browser_imputer_benchmark_by_well_repeat.csv');
  fs.writeFileSync(summaryPath, `${summaryLines.join('\n')}\n`, 'utf8');
  fs.writeFileSync(detailPath, `${detailRows.join('\n')}\n`, 'utf8');
  console.log(summaryLines.join('\n'));
  console.log(`Wrote:\n- ${summaryPath}\n- ${detailPath}`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
