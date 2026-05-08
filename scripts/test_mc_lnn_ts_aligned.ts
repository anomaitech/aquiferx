/**
 * Test MC-LNN TypeScript using EXACT same folds as Python.
 * Loads fold indices from gslb_cv_folds.json and compares results.
 *
 * Run: npx tsx scripts/test_mc_lnn_ts_aligned.ts
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import {
  runMcLnnImputation,
  computeKGE,
  pchipFill,
} from '../services/mcLnnPureBrowser';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DATA_DIR = path.resolve(__dirname, '../../liquid-neural-imputer v2.0/datas');

const TARGET_CSV = path.join(DATA_DIR, 'measurements_till_2023_to_lnn_imputation.csv');
const AUX_CSV = path.join(DATA_DIR, 'lnn_imputation_gslb_gldas_df_excercise.csv');
const FOLDS_JSON = path.join(DATA_DIR, 'gslb_cv_folds.json');
const PYTHON_RESULTS = path.join(DATA_DIR, 'gslb_cv_python_results.json');

const DATE_START = '2000-01-01';
const DATE_END = '2023-12-21';
const AUX_COLS = ['soilw', 'soilw_yr01', 'soilw_yr03', 'soilw_yr05', 'soilw_yr10'];

function parseCSV(filepath: string): Record<string, string>[] {
  const text = fs.readFileSync(filepath, 'utf-8');
  const lines = text.trim().split('\n');
  const headers = lines[0].split(',');
  return lines.slice(1).map(line => {
    const vals = line.split(',');
    const row: Record<string, string> = {};
    for (let i = 0; i < headers.length; i++) row[headers[i]] = vals[i] ?? '';
    return row;
  });
}

function generateMonths(start: string, end: string): string[] {
  const months: string[] = [];
  const s = new Date(start);
  const e = new Date(end);
  let y = s.getFullYear(), m = s.getMonth();
  while (true) {
    const d = new Date(y, m, 1);
    if (d > e) break;
    months.push(`${y}-${String(m + 1).padStart(2, '0')}`);
    m++;
    if (m > 11) { m = 0; y++; }
  }
  return months;
}

function monthlySeries(mainDf: Record<string, string>[], wellId: string, months: string[]): (number | null)[] {
  const byMonth = new Map<string, number[]>();
  for (const row of mainDf) {
    if (row['Well_ID'] !== wellId) continue;
    const d = new Date(row['Date']);
    const ym = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`;
    if (!byMonth.has(ym)) byMonth.set(ym, []);
    byMonth.get(ym)!.push(parseFloat(row['WTE']));
  }
  return months.map(m => {
    const vals = byMonth.get(m);
    return vals ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  });
}

interface Fold {
  well_id: string;
  n_years: number;
  rep: number;
  holdout_indices: number[];
  holdout_values: Record<string, number>;
  mc_seed: number;
}

interface PyResult {
  well_id: string;
  n_years: number;
  rep: number;
  kge: number;
  rmse: number;
}

async function main() {
  console.log('Loading data...');
  const mainDf = parseCSV(TARGET_CSV);
  const auxDf = parseCSV(AUX_CSV);
  const months = generateMonths(DATE_START, DATE_END);

  const filtered = mainDf.filter(r => r['Date'] >= DATE_START && r['Date'] <= DATE_END);

  // Load folds and Python results
  const folds: Fold[] = JSON.parse(fs.readFileSync(FOLDS_JSON, 'utf-8'));
  const pyResults: PyResult[] = JSON.parse(fs.readFileSync(PYTHON_RESULTS, 'utf-8'));
  console.log(`Loaded ${folds.length} folds, ${pyResults.length} Python results`);

  // Get all well IDs needed
  const allWellIds = new Set<string>();
  for (const f of folds) allWellIds.add(f.well_id);

  // Get eligible wells for donor pool
  const wellCounts = new Map<string, number>();
  for (const r of filtered) wellCounts.set(r['Well_ID'], (wellCounts.get(r['Well_ID']) ?? 0) + 1);
  const eligible = Array.from(wellCounts.entries())
    .filter(([, n]) => n >= 10)
    .sort((a, b) => b[1] - a[1])
    .map(([w]) => w)
    .slice(0, 50);

  // Build monthly series
  const wellSeries = new Map<string, (number | null)[]>();
  for (const wid of eligible) wellSeries.set(wid, monthlySeries(filtered, wid, months));

  // PCHIP prefill all donors
  const filledSeries = new Map<string, (number | null)[]>();
  for (const [wid, raw] of Array.from(wellSeries.entries())) {
    filledSeries.set(wid, pchipFill(raw, 48));
  }

  // Build aux
  const auxByM = new Map<string, number[]>();
  for (const r of auxDf) {
    const d = new Date(r['time']);
    const ym = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`;
    auxByM.set(ym, AUX_COLS.map(c => { const v = parseFloat(r[c]); return isNaN(v) ? 0 : v; }));
  }
  const auxData = months.map(m => auxByM.get(m) ?? [0, 0, 0, 0, 0]);

  console.log(`Wells: ${eligible.length}, Months: ${months.length}`);
  console.log();

  // Run each fold with exact same holdout indices
  const tsResults: { well_id: string; n_years: number; rep: number; kge: number; rmse: number }[] = [];
  const t0 = Date.now();

  for (let fi = 0; fi < folds.length; fi++) {
    const fold = folds[fi];
    const truth = new Map<number, number>();
    for (const [k, v] of Object.entries(fold.holdout_values)) {
      truth.set(parseInt(k), v);
    }

    // Build fold: use PCHIP-filled donors, remove exact holdout indices
    const modSeries = new Map<string, (number | null)[]>();
    for (const [w, s] of Array.from(filledSeries.entries())) {
      if (w === fold.well_id) {
        const mod = [...s];
        for (const idx of fold.holdout_indices) mod[idx] = null;
        modSeries.set(w, mod);
      } else {
        modSeries.set(w, s);
      }
    }

    try {
      const imputed = await runMcLnnImputation(
        modSeries, auxData, [fold.well_id],
        { seed: fold.mc_seed, maxDonors: 50 },
      );

      const pred = imputed.get(fold.well_id);
      if (!pred) continue;

      const obsVals: number[] = [];
      const predVals: number[] = [];
      for (const [t, v] of Array.from(truth.entries())) {
        if (isFinite(pred[t])) {
          obsVals.push(v);
          predVals.push(pred[t]);
        }
      }

      if (obsVals.length >= 2) {
        const kge = computeKGE(obsVals, predVals);
        const rmse = Math.sqrt(obsVals.reduce((acc, v, i) => acc + (v - predVals[i]) ** 2, 0) / obsVals.length);
        tsResults.push({ well_id: fold.well_id, n_years: fold.n_years, rep: fold.rep, kge, rmse });
      }
    } catch {
      // skip
    }

    if ((fi + 1) % 10 === 0 || fi === folds.length - 1) {
      console.log(`  [${fi + 1}/${folds.length}] ${((Date.now() - t0) / 1000).toFixed(0)}s`);
    }
  }

  const elapsed = (Date.now() - t0) / 1000;

  // Build Python lookup
  const pyLookup = new Map<string, PyResult>();
  for (const r of pyResults) {
    pyLookup.set(`${r.well_id}_${r.n_years}_${r.rep}`, r);
  }

  // Summary comparison
  console.log(`\n${'='.repeat(70)}`);
  console.log(`Aligned Fold Comparison (${elapsed.toFixed(0)}s)`);
  console.log(`${'='.repeat(70)}`);
  console.log(`${'Gap'.padEnd(6)} ${'TS KGE'.padStart(10)} ${'Py KGE'.padStart(10)} ${'Delta'.padStart(10)} ${'TS RMSE'.padStart(10)} ${'Py RMSE'.padStart(10)} ${'Delta'.padStart(10)}`);
  console.log('-'.repeat(70));

  for (const ny of [1, 2, 3, 4, 5]) {
    const tsSub = tsResults.filter(r => r.n_years === ny);
    const pySub = pyResults.filter(r => r.n_years === ny);
    if (tsSub.length && pySub.length) {
      const tsKge = tsSub.reduce((a, r) => a + r.kge, 0) / tsSub.length;
      const pyKge = pySub.reduce((a, r) => a + r.kge, 0) / pySub.length;
      const tsRmse = tsSub.reduce((a, r) => a + r.rmse, 0) / tsSub.length;
      const pyRmse = pySub.reduce((a, r) => a + r.rmse, 0) / pySub.length;
      console.log(
        `${ny}yr${' '.repeat(3)} ${tsKge.toFixed(4).padStart(10)} ${pyKge.toFixed(4).padStart(10)} ${(tsKge - pyKge).toFixed(4).padStart(10)}` +
        ` ${tsRmse.toFixed(4).padStart(10)} ${pyRmse.toFixed(4).padStart(10)} ${(tsRmse - pyRmse).toFixed(4).padStart(10)}`
      );
    }
  }

  const tsAll = tsResults.reduce((a, r) => a + r.kge, 0) / tsResults.length;
  const pyAll = pyResults.reduce((a, r) => a + r.kge, 0) / pyResults.length;
  const tsRmseAll = tsResults.reduce((a, r) => a + r.rmse, 0) / tsResults.length;
  const pyRmseAll = pyResults.reduce((a, r) => a + r.rmse, 0) / pyResults.length;
  console.log(
    `${'ALL'.padEnd(6)} ${tsAll.toFixed(4).padStart(10)} ${pyAll.toFixed(4).padStart(10)} ${(tsAll - pyAll).toFixed(4).padStart(10)}` +
    ` ${tsRmseAll.toFixed(4).padStart(10)} ${pyRmseAll.toFixed(4).padStart(10)} ${(tsRmseAll - pyRmseAll).toFixed(4).padStart(10)}`
  );

  // Per-fold comparison
  console.log(`\nPer-fold deltas (TS - Python):`);
  let matchCount = 0, totalCompared = 0;
  for (const ts of tsResults) {
    const key = `${ts.well_id}_${ts.n_years}_${ts.rep}`;
    const py = pyLookup.get(key);
    if (py) {
      totalCompared++;
      if (Math.abs(ts.kge - py.kge) < 0.1) matchCount++;
    }
  }
  console.log(`  ${matchCount}/${totalCompared} folds within 0.1 KGE of Python`);
}

main().catch(console.error);
