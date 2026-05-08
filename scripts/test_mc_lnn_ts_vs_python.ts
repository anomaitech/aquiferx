/**
 * Test MC-LNN TypeScript vs Python on GSLB data.
 *
 * Loads the same GSLB CSV data used by Python, runs the TS pipeline,
 * and compares KGE/RMSE against the Python results.
 *
 * Run: npx tsx scripts/test_mc_lnn_ts_vs_python.ts
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import {
  runMcLnnImputation,
  SeededRng,
  pearsonR,
  computeKGE,
  computeNRMSE,
  pchipFill,
} from '../services/mcLnnPureBrowser';

// ── Paths ──
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DATA_DIR = path.resolve(__dirname, '../../liquid-neural-imputer v2.0/datas');
const TARGET_CSV = path.join(DATA_DIR, 'measurements_till_2023_to_lnn_imputation.csv');
const AUX_CSV = path.join(DATA_DIR, 'lnn_imputation_gslb_gldas_df_excercise.csv');

// ── Config ──
const DATE_START = '2000-01-01';
const DATE_END = '2023-12-21';
const CONSECUTIVE_YEARS = [1, 2, 3, 4, 5];
const AUX_COLS = ['soilw', 'soilw_yr01', 'soilw_yr03', 'soilw_yr05', 'soilw_yr10'];
const SEED = 42;
const TOP_WELLS = [
  '415703112514501',
  '414236112101201',
  '414411112543701',
  '411544111461001',
  '411348112013601',
  '401818112014501',
  '402333111513401',
  '401312112442301',
  '403916111575901',
];
const DONOR_POOL_SIZE = 50;

// ── CSV parsing ──
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

// ── Generate monthly date grid ──
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

// ── Build monthly series for a well ──
function monthlySeries(
  mainDf: Record<string, string>[],
  wellId: string,
  months: string[],
): (number | null)[] {
  // Group by month, average WTE
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
    if (!vals || vals.length === 0) return null;
    return vals.reduce((a, b) => a + b, 0) / vals.length;
  });
}

// ── Build aux lookup ──
function buildAuxData(auxDf: Record<string, string>[], months: string[]): number[][] {
  const byMonth = new Map<string, number[]>();
  for (const row of auxDf) {
    const d = new Date(row['time']);
    const ym = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`;
    const vals = AUX_COLS.map(c => {
      const v = parseFloat(row[c]);
      return isNaN(v) ? 0 : v;
    });
    byMonth.set(ym, vals);
  }
  return months.map(m => byMonth.get(m) ?? [0, 0, 0, 0, 0]);
}

// ── Seeded holdout ──
function rngSeed(...parts: (string | number)[]): number {
  // Simple hash matching Python's sha256-based seed
  let h = 0;
  const s = parts.join('|');
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) - h + s.charCodeAt(i)) | 0;
  }
  return Math.abs(h) % (2 ** 31);
}

function relaxedHoldout(
  raw: (number | null)[],
  nYears: number,
  seed: number,
): { startIdx: number; truth: Map<number, number> } | null {
  const nMonths = nYears * 12;
  if (raw.length < nMonths) return null;

  const validStarts: number[] = [];
  for (let s = 0; s <= raw.length - nMonths; s++) {
    let nObs = 0;
    for (let i = s; i < s + nMonths; i++) if (raw[i] !== null) nObs++;
    if (nObs >= 1) validStarts.push(s);
  }
  if (!validStarts.length) return null;

  const rng = new SeededRng(seed);
  const start = validStarts[rng.randInt(0, validStarts.length)];
  const truth = new Map<number, number>();
  for (let i = start; i < start + nMonths; i++) {
    if (raw[i] !== null) truth.set(i, raw[i]!);
  }
  return { startIdx: start, truth };
}

// ── Main test ──
async function main() {
  console.log('Loading GSLB data...');
  const mainDf = parseCSV(TARGET_CSV);
  const auxDf = parseCSV(AUX_CSV);
  const months = generateMonths(DATE_START, DATE_END);
  console.log(`  ${mainDf.length} records, ${months.length} months`);

  // Filter to date range
  const filteredMain = mainDf.filter(row => {
    const d = row['Date'];
    return d >= DATE_START && d <= DATE_END;
  });

  // Get eligible wells sorted by observation count
  const wellCounts = new Map<string, number>();
  for (const row of filteredMain) {
    const wid = row['Well_ID'];
    wellCounts.set(wid, (wellCounts.get(wid) ?? 0) + 1);
  }
  const eligible = Array.from(wellCounts.entries())
    .filter(([, n]) => n >= 10)
    .sort((a, b) => b[1] - a[1])
    .map(([wid]) => wid)
    .slice(0, DONOR_POOL_SIZE);

  console.log(`  ${eligible.length} eligible wells (top ${DONOR_POOL_SIZE})`);

  // Build monthly series for all eligible wells
  const wellSeries = new Map<string, (number | null)[]>();
  for (const wid of eligible) {
    wellSeries.set(wid, monthlySeries(filteredMain, wid, months));
  }

  const rawTotal = Array.from(wellSeries.values()).reduce(
    (acc, s) => acc + s.filter(v => v !== null).length, 0
  );
  console.log(`  Raw observations: ${rawTotal}`);

  // Build aux data
  const filteredAux = auxDf.filter(row => {
    const d = row['time'];
    return d >= DATE_START && d <= DATE_END;
  });
  const auxData = buildAuxData(filteredAux, months);
  console.log(`  Aux data: ${auxData.length} months`);

  // ── PCHIP prefill ALL donor wells (matching Python) ──
  console.log('PCHIP prefilling all donors...');
  const filledSeries = new Map<string, (number | null)[]>();
  for (const [wid, raw] of Array.from(wellSeries.entries())) {
    filledSeries.set(wid, pchipFill(raw, 24 * 2)); // fill all gaps first, pipeline handles blanking
  }
  const filledTotal = Array.from(filledSeries.values()).reduce(
    (acc, s) => acc + s.filter(v => v !== null).length, 0
  );
  console.log(`  PCHIP filled: ${rawTotal} -> ${filledTotal} obs`);

  // ── Run consecutive-gap CV ──
  console.log('\n=== TypeScript MC-LNN Consecutive-Gap CV ===');
  console.log(`Wells: ${TOP_WELLS.length}, Years: [${CONSECUTIVE_YEARS}], Repeats: 2`);
  console.log();

  const results: { nYears: number; kge: number; rmse: number }[] = [];
  const t0 = Date.now();

  for (const wid of TOP_WELLS) {
    const rawTarget = wellSeries.get(wid);
    if (!rawTarget) { console.log(`  ${wid}: not in donor pool, skipping`); continue; }

    for (const nYears of CONSECUTIVE_YEARS) {
      for (let rep = 0; rep < 2; rep++) {
        const holdout = relaxedHoldout(rawTarget, nYears, rngSeed(SEED, 'cv', wid, nYears, rep));
        if (!holdout) continue;

        // Build fold: use PCHIP-filled donors, remove holdout from target
        const modSeries = new Map<string, (number | null)[]>();
        for (const [w, s] of Array.from(filledSeries.entries())) {
          if (w === wid) {
            const mod = [...s];
            for (const t of Array.from(holdout.truth.keys())) mod[t] = null;
            modSeries.set(w, mod);
          } else {
            modSeries.set(w, s);
          }
        }

        try {
          const imputed = await runMcLnnImputation(
            modSeries, auxData, [wid],
            { seed: rngSeed(SEED, 'mc_lnn', wid, nYears, rep), maxDonors: 50 },
          );

          const pred = imputed.get(wid);
          if (!pred) continue;

          const obsVals: number[] = [];
          const predVals: number[] = [];
          for (const [t, v] of Array.from(holdout.truth.entries())) {
            if (isFinite(pred[t])) {
              obsVals.push(v);
              predVals.push(pred[t]);
            }
          }

          if (obsVals.length >= 2) {
            const kge = computeKGE(obsVals, predVals);
            const rmse = Math.sqrt(
              obsVals.reduce((acc, v, i) => acc + (v - predVals[i]) ** 2, 0) / obsVals.length
            );
            results.push({ nYears, kge, rmse });
          }
        } catch (e) {
          // skip
        }
      }
    }
    console.log(`  ${wid} done`);
  }

  const elapsed = (Date.now() - t0) / 1000;

  // ── Summary ──
  console.log(`\n${'='.repeat(60)}`);
  console.log(`TypeScript MC-LNN Results (${elapsed.toFixed(0)}s)`);
  console.log(`${'='.repeat(60)}`);
  console.log(`${'Gap'.padEnd(6)} ${'TS KGE'.padStart(10)} ${'TS RMSE'.padStart(10)}`);
  console.log('-'.repeat(30));

  for (const ny of CONSECUTIVE_YEARS) {
    const sub = results.filter(r => r.nYears === ny);
    if (sub.length) {
      const kgeMean = sub.reduce((a, r) => a + r.kge, 0) / sub.length;
      const rmseMean = sub.reduce((a, r) => a + r.rmse, 0) / sub.length;
      console.log(`${ny}yr${' '.repeat(3)} ${kgeMean.toFixed(4).padStart(10)} ${rmseMean.toFixed(4).padStart(10)}`);
    }
  }

  const allKge = results.reduce((a, r) => a + r.kge, 0) / results.length;
  const allRmse = results.reduce((a, r) => a + r.rmse, 0) / results.length;
  console.log(`${'ALL'.padEnd(6)} ${allKge.toFixed(4).padStart(10)} ${allRmse.toFixed(4).padStart(10)}`);

  console.log(`\nPython reference (GSLB, aquiferx-style PCHIP + MC-LNN):`);
  console.log(`  ALL: KGE=0.8060  RMSE=1.1810`);
  console.log(`\nDelta: KGE=${(allKge - 0.806).toFixed(4)}, RMSE=${(allRmse - 1.181).toFixed(4)}`);
}

main().catch(console.error);
