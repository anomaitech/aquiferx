import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { SeededRng, pearsonR, computeKGE, pchipFill, archiRegression, mcSoftImpute, runLnnCfc, optimizeLnnParams } from '../services/mcLnnPureBrowser';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DATA_DIR = path.resolve(__dirname, '../../liquid-neural-imputer v2.0/datas');

function parseCSV(fp: string) {
  const lines = fs.readFileSync(fp, 'utf-8').trim().split('\n');
  const h = lines[0].split(',');
  return lines.slice(1).map(l => { const v = l.split(','); const r: any = {}; h.forEach((k,i) => r[k]=v[i]); return r; });
}

const mainDf = parseCSV(path.join(DATA_DIR, 'measurements_till_2023_to_lnn_imputation.csv'));
const auxDf = parseCSV(path.join(DATA_DIR, 'lnn_imputation_gslb_gldas_df_excercise.csv'));

const months: string[] = [];
let y=2000, m=0;
while(!(y===2024&&m>0)) { months.push(y+'-'+String(m+1).padStart(2,'0')); m++; if(m>11){m=0;y++;} }

const filtered = mainDf.filter((r: any) => r['Date']>='2000-01-01' && r['Date']<='2023-12-21');

function monthlySeries(wid: string): (number|null)[] {
  const byM = new Map<string, number[]>();
  for (const r of filtered) {
    if (r['Well_ID'] !== wid) continue;
    const d = new Date(r['Date']);
    const ym = d.getFullYear()+'-'+String(d.getMonth()+1).padStart(2,'0');
    if (!byM.has(ym)) byM.set(ym, []);
    byM.get(ym)!.push(parseFloat(r['WTE']));
  }
  return months.map(m => { const v = byM.get(m); return v ? v.reduce((a: number,b: number)=>a+b,0)/v.length : null; });
}

const wellCounts = new Map<string, number>();
for (const r of filtered) wellCounts.set(r['Well_ID'], (wellCounts.get(r['Well_ID'])??0)+1);
const eligible = Array.from(wellCounts.entries()).filter(([,n])=>n>=10).sort((a,b)=>b[1]-a[1]).map(([w])=>w).slice(0,50);

const wellSeries = new Map<string, (number|null)[]>();
for (const w of eligible) wellSeries.set(w, monthlySeries(w));

const auxCols = ['soilw','soilw_yr01','soilw_yr03','soilw_yr05','soilw_yr10'];
const auxByM = new Map<string, number[]>();
for (const r of auxDf) {
  const d = new Date(r['time']);
  const ym = d.getFullYear()+'-'+String(d.getMonth()+1).padStart(2,'0');
  auxByM.set(ym, auxCols.map(c => { const v = parseFloat(r[c]); return isNaN(v)?0:v; }));
}
const auxData = months.map(m => auxByM.get(m) ?? [0,0,0,0,0]);

const wid = '403916111575901';
const raw = wellSeries.get(wid)!;
console.log('Well:', wid);
console.log('Obs count:', raw.filter(v=>v!==null).length, '/', raw.length);
console.log('Value range:', Math.min(...raw.filter(v=>v!==null) as number[]).toFixed(1), '-', Math.max(...raw.filter(v=>v!==null) as number[]).toFixed(1));

// Create 2-year holdout
const holdoutStart = 120;
const truth = new Map<number, number>();
const modTarget = [...raw];
for (let i = holdoutStart; i < holdoutStart + 24; i++) {
  if (raw[i] !== null) { truth.set(i, raw[i]!); modTarget[i] = null; }
}
console.log('Holdout:', truth.size, 'obs at months', holdoutStart, '-', holdoutStart+23);

// Donor pool
const modSeries = new Map(Array.from(wellSeries.entries()).map(([w,s]): [string, (number|null)[]] => {
  return w === wid ? [w, modTarget] : [w, s];
}));

const targetObs = new Map<number, number>();
for (let i = 0; i < modTarget.length; i++) if (modTarget[i] !== null) targetObs.set(i, modTarget[i]!);

// Step 1: ARCHI
console.log('\n=== STEP 1: ARCHI DONOR REGRESSION ===');
const { preds: archiPreds, donors } = archiRegression(targetObs, modSeries, wid, 15);
console.log('Donors found:', donors.length);
if (donors.length) console.log('Top donor r:', donors[0].r.toFixed(4));
const archiO: number[] = [], archiP: number[] = [];
for (const [t, v] of Array.from(truth.entries())) {
  if (archiPreds.has(t)) { archiO.push(v); archiP.push(archiPreds.get(t)!); }
}
console.log('ARCHI scored:', archiO.length, '/', truth.size);
if (archiO.length>=2) {
  console.log('ARCHI KGE:', computeKGE(archiO, archiP).toFixed(4));
  console.log('ARCHI pred sample:', archiP.slice(0,5).map(v=>v.toFixed(1)));
  console.log('Truth sample:', archiO.slice(0,5).map(v=>v.toFixed(1)));
}

// Step 2: MC
console.log('\n=== STEP 2: MC SOFTIMPUTE ===');
const mcPreds = mcSoftImpute(modTarget, modSeries, donors, archiPreds, auxData, new SeededRng(42));
const mcO: number[] = [], mcP: number[] = [];
for (const [t, v] of Array.from(truth.entries())) {
  if (mcPreds.has(t)) { mcO.push(v); mcP.push(mcPreds.get(t)!); }
}
console.log('MC scored:', mcO.length, '/', truth.size);
if (mcO.length>=2) {
  console.log('MC KGE:', computeKGE(mcO, mcP).toFixed(4));
  console.log('MC pred sample:', mcP.slice(0,5).map(v=>v.toFixed(1)));
  console.log('MC pred range:', Math.min(...mcP).toFixed(1), '-', Math.max(...mcP).toFixed(1));
}

// Step 3: LNN
console.log('\n=== STEP 3: LNN CFC ===');
const mcPlaceholders = new Map<number, number>();
for (let t = 0; t < modTarget.length; t++) {
  if (modTarget[t] === null && mcPreds.has(t)) mcPlaceholders.set(t, mcPreds.get(t)!);
}
console.log('MC placeholders:', mcPlaceholders.size);

const bestParams = optimizeLnnParams(modTarget, mcPlaceholders, auxData, new SeededRng(42), 8);
console.log('Best params:', JSON.stringify(bestParams));

const lnnPred = runLnnCfc(modTarget, mcPlaceholders, auxData, bestParams, new SeededRng(42));
const lnnO: number[] = [], lnnP: number[] = [];
for (const [t, v] of Array.from(truth.entries())) {
  if (isFinite(lnnPred[t])) { lnnO.push(v); lnnP.push(lnnPred[t]); }
}
console.log('LNN scored:', lnnO.length);
if (lnnO.length>=2) {
  console.log('LNN KGE:', computeKGE(lnnO, lnnP).toFixed(4));
  console.log('LNN pred sample:', lnnP.slice(0,5).map(v=>v.toFixed(1)));
  console.log('LNN pred range:', Math.min(...lnnP).toFixed(1), '-', Math.max(...lnnP).toFixed(1));
  console.log('Truth range:', Math.min(...lnnO).toFixed(1), '-', Math.max(...lnnO).toFixed(1));
}
