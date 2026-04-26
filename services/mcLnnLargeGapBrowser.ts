import { BrowserLnnCfcParams, BrowserLnnPoint, BrowserLnnResultPoint, runExactLnnCfcAux } from './lnnCfcAuxBrowser';
import { softImpute, SoftImputeOptions } from './mcSoftImpute';

export interface AuxMonthLookup {
  [isoMonth: string]: number[];
}

export interface BrowserDonorSeries {
  wellId: string;
  points: BrowserLnnPoint[];
}

export interface BrowserMcSinglePassInput {
  targetWellId: string;
  targetSeries: Array<number | null>;
  donorSeries: Record<string, Array<number | null>>;
  monthLabels: string[];
  auxByMonth: AuxMonthLookup;
  lnnParams: BrowserLnnCfcParams;
  softImpute: SoftImputeOptions;
}

const MIN_DONOR_CORR = 0.3;
const MAX_DONORS = 10;

function mean(values: number[]): number {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
}

function std(values: number[]): number {
  if (values.length < 2) return 0;
  const m = mean(values);
  return Math.sqrt(values.reduce((acc, v) => acc + (v - m) ** 2, 0) / values.length);
}

function pearson(x: number[], y: number[]): number {
  if (!x.length || x.length !== y.length) return 0;
  const mx = mean(x);
  const my = mean(y);
  let num = 0;
  let dx = 0;
  let dy = 0;
  for (let i = 0; i < x.length; i++) {
    const ax = x[i] - mx;
    const ay = y[i] - my;
    num += ax * ay;
    dx += ax * ax;
    dy += ay * ay;
  }
  if (dx === 0 || dy === 0) return 0;
  return num / Math.sqrt(dx * dy);
}

function archiRegression(
  targetObs: Record<number, number>,
  donorObs: Record<string, Array<number | null>>,
  targetId: string,
): { archiPreds: Record<number, number>; donors: Array<{ wid: string; r: number; dobs: Record<number, number> }> } {
  if (Object.keys(targetObs).length < 5) return { archiPreds: {}, donors: [] };
  const sampleSeries = donorObs[targetId] ?? Object.values(donorObs)[0] ?? [];
  const gapTimes = Array.from({ length: sampleSeries.length }, (_, i) => i).filter(i => !(i in targetObs));
  const donors: Array<{ wid: string; r: number; dobs: Record<number, number> }> = [];

  for (const [wid, series] of Object.entries(donorObs)) {
    if (wid === targetId) continue;
    const dobs: Record<number, number> = {};
    series.forEach((v, i) => {
      if (v != null) dobs[i] = v;
    });
    const common = Object.keys(targetObs).map(Number).filter(i => i in dobs).sort((a, b) => a - b);
    if (common.length < 8) continue;
    const tv = common.map(i => targetObs[i]);
    const dv = common.map(i => dobs[i]);
    if (std(tv) < 1e-10 || std(dv) < 1e-10) continue;
    const r = pearson(tv, dv);
    if (Math.abs(r) < MIN_DONOR_CORR) continue;
    donors.push({ wid, r, dobs });
  }
  donors.sort((a, b) => Math.abs(b.r) - Math.abs(a.r));
  const topDonors = donors.slice(0, MAX_DONORS);

  const predsByDonor: Array<Record<number, number>> = [];
  const weights: number[] = [];
  for (const donor of topDonors) {
    const common = Object.keys(targetObs).map(Number).filter(i => i in donor.dobs).sort((a, b) => a - b);
    if (common.length < 5) continue;
    const tv = common.map(i => targetObs[i]);
    const dv = common.map(i => donor.dobs[i]);
    const dm = mean(dv);
    const tm = mean(tv);
    const ss = dv.reduce((acc, v) => acc + (v - dm) ** 2, 0);
    if (ss < 1e-10) continue;
    const a = dv.reduce((acc, v, idx) => acc + (v - dm) * (tv[idx] - tm), 0) / ss;
    const b = tm - a * dm;
    const preds: Record<number, number> = {};
    for (const t of gapTimes) {
      if (t in donor.dobs) preds[t] = a * donor.dobs[t] + b;
    }
    if (Object.keys(preds).length) {
      predsByDonor.push(preds);
      weights.push(donor.r ** 2);
    }
  }

  const combined: Record<number, number> = {};
  for (const t of gapTimes) {
    let ws = 0;
    let wc = 0;
    for (let i = 0; i < predsByDonor.length; i++) {
      const pred = predsByDonor[i][t];
      if (pred != null) {
        ws += pred * weights[i];
        wc += weights[i];
      }
    }
    if (wc > 0) combined[t] = ws / wc;
  }
  return { archiPreds: combined, donors: topDonors };
}

function chooseRankByObservedError(M: number[][], candidateRanks: number[], softOptions: SoftImputeOptions): number {
  const observedIdx: Array<[number, number]> = [];
  for (let r = 0; r < M.length; r++) {
    for (let c = 0; c < M[r].length; c++) {
      if (Number.isFinite(M[r][c])) observedIdx.push([r, c]);
    }
  }
  let bestRank = candidateRanks[0] ?? 5;
  let bestErr = Number.POSITIVE_INFINITY;
  for (const rank of candidateRanks) {
    const filled = softImpute(M, { ...softOptions, rank, maxIterations: 20, tolerance: 5e-5 }).completed;
    let err = 0;
    let count = 0;
    for (const [r, c] of observedIdx) {
      err += (filled[r][c] - M[r][c]) ** 2;
      count++;
    }
    if (count && err / count < bestErr) {
      bestErr = err / count;
      bestRank = rank;
    }
  }
  return bestRank;
}

function matrixCompletionArchiInit(
  targetSeries: Array<number | null>,
  targetAux: number[][],
  donorObs: Record<string, Array<number | null>>,
  donors: Array<{ wid: string; r: number }>,
  archiPreds: Record<number, number>,
  softOptions: SoftImputeOptions,
  targetInitPreds?: Record<number, number> | null,
): Record<number, number> {
  const nTimes = targetSeries.length;
  const targetObsIdx = targetSeries.map((v, i) => (v != null ? i : -1)).filter(i => i >= 0);
  const selWids = [null, ...donors.map(d => d.wid)];
  const nw = selWids.length;
  const nAux = targetAux[0]?.length ?? 0;
  const MRaw = Array.from({ length: nw }, () => new Array(nTimes).fill(Number.NaN));

  targetSeries.forEach((v, t) => { if (v != null) MRaw[0][t] = v; });
  selWids.slice(1).forEach((wid, wi) => {
    donorObs[wid as string].forEach((v, t) => {
      if (v != null) MRaw[wi + 1][t] = v;
    });
  });

  const M = Array.from({ length: nw + nAux + 2 }, () => new Array(nTimes).fill(Number.NaN));
  const rmeans = new Array(nw).fill(0);
  const rstds = new Array(nw).fill(1);

  for (let wi = 0; wi < nw; wi++) {
    const vals = MRaw[wi].filter(v => Number.isFinite(v));
    if (vals.length >= 3) {
      rmeans[wi] = mean(vals);
      rstds[wi] = Math.max(std(vals), 1e-10);
    } else if (vals.length) {
      rmeans[wi] = mean(vals);
    }
    for (let t = 0; t < nTimes; t++) {
      const v = MRaw[wi][t];
      if (Number.isFinite(v)) M[wi][t] = (v - rmeans[wi]) / rstds[wi];
    }
  }

  donors.forEach((d, di) => {
    for (let t = 0; t < nTimes; t++) {
      if (Number.isFinite(M[di + 1][t])) M[di + 1][t] *= Math.abs(d.r);
    }
  });

  for (let j = 0; j < nAux; j++) {
    const col = targetAux.map(row => row[j] ?? 0);
    const cm = mean(col);
    const cs = Math.max(std(col), 1e-10);
    for (let t = 0; t < nTimes; t++) M[nw + j][t] = (col[t] - cm) / cs;
  }

  for (let t = 0; t < nTimes; t++) {
    M[nw + nAux][t] = Math.sin(2 * Math.PI * t / 12);
    M[nw + nAux + 1][t] = Math.cos(2 * Math.PI * t / 12);
    if (!Number.isFinite(M[0][t])) {
      if (archiPreds[t] != null) M[0][t] = (archiPreds[t] - rmeans[0]) / rstds[0];
      else M[0][t] = 0;
    }
  }

  const candidateRanks = [3, 5, 8].filter(r => r < Math.min(M.length, nTimes));
  const bestRank = chooseRankByObservedError(M, candidateRanks.length ? candidateRanks : [Math.max(1, Math.min(5, M.length - 1, nTimes - 1))], softOptions);
  const X = softImpute(M, { ...softOptions, rank: bestRank, maxIterations: 50, tolerance: 5e-5 }).completed;
  donors.forEach((d, di) => {
    if (Math.abs(d.r) > 1e-10) {
      for (let t = 0; t < nTimes; t++) X[di + 1][t] /= Math.abs(d.r);
    }
  });

  const pred = X[0].map(v => v * rstds[0] + rmeans[0]);
  if (targetInitPreds) {
    for (const [tStr, v] of Object.entries(targetInitPreds)) {
      const t = Number(tStr);
      if (targetSeries[t] == null && Number.isFinite(v) && Number.isFinite(pred[t])) pred[t] = 0.35 * v + 0.65 * pred[t];
    }
  }
  if (targetObsIdx.length >= 3) {
    const ov = targetObsIdx.map(i => MRaw[0][i]);
    const mv = targetObsIdx.map(i => pred[i]);
    const om = mean(ov);
    const os = Math.max(std(ov), 1e-10);
    const mm = mean(mv);
    const ms = Math.max(std(mv), 1e-10);
    for (let t = 0; t < pred.length; t++) pred[t] = ((pred[t] - mm) / ms) * os + om;
  }
  const out: Record<number, number> = {};
  pred.forEach((v, t) => { out[t] = v; });
  return out;
}

function buildAuxTimeline(
  obsList: Array<number | null>,
  monthLabels: string[],
  auxByMonth: AuxMonthLookup,
): BrowserLnnPoint[] {
  return monthLabels.map((month, i) => ({
    date: month,
    time: i,
    observed: obsList[i],
    auxiliaries: [
      ...(auxByMonth[month] ?? [0, 0, 0, 0, 0]),
      Math.sin(2 * Math.PI * i / 12),
      Math.cos(2 * Math.PI * i / 12),
    ],
  }));
}

export function runExactMcLnnSinglePass(input: BrowserMcSinglePassInput, targetInitPreds?: Record<number, number> | null): BrowserLnnResultPoint[] {
  const targetObs: Record<number, number> = {};
  input.targetSeries.forEach((v, i) => { if (v != null) targetObs[i] = v; });
  const { archiPreds, donors } = archiRegression(targetObs, input.donorSeries, input.targetWellId);
  const targetAux = input.monthLabels.map(month => [...(input.auxByMonth[month] ?? [0, 0, 0, 0, 0])]);
  if (!donors.length) {
    return runExactLnnCfcAux(buildAuxTimeline(input.targetSeries, input.monthLabels, input.auxByMonth), input.lnnParams).points;
  }
  const mcPreds = matrixCompletionArchiInit(
    input.targetSeries,
    targetAux,
    input.donorSeries,
    donors.map(d => ({ wid: d.wid, r: d.r })),
    archiPreds,
    input.softImpute,
    targetInitPreds,
  );
  const enriched = input.targetSeries.map((v, i) => (v != null ? v : (Number.isFinite(mcPreds[i]) ? mcPreds[i] : null)));
  return runExactLnnCfcAux(buildAuxTimeline(enriched, input.monthLabels, input.auxByMonth), input.lnnParams).points;
}
