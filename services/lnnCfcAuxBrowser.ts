import { EigenvalueDecomposition, Matrix, solve } from 'ml-matrix';

export interface BrowserLnnPoint {
  date: string;
  time: number;
  observed: number | null;
  auxiliaries?: number[];
}

export interface BrowserLnnCfcParams {
  reservoirSize: number;
  spectralRadius: number;
  leakRate: number;
  inputScaling: number;
  ridgeAlpha: number;
  maxGapThreshold: number;
  smallGapOptimizeTrials: number;
}

export interface BrowserLnnResultPoint extends BrowserLnnPoint {
  imputed: number | null;
}

export interface BrowserLnnRunResult {
  points: BrowserLnnResultPoint[];
  params: BrowserLnnCfcParams;
  r2: number;
}

interface BrowserNormRow {
  observed: number | null;
  auxiliaries: number[];
}

interface Scaler {
  min: number;
  max: number;
}

interface PreparedContext {
  hasAux: boolean;
  numAux: number;
  inputDim: number;
  obsScaler: Scaler;
  auxScalers: Scaler[];
  normData: BrowserNormRow[];
  phWeights: number[];
  resWeights: number[];
  placeholderNorm: number[];
  inLargeGap: boolean[];
}

interface LocalRng {
  next(): number;
  int(minInclusive: number, maxExclusive: number): number;
}

function mean(values: number[]): number {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
}

function stdDev(values: number[]): number {
  if (values.length < 2) return 0;
  const m = mean(values);
  const variance = values.reduce((acc, v) => acc + (v - m) ** 2, 0) / (values.length - 1);
  return Math.sqrt(variance);
}

function pearson(x: number[], y: number[]): number {
  if (x.length !== y.length || x.length === 0) return 0;
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

function calculateR2(observed: number[], predicted: number[]): number {
  if (observed.length !== predicted.length || observed.length === 0) return Number.NEGATIVE_INFINITY;
  const obsMean = mean(observed);
  const ssTot = observed.reduce((acc, v) => acc + (v - obsMean) ** 2, 0);
  if (ssTot === 0) {
    const ssResFlat = observed.reduce((acc, v, i) => acc + (v - predicted[i]) ** 2, 0);
    return ssResFlat < 1e-12 ? 1 : Number.NEGATIVE_INFINITY;
  }
  const ssRes = observed.reduce((acc, v, i) => acc + (v - predicted[i]) ** 2, 0);
  return 1 - ssRes / ssTot;
}

function getScaler(values: number[]): Scaler {
  if (!values.length) return { min: 0, max: 1 };
  return { min: Math.min(...values), max: Math.max(...values) };
}

function createLocalRng(seed = 123456789): LocalRng {
  let state = seed >>> 0;
  const next = (): number => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 0x100000000;
  };
  return {
    next,
    int(minInclusive: number, maxExclusive: number): number {
      return minInclusive + Math.floor(next() * (maxExclusive - minInclusive));
    },
  };
}

function normalizeValue(val: number, scaler: Scaler): number {
  if (scaler.max === scaler.min) return 0;
  return ((val - scaler.min) / (scaler.max - scaler.min)) * 1.6 - 0.8;
}

function denormalizeValue(normVal: number, scaler: Scaler): number {
  if (scaler.max === scaler.min) return scaler.min;
  return ((normVal + 0.8) / 1.6) * (scaler.max - scaler.min) + scaler.min;
}

function ridgeRegression(X: number[][], y: number[], alpha: number): number[] {
  const xMat = new Matrix(X);
  const yMat = Matrix.columnVector(y);
  const xtx = xMat.transpose().mmul(xMat);
  for (let i = 0; i < xtx.rows; i++) xtx.set(i, i, xtx.get(i, i) + alpha);
  const xty = xMat.transpose().mmul(yMat);
  return solve(xtx, xty, true).getColumn(0);
}

function prepareContext(data: BrowserLnnPoint[], params: BrowserLnnCfcParams): PreparedContext {
  const firstWithAux = data.find(d => (d.auxiliaries?.length ?? 0) > 0);
  const numAux = firstWithAux?.auxiliaries?.length ?? 0;
  const hasAux = numAux > 0;
  const inputDim = 1 + numAux;
  const obsScaler = getScaler(data.flatMap(d => (d.observed != null ? [d.observed] : [])));
  const auxScalers: Scaler[] = [];
  for (let i = 0; i < numAux; i++) {
    auxScalers.push(getScaler(data.map(d => d.auxiliaries?.[i] ?? 0)));
  }
  const normData: BrowserNormRow[] = data.map(d => ({
    observed: d.observed != null ? normalizeValue(d.observed, obsScaler) : null,
    auxiliaries: Array.from({ length: numAux }, (_, i) => normalizeValue(d.auxiliaries?.[i] ?? 0, auxScalers[i])),
  }));

  const { phWeights, resWeights } = computeAuxWeights(normData, numAux);
  const placeholderNorm = computeAuxPlaceholders(data, normData, hasAux, numAux, obsScaler, params.ridgeAlpha * 10);
  const inLargeGap = computeLargeGapMask(data, params.maxGapThreshold);

  return {
    hasAux,
    numAux,
    inputDim,
    obsScaler,
    auxScalers,
    normData,
    phWeights,
    resWeights,
    placeholderNorm,
    inLargeGap,
  };
}

function computeAuxWeights(normData: BrowserNormRow[], numAux: number): { phWeights: number[]; resWeights: number[] } {
  if (numAux === 0) return { phWeights: [], resWeights: [] };
  const target = normData.filter(r => r.observed != null).map(r => r.observed as number);
  if (target.length < 3) return { phWeights: new Array(numAux).fill(1), resWeights: new Array(numAux).fill(1) };
  const auxSeries = Array.from({ length: numAux }, () => [] as number[]);
  for (const row of normData) {
    if (row.observed == null) continue;
    for (let j = 0; j < numAux; j++) auxSeries[j].push(row.auxiliaries[j] ?? 0);
  }
  const correlations = auxSeries.map(col => pearson(col, target));
  const phWeights = correlations.map(r => (Math.abs(r) >= 0.15 ? Math.abs(r) ** 2 : 0));
  if (Math.max(...phWeights, 0) < 1e-10) {
    const bestIdx = correlations.reduce((best, r, i, arr) => (Math.abs(r) > Math.abs(arr[best]) ? i : best), 0);
    phWeights[bestIdx] = Math.abs(correlations[bestIdx]) || 1;
  }
  const phMax = Math.max(...phWeights, 1);
  const phNorm = phWeights.map(w => w / phMax);
  const resWeights = correlations.map(r => Math.max(0.1, Math.abs(r) ** 2));
  const resMax = Math.max(...resWeights, 1);
  const resNorm = resWeights.map(w => w / resMax);
  return { phWeights: phNorm, resWeights: resNorm };
}

function computeAuxPlaceholders(
  data: BrowserLnnPoint[],
  normData: BrowserNormRow[],
  hasAux: boolean,
  numAux: number,
  obsScaler: Scaler,
  alpha: number,
): number[] {
  const n = data.length;
  if (!n) return [];
  const times = data.map(d => d.time);
  const minT = Math.min(...times);
  const maxT = Math.max(...times);
  const scaleT = Math.max(maxT - minT, 1e-10);
  const timeNorm = times.map(t => ((t - minT) / scaleT) * 1.6 - 0.8);

  const X = Array.from({ length: n }, (_, i) => {
    const row = [1, timeNorm[i]];
    if (hasAux) {
      for (let j = 0; j < numAux; j++) row.push(normData[i].auxiliaries[j] ?? 0);
    }
    return row;
  });
  const trainIdx = normData.map((r, i) => (r.observed != null ? i : -1)).filter(i => i >= 0);
  if (trainIdx.length < 2 || (hasAux && trainIdx.length < numAux + 1)) return [];
  const XTrain = trainIdx.map(i => X[i]);
  const yTrain = trainIdx.map(i => normData[i].observed as number);
  const w = ridgeRegression(XTrain, yTrain, alpha);
  return X.map((row, i) => {
    const pred = row.reduce((acc, v, j) => acc + v * w[j], 0);
    return normData[i].observed != null ? (normData[i].observed as number) : pred;
  });
}

function computeLargeGapMask(data: BrowserLnnPoint[], maxGapThreshold: number): boolean[] {
  const mask = new Array(data.length).fill(false);
  let gapStart = -1;
  let gapLen = 0;
  for (let i = 0; i < data.length; i++) {
    if (data[i].observed == null) {
      if (gapLen === 0) gapStart = i;
      gapLen++;
    } else {
      if (gapLen > maxGapThreshold) {
        for (let j = gapStart; j < gapStart + gapLen; j++) mask[j] = true;
      }
      gapStart = -1;
      gapLen = 0;
    }
  }
  if (gapLen > maxGapThreshold && gapStart >= 0) {
    for (let j = gapStart; j < gapStart + gapLen; j++) mask[j] = true;
  }
  return mask;
}

function spectralRadiusNormalize(Wres: Matrix, targetRadius: number): Matrix {
  const evd = new EigenvalueDecomposition(Wres);
  const real = evd.realEigenvalues;
  const imag = evd.imaginaryEigenvalues;
  let actualRadius = 0;
  for (let i = 0; i < real.length; i++) {
    actualRadius = Math.max(actualRadius, Math.hypot(real[i] ?? 0, imag[i] ?? 0));
  }
  if (actualRadius > 1e-10) return Wres.mul(targetRadius / actualRadius);
  return Wres.mul(targetRadius);
}

function identifyOutlierIndices(points: BrowserLnnResultPoint[]): number[] {
  const valid = points
    .map((point, index) => ({ index, value: point.imputed }))
    .filter((entry): entry is { index: number; value: number } => entry.value != null);
  if (valid.length < 5) return [];
  const values = valid.map(entry => entry.value);
  const mu = mean(values);
  const sigma = stdDev(values);
  if (sigma === 0) return [];
  return valid
    .filter(entry => Math.abs((entry.value - mu) / sigma) > 2.5)
    .map(entry => entry.index);
}

function getCurrentObservationValue(
  row: BrowserNormRow,
  lastValidObserved: number,
  hasAux: boolean,
): number {
  if (row.observed != null) return row.observed;
  if (hasAux && row.auxiliaries.length) {
    const valid = row.auxiliaries.filter(v => Number.isFinite(v) && v !== 0);
    if (valid.length) return mean(valid);
  }
  return lastValidObserved;
}

export function runExactLnnCfcAux(
  data: BrowserLnnPoint[],
  params: BrowserLnnCfcParams,
  rng: LocalRng = createLocalRng(),
): BrowserLnnRunResult {
  const ctx = prepareContext(data, params);
  const { hasAux, numAux, inputDim, obsScaler, normData, placeholderNorm, resWeights } = ctx;
  const reservoirSize = params.reservoirSize;
  const leak = Math.max(params.leakRate, 1e-6);
  const winVals = Array.from({ length: reservoirSize }, () =>
    Array.from({ length: inputDim }, () => (rng.next() * 2 - 1) * params.inputScaling),
  );
  const Win = new Matrix(winVals);
  const sparse = Array.from({ length: reservoirSize }, () =>
    Array.from({ length: reservoirSize }, () => (rng.next() < 0.2 ? rng.next() * 2 - 1 : 0)),
  );
  let Wres = new Matrix(sparse);
  Wres = spectralRadiusNormalize(Wres, params.spectralRadius);

  const states: number[][] = [];
  const targets: number[] = [];
  const allStates: number[][] = [];
  let x = new Array(reservoirSize).fill(0);
  let lastValidObserved = 0;

  for (let i = 0; i < data.length; i++) {
    const d = normData[i];
    const currentObs = placeholderNorm.length ? placeholderNorm[i] : getCurrentObservationValue(d, lastValidObserved, hasAux);
    if (d.observed != null) lastValidObserved = d.observed;
    const inputVector = [currentObs];
    if (hasAux) {
      for (let j = 0; j < numAux; j++) {
        inputVector.push((d.auxiliaries[j] ?? 0) * (resWeights[j] ?? 1));
      }
    }
    while (inputVector.length < inputDim) inputVector.push(0);

    const xVec = Matrix.columnVector(x);
    const inputVec = Matrix.columnVector(inputVector);
    const stepExp = Math.exp(-leak * 0.1);
    const stepCoef = (1 - stepExp) / leak;
    const b = Win.mmul(inputVec).add(Wres.mmul(xVec)).to1DArray().map(v => Math.tanh(v));
    x = x.map((prev, idx) => prev * stepExp + stepCoef * b[idx]);
    allStates.push([...x]);
    if (d.observed != null) {
      states.push([1, ...x]);
      targets.push(d.observed);
    }
  }

  if (!states.length) {
    return {
      points: data.map(d => ({ ...d, imputed: d.observed })),
      params,
      r2: Number.NEGATIVE_INFINITY,
    };
  }

  const Wout = ridgeRegression(states, targets, params.ridgeAlpha);
  const result = data.map((d, i) => {
    const row = [1, ...allStates[i]];
    const predNorm = row.reduce((acc, v, j) => acc + v * Wout[j], 0);
    return { ...d, imputed: denormalizeValue(predNorm, obsScaler) };
  });
  const obs = result.filter(r => r.observed != null && r.imputed != null).map(r => r.observed as number);
  const pred = result.filter(r => r.observed != null && r.imputed != null).map(r => r.imputed as number);
  return { points: result, params, r2: calculateR2(obs, pred) };
}

export function optimizeExactLnnCfcAux(
  data: BrowserLnnPoint[],
  baseParams: BrowserLnnCfcParams,
): BrowserLnnCfcParams {
  let best = baseParams;
  let bestScore = Number.NEGATIVE_INFINITY;
  const trials = Math.max(1, baseParams.smallGapOptimizeTrials);
  const rng = createLocalRng();
  for (let i = 0; i < trials; i++) {
    const candidate: BrowserLnnCfcParams = {
      ...baseParams,
      reservoirSize: rng.int(10, 81),
      leakRate: 0.05 + rng.next() * 0.9,
    };
    const result = runExactLnnCfcAux(data, candidate, rng);
    const outlierCount = identifyOutlierIndices(result.points).length;
    const finalScore = result.r2 - outlierCount * 0.05;
    if (finalScore > bestScore) {
      bestScore = finalScore;
      best = candidate;
    }
    if (bestScore >= 0.9) break;
  }
  return best;
}

export function computeProtectedMissingIndices(
  data: BrowserLnnPoint[],
  maxGapThreshold: number,
): Set<number> {
  const protectedIdx = new Set<number>();
  const observedIndices = data.map((d, i) => (d.observed != null ? i : -1)).filter(i => i >= 0);
  if (!observedIndices.length) {
    data.forEach((_, i) => protectedIdx.add(i));
    return protectedIdx;
  }
  const firstObs = observedIndices[0];
  const lastObs = observedIndices[observedIndices.length - 1];
  data.forEach((d, i) => {
    if (d.observed == null && (i < firstObs || i > lastObs)) protectedIdx.add(i);
  });
  const largeMask = computeLargeGapMask(data, maxGapThreshold);
  largeMask.forEach((isLarge, i) => {
    if (isLarge) protectedIdx.add(i);
  });
  return protectedIdx;
}

export function runExactSmallGapLnnCfcAux(
  data: BrowserLnnPoint[],
  params: BrowserLnnCfcParams,
): BrowserLnnRunResult {
  const protectedIdx = computeProtectedMissingIndices(data, params.maxGapThreshold);
  const masked = data.map((d, i) => ({
    ...d,
    observed: protectedIdx.has(i) ? null : d.observed,
  }));
  const bestParams = optimizeExactLnnCfcAux(masked, params);
  const result = runExactLnnCfcAux(masked, bestParams);
  return {
    ...result,
    points: result.points.map((p, i) => ({
      ...data[i],
      imputed: protectedIdx.has(i) ? null : p.imputed,
    })),
  };
}
