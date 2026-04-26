export interface IterativeMetrics {
  r2?: number;
  rmse?: number;
}

export interface IterativeRefineResult<TPoint> {
  points: TPoint[];
  metrics?: IterativeMetrics;
}

export interface IterativeMeta {
  outerIterationsUsed: number;
  supportPoints: number;
  bestSupportR2: number | null;
  bestSupportRmse: number | null;
}

function mean(values: number[]): number {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
}

function calculateR2(observed: number[], predicted: number[]): number {
  if (!observed.length || observed.length !== predicted.length) return Number.NEGATIVE_INFINITY;
  const obsMean = mean(observed);
  const ssTot = observed.reduce((acc, v) => acc + (v - obsMean) ** 2, 0);
  if (ssTot === 0) {
    const ssResFlat = observed.reduce((acc, v, i) => acc + (v - predicted[i]) ** 2, 0);
    return ssResFlat < 1e-12 ? 1 : Number.NEGATIVE_INFINITY;
  }
  const ssRes = observed.reduce((acc, v, i) => acc + (v - predicted[i]) ** 2, 0);
  return 1 - ssRes / ssTot;
}

export function selectSupportTruth(
  rawTarget: Array<number | null>,
  truthIdx: number[],
  random: () => number,
  supportFrac: number,
  minSupport: number,
  maxSupport: number,
): Record<number, number> {
  const truthSet = new Set(truthIdx);
  const candidates = rawTarget
    .map((v, i) => ({ v, i }))
    .filter(({ v, i }) => v != null && !truthSet.has(i))
    .map(({ v, i }) => ({ i, v: v as number }));

  if (candidates.length < minSupport) return {};
  let nPick = Math.round(candidates.length * supportFrac);
  nPick = Math.max(minSupport, Math.min(maxSupport, nPick, candidates.length));

  const pool = [...candidates];
  const picked: Record<number, number> = {};
  for (let k = 0; k < nPick; k++) {
    const idx = Math.floor(random() * pool.length);
    const chosen = pool.splice(idx, 1)[0];
    picked[chosen.i] = chosen.v;
  }
  return picked;
}

export function scoreTuple(metrics?: IterativeMetrics | null): [number, number] {
  if (!metrics) return [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
  const r2 = Number.isFinite(metrics.r2 ?? NaN) ? (metrics.r2 as number) : Number.NEGATIVE_INFINITY;
  const rmse = Number.isFinite(metrics.rmse ?? NaN) ? (metrics.rmse as number) : Number.POSITIVE_INFINITY;
  return [r2, -rmse];
}

export function blendPredictions(
  previous: Record<number, number> | null,
  next: Record<number, number>,
  prevWeight: number,
): Record<number, number> {
  const out: Record<number, number> = {};
  for (const [idxStr, nextVal] of Object.entries(next)) {
    const idx = Number(idxStr);
    const prevVal = previous?.[idx];
    if (prevVal != null && Number.isFinite(prevVal)) {
      out[idx] = prevWeight * prevVal + (1 - prevWeight) * nextVal;
    } else {
      out[idx] = nextVal;
    }
  }
  return out;
}

export function runIterativeSupportLoop<TPoint>(
  rawTarget: Array<number | null>,
  maskedTarget: Array<number | null>,
  truthIdx: number[],
  random: () => number,
  options: {
    outerIterations: number;
    feedbackPrevWeight: number;
    supportFrac: number;
    minSupport: number;
    maxSupport: number;
  },
  singlePass: (workingTarget: Array<number | null>, init?: Record<number, number> | null) => IterativeRefineResult<TPoint>,
  extractPredictions: (result: IterativeRefineResult<TPoint>) => Record<number, number>,
): { bestInit: Record<number, number> | null; meta: IterativeMeta } {
  const supportTruth = selectSupportTruth(
    rawTarget,
    truthIdx,
    random,
    options.supportFrac,
    options.minSupport,
    options.maxSupport,
  );

  if (!Object.keys(supportTruth).length) {
    return {
      bestInit: null,
      meta: {
        outerIterationsUsed: 0,
        supportPoints: 0,
        bestSupportR2: null,
        bestSupportRmse: null,
      },
    };
  }

  const supportIndices = Object.keys(supportTruth).map(Number);
  const workingTarget = [...maskedTarget];
  for (const i of supportIndices) workingTarget[i] = null;

  let bestInit: Record<number, number> | null = null;
  let bestMetrics: IterativeMetrics | null = null;
  let accepted = 0;

  for (let outer = 0; outer < options.outerIterations; outer++) {
    const result = singlePass(workingTarget, bestInit);
    const extracted = extractPredictions(result);
    const candidateInit = blendPredictions(bestInit, extracted, options.feedbackPrevWeight);
    const supportObs: number[] = [];
    const supportPred: number[] = [];
    for (const idx of supportIndices) {
      const truth = supportTruth[idx];
      const pred = candidateInit[idx];
      if (truth != null && Number.isFinite(pred)) {
        supportObs.push(truth);
        supportPred.push(pred);
      }
    }
    const rmse = supportObs.length
      ? Math.sqrt(supportObs.reduce((acc, v, i) => acc + (supportPred[i] - v) ** 2, 0) / supportObs.length)
      : Number.POSITIVE_INFINITY;
    const r2 = supportObs.length ? calculateR2(supportObs, supportPred) : Number.NEGATIVE_INFINITY;
    const candidateMetrics: IterativeMetrics = { r2, rmse };
    if (scoreTuple(candidateMetrics) > scoreTuple(bestMetrics)) {
      bestMetrics = candidateMetrics;
      bestInit = candidateInit;
      accepted = outer + 1;
    } else {
      break;
    }
  }

  return {
    bestInit,
    meta: {
      outerIterationsUsed: accepted,
      supportPoints: supportIndices.length,
      bestSupportR2: bestMetrics?.r2 ?? null,
      bestSupportRmse: bestMetrics?.rmse ?? null,
    },
  };
}
