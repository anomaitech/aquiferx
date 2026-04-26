import { softImpute, SoftImputeOptions, SoftImputeResult } from './mcSoftImpute';
import { BrowserLnnPoint, runExactSmallGapLnnCfcAux } from './lnnCfcAuxBrowser';
import { runExactMcLnnSinglePass } from './mcLnnLargeGapBrowser';
import { runIterativeSupportLoop } from './mcLnnIterativeExactBrowser';

export interface BrowserMcLnnConfig {
  smallGapThresholdMonths: number;
  outerIterations: number;
  feedbackPrevWeight: number;
  softImpute: SoftImputeOptions;
}

export interface BrowserMcLnnPoint extends BrowserLnnPoint {}

export interface BrowserMcLnnSeriesRow {
  date: string;
  raw: number | null;
  smallGap: number | null;
  final: number | null;
  fillStage: 'observed' | 'small_gap_lnn_cfc_aux' | 'large_gap_mc_lnn_iterative' | 'unfilled';
}

export interface BrowserMcLnnWellResult {
  wellId: string;
  rows: BrowserMcLnnSeriesRow[];
}

export interface BrowserMcLnnRunResult {
  wells: BrowserMcLnnWellResult[];
  mc: SoftImputeResult;
  notes: string[];
}

export interface BrowserMcLnnProgressCallbacks {
  onStage?: (label: string, pct: number) => void;
  onNote?: (note: string) => void;
}

export interface BrowserMcLnnInputWell {
  wellId: string;
  points: BrowserMcLnnPoint[];
}

function cloneWells(wells: BrowserMcLnnInputWell[]): BrowserMcLnnInputWell[] {
  return wells.map(w => ({
    wellId: w.wellId,
    points: w.points.map(p => ({ ...p, auxiliaries: p.auxiliaries ? [...p.auxiliaries] : undefined })),
  }));
}

function buildMatrix(wells: BrowserMcLnnInputWell[]): number[][] {
  if (wells.length === 0) return [];
  const nTime = wells[0].points.length;
  return Array.from({ length: nTime }, (_, t) =>
    wells.map(w => {
      const v = w.points[t]?.observed;
      return v == null ? Number.NaN : v;
    }),
  );
}

function applyCompletedMatrix(
  original: BrowserMcLnnInputWell[],
  smallGapFilled: BrowserMcLnnInputWell[],
  finalRefined: BrowserMcLnnInputWell[],
  completed: number[][],
): BrowserMcLnnWellResult[] {
  return finalRefined.map((well, wIdx) => ({
    wellId: well.wellId,
    rows: well.points.map((p, tIdx) => {
      const raw = original[wIdx].points[tIdx]?.observed ?? null;
      const sg = smallGapFilled[wIdx].points[tIdx]?.observed ?? null;
      const final = Number.isFinite(completed[tIdx]?.[wIdx]) ? completed[tIdx][wIdx] : sg;
      let fillStage: BrowserMcLnnSeriesRow['fillStage'] = 'unfilled';
      if (raw != null) fillStage = 'observed';
      else if (sg != null) fillStage = 'small_gap_lnn_cfc_aux';
      else if (final != null) fillStage = 'large_gap_mc_lnn_iterative';
      return {
        date: p.date,
        raw,
        smallGap: sg,
        final,
        fillStage,
      };
    }),
  }));
}

/**
 * Browser-port foundation for the MC+LNN pipeline.
 *
 * Current scope:
 * - ships a browser-safe SoftImpute implementation for the large-gap MC core
 * - reserves the exact LNN-CFC auxiliary small-gap and iterative refinement stages
 *   for a direct TypeScript port next
 *
 * This keeps the algorithm path honest: AquiferX gets a real browser MC backend
 * first instead of a placeholder that claims to run the full validated pipeline.
 */
export function runBrowserMcLnnFoundation(
  matrix: number[][],
  config: BrowserMcLnnConfig,
): BrowserMcLnnRunResult {
  const mc = softImpute(matrix, config.softImpute);
  return {
    wells: [],
    mc,
    notes: [
      'Browser MC core completed with SoftImpute.',
      'Next port steps: LNN-CFC auxiliary small-gap fill and iterative MC->LNN feedback.',
      `Configured outer iterations target: ${config.outerIterations}`,
      `Configured small-gap threshold (months): ${config.smallGapThresholdMonths}`,
    ],
  };
}

export function runBrowserMcLnnCandidate(
  inputWells: BrowserMcLnnInputWell[],
  config: BrowserMcLnnConfig,
): BrowserMcLnnRunResult {
  const original = cloneWells(inputWells);
  const working = cloneWells(inputWells);
  const notes: string[] = [];

  for (const well of working) {
    const small = runExactSmallGapLnnCfcAux(well.points, {
      reservoirSize: 20,
      spectralRadius: 0.9,
      leakRate: 0.2,
      inputScaling: 0.5,
      ridgeAlpha: 1e-4,
      maxGapThreshold: config.smallGapThresholdMonths,
      smallGapOptimizeTrials: 8,
    });
    well.points = small.points.map(p => ({
      date: p.date,
      time: p.time,
      observed: p.observed != null ? p.observed : p.imputed,
      auxiliaries: p.auxiliaries,
    }));
    notes.push(`[${well.wellId}] exact small-gap CFC R²=${Number.isFinite(small.r2) ? small.r2.toFixed(3) : 'NA'}`);
  }

  const mc = softImpute(buildMatrix(working), config.softImpute);
  const wells = applyCompletedMatrix(original, working, working, mc.completed);

  notes.push('Browser pipeline now uses the exact small-gap LNN-CFC auxiliary port.');
  notes.push('Remaining exact port step: iterative large-gap MC -> LNN feedback refinement.');

  return { wells, mc, notes };
}

function makeSeededRandom(seed = 42): () => number {
  let s = seed >>> 0;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

function yieldToBrowser(): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, 0));
}

export async function runBrowserMcLnnExact(
  inputWells: BrowserMcLnnInputWell[],
  config: BrowserMcLnnConfig,
  callbacks: BrowserMcLnnProgressCallbacks = {},
): Promise<BrowserMcLnnRunResult> {
  const original = cloneWells(inputWells);
  const smallGapWells = cloneWells(inputWells);
  const notes: string[] = [];
  let noLargeGapNeeded = 0;

  callbacks.onStage?.('Small-gap LNN-CFC auxiliary...', 30);
  for (let wellIndex = 0; wellIndex < smallGapWells.length; wellIndex++) {
    const well = smallGapWells[wellIndex];
    const small = runExactSmallGapLnnCfcAux(well.points, {
      reservoirSize: 20,
      spectralRadius: 0.9,
      leakRate: 0.2,
      inputScaling: 0.5,
      ridgeAlpha: 1e-4,
      maxGapThreshold: config.smallGapThresholdMonths,
      smallGapOptimizeTrials: 8,
    });
    well.points = small.points.map(p => ({
      date: p.date,
      time: p.time,
      observed: p.observed != null ? p.observed : p.imputed,
      auxiliaries: p.auxiliaries,
    }));
    const rawMissing = original[wellIndex].points.filter(p => p.observed == null).length;
    const remainingAfterSmall = small.points.filter(p => p.imputed == null && p.observed == null).length;
    if (remainingAfterSmall === 0) noLargeGapNeeded += 1;
    const note = `[${well.wellId}] small-gap R²=${Number.isFinite(small.r2) ? small.r2.toFixed(3) : 'NA'} | rawMissing=${rawMissing} | remainingAfterSmall=${remainingAfterSmall}`;
    notes.push(note);
    callbacks.onNote?.(note);
    callbacks.onStage?.('Small-gap LNN-CFC auxiliary...', 30 + ((wellIndex + 1) / Math.max(1, smallGapWells.length)) * 20);
    if ((wellIndex + 1) % 5 === 0) await yieldToBrowser();
  }

  const monthLabels = smallGapWells[0]?.points.map(p => p.date) ?? [];
  const auxByMonth = Object.fromEntries(monthLabels.map((m, i) => [m, smallGapWells[0].points[i]?.auxiliaries?.slice(0, 5) ?? [0, 0, 0, 0, 0]]));
  const donorMap = Object.fromEntries(smallGapWells.map(w => [w.wellId, w.points.map(p => p.observed)]));

  const wellsNeedingLargeGap = smallGapWells.filter(w => w.points.some(p => p.observed == null));
  callbacks.onNote?.(`Stage summary: ${smallGapWells.length - noLargeGapNeeded}/${smallGapWells.length} wells require large-gap refinement after small-gap fill.`);
  callbacks.onStage?.(`Large-gap iterative MC + LNN... (${wellsNeedingLargeGap.length}/${smallGapWells.length} wells)`, 50);
  const refinedWells: BrowserMcLnnInputWell[] = [];
  let largeGapProcessed = 0;
  for (let wellIndex = 0; wellIndex < smallGapWells.length; wellIndex++) {
    const well = smallGapWells[wellIndex];
    if (!well.points.some(p => p.observed == null)) {
      refinedWells.push(well);
      continue;
    }
    largeGapProcessed += 1;
    const rawTarget = original.find(w => w.wellId === well.wellId)?.points.map(p => p.observed) ?? well.points.map(p => p.observed);
    const maskedTarget = well.points.map(p => p.observed);
    const truthIdx = rawTarget.map((v, i) => (maskedTarget[i] == null && v != null ? i : -1)).filter(i => i >= 0);
    const random = makeSeededRandom(42 + truthIdx.length);

    const { bestInit, meta } = runIterativeSupportLoop(
      rawTarget,
      maskedTarget,
      truthIdx,
      random,
      {
        outerIterations: config.outerIterations,
        feedbackPrevWeight: config.feedbackPrevWeight,
        supportFrac: 0.12,
        minSupport: 6,
        maxSupport: 24,
      },
      (workingTarget, init) => {
        const points = runExactMcLnnSinglePass(
          {
            targetWellId: well.wellId,
            targetSeries: workingTarget,
            donorSeries: donorMap,
            monthLabels,
            auxByMonth,
            lnnParams: {
              reservoirSize: 20,
              spectralRadius: 0.9,
              leakRate: 0.2,
              inputScaling: 0.5,
              ridgeAlpha: 1e-4,
              maxGapThreshold: config.smallGapThresholdMonths,
              smallGapOptimizeTrials: 8,
            },
            softImpute: config.softImpute,
          },
          init ?? undefined,
        );
        const obs: number[] = [];
        const pred: number[] = [];
        for (const idx of truthIdx) {
          const rv = rawTarget[idx];
          const pv = points[idx]?.imputed;
          if (rv != null && pv != null) {
            obs.push(rv);
            pred.push(pv);
          }
        }
        const rmse = obs.length ? Math.sqrt(obs.reduce((acc, v, i) => acc + (pred[i] - v) ** 2, 0) / obs.length) : undefined;
        return { points, metrics: { rmse } };
      },
      result => {
        const out: Record<number, number> = {};
        result.points.forEach((p, i) => {
          if (p.imputed != null) out[i] = p.imputed;
        });
        return out;
      },
    );

    const finalPoints = runExactMcLnnSinglePass(
      {
        targetWellId: well.wellId,
        targetSeries: maskedTarget,
        donorSeries: donorMap,
        monthLabels,
        auxByMonth,
        lnnParams: {
          reservoirSize: 20,
          spectralRadius: 0.9,
          leakRate: 0.2,
          inputScaling: 0.5,
          ridgeAlpha: 1e-4,
          maxGapThreshold: config.smallGapThresholdMonths,
          smallGapOptimizeTrials: 8,
        },
        softImpute: config.softImpute,
      },
      bestInit ?? undefined,
    );
    const note = `[${well.wellId}] large-gap support: outer=${meta.outerIterationsUsed}, support=${meta.supportPoints}, supportR²=${meta.bestSupportR2 != null ? meta.bestSupportR2.toFixed(3) : 'NA'}, supportRMSE=${meta.bestSupportRmse != null ? meta.bestSupportRmse.toFixed(3) : 'NA'}`;
    notes.push(note);
    callbacks.onNote?.(note);
    refinedWells.push({
      wellId: well.wellId,
      points: finalPoints.map(p => ({
        date: p.date,
        time: p.time,
        observed: p.observed != null ? p.observed : p.imputed,
        auxiliaries: p.auxiliaries,
      })),
    });
    callbacks.onStage?.(`Large-gap iterative MC + LNN... (${largeGapProcessed}/${Math.max(1, wellsNeedingLargeGap.length)} wells, R² ${meta.bestSupportR2 != null ? meta.bestSupportR2.toFixed(3) : 'NA'})`, 50 + (largeGapProcessed / Math.max(1, wellsNeedingLargeGap.length)) * 30);
    await yieldToBrowser();
  }

  callbacks.onStage?.('Final SoftImpute assembly...', 82);
  const mc = softImpute(buildMatrix(refinedWells), config.softImpute);
  const wells = applyCompletedMatrix(original, smallGapWells, refinedWells, mc.completed);
  notes.push('Browser exact path: small-gap LNN-CFC auxiliary + large-gap MC+LNN iterative scaffold connected.');
  return { wells, mc, notes };
}
