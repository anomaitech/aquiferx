import { Matrix, solve } from 'ml-matrix';
import { interpolatePCHIP } from '../utils/interpolation';

export interface BrowserGapPoint {
  date: string;
  time: number;
  value: number | null;
  auxiliaries?: number[];
}

export interface SmallGapBrowserOptions {
  maxGapMonths: number;
  minTrainingPoints?: number;
  ridgeLambda?: number;
}

export interface SmallGapBrowserResult {
  filled: BrowserGapPoint[];
  filledIndices: number[];
  notes: string[];
}

interface GapBlock {
  start: number;
  end: number;
  size: number;
}

function contiguousMissingBlocks(points: BrowserGapPoint[]): GapBlock[] {
  const blocks: GapBlock[] = [];
  let i = 0;
  while (i < points.length) {
    if (points[i].value == null) {
      const start = i;
      while (i < points.length && points[i].value == null) i++;
      blocks.push({ start, end: i - 1, size: i - start });
    } else {
      i++;
    }
  }
  return blocks;
}

function withinObservedSpan(points: BrowserGapPoint[], start: number, end: number): boolean {
  const leftObserved = points.slice(0, start).some(p => p.value != null);
  const rightObserved = points.slice(end + 1).some(p => p.value != null);
  return leftObserved && rightObserved;
}

function normalizeTimes(points: BrowserGapPoint[]): number[] {
  const minT = points[0]?.time ?? 0;
  const maxT = points[points.length - 1]?.time ?? minT + 1;
  const span = Math.max(maxT - minT, 1);
  return points.map(p => ((p.time - minT) / span) * 1.6 - 0.8);
}

function buildTrainingRows(
  points: BrowserGapPoint[],
  timeNorm: number[],
): { X: number[][]; y: number[] } {
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    if (p.value == null) continue;
    X.push([1, timeNorm[i], ...(p.auxiliaries ?? [])]);
    y.push(p.value);
  }
  return { X, y };
}

function fitRidge(X: number[][], y: number[], lambda: number): number[] | null {
  if (X.length === 0) return null;
  const xMat = new Matrix(X);
  const yMat = Matrix.columnVector(y);
  const xtx = xMat.transpose().mmul(xMat);
  for (let i = 0; i < xtx.rows; i++) {
    xtx.set(i, i, xtx.get(i, i) + lambda);
  }
  const xty = xMat.transpose().mmul(yMat);
  try {
    return solve(xtx, xty, true).getColumn(0);
  } catch {
    return null;
  }
}

function predictRow(weights: number[], row: number[]): number {
  let sum = 0;
  for (let i = 0; i < weights.length; i++) sum += weights[i] * row[i];
  return sum;
}

function pchipFillBlock(points: BrowserGapPoint[], start: number, end: number): number[] | null {
  const observed = points
    .map((p, i) => ({ i, t: p.time, v: p.value }))
    .filter(p => p.v != null) as Array<{ i: number; t: number; v: number }>;
  if (observed.length < 2) return null;
  const x = observed.map(p => p.t);
  const y = observed.map(p => p.v);
  const target = points.slice(start, end + 1).map(p => p.time);
  return interpolatePCHIP(x, y, target);
}

export function fillSmallGapsWithAuxBrowser(
  input: BrowserGapPoint[],
  options: SmallGapBrowserOptions,
): SmallGapBrowserResult {
  const filled = input.map(p => ({ ...p, auxiliaries: p.auxiliaries ? [...p.auxiliaries] : undefined }));
  const notes: string[] = [];
  const filledIndices: number[] = [];
  const maxGapMonths = options.maxGapMonths;
  const minTrainingPoints = options.minTrainingPoints ?? 8;
  const ridgeLambda = options.ridgeLambda ?? 1e-2;
  const timeNorm = normalizeTimes(filled);
  const blocks = contiguousMissingBlocks(filled);

  for (const block of blocks) {
    if (block.size > maxGapMonths) continue;
    if (!withinObservedSpan(filled, block.start, block.end)) continue;

    const { X, y } = buildTrainingRows(filled, timeNorm);
    let used = 'ridge';
    let predicted: number[] | null = null;

    if (X.length >= minTrainingPoints) {
      const weights = fitRidge(X, y, ridgeLambda);
      if (weights) {
        predicted = [];
        for (let i = block.start; i <= block.end; i++) {
          const row = [1, timeNorm[i], ...(filled[i].auxiliaries ?? [])];
          predicted.push(predictRow(weights, row));
        }
      }
    }

    if (!predicted) {
      used = 'pchip';
      predicted = pchipFillBlock(filled, block.start, block.end);
    }

    if (!predicted) {
      notes.push(`gap ${block.start}-${block.end}: unable to fill`);
      continue;
    }

    for (let offset = 0; offset < predicted.length; offset++) {
      const idx = block.start + offset;
      filled[idx].value = predicted[offset];
      filledIndices.push(idx);
    }
    notes.push(`gap ${block.start}-${block.end}: filled with ${used}`);
  }

  return { filled, filledIndices, notes };
}
