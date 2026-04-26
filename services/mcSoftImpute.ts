import { Matrix, SingularValueDecomposition } from 'ml-matrix';

export interface SoftImputeOptions {
  rank?: number;
  shrinkage?: number;
  maxIterations?: number;
  tolerance?: number;
  verbose?: boolean;
}

export interface SoftImputeResult {
  completed: number[][];
  iterations: number;
  converged: boolean;
  relativeChange: number;
}

function clone2d(values: number[][]): number[][] {
  return values.map(row => [...row]);
}

function finiteMean(values: number[]): number {
  const finite = values.filter(v => Number.isFinite(v));
  if (finite.length === 0) return 0;
  return finite.reduce((acc, v) => acc + v, 0) / finite.length;
}

function fillMissingWithColumnMeans(values: number[][], missingMask: boolean[][]): number[][] {
  const rows = values.length;
  const cols = values[0]?.length ?? 0;
  const filled = clone2d(values);
  const colMeans = new Array(cols).fill(0);

  for (let c = 0; c < cols; c++) {
    const col: number[] = [];
    for (let r = 0; r < rows; r++) {
      const v = values[r][c];
      if (!missingMask[r][c] && Number.isFinite(v)) col.push(v);
    }
    colMeans[c] = finiteMean(col);
  }

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (missingMask[r][c] || !Number.isFinite(filled[r][c])) {
        filled[r][c] = colMeans[c];
      }
    }
  }
  return filled;
}

function buildMissingMask(values: number[][]): boolean[][] {
  return values.map(row => row.map(v => !Number.isFinite(v)));
}

function frobeniusNorm(values: number[][]): number {
  let sumSq = 0;
  for (const row of values) {
    for (const v of row) {
      if (Number.isFinite(v)) sumSq += v * v;
    }
  }
  return Math.sqrt(sumSq);
}

function diffFrobeniusNorm(a: number[][], b: number[][]): number {
  let sumSq = 0;
  for (let r = 0; r < a.length; r++) {
    for (let c = 0; c < a[r].length; c++) {
      const d = a[r][c] - b[r][c];
      if (Number.isFinite(d)) sumSq += d * d;
    }
  }
  return Math.sqrt(sumSq);
}

function reconstructLowRank(
  filled: number[][],
  rank: number,
  shrinkage: number,
): number[][] {
  const matrix = new Matrix(filled);
  const svd = new SingularValueDecomposition(matrix, { autoTranspose: true });
  const U = svd.leftSingularVectors;
  const V = svd.rightSingularVectors;
  const s = svd.diagonal;
  const useRank = Math.max(1, Math.min(rank, s.length));

  const sigma = Matrix.zeros(useRank, useRank);
  let kept = 0;
  for (let i = 0; i < useRank; i++) {
    const shrunk = Math.max(0, s[i] - shrinkage);
    sigma.set(i, i, shrunk);
    if (shrunk > 0) kept++;
  }

  if (kept === 0) {
    return Matrix.zeros(matrix.rows, matrix.columns).to2DArray();
  }

  const U_r = U.subMatrix(0, U.rows - 1, 0, useRank - 1);
  const V_r = V.subMatrix(0, V.rows - 1, 0, useRank - 1);
  return U_r.mmul(sigma).mmul(V_r.transpose()).to2DArray();
}

export function softImpute(
  values: number[][],
  options: SoftImputeOptions = {},
): SoftImputeResult {
  if (values.length === 0 || values[0].length === 0) {
    return { completed: [], iterations: 0, converged: true, relativeChange: 0 };
  }

  const rank = options.rank ?? Math.min(8, values.length, values[0].length);
  const shrinkage = options.shrinkage ?? 0;
  const maxIterations = options.maxIterations ?? 100;
  const tolerance = options.tolerance ?? 1e-5;
  const verbose = options.verbose ?? false;

  const missingMask = buildMissingMask(values);
  let current = fillMissingWithColumnMeans(values, missingMask);
  let converged = false;
  let relativeChange = Number.POSITIVE_INFINITY;
  let iteration = 0;

  for (iteration = 1; iteration <= maxIterations; iteration++) {
    const previous = clone2d(current);
    const reconstructed = reconstructLowRank(previous, rank, shrinkage);

    current = clone2d(reconstructed);
    for (let r = 0; r < values.length; r++) {
      for (let c = 0; c < values[r].length; c++) {
        if (!missingMask[r][c] && Number.isFinite(values[r][c])) {
          current[r][c] = values[r][c];
        }
      }
    }

    const diff = diffFrobeniusNorm(current, previous);
    const denom = Math.max(frobeniusNorm(previous), 1e-12);
    relativeChange = diff / denom;
    if (verbose) {
      console.log(`[softImpute] iter=${iteration} relChange=${relativeChange.toExponential(3)}`);
    }
    if (relativeChange <= tolerance) {
      converged = true;
      break;
    }
  }

  return {
    completed: current,
    iterations: iteration,
    converged,
    relativeChange,
  };
}
