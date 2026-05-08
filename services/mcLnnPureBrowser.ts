/**
 * MC-LNN Pure Browser Implementation
 *
 * Standalone TypeScript port of the Python MC-LNN groundwater imputer.
 * Pipeline: PCHIP (small gaps) -> SoftImpute MC (spatial) -> LNN CFC (temporal)
 *
 * Key design: MC predictions are used as reservoir INPUT (placeholders)
 * but LNN readout is trained ONLY on real observations.
 */

import { Matrix, SingularValueDecomposition } from 'ml-matrix';

function yieldToUI(): Promise<void> {
  return new Promise(r => setTimeout(r, 0));
}

// ═══════════════════════════════════════════════════════════════════
// Seeded PRNG (xoshiro128**)
// ═══════════════════════════════════════════════════════════════════

class SeededRng {
  private s: Uint32Array;

  constructor(seed: number) {
    this.s = new Uint32Array(4);
    // SplitMix64 seed expansion
    let z = (seed >>> 0) + 0x9e3779b9;
    for (let i = 0; i < 4; i++) {
      z = (z ^ (z >>> 16)) * 0x85ebca6b;
      z = (z ^ (z >>> 13)) * 0xc2b2ae35;
      z = z ^ (z >>> 16);
      this.s[i] = z >>> 0;
    }
  }

  private rotl(x: number, k: number): number {
    return ((x << k) | (x >>> (32 - k))) >>> 0;
  }

  nextU32(): number {
    const result = (this.rotl((this.s[1] * 5) >>> 0, 7) * 9) >>> 0;
    const t = (this.s[1] << 9) >>> 0;
    this.s[2] ^= this.s[0];
    this.s[3] ^= this.s[1];
    this.s[1] ^= this.s[2];
    this.s[0] ^= this.s[3];
    this.s[2] ^= t;
    this.s[3] = this.rotl(this.s[3], 11);
    return result;
  }

  /** Uniform [0, 1) */
  random(): number {
    return this.nextU32() / 0x100000000;
  }

  /** Uniform [-1, 1) */
  randomSigned(): number {
    return this.random() * 2 - 1;
  }

  /** Integer in [lo, hi) */
  randInt(lo: number, hi: number): number {
    return lo + Math.floor(this.random() * (hi - lo));
  }

  /** Normal(0, 1) via Box-Muller */
  normal(): number {
    const u1 = Math.max(1e-10, this.random());
    const u2 = this.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
}

// ═══════════════════════════════════════════════════════════════════
// Math Utilities
// ═══════════════════════════════════════════════════════════════════

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function matMul(A: number[][], B: number[][]): number[][] {
  const m = A.length, k = B.length, n = B[0].length;
  const C: number[][] = Array.from({ length: m }, () => new Array(n).fill(0));
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++)
      for (let p = 0; p < k; p++)
        C[i][j] += A[i][p] * B[p][j];
  return C;
}

function matVecMul(A: number[][], v: number[]): number[] {
  return A.map(row => dot(row, v));
}

function transpose(A: number[][]): number[][] {
  const m = A.length, n = A[0].length;
  const T: number[][] = Array.from({ length: n }, () => new Array(m));
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++)
      T[j][i] = A[i][j];
  return T;
}

function ridgeRegression(X: number[][], y: number[], alpha: number = 1e-4): number[] {
  const n = X.length, p = X[0].length;
  if (n === 0 || p === 0) return [];
  const Xt = transpose(X);
  const XtX = matMul(Xt, X.map(r => [r]).map(r => r[0]).length ? Xt.map((row, i) => X.map(xr => xr[i])).map(col => {
    // Rebuild XtX properly
    return [];
  }) : []);

  // Direct computation: XtX = Xt @ X
  const A: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
  for (let i = 0; i < p; i++)
    for (let j = 0; j < p; j++)
      for (let k = 0; k < n; k++)
        A[i][j] += X[k][i] * X[k][j];

  // Add ridge: A + alpha * I
  for (let i = 0; i < p; i++) A[i][i] += alpha;

  // Xty = Xt @ y
  const b: number[] = new Array(p).fill(0);
  for (let i = 0; i < p; i++)
    for (let k = 0; k < n; k++)
      b[i] += X[k][i] * y[k];

  // Solve A w = b via Gaussian elimination with partial pivoting
  return solveLinear(A, b);
}

function solveLinear(A: number[][], b: number[]): number[] {
  const n = A.length;
  const M: number[][] = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    // Partial pivot
    let maxRow = col;
    let maxVal = Math.abs(M[col][col]);
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(M[row][col]) > maxVal) {
        maxVal = Math.abs(M[row][col]);
        maxRow = row;
      }
    }
    [M[col], M[maxRow]] = [M[maxRow], M[col]];

    if (Math.abs(M[col][col]) < 1e-12) continue;

    for (let row = col + 1; row < n; row++) {
      const factor = M[row][col] / M[col][col];
      for (let j = col; j <= n; j++) M[row][j] -= factor * M[col][j];
    }
  }

  // Back substitution
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = M[i][n];
    for (let j = i + 1; j < n; j++) sum -= M[i][j] * x[j];
    x[i] = Math.abs(M[i][i]) > 1e-12 ? sum / M[i][i] : 0;
  }
  return x;
}

function pearsonR(x: number[], y: number[]): number {
  const n = x.length;
  if (n < 3) return 0;
  const mx = x.reduce((a, b) => a + b, 0) / n;
  const my = y.reduce((a, b) => a + b, 0) / n;
  let num = 0, dx2 = 0, dy2 = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - mx, dy = y[i] - my;
    num += dx * dy;
    dx2 += dx * dx;
    dy2 += dy * dy;
  }
  const denom = Math.sqrt(dx2 * dy2);
  return denom > 1e-10 ? num / denom : 0;
}

function computeKGE(obs: number[], pred: number[]): number {
  if (obs.length < 2) return -Infinity;
  const r = pearsonR(obs, pred);
  const mObs = obs.reduce((a, b) => a + b, 0) / obs.length;
  const mPred = pred.reduce((a, b) => a + b, 0) / pred.length;
  const sObs = Math.sqrt(obs.reduce((a, b) => a + (b - mObs) ** 2, 0) / (obs.length - 1));
  const sPred = Math.sqrt(pred.reduce((a, b) => a + (b - mPred) ** 2, 0) / (pred.length - 1));
  const alpha = sObs > 1e-10 ? sPred / sObs : 1;
  const beta = Math.abs(mObs) > 1e-10 ? mPred / mObs : 1;
  return 1 - Math.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2);
}

function computeNRMSE(obs: number[], pred: number[]): number {
  if (obs.length < 2) return Infinity;
  let mse = 0;
  for (let i = 0; i < obs.length; i++) mse += (obs[i] - pred[i]) ** 2;
  const rmse = Math.sqrt(mse / obs.length);
  const mObs = obs.reduce((a, b) => a + b, 0) / obs.length;
  const sd = Math.sqrt(obs.reduce((a, b) => a + (b - mObs) ** 2, 0) / (obs.length - 1));
  return sd > 1e-10 ? rmse / sd : Infinity;
}

// ═══════════════════════════════════════════════════════════════════
// Truncated SVD via Power Iteration
// ═══════════════════════════════════════════════════════════════════

interface SvdResult {
  U: number[][]; // m x k
  S: number[];   // k
  Vt: number[][]; // k x n
}

function truncatedSvd(M: number[][], k: number, rng: SeededRng, nIter: number = 20): SvdResult {
  const m = M.length, n = M[0].length;
  const actualK = Math.min(k, m, n);

  // Randomized SVD: Q = orth(M @ Omega), then SVD of Q' @ M
  // Step 1: Random projection
  const Omega: number[][] = Array.from({ length: n }, () =>
    Array.from({ length: actualK }, () => rng.normal())
  );
  let Y = matMul(M, Omega); // m x k

  // Power iteration for better accuracy
  const Mt = transpose(M);
  for (let iter = 0; iter < nIter; iter++) {
    Y = matMul(Mt, Y);  // n x k
    Y = matMul(M, Y);   // m x k
  }

  // QR decomposition of Y (modified Gram-Schmidt)
  const Q = gramSchmidt(Y, actualK);

  // B = Q' @ M  (k x n)
  const Qt = transpose(Q);
  const B = matMul(Qt, M);

  // SVD of small matrix B (k x n) using eigendecomposition of B @ B'
  const BBt = matMul(B, transpose(B)); // k x k
  const { values: eigVals, vectors: eigVecs } = symmetricEigen(BBt);

  const S: number[] = [];
  const U: number[][] = Array.from({ length: m }, () => new Array(actualK).fill(0));
  const Vt: number[][] = Array.from({ length: actualK }, () => new Array(n).fill(0));

  for (let i = 0; i < actualK; i++) {
    const si = Math.sqrt(Math.max(eigVals[i], 0));
    S.push(si);
    if (si < 1e-10) continue;

    // V_i = B' @ eigVec_i / sigma_i
    const bCol = B[0].map((_, j) => {
      let s = 0;
      for (let r = 0; r < actualK; r++) s += B[r][j] * eigVecs[r][i];
      return s / si;
    });
    Vt[i] = bCol;

    // U_i = Q @ eigVec_i
    for (let r = 0; r < m; r++) {
      let s = 0;
      for (let c = 0; c < actualK; c++) s += Q[r][c] * eigVecs[c][i];
      U[r][i] = s;
    }
  }

  return { U, S, Vt };
}

function gramSchmidt(Y: number[][], k: number): number[][] {
  const m = Y.length;
  const Q: number[][] = Array.from({ length: m }, () => new Array(k).fill(0));

  for (let j = 0; j < k; j++) {
    // Copy column j
    for (let i = 0; i < m; i++) Q[i][j] = Y[i][j];

    // Orthogonalize against previous columns
    for (let p = 0; p < j; p++) {
      let d = 0;
      for (let i = 0; i < m; i++) d += Q[i][j] * Q[i][p];
      for (let i = 0; i < m; i++) Q[i][j] -= d * Q[i][p];
    }

    // Normalize
    let norm = 0;
    for (let i = 0; i < m; i++) norm += Q[i][j] ** 2;
    norm = Math.sqrt(norm);
    if (norm > 1e-10) {
      for (let i = 0; i < m; i++) Q[i][j] /= norm;
    }
  }
  return Q;
}

function symmetricEigen(A: number[][]): { values: number[]; vectors: number[][] } {
  // Jacobi eigenvalue algorithm for small symmetric matrices
  const n = A.length;
  const V: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => i === j ? 1 : 0)
  );
  const D: number[][] = A.map(r => [...r]);

  for (let sweep = 0; sweep < 100; sweep++) {
    let offDiag = 0;
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++)
        offDiag += D[i][j] ** 2;
    if (offDiag < 1e-20) break;

    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(D[p][q]) < 1e-15) continue;
        const theta = 0.5 * Math.atan2(2 * D[p][q], D[p][p] - D[q][q]);
        const c = Math.cos(theta), s = Math.sin(theta);

        // Rotate D
        const newPP = c * c * D[p][p] + 2 * s * c * D[p][q] + s * s * D[q][q];
        const newQQ = s * s * D[p][p] - 2 * s * c * D[p][q] + c * c * D[q][q];
        D[p][q] = 0;
        D[q][p] = 0;
        D[p][p] = newPP;
        D[q][q] = newQQ;

        for (let i = 0; i < n; i++) {
          if (i === p || i === q) continue;
          const dip = D[i][p], diq = D[i][q];
          D[i][p] = D[p][i] = c * dip + s * diq;
          D[i][q] = D[q][i] = -s * dip + c * diq;
        }

        // Rotate V
        for (let i = 0; i < n; i++) {
          const vip = V[i][p], viq = V[i][q];
          V[i][p] = c * vip + s * viq;
          V[i][q] = -s * vip + c * viq;
        }
      }
    }
  }

  // Sort by descending eigenvalue
  const indices = Array.from({ length: n }, (_, i) => i);
  indices.sort((a, b) => D[b][b] - D[a][a]);

  return {
    values: indices.map(i => D[i][i]),
    vectors: Array.from({ length: n }, (_, r) =>
      indices.map(i => V[r][i])
    ),
  };
}

// ═══════════════════════════════════════════════════════════════════
// SoftImpute Matrix Completion
// ═══════════════════════════════════════════════════════════════════

function softImpute(
  M: number[][], // matrix with NaN for missing
  maxRank: number,
  maxIters: number = 100,
  tol: number = 1e-6,
  _rng: SeededRng,
): number[][] {
  const m = M.length, n = M[0].length;
  const obsMask: boolean[][] = M.map(r => r.map(v => !isNaN(v)));

  // Initialize: fill NaN with row means
  const X: number[][] = M.map(r => {
    const vals = r.filter(v => !isNaN(v));
    const rowMean = vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
    return r.map(v => isNaN(v) ? rowMean : v);
  });

  const k = Math.min(maxRank, m, n);

  for (let iter = 0; iter < maxIters; iter++) {
    // SVD via ml-matrix (fast, compiled-quality)
    const mat = new Matrix(X);
    const svd = new SingularValueDecomposition(mat, { autoTranspose: true });
    const U = svd.leftSingularVectors;
    const S = svd.diagonal;
    const Vt = svd.rightSingularVectors.transpose();

    // Truncated reconstruction: keep only top-k singular values
    const Xnew: number[][] = Array.from({ length: m }, () => new Array(n).fill(0));
    for (let i = 0; i < m; i++)
      for (let j = 0; j < n; j++) {
        let val = 0;
        for (let c = 0; c < k && c < S.length; c++)
          val += U.get(i, c) * S[c] * Vt.get(c, j);
        Xnew[i][j] = val;
      }

    // Re-insert observed values
    for (let i = 0; i < m; i++)
      for (let j = 0; j < n; j++)
        if (obsMask[i][j]) Xnew[i][j] = M[i][j];

    // Check convergence
    let maxDiff = 0;
    for (let i = 0; i < m; i++)
      for (let j = 0; j < n; j++)
        maxDiff = Math.max(maxDiff, Math.abs(Xnew[i][j] - X[i][j]));

    for (let i = 0; i < m; i++)
      for (let j = 0; j < n; j++)
        X[i][j] = Xnew[i][j];

    if (maxDiff < tol) break;
  }
  return X;
}

// ═══════════════════════════════════════════════════════════════════
// PCHIP Interpolation
// ═══════════════════════════════════════════════════════════════════

function pchipFill(series: (number | null)[], maxGap: number = 24): (number | null)[] {
  const out = [...series];
  const obsIdx: number[] = [];
  const obsVals: number[] = [];
  for (let i = 0; i < series.length; i++) {
    if (series[i] !== null) {
      obsIdx.push(i);
      obsVals.push(series[i]!);
    }
  }
  if (obsIdx.length < 3) return out;

  // Identify small gaps
  const gaps: [number, number][] = [];
  let i = 0;
  while (i < series.length) {
    if (series[i] === null) {
      const start = i;
      while (i < series.length && series[i] === null) i++;
      if (i - start <= maxGap) gaps.push([start, i - 1]);
    } else {
      i++;
    }
  }
  if (!gaps.length) return out;

  // Compute PCHIP derivatives (Fritsch-Carlson)
  const n = obsIdx.length;
  const h: number[] = [];
  const delta: number[] = [];
  for (let i = 0; i < n - 1; i++) {
    h.push(obsIdx[i + 1] - obsIdx[i]);
    delta.push(h[i] === 0 ? 0 : (obsVals[i + 1] - obsVals[i]) / h[i]);
  }

  const d = new Array(n).fill(0);
  for (let i = 1; i < n - 1; i++) {
    if (delta[i - 1] * delta[i] <= 0) {
      d[i] = 0;
    } else {
      const w1 = 2 * h[i] + h[i - 1];
      const w2 = h[i] + 2 * h[i - 1];
      d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
    }
  }
  // Endpoints
  if (n >= 3) {
    d[0] = pchipEndSlope(h[0], h[1], delta[0], delta[1]);
    d[n - 1] = pchipEndSlope(h[n - 2], h[n - 3] ?? h[n - 2], delta[n - 2], delta[n - 3] ?? delta[n - 2]);
  }

  // Interpolate
  for (const [gStart, gEnd] of gaps) {
    for (let t = gStart; t <= gEnd; t++) {
      // Find interval
      let lo = 0, hi = n - 1;
      while (lo < hi - 1) {
        const mid = (lo + hi) >> 1;
        if (obsIdx[mid] <= t) lo = mid; else hi = mid;
      }
      if (h[lo] === 0) { out[t] = obsVals[lo]; continue; }
      const u = (t - obsIdx[lo]) / h[lo];
      const u2 = u * u, u3 = u2 * u;
      const h00 = 2 * u3 - 3 * u2 + 1;
      const h10 = u3 - 2 * u2 + u;
      const h01 = -2 * u3 + 3 * u2;
      const h11 = u3 - u2;
      const val = h00 * obsVals[lo] + h10 * h[lo] * d[lo] + h01 * obsVals[lo + 1] + h11 * h[lo] * d[lo + 1];
      if (isFinite(val)) out[t] = val;
    }
  }
  return out;
}

function pchipEndSlope(h0: number, h1: number, d0: number, d1: number): number {
  if (h0 === 0) return d1;
  if (h1 === 0) return d0;
  const slope = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
  if (Math.sign(slope) !== Math.sign(d0)) return 0;
  if (Math.sign(d0) !== Math.sign(d1) && Math.abs(slope) > 3 * Math.abs(d0)) return 3 * d0;
  return slope;
}

// ═══════════════════════════════════════════════════════════════════
// ARCHI Donor Regression
// ═══════════════════════════════════════════════════════════════════

interface DonorInfo {
  wid: string;
  r: number;
  dobs: Map<number, number>;
}

function archiRegression(
  targetObs: Map<number, number>,
  donorObs: Map<string, (number | null)[]>,
  targetId: string,
  maxDonors: number = 50,
): { preds: Map<number, number>; donors: DonorInfo[] } {
  if (targetObs.size < 5) return { preds: new Map(), donors: [] };

  const nTimes = Math.max(...Array.from(donorObs.values()).map(s => s.length));
  const gapTimes = new Set<number>();
  for (let t = 0; t < nTimes; t++) if (!targetObs.has(t)) gapTimes.add(t);

  const donors: DonorInfo[] = [];
  for (const [wid, series] of Array.from(donorObs.entries())) {
    if (wid === targetId) continue;
    const dobs = new Map<number, number>();
    for (let t = 0; t < series.length; t++) if (series[t] !== null) dobs.set(t, series[t]!);
    const common: number[] = [];
    for (const t of Array.from(targetObs.keys())) if (dobs.has(t)) common.push(t);
    if (common.length < 8) continue;

    const tv = common.map(t => targetObs.get(t)!);
    const dv = common.map(t => dobs.get(t)!);
    const r = pearsonR(tv, dv);
    if (!isFinite(r)) continue;
    donors.push({ wid, r, dobs });
  }

  donors.sort((a, b) => Math.abs(b.r) - Math.abs(a.r));
  donors.length = Math.min(donors.length, maxDonors);
  if (!donors.length) return { preds: new Map(), donors: [] };

  // Per-donor OLS + weighted average
  const predsByDonor: Map<number, number>[] = [];
  const weights: number[] = [];
  for (const di of donors) {
    const common: number[] = [];
    for (const t of Array.from(targetObs.keys())) if (di.dobs.has(t)) common.push(t);
    if (common.length < 5) continue;
    const tv = common.map(t => targetObs.get(t)!);
    const dv = common.map(t => di.dobs.get(t)!);
    const dm = dv.reduce((a, b) => a + b, 0) / dv.length;
    const tm = tv.reduce((a, b) => a + b, 0) / tv.length;
    const ss = dv.reduce((a, v) => a + (v - dm) ** 2, 0);
    if (ss < 1e-10) continue;
    const a = dv.reduce((acc, v, i) => acc + (v - dm) * (tv[i] - tm), 0) / ss;
    const b = tm - a * dm;
    const preds = new Map<number, number>();
    for (const t of Array.from(gapTimes)) {
      if (di.dobs.has(t)) preds.set(t, a * di.dobs.get(t)! + b);
    }
    if (preds.size) {
      predsByDonor.push(preds);
      weights.push(di.r ** 2);
    }
  }

  const combined = new Map<number, number>();
  for (const t of Array.from(gapTimes)) {
    let ws = 0, wc = 0;
    for (let i = 0; i < predsByDonor.length; i++) {
      const v = predsByDonor[i].get(t);
      if (v !== undefined) { ws += v * weights[i]; wc += weights[i]; }
    }
    if (wc > 0) combined.set(t, ws / wc);
  }
  return { preds: combined, donors };
}

// ═══════════════════════════════════════════════════════════════════
// Matrix Completion (SoftImpute with ARCHI init)
// ═══════════════════════════════════════════════════════════════════

function mcSoftImpute(
  targetSeries: (number | null)[],
  donorObs: Map<string, (number | null)[]>,
  donors: DonorInfo[],
  archiPreds: Map<number, number>,
  targetAux: number[][] | null,
  rng: SeededRng,
): Map<number, number> {
  const nTimes = targetSeries.length;
  const targetObsIdx: number[] = [];
  for (let i = 0; i < nTimes; i++) if (targetSeries[i] !== null) targetObsIdx.push(i);

  const selWids = [null, ...donors.map(d => d.wid)];
  const nw = selWids.length;
  const nAux = targetAux ? targetAux[0].length : 0;

  // Build raw matrix
  const MRaw: number[][] = Array.from({ length: nw }, () => new Array(nTimes).fill(NaN));
  for (let t = 0; t < nTimes; t++) if (targetSeries[t] !== null) MRaw[0][t] = targetSeries[t]!;
  for (let wi = 1; wi < nw; wi++) {
    const series = donorObs.get(selWids[wi]!)!;
    for (let t = 0; t < Math.min(series.length, nTimes); t++)
      if (series[t] !== null) MRaw[wi][t] = series[t]!;
  }

  // Normalize per row
  const rmeans = new Array(nw).fill(0);
  const rstds = new Array(nw).fill(1);
  const M: number[][] = Array.from({ length: nw + nAux + 2 }, () => new Array(nTimes).fill(NaN));

  for (let wi = 0; wi < nw; wi++) {
    const vals = MRaw[wi].filter(v => !isNaN(v));
    if (vals.length >= 3) {
      rmeans[wi] = vals.reduce((a, b) => a + b, 0) / vals.length;
      rstds[wi] = Math.max(Math.sqrt(vals.reduce((a, v) => a + (v - rmeans[wi]) ** 2, 0) / vals.length), 1e-10);
    } else if (vals.length > 0) {
      rmeans[wi] = vals.reduce((a, b) => a + b, 0) / vals.length;
    }
    for (let t = 0; t < nTimes; t++)
      M[wi][t] = isNaN(MRaw[wi][t]) ? NaN : (MRaw[wi][t] - rmeans[wi]) / rstds[wi];
  }

  // Weight donor rows by |r|
  for (let di = 0; di < donors.length; di++)
    for (let t = 0; t < nTimes; t++)
      if (!isNaN(M[di + 1][t])) M[di + 1][t] *= Math.abs(donors[di].r);

  // Auxiliary rows
  if (targetAux) {
    for (let j = 0; j < nAux; j++) {
      const ac = targetAux.map(row => row[j]);
      const am = ac.reduce((a, b) => a + b, 0) / ac.length;
      const ast = Math.max(Math.sqrt(ac.reduce((a, v) => a + (v - am) ** 2, 0) / ac.length), 1e-10);
      for (let t = 0; t < nTimes; t++) M[nw + j][t] = (ac[t] - am) / ast;
    }
  }

  // Temporal encoding rows
  for (let t = 0; t < nTimes; t++) {
    M[nw + nAux][t] = Math.sin(2 * Math.PI * t / 12);
    M[nw + nAux + 1][t] = Math.cos(2 * Math.PI * t / 12);
  }

  // Adaptive rank selection
  let bestK = 5, bestErr = Infinity;
  for (const kTry of [3, 5, 8, 10, 12]) {
    if (kTry >= Math.min(M.length, nTimes)) continue;
    const Xt = softImpute(M, kTry, 50, 1e-6, rng);
    if (targetObsIdx.length > 3) {
      let err = 0;
      for (const t of targetObsIdx) err += (Xt[0][t] - M[0][t]) ** 2;
      err /= targetObsIdx.length;
      if (err < bestErr) { bestErr = err; bestK = kTry; }
    }
  }

  // Final run with best rank
  const X = softImpute(M, bestK, 100, 1e-7, rng);

  // Undo donor weighting
  for (let di = 0; di < donors.length; di++)
    if (Math.abs(donors[di].r) > 1e-10)
      for (let t = 0; t < nTimes; t++) X[di + 1][t] /= Math.abs(donors[di].r);

  // Denormalize
  const pred = X[0].map(v => v * rstds[0] + rmeans[0]);

  // MOVE.1 variance-preserving bias correction
  if (targetObsIdx.length >= 3) {
    const ov = targetObsIdx.map(t => MRaw[0][t]);
    const mv = targetObsIdx.map(t => pred[t]);
    const om = ov.reduce((a, b) => a + b, 0) / ov.length;
    const os = Math.max(Math.sqrt(ov.reduce((a, v) => a + (v - om) ** 2, 0) / (ov.length - 1)), 1e-10);
    const mm = mv.reduce((a, b) => a + b, 0) / mv.length;
    const ms = Math.max(Math.sqrt(mv.reduce((a, v) => a + (v - mm) ** 2, 0) / (mv.length - 1)), 1e-10);
    for (let t = 0; t < nTimes; t++) pred[t] = om + (os / ms) * (pred[t] - mm);
  }

  const result = new Map<number, number>();
  for (let t = 0; t < nTimes; t++) if (isFinite(pred[t])) result.set(t, pred[t]);
  return result;
}

// ═══════════════════════════════════════════════════════════════════
// LNN CFC Reservoir
// ═══════════════════════════════════════════════════════════════════

interface LnnParams {
  reservoirSize: number;
  leakRate: number;
  inputScaling: number;
  spectralRadius: number;
  ridgeAlpha: number;
}

function runLnnCfc(
  observed: (number | null)[],  // real observations (null = missing)
  mcPlaceholders: Map<number, number>,  // MC predictions for reservoir input
  aux: number[][] | null,  // auxiliary features per timestep
  params: LnnParams,
  rng: SeededRng,
): number[] {
  const n = observed.length;
  const nAux = aux ? aux[0].length : 0;
  const inputDim = 1 + nAux + 2; // obs/placeholder + aux + sin/cos
  const { reservoirSize, leakRate, inputScaling, spectralRadius, ridgeAlpha } = params;

  // Normalize to [-0.8, 0.8] using ALL available values (obs + MC placeholders)
  // This matches Python which uses obs_scaler from prepare_data on the enriched timeline
  const allVals: number[] = [];
  for (let i = 0; i < n; i++) {
    if (observed[i] !== null) allVals.push(observed[i]!);
    else if (mcPlaceholders.has(i)) allVals.push(mcPlaceholders.get(i)!);
  }
  if (allVals.length < 2) return observed.map(v => v ?? 0);
  const obsMin = Math.min(...allVals);
  const obsMax = Math.max(...allVals);
  const obsRange = Math.max(obsMax - obsMin, 1e-10);

  function normalize(v: number): number {
    return ((v - obsMin) / obsRange) * 1.6 - 0.8;
  }
  function denormalize(v: number): number {
    return ((v + 0.8) / 1.6) * obsRange + obsMin;
  }

  // Normalize aux
  const auxMeans: number[] = [], auxStds: number[] = [];
  if (aux) {
    for (let j = 0; j < nAux; j++) {
      const col = aux.map(r => r[j]);
      const m = col.reduce((a, b) => a + b, 0) / col.length;
      const s = Math.max(Math.sqrt(col.reduce((a, v) => a + (v - m) ** 2, 0) / col.length), 1e-10);
      auxMeans.push(m);
      auxStds.push(s);
    }
  }

  // Initialize weights
  const Win: number[][] = Array.from({ length: reservoirSize }, () =>
    Array.from({ length: inputDim }, () => rng.randomSigned() * inputScaling)
  );

  const Wres: number[][] = Array.from({ length: reservoirSize }, () =>
    Array.from({ length: reservoirSize }, () => rng.random() < 0.2 ? rng.randomSigned() : 0)
  );

  // Spectral radius normalization via ml-matrix eigenvalues
  try {
    const wresMat = new Matrix(Wres);
    const eig = new SingularValueDecomposition(wresMat, { autoTranspose: true });
    const actualSr = eig.diagonal.length > 0 ? eig.diagonal[0] : 1;
    if (actualSr > 1e-10) {
      const scale = spectralRadius / actualSr;
      for (let i = 0; i < reservoirSize; i++)
        for (let j = 0; j < reservoirSize; j++)
          Wres[i][j] *= scale;
    }
  } catch {
    // Fallback: power iteration
    let v = Array.from({ length: reservoirSize }, () => rng.random());
    for (let iter = 0; iter < 50; iter++) {
      const Av = matVecMul(Wres, v);
      const norm = Math.sqrt(Av.reduce((a, x) => a + x * x, 0));
      if (norm < 1e-10) break;
      v = Av.map(x => x / norm);
    }
    const actualSr = Math.sqrt(matVecMul(Wres, v).reduce((a, x) => a + x * x, 0));
    if (actualSr > 1e-10) {
      const scale = spectralRadius / actualSr;
      for (let i = 0; i < reservoirSize; i++)
        for (let j = 0; j < reservoirSize; j++)
          Wres[i][j] *= scale;
    }
  }

  // Run reservoir
  const x = new Array(reservoirSize).fill(0);
  const allStates: number[][] = [];
  const trainStates: number[][] = [];
  const trainTargets: number[] = [];
  let lastValidObs = 0;
  const leak = Math.max(leakRate, 1e-6);
  const dt = 0.1;

  for (let i = 0; i < n; i++) {
    // Priority: real obs > MC placeholder > last valid
    let currentInput: number;
    if (observed[i] !== null) {
      currentInput = normalize(observed[i]!);
      lastValidObs = currentInput;
    } else if (mcPlaceholders.has(i)) {
      currentInput = normalize(mcPlaceholders.get(i)!);
    } else {
      currentInput = lastValidObs;
    }

    // Build input vector
    const inputVec: number[] = [currentInput];
    if (aux) {
      for (let j = 0; j < nAux; j++) {
        inputVec.push((aux[i][j] - auxMeans[j]) / auxStds[j]);
      }
    }
    inputVec.push(Math.sin(2 * Math.PI * i / 12));
    inputVec.push(Math.cos(2 * Math.PI * i / 12));
    while (inputVec.length < inputDim) inputVec.push(0);

    // CFC update
    const stepExp = Math.exp(-leak * dt);
    const stepCoef = (1 - stepExp) / leak;
    const pre = matVecMul(Win, inputVec);
    const b = pre.map((v, idx) => Math.tanh(v + dot(Wres[idx], x)));
    for (let j = 0; j < reservoirSize; j++) {
      x[j] = x[j] * stepExp + stepCoef * b[j];
    }

    allStates.push([1, ...x]);

    // Train ONLY on real observations (not MC placeholders)
    if (observed[i] !== null) {
      trainStates.push([1, ...x]);
      trainTargets.push(normalize(observed[i]!));
    }
  }

  if (trainStates.length === 0) return observed.map(v => v ?? 0);

  // Ridge readout
  const Wout = ridgeRegression(trainStates, trainTargets, ridgeAlpha);
  if (Wout.length === 0) return observed.map(v => v ?? 0);

  // Predict all points
  return allStates.map(state => denormalize(dot(Wout, state)));
}

// ═══════════════════════════════════════════════════════════════════
// Hyperparameter Optimization
// ═══════════════════════════════════════════════════════════════════

function optimizeLnnParams(
  observed: (number | null)[],
  mcPlaceholders: Map<number, number>,
  aux: number[][] | null,
  rng: SeededRng,
  nTrials: number = 8,
): LnnParams {
  let bestParams: LnnParams = {
    reservoirSize: 30, leakRate: 0.2, inputScaling: 0.1,
    spectralRadius: 0.9, ridgeAlpha: 1e-4,
  };
  let bestScore = -Infinity;

  const obsIdx: number[] = [];
  const obsVals: number[] = [];
  for (let i = 0; i < observed.length; i++) {
    if (observed[i] !== null) { obsIdx.push(i); obsVals.push(observed[i]!); }
  }
  if (obsVals.length < 3) return bestParams;

  for (let trial = 0; trial < nTrials; trial++) {
    const params: LnnParams = {
      reservoirSize: rng.randInt(10, 81),
      leakRate: 0.05 + rng.random() * 0.90,
      inputScaling: 0.01 + rng.random() * 0.39,
      spectralRadius: 0.9,
      ridgeAlpha: 1e-4,
    };

    const pred = runLnnCfc(observed, mcPlaceholders, aux, params, new SeededRng(rng.nextU32()));
    const predAtObs = obsIdx.map(i => pred[i]);

    const kge = computeKGE(obsVals, predAtObs);
    const nrmse = computeNRMSE(obsVals, predAtObs);

    // Blend score: 50% KGE + 50% (1 - NRMSE)
    const score = 0.5 * kge + 0.5 * (1 - nrmse);
    if (score > bestScore) {
      bestScore = score;
      bestParams = params;
    }
    if (bestScore >= 0.9) break;
  }
  return bestParams;
}

// ═══════════════════════════════════════════════════════════════════
// Full MC-LNN Pipeline
// ═══════════════════════════════════════════════════════════════════

export interface McRunConfig {
  seed?: number;
  maxDonors?: number;
  gapSize?: number;  // months (default 24 = ~730 days)
  padSize?: number;  // months (default 6 = ~180 days)
  optimizeTrials?: number;
  lnnEnsembleSize?: number;
}

export async function runMcLnnImputation(
  wellSeries: Map<string, (number | null)[]>,
  auxData: number[][] | null,  // [soilw, yr01, yr03, yr05, yr10] per month
  targetWellIds: string[],
  config?: McRunConfig,
  onProgress?: (msg: string, pct: number) => void,
): Promise<Map<string, number[]>> {
  const seed = config?.seed ?? 42;
  const maxDonors = config?.maxDonors ?? 50;
  const gapSize = config?.gapSize ?? 24;
  const padSize = config?.padSize ?? 6;
  const nTrials = config?.optimizeTrials ?? 8;
  const ensembleSize = config?.lnnEnsembleSize ?? 3;
  const rng = new SeededRng(seed);

  // Phase 1: PCHIP small-gap fill + large-gap blanking (aquiferx style)
  onProgress?.('PCHIP small-gap fill...', 5);
  const filled = new Map<string, (number | null)[]>();
  for (const [wid, raw] of Array.from(wellSeries.entries())) {
    // PCHIP fill everything within observation range
    let out = pchipFill(raw, gapSize * 2); // fill all gaps first

    // Blank interior of large gaps
    const obsIdx: number[] = [];
    for (let i = 0; i < raw.length; i++) if (raw[i] !== null) obsIdx.push(i);
    for (let g = 0; g < obsIdx.length - 1; g++) {
      const gapLen = obsIdx[g + 1] - obsIdx[g] - 1;
      if (gapLen > gapSize) {
        const blankStart = obsIdx[g] + padSize + 1;
        const blankEnd = obsIdx[g + 1] - padSize - 1;
        for (let t = Math.max(blankStart, 0); t <= Math.min(blankEnd, out.length - 1); t++) {
          out[t] = null;
        }
      }
    }
    filled.set(wid, out);
  }

  // Phase 2: MC-LNN for each target well
  const results = new Map<string, number[]>();

  for (let wi = 0; wi < targetWellIds.length; wi++) {
    const wid = targetWellIds[wi];
    const pct = 10 + (wi / targetWellIds.length) * 85;
    onProgress?.(`MC-LNN well ${wi + 1}/${targetWellIds.length}...`, pct);
    await yieldToUI();

    const modTarget = filled.get(wid);
    if (!modTarget) continue;
    const nTimes = modTarget.length;

    // Build target observations
    const targetObs = new Map<number, number>();
    for (let t = 0; t < nTimes; t++) if (modTarget[t] !== null) targetObs.set(t, modTarget[t]!);

    // ARCHI donor regression
    const { preds: archiPreds, donors } = archiRegression(targetObs, filled, wid, maxDonors);

    // Build aux for this well
    const targetAux = auxData ? auxData.slice(0, nTimes) : null;

    let mcPreds: Map<number, number>;
    if (donors.length === 0) {
      mcPreds = archiPreds;
    } else {
      // SoftImpute matrix completion
      mcPreds = mcSoftImpute(
        modTarget, filled, donors, archiPreds, targetAux,
        new SeededRng(rng.nextU32()),
      );
    }

    // Identify which are real observations vs MC placeholders
    const realObs: (number | null)[] = [...modTarget];
    const mcPlaceholders = new Map<number, number>();
    for (let t = 0; t < nTimes; t++) {
      if (modTarget[t] === null && mcPreds.has(t) && isFinite(mcPreds.get(t)!)) {
        mcPlaceholders.set(t, mcPreds.get(t)!);
      }
    }

    // LNN with ensemble
    await yieldToUI();
    const bestParams = optimizeLnnParams(realObs, mcPlaceholders, targetAux, new SeededRng(rng.nextU32()), nTrials);

    await yieldToUI();
    let bestPred: number[] | null = null;
    let bestKge = -Infinity;
    const obsIdx = Array.from(targetObs.keys());
    const obsVals = obsIdx.map(t => targetObs.get(t)!);

    for (let e = 0; e < ensembleSize; e++) {
      const pred = runLnnCfc(realObs, mcPlaceholders, targetAux, bestParams, new SeededRng(rng.nextU32() + e));
      const predAtObs = obsIdx.map(t => pred[t]);
      const kge = computeKGE(obsVals, predAtObs);
      if (kge > bestKge) { bestKge = kge; bestPred = pred; }
      await yieldToUI();
    }

    const finalPred = bestPred ?? new Array(nTimes).fill(0);
    results.set(wid, finalPred);

    // Report per-well metrics
    if (obsVals.length >= 2 && bestPred) {
      const predAtObs = obsIdx.map(t => finalPred[t]);
      const kge = computeKGE(obsVals, predAtObs);
      const meanObs = obsVals.reduce((a, b) => a + b, 0) / obsVals.length;
      let ssTot = 0, ssRes = 0;
      for (let i = 0; i < obsVals.length; i++) {
        ssTot += (obsVals[i] - meanObs) ** 2;
        ssRes += (obsVals[i] - predAtObs[i]) ** 2;
      }
      const r2 = ssTot === 0 ? 0 : 1 - ssRes / ssTot;
      const rmse = Math.sqrt(ssRes / obsVals.length);
      onProgress?.(`Well ${wi + 1}/${targetWellIds.length}: R²=${r2.toFixed(4)}, RMSE=${rmse.toFixed(2)}, KGE=${kge.toFixed(4)}`, pct);
    }
  }

  onProgress?.('Complete', 100);
  return results;
}

// ═══════════════════════════════════════════════════════════════════
// Exports for testing
// ═══════════════════════════════════════════════════════════════════

export {
  SeededRng,
  pearsonR,
  computeKGE,
  computeNRMSE,
  ridgeRegression,
  pchipFill,
  archiRegression,
  mcSoftImpute,
  runLnnCfc,
  optimizeLnnParams,
  softImpute,
  truncatedSvd,
};
