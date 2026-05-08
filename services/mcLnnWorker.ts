/**
 * Web Worker for MC-LNN imputation.
 * Runs the heavy computation off the main thread.
 */

import { runMcLnnImputation, type McRunConfig } from './mcLnnPureBrowser';

interface WorkerInput {
  type: 'run';
  wellSeries: [string, (number | null)[]][];  // serialized Map
  auxData: number[][] | null;
  targetWellIds: string[];
  config?: McRunConfig;
}

interface WorkerOutput {
  type: 'progress' | 'result' | 'error';
  message?: string;
  pct?: number;
  results?: [string, number[]][];  // serialized Map
  error?: string;
}

self.onmessage = async (e: MessageEvent<WorkerInput>) => {
  const { wellSeries, auxData, targetWellIds, config } = e.data;

  try {
    const seriesMap = new Map(wellSeries);

    const results = await runMcLnnImputation(
      seriesMap,
      auxData,
      targetWellIds,
      config,
      (msg: string, pct: number) => {
        (self as any).postMessage({ type: 'progress', message: msg, pct } as WorkerOutput);
      },
    );

    const serialized: [string, number[]][] = Array.from(results.entries());
    (self as any).postMessage({ type: 'result', results: serialized } as WorkerOutput);
  } catch (err) {
    (self as any).postMessage({
      type: 'error',
      error: err instanceof Error ? err.message : String(err),
    } as WorkerOutput);
  }
};
