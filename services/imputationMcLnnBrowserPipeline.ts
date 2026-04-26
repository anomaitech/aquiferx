import {
  Aquifer, Region, Well, Measurement,
  ImputationParams, ImputationDataRow, ImputationModelResult, ImputationWellMetrics,
} from '../types';
import { fetchGldasFeatures } from './gldasFetch';
import { slugify } from '../utils/strings';

interface BrowserMcLnnInputPoint {
  date: string;
  time: number;
  observed: number | null;
  auxiliaries: number[];
}

interface BrowserMcLnnInputWell {
  wellId: string;
  points: BrowserMcLnnInputPoint[];
}

interface PythonMcLnnResponse {
  wells: Array<{
    wellId: string;
    rows: Array<{
      date: string;
      raw: number | null;
      smallGap: number | null;
      final: number | null;
      fillStage: string;
    }>;
  }>;
  notes: string[];
}

interface PythonMcLnnStreamEvent {
  type: 'progress' | 'note' | 'well_metrics' | 'result' | 'error';
  label?: string;
  pct?: number;
  message?: string;
  wellId?: string;
  rawMissing?: number;
  remainingAfterSmall?: number;
  outerIterationsUsed?: number;
  supportPoints?: number;
  supportKGE?: number | null;
  supportRMSE?: number | null;
  payload?: PythonMcLnnResponse;
}

export interface BrowserMcLnnPipelineInput {
  title: string;
  startDate: string;
  endDate: string;
  gldasStartDate: string;
  gldasEndDate: string;
  minSamples: number;
  gapSize: number; // days
  padSize: number; // retained for result-schema parity
  hiddenUnits: number; // retained for result-schema parity
  lambda: number; // retained for result-schema parity
}

function yieldToUI(): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, 0));
}

function generateMonthlyDates(startDate: string, endDate: string): string[] {
  const dates: string[] = [];
  const start = new Date(startDate);
  const end = new Date(endDate);
  let year = start.getUTCFullYear();
  let month = start.getUTCMonth();
  while (true) {
    const d = new Date(Date.UTC(year, month, 1));
    if (d > end) break;
    dates.push(d.toISOString().slice(0, 10));
    month++;
    if (month > 11) { month = 0; year++; }
  }
  return dates;
}

function mean(values: number[]): number | null {
  if (!values.length) return null;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function clampDateRange(
  startDate: string,
  endDate: string,
  gldasStartDate: string,
  gldasEndDate: string,
): { min: string; max: string } | null {
  const min = gldasStartDate > startDate ? gldasStartDate : startDate;
  const max = gldasEndDate < endDate ? gldasEndDate : endDate;
  return min <= max ? { min, max } : null;
}

async function readPythonMcLnnStream(
  response: Response,
  onLog: (msg: string) => void,
  onProgress: (step: string, pct: number) => void,
): Promise<PythonMcLnnResponse> {
  const reader = response.body?.getReader();
  if (!reader) throw new Error('Python MC+LNN backend returned no stream body.');
  const decoder = new TextDecoder();
  let buffer = '';
  let finalPayload: PythonMcLnnResponse | null = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newlineIdx = buffer.indexOf('\n');
    while (newlineIdx >= 0) {
      const line = buffer.slice(0, newlineIdx).trim();
      buffer = buffer.slice(newlineIdx + 1);
      if (line) {
        const event = JSON.parse(line) as PythonMcLnnStreamEvent;
        if (event.type === 'progress' && typeof event.label === 'string' && typeof event.pct === 'number') {
          onProgress(event.label, event.pct);
        } else if (event.type === 'note' && typeof event.message === 'string') {
          onLog(event.message);
        } else if (event.type === 'well_metrics' && event.wellId) {
          onLog(
            `[${event.wellId}] metrics | rawMissing=${event.rawMissing ?? 'NA'} | remainingAfterSmall=${event.remainingAfterSmall ?? 'NA'}`
            + ` | outer=${event.outerIterationsUsed ?? 'NA'} | support=${event.supportPoints ?? 'NA'}`
            + ` | supportKGE=${event.supportKGE == null ? 'NA' : event.supportKGE.toFixed(3)}`
            + ` | supportRMSE=${event.supportRMSE == null ? 'NA' : event.supportRMSE.toFixed(3)}`
          );
        } else if (event.type === 'error') {
          throw new Error(event.message || 'Python MC+LNN backend failed.');
        } else if (event.type === 'result' && event.payload) {
          finalPayload = event.payload;
        }
      }
      newlineIdx = buffer.indexOf('\n');
    }
  }
  if (!finalPayload) throw new Error('Python MC+LNN backend finished without a result payload.');
  return finalPayload;
}

export async function runBrowserMcLnnImputationPipeline(
  input: BrowserMcLnnPipelineInput,
  aquifer: Aquifer,
  region: Region,
  wells: Well[],
  measurements: Measurement[],
  onLog: (msg: string) => void,
  onProgress: (step: string, pct: number) => void,
): Promise<ImputationModelResult> {
  const { title, startDate, endDate, gldasStartDate, gldasEndDate, minSamples, gapSize, padSize, hiddenUnits, lambda } = input;
  const effectiveRange = clampDateRange(startDate, endDate, gldasStartDate, gldasEndDate);
  if (!effectiveRange) throw new Error('Selected date range does not overlap the GLDAS feature range.');

  onProgress('Fetching GLDAS data...', 0);
  onLog(`Fetching GLDAS soil moisture data for aquifer "${aquifer.name}"...`);
  await yieldToUI();

  const gldas = await fetchGldasFeatures(aquifer.id, aquifer.geojson, gldasStartDate, gldasEndDate);
  onLog(`GLDAS data loaded: ${gldas.dates.length} monthly records, range ${gldas.dates[0]} to ${gldas.dates[gldas.dates.length - 1]}`);
  onProgress('GLDAS data loaded', 8);

  const wteMeasurements = measurements.filter(m => m.dataType === 'wte');
  const monthlyObsByWell = new Map<string, Map<string, number[]>>();
  for (const m of wteMeasurements) {
    const month = m.date.slice(0, 7) + '-01';
    if (month < effectiveRange.min || month > effectiveRange.max) continue;
    if (!monthlyObsByWell.has(m.wellId)) monthlyObsByWell.set(m.wellId, new Map());
    const perMonth = monthlyObsByWell.get(m.wellId)!;
    if (!perMonth.has(month)) perMonth.set(month, []);
    perMonth.get(month)!.push(m.value);
  }

  const qualified = wells.filter(well => {
    const monthMap = monthlyObsByWell.get(well.id);
    if (!monthMap) return false;
    let count = 0;
    for (const values of monthMap.values()) {
      if (values.length && mean(values) != null) count++;
    }
    return count >= minSamples;
  });

  onLog(`${qualified.length} wells qualified (>= ${minSamples} monthly samples in ${effectiveRange.min} to ${effectiveRange.max}), ${wells.length - qualified.length} omitted`);
  if (!qualified.length) throw new Error(`No wells have >= ${minSamples} monthly samples. Adjust min samples or date range.`);

  onProgress('Preparing MC+LNN inputs...', 15);
  await yieldToUI();

  const monthDates = generateMonthlyDates(startDate, endDate);
  const gldasIdx = new Map<string, number>(gldas.dates.map((d, i) => [d, i]));
  const usableMonths = monthDates.filter(d => gldasIdx.has(d));

  const inputWells: BrowserMcLnnInputWell[] = qualified.map(well => {
    const monthMap = monthlyObsByWell.get(well.id) ?? new Map<string, number[]>();
    return {
      wellId: well.id,
      points: usableMonths.map((date, monthIdx) => {
        const vals = monthMap.get(date) ?? [];
        const observed = mean(vals);
        const gi = gldasIdx.get(date)!;
        return {
          date,
          time: monthIdx,
          observed,
          auxiliaries: [
            gldas.soilw[gi],
            gldas.soilw_yr01[gi],
            gldas.soilw_yr03[gi],
            gldas.soilw_yr05[gi],
            gldas.soilw_yr10[gi],
          ],
        };
      }),
    };
  });

  onProgress('Running validated Python MC + LNN...', 25);
  await yieldToUI();

  const mcLnnResp = await fetch('/api/impute-mc-lnn', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      wells: inputWells,
      config: {
        seed: 42,
        outerIterations: 2,
        feedbackPrevWeight: 0.35,
        supportFrac: 0.12,
        minSupport: 6,
        maxSupport: 24,
      },
    }),
  });
  if (!mcLnnResp.ok) {
    throw new Error(await mcLnnResp.text());
  }
  const mcLnn = await readPythonMcLnnStream(mcLnnResp, onLog, onProgress);

  onProgress('Assembling output...', 85);
  await yieldToUI();

  const outputStartTs = new Date(startDate).getTime();
  const outputEndTs = new Date(endDate).getTime();

  const data: ImputationDataRow[] = [];
  const wellMetrics: Record<string, ImputationWellMetrics> = {};

  for (const well of mcLnn.wells) {
    for (const row of well.rows) {
      const ts = new Date(row.date).getTime();
      if (ts < outputStartTs || ts > outputEndTs) continue;
      const pchipLike = row.raw != null ? row.raw : row.smallGap;
      const modelLike = row.raw == null && row.smallGap == null ? row.final : null;
      data.push({
        well_id: well.wellId,
        date: row.date,
        pchip: pchipLike,
        model: modelLike,
        combined: row.final ?? pchipLike ?? 0,
      });
    }
    wellMetrics[well.wellId] = { r2: Number.NaN, rmse: Number.NaN };
  }

  const code = slugify(title);
  const aquiferSlug = slugify(aquifer.name);
  const filePath = `${region.id}/${aquiferSlug}/model_wte_${code}.json`;

  const params: ImputationParams = {
    startDate,
    endDate,
    gldasStartDate: gldas.dates[0],
    gldasEndDate: gldas.dates[gldas.dates.length - 1],
    minSamples,
    gapSize,
    padSize,
    hiddenUnits,
    lambda,
  };

  const result: ImputationModelResult = {
    title,
    code,
    aquiferId: aquifer.id,
    aquiferName: aquifer.name,
    regionId: region.id,
    dataType: 'wte',
    filePath,
    createdAt: new Date().toISOString(),
    params,
    wellMetrics,
    data,
    log: [],
  };

  await fetch('/api/save-data', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      files: [{
        path: filePath,
        content: JSON.stringify(result),
      }],
    }),
  });

  onProgress('Complete!', 100);
  onLog('Validated Python MC+LNN imputation complete! Model saved.');
  return result;
}
