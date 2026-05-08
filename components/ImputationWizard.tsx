import React, { useState, useMemo, useRef, useEffect } from 'react';
import { X, ChevronLeft, ChevronRight, Play, Loader2, CheckCircle2, AlertTriangle } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, ReferenceArea, Cell,
} from 'recharts';
import { Aquifer, Region, Well, Measurement, ImputationModelResult } from '../types';
import { runImputationPipeline, ImputationPipelineInput } from '../services/imputationPipeline';
import { runBrowserMcLnnImputationPipeline } from '../services/imputationMcLnnBrowserPipeline';
import { runMcLnnPureImputationPipeline } from '../services/imputationMcLnnPurePipeline';
import { slugify } from '../utils/strings';
import PchipPreviewCanvas from './PchipPreviewCanvas';

interface ImputationWizardProps {
  aquifer: Aquifer;
  region: Region;
  wells: Well[];
  measurements: Measurement[];
  existingModelCodes: string[];
  gldasDateRange: { min: string; max: string } | null;
  minSamples: number;
  onClose: () => void;
  onComplete: (result: ImputationModelResult) => void;
}

type Step = 1 | 2 | 'running' | 'complete';
type ImputationMethod = 'original-elm' | 'browser-mc-lnn' | 'browser-mc-lnn-pure';

const STEP_LABELS = ['Wells & Options', 'Title & Run'];

const ImputationWizard: React.FC<ImputationWizardProps> = ({
  aquifer, region, wells, measurements, existingModelCodes,
  gldasDateRange, minSamples, onClose, onComplete,
}) => {
  const [step, setStep] = useState<Step>(1);
  const [progressText, setProgressText] = useState('');
  const [progressPct, setProgressPct] = useState(0);
  const [liveR2, setLiveR2] = useState<string | null>(null);
  const [result, setResult] = useState<ImputationModelResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const cancelledRef = useRef(false);

  // --- Log state ---
  const [logMessages, setLogMessages] = useState<string[]>([]);
  const logContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll log
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logMessages]);

  // --- Step 1 options ---
  const [gapSizeDays, setGapSizeDays] = useState(730);
  const [padSizeDays, setPadSizeDays] = useState(180);
  const [method, setMethod] = useState<ImputationMethod>('original-elm');

  // --- Step 2 ---
  const [title, setTitle] = useState('');

  // Build well key set for fast lookup
  const wellKeySet = useMemo(() => new Set(wells.map(w => `${w.regionId}:${w.aquiferId}:${w.id}`)), [wells]);

  // WTE measurements for this aquifer
  const wteMeasurements = useMemo(() =>
    measurements.filter(m => m.dataType === 'wte' && wellKeySet.has(`${m.regionId}:${m.aquiferId}:${m.wellId}`)),
  [measurements, wellKeySet]);

  // Data density analysis (same as SpatialAnalysisDialog but with 3%/5 threshold)
  const { densityData, defaultStartDate, defaultEndDate, gldasFirstLabel, gldasLastLabel } = useMemo(() => {
    const byWellDate = new Map<string, Set<string>>();
    for (const m of wteMeasurements) {
      if (!byWellDate.has(m.wellId)) byWellDate.set(m.wellId, new Set());
      byWellDate.get(m.wellId)!.add(m.date);
    }

    let allMin = Infinity, allMax = -Infinity;
    for (const m of wteMeasurements) {
      const t = new Date(m.date).getTime();
      if (!isNaN(t)) {
        if (t < allMin) allMin = t;
        if (t > allMax) allMax = t;
      }
    }

    if (allMin === Infinity) {
      return { densityData: [], defaultStartDate: '', defaultEndDate: '' };
    }

    const minYear = new Date(allMin).getFullYear();
    const maxYear = new Date(allMax).getFullYear();
    const minDateStr = `${minYear}-01-01`;
    const maxDateStr = `${maxYear + 1}-01-01`;

    const bins: { label: string; start: Date; end: Date }[] = [];
    for (let y = minYear; y <= maxYear; y++) {
      bins.push({ label: `${y} H1`, start: new Date(y, 0, 1), end: new Date(y, 6, 1) });
      bins.push({ label: `${y} H2`, start: new Date(y, 6, 1), end: new Date(y + 1, 0, 1) });
    }

    const densityData = bins.map(bin => {
      const wellsInBin = new Set<string>();
      for (const [wellId, dates] of byWellDate) {
        for (const d of dates) {
          const t = new Date(d);
          if (t >= bin.start && t < bin.end) {
            wellsInBin.add(wellId);
            break;
          }
        }
      }
      return { label: bin.label, count: wellsInBin.size, startTs: bin.start.getTime() };
    }).filter(b => b.startTs >= allMin - 365 * 86400000 && b.startTs <= allMax + 365 * 86400000);

    // Default dates: full GLDAS feature range (matches Python notebook behavior)
    let defStart = gldasDateRange?.min ?? minDateStr;
    let defEnd = gldasDateRange?.max ?? maxDateStr;

    // Compute GLDAS range bin labels for histogram highlight
    let gldasFirstLabel: string | undefined;
    let gldasLastLabel: string | undefined;
    if (gldasDateRange) {
      const gMin = new Date(gldasDateRange.min).getTime();
      const gMax = new Date(gldasDateRange.max).getTime();
      for (const b of densityData) {
        if (b.startTs >= gMin && !gldasFirstLabel) gldasFirstLabel = b.label;
        if (b.startTs <= gMax) gldasLastLabel = b.label;
      }
    }

    return { densityData, defaultStartDate: defStart, defaultEndDate: defEnd, gldasFirstLabel, gldasLastLabel };
  }, [wteMeasurements, gldasDateRange]);

  const [startDate, setStartDate] = useState(defaultStartDate);
  const [endDate, setEndDate] = useState(defaultEndDate);

  const effectiveDateRange = useMemo(() => {
    if (!startDate || !endDate) return null;
    const min = gldasDateRange?.min && gldasDateRange.min > startDate ? gldasDateRange.min : startDate;
    const max = gldasDateRange?.max && gldasDateRange.max < endDate ? gldasDateRange.max : endDate;
    if (min > max) return null;
    return { min, max };
  }, [startDate, endDate, gldasDateRange]);

  // Sync defaults when they compute
  useEffect(() => {
    if (defaultStartDate && !startDate) setStartDate(defaultStartDate);
    if (defaultEndDate && !endDate) setEndDate(defaultEndDate);
  }, [defaultStartDate, defaultEndDate]);

  const code = slugify(title);
  const hasConflict = existingModelCodes.includes(code);

  // Usable well count
  const usableWellCount = useMemo(() => {
    if (!effectiveDateRange) return 0;
    const counts = new Map<string, Set<string>>();
    for (const m of wteMeasurements) {
      if (m.date < effectiveDateRange.min || m.date > effectiveDateRange.max) continue;
      if (!counts.has(m.wellId)) counts.set(m.wellId, new Set());
      counts.get(m.wellId)!.add(m.date.slice(0, 7));
    }
    let count = 0;
    for (const months of counts.values()) if (months.size >= 2) count++;
    return count;
  }, [wteMeasurements, effectiveDateRange]);

  // Well qualification based on minSamples
  const { qualifiedWellCount, omittedWellCount } = useMemo(() => {
    if (!effectiveDateRange) return { qualifiedWellCount: 0, omittedWellCount: 0 };
    const byWell = new Map<string, Set<string>>();
    for (const m of wteMeasurements) {
      if (m.date < effectiveDateRange.min || m.date > effectiveDateRange.max) continue;
      if (!byWell.has(m.wellId)) byWell.set(m.wellId, new Set());
      byWell.get(m.wellId)!.add(m.date.slice(0, 7));
    }
    let qualified = 0;
    for (const well of wells) {
      const months = byWell.get(well.id);
      if ((months?.size ?? 0) >= minSamples) qualified++;
    }
    return { qualifiedWellCount: qualified, omittedWellCount: wells.length - qualified };
  }, [wteMeasurements, wells, minSamples, effectiveDateRange]);

  // GLDAS range info
  const gldasInfo = useMemo(() => {
    if (!gldasDateRange) return 'GLDAS date range not available. The wizard will attempt to fetch GLDAS data during processing.';
    return `Training uses full GLDAS range: ${gldasDateRange.min} to ${gldasDateRange.max}. Output dates clip the results.`;
  }, [gldasDateRange]);

  // Validation
  const step1Valid = startDate && endDate && startDate < endDate && wells.length > 0 && qualifiedWellCount > 0;
  const step2Valid = title.trim().length > 0 && !hasConflict;

  const handleRun = async () => {
    cancelledRef.current = false;
    setStep('running');
    setErrorMessage(null);
    setLogMessages([]);
    setLiveR2(null);

    const input: ImputationPipelineInput = {
      title,
      startDate,
      endDate,
      gldasStartDate: gldasDateRange?.min ?? startDate,
      gldasEndDate: gldasDateRange?.max ?? endDate,
      minSamples,
      gapSize: gapSizeDays,
      padSize: padSizeDays,
      hiddenUnits: 500,
      lambda: 100,
    };

    try {
      const logAccumulator: string[] = [];
      const scopedMeasurements = measurements.filter(m => wellKeySet.has(`${m.regionId}:${m.aquiferId}:${m.wellId}`));
      const handleLog = (msg: string) => {
        if (!cancelledRef.current) {
          logAccumulator.push(msg);
          setLogMessages([...logAccumulator]);
          const match = msg.match(/supportR²=([0-9.+-]+)/i)
            || msg.match(/supportR2=([0-9.+-]+)/i)
            || msg.match(/small-gap R²=([0-9.+-]+)/i)
            || msg.match(/small-gap R2=([0-9.+-]+)/i);
          if (match) setLiveR2(match[1]);
        }
      };
      const handleProgress = (stepText: string, pct: number) => {
        if (!cancelledRef.current) {
          setProgressText(stepText);
          setProgressPct(pct);
        }
      };
      const result = method === 'browser-mc-lnn'
        ? await runBrowserMcLnnImputationPipeline(input, aquifer, region, wells, scopedMeasurements, handleLog, handleProgress)
        : method === 'browser-mc-lnn-pure'
        ? await runMcLnnPureImputationPipeline(input, aquifer, region, wells, scopedMeasurements, handleLog, handleProgress)
        : await runImputationPipeline(input, aquifer, region, wells, scopedMeasurements, handleLog, handleProgress);

      if (!cancelledRef.current) {
        result.log = logAccumulator;
        setResult(result);
        setStep('complete');
      }
    } catch (err) {
      console.error('Imputation failed:', err);
      setErrorMessage(err instanceof Error ? err.message : String(err));
      setStep(2);
    }
  };

  const handleCancel = () => {
    if (step === 'running') {
      cancelledRef.current = true;
    }
    setStep(1);
    setProgressPct(0);
    setProgressText('');
    setLiveR2(null);
    setErrorMessage(null);
  };

  const startTs = startDate ? new Date(startDate).getTime() : undefined;
  const endTs = endDate ? new Date(endDate).getTime() : undefined;

  const stepNumber = typeof step === 'number' ? step : null;

  const inputCls = "w-full px-2 py-1.5 border border-slate-300 rounded-md text-sm focus:ring-2 focus:ring-amber-500 focus:border-amber-500";
  const labelCls = "block text-xs font-medium text-slate-600 mb-1";

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/40">
      <div className="bg-white rounded-xl shadow-2xl w-[900px] max-h-[90vh] flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200">
          <div>
            <h2 className="text-lg font-bold text-slate-800">Impute WTE Gaps</h2>
            <p className="text-sm text-slate-500">{aquifer.name} &mdash; {region.name}</p>
          </div>
          <div className="flex items-center gap-4">
            {stepNumber && (
              <div className="flex items-center gap-2 text-xs text-slate-500">
                {STEP_LABELS.map((label, i) => {
                  const sn = i + 1;
                  const isActive = sn === stepNumber;
                  const isDone = sn < stepNumber;
                  return (
                    <React.Fragment key={i}>
                      {i > 0 && <div className="w-4 h-px bg-slate-300" />}
                      <div className={`flex items-center gap-1 ${isActive ? 'text-amber-600 font-semibold' : isDone ? 'text-amber-500' : 'text-slate-400'}`}>
                        <div className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold ${
                          isActive ? 'bg-amber-600 text-white' : isDone ? 'bg-amber-100 text-amber-600' : 'bg-slate-100 text-slate-400'
                        }`}>{sn}</div>
                        <span className="hidden sm:inline">{label}</span>
                      </div>
                    </React.Fragment>
                  );
                })}
              </div>
            )}
            <button onClick={onClose} className="p-1 hover:bg-slate-100 rounded-lg transition-colors">
              <X size={20} className="text-slate-400" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          <div className="mb-4 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
            <div className="font-semibold">Bundled standalone MC+LNN backend</div>
            <div className="mt-1">
              This wizard runs the browser ELM pipeline. The repository also includes a standalone Python backend at
              <code className="mx-1 rounded bg-amber-100 px-1 py-0.5">python/mc_lnn_imputer</code>
              for the validated GSLB workflow:
              <span className="font-medium"> small-gap LNN-CFC auxiliary</span>
              {' -> '}
              <span className="font-medium">large-gap iterative SoftImpute MC + LNN</span>.
            </div>
          </div>

          {/* ===== STEP 1: Wells & Options ===== */}
          {step === 1 && (
            <div className="space-y-5">
              {/* PCHIP Preview */}
              {wteMeasurements.length > 0 && (
                <div>
                  <h3 className="text-sm font-semibold text-slate-700 mb-1">
                    WTE Time Series Preview ({usableWellCount} wells with 2+ observations)
                  </h3>
                  <div className="h-[180px] bg-slate-50 rounded-lg border border-slate-200">
                    <PchipPreviewCanvas
                      wells={wells}
                      wteMeasurements={wteMeasurements}
                      startTs={startTs}
                      endTs={endTs}
                      gldasRange={gldasDateRange ? { min: new Date(gldasDateRange.min).getTime(), max: new Date(gldasDateRange.max).getTime() } : undefined}
                    />
                  </div>
                </div>
              )}

              {/* Data Density Histogram */}
              {densityData.length > 0 && (
                <div>
                  <h3 className="text-sm font-semibold text-slate-700 mb-1">Data Density (wells per 6-month bin)</h3>
                  <div className="h-[130px] bg-slate-50 rounded-lg border border-slate-200 p-1">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={densityData} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                        <XAxis dataKey="label" stroke="#94a3b8" fontSize={9} interval="preserveStartEnd" />
                        <YAxis stroke="#94a3b8" fontSize={10} />
                        {gldasFirstLabel && gldasLastLabel && (
                          <ReferenceArea x1={gldasFirstLabel} x2={gldasLastLabel} {...{ fill: '#3b82f6', fillOpacity: 0.08 } as any} />
                        )}
                        <ReferenceLine y={5} stroke="#f59e0b" strokeDasharray="4 3" strokeWidth={1.5} />
                        <Tooltip
                          content={({ payload }) => {
                            if (!payload || payload.length === 0) return null;
                            const d = payload[0]?.payload;
                            return (
                              <div className="bg-white rounded shadow-md px-2 py-1 text-[10px] border border-slate-200">
                                <div className="text-slate-700 font-medium">{d?.label}</div>
                                <div className="text-slate-500">{d?.count} wells</div>
                              </div>
                            );
                          }}
                        />
                        <Bar dataKey="count" radius={[2, 2, 0, 0]}>
                          {densityData.map((entry, i) => (
                            <Cell
                              key={i}
                              fill={entry.count >= 5 ? '#f59e0b' : '#94a3b8'}
                              fillOpacity={entry.count >= 5 ? 0.8 : 0.4}
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {/* Date Controls */}
              <div className="grid grid-cols-2 gap-4">
                <div className="col-span-2">
                  <label className={labelCls}>Imputation Method</label>
                  <select className={inputCls} value={method} onChange={e => setMethod(e.target.value as ImputationMethod)}>
                    <option value="original-elm">Original PCHIP + ELM</option>
                    <option value="browser-mc-lnn">MC + LNN (Validated Python)</option>
                    <option value="browser-mc-lnn-pure">MC + LNN (Pure Browser)</option>
                  </select>
                  <p className="text-[10px] text-slate-400 mt-0.5">
                    {method === 'browser-mc-lnn'
                      ? 'Runs the validated local Python backend: PCHIP small gaps, then iterative SoftImpute MC -> LNN for large gaps.'
                      : method === 'browser-mc-lnn-pure'
                      ? 'Pure browser: PCHIP small gaps, then Matrix Completion + Liquid Neural Network for large gaps.'
                      : 'Original AquiferX pipeline: PCHIP small gaps, then ELM model for large gaps.'}
                  </p>
                </div>
                <div>
                  <label className={labelCls}>Output Start Date</label>
                  <div className="flex items-center gap-1">
                    <input type="date" value={startDate}
                      min={gldasDateRange?.min}
                      max={endDate || gldasDateRange?.max}
                      onChange={e => {
                        let v = e.target.value;
                        if (gldasDateRange && v < gldasDateRange.min) v = gldasDateRange.min;
                        setStartDate(v);
                      }}
                      className={inputCls + ' flex-1'} />
                    {!(gldasDateRange && startDate <= gldasDateRange.min) && (
                      <button type="button" className="p-1 text-slate-400 hover:text-slate-700" onClick={() => {
                        const y = parseInt(startDate); if (isNaN(y)) return;
                        let next = `${y - 1}-01-01`;
                        if (gldasDateRange && next < gldasDateRange.min) next = gldasDateRange.min;
                        setStartDate(next);
                      }}>
                        <svg width="14" height="14" viewBox="0 0 14 14"><path d="M9 2L4 7l5 5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
                      </button>
                    )}
                    {!(gldasDateRange && startDate >= gldasDateRange.max) && (
                      <button type="button" className="p-1 text-slate-400 hover:text-slate-700" onClick={() => {
                        const y = parseInt(startDate); if (isNaN(y)) return;
                        let next = `${y + 1}-01-01`;
                        if (gldasDateRange && next > gldasDateRange.max) next = gldasDateRange.max;
                        setStartDate(next);
                      }}>
                        <svg width="14" height="14" viewBox="0 0 14 14"><path d="M5 2l5 5-5 5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
                      </button>
                    )}
                  </div>
                </div>
                <div>
                  <label className={labelCls}>Output End Date</label>
                  <div className="flex items-center gap-1">
                    <input type="date" value={endDate}
                      min={startDate || gldasDateRange?.min}
                      max={gldasDateRange?.max}
                      onChange={e => {
                        let v = e.target.value;
                        if (gldasDateRange && v > gldasDateRange.max) v = gldasDateRange.max;
                        setEndDate(v);
                      }}
                      className={inputCls + ' flex-1'} />
                    {!(gldasDateRange && endDate <= gldasDateRange.min) && (
                      <button type="button" className="p-1 text-slate-400 hover:text-slate-700" onClick={() => {
                        const y = parseInt(endDate); if (isNaN(y)) return;
                        let next = `${y - 1}-01-01`;
                        if (gldasDateRange && next < gldasDateRange.min) next = gldasDateRange.min;
                        setEndDate(next);
                      }}>
                        <svg width="14" height="14" viewBox="0 0 14 14"><path d="M9 2L4 7l5 5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
                      </button>
                    )}
                    {!(gldasDateRange && endDate >= gldasDateRange.max) && (
                      <button type="button" className="p-1 text-slate-400 hover:text-slate-700" onClick={() => {
                        const y = parseInt(endDate); if (isNaN(y)) return;
                        let next = `${y + 1}-01-01`;
                        if (gldasDateRange && next > gldasDateRange.max) next = gldasDateRange.max;
                        setEndDate(next);
                      }}>
                        <svg width="14" height="14" viewBox="0 0 14 14"><path d="M5 2l5 5-5 5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
                      </button>
                    )}
                  </div>
                </div>

                <>
                  <div>
                    <label className={labelCls}>Min Samples / Well</label>
                    <div className={inputCls + ' bg-slate-50 text-slate-700 flex items-center justify-between'}>
                      <span>{minSamples}</span>
                      <span className="text-[10px] text-slate-400">from map Min obs</span>
                    </div>
                  </div>
                  <div>
                    <label className={labelCls}>Gap Size (days)</label>
                    <input type="number" value={gapSizeDays} min={1} max={7300} step={1}
                      onChange={e => setGapSizeDays(Math.max(1, parseInt(e.target.value) || 730))} className={inputCls} />
                    <p className="text-[10px] text-slate-400 mt-0.5">Gaps larger than this use {method === 'browser-mc-lnn-pure' ? 'MC+LNN' : method === 'browser-mc-lnn' ? 'MC+LNN' : 'ELM'} model</p>
                  </div>
                  <div>
                    <label className={labelCls}>Pad Size (days)</label>
                    <input type="number" value={padSizeDays} min={0} max={1800} step={1}
                      onChange={e => setPadSizeDays(Math.max(0, parseInt(e.target.value) || 180))} className={inputCls} />
                    <p className="text-[10px] text-slate-400 mt-0.5">PCHIP padding at gap boundaries</p>
                  </div>
                </>

                <div className="col-span-2">
                  <div className="flex items-center gap-3 text-xs">
                    <span className="text-amber-600 font-medium">{qualifiedWellCount} wells qualify</span>
                    {omittedWellCount > 0 && (
                      <span className="text-slate-400">{omittedWellCount} omitted (fewer than {minSamples} samples)</span>
                    )}
                  </div>
                </div>
              </div>

              {/* GLDAS Info */}
              {gldasInfo && (
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 flex items-start gap-2">
                  <AlertTriangle size={14} className="text-blue-500 mt-0.5 shrink-0" />
                  <p className="text-xs text-blue-700">{gldasInfo}</p>
                </div>
              )}
            </div>
          )}

          {/* ===== STEP 2: Title & Summary ===== */}
          {step === 2 && (
            <div className="space-y-5">
              {errorMessage && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-3 flex items-start gap-2">
                  <AlertTriangle size={16} className="text-red-500 mt-0.5 shrink-0" />
                  <div>
                    <p className="text-sm font-medium text-red-800">Imputation Failed</p>
                    <p className="text-xs text-red-600 mt-1">{errorMessage}</p>
                  </div>
                </div>
              )}

              <div>
                <label className={labelCls}>Title</label>
                <input type="text" value={title} onChange={e => {
                  const v = e.target.value.replace(/[^a-zA-Z0-9 _-]/g, '');
                  setTitle(v);
                }}
                  placeholder="e.g. ELM Imputation 2024"
                  className={inputCls} />
                {title && (
                  <div className="mt-1 flex items-center gap-2">
                    <span className="text-[10px] text-slate-400">{slugify(aquifer.name)}/model_wte_{code}.json</span>
                    {hasConflict && <span className="text-[10px] text-red-500 font-medium">Name already exists</span>}
                  </div>
                )}
              </div>

              {/* Options summary */}
              <div className="bg-slate-50 rounded-lg border border-slate-200 p-4">
                <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wide mb-3">Options Summary</h4>
                <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs text-slate-600">
                  <div><span className="text-slate-400">Dates:</span> {startDate} to {endDate}</div>
                  <div><span className="text-slate-400">Method:</span> {method === 'browser-mc-lnn' ? 'MC + LNN (Validated Python)' : method === 'browser-mc-lnn-pure' ? 'MC + LNN (Pure Browser)' : 'PCHIP + ELM'}</div>
                  <div><span className="text-slate-400">Wells:</span> {qualifiedWellCount} qualified</div>
                  {method === 'original-elm' ? (
                    <>
                      <div><span className="text-slate-400">Min Samples:</span> {minSamples}</div>
                      <div><span className="text-slate-400">Gap Size:</span> {gapSizeDays} days</div>
                      <div><span className="text-slate-400">Pad Size:</span> {padSizeDays} days</div>
                    </>
                  ) : (
                    <div><span className="text-slate-400">Defaults:</span> built-in MC+LNN browser settings</div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* ===== RUNNING ===== */}
          {step === 'running' && (
            <div className="space-y-4">
              <div className="flex flex-col items-center justify-center py-6 space-y-3">
                <Loader2 className="w-8 h-8 text-amber-500 animate-spin" />
                <p className="text-sm text-slate-600">{progressText}</p>
                <div className="w-80 bg-slate-200 rounded-full h-2.5">
                  <div className="bg-amber-500 h-2.5 rounded-full transition-all duration-300"
                    style={{ width: `${progressPct}%` }} />
                </div>
                <p className="text-xs text-slate-400">{Math.round(progressPct)}%</p>
                {liveR2 && (
                  <p className="text-xs text-slate-500">Latest R²: <span className="font-medium text-slate-700">{liveR2}</span></p>
                )}
              </div>

              {/* Log viewer */}
              <div className="bg-slate-900 rounded-lg border border-slate-700 overflow-hidden">
                <div className="px-3 py-1.5 bg-slate-800 text-[10px] text-slate-400 font-mono">Log</div>
                <div ref={logContainerRef} className="h-[200px] overflow-y-auto px-3 py-2 font-mono text-[11px] text-slate-300 space-y-0.5">
                  {logMessages.map((msg, i) => (
                    <div key={i} className={msg.startsWith('ERROR') ? 'text-red-400' : msg.includes('R²') ? 'text-emerald-400' : ''}>
                      {msg}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* ===== COMPLETE ===== */}
          {step === 'complete' && result && (
            <div className="space-y-4">
              <div className="flex flex-col items-center justify-center py-8 space-y-3">
                <CheckCircle2 className="w-12 h-12 text-amber-500" />
                <h3 className="text-lg font-semibold text-slate-800">Imputation Complete</h3>
                <div className="text-sm text-slate-600 text-center space-y-1">
                  <p>{Object.keys(result.wellMetrics).length} wells modeled &bull; {result.data.length} data rows</p>
                  <p>{result.params.startDate} to {result.params.endDate}</p>
                  <p>{result.aquiferName}</p>
                </div>
              </div>

              {/* Log viewer (collapsed) */}
              <details className="bg-slate-900 rounded-lg border border-slate-700 overflow-hidden">
                <summary className="px-3 py-1.5 bg-slate-800 text-[10px] text-slate-400 font-mono cursor-pointer hover:bg-slate-750">
                  Processing Log ({logMessages.length} entries)
                </summary>
                <div className="h-[150px] overflow-y-auto px-3 py-2 font-mono text-[11px] text-slate-300 space-y-0.5">
                  {logMessages.map((msg, i) => (
                    <div key={i} className={msg.startsWith('ERROR') ? 'text-red-400' : msg.includes('R²') ? 'text-emerald-400' : ''}>
                      {msg}
                    </div>
                  ))}
                </div>
              </details>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-3 border-t border-slate-200 bg-slate-50">
          <div>
            {(stepNumber || step === 'running') && (
              <button onClick={step === 'running' ? handleCancel : onClose}
                className="px-4 py-2 text-sm text-slate-600 hover:bg-slate-100 rounded-lg transition-colors">
                Cancel
              </button>
            )}
          </div>

          <div>
            {stepNumber && stepNumber > 1 && (
              <button onClick={() => setStep((stepNumber - 1) as Step)}
                className="flex items-center gap-1 px-4 py-2 text-sm text-slate-600 hover:bg-slate-100 rounded-lg transition-colors">
                <ChevronLeft size={16} />
                Back
              </button>
            )}
          </div>

          <div>
            {step === 1 && (
              <button onClick={() => setStep(2)} disabled={!step1Valid}
                className="flex items-center gap-1 px-4 py-2 bg-amber-600 text-white text-sm font-medium rounded-lg hover:bg-amber-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed">
                Next
                <ChevronRight size={16} />
              </button>
            )}
            {step === 2 && (
              <button onClick={handleRun} disabled={!step2Valid}
                className="flex items-center gap-2 px-4 py-2 bg-amber-600 text-white text-sm font-medium rounded-lg hover:bg-amber-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed">
                <Play size={16} />
                Run Imputation
              </button>
            )}
            {step === 'running' && <div />}
            {step === 'complete' && result && (
              <button onClick={() => onComplete(result)}
                className="flex items-center gap-2 px-4 py-2 bg-amber-600 text-white text-sm font-medium rounded-lg hover:bg-amber-700 transition-colors">
                View Results
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImputationWizard;
