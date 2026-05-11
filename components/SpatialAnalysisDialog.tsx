import React, { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import { X, ChevronLeft, ChevronRight, Play, Loader2, CheckCircle2, AlertTriangle } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Cell
} from 'recharts';
import {
  Aquifer, Region, Well, Measurement, DataType, RasterAnalysisResult,
  VariogramModel, KrigingRangeMode, IdwNodalFunction, IdwNeighborMode, SpatialMethod,
  ImputationModelMeta,
} from '../types';
import { runRasterAnalysis, RasterPipelineInput } from '../services/rasterAnalysis';
import { slugify } from '../utils/strings';
import PchipPreviewCanvas from './PchipPreviewCanvas';

interface SpatialAnalysisDialogProps {
  aquifer: Aquifer;
  region: Region;
  wells: Well[];
  measurements: Measurement[];
  dataType: DataType;
  existingCodes: string[];
  onClose: () => void;
  onComplete: (result: RasterAnalysisResult) => void;
  availableModels?: ImputationModelMeta[];
}

type Step = 1 | 2 | 3 | 'running' | 'complete';

const STEP_LABELS = ['Temporal', 'Spatial', 'Title & Run'];

function formatTemporalMethodLabel(
  method: 'pchip' | 'linear' | 'moving-average' | 'model' | 'model-direct' | 'model-mavg',
  maWindow: number,
  modelCode: string,
): string {
  switch (method) {
    case 'pchip': return 'pchip';
    case 'linear': return 'linear';
    case 'moving-average': return `ma${maWindow}`;
    case 'model': return modelCode ? `model_${modelCode}` : 'model';
    case 'model-direct': return modelCode ? `model_direct_${modelCode}` : 'model_direct';
    case 'model-mavg': return modelCode ? `model_ma${maWindow}_${modelCode}` : `model_ma${maWindow}`;
    default: return method;
  }
}

function formatSpatialMethodLabel(
  spatialMethod: SpatialMethod,
  variogramModel: VariogramModel,
  nodalFunction: IdwNodalFunction,
): string {
  if (spatialMethod === 'kriging') return `kriging_${variogramModel}`;
  if (spatialMethod === 'mc') return 'mc_lowrank';
  return nodalFunction === 'classic' ? 'idw' : nodalFunction === 'gradient' ? 'idw_gp' : 'idw_quad';
}

const SpatialAnalysisDialog: React.FC<SpatialAnalysisDialogProps> = ({
  aquifer, region, wells, measurements, dataType, existingCodes, onClose, onComplete, availableModels,
}) => {
  const [step, setStep] = useState<Step>(1);
  const [progressText, setProgressText] = useState('');
  const [progressPct, setProgressPct] = useState(0);
  const [result, setResult] = useState<RasterAnalysisResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const cancelledRef = useRef(false);

  // --- Temporal options (Step 1) ---
  const [resolution, setResolution] = useState(50);
  const [interval, setInterval] = useState<'3months' | '6months' | '1year'>('1year');
  const [minObs, setMinObs] = useState(5);
  const [minSpanYears, setMinSpanYears] = useState(5);
  const [smoothingMethod, setSmoothingMethod] = useState<'pchip' | 'linear' | 'moving-average' | 'model' | 'model-direct' | 'model-mavg'>('pchip');
  const [smoothingMonths, setSmoothingMonths] = useState(12);
  const [selectedModelCode, setSelectedModelCode] = useState<string>('');
  const [selectedModelFilePath, setSelectedModelFilePath] = useState<string>('');
  const [dataSource, setDataSource] = useState<'raw' | 'model'>('raw');

  // --- Spatial options (Step 2) ---
  const [spatialMethod, setSpatialMethod] = useState<SpatialMethod>('kriging');
  const [variogramModel, setVariogramModel] = useState<VariogramModel>('gaussian');
  const [nuggetEnabled, setNuggetEnabled] = useState(true);
  const [rangeMode, setRangeMode] = useState<KrigingRangeMode>('auto');
  const [rangeValue, setRangeValue] = useState<number>(33);
  const [idwExponent, setIdwExponent] = useState(2);
  const [nodalFunction, setNodalFunction] = useState<IdwNodalFunction>('classic');
  const [neighborMode, setNeighborMode] = useState<IdwNeighborMode>('all');
  const [neighborCount, setNeighborCount] = useState(12);
  const [eofModes, setEofModes] = useState(20);
  const [eofNeighbors, setEofNeighbors] = useState(30);

  // --- General options (Step 2) ---
  const [truncateLow, setTruncateLow] = useState(false);
  const [truncateLowValue, setTruncateLowValue] = useState(0);
  const [truncateHigh, setTruncateHigh] = useState(false);
  const [truncateHighValue, setTruncateHighValue] = useState(0);
  const [logInterpolation, setLogInterpolation] = useState(false);

  // --- Title (Step 3) ---
  const [title, setTitle] = useState('');

  // Build well key set for fast lookup (regionId:aquiferId:wellId for global uniqueness)
  const wellKeySet = useMemo(() => new Set(wells.map(w => `${w.regionId}:${w.aquiferId}:${w.id}`)), [wells]);

  // Compute measurements for the target data type in this aquifer
  const wteMeasurements = useMemo(() =>
    measurements.filter(m => m.dataType === dataType.code && wellKeySet.has(`${m.regionId}:${m.aquiferId}:${m.wellId}`)),
  [measurements, wellKeySet, dataType.code]);

  // Check if any values are non-positive (disables log interpolation)
  const hasNonPositiveValues = useMemo(() => {
    for (const m of wteMeasurements) {
      if (m.value <= 0) return true;
    }
    return false;
  }, [wteMeasurements]);

  // Initialize max observed for truncation default
  useEffect(() => {
    if (wteMeasurements.length > 0) {
      let max = -Infinity;
      for (const m of wteMeasurements) if (m.value > max) max = m.value;
      setTruncateHighValue(Math.ceil(max));
    }
  }, [wteMeasurements]);

  // Data density analysis: 6-month bins
  const { densityData, defaultStartDate, defaultEndDate } = useMemo(() => {
    const byWellDate = new Map<string, Set<string>>();
    for (const m of wteMeasurements) {
      if (!byWellDate.has(m.wellId)) byWellDate.set(m.wellId, new Set());
      byWellDate.get(m.wellId)!.add(m.date);
    }

    // Find overall date range
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

    // Build bins
    const bins: { label: string; start: Date; end: Date }[] = [];
    for (let y = minYear; y <= maxYear; y++) {
      bins.push({ label: `${y} H1`, start: new Date(y, 0, 1), end: new Date(y, 6, 1) });
      bins.push({ label: `${y} H2`, start: new Date(y, 6, 1), end: new Date(y + 1, 0, 1) });
    }

    // Count wells per bin
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

    // Compute per-well time series span [minDate, maxDate]
    const wellSpans: { min: number; max: number }[] = [];
    for (const [, dates] of byWellDate) {
      let wMin = Infinity, wMax = -Infinity;
      for (const d of dates) {
        const t = new Date(d).getTime();
        if (!isNaN(t)) { if (t < wMin) wMin = t; if (t > wMax) wMax = t; }
      }
      if (wMin < Infinity) wellSpans.push({ min: wMin, max: wMax });
    }

    // Find default start/end: first/last Jan 1 where ≥5% of wells (min 10) have spanning coverage
    const threshold = Math.max(10, Math.ceil(wellSpans.length * 0.05));
    let defStartYear = -1, defEndYear = -1;
    for (let y = minYear; y <= maxYear; y++) {
      const jan1 = new Date(y, 0, 1).getTime();
      let spanning = 0;
      for (const span of wellSpans) {
        if (span.min <= jan1 && span.max >= jan1) spanning++;
      }
      if (spanning >= threshold) {
        if (defStartYear === -1) defStartYear = y;
        defEndYear = y;
      }
    }

    let defStart: string, defEnd: string;
    if (defStartYear >= 0) {
      defStart = `${defStartYear}-01-01`;
      defEnd = `${defEndYear}-01-01`;
    } else {
      defStart = minDateStr;
      defEnd = maxDateStr;
    }

    return { densityData, defaultStartDate: defStart, defaultEndDate: defEnd };
  }, [wteMeasurements]);

  const [startDate, setStartDate] = useState(defaultStartDate);
  const [endDate, setEndDate] = useState(defaultEndDate);

  // Model date range for clamping
  const modelDateRange = useMemo(() => {
    if (dataSource !== 'model' || !selectedModelCode || !availableModels) return null;
    const m = availableModels.find(mm => mm.code === selectedModelCode);
    if (!m?.params) return null;
    return { min: m.params.startDate, max: m.params.endDate };
  }, [dataSource, selectedModelCode, availableModels]);

  // Show data source toggle only when WTE + models exist
  const showDataSource = dataType.code === 'wte' && availableModels && availableModels.length > 0;

  // Handle switching between raw and model data source
  const handleDataSourceChange = useCallback((source: 'raw' | 'model') => {
    setDataSource(source);
    if (source === 'model') {
      // Auto-select first model if none selected
      if (availableModels && availableModels.length > 0) {
        const m = availableModels[0];
        setSelectedModelCode(m.code);
        setSelectedModelFilePath(m.filePath);
        if (m.params) {
          setStartDate(m.params.startDate);
          setEndDate(m.params.endDate);
        }
      }
      setSmoothingMethod('model-direct');
    } else {
      setSmoothingMethod('pchip');
      setStartDate(defaultStartDate);
      setEndDate(defaultEndDate);
    }
  }, [availableModels, defaultStartDate, defaultEndDate]);

  // When model selection changes, re-clamp dates
  const handleModelChange = useCallback((code: string) => {
    setSelectedModelCode(code);
    const m = availableModels?.find(mm => mm.code === code);
    if (m) {
      setSelectedModelFilePath(m.filePath);
      if (m.params) {
        // Clamp dates to new model range
        setStartDate(prev => prev < m.params.startDate ? m.params.startDate : prev > m.params.endDate ? m.params.endDate : prev);
        setEndDate(prev => prev > m.params.endDate ? m.params.endDate : prev < m.params.startDate ? m.params.startDate : prev);
      }
    }
  }, [availableModels]);

  const code = slugify(title);
  const hasConflict = existingCodes.includes(code);

  // Validation
  const step1Valid = startDate && endDate && startDate < endDate && wells.length > 0;
  const step2Valid = (() => {
    if (spatialMethod === 'idw' && nodalFunction === 'quadratic' && neighborMode === 'nearest' && neighborCount < 6) return false;
    if (rangeMode === 'custom' && (rangeValue <= 0)) return false;
    if (rangeMode === 'percentage' && (rangeValue <= 0 || rangeValue > 100)) return false;
    return true;
  })();
  const step3Valid = title.trim().length > 0 && !hasConflict;

  const handleRun = async () => {
    cancelledRef.current = false;
    setStep('running');
    setErrorMessage(null);

    const input: RasterPipelineInput = {
      temporal: {
        method: smoothingMethod,
        maWindow: smoothingMonths,
        startDate,
        endDate,
        interval,
        minObservations: minObs,
        minTimeSpan: minSpanYears,
        ...((smoothingMethod === 'model' || smoothingMethod === 'model-direct' || smoothingMethod === 'model-mavg') ? { modelCode: selectedModelCode, modelFilePath: selectedModelFilePath } : {}),
      },
      spatial: {
        method: spatialMethod,
        resolution,
        kriging: {
          variogramModel,
          nugget: nuggetEnabled,
          rangeMode,
          rangeValue: rangeMode === 'auto' ? null : rangeValue,
        },
        idw: {
          exponent: idwExponent,
          nodalFunction,
          neighborMode,
          neighborCount,
        },
        eof: {
          nModes: eofModes,
          maxNeighbors: eofNeighbors,
        },
      },
      general: {
        truncateLow,
        truncateLowValue,
        truncateHigh,
        truncateHighValue,
        logInterpolation,
      },
      title,
    };

    try {
      const result = await runRasterAnalysis(
        input, dataType.code, aquifer, region, wells,
        measurements.filter(m => wellKeySet.has(`${m.regionId}:${m.aquiferId}:${m.wellId}`)),
        (stepText, pct) => {
          if (!cancelledRef.current) {
            setProgressText(stepText);
            setProgressPct(pct);
          }
        }
      );
      if (!cancelledRef.current) {
        setResult(result);
        setStep('complete');
      }
    } catch (err) {
      console.error('Raster analysis failed:', err);
      setErrorMessage(err instanceof Error ? err.message : String(err));
      setStep(3);
    }
  };

  const handleCancel = () => {
    if (step === 'running') {
      cancelledRef.current = true;
    }
    setStep(1);
    setProgressPct(0);
    setProgressText('');
    setErrorMessage(null);
  };

  // Date range markers for charts
  const startTs = startDate ? new Date(startDate).getTime() : undefined;
  const endTs = endDate ? new Date(endDate).getTime() : undefined;

  // Count wells with >= 2 measurements (usable for PCHIP preview)
  const usableWellCount = useMemo(() => {
    const counts = new Map<string, number>();
    for (const m of wteMeasurements) {
      counts.set(m.wellId, (counts.get(m.wellId) || 0) + 1);
    }
    let count = 0;
    for (const c of counts.values()) if (c >= 2) count++;
    return count;
  }, [wteMeasurements]);

  // Well qualification based on minObs and minSpanYears
  const { qualifiedWellCount, omittedWellCount } = useMemo(() => {
    const MS_PER_YEAR = 365.25 * 86400000;
    const byWell = new Map<string, number[]>();
    for (const m of wteMeasurements) {
      const t = new Date(m.date).getTime();
      if (!isNaN(t)) {
        if (!byWell.has(m.wellId)) byWell.set(m.wellId, []);
        byWell.get(m.wellId)!.push(t);
      }
    }
    let qualified = 0, omitted = 0;
    for (const [, times] of byWell) {
      if (times.length < 2) { omitted++; continue; } // need >= 2 for PCHIP
      const obsCount = times.length;
      const span = (Math.max(...times) - Math.min(...times)) / MS_PER_YEAR;
      if (obsCount >= minObs && span >= minSpanYears) {
        qualified++;
      } else {
        omitted++;
      }
    }
    return { qualifiedWellCount: qualified, omittedWellCount: omitted };
  }, [wteMeasurements, minObs, minSpanYears]);

  const stepNumber = typeof step === 'number' ? step : null;

  // --- Input class helpers ---
  const inputCls = "w-full px-2 py-1.5 border border-slate-300 rounded-md text-sm focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500";
  const labelCls = "block text-xs font-medium text-slate-600 mb-1";
  const radioCls = (active: boolean) =>
    `px-3 py-1.5 text-xs font-medium rounded-md border cursor-pointer transition-colors ${
      active
        ? 'bg-emerald-600 text-white border-emerald-600'
        : 'bg-white text-slate-600 border-slate-300 hover:bg-slate-50'
    }`;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/40">
      <div className="bg-white rounded-xl shadow-2xl w-[900px] max-h-[90vh] flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200">
          <div>
            <h2 className="text-lg font-bold text-slate-800">Build Spatial Raster</h2>
            <p className="text-sm text-slate-500">{aquifer.name} &mdash; {region.name}</p>
          </div>
          <div className="flex items-center gap-4">
            {/* Step indicator */}
            {stepNumber && (
              <div className="flex items-center gap-2 text-xs text-slate-500">
                {STEP_LABELS.map((label, i) => {
                  const sn = i + 1;
                  const isActive = sn === stepNumber;
                  const isDone = sn < stepNumber;
                  return (
                    <React.Fragment key={i}>
                      {i > 0 && <div className="w-4 h-px bg-slate-300" />}
                      <div className={`flex items-center gap-1 ${isActive ? 'text-emerald-600 font-semibold' : isDone ? 'text-emerald-500' : 'text-slate-400'}`}>
                        <div className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold ${
                          isActive ? 'bg-emerald-600 text-white' : isDone ? 'bg-emerald-100 text-emerald-600' : 'bg-slate-100 text-slate-400'
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
          {/* ===== STEP 1: Temporal ===== */}
          {step === 1 && (
            <div className="space-y-5">
              {/* Data Source selector (WTE + models only) */}
              {showDataSource && (
                <div>
                  <h3 className="text-sm font-semibold text-slate-700 mb-2">Data Source</h3>
                  <div className="flex items-center gap-3">
                    <div className="flex gap-2">
                      <button onClick={() => handleDataSourceChange('raw')} className={radioCls(dataSource === 'raw')}>
                        Raw Data
                      </button>
                      <button onClick={() => handleDataSourceChange('model')} className={radioCls(dataSource === 'model')}>
                        Imputed Data
                      </button>
                    </div>
                    {dataSource === 'model' && availableModels && (
                      availableModels.length === 1 ? (
                        <span className="text-xs text-slate-500">{availableModels[0].title}</span>
                      ) : (
                        <select
                          value={selectedModelCode}
                          onChange={e => handleModelChange(e.target.value)}
                          className="px-2 py-1 border border-slate-300 rounded-md text-xs focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                        >
                          {availableModels.map(m => (
                            <option key={m.code} value={m.code}>{m.title}</option>
                          ))}
                        </select>
                      )
                    )}
                  </div>
                </div>
              )}

              {/* PCHIP Preview */}
              {wteMeasurements.length > 0 && (
                <div>
                  <h3 className="text-sm font-semibold text-slate-700 mb-1">
                    {dataType.name} Time Series Preview ({usableWellCount} wells with 2+ observations)
                  </h3>
                  <div className="h-[180px] bg-slate-50 rounded-lg border border-slate-200">
                    <PchipPreviewCanvas
                      wells={wells}
                      wteMeasurements={wteMeasurements}
                      startTs={startTs}
                      endTs={endTs}
                      gldasRange={modelDateRange ? {
                        min: new Date(modelDateRange.min).getTime(),
                        max: new Date(modelDateRange.max).getTime(),
                      } : undefined}
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
                        <ReferenceLine y={10} stroke="#ef4444" strokeDasharray="4 3" strokeWidth={1.5} />
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
                              fill={entry.count >= 10 ? '#10b981' : '#94a3b8'}
                              fillOpacity={entry.count >= 10 ? 0.8 : 0.4}
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {/* Options Form */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className={labelCls}>Start Date</label>
                  <div className="flex items-center gap-1">
                    <input type="date" value={startDate}
                      min={modelDateRange?.min} max={modelDateRange?.max}
                      onChange={e => setStartDate(e.target.value)} className={inputCls + ' flex-1'} />
                    <button type="button" className="p-1 text-slate-400 hover:text-slate-700" onClick={() => {
                      const y = parseInt(startDate); if (isNaN(y)) return;
                      let next = `${y - 1}-01-01`;
                      if (modelDateRange && next < modelDateRange.min) next = modelDateRange.min;
                      setStartDate(next);
                    }}>
                      <svg width="14" height="14" viewBox="0 0 14 14"><path d="M9 2L4 7l5 5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
                    </button>
                    <button type="button" className="p-1 text-slate-400 hover:text-slate-700" onClick={() => {
                      const y = parseInt(startDate); if (isNaN(y)) return;
                      let next = `${y + 1}-01-01`;
                      if (modelDateRange && next > modelDateRange.max) next = modelDateRange.max;
                      setStartDate(next);
                    }}>
                      <svg width="14" height="14" viewBox="0 0 14 14"><path d="M5 2l5 5-5 5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
                    </button>
                  </div>
                </div>
                <div>
                  <label className={labelCls}>End Date</label>
                  <div className="flex items-center gap-1">
                    <input type="date" value={endDate}
                      min={modelDateRange?.min} max={modelDateRange?.max}
                      onChange={e => setEndDate(e.target.value)} className={inputCls + ' flex-1'} />
                    <button type="button" className="p-1 text-slate-400 hover:text-slate-700" onClick={() => {
                      const y = parseInt(endDate); if (isNaN(y)) return;
                      let next = `${y - 1}-01-01`;
                      if (modelDateRange && next < modelDateRange.min) next = modelDateRange.min;
                      setEndDate(next);
                    }}>
                      <svg width="14" height="14" viewBox="0 0 14 14"><path d="M9 2L4 7l5 5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
                    </button>
                    <button type="button" className="p-1 text-slate-400 hover:text-slate-700" onClick={() => {
                      const y = parseInt(endDate); if (isNaN(y)) return;
                      let next = `${y + 1}-01-01`;
                      if (modelDateRange && next > modelDateRange.max) next = modelDateRange.max;
                      setEndDate(next);
                    }}>
                      <svg width="14" height="14" viewBox="0 0 14 14"><path d="M5 2l5 5-5 5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
                    </button>
                  </div>
                </div>
                <div>
                  <label className={labelCls}>Resolution (columns)</label>
                  <input type="number" value={resolution} min={10} max={500} step={10}
                    onChange={e => setResolution(Math.max(10, parseInt(e.target.value) || 100))} className={inputCls} />
                </div>
                {dataSource !== 'model' && (
                  <div>
                    <label className={labelCls}>Min Observations / Well</label>
                    <input type="number" value={minObs} min={2} max={100} step={1}
                      onChange={e => setMinObs(Math.max(2, parseInt(e.target.value) || 5))} className={inputCls} />
                  </div>
                )}
                {dataSource !== 'model' && (
                  <div>
                    <label className={labelCls}>Min Time Span / Well (years)</label>
                    <input type="number" value={minSpanYears} min={0} max={50} step={1}
                      onChange={e => setMinSpanYears(Math.max(0, parseFloat(e.target.value) || 5))} className={inputCls} />
                  </div>
                )}
                {dataSource !== 'model' && (
                  <div className="col-span-2">
                    <div className="flex items-center gap-3 text-xs">
                      <span className="text-emerald-600 font-medium">{qualifiedWellCount} wells qualify</span>
                      {omittedWellCount > 0 && (
                        <span className="text-slate-400">{omittedWellCount} omitted</span>
                      )}
                    </div>
                  </div>
                )}
                <div>
                  <label className={labelCls}>Interval</label>
                  <select value={interval} onChange={e => setInterval(e.target.value as any)} className={inputCls}>
                    <option value="3months">3 months</option>
                    <option value="6months">6 months</option>
                    <option value="1year">1 year</option>
                  </select>
                </div>
                <div>
                  <label className={labelCls}>Temporal Method</label>
                  {dataSource === 'model' ? (
                    <select value={smoothingMethod} onChange={e => setSmoothingMethod(e.target.value as any)} className={inputCls}>
                      <option value="model-direct">Direct</option>
                      <option value="model-mavg">Moving Average</option>
                    </select>
                  ) : (
                    <select value={smoothingMethod} onChange={e => setSmoothingMethod(e.target.value as any)} className={inputCls}>
                      <option value="pchip">PCHIP</option>
                      <option value="linear">Linear</option>
                      <option value="moving-average">Moving Average</option>
                    </select>
                  )}
                </div>
                {(smoothingMethod === 'moving-average' || smoothingMethod === 'model-mavg') && (
                  <div>
                    <label className={labelCls}>MA Window (months)</label>
                    <input type="number" value={smoothingMonths} min={1} max={60} step={1}
                      onChange={e => setSmoothingMonths(Math.max(1, Math.min(60, parseInt(e.target.value) || 12)))}
                      className={inputCls} />
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ===== STEP 2: Spatial ===== */}
          {step === 2 && (
            <div className="space-y-5">
              {/* Spatial Method selector */}
              <div>
                <h3 className="text-sm font-semibold text-slate-700 mb-2">Spatial Method</h3>
                <div className="flex gap-2">
                  <button onClick={() => setSpatialMethod('kriging')} className={radioCls(spatialMethod === 'kriging')}>
                    Kriging
                  </button>
                  <button onClick={() => setSpatialMethod('idw')} className={radioCls(spatialMethod === 'idw')}>
                    IDW (Inverse Distance)
                  </button>
                  <button onClick={() => setSpatialMethod('mc')} className={radioCls(spatialMethod === 'mc')}>
                    MC (Low-Rank)
                  </button>
                  <button onClick={() => setSpatialMethod('eof')} className={radioCls(spatialMethod === 'eof')}>
                    EOF (Trend+SVD)
                  </button>
                </div>
              </div>

              {/* Method-specific options */}
              {spatialMethod === 'kriging' && (
                <div className="bg-slate-50 rounded-lg border border-slate-200 p-4 space-y-4">
                  <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wide">Kriging Options</h4>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className={labelCls}>Variogram Model</label>
                      <select value={variogramModel} onChange={e => setVariogramModel(e.target.value as VariogramModel)} className={inputCls}>
                        <option value="gaussian">Gaussian</option>
                        <option value="spherical">Spherical</option>
                        <option value="exponential">Exponential</option>
                      </select>
                    </div>
                    <div className="flex items-end pb-1">
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input type="checkbox" checked={nuggetEnabled}
                          onChange={e => setNuggetEnabled(e.target.checked)}
                          className="w-3.5 h-3.5 rounded accent-emerald-600" />
                        <span className="text-xs text-slate-600">Enable nugget</span>
                      </label>
                    </div>
                  </div>

                  <div>
                    <label className={labelCls}>Range</label>
                    <div className="flex gap-2 mb-2">
                      {(['auto', 'custom', 'percentage'] as KrigingRangeMode[]).map(mode => (
                        <button key={mode} onClick={() => setRangeMode(mode)} className={radioCls(rangeMode === mode)}>
                          {mode === 'auto' ? 'Auto (1/3 diagonal)' : mode === 'custom' ? 'Custom (meters)' : 'Percentage'}
                        </button>
                      ))}
                    </div>
                    {rangeMode !== 'auto' && (
                      <input type="number" value={rangeValue} min={0.1}
                        step={rangeMode === 'percentage' ? 1 : 100}
                        onChange={e => setRangeValue(parseFloat(e.target.value) || 0)}
                        className={inputCls}
                        placeholder={rangeMode === 'custom' ? 'Range in meters' : 'Percentage of diagonal'} />
                    )}
                  </div>
                </div>
              )}

              {spatialMethod === 'idw' && (
                <div className="bg-slate-50 rounded-lg border border-slate-200 p-4 space-y-4">
                  <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wide">IDW Options</h4>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className={labelCls}>Exponent</label>
                      <input type="number" value={idwExponent} min={0.5} max={10} step={0.1}
                        onChange={e => setIdwExponent(Math.max(0.5, Math.min(10, parseFloat(e.target.value) || 2)))}
                        className={inputCls} />
                    </div>
                    <div>
                      <label className={labelCls}>Nodal Function</label>
                      <select value={nodalFunction} onChange={e => setNodalFunction(e.target.value as IdwNodalFunction)} className={inputCls}>
                        <option value="classic">Classic</option>
                        <option value="gradient">Gradient Plane</option>
                        <option value="quadratic">Quadratic</option>
                      </select>
                    </div>
                  </div>

                  <div>
                    <label className={labelCls}>Neighbors</label>
                    <div className="flex gap-2 mb-2">
                      <button onClick={() => setNeighborMode('all')} className={radioCls(neighborMode === 'all')}>
                        All Wells
                      </button>
                      <button onClick={() => setNeighborMode('nearest')} className={radioCls(neighborMode === 'nearest')}>
                        Nearest N
                      </button>
                    </div>
                    {neighborMode === 'nearest' && (
                      <div>
                        <input type="number" value={neighborCount} min={3} max={100} step={1}
                          onChange={e => setNeighborCount(Math.max(3, parseInt(e.target.value) || 12))}
                          className={inputCls} />
                        {nodalFunction === 'quadratic' && neighborCount < 6 && (
                          <p className="text-[10px] text-red-500 mt-1">Quadratic nodal function requires at least 6 neighbors</p>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {spatialMethod === 'mc' && (
                <div className="bg-slate-50 rounded-lg border border-slate-200 p-4 space-y-3">
                  <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wide">MC Options</h4>
                  <p className="text-xs text-slate-600">
                    MC uses low-rank matrix completion across the full time × well matrix, then projects the completed well field to the grid.
                  </p>
                  <p className="text-[11px] text-slate-500">
                    Grid projection currently uses the same IDW settings below the hood after matrix completion. No separate MC tuning is exposed yet.
                  </p>
                </div>
              )}

              {spatialMethod === 'eof' && (
                <div className="bg-slate-50 rounded-lg border border-slate-200 p-4 space-y-4">
                  <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wide">EOF Options</h4>
                  <p className="text-xs text-slate-600">
                    EOF fits a spatial trend surface (polynomial of lat/lon/elevation), then decomposes the residuals via SVD into shared temporal modes. Grid cells are predicted by interpolating the spatial loadings from nearby wells.
                  </p>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className={labelCls}>EOF Modes</label>
                      <input type="number" value={eofModes} min={3} max={50} step={1}
                        onChange={e => setEofModes(Math.max(3, Math.min(50, parseInt(e.target.value) || 20)))}
                        className={inputCls} />
                      <p className="text-[10px] text-slate-400 mt-1">Number of temporal patterns to retain (20 recommended)</p>
                    </div>
                    <div>
                      <label className={labelCls}>IDW Neighbors</label>
                      <input type="number" value={eofNeighbors} min={5} max={100} step={1}
                        onChange={e => setEofNeighbors(Math.max(5, Math.min(100, parseInt(e.target.value) || 30)))}
                        className={inputCls} />
                      <p className="text-[10px] text-slate-400 mt-1">Wells used to interpolate loadings (30 recommended)</p>
                    </div>
                  </div>
                </div>
              )}

              {/* General interpolation options */}
              <div className="bg-slate-50 rounded-lg border border-slate-200 p-4 space-y-3">
                <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wide">General Options</h4>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="flex items-center gap-2 cursor-pointer mb-1">
                      <input type="checkbox" checked={truncateLow}
                        onChange={e => setTruncateLow(e.target.checked)}
                        className="w-3.5 h-3.5 rounded accent-emerald-600" />
                      <span className="text-xs text-slate-600">Truncate Low</span>
                    </label>
                    {truncateLow && (
                      <input type="number" value={truncateLowValue} step={1}
                        onChange={e => setTruncateLowValue(parseFloat(e.target.value) || 0)}
                        className={inputCls} />
                    )}
                  </div>
                  <div>
                    <label className="flex items-center gap-2 cursor-pointer mb-1">
                      <input type="checkbox" checked={truncateHigh}
                        onChange={e => setTruncateHigh(e.target.checked)}
                        className="w-3.5 h-3.5 rounded accent-emerald-600" />
                      <span className="text-xs text-slate-600">Truncate High</span>
                    </label>
                    {truncateHigh && (
                      <input type="number" value={truncateHighValue} step={1}
                        onChange={e => setTruncateHighValue(parseFloat(e.target.value) || 0)}
                        className={inputCls} />
                    )}
                  </div>
                </div>

                <div>
                  <label className="flex items-center gap-2 cursor-pointer" title={hasNonPositiveValues ? 'Cannot use log interpolation: dataset contains zero or negative values' : ''}>
                    <input type="checkbox" checked={logInterpolation}
                      disabled={hasNonPositiveValues}
                      onChange={e => setLogInterpolation(e.target.checked)}
                      className="w-3.5 h-3.5 rounded accent-emerald-600 disabled:opacity-40" />
                    <span className={`text-xs ${hasNonPositiveValues ? 'text-slate-400' : 'text-slate-600'}`}>
                      Log Interpolation
                      {hasNonPositiveValues && <span className="ml-1 text-[10px] text-amber-500">(non-positive values present)</span>}
                    </span>
                  </label>
                </div>
              </div>
            </div>
          )}

          {/* ===== STEP 3: Title & Summary ===== */}
          {step === 3 && (
            <div className="space-y-5">
              {errorMessage && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-3 flex items-start gap-2">
                  <AlertTriangle size={16} className="text-red-500 mt-0.5 shrink-0" />
                  <div>
                    <p className="text-sm font-medium text-red-800">Analysis Failed</p>
                    <p className="text-xs text-red-600 mt-1">{errorMessage}</p>
                  </div>
                </div>
              )}

              <div>
                <label className={labelCls}>Title</label>
                <input type="text" value={title} onChange={e => {
                  // Only allow [a-zA-Z0-9 _-]
                  const v = e.target.value.replace(/[^a-zA-Z0-9 _-]/g, '');
                  setTitle(v);
                }}
                  placeholder="e.g. Annual Analysis 2024"
                  className={inputCls} />
                {title && (
                  <div className="mt-1 flex items-center gap-2">
                    <span className="text-[10px] text-slate-400">{slugify(aquifer.name)}/raster_{dataType.code}_{code}.json</span>
                    {hasConflict && <span className="text-[10px] text-red-500 font-medium">Name already exists</span>}
                  </div>
                )}
              </div>

              {/* Options summary */}
              <div className="bg-slate-50 rounded-lg border border-slate-200 p-4">
                <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wide mb-3">Options Summary</h4>
                <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs text-slate-600">
                  <div><span className="text-slate-400">Dates:</span> {startDate} to {endDate}</div>
                  <div><span className="text-slate-400">Interval:</span> {interval}</div>
                  <div><span className="text-slate-400">Resolution:</span> {resolution} cols</div>
                  <div><span className="text-slate-400">Temporal:</span> {
                    smoothingMethod === 'pchip' ? 'PCHIP' :
                    smoothingMethod === 'linear' ? 'Linear' :
                    smoothingMethod === 'model' ? `Model (${selectedModelCode})` :
                    smoothingMethod === 'model-direct' ? `Model Direct (${selectedModelCode})` :
                    smoothingMethod === 'model-mavg' ? `Model MA ${smoothingMonths}mo (${selectedModelCode})` :
                    `MA ${smoothingMonths}mo`
                  }</div>
                  <div><span className="text-slate-400">Wells:</span> {qualifiedWellCount} qualified</div>
                  <div><span className="text-slate-400">Method:</span> {spatialMethod === 'kriging' ? `Kriging (${variogramModel})` : spatialMethod === 'mc' ? 'MC (Low-Rank)' : `IDW (p=${idwExponent})`}</div>
                  {spatialMethod === 'kriging' && (
                    <>
                      <div><span className="text-slate-400">Nugget:</span> {nuggetEnabled ? 'Yes' : 'No'}</div>
                      <div><span className="text-slate-400">Range:</span> {rangeMode === 'auto' ? 'Auto' : rangeMode === 'custom' ? `${rangeValue}m` : `${rangeValue}%`}</div>
                    </>
                  )}
                  {spatialMethod === 'idw' && (
                    <>
                      <div><span className="text-slate-400">Nodal:</span> {nodalFunction}</div>
                      <div><span className="text-slate-400">Neighbors:</span> {neighborMode === 'all' ? 'All' : `Nearest ${neighborCount}`}</div>
                    </>
                  )}
                  {spatialMethod === 'mc' && (
                    <>
                      <div><span className="text-slate-400">Projection:</span> IDW after MC completion</div>
                      <div><span className="text-slate-400">MC Rank:</span> Auto (up to 8)</div>
                    </>
                  )}
                  {(truncateLow || truncateHigh || logInterpolation) && (
                    <div className="col-span-2 mt-1 pt-1 border-t border-slate-200">
                      <span className="text-slate-400">Post-processing:</span>{' '}
                      {[
                        truncateLow && `low >= ${truncateLowValue}`,
                        truncateHigh && `high <= ${truncateHighValue}`,
                        logInterpolation && 'log transform',
                      ].filter(Boolean).join(', ')}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* ===== RUNNING ===== */}
          {step === 'running' && (
            <div className="flex flex-col items-center justify-center py-12 space-y-4">
              <Loader2 className="w-10 h-10 text-emerald-500 animate-spin" />
              <p className="text-sm text-slate-600">{progressText}</p>
              <div className="w-80 bg-slate-200 rounded-full h-2.5">
                <div className="bg-emerald-500 h-2.5 rounded-full transition-all duration-300"
                  style={{ width: `${progressPct}%` }} />
              </div>
              <p className="text-xs text-slate-400">{Math.round(progressPct)}%</p>
            </div>
          )}

          {/* ===== COMPLETE ===== */}
          {step === 'complete' && result && (
            <div className="flex flex-col items-center justify-center py-12 space-y-4">
              <CheckCircle2 className="w-12 h-12 text-emerald-500" />
              <h3 className="text-lg font-semibold text-slate-800">Analysis Complete</h3>
              <div className="text-sm text-slate-600 text-center space-y-1">
                <p>{result.frames.length} timesteps &bull; {result.params.startDate} to {result.params.endDate}</p>
                <p>{result.aquiferName}</p>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-3 border-t border-slate-200 bg-slate-50">
          {/* Left: Cancel */}
          <div>
            {(stepNumber || step === 'running') && (
              <button onClick={step === 'running' ? handleCancel : onClose}
                className="px-4 py-2 text-sm text-slate-600 hover:bg-slate-100 rounded-lg transition-colors">
                Cancel
              </button>
            )}
          </div>

          {/* Center: Back */}
          <div>
            {stepNumber && stepNumber > 1 && (
              <button onClick={() => setStep((stepNumber - 1) as Step)}
                className="flex items-center gap-1 px-4 py-2 text-sm text-slate-600 hover:bg-slate-100 rounded-lg transition-colors">
                <ChevronLeft size={16} />
                Back
              </button>
            )}
          </div>

          {/* Right: Next / Run / View */}
          <div>
            {step === 1 && (
              <button onClick={() => setStep(2)} disabled={!step1Valid}
                className="flex items-center gap-1 px-4 py-2 bg-emerald-600 text-white text-sm font-medium rounded-lg hover:bg-emerald-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed">
                Next
                <ChevronRight size={16} />
              </button>
            )}
            {step === 2 && (
              <button onClick={() => {
                  if (!title) {
                    const temporalLabel = formatTemporalMethodLabel(smoothingMethod, smoothingMonths, selectedModelCode);
                    const spatialLabel = formatSpatialMethodLabel(spatialMethod, variogramModel, nodalFunction);
                    setTitle(`${temporalLabel}_${spatialLabel}`);
                  }
                  setStep(3);
                }} disabled={!step2Valid}
                className="flex items-center gap-1 px-4 py-2 bg-emerald-600 text-white text-sm font-medium rounded-lg hover:bg-emerald-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed">
                Next
                <ChevronRight size={16} />
              </button>
            )}
            {step === 3 && (
              <button onClick={handleRun} disabled={!step3Valid}
                className="flex items-center gap-2 px-4 py-2 bg-emerald-600 text-white text-sm font-medium rounded-lg hover:bg-emerald-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed">
                <Play size={16} />
                Run Analysis
              </button>
            )}
            {step === 'running' && (
              <div /> /* Cancel is on the left */
            )}
            {step === 'complete' && result && (
              <button onClick={() => onComplete(result)}
                className="flex items-center gap-2 px-4 py-2 bg-emerald-600 text-white text-sm font-medium rounded-lg hover:bg-emerald-700 transition-colors">
                View Results
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SpatialAnalysisDialog;
