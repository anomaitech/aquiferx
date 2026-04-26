import React, { useMemo, useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Legend, Scatter, ScatterChart, ComposedChart, ReferenceLine,
} from 'recharts';
import { ImputationModelResult, Well, Measurement } from '../types';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { smoothModelCombined } from '../utils/interpolation';

interface ModelTimeSeriesProps {
  model: ImputationModelResult;
  well: Well;
  measurements: Measurement[];
  showCombined: boolean;
  onToggleCombined: () => void;
  lengthUnit: 'ft' | 'm';
  showSmooth?: boolean;
  smoothMonths?: number;
  showGSE?: boolean;
}

const ModelTimeSeries: React.FC<ModelTimeSeriesProps> = ({
  model, well, measurements, showCombined, onToggleCombined, lengthUnit,
  showSmooth = false, smoothMonths = 3, showGSE = false,
}) => {
  const [logExpanded, setLogExpanded] = useState(false);

  const wellMetrics = model.wellMetrics[well.id];
  const hasFiniteMetrics = !!wellMetrics && Number.isFinite(wellMetrics.r2) && Number.isFinite(wellMetrics.rmse);

  const modelWindowMonthlyCount = useMemo(() => {
    const startDate = typeof model.params.startDate === 'string' ? model.params.startDate : null;
    const endDate = typeof model.params.endDate === 'string' ? model.params.endDate : null;
    const months = new Set<string>();
    for (const m of measurements) {
      if (m.wellId !== well.id || m.dataType !== 'wte') continue;
      if (startDate && m.date < startDate) continue;
      if (endDate && m.date > endDate) continue;
      months.add(m.date.slice(0, 7));
    }
    return months.size;
  }, [measurements, model.params.endDate, model.params.startDate, well.id]);

  // Get original WTE measurements for this well
  const wteMeas = useMemo(() =>
    measurements.filter(m => m.wellId === well.id && m.dataType === 'wte')
      .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()),
  [measurements, well.id]);

  // Get model data rows for this well
  const modelRows = useMemo(() =>
    model.data.filter(r => r.well_id === well.id).sort((a, b) => a.date.localeCompare(b.date)),
  [model.data, well.id]);

  // Compute smoothed combined data when in combined mode with smooth enabled
  const smoothedData = useMemo(() => {
    if (!showCombined || !showSmooth || modelRows.length < 2) return null;
    const combinedRows = modelRows.map(row => ({
      date: row.date,
      combined: row.pchip !== null ? row.pchip : row.model,
    }));
    const { dates, values } = smoothModelCombined(combinedRows, smoothMonths);
    const map = new Map<number, number>();
    for (let i = 0; i < dates.length; i++) map.set(dates[i], values[i]);
    return map;
  }, [showCombined, showSmooth, smoothMonths, modelRows]);

  // Build chart data
  const chartData = useMemo(() => {
    if (showCombined) {
      // Combined view: red for PCHIP, blue for ELM
      return modelRows.map(row => {
        const ts = new Date(row.date).getTime();
        return {
          date: ts,
          pchip: row.pchip,
          elm: row.pchip !== null ? null : row.model,
          smooth: smoothedData?.get(ts) ?? null,
        };
      });
    } else {
      // Uncombined view: green dots for measurements, red/blue lines
      const dataMap = new Map<number, any>();

      // Add model rows (monthly grid: 1st of each month)
      const modelTimestamps: number[] = [];
      for (const row of modelRows) {
        const ts = new Date(row.date).getTime();
        modelTimestamps.push(ts);
        dataMap.set(ts, {
          date: ts,
          model: row.model,
          pchip: row.pchip,
        });
      }

      // Snap measurement dots to nearest monthly grid date so they don't
      // break the PCHIP/ELM lines (measurements between grid dates would
      // create entries with undefined pchip, severing connectNulls={false} lines).
      // Measurements outside the model range keep their real dates so the
      // chart x-axis expands to show them.
      const modelMin = modelTimestamps.length > 0 ? modelTimestamps[0] : Infinity;
      const modelMax = modelTimestamps.length > 0 ? modelTimestamps[modelTimestamps.length - 1] : -Infinity;
      for (const m of wteMeas) {
        const ts = new Date(m.date).getTime();
        let snapTs = ts;
        if (ts >= modelMin && ts <= modelMax) {
          // Inside model range: snap to nearest grid date
          let bestDist = Infinity;
          for (const mt of modelTimestamps) {
            const dist = Math.abs(mt - ts);
            if (dist < bestDist) { bestDist = dist; snapTs = mt; }
          }
        }
        const existing = dataMap.get(snapTs) || { date: snapTs };
        existing.measurement = m.value;
        dataMap.set(snapTs, existing);
      }

      return Array.from(dataMap.values()).sort((a, b) => a.date - b.date);
    }
  }, [modelRows, wteMeas, showCombined, smoothedData]);

  // Compute tight Y domain from all values
  const yDomain = useMemo(() => {
    let min = Infinity, max = -Infinity;
    for (const d of chartData) {
      for (const key of ['pchip', 'elm', 'model', 'measurement', 'smooth'] as const) {
        const v = d[key];
        if (v != null && isFinite(v)) {
          if (v < min) min = v;
          if (v > max) max = v;
        }
      }
    }
    if (showGSE && well.gse != null && !isNaN(well.gse)) {
      if (well.gse < min) min = well.gse;
      if (well.gse > max) max = well.gse;
    }
    if (!isFinite(min)) return [0, 1];
    const pad = (max - min) * 0.05 || 1;
    return [min - pad, max + pad];
  }, [chartData, showGSE, well.gse]);

  const formatDate = (ts: number) => new Date(ts).getFullYear().toString();

  // Generate ticks at Jan 1 of each year
  const yearTicks = useMemo(() => {
    if (chartData.length === 0) return [];
    const minTs = chartData[0].date;
    const maxTs = chartData[chartData.length - 1].date;
    const startYear = new Date(minTs).getFullYear();
    const endYear = new Date(maxTs).getFullYear();
    const ticks: number[] = [];
    for (let y = startYear; y <= endYear; y++) {
      ticks.push(new Date(y, 0, 1).getTime());
    }
    return ticks;
  }, [chartData]);

  if (modelRows.length === 0) {
    const rangeLabel = typeof model.params.startDate === 'string' && typeof model.params.endDate === 'string'
      ? `${model.params.startDate} to ${model.params.endDate}`
      : 'the model date range';
    return (
      <div className="flex h-full items-center justify-center px-6 text-center">
        <div className="space-y-1">
          <div className="text-sm font-medium text-slate-500">No model data for this well</div>
          <div className="text-xs text-slate-400">
            This well was likely omitted from the imputation run because it did not meet the model qualification threshold.
          </div>
          <div className="text-[11px] text-slate-400">
            Monthly WTE observations in {rangeLabel}: <span className="font-medium text-slate-500">{modelWindowMonthlyCount}</span>
          </div>
          <div className="text-[11px] text-slate-400">
            Model minimum samples: <span className="font-medium text-slate-500">{model.params.minSamples}</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header with toggle and metrics */}
      <div className="flex items-center justify-between px-3 py-1.5 bg-amber-50 border-b border-amber-200">
        <div className="flex items-center gap-3">
          <span className="text-xs font-semibold text-amber-800">Model: {model.title}</span>
          <label className="flex items-center gap-1 text-[10px] text-amber-800 cursor-pointer select-none">
            <input type="checkbox" checked={showCombined} onChange={onToggleCombined} className="accent-amber-600" />
            Combined
          </label>
        </div>
        {hasFiniteMetrics && wellMetrics && (
          <div className="flex items-center gap-2">
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-700 font-medium">
              R² = {wellMetrics.r2.toFixed(3)}
            </span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-100 text-blue-700 font-medium">
              RMSE = {wellMetrics.rmse.toFixed(2)} {lengthUnit}
            </span>
          </div>
        )}
      </div>

      {/* Chart */}
      <div className="flex-1 min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 10, right: 20, bottom: 5, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="date"
              type="number"
              scale="time"
              domain={['dataMin', 'dataMax']}
              ticks={yearTicks}
              tickFormatter={formatDate}
              stroke="#94a3b8"
              fontSize={10}
            />
            <YAxis stroke="#94a3b8" fontSize={10} domain={yDomain} tickFormatter={(v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 1 })} />
            <Tooltip
              labelFormatter={(ts: number) => { const d = new Date(ts); return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`; }}
              formatter={(value: number, name: string) => [
                value?.toFixed(2) ?? 'N/A',
                name === 'measurement' ? 'Measured' : name === 'pchip' ? 'PCHIP' : name === 'elm' ? 'ELM' : name === 'model' ? 'ELM' : name === 'smooth' ? 'MAvg' : name
              ]}
            />

            {showCombined ? (
              <>
                <Line
                  type="monotone"
                  dataKey="pchip"
                  stroke="#ef4444"
                  strokeWidth={1.5}
                  dot={false}
                  connectNulls={false}
                  name="PCHIP"
                  animationDuration={400}
                />
                <Line
                  type="monotone"
                  dataKey="elm"
                  stroke="#3b82f6"
                  strokeWidth={1.5}
                  dot={false}
                  connectNulls={false}
                  name="ELM"
                  animationDuration={400}
                />
                {smoothedData && (
                  <Line
                    type="monotone"
                    dataKey="smooth"
                    stroke="#f97316"
                    strokeWidth={2}
                    strokeDasharray="6 3"
                    dot={false}
                    connectNulls
                    name="MAvg"
                    animationDuration={400}
                  />
                )}
              </>
            ) : (
              <>
                <Line
                  type="monotone"
                  dataKey="pchip"
                  stroke="#ef4444"
                  strokeWidth={2}
                  dot={false}
                  connectNulls={false}
                  name="PCHIP"
                  animationDuration={400}
                />
                <Line
                  type="monotone"
                  dataKey="model"
                  stroke="#3b82f6"
                  strokeWidth={1.5}
                  dot={false}
                  connectNulls={false}
                  name="ELM"
                  animationDuration={400}
                />
                <Line
                  type="monotone"
                  dataKey="measurement"
                  stroke="#22c55e"
                  strokeWidth={0}
                  dot={{ fill: '#22c55e', r: 3, strokeWidth: 0 }}
                  connectNulls={false}
                  name="Measured"
                  isAnimationActive={false}
                />
              </>
            )}

            {(
              <Legend
                content={({ payload }: any) => (
                  <div style={{ display: 'flex', justifyContent: 'center', gap: 16, fontSize: 11 }}>
                    {payload?.map((entry: any) => (
                      <span key={entry.value} style={{ display: 'inline-flex', alignItems: 'center', gap: 4, color: entry.color || '#666' }}>
                        {entry.value === 'Measured' ? (
                          <svg width="8" height="8"><circle cx="4" cy="4" r="4" fill="#22c55e" /></svg>
                        ) : entry.value === 'MAvg' ? (
                          <svg width="14" height="2"><line x1="0" y1="1" x2="14" y2="1" stroke={entry.color} strokeWidth="2" strokeDasharray="3 2" /></svg>
                        ) : (
                          <svg width="14" height="2"><line x1="0" y1="1" x2="14" y2="1" stroke={entry.color} strokeWidth="2" /></svg>
                        )}
                        {entry.value}
                      </span>
                    ))}
                  </div>
                )}
              />
            )}
            {showGSE && well.gse != null && !isNaN(well.gse) && (
              <ReferenceLine
                y={well.gse}
                stroke="#8B4513"
                strokeDasharray="6 3"
                strokeWidth={1.5}
                label={{ value: 'GSE', position: 'right', fill: '#8B4513', fontSize: 10 }}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Collapsible log */}
      {model.log && model.log.length > 0 && (
        <div className="border-t border-slate-200">
          <button
            onClick={() => setLogExpanded(!logExpanded)}
            className="w-full flex items-center gap-1 px-3 py-1 text-[10px] text-slate-500 hover:bg-slate-50"
          >
            {logExpanded ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
            Processing Log ({model.log.length} entries)
          </button>
          {logExpanded && (
            <div className="max-h-[120px] overflow-y-auto px-3 py-1 bg-slate-900 font-mono text-[10px] text-slate-300 space-y-0.5">
              {model.log.map((msg, i) => (
                <div key={i} className={msg.startsWith('ERROR') ? 'text-red-400' : msg.includes('R²') ? 'text-emerald-400' : ''}>
                  {msg}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ModelTimeSeries;
