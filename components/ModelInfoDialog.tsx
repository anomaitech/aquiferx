import React from 'react';
import { X } from 'lucide-react';
import { ImputationModelMeta, ImputationWellMetrics } from '../types';

interface ModelInfoDialogProps {
  meta: ImputationModelMeta;
  allModels: ImputationModelMeta[];
  onClose: () => void;
}

function summarizeMetrics(metrics: ImputationWellMetrics[]) {
  const finiteR2 = metrics.map(w => w.r2).filter(Number.isFinite);
  const finiteRmse = metrics.map(w => w.rmse).filter(Number.isFinite);
  return {
    avgR2: finiteR2.length ? finiteR2.reduce((a, b) => a + b, 0) / finiteR2.length : null,
    avgRmse: finiteRmse.length ? finiteRmse.reduce((a, b) => a + b, 0) / finiteRmse.length : null,
    count: metrics.length,
    finiteCount: Math.min(finiteR2.length, finiteRmse.length),
  };
}

const ModelInfoDialog: React.FC<ModelInfoDialogProps> = ({ meta, allModels, onClose }) => {
  const rowCls = "flex justify-between py-1 border-b border-slate-100 last:border-0";
  const labelCls = "text-slate-400";
  const valueCls = "text-slate-700 font-medium";

  const wellCount = Object.keys(meta.wellMetrics).length;
  const metrics = Object.values(meta.wellMetrics) as ImputationWellMetrics[];
  const summary = summarizeMetrics(metrics);
  const exactCounterpart = allModels.find(model =>
    model.filePath !== meta.filePath
    && model.regionId === meta.regionId
    && model.aquiferId === meta.aquiferId
    && model.dataType === meta.dataType
    && model.params.startDate === meta.params.startDate
    && model.params.endDate === meta.params.endDate
    && model.params.minSamples === meta.params.minSamples
    && model.method !== meta.method
  ) || null;
  const fallbackCounterpart = allModels.find(model =>
    model.filePath !== meta.filePath
    && model.regionId === meta.regionId
    && model.aquiferId === meta.aquiferId
    && model.dataType === meta.dataType
    && model.params.startDate === meta.params.startDate
    && model.params.endDate === meta.params.endDate
    && model.method !== meta.method
  ) || null;
  const counterpart = exactCounterpart || fallbackCounterpart;
  const counterpartSummary = counterpart
    ? summarizeMetrics(Object.values(counterpart.wellMetrics) as ImputationWellMetrics[])
    : null;
  const methodLabel = meta.method === 'browser-mc-lnn' ? 'MC + LNN (Validated Python)' : 'PCHIP + ELM';
  const counterpartMethodLabel = counterpart?.method === 'browser-mc-lnn' ? 'MC + LNN (Validated Python)' : 'PCHIP + ELM';
  const isFallbackComparison = !!counterpart && !exactCounterpart;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/40">
      <div className="bg-white rounded-xl shadow-2xl w-[480px] max-h-[80vh] flex flex-col overflow-hidden">
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200">
          <h2 className="text-lg font-bold text-slate-800">Model Info</h2>
          <button onClick={onClose} className="p-1 hover:bg-slate-100 rounded-lg transition-colors">
            <X size={20} className="text-slate-400" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-6 py-4 text-xs space-y-4">
          {/* General */}
          <section>
            <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-wide mb-2">General</h3>
            <div className="space-y-0">
              <div className={rowCls}><span className={labelCls}>Title</span><span className={valueCls}>{meta.title}</span></div>
              <div className={rowCls}><span className={labelCls}>Code</span><span className={valueCls}>{meta.code}</span></div>
              <div className={rowCls}><span className={labelCls}>Method</span><span className={valueCls}>{methodLabel}</span></div>
              <div className={rowCls}><span className={labelCls}>Data Type</span><span className={valueCls}>{meta.dataType.toUpperCase()}</span></div>
              <div className={rowCls}><span className={labelCls}>Aquifer</span><span className={valueCls}>{meta.aquiferName}</span></div>
              <div className={rowCls}><span className={labelCls}>Created</span><span className={valueCls}>{meta.createdAt ? new Date(meta.createdAt).toLocaleString() : 'N/A'}</span></div>
            </div>
          </section>

          {/* Parameters */}
          <section>
            <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-wide mb-2">Parameters</h3>
            <div className="space-y-0">
              <div className={rowCls}><span className={labelCls}>Output Dates</span><span className={valueCls}>{meta.params.startDate} to {meta.params.endDate}</span></div>
              <div className={rowCls}><span className={labelCls}>GLDAS Training</span><span className={valueCls}>{meta.params.gldasStartDate} to {meta.params.gldasEndDate}</span></div>
              <div className={rowCls}><span className={labelCls}>Min Samples</span><span className={valueCls}>{meta.params.minSamples}</span></div>
              <div className={rowCls}><span className={labelCls}>Gap Size</span><span className={valueCls}>{meta.params.gapSize} days</span></div>
              <div className={rowCls}><span className={labelCls}>Pad Size</span><span className={valueCls}>{meta.params.padSize} days</span></div>
              <div className={rowCls}><span className={labelCls}>Hidden Units</span><span className={valueCls}>{meta.params.hiddenUnits}</span></div>
              <div className={rowCls}><span className={labelCls}>Lambda</span><span className={valueCls}>{meta.params.lambda}</span></div>
            </div>
          </section>

          {/* Well Metrics */}
          <section>
            <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-wide mb-2">Well Metrics</h3>
            <div className="space-y-0">
              <div className={rowCls}><span className={labelCls}>Wells Modeled</span><span className={valueCls}>{wellCount}</span></div>
              {summary.avgR2 !== null && (
                <div className={rowCls}><span className={labelCls}>Avg R²</span><span className={valueCls}>{summary.avgR2.toFixed(4)}</span></div>
              )}
              {summary.avgRmse !== null && (
                <div className={rowCls}><span className={labelCls}>Avg RMSE</span><span className={valueCls}>{summary.avgRmse.toFixed(4)}</span></div>
              )}
            </div>
          </section>

          <section>
            <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-wide mb-2">Method Comparison</h3>
            {counterpart && counterpartSummary ? (
              <div className="space-y-0">
                <div className={rowCls}><span className={labelCls}>Compared Against</span><span className={valueCls}>{counterpartMethodLabel}</span></div>
                <div className={rowCls}><span className={labelCls}>Comparison Basis</span><span className={valueCls}>{isFallbackComparison ? 'Same aquifer/date window' : 'Same aquifer/date/min samples'}</span></div>
                {isFallbackComparison && (
                  <div className={rowCls}><span className={labelCls}>Other Min Samples</span><span className={valueCls}>{counterpart.params.minSamples}</span></div>
                )}
                <div className={rowCls}><span className={labelCls}>Current Avg R²</span><span className={valueCls}>{summary.avgR2 == null ? 'N/A' : summary.avgR2.toFixed(4)}</span></div>
                <div className={rowCls}><span className={labelCls}>Other Avg R²</span><span className={valueCls}>{counterpartSummary.avgR2 == null ? 'N/A' : counterpartSummary.avgR2.toFixed(4)}</span></div>
                <div className={rowCls}><span className={labelCls}>Current Avg RMSE</span><span className={valueCls}>{summary.avgRmse == null ? 'N/A' : summary.avgRmse.toFixed(4)}</span></div>
                <div className={rowCls}><span className={labelCls}>Other Avg RMSE</span><span className={valueCls}>{counterpartSummary.avgRmse == null ? 'N/A' : counterpartSummary.avgRmse.toFixed(4)}</span></div>
                {summary.avgR2 != null && counterpartSummary.avgR2 != null && (
                  <div className={rowCls}><span className={labelCls}>R² Delta</span><span className={valueCls}>{(summary.avgR2 - counterpartSummary.avgR2).toFixed(4)}</span></div>
                )}
                {summary.avgRmse != null && counterpartSummary.avgRmse != null && (
                  <div className={rowCls}><span className={labelCls}>RMSE Delta</span><span className={valueCls}>{(summary.avgRmse - counterpartSummary.avgRmse).toFixed(4)}</span></div>
                )}
              </div>
            ) : (
              <div className="text-slate-500 text-[11px]">
                No matching run from the other method was found for this aquifer and date window.
              </div>
            )}
          </section>
        </div>

        <div className="flex justify-end px-6 py-3 border-t border-slate-200 bg-slate-50">
          <button onClick={onClose}
            className="px-4 py-2 text-sm text-slate-600 hover:bg-slate-100 rounded-lg transition-colors">
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default ModelInfoDialog;
