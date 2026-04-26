import React from 'react';
import { X } from 'lucide-react';
import { RasterAnalysisMeta } from '../types';

interface RasterInfoDialogProps {
  meta: RasterAnalysisMeta;
  onClose: () => void;
}

const RasterInfoDialog: React.FC<RasterInfoDialogProps> = ({ meta, onClose }) => {
  const opts = meta.options;
  const rowCls = "flex justify-between py-1 border-b border-slate-100 last:border-0";
  const labelCls = "text-slate-400";
  const valueCls = "text-slate-700 font-medium";

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/40">
      <div className="bg-white rounded-xl shadow-2xl w-[480px] max-h-[80vh] flex flex-col overflow-hidden">
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200">
          <h2 className="text-lg font-bold text-slate-800">Raster Info</h2>
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
              <div className={rowCls}><span className={labelCls}>Data Type</span><span className={valueCls}>{meta.dataType.toUpperCase()}</span></div>
              <div className={rowCls}><span className={labelCls}>Aquifer</span><span className={valueCls}>{meta.aquiferName}</span></div>
              <div className={rowCls}><span className={labelCls}>Created</span><span className={valueCls}>{meta.createdAt ? new Date(meta.createdAt).toLocaleString() : 'N/A'}</span></div>
            </div>
          </section>

          {/* Temporal */}
          <section>
            <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-wide mb-2">Temporal</h3>
            {opts ? (
              <div className="space-y-0">
                <div className={rowCls}><span className={labelCls}>Dates</span><span className={valueCls}>{opts.temporal.startDate} to {opts.temporal.endDate}</span></div>
                <div className={rowCls}><span className={labelCls}>Interval</span><span className={valueCls}>{opts.temporal.interval}</span></div>
                <div className={rowCls}><span className={labelCls}>Method</span><span className={valueCls}>{opts.temporal.method === 'pchip' ? 'PCHIP' : opts.temporal.method === 'linear' ? 'Linear' : `Moving Average (${opts.temporal.maWindow}mo)`}</span></div>
                <div className={rowCls}><span className={labelCls}>Min Observations</span><span className={valueCls}>{opts.temporal.minObservations}</span></div>
                <div className={rowCls}><span className={labelCls}>Min Time Span</span><span className={valueCls}>{opts.temporal.minTimeSpan} years</span></div>
              </div>
            ) : (
              <div className="space-y-0">
                <div className={rowCls}><span className={labelCls}>Dates</span><span className={valueCls}>{meta.params.startDate} to {meta.params.endDate}</span></div>
                <div className={rowCls}><span className={labelCls}>Interval</span><span className={valueCls}>{meta.params.interval}</span></div>
                <div className={rowCls}><span className={labelCls}>Method</span><span className={valueCls}>{meta.params.smoothingMethod === 'pchip' ? 'PCHIP' : meta.params.smoothingMethod === 'linear' ? 'Linear' : `Moving Average (${meta.params.smoothingMonths}mo)`}</span></div>
              </div>
            )}
          </section>

          {/* Spatial */}
          <section>
            <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-wide mb-2">Spatial</h3>
            {opts ? (
              <div className="space-y-0">
                <div className={rowCls}><span className={labelCls}>Method</span><span className={valueCls}>{opts.spatial.method === 'kriging' ? 'Kriging' : opts.spatial.method === 'mc' ? 'MC (Low-Rank)' : 'IDW'}</span></div>
                <div className={rowCls}><span className={labelCls}>Resolution</span><span className={valueCls}>{opts.spatial.resolution} cols</span></div>
                {opts.spatial.method === 'kriging' && (
                  <>
                    <div className={rowCls}><span className={labelCls}>Variogram</span><span className={valueCls}>{opts.spatial.kriging.variogramModel}</span></div>
                    <div className={rowCls}><span className={labelCls}>Nugget</span><span className={valueCls}>{opts.spatial.kriging.nugget ? 'Enabled' : 'Disabled'}</span></div>
                    <div className={rowCls}><span className={labelCls}>Range</span><span className={valueCls}>
                      {opts.spatial.kriging.rangeMode === 'auto' ? 'Auto (1/3 diagonal)' :
                        opts.spatial.kriging.rangeMode === 'custom' ? `${opts.spatial.kriging.rangeValue}m` :
                          `${opts.spatial.kriging.rangeValue}%`}
                    </span></div>
                  </>
                )}
                {opts.spatial.method === 'idw' && (
                  <>
                    <div className={rowCls}><span className={labelCls}>Exponent</span><span className={valueCls}>{opts.spatial.idw.exponent}</span></div>
                    <div className={rowCls}><span className={labelCls}>Nodal Function</span><span className={valueCls}>{opts.spatial.idw.nodalFunction}</span></div>
                    <div className={rowCls}><span className={labelCls}>Neighbors</span><span className={valueCls}>{opts.spatial.idw.neighborMode === 'all' ? 'All' : `Nearest ${opts.spatial.idw.neighborCount}`}</span></div>
                  </>
                )}
                {opts.spatial.method === 'mc' && (
                  <>
                    <div className={rowCls}><span className={labelCls}>Completion</span><span className={valueCls}>SoftImpute low-rank matrix completion</span></div>
                    <div className={rowCls}><span className={labelCls}>Projection</span><span className={valueCls}>IDW from completed well field</span></div>
                  </>
                )}
              </div>
            ) : (
              <div className="space-y-0">
                <div className={rowCls}><span className={labelCls}>Method</span><span className={valueCls}>Kriging (Gaussian)</span></div>
                <div className={rowCls}><span className={labelCls}>Resolution</span><span className={valueCls}>{meta.params.resolution} cols</span></div>
              </div>
            )}
          </section>

          {/* General options */}
          {opts && (opts.general.truncateLow || opts.general.truncateHigh || opts.general.logInterpolation) && (
            <section>
              <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-wide mb-2">Post-Processing</h3>
              <div className="space-y-0">
                {opts.general.truncateLow && (
                  <div className={rowCls}><span className={labelCls}>Truncate Low</span><span className={valueCls}>{opts.general.truncateLowValue}</span></div>
                )}
                {opts.general.truncateHigh && (
                  <div className={rowCls}><span className={labelCls}>Truncate High</span><span className={valueCls}>{opts.general.truncateHighValue}</span></div>
                )}
                {opts.general.logInterpolation && (
                  <div className={rowCls}><span className={labelCls}>Log Transform</span><span className={valueCls}>Enabled</span></div>
                )}
              </div>
            </section>
          )}
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

export default RasterInfoDialog;
