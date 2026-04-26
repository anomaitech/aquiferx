
import React, { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import { Layers, Map as MapIcon, Database, ChevronRight, Activity, Upload, Loader2, Download, Table, BarChart3, Maximize2, X } from 'lucide-react';
import { Region, Aquifer, Well, Measurement, DataType, RasterAnalysisResult, RasterAnalysisMeta, CrossSectionProfile, ImputationModelResult, ImputationModelMeta } from './types';
import { loadAllData } from './services/dataLoader';
import { freshFetch } from './services/importUtils';
import { slugify } from './utils/strings';
import MapView, { MapViewHandle } from './components/MapView';
import Sidebar from './components/Sidebar';
import TimeSeriesChart from './components/TimeSeriesChart';
import ImportDataHub from './components/import/ImportDataHub';
import DataEditor from './components/DataEditor';
import SpatialAnalysisDialog from './components/SpatialAnalysisDialog';
import ImputationWizard from './components/ImputationWizard';
import ModelTimeSeries from './components/ModelTimeSeries';
import { fetchGldasDateRange } from './services/gldasFetch';
import RasterInfoDialog from './components/RasterInfoDialog';
import ModelInfoDialog from './components/ModelInfoDialog';
import RasterOverlay from './components/RasterOverlay';
import CrossSectionChart from './components/CrossSectionChart';
import { computeStorageChange } from './utils/storageVolume';
import RasterStatsChart from './components/RasterStatsChart';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts';
import JSZip from 'jszip';

const TREND_THRESHOLDS_FT = { extreme: 2.0, moderate: 0.5 };
const TREND_THRESHOLDS_M = { extreme: 0.6, moderate: 0.15 };
const AQUIFER_TREND_THRESHOLDS_FT = { extreme: 1.0, moderate: 0.25 };
const AQUIFER_TREND_THRESHOLDS_M = { extreme: 0.3, moderate: 0.075 };

type TrendThresholds = { extreme: number; moderate: number };

const TREND_CATEGORIES: { label: string; color: string; test: (s: number, t: TrendThresholds) => boolean }[] = [
  { label: 'Extreme Decline', color: '#DC2626', test: (s, t) => s < -t.extreme },
  { label: 'Decline', color: '#FB923C', test: (s, t) => s < -t.moderate },
  { label: 'Static', color: '#FACC15', test: (s, t) => s <= t.moderate },
  { label: 'Increase', color: '#38BDF8', test: (s, t) => s <= t.extreme },
  { label: 'Extreme Increase', color: '#2563EB', test: () => true },
];

const INSUFFICIENT_COLOR = '#1E293B';
const MS_PER_YEAR = 365.25 * 86400000;
const STORAGE_COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'];

function computeSlope(meas: Measurement[]): number | null {
  if (meas.length < 2) return null;
  let n = 0, sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
  for (const m of meas) {
    const x = new Date(m.date).getTime() / MS_PER_YEAR;
    const y = m.value;
    n++; sumX += x; sumY += y; sumXY += x * y; sumX2 += x * x;
  }
  const denom = n * sumX2 - sumX * sumX;
  if (denom === 0) return null;
  return (n * sumXY - sumX * sumY) / denom;
}

function classifySlope(slope: number | null, thresholds: TrendThresholds): string {
  if (slope === null) return INSUFFICIENT_COLOR;
  for (const cat of TREND_CATEGORIES) {
    if (cat.test(slope, thresholds)) return cat.color;
  }
  return INSUFFICIENT_COLOR;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function getRasterMethodSummary(result: RasterAnalysisResult | null): string {
  const opts = result?.options;
  if (!opts) return '';
  const temporal = opts.temporal.method === 'pchip'
    ? 'PCHIP'
    : opts.temporal.method === 'linear'
      ? 'Linear'
      : opts.temporal.method === 'moving-average'
        ? `MA (${opts.temporal.maWindow}mo)`
        : opts.temporal.method === 'model-direct'
          ? `Imputed (${opts.temporal.modelCode || 'model'})`
          : opts.temporal.method === 'model-mavg'
            ? `Imputed MA (${opts.temporal.maWindow}mo, ${opts.temporal.modelCode || 'model'})`
            : `Imputed (${opts.temporal.modelCode || 'model'})`;
  const spatial = opts.spatial.method === 'kriging'
    ? `Kriging (${opts.spatial.kriging.variogramModel})`
    : opts.spatial.method === 'mc'
      ? 'MC (Low-Rank)'
      : `IDW (p=${opts.spatial.idw.exponent})`;
  return `${temporal} -> ${spatial}`;
}

// --- Draggable / resizable floating window for expanded chart ---
const MIN_WIN_W = 480;
const MIN_WIN_H = 320;

const ExpandedChartWindow: React.FC<{
  onClose: () => void;
  title: string;
  subtitle: string;
  children: React.ReactNode;
}> = ({ onClose, title, subtitle, children }) => {
  const winRef = useRef<HTMLDivElement>(null);

  // Initialise centered at 80% viewport
  const initRect = useRef({
    x: Math.round(window.innerWidth * 0.1),
    y: Math.round(window.innerHeight * 0.05),
    w: Math.round(window.innerWidth * 0.8),
    h: Math.round(window.innerHeight * 0.85),
  });

  // --- Title-bar drag to move ---
  const handleTitleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    e.preventDefault();
    const el = winRef.current;
    if (!el) return;
    const startX = e.clientX, startY = e.clientY;
    const startLeft = el.offsetLeft, startTop = el.offsetTop;

    const onMove = (me: MouseEvent) => {
      const nx = Math.max(0, Math.min(window.innerWidth - 100, startLeft + me.clientX - startX));
      const ny = Math.max(0, Math.min(window.innerHeight - 40, startTop + me.clientY - startY));
      el.style.left = `${nx}px`;
      el.style.top = `${ny}px`;
    };
    const onUp = () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  };

  // --- Corner / edge resize ---
  const handleResizeMouseDown = (e: React.MouseEvent, edgeX: -1 | 0 | 1, edgeY: -1 | 0 | 1) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    const el = winRef.current;
    if (!el) return;
    const startMX = e.clientX, startMY = e.clientY;
    const startL = el.offsetLeft, startT = el.offsetTop, startW = el.offsetWidth, startH = el.offsetHeight;

    const onMove = (me: MouseEvent) => {
      const dx = me.clientX - startMX, dy = me.clientY - startMY;
      let newL = startL, newT = startT, newW = startW, newH = startH;

      if (edgeX === 1) newW = Math.max(MIN_WIN_W, startW + dx);
      if (edgeX === -1) { newW = Math.max(MIN_WIN_W, startW - dx); newL = startL + startW - newW; }
      if (edgeY === 1) newH = Math.max(MIN_WIN_H, startH + dy);
      if (edgeY === -1) { newH = Math.max(MIN_WIN_H, startH - dy); newT = startT + startH - newH; }

      el.style.left = `${newL}px`;
      el.style.top = `${newT}px`;
      el.style.width = `${newW}px`;
      el.style.height = `${newH}px`;
    };
    const onUp = () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  };

  const r = initRect.current;
  return (
    <div
      ref={winRef}
      className="fixed z-[100] bg-white rounded-lg shadow-2xl border border-slate-300 flex flex-col overflow-hidden"
      style={{ left: r.x, top: r.y, width: r.w, height: r.h }}
    >
      {/* Title bar — drag to move */}
      <div
        className="flex items-center justify-between px-4 py-2 border-b border-slate-200 bg-slate-50 cursor-move select-none flex-shrink-0"
        onMouseDown={handleTitleMouseDown}
      >
        <div className="flex items-center space-x-3 min-w-0">
          <Activity size={16} className="text-blue-500 flex-shrink-0" />
          <span className="font-semibold text-sm text-slate-800 truncate">{title}</span>
          <span className="text-[11px] text-slate-400 flex-shrink-0">{subtitle}</span>
        </div>
        <button
          onClick={onClose}
          onMouseDown={e => e.stopPropagation()}
          className="p-1 text-slate-400 hover:text-slate-600 hover:bg-slate-200 rounded transition-colors flex-shrink-0"
          title="Close"
        >
          <X size={16} />
        </button>
      </div>

      {/* Chart body */}
      <div className="flex-1 min-h-0 p-3">
        {children}
      </div>

      {/* Resize handles (edges + corners) */}
      {/* right */}
      <div className="absolute top-0 right-0 w-1.5 h-full cursor-ew-resize" onMouseDown={e => handleResizeMouseDown(e, 1, 0)} />
      {/* bottom */}
      <div className="absolute bottom-0 left-0 h-1.5 w-full cursor-ns-resize" onMouseDown={e => handleResizeMouseDown(e, 0, 1)} />
      {/* left */}
      <div className="absolute top-0 left-0 w-1.5 h-full cursor-ew-resize" onMouseDown={e => handleResizeMouseDown(e, -1, 0)} />
      {/* top */}
      <div className="absolute top-0 left-0 h-1.5 w-full cursor-ns-resize" onMouseDown={e => handleResizeMouseDown(e, 0, -1)} />
      {/* bottom-right corner */}
      <div className="absolute bottom-0 right-0 w-3 h-3 cursor-nwse-resize" onMouseDown={e => handleResizeMouseDown(e, 1, 1)} />
      {/* bottom-left corner */}
      <div className="absolute bottom-0 left-0 w-3 h-3 cursor-nesw-resize" onMouseDown={e => handleResizeMouseDown(e, -1, 1)} />
      {/* top-right corner */}
      <div className="absolute top-0 right-0 w-3 h-3 cursor-nesw-resize" onMouseDown={e => handleResizeMouseDown(e, 1, -1)} />
      {/* top-left corner */}
      <div className="absolute top-0 left-0 w-3 h-3 cursor-nwse-resize" onMouseDown={e => handleResizeMouseDown(e, -1, -1)} />
    </div>
  );
};

const App: React.FC = () => {

  const [regions, setRegions] = useState<Region[]>([]);
  const [aquifers, setAquifers] = useState<Aquifer[]>([]);
  const [wells, setWells] = useState<Well[]>([]);
  const [measurements, setMeasurements] = useState<Measurement[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [selectedRegion, setSelectedRegion] = useState<Region | null>(null);
  const [selectedAquifer, setSelectedAquifer] = useState<Aquifer | null>(null);
  const [selectedWells, setSelectedWells] = useState<Well[]>([]);
  const [isDataManagerOpen, setIsDataManagerOpen] = useState(false);
  const [isDataEditorOpen, setIsDataEditorOpen] = useState(false);
  const [showGSE, setShowGSE] = useState(false);
  const [showTrendLine, setShowTrendLine] = useState(false);
  const [showSmooth, setShowSmooth] = useState(false);
  const [usePCHIP, setUsePCHIP] = useState(true);
  const [smoothMonths, setSmoothMonths] = useState(3);
  const [trendColors, setTrendColors] = useState<Map<string, string> | null>(null);
  const [aquiferTrendColors, setAquiferTrendColors] = useState<Map<string, string> | null>(null);
  const [showTrends, setShowTrends] = useState(false);
  const [showWellsOnMap, setShowWellsOnMap] = useState(true);
  const [mapMinObs, setMapMinObs] = useState(5);
  const [dateFilter, setDateFilter] = useState<{ minYear: number; maxYear: number } | null>(null);
  const [showWellIdsOnMap, setShowWellIdsOnMap] = useState(false);
  const [trendWindowYears, setTrendWindowYears] = useState(30);
  const [selectedDataType, setSelectedDataType] = useState<string>('wte');
  const [visibleRegionIds, setVisibleRegionIds] = useState<Set<string>>(new Set());
  const [rasterDialogOpen, setRasterDialogOpen] = useState(false);
  const [rasterResult, setRasterResult] = useState<RasterAnalysisResult | null>(null);
  const [compareRasterResults, setCompareRasterResults] = useState<RasterAnalysisResult[]>([]);
  const [rasterMeta, setRasterMeta] = useState<RasterAnalysisMeta[]>([]);
  const [loadingRasterCode, setLoadingRasterCode] = useState<string | null>(null);
  const [modelMeta, setModelMeta] = useState<ImputationModelMeta[]>([]);
  const [imputationDialogOpen, setImputationDialogOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState<ImputationModelResult | null>(null);
  const [gldasDateRange, setGldasDateRange] = useState<{ min: string; max: string } | null>(null);
  const [showCombinedModel, setShowCombinedModel] = useState(false);
  const [crossSectionProfile, setCrossSectionProfile] = useState<CrossSectionProfile | null>(null);
  const [activeTimeSeriesTab, setActiveTimeSeriesTab] = useState<'waterLevel' | 'storageChange' | 'rasterStats' | 'crossSection'>('waterLevel');
  const [rasterFrameDate, setRasterFrameDate] = useState<{ date: string; dateTs: number } | null>(null);
  const [showActiveWells, setShowActiveWells] = useState(false);
  const [storageCoeff, setStorageCoeff] = useState(0.1);
  const [storageVolumeUnit, setStorageVolumeUnit] = useState('');
  const [chartHeight, setChartHeight] = useState(250);
  const [isChartExpanded, setIsChartExpanded] = useState(false);
  const isDraggingDividerRef = useRef(false);
  const dragStartYRef = useRef(0);
  const dragStartHeightRef = useRef(0);
  const mapViewRef = useRef<MapViewHandle>(null);
  const splitContainerRef = useRef<HTMLDivElement>(null);
  const chartPanelRef = useRef<HTMLDivElement>(null);

  // Refs for async handlers to avoid stale closures after await
  const selectedAquiferRef = useRef(selectedAquifer);
  selectedAquiferRef.current = selectedAquifer;
  const selectedDataTypeRef = useRef(selectedDataType);
  selectedDataTypeRef.current = selectedDataType;
  // Guards: prevent the aquifer-change effect from clearing state that a
  // concurrent raster/model load is about to set
  const loadingRasterRef = useRef(false);
  const loadingModelRef = useRef(false);

  // Divider drag handlers — direct DOM manipulation during drag to avoid full re-renders
  const handleDividerMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isDraggingDividerRef.current = true;
    dragStartYRef.current = e.clientY;
    dragStartHeightRef.current = chartHeight;

    const onMove = (me: MouseEvent) => {
      if (!isDraggingDividerRef.current) return;
      const dy = dragStartYRef.current - me.clientY;
      const containerH = splitContainerRef.current?.clientHeight || 600;
      const newHeight = Math.min(Math.max(100, dragStartHeightRef.current + dy), containerH - 100);
      if (chartPanelRef.current) chartPanelRef.current.style.height = `${newHeight}px`;
    };

    const onUp = (me: MouseEvent) => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      isDraggingDividerRef.current = false;
      const dy = dragStartYRef.current - me.clientY;
      const containerH = splitContainerRef.current?.clientHeight || 600;
      const finalHeight = Math.min(Math.max(100, dragStartHeightRef.current + dy), containerH - 100);
      setChartHeight(finalHeight);
    };

    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  }, [chartHeight]);

  // Keep visibleRegionIds in sync when regions load/change
  useEffect(() => {
    setVisibleRegionIds(prev => {
      const updated = new Set(prev);
      for (const r of regions) {
        if (!prev.has(r.id)) updated.add(r.id);
      }
      return updated.size !== prev.size ? updated : prev;
    });
  }, [regions]);

  const toggleRegionVisibility = (id: string) => {
    setVisibleRegionIds(prev => {
      const updated = new Set(prev);
      if (updated.has(id)) updated.delete(id);
      else updated.add(id);
      return updated;
    });
  };

  // Reset selectedDataType when region changes
  useEffect(() => {
    setSelectedDataType('wte');
  }, [selectedRegion?.id]);

  // Clear trend analysis when region or data type changes
  useEffect(() => {
    setTrendColors(null);
    setAquiferTrendColors(null);
    setShowTrends(false);
  }, [selectedRegion?.id, selectedDataType]);

  // Clear active raster overlay and model when aquifer changes — but not
  // when the change was triggered by an in-flight raster/model load that is
  // about to set new results (otherwise the effect races against the load).
  useEffect(() => {
    if (!loadingRasterRef.current) setRasterResult(null);
    if (!loadingModelRef.current) setSelectedModel(null);
  }, [selectedAquifer?.id]);

  // Auto-switch time series tab based on raster result
  useEffect(() => {
    if (rasterResult) {
      if (rasterResult.dataType === 'wte') {
        setActiveTimeSeriesTab('storageChange');
      } else if (rasterResult.stats) {
        setActiveTimeSeriesTab('rasterStats');
      }
    } else {
      setActiveTimeSeriesTab('waterLevel');
      setRasterFrameDate(null);
      setCrossSectionProfile(null);
    }
    setCompareRasterResults([]);
  }, [rasterResult]);

  // Auto-switch to cross-section tab when profile is set/cleared.
  // Reads rasterResult only to pick a fallback tab on dismiss — intentionally
  // excluded from deps so a raster change doesn't yank the user off this tab.
  useEffect(() => {
    if (crossSectionProfile) {
      setActiveTimeSeriesTab('crossSection');
    } else if (activeTimeSeriesTab === 'crossSection') {
      const rr = rasterResult;
      setActiveTimeSeriesTab(rr?.dataType === 'wte' ? 'storageChange' : rr?.stats ? 'rasterStats' : 'waterLevel');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [crossSectionProfile]);

  // Compute the data time range (in years) for the selected region/data type
  const dataTimeRangeYears = useMemo(() => {
    if (!selectedRegion) return 0;
    let minT = Infinity;
    for (const m of measurements) {
      if (m.dataType !== selectedDataType) continue;
      const t = new Date(m.date).getTime();
      if (!isNaN(t) && t < minT) minT = t;
    }
    if (!isFinite(minT)) return 0;
    return (Date.now() - minT) / MS_PER_YEAR;
  }, [selectedRegion, measurements, selectedDataType]);

  const trendWindowMax = Math.max(5, Math.ceil(dataTimeRangeYears / 5) * 5);

  const computeTrendColors = useCallback((windowYears: number) => {
    if (!selectedRegion) return;

    const cutoffTime = Date.now() - windowYears * MS_PER_YEAR;
    const regionWells = wells.filter(w => w.regionId === selectedRegion.id);

    // Group measurements by regionId:aquiferId:wellId for this data type, filtered by window
    const byWell = new Map<string, Measurement[]>();
    for (const m of measurements) {
      if (m.dataType === selectedDataType) {
        const t = new Date(m.date).getTime();
        if (!isNaN(t) && t >= cutoffTime) {
          const key = `${m.regionId}:${m.aquiferId}:${m.wellId}`;
          const arr = byWell.get(key);
          if (arr) arr.push(m);
          else byWell.set(key, [m]);
        }
      }
    }

    // Per-well colors
    const wellThresholds = selectedRegion.lengthUnit === 'm' ? TREND_THRESHOLDS_M : TREND_THRESHOLDS_FT;
    const wellColorMap = new Map<string, string>();
    for (const w of regionWells) {
      wellColorMap.set(w.id, classifySlope(computeSlope(byWell.get(`${w.regionId}:${w.aquiferId}:${w.id}`) || []), wellThresholds));
    }

    // Per-aquifer colors (median of well slopes)
    const aqThresholds = selectedRegion.lengthUnit === 'm' ? AQUIFER_TREND_THRESHOLDS_M : AQUIFER_TREND_THRESHOLDS_FT;
    const wellsByAquifer = new Map<string, Well[]>();
    for (const w of regionWells) {
      const arr = wellsByAquifer.get(w.aquiferId);
      if (arr) arr.push(w);
      else wellsByAquifer.set(w.aquiferId, [w]);
    }
    const regionAquifers = aquifers.filter(a => a.regionId === selectedRegion.id);
    const aqColorMap = new Map<string, string>();
    for (const a of regionAquifers) {
      const aqWells = wellsByAquifer.get(a.id) || [];
      const slopes: number[] = [];
      for (const w of aqWells) {
        const s = computeSlope(byWell.get(`${w.regionId}:${w.aquiferId}:${w.id}`) || []);
        if (s !== null) slopes.push(s);
      }
      if (slopes.length === 0) {
        aqColorMap.set(a.id, INSUFFICIENT_COLOR);
      } else {
        aqColorMap.set(a.id, classifySlope(median(slopes), aqThresholds));
      }
    }

    setTrendColors(wellColorMap);
    setAquiferTrendColors(aqColorMap);
  }, [selectedRegion, wells, aquifers, measurements, selectedDataType]);

  const analyzeTrends = () => {
    if (showTrends) {
      setTrendColors(null);
      setAquiferTrendColors(null);
      setShowTrends(false);
      setShowTrendLine(false);
      return;
    }
    if (!selectedRegion) return;
    setShowTrendLine(true);
    computeTrendColors(trendWindowYears);
    setShowTrends(true);
  };

  const handleTrendWindowChange = (newWindow: number) => {
    setTrendWindowYears(newWindow);
    computeTrendColors(newWindow);
  };

  // Load data on mount
  useEffect(() => {
    const loadData = async () => {
      try {
        setIsLoading(true);
        setLoadError(null);
        const data = await loadAllData();
        setRegions(data.regions);
        setAquifers(data.aquifers);
        setWells(data.wells);
        setMeasurements(data.measurements);
        setRasterMeta(data.storageMeta);
        setModelMeta(data.modelMeta);
        console.log(`Loaded: ${data.regions.length} regions, ${data.aquifers.length} aquifers, ${data.wells.length} wells, ${data.measurements.length} measurements, ${data.storageMeta.length} raster analyses, ${data.modelMeta.length} models`);
      } catch (e) {
        console.error('Failed to load data:', e);
        setLoadError(e instanceof Error ? e.message : 'Failed to load data');
      } finally {
        setIsLoading(false);
      }
    };
    loadData();
  }, []);

  // Callback for ImportDataHub / DataManager to refresh data
  const handleDataChanged = async () => {
    try {
      const data = await loadAllData();
      setRegions(data.regions);
      setAquifers(data.aquifers);
      setWells(data.wells);
      setMeasurements(data.measurements);
      setRasterMeta(data.storageMeta);
      setModelMeta(data.modelMeta);
      // Re-resolve the current selection against the fresh data so that
      // derived state like effectiveDataTypes / filteredWells picks up
      // changes from the import (new wells, new data types, etc).
      if (selectedRegion) {
        const updatedRegion = data.regions.find(r => r.id === selectedRegion.id);
        if (updatedRegion) setSelectedRegion(updatedRegion);
      }
      if (selectedAquifer) {
        const updatedAquifer = data.aquifers.find(
          a => a.id === selectedAquifer.id && a.regionId === selectedAquifer.regionId
        );
        if (updatedAquifer) setSelectedAquifer(updatedAquifer);
      }
    } catch (e) {
      console.error('Failed to reload data:', e);
    }
  };

  // Load full raster analysis from file
  const handleLoadRaster = async (meta: RasterAnalysisMeta) => {
    loadingRasterRef.current = true;
    setLoadingRasterCode(meta.code);
    try {
      const res = await fetch(`/data/${meta.filePath}`);
      if (res.ok) {
        const fullResult: RasterAnalysisResult = await res.json();
        // Select the aquifer if not already selected (use ref for current value)
        if (!selectedAquiferRef.current || selectedAquiferRef.current.id !== meta.aquiferId) {
          const aq = aquifers.find(a => a.id === meta.aquiferId && a.regionId === meta.regionId);
          if (aq) setSelectedAquifer(aq);
        }
        setRasterResult(fullResult);
        // Sync data type selector to match the raster's data type
        if (meta.dataType !== selectedDataTypeRef.current) {
          setSelectedDataType(meta.dataType);
        }
      }
    } catch (e) {
      console.error('Failed to load raster analysis:', e);
    } finally {
      loadingRasterRef.current = false;
      setLoadingRasterCode(null);
    }
  };

  const handleRasterFrameChange = useCallback((date: string, dateTs: number) => {
    setRasterFrameDate(prev => {
      if (prev && prev.date === date && prev.dateTs === dateTs) return prev;
      return { date, dateTs };
    });
  }, []);

  const handleCrossSectionChange = useCallback((profile: CrossSectionProfile | null) => {
    setCrossSectionProfile(profile);
  }, []);

  const handleUnloadRaster = useCallback(() => {
    setRasterResult(null);
    setShowActiveWells(false);
  }, []);

  const handleToggleActiveWells = useCallback(() => {
    setShowActiveWells(v => !v);
  }, []);

  const handleToggleCompareRaster = async (meta: RasterAnalysisMeta) => {
    if (rasterResult && rasterResult.code === meta.code && rasterResult.dataType === meta.dataType && rasterResult.regionId === meta.regionId) return;
    const existingIdx = compareRasterResults.findIndex(r => r.code === meta.code && r.dataType === meta.dataType && r.regionId === meta.regionId);
    if (existingIdx >= 0) {
      setCompareRasterResults(prev => prev.filter((_, i) => i !== existingIdx));
      return;
    }
    try {
      const res = await fetch(`/data/${meta.filePath}`);
      if (res.ok) {
        const fullResult: RasterAnalysisResult = await res.json();
        setCompareRasterResults(prev => [...prev, fullResult]);
      }
    } catch (e) {
      console.error('Failed to load raster for comparison:', e);
    }
  };

  const handleDeleteRaster = async (meta: RasterAnalysisMeta) => {
    // Unload if this is the active raster
    if (rasterResult?.code === meta.code && rasterResult?.dataType === meta.dataType) {
      setRasterResult(null);
    }
    // Remove from metadata list
    setRasterMeta(prev => prev.filter(m => !(m.code === meta.code && m.dataType === meta.dataType && m.regionId === meta.regionId)));
    // Delete the file on disk
    try {
      await fetch('/api/delete-file', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filePath: meta.filePath }),
      });
    } catch (e) {
      console.error('Failed to delete raster analysis file:', e);
    }
  };

  const handleRenameRaster = async (meta: RasterAnalysisMeta, newTitle: string) => {
    const newCode = slugify(newTitle);
    const oldPath = meta.filePath;
    // Build new file path: same directory, new code in filename
    const dir = oldPath.substring(0, oldPath.lastIndexOf('/'));
    const newPath = `${dir}/raster_${meta.dataType}_${newCode}.json`;

    try {
      const res = await fetch('/api/rename-raster', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ oldPath, newPath, newCode, newTitle }),
      });
      if (res.ok) {
        setRasterMeta(prev => prev.map(m =>
          m.code === meta.code && m.dataType === meta.dataType && m.regionId === meta.regionId
            ? { ...m, title: newTitle, code: newCode, filePath: newPath }
            : m
        ));
        // Update active raster if it was the renamed one
        if (rasterResult?.code === meta.code && rasterResult?.dataType === meta.dataType && rasterResult?.regionId === meta.regionId) {
          setRasterResult(prev => prev ? { ...prev, title: newTitle, code: newCode } : null);
        }
      }
    } catch (e) {
      console.error('Failed to rename raster:', e);
    }
  };

  const [rasterInfoMeta, setRasterInfoMeta] = useState<RasterAnalysisMeta | null>(null);
  const handleGetRasterInfo = (meta: RasterAnalysisMeta) => {
    setRasterInfoMeta(meta);
  };

  const [modelInfoMeta, setModelInfoMeta] = useState<ImputationModelMeta | null>(null);
  const handleGetModelInfo = (meta: ImputationModelMeta) => {
    setModelInfoMeta(meta);
  };

  // --- Imputation model handlers ---
  const handleLoadModel = async (meta: ImputationModelMeta) => {
    loadingModelRef.current = true;
    try {
      const res = await fetch(`/data/${meta.filePath}`);
      if (res.ok) {
        const fullResult: ImputationModelResult = await res.json();
        if (!selectedAquiferRef.current || selectedAquiferRef.current.id !== meta.aquiferId) {
          const aq = aquifers.find(a => a.id === meta.aquiferId && a.regionId === meta.regionId);
          if (aq) setSelectedAquifer(aq);
        }
        setSelectedModel(fullResult);
        // Sync data type selector to match the model's data type
        if (meta.dataType !== selectedDataTypeRef.current) {
          setSelectedDataType(meta.dataType);
        }
      }
    } catch (e) {
      console.error('Failed to load model:', e);
    } finally {
      loadingModelRef.current = false;
    }
  };

  const handleUnloadModel = () => {
    setSelectedModel(null);
  };

  const handleDeleteModel = async (meta: ImputationModelMeta) => {
    if (selectedModel?.code === meta.code) {
      setSelectedModel(null);
    }
    setModelMeta(prev => prev.filter(m => !(m.code === meta.code && m.regionId === meta.regionId)));
    try {
      await fetch('/api/delete-file', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filePath: meta.filePath }),
      });
    } catch (e) {
      console.error('Failed to delete model file:', e);
    }
  };

  const handleRenameModel = async (meta: ImputationModelMeta, newTitle: string) => {
    const newCode = slugify(newTitle);
    const oldPath = meta.filePath;
    const dir = oldPath.substring(0, oldPath.lastIndexOf('/'));
    const newPath = `${dir}/model_wte_${newCode}.json`;

    try {
      const res = await fetch('/api/rename-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ oldPath, newPath, newCode, newTitle }),
      });
      if (res.ok) {
        setModelMeta(prev => prev.map(m =>
          m.code === meta.code && m.regionId === meta.regionId
            ? { ...m, title: newTitle, code: newCode, filePath: newPath }
            : m
        ));
        if (selectedModel?.code === meta.code && selectedModel?.regionId === meta.regionId) {
          setSelectedModel(prev => prev ? { ...prev, title: newTitle, code: newCode } : null);
        }
      }
    } catch (e) {
      console.error('Failed to rename model:', e);
    }
  };

  const handleOpenImputationWizard = async () => {
    // Pre-fetch GLDAS date range
    try {
      const range = await fetchGldasDateRange();
      setGldasDateRange(range);
    } catch (e) {
      console.warn('Could not pre-fetch GLDAS date range:', e);
    }
    setImputationDialogOpen(true);
  };

  // Active data type object
  const activeDataType = useMemo<DataType>(() => {
    if (selectedRegion) {
      const dt = selectedRegion.effectiveDataTypes.find(d => d.code === selectedDataType);
      if (dt) return dt;
    }
    return { code: 'wte', name: 'Water Table Elevation', unit: 'ft' };
  }, [selectedRegion, selectedDataType]);

  // Filtered views
  const filteredAquifers = useMemo(() =>
    selectedRegion ? aquifers.filter(a => a.regionId === selectedRegion.id) : [],
  [selectedRegion, aquifers]);

  const filteredWells = useMemo(() =>
    selectedAquifer ? wells.filter(w => w.aquiferId === selectedAquifer.id && w.regionId === selectedAquifer.regionId) : [],
  [selectedAquifer, wells]);

  const selectedWellKeys = useMemo(() =>
    new Set(selectedWells.map(w => `${w.regionId}:${w.aquiferId}:${w.id}`)),
  [selectedWells]);

  const selectedWellMeasurements = useMemo(() =>
    selectedWellKeys.size > 0
      ? measurements.filter(m => m.dataType === selectedDataType && selectedWellKeys.has(`${m.regionId}:${m.aquiferId}:${m.wellId}`))
      : [],
  [selectedWellKeys, measurements, selectedDataType]);

  // Precompute eligible well time ranges once (stable across frames)
  const wellTimeRanges = useMemo<Map<string, [number, number]> | null>(() => {
    if (!showActiveWells || !rasterResult) return null;

    const opts = rasterResult.options;
    const dataType = rasterResult.dataType;
    const minObs = opts?.temporal.minObservations ?? 2;
    const minSpanYears = opts?.temporal.minTimeSpan ?? 0;

    const ranges = new Map<string, [number, number]>();

    for (const w of filteredWells) {
      const times: number[] = [];
      for (const m of measurements) {
        if (m.wellId !== w.id || m.regionId !== w.regionId || m.aquiferId !== w.aquiferId || m.dataType !== dataType) continue;
        const t = new Date(m.date).getTime();
        if (!isNaN(t)) times.push(t);
      }
      if (times.length < Math.max(2, minObs)) continue;
      times.sort((a, b) => a - b);
      const minT = times[0];
      const maxT = times[times.length - 1];
      const spanYears = (maxT - minT) / (365.25 * 24 * 60 * 60 * 1000);
      if (spanYears < minSpanYears) continue;
      ranges.set(w.id, [minT, maxT]);
    }
    return ranges;
  }, [showActiveWells, rasterResult, filteredWells, measurements]);

  // Cheap per-frame check: just test frame date against precomputed ranges
  const rasterActiveWellIds = useMemo<Set<string> | null>(() => {
    if (!wellTimeRanges || !rasterFrameDate) return null;
    const frameTs = rasterFrameDate.dateTs;
    const active = new Set<string>();
    for (const [wellId, [minT, maxT]] of wellTimeRanges) {
      if (frameTs >= minT && frameTs <= maxT) active.add(wellId);
    }
    return active;
  }, [wellTimeRanges, rasterFrameDate]);

  const allRasterResults = useMemo(() => {
    if (!rasterResult) return [];
    return [rasterResult, ...compareRasterResults];
  }, [rasterResult, compareRasterResults]);

  // Time range of the active raster (for chart X-axis override)
  const rasterTimeRange = useMemo<[number, number] | undefined>(() => {
    if (!rasterResult || rasterResult.frames.length === 0) return undefined;
    const first = new Date(rasterResult.frames[0].date).getTime();
    const last = new Date(rasterResult.frames[rasterResult.frames.length - 1].date).getTime();
    return [first, last];
  }, [rasterResult]);

  // Set default volume unit based on region when raster result loads
  useEffect(() => {
    if (rasterResult && selectedRegion) {
      setStorageVolumeUnit(prev => prev || (selectedRegion.lengthUnit === 'ft' ? 'acre-ft' : 'MCM'));
    }
  }, [rasterResult, selectedRegion]);

  // Reset volume unit when region changes
  useEffect(() => {
    if (selectedRegion) {
      setStorageVolumeUnit(selectedRegion.lengthUnit === 'ft' ? 'acre-ft' : 'MCM');
    }
  }, [selectedRegion?.id]);

  // Compute storage change on-the-fly for WTE rasters
  const storageChartData = useMemo(() => {
    if (allRasterResults.length === 0) return [];
    if (allRasterResults[0].dataType !== 'wte') return [];

    const lengthUnit = selectedRegion?.lengthUnit || 'ft';
    const allSeries = allRasterResults.map(r =>
      computeStorageChange(r, storageCoeff, storageVolumeUnit, lengthUnit)
    );

    const dateSet = new Set<number>();
    for (const series of allSeries) {
      for (const s of series) dateSet.add(new Date(s.date).getTime());
    }
    const dates = [...dateSet].sort((a, b) => a - b);

    const lookups = allSeries.map(series => {
      const map = new Map<number, number>();
      for (const s of series) map.set(new Date(s.date).getTime(), s.value);
      return map;
    });

    return dates.map(d => {
      const row: Record<string, number> = { date: d };
      for (let i = 0; i < lookups.length; i++) {
        const val = lookups[i].get(d);
        if (val !== undefined) row[`value_${i}`] = val;
      }
      return row;
    });
  }, [allRasterResults, storageCoeff, storageVolumeUnit, selectedRegion?.lengthUnit]);

  const handleWellClick = (well: Well, shiftKey: boolean) => {
    if (shiftKey) {
      setSelectedWells(prev =>
        prev.some(w => w.id === well.id)
          ? prev.filter(w => w.id !== well.id)
          : [...prev, well]
      );
    } else {
      setSelectedWells([well]);
    }
    setActiveTimeSeriesTab('waterLevel');
  };

  const handleWellBoxSelect = (wells: Well[]) => {
    setSelectedWells(wells);
    setActiveTimeSeriesTab('waterLevel');
  };

  // --- Region/Aquifer rename & delete handlers ---

  const handleEditRegion = async (regionId: string, newName: string, lengthUnit: 'ft' | 'm') => {
    const region = regions.find(r => r.id === regionId);
    if (!region) return;

    setRegions(prev => prev.map(r => r.id === regionId ? { ...r, name: newName, lengthUnit } : r));

    // Persist updated region.json (per-folder)
    const regionMeta = {
      id: regionId,
      name: newName,
      lengthUnit,
      singleUnit: region.singleUnit,
      customDataTypes: region.customDataTypes,
    };
    await fetch('/api/save-data', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ files: [{ path: `${regionId}/region.json`, content: JSON.stringify(regionMeta, null, 2) }] }),
    });
  };

  const handleDeleteRegion = async (regionId: string) => {
    // Delete folder on disk first — keep the row mounted so the sidebar can
    // show its deleting spinner until the backend confirms completion.
    try {
      const res = await fetch('/api/delete-folder', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder: regionId }),
      });
      if (!res.ok) throw new Error(`delete-folder failed: ${res.status}`);
    } catch (err) {
      console.error('Failed to delete region folder:', err);
      alert(`Failed to delete region. ${err instanceof Error ? err.message : ''}`);
      return;
    }
    // Clear selection if needed
    if (selectedRegion?.id === regionId) {
      setSelectedRegion(null);
      setSelectedAquifer(null);
      setSelectedWells([]);
    }
    // Remove from state
    const deletedWellIds = new Set(wells.filter(w => w.regionId === regionId).map(w => `${w.regionId}:${w.aquiferId}:${w.id}`));
    setRegions(prev => prev.filter(r => r.id !== regionId));
    setAquifers(prev => prev.filter(a => a.regionId !== regionId));
    setWells(prev => prev.filter(w => w.regionId !== regionId));
    setMeasurements(prev => prev.filter(m => !deletedWellIds.has(`${m.regionId}:${m.aquiferId}:${m.wellId}`)));
  };

  const handleDownloadRegion = async (regionId: string) => {
    const region = regions.find(r => r.id === regionId);
    if (!region) return;
    const basePath = `/data/${regionId}`;
    const zip = new JSZip();

    // Always include region.json and region.geojson
    const staticFiles = ['region.json', 'region.geojson', 'aquifers.geojson', 'wells.csv'];
    for (const name of staticFiles) {
      try {
        const res = await freshFetch(`${basePath}/${name}`);
        if (res.ok) {
          zip.file(name, await res.text());
        }
      } catch {
        // skip files that don't exist
      }
    }

    // Dynamically include all data_*.csv files
    for (const dt of region.effectiveDataTypes) {
      try {
        const res = await freshFetch(`${basePath}/data_${dt.code}.csv`);
        if (res.ok) {
          zip.file(`data_${dt.code}.csv`, await res.text());
        }
      } catch {
        // skip
      }
    }

    const blob = await zip.generateAsync({ type: 'blob' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${region.name.replace(/[^a-z0-9]+/gi, '_')}.zip`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleRenameAquifer = async (aquiferId: string, newName: string) => {
    setAquifers(prev => prev.map(a => a.id === aquiferId ? { ...a, name: newName } : a));
    // Rebuild and persist the aquifers.geojson for the affected region
    const aquifer = aquifers.find(a => a.id === aquiferId);
    if (!aquifer) return;
    const regionId = aquifer.regionId;
    const regionAquifers = aquifers
      .filter(a => a.regionId === regionId)
      .map(a => a.id === aquiferId ? { ...a, name: newName } : a);
    const features = regionAquifers.flatMap(a =>
      (a.geojson?.features || []).map((f: any) => ({
        ...f,
        properties: { ...f.properties, aquifer_id: a.id, aquifer_name: a.name },
      }))
    );
    const geojsonContent = JSON.stringify({ type: 'FeatureCollection', features }, null, 2);
    await fetch('/api/save-data', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ files: [{ path: `${regionId}/aquifers.geojson`, content: geojsonContent }] }),
    });
  };

  const handleDeleteAquifer = async (aquiferId: string) => {
    const aquifer = aquifers.find(a => a.id === aquiferId);
    if (!aquifer) return;
    const regionId = aquifer.regionId;
    const region = regions.find(r => r.id === regionId);
    // Clear selection if needed
    if (selectedAquifer?.id === aquiferId) {
      setSelectedAquifer(null);
      setSelectedWells([]);
    }
    // Compute remaining data for the region before removing from state
    const remainingAquifers = aquifers.filter(a => !(a.id === aquiferId && a.regionId === regionId));
    const remainingWells = wells.filter(w => !(w.aquiferId === aquiferId && w.regionId === regionId));
    const deletedWellKeys = new Set(wells.filter(w => w.aquiferId === aquiferId && w.regionId === regionId).map(w => `${w.regionId}:${w.aquiferId}:${w.id}`));
    const remainingMeasurements = measurements.filter(m => !deletedWellKeys.has(`${m.regionId}:${m.aquiferId}:${m.wellId}`));
    // Update state
    setAquifers(remainingAquifers);
    setWells(remainingWells);
    setMeasurements(remainingMeasurements);
    // Rebuild files for the region
    const regionAquifers = remainingAquifers.filter(a => a.regionId === regionId);
    const regionWells = remainingWells.filter(w => w.regionId === regionId);
    const regionMeasurements = remainingMeasurements.filter(m => regionWells.some(w => w.id === m.wellId && w.regionId === m.regionId && w.aquiferId === m.aquiferId));
    // Aquifers GeoJSON
    const features = regionAquifers.flatMap(a =>
      (a.geojson?.features || []).map((f: any) => ({
        ...f,
        properties: { ...f.properties, aquifer_id: a.id, aquifer_name: a.name },
      }))
    );
    const geojsonContent = JSON.stringify({ type: 'FeatureCollection', features }, null, 2);
    // Wells CSV
    const wellsCsvHeader = 'well_id,well_name,lat,long,gse,aquifer_id,aquifer_name';
    const wellsCsvRows = regionWells.map(w =>
      `${w.id},"${w.name}",${w.lat},${w.lng},${w.gse},${w.aquiferId},"${w.aquiferName}"`
    );
    const wellsCsvContent = [wellsCsvHeader, ...wellsCsvRows].join('\n');

    // Rebuild data CSVs per data type
    const regionDataTypes = region?.effectiveDataTypes || [{ code: 'wte', name: 'Water Table Elevation', unit: 'ft' }];
    const dataFiles: { path: string; content: string }[] = [];
    for (const dt of regionDataTypes) {
      const dtMeasurements = regionMeasurements.filter(m => m.dataType === dt.code);
      const header = 'well_id,well_name,date,value,aquifer_id';
      const rows = dtMeasurements.map(m =>
        `${m.wellId},"${m.wellName}",${m.date},${m.value},${m.aquiferId}`
      );
      dataFiles.push({ path: `${regionId}/data_${dt.code}.csv`, content: [header, ...rows].join('\n') });
    }

    await fetch('/api/save-data', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        files: [
          { path: `${regionId}/aquifers.geojson`, content: geojsonContent },
          { path: `${regionId}/wells.csv`, content: wellsCsvContent },
          ...dataFiles,
        ],
      }),
    });
  };

  // Export time series data to CSV
  const exportToCSV = () => {
    if (selectedWells.length === 0) return;

    // Model-aware export
    if (selectedModel && selectedWells.length === 1) {
      const modelRows = selectedModel.data
        .filter(r => r.well_id === selectedWells[0].id)
        .sort((a, b) => a.date.localeCompare(b.date));
      if (modelRows.length === 0) return;

      const headers = 'Date,PCHIP,ELM,Combined';
      const csvRows = modelRows.map(r => {
        const combined = r.pchip != null ? r.pchip : r.model;
        return `${r.date},${r.pchip ?? ''},${r.model ?? ''},${combined ?? ''}`;
      });

      const csvContent = [headers, ...csvRows].join('\n');

      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      const wellName = selectedWells[0].name.replace(/[^a-z0-9]/gi, '_');
      link.download = `${wellName}_model_${selectedModel.code}.csv`;
      link.click();
      URL.revokeObjectURL(url);
      return;
    }

    if (selectedWellMeasurements.length === 0) return;

    const unit = activeDataType.unit;
    const headers = ['Date', `${activeDataType.name} (${unit})`, 'Well Name', 'Aquifer ID'];
    const rows = selectedWellMeasurements
      .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
      .map(m => [
        new Date(m.date).toLocaleDateString(),
        m.value.toString(),
        m.wellName,
        m.aquiferId
      ]);

    const csvContent = [headers, ...rows]
      .map(row => row.map(cell => `"${cell}"`).join(','))
      .join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    const firstName = selectedWells[0].name.replace(/[^a-z0-9]/gi, '_');
    const suffix = selectedWells.length > 1 ? `_and_${selectedWells.length - 1}_others` : '';
    link.download = `${firstName}${suffix}_${activeDataType.code}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Chart inline edit/delete handlers
  const handleChartEditMeasurement = async (wellId: string, date: number, newValue: number) => {
    const well = selectedWells.find(w => w.id === wellId);
    const updatedMeasurements = measurements.map(m =>
      m.wellId === wellId && m.regionId === well?.regionId && m.aquiferId === well?.aquiferId && new Date(m.date).getTime() === date && m.dataType === selectedDataType
        ? { ...m, value: newValue }
        : m
    );
    await handleDataEditorSave(updatedMeasurements);
  };

  const handleChartDeleteMeasurement = async (wellId: string, date: number) => {
    const well = selectedWells.find(w => w.id === wellId);
    const updatedMeasurements = measurements.filter(m =>
      !(m.wellId === wellId && m.regionId === well?.regionId && m.aquiferId === well?.aquiferId && new Date(m.date).getTime() === date && m.dataType === selectedDataType)
    );
    await handleDataEditorSave(updatedMeasurements);
  };

  // Save handler for DataEditor
  const handleDataEditorSave = async (updatedMeasurements: Measurement[]) => {
    setMeasurements(updatedMeasurements);
    // Rebuild and persist the data CSV for the active data type in this region
    const regionId = selectedWells[0].regionId;
    const region = regions.find(r => r.id === regionId);
    const regionWells = wells.filter(w => w.regionId === regionId);
    // Write the CSV for the active data type
    const dtCode = selectedDataType;
    const regionMeasurements = updatedMeasurements.filter(m => m.regionId === regionId && m.dataType === dtCode);
    const header = 'well_id,well_name,date,value,aquifer_id';
    const rows = regionMeasurements.map(m =>
      `${m.wellId},"${m.wellName}",${m.date},${m.value},${m.aquiferId}`
    );
    const csvContent = [header, ...rows].join('\n');
    await fetch('/api/save-data', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ files: [{ path: `${regionId}/data_${dtCode}.csv`, content: csvContent }] }),
    });
  };

  // Loading state
  if (isLoading) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-slate-50">
        <div className="text-center">
          <Loader2 className="w-12 h-12 text-blue-600 animate-spin mx-auto mb-4" />
          <p className="text-slate-600 font-medium">Loading groundwater data...</p>
        </div>
      </div>
    );
  }

  // Error state
  if (loadError) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-slate-50">
        <div className="text-center max-w-md">
          <div className="text-red-500 text-6xl mb-4">!</div>
          <h2 className="text-xl font-bold text-slate-800 mb-2">Failed to Load Data</h2>
          <p className="text-slate-600 mb-4">{loadError}</p>
          <button
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  const hasMultipleDataTypes = selectedRegion && selectedRegion.effectiveDataTypes.length > 1;

  return (
    <div className="flex h-screen w-full overflow-hidden bg-slate-50 font-sans">
      {/* Sidebar */}
      <Sidebar
        regions={regions}
        selectedRegion={selectedRegion}
        setSelectedRegion={(r) => {
          setSelectedRegion(r);
          setSelectedWells([]);
          // For single-unit regions, auto-select the default aquifer
          if (r && r.singleUnit) {
            const singleAquifer = aquifers.find(a => a.regionId === r.id);
            setSelectedAquifer(singleAquifer || null);
          } else {
            setSelectedAquifer(null);
          }
        }}
        allAquifers={aquifers}
        selectedAquifer={selectedAquifer}
        setSelectedAquifer={(a) => {
          setSelectedAquifer(a);
          setSelectedWells([]);
        }}
        visibleRegionIds={visibleRegionIds}
        onToggleRegionVisibility={toggleRegionVisibility}
        onEditRegion={handleEditRegion}
        onDownloadRegion={handleDownloadRegion}
        onDeleteRegion={handleDeleteRegion}
        onRenameAquifer={handleRenameAquifer}
        onDeleteAquifer={handleDeleteAquifer}
        rasterMeta={rasterMeta}
        activeRasterCode={rasterResult ? `${rasterResult.dataType}_${rasterResult.code}` : null}
        compareRasterCodes={compareRasterResults.map(r => `${r.dataType}_${r.code}`)}
        loadingRasterCode={loadingRasterCode}
        onLoadRaster={handleLoadRaster}
        onUnloadRaster={handleUnloadRaster}
        onToggleCompareRaster={handleToggleCompareRaster}
        onDeleteRaster={handleDeleteRaster}
        onRenameRaster={handleRenameRaster}
        onGetRasterInfo={handleGetRasterInfo}
        modelMeta={modelMeta}
        activeModelCode={selectedModel?.code || null}
        onLoadModel={handleLoadModel}
        onUnloadModel={handleUnloadModel}
        onDeleteModel={handleDeleteModel}
        onRenameModel={handleRenameModel}
        onGetModelInfo={handleGetModelInfo}
      />

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col relative overflow-hidden">
        {/* Top Navigation / Breadcrumbs */}
        <header className="bg-white border-b border-slate-200 px-6 py-3 flex items-center justify-between shadow-sm z-10">
          <div className="flex items-center space-x-2 text-sm text-slate-600">
            <MapIcon size={16} />
            <button
              onClick={() => {
                setSelectedRegion(null);
                setSelectedAquifer(null);
                setSelectedWells([]);
              }}
              className="font-semibold text-slate-800 hover:text-blue-600 transition-colors"
            >
              Home
            </button>
            {selectedRegion && (
              <>
                <ChevronRight size={14} className="text-slate-400" />
                <button
                  onClick={() => {
                    if (selectedRegion.singleUnit) {
                      // Single-unit: just clear wells, keep aquifer auto-selected
                      setSelectedWells([]);
                    } else {
                      setSelectedAquifer(null);
                      setSelectedWells([]);
                    }
                  }}
                  className="hover:text-blue-600 transition-colors"
                >
                  {selectedRegion.name}
                </button>
              </>
            )}
            {selectedAquifer && !selectedRegion?.singleUnit && (
              <>
                <ChevronRight size={14} className="text-slate-400" />
                <button
                  onClick={() => setSelectedWells([])}
                  className="hover:text-blue-600 transition-colors"
                >
                  {selectedAquifer.name}
                </button>
              </>
            )}
            {selectedWells.length > 0 && (
              <>
                <ChevronRight size={14} className="text-slate-400" />
                <span className="font-medium text-blue-600">
                  {selectedWells[0].name}
                  {selectedWells.length > 1 && ` + ${selectedWells.length - 1} more`}
                </span>
              </>
            )}
          </div>
          <div className="flex items-center space-x-3">
            {hasMultipleDataTypes && (
              <select
                value={selectedDataType}
                onChange={(e) => {
                  const newType = e.target.value;
                  setSelectedDataType(newType);
                  // Unload raster/model if its data type no longer matches
                  if (rasterResult && rasterResult.dataType !== newType) {
                    setRasterResult(null);
                    setCompareRasterResults([]);
                  }
                  if (selectedModel && selectedModel.dataType !== newType) {
                    setSelectedModel(null);
                  }
                }}
                className="px-2 py-1.5 border border-slate-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                {selectedRegion!.effectiveDataTypes.map(dt => (
                  <option key={dt.code} value={dt.code}>{dt.name} ({dt.unit})</option>
                ))}
              </select>
            )}
            {selectedRegion && (
              <>
                <button
                  onClick={analyzeTrends}
                  className={`flex items-center space-x-2 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                    showTrends
                      ? 'bg-purple-100 text-purple-700 hover:bg-purple-200'
                      : 'bg-blue-50 text-blue-700 hover:bg-blue-100'
                  }`}
                >
                  <Activity size={16} />
                  <span>Analyze Trends</span>
                </button>
                {showTrends && (
                  <div className="flex items-center gap-1.5 text-xs text-slate-600">
                    <span className="whitespace-nowrap">Window:</span>
                    <button
                      onClick={() => handleTrendWindowChange(Math.max(5, trendWindowYears - 5))}
                      disabled={trendWindowYears <= 5}
                      className="w-5 h-5 flex items-center justify-center rounded bg-slate-100 hover:bg-slate-200 disabled:opacity-30 disabled:cursor-not-allowed font-bold text-slate-600"
                    >&minus;</button>
                    <span className="w-12 text-center font-medium">{trendWindowYears} yr</span>
                    <button
                      onClick={() => handleTrendWindowChange(Math.min(trendWindowMax, trendWindowYears + 5))}
                      disabled={trendWindowYears >= trendWindowMax}
                      className="w-5 h-5 flex items-center justify-center rounded bg-slate-100 hover:bg-slate-200 disabled:opacity-30 disabled:cursor-not-allowed font-bold text-slate-600"
                    >+</button>
                  </div>
                )}
              </>
            )}
            {selectedAquifer && (
              <button
                onClick={() => setRasterDialogOpen(true)}
                className="flex items-center space-x-2 px-3 py-1.5 bg-emerald-50 text-emerald-700 rounded-md text-sm font-medium hover:bg-emerald-100 transition-colors"
              >
                <BarChart3 size={16} />
                <span>Spatial Analysis</span>
              </button>
            )}
            {selectedAquifer && selectedDataType === 'wte' && (
              <button
                onClick={handleOpenImputationWizard}
                className="flex items-center space-x-2 px-3 py-1.5 bg-amber-50 text-amber-700 rounded-md text-sm font-medium hover:bg-amber-100 transition-colors"
              >
                <Activity size={16} />
                <span>Impute Gaps</span>
              </button>
            )}
            <button
              onClick={() => setIsDataManagerOpen(true)}
              className="flex items-center space-x-2 px-3 py-1.5 bg-blue-50 text-blue-700 rounded-md text-sm font-medium hover:bg-blue-100 transition-colors"
            >
              <Database size={16} />
              <span>Manage Data</span>
            </button>
          </div>
        </header>

        {/* Map and Chart Split View */}
        <div ref={splitContainerRef} className="flex-1 flex flex-col min-h-0">
          <div className="flex-1 relative">
            <MapView
              ref={mapViewRef}
              regions={regions.filter(r => visibleRegionIds.has(r.id))}
              aquifers={filteredAquifers}
              wells={filteredWells}
              measurements={measurements}
              selectedRegion={selectedRegion}
              selectedAquifer={selectedAquifer}
              selectedWells={selectedWells}
              selectedDataType={selectedDataType}
              wellColors={showTrends ? trendColors : null}
              aquiferColors={showTrends && !selectedAquifer ? aquiferTrendColors : null}
              rasterActiveWellIds={rasterActiveWellIds}
              minObs={mapMinObs}
              onMinObsChange={setMapMinObs}
              onRegionClick={(r) => {
                if (selectedRegion?.id === r.id) {
                  // Clicking the already-selected region: deselect aquifer (if any) or clear wells
                  if (selectedAquifer && !r.singleUnit) {
                    setSelectedAquifer(null);
                  }
                  setSelectedWells([]);
                  return;
                }
                setSelectedRegion(r);
                setSelectedWells([]);
                // For single-unit regions, auto-select the default aquifer
                if (r.singleUnit) {
                  const singleAquifer = aquifers.find(a => a.regionId === r.id);
                  setSelectedAquifer(singleAquifer || null);
                } else {
                  setSelectedAquifer(null);
                }
              }}
              onAquiferClick={setSelectedAquifer}
              onWellClick={handleWellClick}
              onWellBoxSelect={handleWellBoxSelect}
              onShowWellsChange={setShowWellsOnMap}
              onShowWellIdsChange={setShowWellIdsOnMap}
              onDateFilterChange={setDateFilter}
            />
            {/* Well legend — shown when aquifer selected, wells visible, and trends not active */}
            {selectedAquifer && showWellsOnMap && !(showTrends && (trendColors || aquiferTrendColors)) && (
              <div className="absolute top-20 left-2 z-[90] bg-white rounded-lg shadow-lg border border-slate-200 p-3" style={{ width: '180px' }}>
                <div className="text-xs font-semibold text-slate-700 mb-2">Wells</div>
                {rasterActiveWellIds ? (
                  <>
                    <div className="flex items-center gap-2 py-0.5">
                      <span className="inline-block w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: '#22c55e' }} />
                      <span className="text-xs text-slate-600">Active (has data)</span>
                    </div>
                    <div className="flex items-center gap-2 py-0.5">
                      <span className="inline-block w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: '#4b5563' }} />
                      <span className="text-xs text-slate-600">Inactive</span>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="flex items-center gap-2 py-0.5">
                      <span className="inline-block w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: '#3b82f6' }} />
                      <span className="text-xs text-slate-600">2+ observations</span>
                    </div>
                    <div className="flex items-center gap-2 py-0.5">
                      <span className="inline-block w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: '#6b7280' }} />
                      <span className="text-xs text-slate-600">1 observation</span>
                    </div>
                    <div className="flex items-center gap-2 py-0.5">
                      <span className="inline-block w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: '#ef4444' }} />
                      <span className="text-xs text-slate-600">No data</span>
                    </div>
                  </>
                )}
              </div>
            )}
            {/* Trend legend */}
            {showTrends && (trendColors || aquiferTrendColors) && (() => {
              const isAquiferMode = !selectedAquifer && aquiferTrendColors !== null;
              const activeColors = isAquiferMode ? aquiferTrendColors! : trendColors!;
              const unit = selectedRegion?.lengthUnit === 'm' ? 'm' : 'ft';
              const thresholds = isAquiferMode
                ? (unit === 'm' ? AQUIFER_TREND_THRESHOLDS_M : AQUIFER_TREND_THRESHOLDS_FT)
                : (unit === 'm' ? TREND_THRESHOLDS_M : TREND_THRESHOLDS_FT);
              return (
                <div className="absolute top-20 left-2 z-[90] bg-white rounded-lg shadow-lg border border-slate-200 p-3" style={{ width: '210px' }}>
                  <div className="text-xs font-semibold text-slate-700 mb-0.5">
                    {isAquiferMode ? 'Aquifer Trend (median)' : 'Well Trend'} ({unit}/yr)
                  </div>
                  <div className="text-[10px] text-slate-400 mb-2">
                    {thresholds.moderate} / {thresholds.extreme} {unit}/yr
                  </div>
                  {[...TREND_CATEGORIES].reverse().map(cat => {
                    const count = Array.from(activeColors.values()).filter(c => c === cat.color).length;
                    return (
                      <div key={cat.label} className="flex items-center gap-2 py-0.5">
                        <span className="inline-block w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: cat.color }} />
                        <span className="text-xs text-slate-600">{cat.label} ({count})</span>
                      </div>
                    );
                  })}
                  <div className="flex items-center gap-2 py-0.5">
                    <span className="inline-block w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: INSUFFICIENT_COLOR }} />
                    <span className="text-xs text-slate-600">
                      Insufficient data ({Array.from(activeColors.values()).filter(c => c === INSUFFICIENT_COLOR).length})
                    </span>
                  </div>
                </div>
              );
            })()}
            {/* Raster Overlay */}
            {rasterResult && mapViewRef.current?.getMap() && (
              <RasterOverlay
                analysis={rasterResult}
                map={mapViewRef.current.getMap()!}
                onClose={handleUnloadRaster}
                onFrameChange={handleRasterFrameChange}
                lengthUnit={selectedRegion?.lengthUnit || 'ft'}
                onCrossSectionChange={handleCrossSectionChange}
                dataTypeName={activeDataType.name}
                showActiveWells={showActiveWells}
                onToggleActiveWells={handleToggleActiveWells}
              />
            )}
          </div>

          {/* Drag handle + Time Series Section */}
          {(selectedWells.length > 0 || rasterResult || crossSectionProfile) && (
            <div
              onMouseDown={handleDividerMouseDown}
              className="h-1.5 bg-slate-200 hover:bg-blue-400 cursor-row-resize flex-shrink-0 transition-colors relative group"
            >
              <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 flex justify-center">
                <div className="w-8 h-0.5 rounded-full bg-slate-400 group-hover:bg-white transition-colors" />
              </div>
            </div>
          )}
          <div
            ref={chartPanelRef}
            className={`border-t border-slate-200 bg-white flex-shrink-0 ${
              (selectedWells.length > 0 || rasterResult || crossSectionProfile) ? '' : 'h-0 overflow-hidden'
            }`}
            style={(selectedWells.length > 0 || rasterResult || crossSectionProfile) ? { height: chartHeight } : undefined}
          >
            {(selectedWells.length > 0 || rasterResult || crossSectionProfile) && (() => {
              // Build available tabs dynamically
              const availableTabs: ('waterLevel' | 'storageChange' | 'rasterStats' | 'crossSection')[] = [];
              if (selectedWells.length > 0) availableTabs.push('waterLevel');
              if (rasterResult?.dataType === 'wte') availableTabs.push('storageChange');
              if (rasterResult?.stats) availableTabs.push('rasterStats');
              if (crossSectionProfile) availableTabs.push('crossSection');
              const showTabs = availableTabs.length > 1;
              const effectiveTab = availableTabs.includes(activeTimeSeriesTab) ? activeTimeSeriesTab : availableTabs[0];
              return (
                <div className="h-full flex flex-col">
                  {showTabs && (
                    <div className="flex px-4 pt-1 -mb-px" style={{ zIndex: 1 }}>
                      {availableTabs.includes('waterLevel') && (
                        <button
                          onClick={() => setActiveTimeSeriesTab('waterLevel')}
                          className={`px-3 py-1 text-[11px] font-medium rounded-t border border-b-0 transition-colors ${
                            effectiveTab === 'waterLevel'
                              ? 'bg-white text-slate-800 border-slate-200'
                              : 'bg-slate-50 text-slate-400 border-transparent hover:text-slate-600'
                          }`}
                        >
                          {activeDataType.code === 'wte' ? 'Water Level' : activeDataType.name}
                        </button>
                      )}
                      {availableTabs.includes('storageChange') && (
                        <button
                          onClick={() => setActiveTimeSeriesTab('storageChange')}
                          className={`px-3 py-1 text-[11px] font-medium rounded-t border border-b-0 transition-colors ${
                            effectiveTab === 'storageChange'
                              ? 'bg-white text-emerald-700 border-slate-200'
                              : 'bg-slate-50 text-slate-400 border-transparent hover:text-slate-600'
                          }`}
                        >
                          Storage Change
                        </button>
                      )}
                      {availableTabs.includes('rasterStats') && (
                        <button
                          onClick={() => setActiveTimeSeriesTab('rasterStats')}
                          className={`px-3 py-1 text-[11px] font-medium rounded-t border border-b-0 transition-colors ${
                            effectiveTab === 'rasterStats'
                              ? 'bg-white text-violet-700 border-slate-200'
                              : 'bg-slate-50 text-slate-400 border-transparent hover:text-slate-600'
                          }`}
                        >
                          Raster Statistics
                        </button>
                      )}
                      {availableTabs.includes('crossSection') && (
                        <button
                          onClick={() => setActiveTimeSeriesTab('crossSection')}
                          className={`px-3 py-1 text-[11px] font-medium rounded-t border border-b-0 transition-colors ${
                            effectiveTab === 'crossSection'
                              ? 'bg-white text-blue-700 border-slate-200'
                              : 'bg-slate-50 text-slate-400 border-transparent hover:text-slate-600'
                          }`}
                        >
                          Cross Section
                        </button>
                      )}
                    </div>
                  )}
                  <div className={`px-4 pb-4 pt-2 flex-1 flex flex-col ${showTabs ? 'border-t border-slate-200' : 'pt-4'}`}>
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-3">
                      {effectiveTab === 'waterLevel' ? (
                        <>
                          {!showTabs && <Activity size={18} className="text-blue-500" />}
                          <h3 className="font-bold text-slate-800">
                            {activeDataType.name}: {
                              selectedWells.length <= 3
                                ? selectedWells.map(w => showWellIdsOnMap ? `${w.name} (${w.id})` : w.name).join(', ')
                                : `${selectedWells.length} wells selected`
                            }
                          </h3>
                        </>
                      ) : effectiveTab === 'storageChange' ? (
                        <>
                          {!showTabs && <BarChart3 size={18} className="text-emerald-500" />}
                          <div>
                            <h3 className="font-bold text-slate-800">
                              {allRasterResults.length > 1 ? 'Storage Change Comparison' : `Storage Change: ${rasterResult!.title}`}
                            </h3>
                            {allRasterResults.length === 1 && getRasterMethodSummary(rasterResult!) && (
                              <div className="text-[11px] text-slate-400">{getRasterMethodSummary(rasterResult!)}</div>
                            )}
                          </div>
                        </>
                      ) : effectiveTab === 'rasterStats' ? (
                        <>
                          <div>
                            <h3 className="font-bold text-slate-800">
                              Raster Statistics: {rasterResult!.title}
                            </h3>
                            {getRasterMethodSummary(rasterResult!) && (
                              <div className="text-[11px] text-slate-400">{getRasterMethodSummary(rasterResult!)}</div>
                            )}
                          </div>
                        </>
                      ) : effectiveTab === 'crossSection' && crossSectionProfile ? (
                        <>
                          <h3 className="font-bold text-slate-800">
                            Cross Section A–A'
                          </h3>
                          <span className="text-xs text-slate-400">
                            ({crossSectionProfile.totalLength.toFixed(0)} {selectedRegion?.lengthUnit || 'ft'})
                          </span>
                        </>
                      ) : rasterResult ? (
                        <h3 className="font-bold text-slate-800">{rasterResult.title}</h3>
                      ) : null}
                    </div>
                    <div className="flex items-center space-x-4">
                      {effectiveTab === 'waterLevel' ? (
                        <>
                          {activeDataType.code === 'wte' && (
                            <label className="flex items-center gap-1.5 text-xs text-slate-600 cursor-pointer select-none">
                              <input type="checkbox" checked={showGSE} onChange={(e) => setShowGSE(e.target.checked)} className="accent-blue-500" />
                              GSE
                            </label>
                          )}
                          {!(selectedModel && selectedWells.length === 1) && (
                            <>
                              <label className="flex items-center gap-1.5 text-xs text-slate-600 cursor-pointer select-none">
                                <input type="checkbox" checked={usePCHIP} onChange={(e) => setUsePCHIP(e.target.checked)} className="accent-blue-500" />
                                PCHIP
                              </label>
                              <label className="flex items-center gap-1.5 text-xs text-slate-600 cursor-pointer select-none">
                                <input type="checkbox" checked={showTrendLine} onChange={(e) => setShowTrendLine(e.target.checked)} className="accent-blue-500" />
                                Trend Line
                              </label>
                            </>
                          )}
                          {!(selectedModel && selectedWells.length === 1 && !showCombinedModel) && (
                            <>
                              <label className="flex items-center gap-1.5 text-xs text-slate-600 cursor-pointer select-none">
                                <input type="checkbox" checked={showSmooth} onChange={(e) => setShowSmooth(e.target.checked)} className="accent-blue-500" />
                                MAvg
                              </label>
                              {showSmooth && (
                                <label className="flex items-center gap-1 text-xs text-slate-600 select-none">
                                  <span>months:</span>
                                  <input
                                    type="number"
                                    min={3}
                                    step={3}
                                    value={smoothMonths}
                                    onChange={(e) => {
                                      const v = Math.max(3, Math.round(parseInt(e.target.value) / 3) * 3 || 3);
                                      setSmoothMonths(v);
                                    }}
                                    className="w-12 border border-slate-300 rounded px-1 py-0.5 text-xs text-center focus:outline-none focus:ring-1 focus:ring-blue-500"
                                  />
                                </label>
                              )}
                            </>
                          )}
                          {!(selectedModel && selectedWells.length === 1) && (
                            <button
                              onClick={() => setIsDataEditorOpen(true)}
                              disabled={selectedWells.length !== 1}
                              className="flex items-center space-x-1.5 px-3 py-1.5 bg-blue-50 text-blue-700 rounded-md text-sm font-medium hover:bg-blue-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                              title={selectedWells.length !== 1 ? 'Select a single well to edit' : 'View/Edit measurement data'}
                            >
                              <Table size={14} />
                              <span>View/Edit</span>
                            </button>
                          )}
                          <button
                            onClick={exportToCSV}
                            disabled={selectedWellMeasurements.length === 0 && !(selectedModel && selectedWells.length === 1)}
                            className="flex items-center space-x-1.5 px-3 py-1.5 bg-green-50 text-green-700 rounded-md text-sm font-medium hover:bg-green-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                            title="Export data to CSV"
                          >
                            <Download size={14} />
                            <span>Export CSV</span>
                          </button>
                          <button
                            onClick={() => setIsChartExpanded(true)}
                            className="p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded transition-colors"
                            title="Expand chart"
                          >
                            <Maximize2 size={14} />
                          </button>
                          <div className="text-xs text-slate-500 uppercase tracking-wider font-semibold">
                            Units: {activeDataType.unit === 'm' ? 'Meters' : activeDataType.unit === 'ft' ? 'Feet' : activeDataType.unit} ({activeDataType.code.toUpperCase()})
                          </div>
                        </>
                      ) : effectiveTab === 'storageChange' ? (
                        <div className="flex items-center gap-3">
                          <label className="flex items-center gap-1.5 text-xs text-slate-600 cursor-pointer select-none">
                            <input type="checkbox" checked={usePCHIP} onChange={(e) => setUsePCHIP(e.target.checked)} className="accent-blue-500" />
                            PCHIP
                          </label>
                          <label className="flex items-center gap-1.5 text-xs text-slate-600 select-none">
                            <span className="font-medium">Storage Coeff.</span>
                            <button
                              onClick={() => setStorageCoeff(Math.max(0.01, +(storageCoeff - 0.05).toFixed(2)))}
                              disabled={storageCoeff <= 0.01}
                              className="w-5 h-5 flex items-center justify-center rounded bg-slate-100 hover:bg-slate-200 disabled:opacity-30 disabled:cursor-not-allowed font-bold text-slate-600 text-xs"
                            >&minus;</button>
                            <input
                              type="number"
                              value={storageCoeff}
                              min={0.01}
                              max={1}
                              step={0.05}
                              onChange={e => {
                                const v = parseFloat(e.target.value);
                                if (!isNaN(v) && v >= 0.01 && v <= 1) setStorageCoeff(+(v.toFixed(2)));
                              }}
                              className="w-14 border border-slate-300 rounded px-1 py-0.5 text-xs text-center focus:outline-none focus:ring-1 focus:ring-emerald-500"
                            />
                            <button
                              onClick={() => setStorageCoeff(Math.min(1, +(storageCoeff + 0.05).toFixed(2)))}
                              disabled={storageCoeff >= 1}
                              className="w-5 h-5 flex items-center justify-center rounded bg-slate-100 hover:bg-slate-200 disabled:opacity-30 disabled:cursor-not-allowed font-bold text-slate-600 text-xs"
                            >+</button>
                          </label>
                          <select
                            value={storageVolumeUnit}
                            onChange={e => setStorageVolumeUnit(e.target.value)}
                            className="px-1.5 py-0.5 border border-slate-300 rounded text-xs focus:ring-1 focus:ring-emerald-500"
                          >
                            {(selectedRegion?.lengthUnit === 'ft'
                              ? [{ value: 'acre-ft', label: 'acre-ft' }, { value: 'ft3', label: 'ft\u00B3' }]
                              : [{ value: 'MCM', label: 'MCM' }, { value: 'm3', label: 'm\u00B3' }, { value: 'km3', label: 'km\u00B3' }]
                            ).map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                          </select>
                          <span className="text-[10px] text-slate-400">
                            {allRasterResults.length > 1
                              ? `${allRasterResults.length} analyses`
                              : <>{rasterResult!.params.interval} &bull; res={rasterResult!.params.resolution}</>
                            }
                          </span>
                          <button
                            onClick={() => {
                              if (storageChartData.length === 0) return;
                              const headers = ['Date', ...allRasterResults.map(r => `${r.title} (${storageVolumeUnit})`)];
                              const rows = storageChartData.map(row => {
                                const date = new Date(row.date).toISOString().slice(0, 10);
                                const vals = allRasterResults.map((_, i) => {
                                  const v = row[`value_${i}`];
                                  return v !== undefined ? (v as number).toFixed(2) : '';
                                });
                                return [date, ...vals];
                              });
                              const csv = [headers, ...rows].map(r => r.join(',')).join('\n');
                              const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
                              const url = URL.createObjectURL(blob);
                              const link = document.createElement('a');
                              link.href = url;
                              link.download = `storage_change_${rasterResult!.code}.csv`;
                              link.click();
                              URL.revokeObjectURL(url);
                            }}
                            disabled={storageChartData.length === 0}
                            className="flex items-center space-x-1.5 px-3 py-1.5 bg-green-50 text-green-700 rounded-md text-sm font-medium hover:bg-green-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                            title="Export data to CSV"
                          >
                            <Download size={14} />
                            <span>Export CSV</span>
                          </button>
                        </div>
                      ) : effectiveTab === 'rasterStats' ? (
                        <div className="text-xs text-slate-500 uppercase tracking-wider font-semibold">
                          {activeDataType.unit === 'm' ? 'Meters' : activeDataType.unit === 'ft' ? 'Feet' : activeDataType.unit} ({activeDataType.code.toUpperCase()})
                        </div>
                      ) : (
                        <div className="text-[10px] text-slate-400">
                          {rasterFrameDate?.date || ''}
                        </div>
                      )}
                    </div>
                  </div>
                  {effectiveTab === 'storageChange' && allRasterResults.length > 1 && (
                    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 mb-1">
                      {allRasterResults.map((r, i) => (
                        <div key={r.code} className="flex items-center gap-1.5">
                          <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: STORAGE_COLORS[i % STORAGE_COLORS.length] }} />
                          <span className="text-[11px] text-slate-600">{r.title}</span>
                        </div>
                      ))}
                    </div>
                  )}
                  <div className="flex-1 min-h-0">
                    {selectedModel && selectedWells.length === 1 && effectiveTab === 'waterLevel' ? (
                      <ModelTimeSeries
                        model={selectedModel}
                        well={selectedWells[0]}
                        measurements={measurements}
                        showCombined={showCombinedModel}
                        onToggleCombined={() => {
                          setShowCombinedModel(p => {
                            if (p) setShowSmooth(false); // reset smooth when switching to separated
                            return !p;
                          });
                        }}
                        lengthUnit={selectedRegion?.lengthUnit || 'ft'}
                        showSmooth={showSmooth}
                        smoothMonths={smoothMonths}
                        showGSE={showGSE && activeDataType.code === 'wte'}
                      />
                    ) : effectiveTab === 'waterLevel' ? (
                      <TimeSeriesChart
                        measurements={selectedWellMeasurements}
                        selectedWells={selectedWells}
                        showGSE={showGSE && activeDataType.code === 'wte'}
                        showTrendLine={showTrendLine}
                        showSmooth={showSmooth}
                        smoothMonths={smoothMonths}
                        usePCHIP={usePCHIP}
                        dataType={activeDataType}
                        lengthUnit={selectedRegion?.lengthUnit || 'ft'}
                        onEditMeasurement={handleChartEditMeasurement}
                        onDeleteMeasurement={handleChartDeleteMeasurement}
                        referenceDate={rasterResult && rasterFrameDate ? rasterFrameDate.dateTs : undefined}
                        rasterTimeRange={rasterTimeRange}
                        trendWindowStart={showTrends ? Date.now() - trendWindowYears * MS_PER_YEAR : undefined}
                        dateFilter={dateFilter}
                      />
                    ) : effectiveTab === 'rasterStats' && rasterResult?.stats ? (
                      <RasterStatsChart
                        stats={rasterResult.stats}
                        dataTypeName={activeDataType.name}
                        dataTypeUnit={activeDataType.unit}
                        referenceDate={rasterFrameDate?.dateTs}
                      />
                    ) : effectiveTab === 'crossSection' && crossSectionProfile ? (
                      <CrossSectionChart
                        profile={crossSectionProfile}
                        frameIdx={crossSectionProfile.frameDates.indexOf(rasterFrameDate?.date || crossSectionProfile.frameDates[0])}
                        lengthUnit={selectedRegion?.lengthUnit || 'ft'}
                      />
                    ) : rasterResult && storageChartData.length > 0 && (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={storageChartData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                          <XAxis
                            dataKey="date" type="number" domain={['dataMin', 'dataMax']}
                            tickFormatter={(t: number) => new Date(t).getFullYear().toString()}
                            stroke="#94a3b8" fontSize={10}
                          />
                          <YAxis stroke="#94a3b8" fontSize={10}
                            tickFormatter={(v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                            label={allRasterResults.length === 1 ? { value: storageVolumeUnit, angle: -90, position: 'insideLeft', style: { textAnchor: 'middle', fill: '#94a3b8', fontSize: 10 } } : undefined}
                          />
                          <Tooltip
                            content={({ label, payload }) => {
                              if (!payload || payload.length === 0) return null;
                              const items = payload.filter(p => p.value !== undefined && p.value !== null);
                              if (items.length === 0) return null;
                              return (
                                <div className="bg-white rounded shadow-md px-2 py-1.5 text-[10px] border border-slate-200">
                                  <div className="text-slate-400 mb-1">{new Date(label as number).toLocaleDateString()}</div>
                                  {items.map((p, i) => {
                                    const resultIdx = parseInt((p.dataKey as string).replace('value_', ''));
                                    const result = allRasterResults[resultIdx];
                                    if (!result) return null;
                                    return (
                                      <div key={i} className="flex items-center gap-1.5">
                                        {allRasterResults.length > 1 && (
                                          <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: p.stroke as string }} />
                                        )}
                                        <span className="text-slate-700 font-medium">
                                          {allRasterResults.length > 1 ? `${result.title}: ` : ''}
                                          {(p.value as number)?.toLocaleString(undefined, { maximumFractionDigits: 1 })} {storageVolumeUnit}
                                        </span>
                                      </div>
                                    );
                                  })}
                                </div>
                              );
                            }}
                          />
                          {allRasterResults.map((r, i) => (
                            <Line
                              key={r.code}
                              type={usePCHIP ? 'monotone' : 'linear'}
                              dataKey={`value_${i}`}
                              stroke={STORAGE_COLORS[i % STORAGE_COLORS.length]}
                              strokeWidth={i === 0 ? 2 : 1.5}
                              dot={{ r: i === 0 ? 1.5 : 1, fill: STORAGE_COLORS[i % STORAGE_COLORS.length] }}
                              isAnimationActive={false}
                              connectNulls
                            />
                          ))}
                          {rasterFrameDate && (
                            <ReferenceLine x={rasterFrameDate.dateTs} stroke="#ef4444" strokeDasharray="4 3" strokeWidth={1.5} />
                          )}
                        </LineChart>
                      </ResponsiveContainer>
                    )}
                  </div>
                  </div>
                </div>
              );
            })()}
          </div>
        </div>
      </main>

      {/* Import Data Hub */}
      {isDataManagerOpen && (
        <ImportDataHub
          onClose={() => setIsDataManagerOpen(false)}
          onDataChanged={handleDataChanged}
          initialRegionId={selectedRegion?.id || null}
        />
      )}

      {/* Spatial Analysis Dialog */}
      {rasterDialogOpen && selectedAquifer && selectedRegion && (
        <SpatialAnalysisDialog
          aquifer={selectedAquifer}
          region={selectedRegion}
          wells={wells.filter(w => w.aquiferId === selectedAquifer.id && w.regionId === selectedAquifer.regionId)}
          measurements={measurements}
          dataType={activeDataType}
          existingCodes={rasterMeta.filter(a => a.aquiferId === selectedAquifer.id && a.regionId === selectedAquifer.regionId && a.dataType === activeDataType.code).map(a => a.code)}
          availableModels={modelMeta.filter(m => m.aquiferId === selectedAquifer.id && m.regionId === selectedAquifer.regionId)}
          onClose={() => setRasterDialogOpen(false)}
          onComplete={(result) => {
            setRasterResult(result);
            const aquiferSlug = slugify(result.aquiferName);
            setRasterMeta(prev => [...prev, {
              title: result.title,
              code: result.code,
              aquiferId: result.aquiferId,
              aquiferName: result.aquiferName,
              regionId: result.regionId,
              filePath: `${result.regionId}/${aquiferSlug}/raster_${result.dataType}_${result.code}.json`,
              dataType: result.dataType,
              params: result.params,
              createdAt: result.createdAt,
              options: result.options,
              generatedAt: result.generatedAt,
            }]);
            setRasterDialogOpen(false);
          }}
        />
      )}

      {/* Imputation Wizard */}
      {imputationDialogOpen && selectedAquifer && selectedRegion && (
        <ImputationWizard
          aquifer={selectedAquifer}
          region={selectedRegion}
          wells={wells.filter(w => w.aquiferId === selectedAquifer.id && w.regionId === selectedAquifer.regionId)}
          measurements={measurements}
          existingModelCodes={modelMeta.filter(m => m.aquiferId === selectedAquifer.id && m.regionId === selectedAquifer.regionId).map(m => m.code)}
          gldasDateRange={gldasDateRange}
          minSamples={mapMinObs}
          onClose={() => setImputationDialogOpen(false)}
          onComplete={(result) => {
            setSelectedModel(result);
            setModelMeta(prev => [...prev, {
              title: result.title,
              code: result.code,
              aquiferId: result.aquiferId,
              aquiferName: result.aquiferName,
              regionId: result.regionId,
              filePath: result.filePath,
              dataType: result.dataType,
              params: result.params,
              createdAt: result.createdAt,
              wellMetrics: result.wellMetrics,
            }]);
            setImputationDialogOpen(false);
          }}
        />
      )}

      {/* Raster Info Dialog */}
      {rasterInfoMeta && (
        <RasterInfoDialog
          meta={rasterInfoMeta}
          onClose={() => setRasterInfoMeta(null)}
        />
      )}

      {/* Model Info Dialog */}
      {modelInfoMeta && (
        <ModelInfoDialog
          meta={modelInfoMeta}
          onClose={() => setModelInfoMeta(null)}
        />
      )}

      {/* Data Editor Modal */}
      {isDataEditorOpen && selectedWells.length === 1 && (
        <DataEditor
          well={selectedWells[0]}
          measurements={selectedWellMeasurements}
          allMeasurements={measurements}
          regionId={selectedWells[0].regionId}
          dataType={activeDataType}
          onClose={() => setIsDataEditorOpen(false)}
          onSave={handleDataEditorSave}
        />
      )}

      {/* Expanded Chart Window */}
      {isChartExpanded && selectedWells.length > 0 && (
        <ExpandedChartWindow
          onClose={() => setIsChartExpanded(false)}
          title={`${activeDataType.name}: ${
            selectedWells.length <= 3
              ? selectedWells.map(w => showWellIdsOnMap ? `${w.name} (${w.id})` : w.name).join(', ')
              : `${selectedWells.length} wells selected`
          }`}
          subtitle={`Units: ${activeDataType.unit === 'm' ? 'Meters' : activeDataType.unit === 'ft' ? 'Feet' : activeDataType.unit} (${activeDataType.code.toUpperCase()})`}
        >
          <TimeSeriesChart
            measurements={selectedWellMeasurements}
            selectedWells={selectedWells}
            showGSE={showGSE && activeDataType.code === 'wte'}
            showTrendLine={showTrendLine}
            showSmooth={showSmooth}
            smoothMonths={smoothMonths}
            dataType={activeDataType}
            lengthUnit={selectedRegion?.lengthUnit || 'ft'}
            onEditMeasurement={handleChartEditMeasurement}
            onDeleteMeasurement={handleChartDeleteMeasurement}
            referenceDate={rasterResult && rasterFrameDate ? rasterFrameDate.dateTs : undefined}
            rasterTimeRange={rasterTimeRange}
            trendWindowStart={showTrends ? Date.now() - trendWindowYears * MS_PER_YEAR : undefined}
            dateFilter={dateFilter}
            onEscapeUnhandled={() => setIsChartExpanded(false)}
          />
        </ExpandedChartWindow>
      )}
    </div>
  );
};

export default App;
