import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import L from 'leaflet';
import { Play, Pause, X, ChevronDown } from 'lucide-react';
import { RasterAnalysisResult, CrossSectionProfile } from '../types';
import { sampleCrossSection } from '../utils/rasterSampling';

// --- Color ramp system ---

const LUT_SIZE = 64;

function buildLUT(stops: [number, number, number][], n: number): [number, number, number][] {
  const result: [number, number, number][] = [];
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    const idx = t * (stops.length - 1);
    const lo = Math.floor(idx);
    const hi = Math.min(lo + 1, stops.length - 1);
    const f = idx - lo;
    result.push([
      Math.round(stops[lo][0] + f * (stops[hi][0] - stops[lo][0])),
      Math.round(stops[lo][1] + f * (stops[hi][1] - stops[lo][1])),
      Math.round(stops[lo][2] + f * (stops[hi][2] - stops[lo][2])),
    ]);
  }
  return result;
}

interface ColorRampDef {
  id: string;
  name: string;
  lut: [number, number, number][];
}

const COLOR_RAMPS: ColorRampDef[] = [
  {
    id: 'bgyr',
    name: 'BGYR',
    lut: buildLUT([
      [33, 72, 141], [44, 123, 182], [26, 152, 80], [102, 189, 99],
      [166, 217, 106], [217, 239, 139], [255, 255, 191], [254, 224, 139],
      [253, 174, 97], [244, 109, 67], [214, 47, 39], [178, 24, 43],
    ], LUT_SIZE),
  },
  {
    id: 'viridis',
    name: 'Viridis',
    lut: buildLUT([
      [68, 1, 84], [72, 29, 111], [60, 64, 130], [42, 90, 131],
      [31, 104, 124], [24, 117, 113], [24, 126, 102], [35, 136, 84],
      [66, 146, 57], [107, 153, 28], [152, 156, 19], [194, 151, 44],
      [226, 138, 86], [247, 118, 142], [253, 231, 37],
    ], LUT_SIZE),
  },
  {
    id: 'plasma',
    name: 'Plasma',
    lut: buildLUT([
      [13, 8, 135], [75, 3, 161], [126, 3, 168], [168, 34, 150],
      [203, 70, 121], [229, 107, 93], [248, 148, 65], [253, 195, 40],
      [240, 249, 33],
    ], LUT_SIZE),
  },
  {
    id: 'turbo',
    name: 'Turbo',
    lut: buildLUT([
      [48, 18, 59], [50, 60, 170], [29, 114, 243], [10, 162, 230],
      [26, 200, 175], [80, 226, 116], [155, 240, 55], [212, 238, 30],
      [249, 210, 30], [253, 157, 20], [232, 90, 12], [186, 36, 8],
      [122, 4, 3],
    ], LUT_SIZE),
  },
  {
    id: 'inferno',
    name: 'Inferno',
    lut: buildLUT([
      [0, 0, 4], [20, 11, 53], [58, 12, 96], [101, 21, 110],
      [143, 36, 107], [183, 55, 84], [217, 87, 53], [241, 130, 23],
      [246, 183, 12], [230, 240, 72], [252, 255, 164],
    ], LUT_SIZE),
  },
  {
    id: 'blues',
    name: 'Blues',
    lut: buildLUT([
      [247, 251, 255], [198, 219, 239], [158, 202, 225],
      [107, 174, 214], [66, 146, 198], [33, 113, 181],
      [8, 81, 156], [8, 48, 107],
    ], LUT_SIZE),
  },
];

function getColor(t: number, lut: [number, number, number][]): [number, number, number] {
  const idx = Math.max(0, Math.min(lut.length - 1, Math.round(t * (lut.length - 1))));
  return lut[idx];
}

function rampGradientCSS(lut: [number, number, number][]): string {
  const indices = [0, 10, 21, 32, 42, 53, 63];
  const stops = indices.map(i => `rgb(${lut[i].join(',')})`);
  return `linear-gradient(to right, ${stops.join(', ')})`;
}

// --- Marching squares contour generation ---
const NUM_CONTOUR_LEVELS = 8;

function generateContourLines(
  values: (number | null)[],
  mask: (0 | 1)[],
  nx: number, ny: number,
  minLat: number, minLng: number,
  dx: number, dy: number,
  globalMin: number, globalMax: number
): { level: number; segments: L.LatLng[][] }[] {
  const range = globalMax - globalMin;
  if (range <= 0) return [];

  const levels: number[] = [];
  for (let i = 1; i <= NUM_CONTOUR_LEVELS; i++) {
    levels.push(globalMin + (i / (NUM_CONTOUR_LEVELS + 1)) * range);
  }

  const getVal = (row: number, col: number): number | null => {
    if (row < 0 || row >= ny || col < 0 || col >= nx) return null;
    const idx = row * nx + col;
    if (mask[idx] === 0) return null;
    return values[idx];
  };

  const results: { level: number; segments: L.LatLng[][] }[] = [];

  for (const level of levels) {
    const segments: L.LatLng[][] = [];

    for (let row = 0; row < ny - 1; row++) {
      for (let col = 0; col < nx - 1; col++) {
        const bl = getVal(row, col);
        const br = getVal(row, col + 1);
        const tr = getVal(row + 1, col + 1);
        const tl = getVal(row + 1, col);

        if (bl === null || br === null || tr === null || tl === null) continue;

        const caseIdx = ((tl >= level ? 1 : 0) << 3) |
                        ((tr >= level ? 1 : 0) << 2) |
                        ((br >= level ? 1 : 0) << 1) |
                        ((bl >= level ? 1 : 0));

        if (caseIdx === 0 || caseIdx === 15) continue;

        const lerp = (v1: number, v2: number): number => {
          const d = v2 - v1;
          return Math.abs(d) < 1e-10 ? 0.5 : (level - v1) / d;
        };

        const topPt = L.latLng(minLat + (row + 1) * dy, minLng + (col + lerp(tl, tr)) * dx);
        const botPt = L.latLng(minLat + row * dy, minLng + (col + lerp(bl, br)) * dx);
        const leftPt = L.latLng(minLat + (row + lerp(bl, tl)) * dy, minLng + col * dx);
        const rightPt = L.latLng(minLat + (row + lerp(br, tr)) * dy, minLng + (col + 1) * dx);

        switch (caseIdx) {
          case 1: case 14:
            segments.push([leftPt, botPt]); break;
          case 2: case 13:
            segments.push([botPt, rightPt]); break;
          case 3: case 12:
            segments.push([leftPt, rightPt]); break;
          case 4: case 11:
            segments.push([topPt, rightPt]); break;
          case 5: {
            const c = (bl + br + tr + tl) / 4;
            if (c >= level) {
              segments.push([leftPt, topPt]);
              segments.push([botPt, rightPt]);
            } else {
              segments.push([leftPt, botPt]);
              segments.push([topPt, rightPt]);
            }
            break;
          }
          case 6: case 9:
            segments.push([botPt, topPt]); break;
          case 7: case 8:
            segments.push([leftPt, topPt]); break;
          case 10: {
            const c = (bl + br + tr + tl) / 4;
            if (c >= level) {
              segments.push([leftPt, botPt]);
              segments.push([topPt, rightPt]);
            } else {
              segments.push([leftPt, topPt]);
              segments.push([botPt, rightPt]);
            }
            break;
          }
        }
      }
    }

    results.push({ level, segments });
  }

  return results;
}

interface RasterOverlayProps {
  analysis: RasterAnalysisResult;
  map: L.Map;
  onClose: () => void;
  onFrameChange?: (date: string, dateTs: number) => void;
  lengthUnit: 'ft' | 'm';
  onCrossSectionChange?: (profile: CrossSectionProfile | null) => void;
  dataTypeName?: string;
  showActiveWells: boolean;
  onToggleActiveWells: () => void;
}

const RasterOverlay: React.FC<RasterOverlayProps> = ({
  analysis, map, onClose, onFrameChange, lengthUnit, onCrossSectionChange, dataTypeName,
  showActiveWells, onToggleActiveWells
}) => {
  const [frameIdx, setFrameIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [selectedRamp, setSelectedRamp] = useState('bgyr');
  const [showRampPicker, setShowRampPicker] = useState(false);
  const overlayRef = useRef<L.ImageOverlay | null>(null);
  const contourGroupRef = useRef<L.LayerGroup | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const lastFrameTimeRef = useRef(0);
  const [crossSectionMode, setCrossSectionMode] = useState(false);
  const [crossSectionStart, setCrossSectionStart] = useState<L.LatLng | null>(null);
  const [crossSectionEnd, setCrossSectionEnd] = useState<L.LatLng | null>(null);
  const crossSectionLayerRef = useRef<L.LayerGroup | null>(null);
  const rubberBandRef = useRef<L.Polyline | null>(null);
  const startMarkerRef = useRef<L.CircleMarker | null>(null);

  const { grid, frames } = analysis;
  const { nx, ny, mask, minLng, minLat, dx, dy } = grid;

  const methodSummary = useMemo(() => {
    const opts = analysis.options;
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
  }, [analysis.options]);

  const activeLUT = useMemo(() => {
    const ramp = COLOR_RAMPS.find(r => r.id === selectedRamp);
    return ramp ? ramp.lut : COLOR_RAMPS[0].lut;
  }, [selectedRamp]);

  // Compute color range using 2nd–98th percentile (masked cells excluded)
  const { globalMin, globalMax } = useMemo(() => {
    const allVals: number[] = [];
    for (const frame of frames) {
      for (let i = 0; i < frame.values.length; i++) {
        const v = frame.values[i];
        if (v !== null && mask[i] === 1) allVals.push(v);
      }
    }
    if (allVals.length === 0) return { globalMin: 0, globalMax: 1 };
    allVals.sort((a, b) => a - b);
    const p2 = allVals[Math.floor(allVals.length * 0.02)];
    const p98 = allVals[Math.ceil(allVals.length * 0.98) - 1];
    return { globalMin: p2, globalMax: p98 };
  }, [frames, mask]);

  const bounds: L.LatLngBoundsExpression = useMemo(() => [
    [minLat, minLng],
    [minLat + ny * dy, minLng + nx * dx]
  ], [minLat, minLng, ny, dy, nx, dx]);

  // Render a frame to canvas and update overlay + contours
  const renderFrame = useCallback((idx: number) => {
    if (idx >= frames.length) return;

    // Create canvas on demand
    if (!canvasRef.current) {
      canvasRef.current = document.createElement('canvas');
      canvasRef.current.width = nx;
      canvasRef.current.height = ny;
    }
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const frame = frames[idx];
    const imageData = ctx.createImageData(nx, ny);
    const range = globalMax - globalMin;

    for (let row = 0; row < ny; row++) {
      for (let col = 0; col < nx; col++) {
        const gridIdx = (ny - 1 - row) * nx + col;
        const canvasIdx = (row * nx + col) * 4;
        const val = frame.values[gridIdx];

        if (val === null || mask[gridIdx] === 0) {
          imageData.data[canvasIdx] = 0;
          imageData.data[canvasIdx + 1] = 0;
          imageData.data[canvasIdx + 2] = 0;
          imageData.data[canvasIdx + 3] = 0;
        } else {
          const t = range > 0 ? Math.max(0, Math.min(1, (val - globalMin) / range)) : 0.5;
          const [r, g, b] = getColor(t, activeLUT);
          imageData.data[canvasIdx] = r;
          imageData.data[canvasIdx + 1] = g;
          imageData.data[canvasIdx + 2] = b;
          imageData.data[canvasIdx + 3] = 255;
        }
      }
    }

    ctx.putImageData(imageData, 0, 0);
    const dataUrl = canvas.toDataURL();

    if (overlayRef.current) {
      overlayRef.current.setUrl(dataUrl);
    } else {
      overlayRef.current = L.imageOverlay(dataUrl, bounds, { opacity: 1.0 }).addTo(map);
    }

    // Generate and render contour lines
    if (contourGroupRef.current) {
      contourGroupRef.current.clearLayers();
    } else {
      contourGroupRef.current = L.layerGroup().addTo(map);
    }

    const contours = generateContourLines(
      frame.values, mask, nx, ny,
      minLat, minLng, dx, dy,
      globalMin, globalMax
    );

    for (const { segments } of contours) {
      if (segments.length === 0) continue;
      L.polyline(segments, {
        color: '#000',
        weight: 1.5,
        opacity: 0.4,
        interactive: false,
      }).addTo(contourGroupRef.current!);
    }
  }, [frames, globalMin, globalMax, map, mask, nx, ny, dx, dy, minLat, minLng, bounds, activeLUT]);

  // Remove and recreate overlay when analysis changes — MUST run before renderFrame effect
  useEffect(() => {
    if (overlayRef.current) {
      map.removeLayer(overlayRef.current);
      overlayRef.current = null;
    }
    if (contourGroupRef.current) {
      map.removeLayer(contourGroupRef.current);
      contourGroupRef.current = null;
    }
    canvasRef.current = null;
    setFrameIdx(0);
    setPlaying(false);
    return () => {
      // StrictMode cleanup: remove layers created by subsequent renderFrame effect
      if (overlayRef.current) { map.removeLayer(overlayRef.current); overlayRef.current = null; }
      if (contourGroupRef.current) { map.removeLayer(contourGroupRef.current); contourGroupRef.current = null; }
      canvasRef.current = null;
    };
  }, [analysis, map]);

  // Render first frame on mount, and re-render on frame/ramp change
  useEffect(() => {
    renderFrame(frameIdx);
  }, [frameIdx, renderFrame]);

  // Animation loop
  useEffect(() => {
    if (!playing) {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      return;
    }

    const animate = (time: number) => {
      if (time - lastFrameTimeRef.current > 500) {
        lastFrameTimeRef.current = time;
        setFrameIdx(prev => {
          const next = prev + 1;
          if (next >= frames.length) {
            setPlaying(false);
            return 0;
          }
          return next;
        });
      }
      animFrameRef.current = requestAnimationFrame(animate);
    };

    animFrameRef.current = requestAnimationFrame(animate);
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [playing, frames.length]);

  // Cleanup overlay + contours on unmount
  useEffect(() => {
    return () => {
      if (overlayRef.current) {
        map.removeLayer(overlayRef.current);
        overlayRef.current = null;
      }
      if (contourGroupRef.current) {
        map.removeLayer(contourGroupRef.current);
        contourGroupRef.current = null;
      }
    };
  }, [map]);

  const currentDate = frames[frameIdx]?.date || '';

  // Notify parent of frame date changes
  useEffect(() => {
    if (currentDate && onFrameChange) {
      onFrameChange(currentDate, new Date(currentDate).getTime());
    }
  }, [currentDate, onFrameChange]);

  const clearRubberBand = useCallback(() => {
    if (rubberBandRef.current) { map.removeLayer(rubberBandRef.current); rubberBandRef.current = null; }
    if (startMarkerRef.current) { map.removeLayer(startMarkerRef.current); startMarkerRef.current = null; }
  }, [map]);

  // Reset cross-section when analysis changes
  useEffect(() => {
    setCrossSectionMode(false);
    setCrossSectionStart(null);
    setCrossSectionEnd(null);
    clearRubberBand();
    if (crossSectionLayerRef.current) {
      map.removeLayer(crossSectionLayerRef.current);
      crossSectionLayerRef.current = null;
    }
    onCrossSectionChange?.(null);
  }, [analysis, map]);

  // Cleanup cross-section layers on unmount
  useEffect(() => {
    return () => {
      if (rubberBandRef.current) { map.removeLayer(rubberBandRef.current); rubberBandRef.current = null; }
      if (startMarkerRef.current) { map.removeLayer(startMarkerRef.current); startMarkerRef.current = null; }
      if (crossSectionLayerRef.current) {
        map.removeLayer(crossSectionLayerRef.current);
        crossSectionLayerRef.current = null;
      }
    };
  }, [map]);

  // Draw cross-section line and labels when both points are set
  useEffect(() => {
    if (crossSectionLayerRef.current) {
      map.removeLayer(crossSectionLayerRef.current);
      crossSectionLayerRef.current = null;
    }
    if (!crossSectionStart || !crossSectionEnd) return;

    const group = L.layerGroup().addTo(map);
    crossSectionLayerRef.current = group;

    // Cross-section line
    const lineColor = '#1e5a8a';
    L.polyline([crossSectionStart, crossSectionEnd], {
      color: lineColor,
      weight: 2.5,
      interactive: false,
    }).addTo(group);

    // Perpendicular sight-direction arrows at each endpoint
    const dlat = crossSectionEnd.lat - crossSectionStart.lat;
    const dlng = crossSectionEnd.lng - crossSectionStart.lng;
    const len = Math.sqrt(dlat * dlat + dlng * dlng);
    if (len > 0) {
      // Perpendicular direction (90° CCW = "left" of A→A' = sight direction)
      const sLat = dlng / len;
      const sLng = -dlat / len;
      // Along-line direction for arrowhead wings
      const wLat = dlat / len;
      const wLng = dlng / len;
      const stemLen = len * 0.07;
      const headLen = stemLen * 0.35;
      const headWidth = stemLen * 0.2;

      for (const base of [crossSectionStart, crossSectionEnd]) {
        const tip = L.latLng(base.lat + sLat * stemLen, base.lng + sLng * stemLen);
        L.polyline([base, tip], { color: lineColor, weight: 2, interactive: false }).addTo(group);

        const wingLeft = L.latLng(
          tip.lat - sLat * headLen + wLat * headWidth,
          tip.lng - sLng * headLen + wLng * headWidth
        );
        const wingRight = L.latLng(
          tip.lat - sLat * headLen - wLat * headWidth,
          tip.lng - sLng * headLen - wLng * headWidth
        );
        L.polygon([tip, wingLeft, wingRight], {
          color: lineColor, fillColor: lineColor, fillOpacity: 1,
          weight: 1, interactive: false,
        }).addTo(group);
      }

      // "A" and "A'" labels
      // iconAnchor = the pixel within the icon that sits on the marker latlng.
      // So to push the label down-and-right of the point, we set the anchor
      // to the top-right of the icon (the icon renders up-and-left of the point).
      const labelStyle = 'font-weight:bold;font-size:17px;color:#1e5a8a;text-shadow:1px 1px 2px #fff,-1px 1px 2px #fff,1px -1px 2px #fff,-1px -1px 2px #fff';

      // "A" — anchor top-right so label renders below-left of start point
      L.marker(crossSectionStart, {
        icon: L.divIcon({
          html: `<div style="${labelStyle}">A</div>`,
          className: '',
          iconSize: [20, 22],
          iconAnchor: [20, -4],
        }),
        interactive: false,
      }).addTo(group);

      // "A'" — anchor top-left so label renders below-right of end point
      L.marker(crossSectionEnd, {
        icon: L.divIcon({
          html: `<div style="${labelStyle}">A'</div>`,
          className: '',
          iconSize: [24, 22],
          iconAnchor: [-4, -4],
        }),
        interactive: false,
      }).addTo(group);
    }
  }, [crossSectionStart, crossSectionEnd, map]);

  // Handle cross-section completion — sample and notify parent
  const handleCrossSectionComplete = useCallback((start: L.LatLng, end: L.LatLng) => {
    const profile = sampleCrossSection(
      { lat: start.lat, lng: start.lng },
      { lat: end.lat, lng: end.lng },
      grid,
      frames,
      lengthUnit
    );
    onCrossSectionChange?.(profile);
  }, [grid, frames, lengthUnit, onCrossSectionChange]);

  const handleClearCrossSection = useCallback(() => {
    setCrossSectionStart(null);
    setCrossSectionEnd(null);
    if (crossSectionLayerRef.current) {
      map.removeLayer(crossSectionLayerRef.current);
      crossSectionLayerRef.current = null;
    }
    onCrossSectionChange?.(null);
  }, [map, onCrossSectionChange]);

  const hasCrossSection = crossSectionStart !== null && crossSectionEnd !== null;

  // Color scale legend
  const legendSteps = 5;
  const legendLabels = useMemo(() =>
    Array.from({ length: legendSteps }, (_, i) => {
      const t = 1 - i / (legendSteps - 1); // top=high, bottom=low
      return globalMin + t * (globalMax - globalMin);
    }),
  [globalMin, globalMax]);

  const verticalGradientCSS = useMemo(() => {
    // Top = high values (t=1), bottom = low values (t=0)
    const stops = [1.0, 0.75, 0.5, 0.25, 0.0].map(t => {
      const [r, g, b] = getColor(t, activeLUT);
      return `rgb(${r},${g},${b})`;
    });
    return `linear-gradient(to bottom, ${stops.join(', ')})`;
  }, [activeLUT]);

  const activeRampName = COLOR_RAMPS.find(r => r.id === selectedRamp)?.name || 'BGYR';

  // --- Cursor value tooltip ---
  const [cursorValue, setCursorValue] = useState<{ x: number; y: number; val: number } | null>(null);

  useEffect(() => {
    const onMouseMove = (e: L.LeafletMouseEvent) => {
      const { lat, lng } = e.latlng;
      const col = Math.floor((lng - minLng) / dx);
      const row = Math.floor((lat - minLat) / dy);

      if (col < 0 || col >= nx || row < 0 || row >= ny) {
        setCursorValue(null);
        return;
      }

      const idx = row * nx + col;
      if (mask[idx] === 0) {
        setCursorValue(null);
        return;
      }

      const frame = frames[frameIdx];
      const val = frame?.values[idx];
      if (val === null || val === undefined) {
        setCursorValue(null);
        return;
      }

      const containerPt = map.latLngToContainerPoint(e.latlng);
      setCursorValue({ x: containerPt.x, y: containerPt.y, val });
    };

    const onMouseOut = () => setCursorValue(null);

    map.on('mousemove', onMouseMove);
    map.on('mouseout', onMouseOut);
    return () => {
      map.off('mousemove', onMouseMove);
      map.off('mouseout', onMouseOut);
    };
  }, [map, frames, frameIdx, minLat, minLng, dx, dy, nx, ny, mask]);

  return (
    <>
      {/* Animation Controls */}
      <div className="absolute left-1/2 -translate-x-1/2 z-[95] bg-white/95 backdrop-blur rounded-lg shadow-lg border border-slate-200 px-4 py-2 flex items-center gap-3"
        style={{ bottom: '8px' }}>
        <div className="flex flex-col mr-1">
          <span className="text-xs font-medium text-slate-600">{analysis.title}</span>
          {methodSummary && <span className="text-[10px] text-slate-400">{methodSummary}</span>}
        </div>

        <button onClick={() => setPlaying(!playing)}
          className="p-1.5 rounded-md hover:bg-slate-100 transition-colors">
          {playing ? <Pause size={16} className="text-slate-700" /> : <Play size={16} className="text-slate-700" />}
        </button>

        <input type="range" min={0} max={frames.length - 1} value={frameIdx}
          onChange={e => { setFrameIdx(parseInt(e.target.value)); setPlaying(false); }}
          className="w-48 h-1.5 accent-emerald-500" />

        <span className="text-xs font-medium text-slate-700 min-w-[80px]">{currentDate}</span>

        <button onClick={onClose}
          className="p-1 hover:bg-slate-100 rounded transition-colors ml-2">
          <X size={14} className="text-slate-400" />
        </button>
      </div>

      {/* Color legend + Cross Section button — top-right, below basemap selector */}
      <div className="absolute top-14 right-3 z-[95] flex flex-col items-end gap-2">
        <div className="bg-white/95 backdrop-blur rounded-lg shadow-lg border border-slate-200 p-3">
          <button
            onClick={() => setShowRampPicker(!showRampPicker)}
            className="flex items-center gap-1.5 text-xs font-semibold text-slate-700 mb-2 hover:text-slate-900 transition-colors"
          >
            <span>{dataTypeName || analysis.dataType.toUpperCase()}</span>
            <ChevronDown size={12} className={`text-slate-400 transition-transform ${showRampPicker ? 'rotate-180' : ''}`} />
          </button>

          {showRampPicker && (
            <div className="mb-3 space-y-1" style={{ width: '150px' }}>
              {COLOR_RAMPS.map(ramp => (
                <button
                  key={ramp.id}
                  onClick={() => { setSelectedRamp(ramp.id); setShowRampPicker(false); }}
                  className={`w-full flex items-center gap-2 px-1.5 py-1 rounded text-left transition-colors ${
                    ramp.id === selectedRamp
                      ? 'bg-slate-100 ring-1 ring-slate-300'
                      : 'hover:bg-slate-50'
                  }`}
                >
                  <div
                    className="h-3 flex-1 rounded-sm border border-slate-200"
                    style={{ background: rampGradientCSS(ramp.lut) }}
                  />
                  <span className="text-[10px] text-slate-600 w-[42px] text-right flex-shrink-0">{ramp.name}</span>
                </button>
              ))}
            </div>
          )}

          <div className="flex items-stretch gap-2">
            <div
              className="w-5 rounded-sm border border-slate-200"
              style={{ height: '120px', background: verticalGradientCSS }}
            />
            <div className="flex flex-col justify-between" style={{ height: '120px' }}>
              {legendLabels.map((val, i) => (
                <span key={i} className="text-[11px] text-slate-600 leading-none">{val.toFixed(0)}</span>
              ))}
            </div>
          </div>
        </div>

        <button
          onClick={() => {
            if (hasCrossSection) {
              handleClearCrossSection();
            } else {
              setCrossSectionMode(true);
              setCrossSectionStart(null);
              setCrossSectionEnd(null);
            }
          }}
          className={`px-3 py-1.5 rounded-lg shadow-lg border text-xs font-medium transition-colors ${
            hasCrossSection
              ? 'bg-red-50 text-red-700 border-red-200 hover:bg-red-100'
              : crossSectionMode
                ? 'bg-blue-100 text-blue-800 border-blue-300'
                : 'bg-white/95 backdrop-blur text-slate-700 border-slate-200 hover:bg-slate-50'
          }`}
        >
          {hasCrossSection ? 'Clear' : 'Cross Section'}
        </button>

        <button
          onClick={onToggleActiveWells}
          className={`px-3 py-1.5 rounded-lg shadow-lg border text-xs font-medium transition-colors ${
            showActiveWells
              ? 'bg-emerald-50 text-emerald-700 border-emerald-200 hover:bg-emerald-100'
              : 'bg-white/95 backdrop-blur text-slate-700 border-slate-200 hover:bg-slate-50'
          }`}
        >
          Active Wells
        </button>
      </div>

      {/* Cursor value tooltip */}
      {cursorValue && !crossSectionMode && (
        <div
          className="absolute z-[96] pointer-events-none"
          style={{ left: cursorValue.x + 14, top: cursorValue.y - 10 }}
        >
          <div className="bg-black/80 text-white text-xs font-medium px-2 py-1 rounded shadow-lg whitespace-nowrap">
            {cursorValue.val.toFixed(1)}
          </div>
        </div>
      )}

      {/* Cross-section drawing overlay */}
      {crossSectionMode && !hasCrossSection && (
        <div
          className="absolute inset-0 z-[500]"
          style={{ cursor: 'crosshair' }}
          onClick={(e) => {
            const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
            const point = L.point(e.clientX - rect.left, e.clientY - rect.top);
            const latlng = map.containerPointToLatLng(point);

            if (!crossSectionStart) {
              setCrossSectionStart(latlng);
              // Show start marker
              startMarkerRef.current = L.circleMarker(latlng, {
                radius: 5, color: '#1e5a8a', fillColor: '#fff', fillOpacity: 1, weight: 2, interactive: false,
              }).addTo(map);
              // Init rubber band line
              rubberBandRef.current = L.polyline([latlng, latlng], {
                color: '#1e5a8a', weight: 2, dashArray: '6 4', interactive: false,
              }).addTo(map);
            } else {
              clearRubberBand();
              setCrossSectionEnd(latlng);
              setCrossSectionMode(false);
              handleCrossSectionComplete(crossSectionStart, latlng);
            }
          }}
          onMouseMove={(e) => {
            if (!crossSectionStart || !rubberBandRef.current) return;
            const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
            const point = L.point(e.clientX - rect.left, e.clientY - rect.top);
            const latlng = map.containerPointToLatLng(point);
            rubberBandRef.current.setLatLngs([crossSectionStart, latlng]);
          }}
          onKeyDown={(e) => {
            if (e.key === 'Escape') {
              setCrossSectionMode(false);
              setCrossSectionStart(null);
              clearRubberBand();
            }
          }}
          tabIndex={0}
          ref={(el) => el?.focus()}
        >
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-black/70 text-white px-4 py-2 rounded-lg text-sm pointer-events-none select-none">
            {crossSectionStart
              ? 'Click to set end point (Esc to cancel)'
              : 'Click to set start point (Esc to cancel)'}
          </div>
        </div>
      )}

    </>
  );
};

export default React.memo(RasterOverlay);
