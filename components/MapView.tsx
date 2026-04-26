
import React, { useEffect, useRef, useMemo, useState, forwardRef, useImperativeHandle } from 'react';
import L from 'leaflet';
import { Layers, ChevronRight, Search } from 'lucide-react';
import { Region, Aquifer, Well, Measurement } from '../types';
import { isPointInGeoJSON } from '../utils/geo';

const BASEMAPS = {
  'OpenStreetMap': {
    url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    thumbnail: 'https://www.arcgis.com/sharing/rest/content/items/b834a68d7a484c5fb473d4ba90571f26/info/thumbnail/ago_downloaded.png'
  },
  'Topographic (Esri)': {
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
    attribution: 'Tiles &copy; Esri',
    thumbnail: 'https://www.arcgis.com/sharing/rest/content/items/67372ff42cd145319639a99152b15bc3/info/thumbnail/ago_downloaded.png'
  },
  'Imagery (Esri)': {
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attribution: 'Tiles &copy; Esri',
    thumbnail: 'https://www.arcgis.com/sharing/rest/content/items/10df2279f9684e4a9f6a7f08febac2a9/info/thumbnail/ago_downloaded.png'
  },
  'Streets (Esri)': {
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
    attribution: 'Tiles &copy; Esri',
    thumbnail: 'https://www.arcgis.com/sharing/rest/content/items/3b93337983e9436f8db950e38a8629af/info/thumbnail/ago_downloaded.png'
  },
  'Light Gray (Esri)': {
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}',
    attribution: 'Tiles &copy; Esri',
    thumbnail: 'https://www.arcgis.com/sharing/rest/content/items/8b3d38c0819547faa83f7b7aca80bd76/info/thumbnail/ago_downloaded.png'
  },
  'Dark Gray (Esri)': {
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
    attribution: 'Tiles &copy; Esri',
    thumbnail: 'https://www.arcgis.com/sharing/rest/content/items/358ec1e175ea41c3bf5c68f0da11ae2b/info/thumbnail/ago_downloaded.png'
  },
  'Terrain (Esri)': {
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}',
    attribution: 'Tiles &copy; Esri',
    thumbnail: 'https://www.arcgis.com/sharing/rest/content/items/c61ad8ab017d49e1a82f580ee1298571/info/thumbnail/ago_downloaded.png'
  }
};

interface MapViewProps {
  regions: Region[];
  aquifers: Aquifer[];
  wells: Well[];
  measurements: Measurement[];
  selectedRegion: Region | null;
  selectedAquifer: Aquifer | null;
  selectedWells: Well[];
  selectedDataType?: string;
  wellColors?: Map<string, string> | null;
  aquiferColors?: Map<string, string> | null;
  rasterActiveWellIds?: Set<string> | null;
  minObs: number;
  onMinObsChange: (value: number) => void;
  onRegionClick: (r: Region) => void;
  onAquiferClick: (a: Aquifer) => void;
  onWellClick: (w: Well, shiftKey: boolean) => void;
  onWellBoxSelect: (wells: Well[]) => void;
  onShowWellsChange?: (show: boolean) => void;
  onShowWellIdsChange?: (show: boolean) => void;
  onDateFilterChange?: (filter: { minYear: number; maxYear: number } | null) => void;
}

export interface MapViewHandle {
  getMap(): L.Map | null;
}

const MapView = forwardRef<MapViewHandle, MapViewProps>(({
  regions,
  aquifers,
  wells,
  measurements,
  selectedRegion,
  selectedAquifer,
  selectedWells,
  selectedDataType = 'wte',
  wellColors,
  aquiferColors,
  rasterActiveWellIds,
  minObs,
  onMinObsChange,
  onRegionClick,
  onAquiferClick,
  onWellClick,
  onWellBoxSelect,
  onShowWellsChange,
  onShowWellIdsChange,
  onDateFilterChange
}, ref) => {
  // Count measurements per well for the active data type (keyed by regionId:aquiferId:wellId)
  const wellMeasurementCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const m of measurements) {
      if (m.dataType === selectedDataType) {
        const key = `${m.regionId}:${m.aquiferId}:${m.wellId}`;
        counts.set(key, (counts.get(key) || 0) + 1);
      }
    }
    return counts;
  }, [measurements, selectedDataType]);

  const mapRef = useRef<L.Map | null>(null);

  useImperativeHandle(ref, () => ({
    getMap: () => mapRef.current,
  }), []);
  const basemapLayerRef = useRef<L.TileLayer | null>(null);
  const regionLayerRef = useRef<L.FeatureGroup | null>(null);
  const aquiferLayerRef = useRef<L.FeatureGroup | null>(null);
  const wellLayerRef = useRef<L.FeatureGroup | null>(null);
  const wellLabelLayerRef = useRef<L.FeatureGroup | null>(null);
  const aquiferLabelLayerRef = useRef<L.FeatureGroup | null>(null);

  const visibleWellsRef = useRef<Well[]>([]);
  const prevSelectedAquiferIdRef = useRef<string | null>(null);
  const prevSelectedRegionIdRef = useRef<string | null>(null);
  const wellJustClickedRef = useRef(false);
  const mouseDownPosRef = useRef<{ x: number; y: number } | null>(null);
  const aquifersRef = useRef(aquifers);
  aquifersRef.current = aquifers;
  const regionsRef = useRef(regions);
  regionsRef.current = regions;
  const selectedRegionRef = useRef(selectedRegion);
  selectedRegionRef.current = selectedRegion;
  const selectedAquiferRef = useRef(selectedAquifer);
  selectedAquiferRef.current = selectedAquifer;
  const onWellBoxSelectRef = useRef(onWellBoxSelect);
  onWellBoxSelectRef.current = onWellBoxSelect;
  const onRegionClickRef = useRef(onRegionClick);
  onRegionClickRef.current = onRegionClick;
  const onAquiferClickRef = useRef(onAquiferClick);
  onAquiferClickRef.current = onAquiferClick;

  const [currentBasemap, setCurrentBasemap] = useState<keyof typeof BASEMAPS>('OpenStreetMap');
  const [isBasemapMenuOpen, setIsBasemapMenuOpen] = useState(false);
  const [showAquiferNames, setShowAquiferNames] = useState(true);
  const [showAquiferIds, setShowAquiferIds] = useState(false);
  const [showWellIds, setShowWellIds] = useState(false);
  const [showWellNames, setShowWellNames] = useState(false);
  const [showWells, setShowWells] = useState(true);
  const [filterDatesEnabled, setFilterDatesEnabled] = useState(false);
  const [filterMinYear, setFilterMinYear] = useState<string>('');
  const [filterMaxYear, setFilterMaxYear] = useState<string>('');
  // "Applied" values only update when input reaches 4 digits — avoids flicker mid-edit
  const [appliedMinYear, setAppliedMinYear] = useState<number | null>(null);
  const [appliedMaxYear, setAppliedMaxYear] = useState<number | null>(null);
  const filterDatesInitialized = useRef(false);
  const [labelFontSize, setLabelFontSize] = useState(9);

  // Compute default date filter bounds from measurements
  const defaultFilterMinYear = useMemo(() => {
    let earliest = Infinity;
    for (const m of measurements) {
      if (m.dataType !== selectedDataType) continue;
      const y = new Date(m.date).getFullYear();
      if (y < earliest) earliest = y;
    }
    return earliest === Infinity ? new Date().getFullYear() : earliest;
  }, [measurements, selectedDataType]);

  // Update applied values only when input is a valid 4-digit year
  useEffect(() => {
    if (filterMinYear.length === 4) {
      const y = parseInt(filterMinYear);
      if (!isNaN(y)) setAppliedMinYear(y);
    }
  }, [filterMinYear]);
  useEffect(() => {
    if (filterMaxYear.length === 4) {
      const y = parseInt(filterMaxYear);
      if (!isNaN(y)) setAppliedMaxYear(y);
    }
  }, [filterMaxYear]);

  // Initialize date filter inputs with defaults when first enabled
  useEffect(() => {
    if (filterDatesEnabled && !filterDatesInitialized.current) {
      filterDatesInitialized.current = true;
      if (!filterMinYear) {
        const defMin = String(defaultFilterMinYear);
        setFilterMinYear(defMin);
        setAppliedMinYear(defaultFilterMinYear);
      }
      if (!filterMaxYear) {
        const curYear = new Date().getFullYear();
        setFilterMaxYear(String(curYear));
        setAppliedMaxYear(curYear);
      }
    }
    if (!filterDatesEnabled) {
      filterDatesInitialized.current = false;
    }
  }, [filterDatesEnabled]);

  // Wells that pass the date filter (measurement span overlaps filter range)
  const dateFilterPassingWells = useMemo(() => {
    if (!filterDatesEnabled || (appliedMinYear == null && appliedMaxYear == null)) return null;
    const filterMin = appliedMinYear != null ? new Date(appliedMinYear, 0, 1).getTime() : -Infinity;
    const filterMax = appliedMaxYear != null ? new Date(appliedMaxYear, 11, 31, 23, 59, 59, 999).getTime() : Infinity;
    // Compute each well's earliest and latest measurement dates
    const wellMin = new Map<string, number>();
    const wellMax = new Map<string, number>();
    for (const m of measurements) {
      if (m.dataType !== selectedDataType) continue;
      const key = `${m.regionId}:${m.aquiferId}:${m.wellId}`;
      const ts = new Date(m.date).getTime();
      const curMin = wellMin.get(key);
      if (curMin === undefined || ts < curMin) wellMin.set(key, ts);
      const curMax = wellMax.get(key);
      if (curMax === undefined || ts > curMax) wellMax.set(key, ts);
    }
    // A well passes if its [earliest, latest] range overlaps [filterMin, filterMax]
    const passing = new Set<string>();
    for (const [key, earliest] of wellMin) {
      const latest = wellMax.get(key)!;
      if (earliest <= filterMax && latest >= filterMin) {
        passing.add(key);
      }
    }
    return passing;
  }, [measurements, selectedDataType, filterDatesEnabled, appliedMinYear, appliedMaxYear]);

  // Notify parent of date filter changes
  useEffect(() => {
    if (filterDatesEnabled && (appliedMinYear != null || appliedMaxYear != null)) {
      onDateFilterChange?.({
        minYear: appliedMinYear ?? 1,
        maxYear: appliedMaxYear ?? 9999,
      });
    } else {
      onDateFilterChange?.(null);
    }
  }, [filterDatesEnabled, appliedMinYear, appliedMaxYear, onDateFilterChange]);

  // Toggle well layer visibility on the map
  useEffect(() => {
    if (!mapRef.current) return;
    if (showWells) {
      if (wellLayerRef.current && !mapRef.current.hasLayer(wellLayerRef.current)) {
        wellLayerRef.current.addTo(mapRef.current);
      }
      if (wellLabelLayerRef.current && !mapRef.current.hasLayer(wellLabelLayerRef.current)) {
        wellLabelLayerRef.current.addTo(mapRef.current);
      }
    } else {
      if (wellLayerRef.current && mapRef.current.hasLayer(wellLayerRef.current)) {
        wellLayerRef.current.removeFrom(mapRef.current);
      }
      if (wellLabelLayerRef.current && mapRef.current.hasLayer(wellLabelLayerRef.current)) {
        wellLabelLayerRef.current.removeFrom(mapRef.current);
      }
    }
    onShowWellsChange?.(showWells);
  }, [showWells, onShowWellsChange]);

  useEffect(() => {
    onShowWellIdsChange?.(showWellIds);
  }, [showWellIds, onShowWellIdsChange]);

  // Box-drag selection state
  // Well search state
  const [wellSearchQuery, setWellSearchQuery] = useState('');
  const [wellSearchFocused, setWellSearchFocused] = useState(false);
  const [wellSearchHighlight, setWellSearchHighlight] = useState(-1);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const searchDropdownRef = useRef<HTMLDivElement>(null);

  const wellSearchMatches = useMemo(() => {
    if (!wellSearchQuery.trim() || !selectedAquifer) return [];
    const q = wellSearchQuery.toLowerCase();
    return visibleWellsRef.current
      .filter(w => w.name.toLowerCase().includes(q) || w.id.toLowerCase().includes(q))
      .slice(0, 8);
  }, [wellSearchQuery, selectedAquifer, wells, minObs, dateFilterPassingWells]);

  const flashWellRef = useRef<L.Marker | null>(null);

  const selectSearchedWell = (well: Well) => {
    onWellClick(well, false);
    const map = mapRef.current;
    if (map) {
      // Remove any previous flash marker
      if (flashWellRef.current) {
        flashWellRef.current.remove();
        flashWellRef.current = null;
      }

      // Delay flyTo so layout settles after chart panel appears
      requestAnimationFrame(() => {
        map.invalidateSize();
        map.flyTo([well.lat, well.lng], Math.max(map.getZoom(), 14), { duration: 1 });

        // Add a pulsing ring after the fly animation settles
        setTimeout(() => {
          if (!mapRef.current) return;
          const flash = L.marker([well.lat, well.lng], {
            interactive: false,
            icon: L.divIcon({
              className: '',
              html: `<div class="well-search-pulse"></div>`,
              iconSize: [0, 0],
              iconAnchor: [0, 0],
            }),
          }).addTo(mapRef.current);
          flashWellRef.current = flash;
          setTimeout(() => { flash.remove(); flashWellRef.current = null; }, 1200);
        }, 1100);
      });
    }
    setWellSearchQuery('');
    setWellSearchFocused(false);
    setWellSearchHighlight(-1);
    searchInputRef.current?.blur();
  };

  // Clear search when aquifer changes
  useEffect(() => {
    setWellSearchQuery('');
    setWellSearchHighlight(-1);
  }, [selectedAquifer]);

  const [shiftHeld, setShiftHeld] = useState(false);
  const [boxDrag, setBoxDrag] = useState<{ startX: number; startY: number; curX: number; curY: number } | null>(null);

  useEffect(() => {
    if (!mapRef.current) {
      mapRef.current = L.map('map-container', { boxZoom: false, zoomSnap: 0.5, zoomDelta: 0.5 }).setView([39, -98], 4);

      const basemap = BASEMAPS[currentBasemap];
      basemapLayerRef.current = L.tileLayer(basemap.url, {
        attribution: basemap.attribution
      }).addTo(mapRef.current);

      regionLayerRef.current = L.featureGroup().addTo(mapRef.current);
      aquiferLayerRef.current = L.featureGroup().addTo(mapRef.current);
      aquiferLabelLayerRef.current = L.featureGroup().addTo(mapRef.current);
      wellLayerRef.current = L.featureGroup().addTo(mapRef.current);
      wellLabelLayerRef.current = L.featureGroup().addTo(mapRef.current);

      // Track mousedown position to distinguish clicks from drags/pans
      const map = mapRef.current;
      map.getContainer().addEventListener('mousedown', (evt: MouseEvent) => {
        mouseDownPosRef.current = { x: evt.clientX, y: evt.clientY };
      });

      // Single native DOM click handler for all map background clicks.
      // Well markers stop native propagation, so this only fires for non-well clicks.
      map.getContainer().addEventListener('click', (evt: MouseEvent) => {
        if (wellJustClickedRef.current) return;
        // Ignore drag/pan gestures — only handle clicks where mouse barely moved
        const down = mouseDownPosRef.current;
        if (down) {
          const dx = evt.clientX - down.x;
          const dy = evt.clientY - down.y;
          if (dx * dx + dy * dy > 25) return; // >5px movement = drag
        }
        const latlng = map.mouseEventToLatLng(evt);
        const { lat, lng } = latlng;
        const curRegion = selectedRegionRef.current;
        const curAquifer = selectedAquiferRef.current;

        if (curAquifer && curRegion) {
          // Aquifer is selected — determine what was clicked
          let clickedAquifer: Aquifer | null = null;
          for (const a of aquifersRef.current) {
            if (isPointInGeoJSON(lat, lng, a.geojson)) { clickedAquifer = a; break; }
          }
          if (clickedAquifer) {
            if (clickedAquifer.id !== curAquifer.id) {
              onAquiferClickRef.current(clickedAquifer);
            } else {
              // Same aquifer — just clear wells
              onWellBoxSelectRef.current([]);
            }
          } else if (isPointInGeoJSON(lat, lng, curRegion.geojson)) {
            // Inside region but outside all aquifers → deselect aquifer
            onRegionClickRef.current(curRegion);
          } else {
            // Check other regions
            let clickedRegion: Region | null = null;
            for (const r of regionsRef.current) {
              if (r.id !== curRegion.id && isPointInGeoJSON(lat, lng, r.geojson)) { clickedRegion = r; break; }
            }
            if (clickedRegion) onRegionClickRef.current(clickedRegion);
            else onWellBoxSelectRef.current([]);
          }
        } else if (curRegion) {
          // Region selected, no aquifer — check for aquifer or other region click
          let clickedAquifer: Aquifer | null = null;
          for (const a of aquifersRef.current) {
            if (isPointInGeoJSON(lat, lng, a.geojson)) { clickedAquifer = a; break; }
          }
          if (clickedAquifer) {
            onAquiferClickRef.current(clickedAquifer);
          } else {
            let clickedRegion: Region | null = null;
            for (const r of regionsRef.current) {
              if (r.id !== curRegion.id && isPointInGeoJSON(lat, lng, r.geojson)) { clickedRegion = r; break; }
            }
            if (clickedRegion) onRegionClickRef.current(clickedRegion);
            else onWellBoxSelectRef.current([]);
          }
        } else {
          // No region selected — check for region click
          let clickedRegion: Region | null = null;
          for (const r of regionsRef.current) {
            if (isPointInGeoJSON(lat, lng, r.geojson)) { clickedRegion = r; break; }
          }
          if (clickedRegion) onRegionClickRef.current(clickedRegion);
          else onWellBoxSelectRef.current([]);
        }
      });
    }
  }, []);

  // Handle basemap changes
  const changeBasemap = (name: keyof typeof BASEMAPS) => {
    if (!mapRef.current) return;

    if (basemapLayerRef.current) {
      mapRef.current.removeLayer(basemapLayerRef.current);
    }

    const basemap = BASEMAPS[name];
    basemapLayerRef.current = L.tileLayer(basemap.url, {
      attribution: basemap.attribution
    }).addTo(mapRef.current);

    // Move basemap to back so other layers stay on top
    basemapLayerRef.current.bringToBack();

    setCurrentBasemap(name);
  };

  // Update Region Layer
  useEffect(() => {
    if (!regionLayerRef.current || !mapRef.current) return;
    regionLayerRef.current.clearLayers();

    regions.forEach(r => {
      const isSelected = selectedRegion?.id === r.id;
      const layer = L.geoJSON(r.geojson, {
        interactive: false,
        style: {
          color: isSelected ? '#2563eb' : '#94a3b8',
          weight: isSelected ? 3 : 1,
          fillOpacity: isSelected ? 0.05 : 0.1,
          fillColor: '#2563eb'
        },
      });
      regionLayerRef.current?.addLayer(layer);
    });

    if (!selectedRegion && regions.length > 0) {
      const allBounds = L.latLngBounds(
        regions.map(r => [
          L.latLng(r.bounds[0], r.bounds[1]),
          L.latLng(r.bounds[2], r.bounds[3])
        ]).flat()
      );
      if (allBounds.isValid()) {
        const zoom = Math.max(mapRef.current.getBoundsZoom(allBounds, false, L.point(20, 20)), 2.5);
        mapRef.current.setView(allBounds.getCenter(), zoom, { animate: false });
      }
    }
  }, [regions, selectedRegion]);

  // Update Aquifer Layer (polygons only)
  useEffect(() => {
    if (!aquiferLayerRef.current || !mapRef.current) return;
    aquiferLayerRef.current.clearLayers();

    if (selectedRegion) {
      aquifers.forEach(a => {
        const isSelected = selectedAquifer?.id === a.id;
        const trendColor = aquiferColors?.get(a.id);
        const layer = L.geoJSON(a.geojson, {
          interactive: false,
          style: {
            color: isSelected ? '#6366f1' : trendColor ? '#000000' : '#475569',
            weight: isSelected ? 5 : trendColor ? 3 : 2,
            fillOpacity: isSelected ? 0 : trendColor ? 0.45 : 0.15,
            fillColor: trendColor || '#64748b'
          }
        });
        aquiferLayerRef.current?.addLayer(layer);
      });

      const regionChanged = selectedRegion.id !== prevSelectedRegionIdRef.current;
      const aquiferDeselected = !selectedAquifer && prevSelectedAquiferIdRef.current !== null;
      if ((regionChanged || aquiferDeselected) && !selectedAquifer) {
        const rBounds = L.latLngBounds(
          [selectedRegion.bounds[0], selectedRegion.bounds[1]],
          [selectedRegion.bounds[2], selectedRegion.bounds[3]]
        );
        mapRef.current.flyToBounds(rBounds, { padding: [40, 40], duration: 1.5 });
      }
      prevSelectedRegionIdRef.current = selectedRegion.id;
    } else {
      prevSelectedRegionIdRef.current = null;
    }
  }, [aquifers, selectedRegion, selectedAquifer, aquiferColors]);

  // Aquifer name labels (separate from polygons so font size changes don't rebuild polygons)
  useEffect(() => {
    aquiferLabelLayerRef.current?.clearLayers();
    if ((!showAquiferNames && !showAquiferIds) || !selectedRegion) return;

    aquifers.forEach(a => {
      if (selectedAquifer?.id === a.id) return;
      const [lat, lng] = a.labelPoint;
      const aFontSize = Math.round(labelFontSize * 1.2);
      const labelText = showAquiferNames && showAquiferIds
        ? `${a.name} (${a.id})`
        : showAquiferNames ? a.name : a.id;
      const label = L.marker([lat, lng], {
        icon: L.divIcon({
          className: '',
          html: `<div style="white-space:nowrap;font-size:${aFontSize}px;color:#fff;text-shadow:0 0 3px #000,0 0 6px #000;pointer-events:none;font-weight:600;text-align:center">${labelText}</div>`,
          iconSize: [400, 20],
          iconAnchor: [200, 10],
        }),
        interactive: false,
      });
      aquiferLabelLayerRef.current?.addLayer(label);
    });
  }, [aquifers, selectedRegion, selectedAquifer, showAquiferNames, showAquiferIds, labelFontSize]);

  // Build well markers — only when wells/aquifer/filter changes, NOT on selection change
  const wellMarkerMapRef = useRef<Map<string, L.CircleMarker>>(new Map());

  useEffect(() => {
    if (!wellLayerRef.current || !mapRef.current) return;
    wellLayerRef.current.clearLayers();
    wellMarkerMapRef.current.clear();

    const visible: Well[] = [];
    if (selectedAquifer) {
      wells.forEach(w => {
        const wellKey = `${w.regionId}:${w.aquiferId}:${w.id}`;
        const measurementCount = wellMeasurementCounts.get(wellKey) || 0;
        if (measurementCount < minObs) return;
        if (dateFilterPassingWells && !dateFilterPassingWells.has(wellKey)) return;
        visible.push(w);
        const trendColor = wellColors?.get(w.id);
        // Color by measurement count: 0=red, 1=gray, 2+=blue
        const defaultColor = measurementCount === 0 ? '#ef4444'
          : measurementCount === 1 ? '#6b7280'
          : '#0000ff';
        const marker = L.circleMarker([w.lat, w.lng], {
          radius: 6,
          fillColor: trendColor || defaultColor,
          color: trendColor ? '#000000' : '#ffffff',
          weight: 1,
          opacity: 1,
          fillOpacity: 0.9
        });
        marker.bindTooltip(`Well: ${w.name}<br/>ID: ${w.id}${w.gse ? `<br/>GSE: ${w.gse}` : ''}<br/>Observations: ${measurementCount}`, { direction: 'top' });
        marker.on('click', (e) => {
          // Stop the native DOM event from reaching the map container
          const ne = (e as any).originalEvent;
          if (ne) ne.stopPropagation();
          // Backup flag in case native stopPropagation doesn't prevent the container listener
          wellJustClickedRef.current = true;
          setTimeout(() => { wellJustClickedRef.current = false; }, 50);
          const shiftKey = ne?.shiftKey ?? false;
          onWellClick(w, shiftKey);
        });
        wellLayerRef.current?.addLayer(marker);
        wellMarkerMapRef.current.set(w.id, marker);
      });

      const aquiferChanged = selectedAquifer.id !== prevSelectedAquiferIdRef.current;
      if (aquiferChanged) {
        const aBounds = L.latLngBounds(
          [selectedAquifer.bounds[0], selectedAquifer.bounds[1]],
          [selectedAquifer.bounds[2], selectedAquifer.bounds[3]]
        );
        mapRef.current.flyToBounds(aBounds, { padding: [40, 40], duration: 1 });
      }
      prevSelectedAquiferIdRef.current = selectedAquifer.id;
    } else {
      prevSelectedAquiferIdRef.current = null;
    }
    visibleWellsRef.current = visible;
  }, [wells, selectedAquifer, wellMeasurementCounts, minObs, dateFilterPassingWells, wellColors]);

  // Update marker styles when selection or raster active wells change — no clearing/recreation
  useEffect(() => {
    const selectedIds = new Set(selectedWells.map(w => w.id));
    const wellKeyMap = new Map(wells.map(w => [w.id, `${w.regionId}:${w.aquiferId}:${w.id}`]));
    wellMarkerMapRef.current.forEach((marker, wellId) => {
      const isSelected = selectedIds.has(wellId);
      const hasTrend = wellColors?.has(wellId);
      // Raster mode: green = active contributor, dark gray = not contributing
      const rasterFill = rasterActiveWellIds
        ? (rasterActiveWellIds.has(wellId) ? '#22c55e' : '#4b5563')
        : null;
      const measurementCount = wellMeasurementCounts.get(wellKeyMap.get(wellId) || '') || 0;
      const defaultFill = measurementCount === 0 ? '#ef4444'
        : measurementCount === 1 ? '#6b7280'
        : '#0000ff';
      marker.setStyle({
        radius: isSelected ? 8 : 6,
        fillColor: hasTrend ? wellColors!.get(wellId)! : rasterFill || defaultFill,
        color: isSelected ? '#f59e0b' : hasTrend ? '#000000' : '#ffffff',
        weight: isSelected ? 2 : 1,
      });
    });
  }, [selectedWells, wells, wellColors, rasterActiveWellIds, wellMeasurementCounts]);

  // Well labels
  useEffect(() => {
    wellLabelLayerRef.current?.clearLayers();
    if (!showWellIds && !showWellNames) return;
    if (!selectedAquifer) return;

    wellMarkerMapRef.current.forEach((marker, wellId) => {
      const well = wells.find(w => w.id === wellId);
      if (!well) return;

      let text = '';
      if (showWellIds && showWellNames) {
        text = `${well.name} (${well.id})`;
      } else if (showWellNames) {
        text = well.name;
      } else {
        text = well.id;
      }

      const label = L.marker([well.lat, well.lng], {
        icon: L.divIcon({
          className: '',
          html: `<div style="white-space:nowrap;font-size:${labelFontSize}px;color:#1e293b;text-shadow:0 0 2px #fff,0 0 4px #fff;pointer-events:none;font-weight:500;margin-left:8px;margin-top:-${labelFontSize + 5}px">${text}</div>`,
          iconSize: [0, 0],
          iconAnchor: [0, 0],
        }),
        interactive: false,
      });
      wellLabelLayerRef.current?.addLayer(label);
    });
  }, [showWellIds, showWellNames, selectedAquifer, wells, minObs, dateFilterPassingWells, wellMeasurementCounts, labelFontSize]);

  // Track shift key for box-drag overlay
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => { if (e.key === 'Shift') setShiftHeld(true); };
    const onKeyUp = (e: KeyboardEvent) => { if (e.key === 'Shift') setShiftHeld(false); };
    const onBlur = () => setShiftHeld(false);
    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', onBlur);
    return () => {
      document.removeEventListener('keydown', onKeyDown);
      document.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', onBlur);
    };
  }, []);

  // Overlay pointer handlers for box-drag selection
  const handleOverlayPointerDown = (e: React.PointerEvent) => {
    e.preventDefault();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    const container = mapRef.current?.getContainer();
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setBoxDrag({ startX: x, startY: y, curX: x, curY: y });
  };

  const handleOverlayPointerMove = (e: React.PointerEvent) => {
    if (!boxDrag) return;
    const container = mapRef.current?.getContainer();
    if (!container) return;
    const rect = container.getBoundingClientRect();
    setBoxDrag(prev => prev ? { ...prev, curX: e.clientX - rect.left, curY: e.clientY - rect.top } : null);
  };

  const handleOverlayPointerUp = (e: React.PointerEvent) => {
    if (!boxDrag) return;
    const map = mapRef.current;
    if (!map) { setBoxDrag(null); return; }

    const container = map.getContainer();
    const rect = container.getBoundingClientRect();
    const endX = e.clientX - rect.left;
    const endY = e.clientY - rect.top;

    const dx = Math.abs(endX - boxDrag.startX);
    const dy = Math.abs(endY - boxDrag.startY);

    if (dx > 5 || dy > 5) {
      // It was a drag — box select
      const minPx = L.point(Math.min(boxDrag.startX, endX), Math.min(boxDrag.startY, endY));
      const maxPx = L.point(Math.max(boxDrag.startX, endX), Math.max(boxDrag.startY, endY));
      const sw = map.containerPointToLatLng(L.point(minPx.x, maxPx.y));
      const ne = map.containerPointToLatLng(L.point(maxPx.x, minPx.y));
      const selBounds = L.latLngBounds(sw, ne);
      const matched = visibleWellsRef.current.filter(w =>
        selBounds.contains(L.latLng(w.lat, w.lng))
      );
      if (matched.length > 0) {
        onWellBoxSelect(matched);
      }
    } else {
      // It was a click — find nearest well and toggle it
      let nearest: Well | null = null;
      let nearestDist = Infinity;
      for (const w of visibleWellsRef.current) {
        const wellPx = map.latLngToContainerPoint(L.latLng(w.lat, w.lng));
        const dist = Math.sqrt((wellPx.x - endX) ** 2 + (wellPx.y - endY) ** 2);
        if (dist < 20 && dist < nearestDist) {
          nearest = w;
          nearestDist = dist;
        }
      }
      if (nearest) {
        onWellClick(nearest, true);
      }
    }

    setBoxDrag(null);
  };

  // Show overlay when shift is held (or mid-drag even if shift released)
  const showOverlay = (shiftHeld || boxDrag) && selectedAquifer;

  return (
    <div className="relative w-full h-full">
      <div id="map-container" className="w-full h-full" />

      {/* Shift-drag overlay for box selection */}
      {showOverlay && (
        <div
          className="absolute inset-0 z-[500] cursor-crosshair"
          onPointerDown={handleOverlayPointerDown}
          onPointerMove={handleOverlayPointerMove}
          onPointerUp={handleOverlayPointerUp}
        >
          {boxDrag && (
            <div
              style={{
                position: 'absolute',
                left: Math.min(boxDrag.startX, boxDrag.curX),
                top: Math.min(boxDrag.startY, boxDrag.curY),
                width: Math.abs(boxDrag.curX - boxDrag.startX),
                height: Math.abs(boxDrag.curY - boxDrag.startY),
                border: '2px dashed #3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.15)',
                pointerEvents: 'none',
              }}
            />
          )}
        </div>
      )}

      {/* Well Search */}
      {selectedAquifer && (
        <div className="absolute top-3 left-14 z-[90]" style={{ width: '260px' }}>
          <div className="relative">
            <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" />
            <input
              ref={searchInputRef}
              type="text"
              placeholder="Search wells..."
              value={wellSearchQuery}
              onChange={(e) => { setWellSearchQuery(e.target.value); setWellSearchHighlight(-1); }}
              onFocus={() => setWellSearchFocused(true)}
              onBlur={() => { setTimeout(() => setWellSearchFocused(false), 150); }}
              onKeyDown={(e) => {
                if (e.key === 'Escape') {
                  setWellSearchQuery('');
                  setWellSearchFocused(false);
                  searchInputRef.current?.blur();
                } else if (e.key === 'ArrowDown') {
                  e.preventDefault();
                  setWellSearchHighlight(prev => Math.min(prev + 1, wellSearchMatches.length - 1));
                } else if (e.key === 'ArrowUp') {
                  e.preventDefault();
                  setWellSearchHighlight(prev => Math.max(prev - 1, 0));
                } else if (e.key === 'Enter' && wellSearchMatches.length > 0) {
                  e.preventDefault();
                  const idx = wellSearchHighlight >= 0 ? wellSearchHighlight : 0;
                  selectSearchedWell(wellSearchMatches[idx]);
                }
              }}
              className="w-full pl-8 pr-3 py-1.5 text-sm bg-white border border-slate-300 rounded-lg shadow-md focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-blue-400"
            />
          </div>
          {wellSearchFocused && wellSearchQuery.trim() && (
            <div ref={searchDropdownRef} className="mt-1 bg-white border border-slate-200 rounded-lg shadow-lg overflow-hidden max-h-64 overflow-y-auto">
              {wellSearchMatches.length === 0 ? (
                <div className="px-3 py-2 text-xs text-slate-400">No matching wells</div>
              ) : (
                wellSearchMatches.map((w, i) => (
                  <button
                    key={w.id}
                    onMouseDown={(e) => { e.preventDefault(); selectSearchedWell(w); }}
                    className={`w-full text-left px-3 py-1.5 text-sm hover:bg-blue-50 flex flex-col ${
                      i === wellSearchHighlight ? 'bg-blue-50' : ''
                    }`}
                  >
                    <span className="font-medium text-slate-800 truncate">{w.name}</span>
                    <span className="text-xs text-slate-400 truncate">ID: {w.id}</span>
                  </button>
                ))
              )}
            </div>
          )}
        </div>
      )}

      {/* Map Options Panel */}
      <div className="absolute bottom-2 left-2 z-[90] flex flex-col gap-1.5 bg-white rounded-lg shadow-lg border border-slate-200 px-2 py-1.5">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1">
            <label htmlFor="min-obs" className="text-xs font-medium text-slate-600 whitespace-nowrap">Min obs</label>
            <input
              id="min-obs"
              type="number"
              min={0}
              step={1}
              value={minObs}
              onChange={(e) => onMinObsChange(Math.max(0, parseInt(e.target.value) || 0))}
              className="w-14 text-xs text-center border border-slate-300 rounded px-1 py-0.5 focus:outline-none focus:ring-1 focus:ring-blue-400"
            />
          </div>
          <div className="flex items-center gap-1">
            <label htmlFor="label-font" className="text-xs font-medium text-slate-600 whitespace-nowrap">Font</label>
            <input
              id="label-font"
              type="number"
              min={6}
              max={24}
              step={1}
              value={labelFontSize}
              onChange={(e) => setLabelFontSize(Math.max(6, Math.min(24, parseInt(e.target.value) || 9)))}
              className="w-12 text-xs text-center border border-slate-300 rounded px-1 py-0.5 focus:outline-none focus:ring-1 focus:ring-blue-400"
            />
          </div>
        </div>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-1 cursor-pointer">
            <input type="checkbox" checked={filterDatesEnabled} onChange={(e) => setFilterDatesEnabled(e.target.checked)} className="w-3 h-3" />
            <span className="text-xs font-medium text-slate-600 whitespace-nowrap">Filter dates:</span>
          </label>
          <input
            type="number"
            min={1800}
            max={2200}
            step={1}
            placeholder="min"
            value={filterMinYear}
            onChange={(e) => setFilterMinYear(e.target.value)}
            disabled={!filterDatesEnabled}
            className="w-14 text-xs text-center border border-slate-300 rounded px-1 py-0.5 focus:outline-none focus:ring-1 focus:ring-blue-400 disabled:opacity-40"
          />
          <span className="text-xs text-slate-400">–</span>
          <input
            type="number"
            min={1800}
            max={2200}
            step={1}
            placeholder="max"
            value={filterMaxYear}
            onChange={(e) => setFilterMaxYear(e.target.value)}
            disabled={!filterDatesEnabled}
            className="w-14 text-xs text-center border border-slate-300 rounded px-1 py-0.5 focus:outline-none focus:ring-1 focus:ring-blue-400 disabled:opacity-40"
          />
        </div>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-1 cursor-pointer">
            <input type="checkbox" checked={showAquiferNames} onChange={(e) => setShowAquiferNames(e.target.checked)} className="w-3 h-3" />
            <span className="text-xs text-slate-600">Aquifer names</span>
          </label>
          <label className="flex items-center gap-1 cursor-pointer">
            <input type="checkbox" checked={showAquiferIds} onChange={(e) => setShowAquiferIds(e.target.checked)} className="w-3 h-3" />
            <span className="text-xs text-slate-600">Aquifer IDs</span>
          </label>
        </div>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-1 cursor-pointer">
            <input type="checkbox" checked={showWells} onChange={(e) => setShowWells(e.target.checked)} className="w-3 h-3" />
            <span className="text-xs text-slate-600">Wells</span>
          </label>
          <label className="flex items-center gap-1 cursor-pointer">
            <input type="checkbox" checked={showWellNames} onChange={(e) => setShowWellNames(e.target.checked)} className="w-3 h-3" />
            <span className="text-xs text-slate-600">Well names</span>
          </label>
          <label className="flex items-center gap-1 cursor-pointer">
            <input type="checkbox" checked={showWellIds} onChange={(e) => setShowWellIds(e.target.checked)} className="w-3 h-3" />
            <span className="text-xs text-slate-600">Well IDs</span>
          </label>
        </div>
      </div>

      {/* Basemap Gallery */}
      <div className="absolute top-3 right-3 z-[90]">
        {!isBasemapMenuOpen ? (
          /* Collapsed - just the icon button */
          <button
            onClick={() => setIsBasemapMenuOpen(true)}
            className="flex items-center justify-center w-8 h-8 bg-white rounded shadow-md border border-slate-300 hover:bg-slate-50 transition-colors"
            title="Basemap Gallery"
          >
            <Layers size={16} className="text-slate-600" />
          </button>
        ) : (
          /* Expanded - gallery panel */
          <div className="bg-white rounded shadow-lg border border-slate-300 overflow-hidden" style={{ width: '260px' }}>
            {/* Header with collapse button */}
            <div className="flex items-center justify-end px-2 py-1 bg-white border-b border-slate-200">
              <button
                onClick={() => setIsBasemapMenuOpen(false)}
                className="flex items-center justify-center w-6 h-6 hover:bg-slate-100 rounded transition-colors"
                title="Collapse"
              >
                <ChevronRight size={16} className="text-slate-500" />
              </button>
            </div>

            {/* Basemap List */}
            <div className="max-h-80 overflow-y-auto">
              {Object.entries(BASEMAPS).map(([name, config]) => (
                <button
                  key={name}
                  onClick={() => changeBasemap(name as keyof typeof BASEMAPS)}
                  className={`w-full flex items-center gap-3 p-2 text-left transition-colors border-2 ${
                    currentBasemap === name
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-transparent hover:bg-slate-50'
                  }`}
                >
                  <img
                    src={config.thumbnail}
                    alt={name}
                    className="w-16 h-16 object-cover rounded flex-shrink-0"
                    onError={(e) => {
                      (e.target as HTMLImageElement).style.display = 'none';
                    }}
                  />
                  <span className={`text-sm ${
                    currentBasemap === name ? 'font-medium text-slate-900' : 'text-slate-700'
                  }`}>
                    {name}
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
});

export default MapView;
