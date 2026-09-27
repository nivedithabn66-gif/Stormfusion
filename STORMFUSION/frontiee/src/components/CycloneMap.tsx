import React, { useState } from 'react';
import { 
  MapContainer, 
  TileLayer, 
  Polyline, 
  Polygon, 
  Marker, 
  Popup, 
  CircleMarker,
  LayersControl
} from 'react-leaflet';
import L from 'leaflet';
import { 
  Layers, 
  Compass, 
  Eye, 
  ShieldAlert, 
  Info, 
  Maximize2, 
  Crosshair,
  Wind,
  Radio,
  Navigation,
  Activity,
  AlertTriangle,
  CloudRain
} from 'lucide-react';
import { CycloneTelemetry, TrackPoint, DistrictRisk } from '../types/cyclone';
import { formatCoords, getCategoryColor, getRiskSeverityColor } from '../utils/formatters';
import { DemoModeBadge } from './DemoModeBadge';
import { useTheme } from '../context/ThemeContext';
import { mockRainfallForecastZones, mockRainfallSwaths } from '../data/mockForecast';

interface CycloneMapProps {
  cyclone: CycloneTelemetry;
  track: TrackPoint[];
  uncertaintyConeCoords: [number, number][];
  districtRisks: DistrictRisk[];
  heightClass?: string;
  defaultShowRainfall?: boolean;
  onSelectDistrict?: (district: DistrictRisk) => void;
}

// Custom Rainfall Pin Icon with pulse & rain styling
const createRainfallMarkerIcon = (intensity: string) => {
  const isExtreme = intensity === 'Extremely Heavy';
  const isVeryHeavy = intensity === 'Very Heavy';
  const color = isExtreme ? '#EF4444' : isVeryHeavy ? '#2563EB' : '#0284C7';

  return L.divIcon({
    className: 'rainfall-marker-pin',
    html: `
      <div class="relative flex items-center justify-center pointer-events-auto cursor-pointer" style="width: 32px; height: 32px;">
        <div class="absolute inset-0 rounded-full animate-ping opacity-60" style="background-color: ${color};"></div>
        <div class="relative w-8 h-8 rounded-full border-2 border-white dark:border-slate-900 shadow-xl flex items-center justify-center font-mono font-black text-xs text-white" style="background: linear-gradient(135deg, ${color}, #1E3A8A);">
          🌧
        </div>
      </div>
    `,
    iconSize: [32, 32],
    iconAnchor: [16, 16],
    popupAnchor: [0, -16]
  });
};

// Realistic Meteorological Cyclone Satellite Vortex (Natural Cloud Bands, Eyewall & Storm Eye)
const createRealisticCycloneIcon = (cyclone: CycloneTelemetry) => {
  return L.divIcon({
    className: 'realistic-cyclone-vortex',
    html: `
      <div class="relative flex items-center justify-center" style="width: 180px; height: 180px;">
        <!-- Satellite-Style Realistic Cloud Vortex & Spiral Bands SVG -->
        <svg width="180" height="180" viewBox="0 0 180 180" class="absolute inset-0 pointer-events-none">
          <defs>
            <!-- Atmospheric Cirrus Outflow Halo -->
            <radialGradient id="satCloudGlow" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stop-color="#FFFFFF" stop-opacity="0.32" />
              <stop offset="35%" stop-color="#F1F5F9" stop-opacity="0.2" />
              <stop offset="70%" stop-color="#94A3B8" stop-opacity="0.08" />
              <stop offset="100%" stop-color="#38BDF8" stop-opacity="0" />
            </radialGradient>

            <!-- Central Dense Overcast (CDO) Core Gradient -->
            <radialGradient id="cdoDenseGrad" cx="48%" cy="48%" r="48%">
              <stop offset="0%" stop-color="#FFFFFF" stop-opacity="0.96" />
              <stop offset="40%" stop-color="#F8FAFC" stop-opacity="0.9" />
              <stop offset="75%" stop-color="#E2E8F0" stop-opacity="0.65" />
              <stop offset="100%" stop-color="#CBD5E1" stop-opacity="0.15" />
            </radialGradient>

            <!-- Primary Cyclonic Spiral Cloud Band Gradient -->
            <linearGradient id="satCloudBand1" x1="10%" y1="90%" x2="80%" y2="20%">
              <stop offset="0%" stop-color="#FFFFFF" stop-opacity="0.15" />
              <stop offset="25%" stop-color="#F8FAFC" stop-opacity="0.82" />
              <stop offset="60%" stop-color="#FFFFFF" stop-opacity="0.95" />
              <stop offset="85%" stop-color="#E2E8F0" stop-opacity="0.8" />
              <stop offset="100%" stop-color="#CBD5E1" stop-opacity="0.2" />
            </linearGradient>

            <!-- Secondary Spiral Inflow Arm Gradient -->
            <linearGradient id="satCloudBand2" x1="90%" y1="10%" x2="20%" y2="80%">
              <stop offset="0%" stop-color="#FFFFFF" stop-opacity="0.15" />
              <stop offset="30%" stop-color="#F1F5F9" stop-opacity="0.75" />
              <stop offset="70%" stop-color="#FFFFFF" stop-opacity="0.92" />
              <stop offset="100%" stop-color="#E2E8F0" stop-opacity="0.25" />
            </linearGradient>

            <!-- Tertiary High-Reflectivity Convective Arm Gradient -->
            <linearGradient id="satCloudBand3" x1="0%" y1="30%" x2="100%" y2="70%">
              <stop offset="0%" stop-color="#F8FAFC" stop-opacity="0.88" />
              <stop offset="50%" stop-color="#FFFFFF" stop-opacity="0.95" />
              <stop offset="100%" stop-color="#CBD5E1" stop-opacity="0.3" />
            </linearGradient>

            <!-- Gaussian Softness Filters for Photorealistic Cloud Texture -->
            <filter id="cloudAtmosphereBlur" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="5.5" result="blur" />
            </filter>
            <filter id="cloudFeatherSoft" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="2.8" result="blur" />
            </filter>
            <filter id="cloudStructureBlur" x="-15%" y="-15%" width="130%" height="130%">
              <feGaussianBlur stdDeviation="1.2" result="blur" />
            </filter>
          </defs>

          <!-- 1. Outer Atmospheric Cirrus Canopy & Natural Irregular Outflow Glow -->
          <path
            d="M 90,12 C 132,8 168,32 174,72 C 180,112 152,156 112,168 C 72,176 26,160 14,120 C 4,80 36,26 90,12 Z"
            fill="url(#satCloudGlow)"
            filter="url(#cloudAtmosphereBlur)"
          />

          <!-- 2. Rotating Satellite Cloud Vortex (Natural Spiral Feeder Arms) -->
          <g class="cyclone-vortex-spin" style="transform-origin: 90px 90px;">
            <!-- Primary Outer Convective Feeder Band (Spiraling Inward Counter-Clockwise) -->
            <path
              d="M 165,125 C 174,88 158,45 124,24 C 90,4 42,20 22,54 C 4,88 18,132 56,152 C 92,172 135,155 148,118 C 158,85 138,54 110,46 C 84,38 64,54 60,78 C 56,98 70,114 88,114 C 102,114 112,102 110,88 C 108,78 98,72 88,75"
              stroke="url(#satCloudBand1)"
              stroke-width="12"
              stroke-linecap="round"
              fill="none"
              filter="url(#cloudFeatherSoft)"
              opacity="0.85"
            />

            <!-- Secondary Spiral Inflow Arm -->
            <path
              d="M 16,62 C 10,105 32,148 72,164 C 112,180 155,158 170,115 C 182,72 155,28 112,15 C 74,3 38,30 34,68 C 30,102 56,130 90,128 C 116,126 136,104 132,78 C 128,56 108,46 88,50 C 72,54 64,68 68,82 C 72,94 84,100 94,94"
              stroke="url(#satCloudBand2)"
              stroke-width="9.5"
              stroke-linecap="round"
              fill="none"
              filter="url(#cloudFeatherSoft)"
              opacity="0.8"
            />

            <!-- Tertiary High-Density Convective Stream -->
            <path
              d="M 172,150 C 152,160 128,152 108,138 C 80,116 70,80 86,58 C 98,40 122,44 130,62 C 136,78 124,96 108,100 C 94,102 85,92 88,82"
              stroke="url(#satCloudBand3)"
              stroke-width="6.5"
              stroke-linecap="round"
              fill="none"
              filter="url(#cloudStructureBlur)"
              opacity="0.88"
            />

            <!-- Outer Wispy Cirrus Outflow Strands (Towards Coast & Northwest Trajectory) -->
            <path
              d="M 145,28 C 160,45 162,72 150,96"
              stroke="#FFFFFF"
              stroke-width="3"
              stroke-linecap="round"
              stroke-opacity="0.65"
              fill="none"
              filter="url(#cloudFeatherSoft)"
            />
            <path
              d="M 35,148 C 18,130 15,102 28,78"
              stroke="#FFFFFF"
              stroke-width="3"
              stroke-linecap="round"
              stroke-opacity="0.6"
              fill="none"
              filter="url(#cloudFeatherSoft)"
            />
            <path
              d="M 72,16 C 42,28 24,56 22,88"
              stroke="#F1F5F9"
              stroke-width="2.5"
              stroke-linecap="round"
              stroke-opacity="0.55"
              fill="none"
              filter="url(#cloudFeatherSoft)"
            />

            <!-- Central Dense Overcast (CDO) Asymmetric Cloud Shield -->
            <path
              d="M 90,62 C 108,60 122,70 125,85 C 128,100 118,116 104,120 C 88,124 72,116 66,102 C 60,86 72,68 86,63 Z"
              fill="url(#cdoDenseGrad)"
              filter="url(#cloudStructureBlur)"
              opacity="0.94"
            />

            <!-- Annular Eyewall Convective Cloud Ring -->
            <circle
              cx="90"
              cy="90"
              r="13"
              stroke="#FFFFFF"
              stroke-width="4.5"
              fill="none"
              stroke-opacity="0.96"
              filter="url(#cloudStructureBlur)"
            />
            <circle
              cx="90"
              cy="90"
              r="10.5"
              stroke="#F8FAFC"
              stroke-width="2.5"
              fill="none"
              stroke-opacity="0.9"
              stroke-dasharray="6 2"
            />
          </g>

          <!-- 3. Fixed Calm Storm Eye (Dark Oceanic Core & Center Reticle) -->
          <circle
            cx="90"
            cy="90"
            r="7"
            fill="rgba(6, 12, 22, 0.88)"
            stroke="rgba(255, 255, 255, 0.7)"
            stroke-width="1"
          />
          <circle cx="90" cy="90" r="2" fill="#FFFFFF" />
          <circle
            cx="90"
            cy="90"
            r="4.5"
            stroke="#38BDF8"
            stroke-width="0.8"
            fill="none"
            class="cyclone-eye-pulse"
          />
        </svg>

        <!-- Floating Tactical Vortex Telemetry Badge -->
        <div class="absolute -bottom-7 left-1/2 -translate-x-1/2 whitespace-nowrap bg-slate-950/95 text-slate-100 border border-cyan-500/50 text-[10px] font-mono font-bold px-2 py-0.5 rounded-md shadow-2xl backdrop-blur-md flex items-center gap-1.5 z-20">
          <span class="w-2 h-2 rounded-full bg-rose-500 animate-pulse shrink-0"></span>
          <span class="text-rose-400 font-extrabold">${cyclone.maxSustainedWindKts} kt</span>
          <span class="text-slate-500">|</span>
          <span class="text-cyan-300 font-semibold">${cyclone.centralPressureHpa} hPa</span>
          <span class="text-slate-500">|</span>
          <span class="text-amber-400">${cyclone.movementDirection} ${cyclone.movementSpeedKmh} km/h</span>
        </div>
      </div>
    `,
    iconSize: [180, 180],
    iconAnchor: [90, 90],
    popupAnchor: [0, -50]
  });
};

// Custom waypoint icon for forecasted steps (+24h, +48h, +72h)
const createForecastIcon = (hour: number, category: string) => {
  const isPeak = hour === 48;
  const isLandfall = hour === 72;
  const borderColor = isPeak ? 'border-purple-400 text-purple-300' : isLandfall ? 'border-rose-400 text-rose-300' : 'border-cyan-400 text-cyan-300';
  const bgColor = isPeak ? 'bg-purple-950/90' : isLandfall ? 'bg-rose-950/90' : 'bg-cyan-950/90';

  return L.divIcon({
    className: 'forecast-marker',
    html: `
      <div class="flex items-center justify-center">
        <div class="w-6 h-6 rounded-full ${bgColor} border-2 ${borderColor} font-mono font-bold text-[10px] flex items-center justify-center shadow-lg backdrop-blur-xs">
          +${hour}
        </div>
      </div>
    `,
    iconSize: [24, 24],
    iconAnchor: [12, 12],
    popupAnchor: [0, -14]
  });
};

export const CycloneMap: React.FC<CycloneMapProps> = ({
  cyclone,
  track,
  uncertaintyConeCoords,
  districtRisks,
  heightClass = 'h-[520px]',
  defaultShowRainfall = false,
  onSelectDistrict
}) => {
  const { resolvedTheme } = useTheme();

  // Layer visibility toggles
  const [showCone, setShowCone] = useState(true);
  const [showForecast, setShowForecast] = useState(true);
  const [showHistorical, setShowHistorical] = useState(true);
  const [showDistricts, setShowDistricts] = useState(true);
  const [showCorridor, setShowCorridor] = useState(true);
  const [showRainfall, setShowRainfall] = useState(defaultShowRainfall);
  const [mapStyle, setMapStyle] = useState<'auto' | 'dark' | 'light' | 'satellite' | 'terrain'>('satellite');

  // Center on Bay of Bengal
  const defaultCenter: [number, number] = [16.2, 85.5];

  // Past track coordinates (observed)
  const pastTrackPoints = track.filter(p => !p.isForecast);
  const pastCoords: [number, number][] = pastTrackPoints.map(p => [p.lat, p.lng] as [number, number]);

  // Forecast track coordinates
  const forecastTrackPoints = track.filter(p => p.isForecast);
  const forecastCoords: [number, number][] = [
    [cyclone.coordinates.lat, cyclone.coordinates.lng] as [number, number],
    ...forecastTrackPoints.map(p => [p.lat, p.lng] as [number, number])
  ];

  // Coastal Approach Monitoring Sector polygon
  const coastalApproachSectorCoords: [number, number][] = [
    [16.8, 82.2],
    [18.5, 84.8],
    [17.8, 85.8],
    [15.8, 83.2]
  ];

  // Effective style based on auto selection or manual choice
  const effectiveStyle = mapStyle === 'auto' ? (resolvedTheme === 'light' ? 'light' : 'dark') : mapStyle;

  // Tile layer URL based on selection
  const getTileLayer = () => {
    switch (effectiveStyle) {
      case 'satellite':
        return {
          url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
          attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
        };
      case 'terrain':
        return {
          url: 'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
          attribution: 'Map data: &copy; OpenStreetMap contributors, SRTM | Map style: &copy; OpenTopoMap (CC-BY-SA)'
        };
      case 'light':
        return {
          url: 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
          attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        };
      case 'dark':
      default:
        return {
          url: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
          attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        };
    }
  };

  const tileInfo = getTileLayer();

  return (
    <div className="relative rounded-2xl overflow-hidden border border-slate-300 dark:border-slate-800 bg-[#E2E8F0] dark:bg-[#06090e] shadow-md flex flex-col transition-colors">
      {/* Top Map HUD Bar */}
      <div className="bg-[#F8FAFC]/98 dark:bg-[#0A0E1A]/95 px-4 py-2.5 border-b border-slate-300 dark:border-slate-800 flex flex-wrap items-center justify-between gap-3 z-10 transition-colors">
        <div className="flex items-center gap-2 flex-wrap">
          <div className="flex items-center gap-1.5 text-xs font-mono font-bold text-slate-950 dark:text-white uppercase tracking-wide">
            <Radio className="w-4 h-4 text-cyan-700 dark:text-cyan-400 animate-pulse" />
            GIS Atmospheric Operations Console
          </div>
          <span className="text-slate-400 dark:text-slate-600 hidden sm:inline">|</span>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-100 dark:bg-cyan-500/10 text-cyan-900 dark:text-cyan-300 border border-cyan-300 dark:border-cyan-500/30 font-bold">
            SIMULATED VORTEX (INSAT-3D IR SYNTHESIS)
          </span>
        </div>

        {/* Map View & Layer Filters */}
        <div className="flex items-center gap-2 text-xs font-mono flex-wrap">
          {/* Basemap switcher */}
          <div className="flex items-center bg-slate-200/90 dark:bg-[#070B16] border border-slate-300 dark:border-cyan-500/20 rounded-xl p-0.5 shadow-xs gap-0.5">
            <button
              onClick={() => setMapStyle('auto')}
              className={`px-2.5 py-1 rounded-lg text-[10px] uppercase transition-all duration-200 cursor-pointer ${
                mapStyle === 'auto'
                  ? 'bg-gradient-to-r from-cyan-700 to-blue-700 text-white dark:from-cyan-950 dark:to-slate-900 dark:text-cyan-300 border border-cyan-500 dark:border-cyan-400/60 shadow-sm dark:shadow-[0_0_10px_rgba(6,182,212,0.3)] ring-1 ring-cyan-400/30 font-bold'
                  : 'text-slate-700 dark:text-slate-300 hover:text-slate-950 dark:hover:text-white hover:bg-white/80 dark:hover:bg-slate-800/80 font-semibold'
              }`}
              title="Automatically match current Dark/Light theme"
            >
              Auto GIS
            </button>
            <button
              onClick={() => setMapStyle('dark')}
              className={`px-2.5 py-1 rounded-lg text-[10px] uppercase transition-all duration-200 cursor-pointer ${
                mapStyle === 'dark'
                  ? 'bg-gradient-to-r from-cyan-700 to-blue-700 text-white dark:from-cyan-950 dark:to-slate-900 dark:text-cyan-300 border border-cyan-500 dark:border-cyan-400/60 shadow-sm dark:shadow-[0_0_10px_rgba(6,182,212,0.3)] ring-1 ring-cyan-400/30 font-bold'
                  : 'text-slate-700 dark:text-slate-300 hover:text-slate-950 dark:hover:text-white hover:bg-white/80 dark:hover:bg-slate-800/80 font-semibold'
              }`}
            >
              Dark
            </button>
            <button
              onClick={() => setMapStyle('light')}
              className={`px-2.5 py-1 rounded-lg text-[10px] uppercase transition-all duration-200 cursor-pointer ${
                mapStyle === 'light'
                  ? 'bg-gradient-to-r from-cyan-700 to-blue-700 text-white dark:from-cyan-950 dark:to-slate-900 dark:text-cyan-300 border border-cyan-500 dark:border-cyan-400/60 shadow-sm dark:shadow-[0_0_10px_rgba(6,182,212,0.3)] ring-1 ring-cyan-400/30 font-bold'
                  : 'text-slate-700 dark:text-slate-300 hover:text-slate-950 dark:hover:text-white hover:bg-white/80 dark:hover:bg-slate-800/80 font-semibold'
              }`}
            >
              Light
            </button>
            <button
              onClick={() => setMapStyle('satellite')}
              className={`px-2.5 py-1 rounded-lg text-[10px] uppercase transition-all duration-200 cursor-pointer ${
                mapStyle === 'satellite'
                  ? 'bg-gradient-to-r from-cyan-700 to-blue-700 text-white dark:from-cyan-950 dark:to-slate-900 dark:text-cyan-300 border border-cyan-500 dark:border-cyan-400/60 shadow-sm dark:shadow-[0_0_10px_rgba(6,182,212,0.3)] ring-1 ring-cyan-400/30 font-bold'
                  : 'text-slate-700 dark:text-slate-300 hover:text-slate-950 dark:hover:text-white hover:bg-white/80 dark:hover:bg-slate-800/80 font-semibold'
              }`}
            >
              Satellite
            </button>
            <button
              onClick={() => setMapStyle('terrain')}
              className={`px-2.5 py-1 rounded-lg text-[10px] uppercase transition-all duration-200 cursor-pointer ${
                mapStyle === 'terrain'
                  ? 'bg-gradient-to-r from-cyan-700 to-blue-700 text-white dark:from-cyan-950 dark:to-slate-900 dark:text-cyan-300 border border-cyan-500 dark:border-cyan-400/60 shadow-sm dark:shadow-[0_0_10px_rgba(6,182,212,0.3)] ring-1 ring-cyan-400/30 font-bold'
                  : 'text-slate-700 dark:text-slate-300 hover:text-slate-950 dark:hover:text-white hover:bg-white/80 dark:hover:bg-slate-800/80 font-semibold'
              }`}
            >
              Terrain
            </button>
          </div>

          {/* Quick layer switches */}
          <div className="hidden lg:flex items-center gap-1.5 bg-slate-200/90 dark:bg-[#070B16] border border-slate-300 dark:border-cyan-500/20 rounded-xl px-2.5 py-1 text-[11px] shadow-xs">
            <label className="flex items-center gap-1 text-slate-800 dark:text-slate-300 cursor-pointer font-bold">
              <input
                type="checkbox"
                checked={showCone}
                onChange={(e) => setShowCone(e.target.checked)}
                className="accent-cyan-600 rounded w-3 h-3"
              />
              Cone
            </label>
            <span className="text-slate-400 dark:text-slate-700">|</span>
            <label className="flex items-center gap-1 text-slate-800 dark:text-slate-300 cursor-pointer font-bold">
              <input
                type="checkbox"
                checked={showForecast}
                onChange={(e) => setShowForecast(e.target.checked)}
                className="accent-cyan-600 rounded w-3 h-3"
              />
              Forecast
            </label>
            <span className="text-slate-400 dark:text-slate-700">|</span>
            <label className="flex items-center gap-1 text-slate-800 dark:text-slate-300 cursor-pointer font-bold">
              <input
                type="checkbox"
                checked={showCorridor}
                onChange={(e) => setShowCorridor(e.target.checked)}
                className="accent-cyan-600 rounded w-3 h-3"
              />
              Approach
            </label>
            <span className="text-slate-400 dark:text-slate-700">|</span>
            <label className="flex items-center gap-1 text-slate-800 dark:text-slate-300 cursor-pointer font-bold">
              <input
                type="checkbox"
                checked={showDistricts}
                onChange={(e) => setShowDistricts(e.target.checked)}
                className="accent-cyan-600 rounded w-3 h-3"
              />
              Districts
            </label>
            <span className="text-slate-400 dark:text-slate-700">|</span>
            <button
              type="button"
              onClick={() => setShowRainfall(!showRainfall)}
              className={`flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-bold transition-all ${
                showRainfall
                  ? 'bg-blue-600 text-white shadow-xs'
                  : 'text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-950/40'
              }`}
              title="Click to toggle Cyclone Rainfall Prediction Layer"
            >
              <CloudRain className={`w-3.5 h-3.5 ${showRainfall ? 'text-white animate-bounce' : 'text-blue-600 dark:text-blue-400'}`} />
              <span>🌧 Rain {showRainfall ? 'ON' : 'OFF'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Leaflet Map Canvas */}
      <div className={`w-full ${heightClass} relative z-0`}>
        {/* Floating Clickable Rain Layer Icon / Control */}
        <div className="absolute top-3 right-3 z-10">
          <button
            type="button"
            onClick={() => setShowRainfall(!showRainfall)}
            className={`px-3 py-2 rounded-xl backdrop-blur-md border transition-all shadow-xl flex items-center gap-2 font-mono text-xs font-bold pointer-events-auto ${
              showRainfall
                ? 'bg-blue-600 text-white border-blue-400 ring-2 ring-blue-400/50 shadow-blue-500/25'
                : 'bg-white/95 dark:bg-slate-900/90 border-slate-300 dark:border-slate-700 text-slate-800 dark:text-slate-200 hover:bg-blue-50 dark:hover:bg-slate-800 hover:border-blue-400'
            }`}
            title="Click to toggle Cyclone Rainfall Prediction Layer"
          >
            <CloudRain className={`w-4 h-4 ${showRainfall ? 'text-cyan-200 animate-bounce' : 'text-blue-600 dark:text-blue-400'}`} />
            <span>🌧 Rainfall Layer: <strong className={showRainfall ? 'text-cyan-200' : 'text-blue-600 dark:text-blue-400'}>{showRainfall ? 'ON' : 'OFF'}</strong></span>
          </button>
        </div>

        <MapContainer
          center={defaultCenter}
          zoom={6}
          scrollWheelZoom={true}
          className="w-full h-full"
          zoomControl={true}
        >
          <TileLayer
            key={tileInfo.url}
            attribution={tileInfo.attribution}
            url={tileInfo.url}
          />

          {/* 1. Uncertainty Cone Polygon (Gradually expands along the forecast vector) */}
          {showCone && (
            <Polygon
              positions={uncertaintyConeCoords}
              pathOptions={{
                color: resolvedTheme === 'light' ? '#0284C7' : '#06B6D4',
                weight: 2,
                fillColor: resolvedTheme === 'light' ? '#0284C7' : '#06B6D4',
                fillOpacity: resolvedTheme === 'light' ? 0.18 : 0.14,
                dashArray: '5, 5'
              }}
            >
              <Popup>
                <div className="p-2 text-xs font-mono space-y-1">
                  <div className="font-bold text-cyan-700 dark:text-cyan-400">72-Hour Forecast Uncertainty Cone</div>
                  <div className="text-slate-700 dark:text-slate-300">Statistical 67% confidence envelope accounting for cross-track & along-track numerical ensemble variance.</div>
                  <div className="text-amber-700 dark:text-amber-300 text-[10px] font-semibold">Cone width expands from ±55 km (24h) to ±180 km (72h).</div>
                </div>
              </Popup>
            </Polygon>
          )}

          {/* 1b. Coastal Approach Monitoring Corridor */}
          {showCorridor && (
            <Polygon
              positions={coastalApproachSectorCoords}
              pathOptions={{
                color: '#F59E0B',
                weight: 1.5,
                fillColor: '#F59E0B',
                fillOpacity: 0.08,
                dashArray: '3, 4'
              }}
            >
              <Popup>
                <div className="p-2 text-xs font-mono space-y-1">
                  <div className="font-bold text-amber-700 dark:text-amber-400 flex items-center gap-1">
                    <AlertTriangle className="w-3.5 h-3.5" />
                    Coastal Approach Corridor (Projected)
                  </div>
                  <div className="text-slate-700 dark:text-slate-300 text-[11px]">
                    Northern Andhra Pradesh / Southern Odisha Coastal Sector under high surveillance for 48–72h atmospheric arrival.
                  </div>
                </div>
              </Popup>
            </Polygon>
          )}

          {/* 2. Historical Observed Track (Solid Sky-Blue Polyline) */}
          {showHistorical && pastCoords.length > 0 && (
            <Polyline
              positions={pastCoords}
              pathOptions={{
                color: resolvedTheme === 'light' ? '#0284C7' : '#38BDF8',
                weight: 3.5,
                opacity: 0.95
              }}
            />
          )}

          {/* 3. Forecast Track (Dashed Rose Polyline with Direction Flow) */}
          {showForecast && forecastCoords.length > 0 && (
            <Polyline
              positions={forecastCoords}
              pathOptions={{
                color: '#EF4444',
                weight: 3.2,
                dashArray: '8, 6',
                opacity: 0.95
              }}
            />
          )}

          {/* 4. Past Observed Waypoints */}
          {showHistorical && pastTrackPoints.map((pt, idx) => (
            <CircleMarker
              key={`past-${idx}`}
              center={[pt.lat, pt.lng]}
              radius={4.5}
              pathOptions={{
                color: '#0284C7',
                fillColor: '#38BDF8',
                fillOpacity: 0.9,
                weight: 1.5
              }}
            >
              <Popup>
                <div className="p-2 text-xs font-mono space-y-1 text-slate-800 dark:text-slate-100">
                  <div className="font-bold text-sky-600 dark:text-sky-400">{pt.label}</div>
                  <div className="text-slate-600 dark:text-slate-300">{pt.timestamp}</div>
                  <div className="text-slate-700 dark:text-slate-200">Wind: <strong>{pt.windSpeedKts} kt</strong> ({pt.centralPressureHpa} hPa)</div>
                  <div className="text-cyan-700 dark:text-cyan-300 text-[11px] font-semibold">{pt.category}</div>
                </div>
              </Popup>
            </CircleMarker>
          ))}

          {/* 5. Forecast Waypoints (+24h, +48h, +72h) */}
          {showForecast && forecastTrackPoints.map((pt, idx) => {
            const hour = pt.forecastHour || (idx + 1) * 24;
            return (
              <Marker
                key={`fc-${idx}`}
                position={[pt.lat, pt.lng]}
                icon={createForecastIcon(hour, pt.category)}
              >
                <Popup>
                  <div className="p-2 text-xs font-mono space-y-1.5">
                    <div className="font-bold text-cyan-700 dark:text-cyan-400 flex items-center justify-between">
                      <span>+{hour}h FORECAST WAYPOINT</span>
                      <span className="text-[10px] text-purple-700 dark:text-purple-400 bg-purple-500/10 px-1 rounded font-bold">
                        {pt.confidencePercent}% Conf.
                      </span>
                    </div>
                    <div className="text-slate-600 dark:text-slate-300 text-[11px]">{pt.timestamp}</div>
                    <div className="text-slate-700 dark:text-slate-200">
                      Coordinates: <strong>{formatCoords(pt.lat, pt.lng)}</strong>
                    </div>
                    <div className="text-slate-700 dark:text-slate-200">
                      Intensity: <strong className="text-rose-600 dark:text-rose-400">{pt.windSpeedKts} kt</strong> ({pt.centralPressureHpa} hPa)
                    </div>
                    <div className="text-slate-600 dark:text-slate-300">
                      Category: <strong>{pt.category}</strong>
                    </div>
                    <div className="text-amber-800 dark:text-amber-300 text-[10px] pt-1 border-t border-slate-200 dark:border-slate-700 font-semibold">
                      Uncertainty Radius: ±{pt.uncertaintyRadiusKm} km
                    </div>
                  </div>
                </Popup>
              </Marker>
            );
          })}

          {/* 6. REALISTIC CYCLONE VORTEX VISUALIZATION (Eye + Spiral Rainbands + Movement Vector) */}
          <Marker
            position={[cyclone.coordinates.lat, cyclone.coordinates.lng]}
            icon={createRealisticCycloneIcon(cyclone)}
          >
            <Popup>
              <div className="p-2.5 text-xs font-mono space-y-2 max-w-xs">
                <div className="flex items-center justify-between pb-1 border-b border-slate-200 dark:border-slate-700">
                  <span className="font-bold text-rose-600 dark:text-rose-400 text-sm flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-rose-500 animate-pulse"></span>
                    {cyclone.name}
                  </span>
                  <span className="text-[10px] font-bold text-emerald-700 dark:text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20">
                    CURRENT STORM EYE
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-2 text-[11px] text-slate-700 dark:text-slate-300">
                  <div>
                    <span className="text-slate-500">Eye Location:</span>
                    <div className="font-bold text-slate-900 dark:text-white">{formatCoords(cyclone.coordinates.lat, cyclone.coordinates.lng)}</div>
                  </div>
                  <div>
                    <span className="text-slate-500">Max Sustained:</span>
                    <div className="font-bold text-rose-600 dark:text-rose-400">{cyclone.maxSustainedWindKts} kt ({cyclone.maxSustainedWindKmh} km/h)</div>
                  </div>
                  <div>
                    <span className="text-slate-500">Central Pressure:</span>
                    <div className="font-bold text-slate-900 dark:text-white">{cyclone.centralPressureHpa} hPa</div>
                  </div>
                  <div>
                    <span className="text-slate-500">Translational Vector:</span>
                    <div className="font-bold text-slate-900 dark:text-white">{cyclone.movementDirection} • {cyclone.movementSpeedKmh} km/h (315°)</div>
                  </div>
                </div>

                <div className="p-2 rounded bg-cyan-50 dark:bg-cyan-950/30 border border-cyan-200 dark:border-cyan-800/40 text-[10px] text-cyan-900 dark:text-cyan-300">
                  <strong>Vortex Simulation:</strong> 8-Band ConvLSTM attention field with axisymmetric convective eyewall consolidation.
                </div>

                <div className="pt-1.5 border-t border-slate-200 dark:border-slate-700 flex justify-between text-[10px]">
                  <span className="text-purple-700 dark:text-purple-400 font-semibold">Model Conf: {cyclone.modelConfidencePercent}%</span>
                  <span className="text-emerald-700 dark:text-emerald-400 font-semibold">INSAT-3D Multispectral</span>
                </div>
              </div>
            </Popup>
          </Marker>

          {/* 7. Affected Coastal District Markers */}
          {showDistricts && districtRisks.map((d, idx) => {
            const riskColors = getRiskSeverityColor(d.riskLevel);
            const circleColor = d.riskLevel === 'CRITICAL' ? '#EF4444' : d.riskLevel === 'HIGH' ? '#F97316' : '#EAB308';

            return (
              <CircleMarker
                key={`dist-${idx}`}
                center={d.coordinates}
                radius={d.riskLevel === 'CRITICAL' ? 9 : 7}
                pathOptions={{
                  color: circleColor,
                  fillColor: circleColor,
                  fillOpacity: 0.75,
                  weight: 2
                }}
              >
                <Popup>
                  <div className="p-2.5 text-xs font-mono space-y-1.5">
                    <div className="flex items-center justify-between pb-1 border-b border-slate-200 dark:border-slate-700">
                      <strong className="text-slate-900 dark:text-white text-sm">{d.district}</strong>
                      <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${riskColors.bg} ${riskColors.text}`}>
                        {d.riskLevel}
                      </span>
                    </div>
                    <div className="text-slate-500 dark:text-slate-400 text-[11px]">{d.state}</div>
                    <div className="space-y-1 text-slate-700 dark:text-slate-300 text-[11px] pt-1">
                      <div>Peak Wind Threat: <strong className="text-rose-600 dark:text-rose-400">{d.windRiskKts} kt</strong></div>
                      <div>Expected Rainfall: <strong className="text-cyan-700 dark:text-cyan-400">{d.estimatedRainfallMm} mm</strong></div>
                      <div>Storm Surge: <strong className="text-sky-700 dark:text-sky-400">{d.stormSurgeMeters} m</strong></div>
                      <div>Exposed Population: <strong className="text-slate-900 dark:text-white">{d.populationExposed}</strong></div>
                    </div>
                    {d.evacuationRecommended && (
                      <div className="pt-1 text-[10px] text-rose-700 dark:text-rose-300 font-bold flex items-center gap-1">
                        <ShieldAlert className="w-3.5 h-3.5 text-rose-600 dark:text-rose-400 shrink-0" />
                        Evacuation Protocol Initiated
                      </div>
                    )}
                  </div>
                </Popup>
              </CircleMarker>
            );
          })}

          {/* 8. CYCLONE RAINFALL FORECAST LAYER (Swaths & District Precipitation Markers) */}
          {showRainfall && (
            <>
              {/* 8a. Outer Heavy to Very Heavy Rainband Swath (65-200 mm) */}
              <Polygon
                positions={mockRainfallSwaths.outer}
                pathOptions={{
                  color: '#38BDF8',
                  weight: 2,
                  fillColor: '#0284C7',
                  fillOpacity: resolvedTheme === 'light' ? 0.22 : 0.28,
                  dashArray: '5, 5'
                }}
              >
                <Popup>
                  <div className="p-2 text-xs font-mono space-y-1">
                    <div className="font-bold text-sky-700 dark:text-sky-300 flex items-center gap-1">
                      🌧 Heavy Precipitation Outer Swath (65–195 mm)
                    </div>
                    <div className="text-slate-700 dark:text-slate-300 text-[11px]">
                      Convective squalls & spiral rainband convergence zone across North Andhra & Odisha coast.
                    </div>
                    <div className="text-[10px] text-cyan-800 dark:text-cyan-300 font-semibold">
                      *Simulated Numerical Forecast
                    </div>
                  </div>
                </Popup>
              </Polygon>

              {/* 8b. Core Torrential Deluge Swath (>200 mm) */}
              <Polygon
                positions={mockRainfallSwaths.core}
                pathOptions={{
                  color: '#2563EB',
                  weight: 2.5,
                  fillColor: '#1D4ED8',
                  fillOpacity: resolvedTheme === 'light' ? 0.32 : 0.38
                }}
              >
                <Popup>
                  <div className="p-2 text-xs font-mono space-y-1">
                    <div className="font-bold text-blue-700 dark:text-blue-300 flex items-center gap-1">
                      🌧 Extremely Heavy Rainfall Core Swath (&gt;200 mm)
                    </div>
                    <div className="text-slate-700 dark:text-slate-300 text-[11px]">
                      Axisymmetric eyewall precipitation corridor; severe flash flooding and inundation threat.
                    </div>
                    <div className="text-rose-600 dark:text-rose-400 text-[10px] font-bold">
                      Critical Inundation Alert
                    </div>
                  </div>
                </Popup>
              </Polygon>

              {/* 8c. Specific Rainfall District Forecast Markers & Popups */}
              {mockRainfallForecastZones.map((rf) => (
                <Marker
                  key={`rf-marker-${rf.id}`}
                  position={rf.coordinates}
                  icon={createRainfallMarkerIcon(rf.intensityCategory)}
                >
                  <Popup>
                    <div className="p-2.5 text-xs font-mono space-y-2 min-w-[190px]">
                      <div className="flex items-center justify-between pb-1 border-b border-blue-200 dark:border-slate-700">
                        <div className="flex items-center gap-1.5 font-bold text-blue-700 dark:text-cyan-400">
                          <span className="text-sm">🌧</span>
                          <span>Rainfall Forecast</span>
                        </div>
                        <span className="text-[9px] px-1.5 py-0.5 rounded bg-blue-100 dark:bg-blue-950/80 text-blue-800 dark:text-blue-300 font-bold">
                          SIMULATED
                        </span>
                      </div>

                      <div className="font-extrabold text-slate-950 dark:text-white text-sm">
                        {rf.district}
                      </div>

                      <div className="space-y-1 text-[11px] text-slate-700 dark:text-slate-300">
                        <div className="flex justify-between">
                          <span className="text-slate-500">24h:</span>
                          <strong className="text-blue-700 dark:text-cyan-300">{rf.rain24h} ({rf.rain24hMm} mm)</strong>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500">48h:</span>
                          <strong className="text-indigo-700 dark:text-indigo-300">{rf.rain48h} ({rf.rain48hMm} mm)</strong>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500">72h:</span>
                          <strong className="text-rose-600 dark:text-rose-400">{rf.rain72h} ({rf.rain72hMm} mm)</strong>
                        </div>
                        <div className="flex justify-between pt-1 border-t border-slate-200 dark:border-slate-700 font-bold">
                          <span className="text-slate-500">Risk:</span>
                          <span className={`px-1.5 py-0.5 rounded text-[10px] ${
                            rf.riskLevel === 'CRITICAL' ? 'bg-rose-100 dark:bg-rose-950/80 text-rose-700 dark:text-rose-300' :
                            rf.riskLevel === 'HIGH' ? 'bg-amber-100 dark:bg-amber-950/80 text-amber-700 dark:text-amber-300' :
                            'bg-yellow-100 dark:bg-yellow-950/80 text-yellow-700 dark:text-yellow-300'
                          }`}>
                            {rf.riskLevel === 'CRITICAL' ? 'Critical' : rf.riskLevel === 'HIGH' ? 'High' : 'Moderate'}
                          </span>
                        </div>
                      </div>

                      <div className="p-1.5 rounded bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-[10px] text-slate-600 dark:text-slate-400">
                        <strong className="text-slate-800 dark:text-slate-200">Peak Period:</strong> {rf.peakWindow}
                      </div>

                      <div className="text-[9px] text-slate-500 dark:text-slate-500 italic pt-0.5 border-t border-slate-200 dark:border-slate-800">
                        *WRF-Chem Numerical Precipitation Model
                      </div>
                    </div>
                  </Popup>
                </Marker>
              ))}
            </>
          )}
        </MapContainer>
      </div>

      {/* Floating Tactical GIS Legend */}
      <div className="absolute bottom-4 left-4 z-10 bg-white/95 dark:bg-slate-900/90 border border-slate-300 dark:border-slate-800 backdrop-blur-md rounded-xl p-3.5 text-[11px] font-mono shadow-xl hidden sm:block max-w-xs pointer-events-auto transition-colors">
        <div className="font-bold text-slate-950 dark:text-slate-200 mb-2 flex items-center justify-between border-b border-slate-200 dark:border-slate-850 pb-1.5">
          <span className="flex items-center gap-1.5">
            <Compass className="w-3.5 h-3.5 text-cyan-700 dark:text-cyan-400" />
            GIS METEOROLOGICAL LEGEND
          </span>
          <span className="text-[10px] text-cyan-800 dark:text-cyan-400 font-bold">Bay of Bengal</span>
        </div>
        <div className="space-y-1.5 text-slate-800 dark:text-slate-400 font-medium">
          <div className="flex items-center gap-2">
            <span className="w-3.5 h-3.5 rounded-full bg-rose-600 border-2 border-white dark:border-slate-900 shrink-0 shadow-xs"></span>
            <span className="font-semibold text-slate-900 dark:text-slate-300">Current Eye (14.52°N, 87.21°E • 82 kt)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-4 h-3 rounded bg-cyan-500/30 border border-cyan-500 shrink-0"></span>
            <span>Satellite Vortex & Spiral Rainbands</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-6 h-0.5 bg-sky-600 dark:bg-sky-400 shrink-0"></span>
            <span>Observed Historical Track (T-60h)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-6 h-0.5 border-b-2 border-dashed border-rose-600 shrink-0"></span>
            <span>AI Projected Trajectory (24h/48h/72h)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-4 h-3 bg-cyan-500/20 border border-cyan-600/50 rounded-sm shrink-0"></span>
            <span>Forecast Uncertainty Cone (67% Conf.)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-4 h-2.5 bg-amber-500/20 border border-amber-600/50 rounded-sm shrink-0"></span>
            <span>Coastal Approach Monitoring Zone</span>
          </div>
          <div className="flex items-center gap-2 pt-1.5 border-t border-slate-200 dark:border-slate-800 font-semibold">
            <span className="w-2.5 h-2.5 rounded-full bg-rose-600 shrink-0"></span>
            <span className="w-2.5 h-2.5 rounded-full bg-orange-500 shrink-0"></span>
            <span className="w-2.5 h-2.5 rounded-full bg-yellow-500 shrink-0"></span>
            <span className="text-slate-900 dark:text-slate-300">District Vulnerability (Critical / High / Mod)</span>
          </div>

          {showRainfall && (
            <div className="pt-2 border-t border-slate-200 dark:border-slate-800 space-y-1">
              <div className="font-bold text-blue-700 dark:text-cyan-400 text-[10px] flex items-center gap-1">
                <CloudRain className="w-3 h-3 text-blue-500 animate-bounce" />
                RAINFALL INTENSITY SCALE (SIMULATED)
              </div>
              <div className="flex items-center gap-1.5 text-[10px]">
                <span className="w-3 h-2 rounded-xs bg-[#1D4ED8] border border-blue-400 shrink-0"></span>
                <span>Extremely Heavy (&gt;200 mm)</span>
              </div>
              <div className="flex items-center gap-1.5 text-[10px]">
                <span className="w-3 h-2 rounded-xs bg-[#0284C7] border border-sky-300 shrink-0"></span>
                <span>Heavy to Very Heavy (65–200 mm)</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

