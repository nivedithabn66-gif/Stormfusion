import React, { useState, useEffect } from 'react';
import { 
  TrendingUp, 
  Clock, 
  MapPin, 
  Compass, 
  ShieldAlert, 
  Cpu, 
  Sparkles, 
  HelpCircle,
  Layers,
  ArrowRight,
  AlertTriangle,
  CloudRain,
  Droplets,
  Wind,
  Gauge,
  Info
} from 'lucide-react';
import { CycloneTelemetry, ForecastHourData, TrackPoint, DistrictRisk, RainfallDistrictForecast } from '../types/cyclone';
import { cycloneApi } from '../services/api';
import { ForecastCard } from '../components/ForecastCard';
import { ForecastTimeline } from '../components/ForecastTimeline';
import { DemoModeBadge } from '../components/DemoModeBadge';
import { mockRainfallForecastZones } from '../data/mockForecast';
import { formatCoords } from '../utils/formatters';

export const ForecastPage: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [cyclone, setCyclone] = useState<CycloneTelemetry | null>(null);
  const [forecastHours, setForecastHours] = useState<ForecastHourData[]>([]);
  const [intensityTrend, setIntensityTrend] = useState<any[]>([]);
  const [uncertaintyCone, setUncertaintyCone] = useState<[number, number][]>([]);
  const [track, setTrack] = useState<TrackPoint[]>([]);
  const [districts, setDistricts] = useState<DistrictRisk[]>([]);
  const [activeNavTab, setActiveNavTab] = useState<'all' | 'track' | 'wind' | 'landfall' | 'rainfall'>('all');

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const [c, f, t, r] = await Promise.all([
          cycloneApi.getCurrentCyclone(),
          cycloneApi.getForecast(),
          cycloneApi.getTrack(),
          cycloneApi.getRiskZones()
        ]);
        setCyclone(c);
        setForecastHours(f.hours);
        setIntensityTrend(f.intensityTrend);
        setUncertaintyCone(f.uncertaintyCone);
        setTrack(t);
        setDistricts(r);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading || !cyclone) {
    return (
      <div className="min-h-screen flex items-center justify-center font-mono text-cyan-400">
        <TrendingUp className="w-8 h-8 animate-spin" />
      </div>
    );
  }

  return (
    <div className="p-4 lg:p-6 space-y-6 max-w-[1700px] mx-auto font-mono">
      {/* 1. Header */}
      <div className="command-card rounded-2xl p-5 border border-slate-200 dark:border-slate-800 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-cyan-500/10 dark:bg-cyan-500/15 border border-cyan-500/30 text-cyan-600 dark:text-cyan-400">
              <TrendingUp className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-wide text-slate-900 dark:text-white uppercase">
                AI Trajectory, Wind & Rainfall Multi-Horizon Forecasting
              </h1>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Spatiotemporal ConvLSTM + Multi-Head Prediction Engine (Track, Intensity, Landfall & Precipitation Swath)
              </p>
            </div>
          </div>
        </div>

        <DemoModeBadge subtle />
      </div>

      {/* 2. Track & Forecast Navigation Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-200/80 dark:bg-[#080D1A] p-2 rounded-2xl border border-slate-300 dark:border-cyan-500/20 shadow-sm backdrop-blur-md">
        <div className="flex items-center flex-wrap gap-1.5 text-xs">
          {/* Tab 1: All Forecasts */}
          <button
            onClick={() => setActiveNavTab('all')}
            className={`px-3.5 py-2 rounded-xl font-bold transition-all duration-200 flex items-center gap-2 cursor-pointer ${
              activeNavTab === 'all'
                ? 'bg-gradient-to-r from-slate-900 to-slate-800 text-white dark:from-cyan-950 dark:to-slate-900 dark:text-cyan-300 border border-slate-700 dark:border-cyan-500/50 shadow-md dark:shadow-[0_0_15px_rgba(6,182,212,0.25)] ring-1 ring-slate-700 dark:ring-cyan-400/30'
                : 'text-slate-700 dark:text-slate-300 hover:text-slate-950 dark:hover:text-white hover:bg-white/80 dark:hover:bg-slate-800/80 border border-transparent'
            }`}
          >
            <Layers className={`w-3.5 h-3.5 ${activeNavTab === 'all' ? 'text-cyan-400 dark:text-cyan-300' : 'text-slate-500 dark:text-slate-400'}`} />
            <span>All Forecasts</span>
          </button>

          {/* Tab 2: Trajectory & 24/48/72h */}
          <button
            onClick={() => setActiveNavTab('track')}
            className={`px-3.5 py-2 rounded-xl font-bold transition-all duration-200 flex items-center gap-2 cursor-pointer ${
              activeNavTab === 'track'
                ? 'bg-gradient-to-r from-cyan-700 to-blue-700 text-white dark:from-cyan-950 dark:to-cyan-900 dark:text-cyan-300 border border-cyan-500 dark:border-cyan-400/60 shadow-md dark:shadow-[0_0_15px_rgba(6,182,212,0.3)] ring-1 ring-cyan-400/40'
                : 'text-slate-700 dark:text-slate-300 hover:text-slate-950 dark:hover:text-white hover:bg-white/80 dark:hover:bg-slate-800/80 border border-transparent'
            }`}
          >
            <Compass className={`w-3.5 h-3.5 ${activeNavTab === 'track' ? 'text-cyan-200' : 'text-slate-500 dark:text-slate-400'}`} />
            <span>Trajectory & 24/48/72h</span>
          </button>

          {/* Tab 3: Wind & Pressure Trend */}
          <button
            onClick={() => setActiveNavTab('wind')}
            className={`px-3.5 py-2 rounded-xl font-bold transition-all duration-200 flex items-center gap-2 cursor-pointer ${
              activeNavTab === 'wind'
                ? 'bg-gradient-to-r from-teal-700 to-cyan-700 text-white dark:from-teal-950 dark:to-cyan-900 dark:text-teal-300 border border-teal-500 dark:border-teal-400/60 shadow-md dark:shadow-[0_0_15px_rgba(20,184,166,0.3)] ring-1 ring-teal-400/40'
                : 'text-slate-700 dark:text-slate-300 hover:text-slate-950 dark:hover:text-white hover:bg-white/80 dark:hover:bg-slate-800/80 border border-transparent'
            }`}
          >
            <Wind className={`w-3.5 h-3.5 ${activeNavTab === 'wind' ? 'text-teal-200' : 'text-slate-500 dark:text-slate-400'}`} />
            <span>Wind & Pressure Trend</span>
          </button>

          {/* Tab 4: Landfall Corridor */}
          <button
            onClick={() => setActiveNavTab('landfall')}
            className={`px-3.5 py-2 rounded-xl font-bold transition-all duration-200 flex items-center gap-2 cursor-pointer ${
              activeNavTab === 'landfall'
                ? 'bg-gradient-to-r from-rose-700 to-amber-700 text-white dark:from-rose-950 dark:to-rose-900 dark:text-rose-300 border border-rose-500 dark:border-rose-400/60 shadow-md dark:shadow-[0_0_15px_rgba(244,63,94,0.3)] ring-1 ring-rose-400/40'
                : 'text-slate-700 dark:text-slate-300 hover:text-slate-950 dark:hover:text-white hover:bg-white/80 dark:hover:bg-slate-800/80 border border-transparent'
            }`}
          >
            <ShieldAlert className={`w-3.5 h-3.5 ${activeNavTab === 'landfall' ? 'text-rose-200' : 'text-slate-500 dark:text-slate-400'}`} />
            <span>Landfall Corridor</span>
          </button>

          {/* Tab 5: Rainfall Forecast */}
          <button
            onClick={() => setActiveNavTab('rainfall')}
            className={`px-3.5 py-2 rounded-xl font-bold transition-all duration-200 flex items-center gap-2 cursor-pointer ${
              activeNavTab === 'rainfall'
                ? 'bg-gradient-to-r from-blue-700 via-indigo-700 to-cyan-700 text-white dark:from-blue-950 dark:to-cyan-950 dark:text-cyan-200 border border-blue-400 dark:border-cyan-400 shadow-md dark:shadow-[0_0_20px_rgba(59,130,246,0.35)] ring-2 ring-blue-400/50'
                : 'text-blue-800 dark:text-cyan-300 bg-blue-100/70 dark:bg-blue-950/40 hover:bg-blue-200/80 dark:hover:bg-blue-900/60 border border-blue-300/80 dark:border-blue-800/60'
            }`}
          >
            <CloudRain className={`w-3.5 h-3.5 ${activeNavTab === 'rainfall' ? 'text-cyan-200 animate-bounce' : 'text-blue-600 dark:text-cyan-400'}`} />
            <span>Rainfall Forecast</span>
            <span className="text-[9px] px-1.5 py-0.2 rounded bg-blue-600 dark:bg-cyan-500 text-white dark:text-slate-950 font-black tracking-wide font-mono shadow-xs">
              NEW
            </span>
          </button>
        </div>

        <div className="flex items-center gap-2 text-[11px] text-slate-700 dark:text-slate-300 px-2 font-semibold">
          <Info className="w-3.5 h-3.5 text-cyan-600 dark:text-cyan-400" />
          <span>WRF-Chem Gridded Model (Simulated)</span>
        </div>
      </div>

      {/* 3. Trajectory / 24h, 48h, 72h Forecast Cards */}
      {(activeNavTab === 'all' || activeNavTab === 'track') && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {forecastHours.map((fc) => (
            <ForecastCard
              key={fc.hour}
              forecast={fc}
              isPeak={fc.hour === 48}
              isLandfall={fc.hour === 72}
            />
          ))}
        </div>
      )}

      {/* 4. Intensity and Pressure Trend Curves */}
      {(activeNavTab === 'all' || activeNavTab === 'wind') && (
        <ForecastTimeline data={intensityTrend} />
      )}

      {/* 5. CYCLONE RAINFALL FORECASTING MODULE */}
      {(activeNavTab === 'all' || activeNavTab === 'rainfall') && (
        <div className="command-card rounded-2xl p-5 border border-blue-300 dark:border-blue-900/50 bg-blue-50/30 dark:bg-slate-900/50 space-y-5 shadow-sm">
          {/* Rainfall Module Header */}
          <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-blue-200 dark:border-slate-800">
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-xl bg-blue-500/15 border border-blue-500/30 text-blue-700 dark:text-cyan-400 shadow-xs">
                <CloudRain className="w-5 h-5 animate-pulse" />
              </div>
              <div>
                <div className="flex items-center gap-2 flex-wrap">
                  <h2 className="text-base font-black font-mono tracking-wide text-slate-950 dark:text-white uppercase">
                    CYCLONE RAINFALL FORECAST & INUNDATION OUTLOOK
                  </h2>
                  <span className="px-2 py-0.5 rounded text-[10px] bg-blue-100 dark:bg-blue-950/80 text-blue-900 dark:text-cyan-300 border border-blue-300 dark:border-blue-700/60 font-bold">
                    24h / 48h / 72h HORIZONS
                  </span>
                </div>
                <p className="text-xs text-slate-600 dark:text-slate-400 font-mono mt-0.5">
                  Cumulative precipitation accumulation, rainband arrival timing & coastal basin inundation threat
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-[10px] font-mono font-bold px-2.5 py-1 rounded bg-amber-100 dark:bg-amber-950/70 text-amber-900 dark:text-amber-300 border border-amber-300 dark:border-amber-700/50">
                PROTOTYPE / SIMULATED DATA
              </span>
            </div>
          </div>

          {/* Horizon Overview Cards (24h / 48h / 72h) */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-mono">
            {/* 24h Rainfall Horizon */}
            <div className="p-4 rounded-xl bg-white dark:bg-slate-900/80 border border-blue-200 dark:border-slate-800 space-y-2.5 shadow-xs">
              <div className="flex items-center justify-between pb-1.5 border-b border-slate-200 dark:border-slate-800">
                <span className="font-bold text-blue-800 dark:text-cyan-300 flex items-center gap-1.5">
                  <Clock className="w-3.5 h-3.5" />
                  24-Hour Period (0-24h)
                </span>
                <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-blue-100 dark:bg-blue-950 text-blue-800 dark:text-blue-300">
                  MODERATE TO HEAVY
                </span>
              </div>
              <div className="text-2xl font-black text-blue-700 dark:text-cyan-400">
                45 – 85 mm
              </div>
              <div className="text-[11px] text-slate-600 dark:text-slate-300 space-y-1">
                <div><strong>Primary Impact:</strong> Kakinada, Visakhapatnam, Outer Delta</div>
                <div><strong>Pattern:</strong> Outer spiral feeder bands & intermittent squally downpours</div>
              </div>
            </div>

            {/* 48h Rainfall Horizon */}
            <div className="p-4 rounded-xl bg-white dark:bg-slate-900/80 border border-indigo-200 dark:border-slate-800 space-y-2.5 shadow-xs">
              <div className="flex items-center justify-between pb-1.5 border-b border-slate-200 dark:border-slate-800">
                <span className="font-bold text-indigo-800 dark:text-indigo-300 flex items-center gap-1.5">
                  <Clock className="w-3.5 h-3.5" />
                  48-Hour Period (24-48h)
                </span>
                <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-indigo-100 dark:bg-indigo-950 text-indigo-800 dark:text-indigo-300">
                  VERY HEAVY
                </span>
              </div>
              <div className="text-2xl font-black text-indigo-700 dark:text-indigo-400">
                115 – 195 mm
              </div>
              <div className="text-[11px] text-slate-600 dark:text-slate-300 space-y-1">
                <div><strong>Primary Impact:</strong> Visakhapatnam, Vizianagaram, Srikakulam</div>
                <div><strong>Pattern:</strong> Deep convective core consolidation & widespread rain squalls</div>
              </div>
            </div>

            {/* 72h Rainfall Horizon (Landfall) */}
            <div className="p-4 rounded-xl bg-white dark:bg-slate-900/80 border border-rose-200 dark:border-slate-800 space-y-2.5 shadow-xs">
              <div className="flex items-center justify-between pb-1.5 border-b border-slate-200 dark:border-slate-800">
                <span className="font-bold text-rose-800 dark:text-rose-300 flex items-center gap-1.5">
                  <Clock className="w-3.5 h-3.5" />
                  72-Hour Period (48-72h)
                </span>
                <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-rose-100 dark:bg-rose-950 text-rose-800 dark:text-rose-300">
                  EXTREMELY HEAVY
                </span>
              </div>
              <div className="text-2xl font-black text-rose-600 dark:text-rose-400">
                200 – 265+ mm
              </div>
              <div className="text-[11px] text-slate-600 dark:text-slate-300 space-y-1">
                <div><strong>Primary Impact:</strong> Visakhapatnam, Srikakulam, South Odisha</div>
                <div><strong>Pattern:</strong> Torrential eyewall deluge & catastrophic local catchment runoff</div>
              </div>
            </div>
          </div>

          {/* Affected Districts / Areas Matrix */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold font-mono text-slate-900 dark:text-white uppercase flex items-center gap-2">
                <MapPin className="w-4 h-4 text-cyan-600 dark:text-cyan-400" />
                District-Level Rainfall & Risk Breakdown
              </h3>
              <span className="text-[10px] text-slate-500 dark:text-slate-400">
                Showing 6 high-exposure coastal sectors
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 font-mono">
              {mockRainfallForecastZones.map((zone) => (
                <div
                  key={zone.id}
                  className="p-3.5 rounded-xl bg-white dark:bg-slate-900/90 border border-slate-300 dark:border-slate-800 space-y-2.5 shadow-xs hover:border-blue-400 dark:hover:border-blue-500/50 transition-all"
                >
                  <div className="flex items-center justify-between pb-1.5 border-b border-slate-200 dark:border-slate-800">
                    <div>
                      <strong className="text-sm text-slate-950 dark:text-white block">{zone.district}</strong>
                      <span className="text-[10px] text-slate-500">{zone.state}</span>
                    </div>
                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                      zone.riskLevel === 'CRITICAL' ? 'bg-rose-100 dark:bg-rose-950/80 text-rose-700 dark:text-rose-300 border border-rose-300 dark:border-rose-800' :
                      zone.riskLevel === 'HIGH' ? 'bg-amber-100 dark:bg-amber-950/80 text-amber-700 dark:text-amber-300 border border-amber-300 dark:border-amber-800' :
                      'bg-yellow-100 dark:bg-yellow-950/80 text-yellow-700 dark:text-yellow-300 border border-yellow-300 dark:border-yellow-800'
                    }`}>
                      {zone.riskLevel}
                    </span>
                  </div>

                  {/* 24h / 48h / 72h Period Accumulations */}
                  <div className="grid grid-cols-3 gap-1.5 text-center bg-slate-50 dark:bg-slate-950/60 p-2 rounded-lg border border-slate-200 dark:border-slate-850 text-[11px]">
                    <div>
                      <span className="text-slate-500 text-[10px] block">24h</span>
                      <strong className="text-blue-700 dark:text-cyan-300">{zone.rain24hMm} mm</strong>
                      <span className="text-[9px] text-slate-400 block">{zone.rain24h}</span>
                    </div>
                    <div className="border-x border-slate-200 dark:border-slate-800">
                      <span className="text-slate-500 text-[10px] block">48h</span>
                      <strong className="text-indigo-700 dark:text-indigo-300">{zone.rain48hMm} mm</strong>
                      <span className="text-[9px] text-slate-400 block">{zone.rain48h}</span>
                    </div>
                    <div>
                      <span className="text-slate-500 text-[10px] block">72h</span>
                      <strong className="text-rose-600 dark:text-rose-400">{zone.rain72hMm} mm</strong>
                      <span className="text-[9px] text-slate-400 block">{zone.rain72h}</span>
                    </div>
                  </div>

                  <div className="text-[11px] text-slate-700 dark:text-slate-300 space-y-1">
                    <div><strong>Peak Window:</strong> {zone.peakWindow}</div>
                    <div className="text-[10px] text-slate-500 dark:text-slate-400 leading-snug">{zone.advisory}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* 7. Detailed Landfall & Uncertainty Breakdown */}
      {(activeNavTab === 'all' || activeNavTab === 'landfall') && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 text-xs">
          {/* Landfall Forecast Box */}
          <div className="command-card rounded-2xl p-5 border border-rose-300 dark:border-rose-500/30 bg-rose-50/50 dark:bg-rose-950/20 space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-rose-200 dark:border-rose-500/20">
              <span className="font-bold text-rose-800 dark:text-rose-300 text-sm flex items-center gap-2">
                <ShieldAlert className="w-4 h-4 text-rose-600 dark:text-rose-400" />
                Predicted Landfall Corridor Assessment
              </span>
              <span className="px-2 py-0.5 rounded text-[10px] bg-rose-200/70 dark:bg-rose-500/30 text-rose-900 dark:text-rose-200 border border-rose-300 dark:border-rose-500/40">
                HIGH RISK
              </span>
            </div>

            <div className="space-y-2 text-slate-700 dark:text-slate-300">
              <div className="flex justify-between py-1 border-b border-rose-100 dark:border-slate-800">
                <span className="text-slate-500">Projected Landfall Window:</span>
                <strong className="text-slate-900 dark:text-white">10 Sep 2026, 12:00 – 16:00 IST (+68 to +72h)</strong>
              </div>
              <div className="flex justify-between py-1 border-b border-rose-100 dark:border-slate-800">
                <span className="text-slate-500">Center Coordinate:</span>
                <strong className="text-cyan-700 dark:text-cyan-300">18.1° N, 82.5° E (North AP / South Odisha coast)</strong>
              </div>
              <div className="flex justify-between py-1 border-b border-rose-100 dark:border-slate-800">
                <span className="text-slate-500">Expected Landfall Intensity:</span>
                <strong className="text-rose-600 dark:text-rose-400">78–82 kt (Severe Cyclonic Storm)</strong>
              </div>
              <div className="flex justify-between py-1 border-b border-rose-100 dark:border-slate-800">
                <span className="text-slate-500">Peak Storm Surge:</span>
                <strong className="text-sky-700 dark:text-sky-300">2.8 to 3.2 meters (Srikakulam & Visakhapatnam)</strong>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-500">Inland Dissipation:</span>
                <span className="text-slate-700 dark:text-slate-300">Rapid weakening to Deep Depression within +18h of crossing</span>
              </div>
            </div>
          </div>

          {/* Uncertainty Methodology Box */}
          <div className="command-card rounded-2xl p-5 border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/40 space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-slate-200 dark:border-slate-800">
              <span className="font-bold text-slate-900 dark:text-white text-sm flex items-center gap-2">
                <Cpu className="w-4 h-4 text-cyan-600 dark:text-cyan-400" />
                Uncertainty Estimation & Error Envelope
              </span>
              <span className="text-[10px] text-cyan-700 dark:text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/20">
                ENSEMBLE VARIANCE
              </span>
            </div>

            <p className="text-slate-600 dark:text-slate-300 text-[11px] leading-relaxed">
              "Forecast confidence decreases monotonically with increasing temporal prediction horizon due to non-linear atmospheric steering oscillations."
            </p>

            <div className="space-y-2 pt-1">
              <div className="flex items-center justify-between p-2 rounded-lg bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-850">
                <span className="text-slate-500 dark:text-slate-400">24-Hour Horizon:</span>
                <div className="flex items-center gap-3">
                  <span className="text-slate-700 dark:text-slate-300">Cross: ±24km | Along: ±32km</span>
                  <span className="text-emerald-600 dark:text-emerald-400 font-bold">89% Conf.</span>
                </div>
              </div>

              <div className="flex items-center justify-between p-2 rounded-lg bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-850">
                <span className="text-slate-500 dark:text-slate-400">48-Hour Horizon:</span>
                <div className="flex items-center gap-3">
                  <span className="text-slate-700 dark:text-slate-300">Cross: ±52km | Along: ±68km</span>
                  <span className="text-cyan-600 dark:text-cyan-400 font-bold">82% Conf.</span>
                </div>
              </div>

              <div className="flex items-center justify-between p-2 rounded-lg bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-850">
                <span className="text-slate-500 dark:text-slate-400">72-Hour Horizon:</span>
                <div className="flex items-center gap-3">
                  <span className="text-slate-700 dark:text-slate-300">Cross: ±95km | Along: ±120km</span>
                  <span className="text-amber-600 dark:text-amber-400 font-bold">74% Conf.</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
