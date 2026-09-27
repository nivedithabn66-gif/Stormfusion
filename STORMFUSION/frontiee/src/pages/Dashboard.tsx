import React, { useEffect, useState } from 'react';
import { 
  Radio, 
  Clock, 
  MapPin, 
  Wind, 
  Activity, 
  ShieldAlert, 
  TrendingUp, 
  Cpu, 
  Sparkles, 
  Eye, 
  HelpCircle,
  Compass,
  AlertTriangle,
  RefreshCw,
  Database,
  GitBranch,
  ArrowRight,
  ArrowDown,
  History,
  Layers,
  Waves
} from 'lucide-react';
import { CycloneTelemetry, ForecastHourData, TrackPoint, DistrictRisk, HistoricalAnalogue } from '../types/cyclone';
import { cycloneApi } from '../services/api';
import { CycloneStatusCard } from '../components/CycloneStatusCard';
import { CycloneMap } from '../components/CycloneMap';
import { DemoModeBadge } from '../components/DemoModeBadge';
import { CycloneLogo } from '../components/CycloneLogo';
import { getRiskSeverityColor } from '../utils/formatters';

export const Dashboard: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [cyclone, setCyclone] = useState<CycloneTelemetry | null>(null);
  const [forecastHours, setForecastHours] = useState<ForecastHourData[]>([]);
  const [intensityTrend, setIntensityTrend] = useState<any[]>([]);
  const [uncertaintyCone, setUncertaintyCone] = useState<[number, number][]>([]);
  const [track, setTrack] = useState<TrackPoint[]>([]);
  const [riskDistricts, setRiskDistricts] = useState<DistrictRisk[]>([]);
  const [analogues, setAnalogues] = useState<HistoricalAnalogue[]>([]);

  const loadAllData = async () => {
    setLoading(true);
    try {
      const [
        cycloneRes,
        forecastRes,
        trackRes,
        riskRes,
        analoguesRes
      ] = await Promise.all([
        cycloneApi.getCurrentCyclone(),
        cycloneApi.getForecast(),
        cycloneApi.getTrack(),
        cycloneApi.getRiskZones(),
        cycloneApi.getHistoricalAnalogues()
      ]);

      setCyclone(cycloneRes);
      setForecastHours(forecastRes.hours);
      setIntensityTrend(forecastRes.intensityTrend);
      setUncertaintyCone(forecastRes.uncertaintyCone);
      setTrack(trackRes);
      setRiskDistricts(riskRes);
      setAnalogues(analoguesRes);
    } catch (err) {
      console.error('Failed to load dashboard telemetry:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAllData();
  }, []);

  if (loading || !cyclone) {
    return (
      <div className="min-h-screen bg-slate-50 dark:bg-[#080C14] flex flex-col items-center justify-center space-y-4 font-mono">
        <div className="relative flex items-center justify-center">
          <div className="w-20 h-20 rounded-full border-2 border-cyan-500/20 border-t-cyan-500 dark:border-t-cyan-400 animate-spin"></div>
          <div className="absolute">
            <CycloneLogo size={38} />
          </div>
        </div>
        <div className="text-center space-y-1">
          <div className="text-sm font-bold text-slate-900 dark:text-white tracking-wider">
            SYNCHRONIZING STORMFUSION COMMAND CENTER
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Fetching INSAT-3D radiances, ConvLSTM inference weights, and GIS layers...
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 lg:p-6 space-y-6 max-w-[1700px] mx-auto">
      {/* 1. System Status Operations Header */}
      <div className="command-card rounded-2xl p-4 sm:p-5 border border-slate-300 dark:border-slate-800 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-3 h-3 rounded-full bg-emerald-600 animate-pulse ring-4 ring-emerald-500/20 shrink-0"></div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl sm:text-2xl font-black font-mono tracking-wider text-slate-950 dark:text-white">
                STORMFUSION <span className="text-cyan-700 dark:text-cyan-400 font-bold">| COMMAND CENTER</span>
              </h1>
            </div>
            <p className="text-xs text-slate-600 dark:text-slate-400 font-mono mt-0.5 font-medium">
              Multi-Source Satellite Intelligence & Automated Early Warning System
            </p>
          </div>
        </div>

        {/* Global Operations Telemetry Stats */}
        <div className="flex flex-wrap items-center gap-3 text-xs font-mono">
          <div className="px-3 py-1.5 rounded-xl bg-amber-50/90 dark:bg-slate-900/90 border border-amber-300 dark:border-slate-800 flex items-center gap-2 shadow-xs">
            <Radio className="w-3.5 h-3.5 text-amber-600 dark:text-cyan-400 animate-pulse" />
            <div>
              <span className="text-amber-900/80 dark:text-slate-500 text-[9px] block font-bold">DATA STATUS</span>
              <span className="text-amber-800 dark:text-amber-300 font-black">DEMO / SIMULATED</span>
            </div>
          </div>

          <div className="px-3 py-1.5 rounded-xl bg-slate-100 dark:bg-slate-900/90 border border-slate-300 dark:border-slate-800 flex items-center gap-2 shadow-xs">
            <Clock className="w-3.5 h-3.5 text-slate-600 dark:text-slate-400" />
            <div>
              <span className="text-slate-600 dark:text-slate-500 text-[9px] block font-bold">LAST UPDATED</span>
              <span className="text-slate-950 dark:text-slate-200 font-bold">{cyclone.lastUpdated}</span>
            </div>
          </div>

          <div className="px-3 py-1.5 rounded-xl bg-purple-50/90 dark:bg-slate-900/90 border border-purple-300 dark:border-slate-800 flex items-center gap-2 shadow-xs">
            <Compass className="w-3.5 h-3.5 text-purple-700 dark:text-purple-400" />
            <div>
              <span className="text-purple-900/80 dark:text-slate-500 text-[9px] block font-bold">BASIN REGION</span>
              <span className="text-purple-900 dark:text-purple-300 font-black">{cyclone.basin}</span>
            </div>
          </div>

          <button
            onClick={loadAllData}
            className="p-2.5 rounded-xl bg-white hover:bg-slate-100 dark:bg-slate-900 dark:hover:bg-slate-850 border border-slate-300 dark:border-slate-800 text-slate-800 hover:text-slate-950 dark:text-slate-300 dark:hover:text-white transition-colors shadow-xs"
            title="Refresh Ingest Pipeline"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* 2. Main Grid: Current Cyclone HUD + GIS Map */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Current Active Cyclone Card + AI Inference Engine */}
        <div className="lg:col-span-5 xl:col-span-5 space-y-6">
          <CycloneStatusCard cyclone={cyclone} />

          {/* AI Inference Engine Card with Telemetry & Pipeline */}
          <div className="command-card rounded-2xl p-5 border border-purple-300 dark:border-purple-500/20 bg-purple-50/40 dark:bg-slate-950/60 font-mono text-xs space-y-3.5">
            <div className="flex items-center justify-between pb-2 border-b border-purple-200 dark:border-slate-800">
              <span className="flex items-center gap-2 font-bold text-slate-950 dark:text-white uppercase text-sm">
                <Cpu className="w-4 h-4 text-purple-700 dark:text-purple-400" />
                AI Inference Engine
              </span>
              <span className="text-[10px] text-purple-900 dark:text-purple-300 bg-purple-100 dark:bg-purple-500/10 px-2 py-0.5 rounded border border-purple-300 dark:border-purple-500/30 font-bold">
                ConvLSTM-B0 v2.4
              </span>
            </div>

            {/* Reused Telemetry & Pipeline Information positioned near AI Inference Engine */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pb-1">
              <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-white/90 dark:bg-slate-900/80 border border-slate-300 dark:border-slate-800 text-[11px] font-mono">
                <Radio className="w-3.5 h-3.5 text-cyan-700 dark:text-cyan-400 shrink-0" />
                <span className="text-slate-600 dark:text-slate-400 font-semibold">TELEMETRY:</span>
                <span className="text-slate-950 dark:text-slate-200 font-bold">INSAT-3D / 3DR (ACTIVE)</span>
              </div>
              <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-white/90 dark:bg-slate-900/80 border border-slate-300 dark:border-slate-800 text-[11px] font-mono">
                <Database className="w-3.5 h-3.5 text-purple-700 dark:text-purple-400 shrink-0" />
                <span className="text-slate-600 dark:text-slate-400 font-semibold">PIPELINE:</span>
                <span className="text-slate-950 dark:text-slate-200 font-bold">ConvLSTM-B0 v2.4</span>
              </div>
            </div>

            <div className="space-y-2 text-slate-800 dark:text-slate-300">
              <div className="flex justify-between py-1 border-b border-purple-100 dark:border-slate-850">
                <span className="text-slate-600 dark:text-slate-500 font-semibold">Pattern Classification:</span>
                <span className="font-bold text-cyan-800 dark:text-cyan-300">{cyclone.cyclonePattern}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-purple-100 dark:border-slate-850">
                <span className="text-slate-600 dark:text-slate-500 font-semibold">Detection Confidence:</span>
                <span className="font-bold text-emerald-800 dark:text-emerald-400">{cyclone.detectionConfidencePercent}%</span>
              </div>
              <div className="flex justify-between py-1 border-b border-purple-100 dark:border-slate-850">
                <span className="text-slate-600 dark:text-slate-500 font-semibold">Predicted Eye Diameter:</span>
                <span className="font-bold text-slate-950 dark:text-white">28 km (Pin-hole developing)</span>
              </div>
              <div className="flex justify-between py-1 border-b border-purple-100 dark:border-slate-850">
                <span className="text-slate-600 dark:text-slate-500 font-semibold">Rapid Intensification (RI):</span>
                <span className="font-bold text-amber-800 dark:text-amber-400">{cyclone.rapidIntensificationRisk} Probability</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-600 dark:text-slate-500 font-semibold">Fusion Architecture:</span>
                <span className="font-bold text-slate-950 dark:text-slate-300">EfficientNet-B0 + ConvLSTM</span>
              </div>
            </div>

            <div className="p-2.5 rounded-lg bg-purple-100/80 dark:bg-purple-950/30 border border-purple-300 dark:border-purple-500/30 text-[11px] text-purple-950 dark:text-purple-200 font-medium">
              <strong>Deep Feature Synopsis:</strong> Strong upper-level divergence combined with 30.8°C SST supports sustained deep convection. Eye boundary exhibits axisymmetric eyewall consolidation over last 6h.
            </div>
          </div>
        </div>

        {/* Right Column: GIS Map + Historical Pattern Intelligence Panel */}
        <div className="lg:col-span-7 xl:col-span-7 space-y-6">
          <CycloneMap
            cyclone={cyclone}
            track={track}
            uncertaintyConeCoords={uncertaintyCone}
            districtRisks={riskDistricts}
            heightClass="h-[440px] xl:h-[460px]"
          />

          {/* Integrated Panel: HISTORICAL PATTERN INTELLIGENCE */}
          <div className="command-card rounded-2xl p-4 sm:p-5 border border-slate-300 dark:border-slate-800 space-y-4">
            {/* Panel Header */}
            <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-200 dark:border-slate-800">
              <div className="flex items-center gap-2.5">
                <div className="p-2 rounded-lg bg-cyan-500/15 border border-cyan-500/30 text-cyan-700 dark:text-cyan-400 shadow-xs">
                  <GitBranch className="w-4 h-4" />
                </div>
                <div>
                  <h2 className="text-sm sm:text-base font-black font-mono tracking-wide text-slate-950 dark:text-white uppercase">
                    HISTORICAL PATTERN INTELLIGENCE
                  </h2>
                  <p className="text-[11px] sm:text-xs text-slate-600 dark:text-slate-400 font-mono font-medium">
                    Historical Data → Pattern Recognition → Prediction
                  </p>
                </div>
              </div>

              <span className="text-[10px] sm:text-[11px] font-mono font-bold px-2.5 py-1 rounded bg-cyan-100 dark:bg-cyan-950/80 text-cyan-900 dark:text-cyan-300 border border-cyan-300 dark:border-cyan-700/60 shadow-xs">
                AI PIPELINE
              </span>
            </div>

            {/* Step-by-Step AI Pipeline */}
            <div className="space-y-3.5 font-mono">
              {/* STEP 1: HISTORICAL IDENTIFICATION */}
              <div className="p-3.5 sm:p-4 rounded-xl bg-[#F8FAFC] dark:bg-slate-900/80 border border-slate-300 dark:border-slate-800 shadow-xs space-y-2.5">
                <div className="flex flex-wrap items-center justify-between gap-2 pb-2.5 border-b border-slate-200 dark:border-slate-800">
                  <div className="flex items-center gap-2.5">
                    <span className="px-2.5 py-1 rounded-md bg-cyan-600 text-white font-black text-[11px] sm:text-xs tracking-wider uppercase shadow-xs">
                      STEP 1
                    </span>
                    <h3 className="font-extrabold uppercase text-slate-950 dark:text-white text-xs sm:text-sm tracking-wide">
                      HISTORICAL IDENTIFICATION
                    </h3>
                  </div>
                  <span className="text-[10px] sm:text-[11px] font-bold px-2.5 py-0.5 rounded bg-cyan-100 dark:bg-cyan-950/80 text-cyan-900 dark:text-cyan-300 border border-cyan-300 dark:border-cyan-700/60">
                    IBTrACS Archive Matching
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 pt-0.5">
                  <div className="p-3 rounded-lg bg-white dark:bg-slate-800/90 border border-slate-200 dark:border-slate-700 flex justify-between items-center shadow-xs">
                    <div>
                      <span className="text-[10px] text-slate-500 dark:text-slate-400 font-bold uppercase block tracking-wider">
                        Analogue Match 1
                      </span>
                      <span className="font-extrabold text-slate-900 dark:text-slate-100 text-xs sm:text-sm">
                        Hudhud (2014)
                      </span>
                    </div>
                    <span className="text-emerald-700 dark:text-emerald-300 font-black text-xs sm:text-sm bg-emerald-50 dark:bg-emerald-950/60 px-2.5 py-1 rounded-md border border-emerald-300 dark:border-emerald-700/60">
                      87.6%
                    </span>
                  </div>

                  <div className="p-3 rounded-lg bg-white dark:bg-slate-800/90 border border-slate-200 dark:border-slate-700 flex justify-between items-center shadow-xs">
                    <div>
                      <span className="text-[10px] text-slate-500 dark:text-slate-400 font-bold uppercase block tracking-wider">
                        Analogue Match 2
                      </span>
                      <span className="font-extrabold text-slate-900 dark:text-slate-100 text-xs sm:text-sm">
                        Phailin (2013)
                      </span>
                    </div>
                    <span className="text-emerald-700 dark:text-emerald-300 font-black text-xs sm:text-sm bg-emerald-50 dark:bg-emerald-950/60 px-2.5 py-1 rounded-md border border-emerald-300 dark:border-emerald-700/60">
                      81.2%
                    </span>
                  </div>
                </div>
              </div>

              {/* STEP 2: PATTERN CLASSIFICATION */}
              <div className="p-3.5 sm:p-4 rounded-xl bg-[#F8FAFC] dark:bg-slate-900/80 border border-slate-300 dark:border-slate-800 shadow-xs space-y-2.5">
                <div className="flex flex-wrap items-center justify-between gap-2 pb-2.5 border-b border-slate-200 dark:border-slate-800">
                  <div className="flex items-center gap-2.5">
                    <span className="px-2.5 py-1 rounded-md bg-purple-600 text-white font-black text-[11px] sm:text-xs tracking-wider uppercase shadow-xs">
                      STEP 2
                    </span>
                    <h3 className="font-extrabold uppercase text-slate-950 dark:text-white text-xs sm:text-sm tracking-wide">
                      PATTERN CLASSIFICATION
                    </h3>
                  </div>
                  <span className="text-[10px] sm:text-[11px] font-bold px-2.5 py-0.5 rounded bg-purple-100 dark:bg-purple-950/80 text-purple-900 dark:text-purple-300 border border-purple-300 dark:border-purple-700/60">
                    EfficientNet + ConvLSTM
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 pt-0.5">
                  <div className="p-3 rounded-lg bg-white dark:bg-slate-800/90 border border-slate-200 dark:border-slate-700 shadow-xs">
                    <span className="text-[10px] text-slate-500 dark:text-slate-400 font-bold uppercase block tracking-wider">
                      Morphology Pattern
                    </span>
                    <span className="font-extrabold text-cyan-800 dark:text-cyan-300 text-xs sm:text-sm block mt-0.5">
                      Curved Band / Developing Eye
                    </span>
                  </div>

                  <div className="p-3 rounded-lg bg-white dark:bg-slate-800/90 border border-slate-200 dark:border-slate-700 flex justify-between items-center shadow-xs">
                    <div>
                      <span className="text-[10px] text-slate-500 dark:text-slate-400 font-bold uppercase block tracking-wider">
                        IMD Intensity Class
                      </span>
                      <span className="font-black text-rose-600 dark:text-rose-400 text-xs sm:text-sm">
                        Very Severe Cyclonic Storm (VSCS)
                      </span>
                    </div>
                    <span className="text-emerald-700 dark:text-emerald-300 font-black text-xs sm:text-sm bg-emerald-50 dark:bg-emerald-950/60 px-2.5 py-1 rounded-md border border-emerald-300 dark:border-emerald-700/60">
                      96% Conf.
                    </span>
                  </div>
                </div>
              </div>

              {/* STEP 3: CYCLONE PREDICTION */}
              <div className="p-3.5 sm:p-4 rounded-xl bg-[#F8FAFC] dark:bg-slate-900/80 border border-slate-300 dark:border-slate-800 shadow-xs space-y-2.5">
                <div className="flex flex-wrap items-center justify-between gap-2 pb-2.5 border-b border-slate-200 dark:border-slate-800">
                  <div className="flex items-center gap-2.5">
                    <span className="px-2.5 py-1 rounded-md bg-rose-600 text-white font-black text-[11px] sm:text-xs tracking-wider uppercase shadow-xs">
                      STEP 3
                    </span>
                    <h3 className="font-extrabold uppercase text-slate-950 dark:text-white text-xs sm:text-sm tracking-wide">
                      CYCLONE PREDICTION
                    </h3>
                  </div>
                  <span className="text-[10px] sm:text-[11px] font-bold px-2.5 py-0.5 rounded bg-rose-100 dark:bg-rose-950/80 text-rose-900 dark:text-rose-300 border border-rose-300 dark:border-rose-700/60">
                    Trajectory + Intensity Forecast
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-2.5 pt-0.5 text-center">
                  <div className="p-3 rounded-lg bg-white dark:bg-slate-800/90 border border-slate-200 dark:border-slate-700 shadow-xs">
                    <span className="text-slate-600 dark:text-slate-400 block text-[10px] sm:text-xs font-bold uppercase tracking-wider">
                      24h Forecast
                    </span>
                    <span className="font-black text-slate-950 dark:text-white text-sm sm:text-base mt-0.5 block">
                      86 kt
                    </span>
                    <span className="block text-[9px] sm:text-[10px] text-slate-500 dark:text-slate-400 mt-1 font-semibold">
                      Track: WNW
                    </span>
                  </div>
                  <div className="p-3 rounded-lg bg-white dark:bg-slate-800/90 border border-slate-200 dark:border-slate-700 shadow-xs">
                    <span className="text-slate-600 dark:text-slate-400 block text-[10px] sm:text-xs font-bold uppercase tracking-wider">
                      48h Peak
                    </span>
                    <span className="font-black text-rose-600 dark:text-rose-400 text-sm sm:text-base mt-0.5 block">
                      91 kt
                    </span>
                    <span className="block text-[9px] sm:text-[10px] text-rose-600 dark:text-rose-400 mt-1 font-bold">
                      Peak Risk
                    </span>
                  </div>
                  <div className="p-3 rounded-lg bg-white dark:bg-slate-800/90 border border-slate-200 dark:border-slate-700 shadow-xs">
                    <span className="text-slate-600 dark:text-slate-400 block text-[10px] sm:text-xs font-bold uppercase tracking-wider">
                      72h Landfall
                    </span>
                    <span className="font-black text-slate-950 dark:text-white text-sm sm:text-base mt-0.5 block">
                      78 kt
                    </span>
                    <span className="block text-[9px] sm:text-[10px] text-emerald-700 dark:text-emerald-400 mt-1 font-bold">
                      North AP
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* RISK INDEX PANEL */}
            <div className="p-3 rounded-xl bg-[#F8FAFC] dark:bg-slate-900/70 border border-slate-300 dark:border-slate-800 space-y-2 font-mono text-xs shadow-xs">
              <div className="flex items-center justify-between pb-1 border-b border-slate-200 dark:border-slate-800">
                <div className="flex items-center gap-2">
                  <ShieldAlert className="w-3.5 h-3.5 text-amber-600 dark:text-amber-400" />
                  <span className="font-bold text-slate-950 dark:text-white uppercase text-[10px] sm:text-[11px]">
                    COASTAL RISK INDEX & VULNERABILITY
                  </span>
                </div>
                <span className="text-[8px] font-bold text-amber-900 dark:text-amber-300 bg-amber-100 dark:bg-amber-500/10 px-1.5 py-0.5 rounded border border-amber-300 dark:border-amber-500/20">
                  POSTGIS EXPOSURE
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-2">
                {riskDistricts.slice(0, 4).map((d) => {
                  const colors = getRiskSeverityColor(d.riskLevel);
                  return (
                    <div key={d.district} className="p-2 rounded-lg bg-white dark:bg-slate-850 border border-slate-200 dark:border-slate-800 space-y-1 shadow-2xs text-[10px]">
                      <div className="flex items-center justify-between">
                        <strong className="text-slate-950 dark:text-white">{d.district}</strong>
                        <span className={`px-1 py-0.2 rounded text-[8px] font-black ${colors.badgeBg}`}>
                          {d.riskLevel}
                        </span>
                      </div>
                      <div className="flex justify-between text-slate-600 dark:text-slate-300 text-[9px]">
                        <span>Wind: <strong className="text-rose-600 dark:text-rose-400">{d.windRiskKts} kt</strong></span>
                        <span>Surge: <strong className="text-sky-600 dark:text-sky-400">{d.stormSurgeMeters}m</strong></span>
                      </div>
                      {d.evacuationRecommended && (
                        <div className="text-[8px] font-bold text-rose-700 dark:text-rose-400 flex items-center gap-0.5">
                          <AlertTriangle className="w-2.5 h-2.5 shrink-0" />
                          Evacuation Protocol
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
