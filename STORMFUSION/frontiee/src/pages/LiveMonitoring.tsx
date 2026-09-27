import React, { useState, useEffect } from 'react';
import { 
  Radio, 
  Clock, 
  MapPin, 
  Layers, 
  Thermometer, 
  Wind, 
  Gauge, 
  Activity,
  RefreshCw,
  Eye,
  Sliders,
  Compass,
  AlertTriangle,
  BrainCircuit
} from 'lucide-react';
import { CycloneTelemetry, SatelliteFrame, GradCamAnalysis } from '../types/cyclone';
import { cycloneApi } from '../services/api';
import { SatelliteViewer } from '../components/SatelliteViewer';
import { GradCAMViewer } from '../components/GradCAMViewer';
import { DemoModeBadge } from '../components/DemoModeBadge';

export const LiveMonitoring: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [cyclone, setCyclone] = useState<CycloneTelemetry | null>(null);
  const [satelliteFrames, setSatelliteFrames] = useState<SatelliteFrame[]>([]);
  const [gradCamAnalysis, setGradCamAnalysis] = useState<GradCamAnalysis | null>(null);
  const [soundings, setSoundings] = useState<any>(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const [c, s, g, env] = await Promise.all([
          cycloneApi.getCurrentCyclone(),
          cycloneApi.getSatelliteFrames(),
          cycloneApi.getGradCamAnalysis(),
          cycloneApi.getEnvironmentalSoundings()
        ]);
        setCyclone(c);
        setSatelliteFrames(s);
        setGradCamAnalysis(g);
        setSoundings(env);
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
        <Activity className="w-8 h-8 animate-spin" />
      </div>
    );
  }

  return (
    <div className="p-4 lg:p-6 space-y-6 max-w-[1700px] mx-auto">
      {/* Header */}
      <div className="command-card rounded-2xl p-5 border border-slate-200 dark:border-slate-800 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-cyan-500/10 dark:bg-cyan-500/15 border border-cyan-500/30 text-cyan-600 dark:text-cyan-400">
              <Radio className="w-6 h-6 animate-pulse" />
            </div>
            <div>
              <h1 className="text-xl font-bold font-mono tracking-wide text-slate-900 dark:text-white uppercase">
                Real-Time Remote Sensing & Multi-Source Satellite Telemetry
              </h1>
              <p className="text-xs text-slate-500 dark:text-slate-400 font-mono">
                Direct downlinks from INSAT-3D/3DR, ECMWF ERA5, and Coastal Doppler Radars (Visakhapatnam & Paradip)
              </p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <DemoModeBadge subtle />
          <div className="px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 font-mono text-xs text-slate-700 dark:text-slate-300">
            Last Sweep: <strong>14:32 IST</strong>
          </div>
        </div>
      </div>

      {/* Atmospheric & Oceanic Environmental Sounding Diagnostics */}
      {soundings && (
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 font-mono text-xs">
          <div className="command-card p-3 rounded-xl border border-slate-200 dark:border-slate-800">
            <span className="text-slate-500 text-[10px] block">SEA SURFACE TEMP (SST)</span>
            <div className="text-lg font-bold text-rose-600 dark:text-rose-400 mt-1">{soundings.seaSurfaceTemperatureC}°C</div>
            <span className="text-[10px] text-slate-500 dark:text-slate-400">Warm thermal anomaly (+1.4°C)</span>
          </div>

          <div className="command-card p-3 rounded-xl border border-slate-200 dark:border-slate-800">
            <span className="text-slate-500 text-[10px] block">VERTICAL WIND SHEAR</span>
            <div className="text-lg font-bold text-emerald-600 dark:text-emerald-400 mt-1">{soundings.verticalWindShearKts} kt</div>
            <span className="text-[10px] text-emerald-600 dark:text-emerald-400">Low (Highly Favorable)</span>
          </div>

          <div className="command-card p-3 rounded-xl border border-slate-200 dark:border-slate-800">
            <span className="text-slate-500 text-[10px] block">OCEAN HEAT CONTENT</span>
            <div className="text-lg font-bold text-cyan-600 dark:text-cyan-400 mt-1">{soundings.oceanHeatContentKjCm2} kJ/cm²</div>
            <span className="text-[10px] text-slate-500 dark:text-slate-400">High Energetic Reserve</span>
          </div>

          <div className="command-card p-3 rounded-xl border border-slate-200 dark:border-slate-800">
            <span className="text-slate-500 text-[10px] block">MID-LEVEL MOISTURE</span>
            <div className="text-lg font-bold text-sky-600 dark:text-sky-400 mt-1">{soundings.midLevelMoisturePercent}%</div>
            <span className="text-[10px] text-slate-500 dark:text-slate-400">700-500 hPa layer</span>
          </div>

          <div className="command-card p-3 rounded-xl border border-slate-200 dark:border-slate-800 col-span-2 sm:col-span-1">
            <span className="text-slate-500 text-[10px] block">OUTFLOW EFFICIENCY</span>
            <div className="text-sm font-bold text-purple-600 dark:text-purple-400 mt-1">Dual Radial</div>
            <span className="text-[10px] text-slate-500 dark:text-slate-400">Strong anti-cyclonic diffluence</span>
          </div>
        </div>
      )}

      {/* 1. Multi-Channel Satellite Viewer (Full Width) */}
      <div className="space-y-2">
        <div className="text-xs font-mono font-bold text-slate-500 dark:text-slate-400 flex items-center justify-between">
          <span>INSAT-3D MULTISPECTRAL RADIOMETER</span>
          <span className="text-purple-600 dark:text-purple-400">Channels: IR / VIS / WV / Enhanced</span>
        </div>
        <SatelliteViewer frames={satelliteFrames} />
      </div>

      {/* 2. Explainable AI (XAI) & Grad-CAM Attention Module */}
      {gradCamAnalysis && (
        <div className="space-y-2">
          <GradCAMViewer analysis={gradCamAnalysis} />
        </div>
      )}

      {/* Sensor Ingestion Pipeline Telemetry */}
      <div className="command-card rounded-2xl p-5 border border-slate-200 dark:border-slate-800 font-mono text-xs">
        <h3 className="font-bold text-slate-900 dark:text-white mb-3 flex items-center gap-2">
          <Layers className="w-4 h-4 text-cyan-600 dark:text-cyan-400" />
          Multi-Source Satellite Data Ingest Matrix
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          <div className="p-3 bg-slate-50 dark:bg-slate-900/60 rounded-xl border border-slate-200 dark:border-slate-850">
            <div className="flex justify-between text-slate-500 dark:text-slate-400 text-[11px] mb-1">
              <span className="font-bold text-slate-800 dark:text-white">INSAT-3D Imager</span>
              <span className="text-emerald-600 dark:text-emerald-400">OPERATIONAL</span>
            </div>
            <p className="text-slate-600 dark:text-slate-400 text-[11px]">Channels: 6 Multispectral (TIR-1, TIR-2, MIR, VIS, SWIR, WV)</p>
            <div className="text-[10px] text-slate-500 mt-2">Latency: 12 mins • MOSDAC Protocol</div>
          </div>

          <div className="p-3 bg-slate-50 dark:bg-slate-900/60 rounded-xl border border-slate-200 dark:border-slate-850">
            <div className="flex justify-between text-slate-500 dark:text-slate-400 text-[11px] mb-1">
              <span className="font-bold text-slate-800 dark:text-white">INSAT-3DR Sounder</span>
              <span className="text-emerald-600 dark:text-emerald-400">OPERATIONAL</span>
            </div>
            <p className="text-slate-600 dark:text-slate-400 text-[11px]">Atmospheric Profiles: 19 Channels (Temperature, Moisture, Ozone)</p>
            <div className="text-[10px] text-slate-500 mt-2">Latency: 15 mins • ISRO Payload</div>
          </div>

          <div className="p-3 bg-slate-50 dark:bg-slate-900/60 rounded-xl border border-slate-200 dark:border-slate-850">
            <div className="flex justify-between text-slate-500 dark:text-slate-400 text-[11px] mb-1">
              <span className="font-bold text-slate-800 dark:text-white">ECMWF ERA5 Reanalysis</span>
              <span className="text-emerald-600 dark:text-emerald-400">SYNCED</span>
            </div>
            <p className="text-slate-600 dark:text-slate-400 text-[11px]">Environmental Boundary: 0.25° Global Gridded Pressure Fields</p>
            <div className="text-[10px] text-slate-500 mt-2">Cycle: 6-hourly Assimilation</div>
          </div>

          <div className="p-3 bg-slate-50 dark:bg-slate-900/60 rounded-xl border border-slate-200 dark:border-slate-850">
            <div className="flex justify-between text-slate-500 dark:text-slate-400 text-[11px] mb-1">
              <span className="font-bold text-slate-800 dark:text-white">DWR Radar Coastal Array</span>
              <span className="text-emerald-600 dark:text-emerald-400">ACTIVE</span>
            </div>
            <p className="text-slate-600 dark:text-slate-400 text-[11px]">Doppler Stations: Visakhapatnam, Machilipatnam, Gopalpur, Paradip</p>
            <div className="text-[10px] text-slate-500 mt-2">Sweep Interval: 10 mins</div>
          </div>
        </div>
      </div>
    </div>
  );
};
