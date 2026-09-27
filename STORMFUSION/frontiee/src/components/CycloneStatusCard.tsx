import React from 'react';
import { 
  Wind, 
  Gauge, 
  Compass, 
  Navigation, 
  Activity, 
  Sparkles, 
  ShieldAlert, 
  Crosshair, 
  TrendingUp, 
  Cpu 
} from 'lucide-react';
import { CycloneTelemetry } from '../types/cyclone';
import { formatCoords, getCategoryColor } from '../utils/formatters';
import { DemoModeBadge } from './DemoModeBadge';
import { CycloneLogo } from './CycloneLogo';

interface CycloneStatusCardProps {
  cyclone: CycloneTelemetry;
}

export const CycloneStatusCard: React.FC<CycloneStatusCardProps> = ({ cyclone }) => {
  const categoryStyles = getCategoryColor(cyclone.category);

  return (
    <div className="command-card rounded-2xl p-4 sm:p-5 relative overflow-hidden space-y-4">
      {/* Decorative Radar Grid Accents (purely background graphics) */}
      <div className="absolute -right-12 -top-12 w-48 h-48 rounded-full border border-cyan-500/10 pointer-events-none" aria-hidden="true"></div>
      <div className="absolute -right-6 -top-6 w-36 h-36 rounded-full border border-cyan-500/15 pointer-events-none" aria-hidden="true"></div>

      {/* 1. CYCLONE DEMO HEADER */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-300 dark:border-slate-800">
        <div className="flex items-center gap-3 min-w-0">
          <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-rose-500/15 via-blue-500/10 to-slate-100 dark:via-cyan-500/10 dark:to-blue-600/15 border border-cyan-500/50 dark:border-cyan-500/30 flex items-center justify-center shadow-[0_2px_10px_rgba(2,132,199,0.15)] dark:shadow-[0_0_20px_rgba(6,182,212,0.2)] shrink-0">
            <CycloneLogo size={30} />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <h2 className="text-xl font-black font-mono tracking-wide text-slate-950 dark:text-white uppercase leading-tight">
                {cyclone.name}
              </h2>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-emerald-100 dark:bg-emerald-500/15 text-emerald-800 dark:text-emerald-400 border border-emerald-300 dark:border-emerald-500/30 flex items-center gap-1.5 shrink-0">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-600 animate-pulse"></span>
                {cyclone.currentStatus}
              </span>
            </div>
            <p className="text-xs text-slate-600 dark:text-slate-400 font-mono mt-1 leading-normal font-semibold">
              Basin: <span className="text-slate-950 dark:text-slate-200 font-bold">{cyclone.basin}</span>
            </p>
          </div>
        </div>

        <div className="shrink-0">
          <DemoModeBadge subtle />
        </div>
      </div>

      {/* 2. CLASSIFICATION SECTION */}
      <div className={`p-3.5 rounded-xl border flex flex-col sm:flex-row sm:items-center justify-between gap-3 transition-colors ${categoryStyles.bg} ${categoryStyles.border}`}>
        {/* Left: Classification */}
        <div className="flex items-center gap-3 min-w-0">
          <div className="p-2 rounded-lg bg-white/90 dark:bg-slate-900/70 border border-rose-200 dark:border-rose-900/50 shadow-xs shrink-0">
            <ShieldAlert className={`w-5 h-5 ${categoryStyles.text}`} />
          </div>
          <div className="min-w-0">
            <div className="text-[10px] font-mono text-slate-700 dark:text-slate-400 uppercase tracking-wider font-bold">
              Classification (IMD Scale)
            </div>
            <div className={`text-sm sm:text-base font-black font-mono tracking-wide ${categoryStyles.text} leading-snug break-words`}>
              {cyclone.category.toUpperCase()}
            </div>
          </div>
        </div>

        {/* Right: Pattern Signature */}
        <div className="sm:text-right border-t sm:border-t-0 pt-2 sm:pt-0 border-slate-200/80 dark:border-slate-800/60 shrink-0">
          <div className="text-[10px] font-mono text-slate-700 dark:text-slate-400 uppercase tracking-wider font-bold">
            Pattern Signature
          </div>
          <div className="text-xs sm:text-sm font-black text-slate-950 dark:text-white font-mono leading-snug">
            {cyclone.cyclonePattern}
          </div>
        </div>
      </div>

      {/* 3. PRIMARY CORE TELEMETRY (2 x 2 Layout: Row 1 = Max Wind, Central Pressure; Row 2 = Vector, Eye Location) */}
      <div className="space-y-2">
        <div className="text-[10px] font-mono uppercase tracking-widest text-slate-700 dark:text-slate-400 font-bold flex items-center gap-1.5">
          <Activity className="w-3.5 h-3.5 text-cyan-700 dark:text-cyan-400 shrink-0" />
          Primary Core Telemetry
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
          {/* ROW 1 - Card 1: MAX WIND */}
          <div className="p-4 rounded-xl bg-[#F8FAFC] dark:bg-slate-900/80 border border-slate-300 dark:border-slate-800 flex flex-col justify-between gap-2.5 transition-all shadow-xs hover:border-cyan-500/60 min-w-0 min-h-[135px]">
            <div className="space-y-1">
              <div className="flex items-center justify-between gap-1.5 font-bold text-[10px] font-mono uppercase tracking-wider text-slate-700 dark:text-slate-400">
                <span className="flex items-center gap-1.5">
                  <Wind className="w-3.5 h-3.5 text-cyan-700 dark:text-cyan-400 shrink-0" />
                  MAX WIND
                </span>
                <span className="text-[9px] font-mono font-bold px-1.5 py-0.5 rounded bg-cyan-100 dark:bg-cyan-500/10 text-cyan-900 dark:text-cyan-300 border border-cyan-300 dark:border-cyan-500/20 inline-block leading-none">
                  1-Min Mean
                </span>
              </div>
            </div>

            <div className="flex items-baseline gap-1.5">
              <span className="text-3xl font-black font-mono text-slate-950 dark:text-white tracking-tight leading-none">
                {cyclone.maxSustainedWindKts}
              </span>
              <span className="text-sm font-mono font-bold text-slate-600 dark:text-slate-400">kt</span>
            </div>

            <div className="pt-2 border-t border-slate-200 dark:border-slate-800 text-xs font-mono font-bold text-cyan-800 dark:text-cyan-400 leading-snug">
              {cyclone.maxSustainedWindKmh} km/h
            </div>
          </div>

          {/* ROW 1 - Card 2: CENTRAL PRESSURE */}
          <div className="p-4 rounded-xl bg-[#F8FAFC] dark:bg-slate-900/80 border border-slate-300 dark:border-slate-800 flex flex-col justify-between gap-2.5 transition-all shadow-xs hover:border-rose-500/60 min-w-0 min-h-[135px]">
            <div className="space-y-1">
              <div className="flex items-center justify-between gap-1.5 font-bold text-[10px] font-mono uppercase tracking-wider text-slate-700 dark:text-slate-400">
                <span className="flex items-center gap-1.5">
                  <Gauge className="w-3.5 h-3.5 text-rose-700 dark:text-rose-400 shrink-0" />
                  CENTRAL PRESSURE
                </span>
                <span className="text-[9px] font-mono font-bold px-1.5 py-0.5 rounded bg-rose-100 dark:bg-rose-500/10 text-rose-900 dark:text-rose-300 border border-rose-300 dark:border-rose-500/20 inline-block leading-none">
                  Barometric
                </span>
              </div>
            </div>

            <div className="flex items-baseline gap-1.5">
              <span className="text-3xl font-black font-mono text-slate-950 dark:text-white tracking-tight leading-none">
                {cyclone.centralPressureHpa}
              </span>
              <span className="text-sm font-mono font-bold text-slate-600 dark:text-slate-400">hPa</span>
            </div>

            <div className="pt-2 border-t border-slate-200 dark:border-slate-800 text-xs font-mono font-bold text-rose-700 dark:text-rose-400 leading-snug">
              -37 hPa DROP
            </div>
          </div>

          {/* ROW 2 - Card 3: VECTOR */}
          <div className="p-4 rounded-xl bg-[#F8FAFC] dark:bg-slate-900/80 border border-slate-300 dark:border-slate-800 flex flex-col justify-between gap-2.5 transition-all shadow-xs hover:border-sky-500/60 min-w-0 min-h-[135px]">
            <div className="space-y-1">
              <div className="flex items-center justify-between gap-1.5 font-bold text-[10px] font-mono uppercase tracking-wider text-slate-700 dark:text-slate-400">
                <span className="flex items-center gap-1.5">
                  <Navigation className="w-3.5 h-3.5 text-sky-700 dark:text-sky-400 shrink-0" />
                  VECTOR
                </span>
                <span className="text-[9px] font-mono font-bold px-1.5 py-0.5 rounded bg-sky-100 dark:bg-sky-500/10 text-sky-900 dark:text-sky-300 border border-sky-300 dark:border-sky-500/20 inline-block leading-none">
                  Track Vector
                </span>
              </div>
            </div>

            <div className="flex items-baseline gap-1.5">
              <span className="text-3xl font-black font-mono text-slate-950 dark:text-white tracking-tight leading-none">
                {cyclone.movementSpeedKmh}
              </span>
              <span className="text-sm font-mono font-bold text-slate-600 dark:text-slate-400">km/h</span>
            </div>

            <div className="pt-2 border-t border-slate-200 dark:border-slate-800 text-xs font-mono font-bold text-slate-800 dark:text-slate-300 leading-snug">
              {cyclone.movementDirection} • 315° AZIMUTH
            </div>
          </div>

          {/* ROW 2 - Card 4: EYE LOCATION */}
          <div className="p-4 rounded-xl bg-[#F8FAFC] dark:bg-slate-900/80 border border-slate-300 dark:border-slate-800 flex flex-col justify-between gap-2.5 transition-all shadow-xs hover:border-emerald-500/60 min-w-0 min-h-[135px]">
            <div className="space-y-1">
              <div className="flex items-center justify-between gap-1.5 font-bold text-[10px] font-mono uppercase tracking-wider text-slate-700 dark:text-slate-400">
                <span className="flex items-center gap-1.5">
                  <Crosshair className="w-3.5 h-3.5 text-emerald-700 dark:text-emerald-400 shrink-0" />
                  EYE LOCATION
                </span>
                <span className="text-[9px] font-mono font-bold px-1.5 py-0.5 rounded bg-emerald-100 dark:bg-emerald-500/10 text-emerald-900 dark:text-emerald-400 border border-emerald-300 dark:border-emerald-500/20 inline-block leading-none">
                  INSAT-3D Fix
                </span>
              </div>
            </div>

            <div className="flex items-baseline gap-2">
              <span className="text-lg font-black font-mono text-slate-950 dark:text-white tracking-tight leading-none">
                {cyclone.coordinates.lat.toFixed(2)}° N
              </span>
              <span className="text-slate-400 dark:text-slate-600 font-bold">•</span>
              <span className="text-lg font-black font-mono text-slate-950 dark:text-white tracking-tight leading-none">
                {cyclone.coordinates.lng.toFixed(2)}° E
              </span>
            </div>

            <div className="pt-2 border-t border-slate-200 dark:border-slate-800 text-[11px] font-mono text-slate-700 dark:text-slate-400 font-semibold leading-snug">
              450 km SE of Visakhapatnam
            </div>
          </div>
        </div>
      </div>

      {/* 4. AI MODEL CONFIDENCE + RISK INDEX (Clearly Separated Cards) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 pt-1">
        {/* Left Card: AI MODEL CONFIDENCE */}
        <div className="p-4 rounded-xl bg-[#F8FAFC] dark:bg-slate-900/70 border border-slate-300 dark:border-slate-800 space-y-3 shadow-xs">
          <div className="flex items-center justify-between pb-2 border-b border-slate-200 dark:border-slate-800">
            <span className="flex items-center gap-1.5 text-xs font-mono font-bold uppercase tracking-wider text-slate-800 dark:text-slate-200">
              <Sparkles className="w-4 h-4 text-purple-700 dark:text-purple-400" />
              AI Model Confidence
            </span>
            <span className="text-[10px] font-mono font-bold text-purple-900 dark:text-purple-300 bg-purple-100 dark:bg-purple-500/10 px-1.5 py-0.5 rounded border border-purple-300 dark:border-purple-500/20">
              CALIBRATED
            </span>
          </div>

          <div className="space-y-2.5">
            {/* Detection Conf */}
            <div className="space-y-1">
              <div className="flex items-center justify-between text-[10px] font-mono font-bold text-slate-600 dark:text-slate-400 uppercase">
                <span>Detection Conf.</span>
                <span className="text-emerald-800 dark:text-emerald-400">HIGH</span>
              </div>
              <div className="flex items-baseline justify-between">
                <span className="text-xl font-black font-mono text-cyan-800 dark:text-cyan-300">{cyclone.detectionConfidencePercent}%</span>
              </div>
              <div className="w-full bg-slate-200 dark:bg-slate-800 h-1.5 rounded-full overflow-hidden">
                <div className="bg-cyan-600 dark:bg-cyan-400 h-full rounded-full" style={{ width: `${cyclone.detectionConfidencePercent}%` }}></div>
              </div>
            </div>

            {/* Model Conf */}
            <div className="space-y-1 pt-1 border-t border-slate-100 dark:border-slate-800/80">
              <div className="flex items-center justify-between text-[10px] font-mono font-bold text-slate-600 dark:text-slate-400 uppercase">
                <span>Model Conf.</span>
                <span className="text-purple-800 dark:text-purple-300">OPTIMAL</span>
              </div>
              <div className="flex items-baseline justify-between">
                <span className="text-xl font-black font-mono text-purple-900 dark:text-purple-300">{cyclone.modelConfidencePercent}%</span>
              </div>
              <div className="w-full bg-slate-200 dark:bg-slate-800 h-1.5 rounded-full overflow-hidden">
                <div className="bg-purple-600 dark:bg-purple-400 h-full rounded-full" style={{ width: `${cyclone.modelConfidencePercent}%` }}></div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Card: RISK INDEX */}
        <div className="p-4 rounded-xl bg-[#F8FAFC] dark:bg-slate-900/70 border border-slate-300 dark:border-slate-800 space-y-3 shadow-xs">
          <div className="flex items-center justify-between pb-2 border-b border-slate-200 dark:border-slate-800">
            <span className="flex items-center gap-1.5 text-xs font-mono font-bold uppercase tracking-wider text-slate-800 dark:text-slate-200">
              <TrendingUp className="w-4 h-4 text-amber-700 dark:text-amber-400" />
              Risk Index
            </span>
            <span className="text-[10px] font-mono font-bold text-amber-900 dark:text-amber-300 bg-amber-100 dark:bg-amber-500/10 px-1.5 py-0.5 rounded border border-amber-300 dark:border-amber-500/20">
              RI WATCH
            </span>
          </div>

          <div className="space-y-2.5">
            {/* RI Risk */}
            <div className="space-y-1">
              <div className="text-[10px] font-mono font-bold text-slate-600 dark:text-slate-400 uppercase">RI Risk (24H)</div>
              <div className="flex items-center gap-2">
                <span className="px-2 py-0.5 rounded text-xs font-black font-mono bg-amber-100 dark:bg-amber-500/15 text-amber-900 dark:text-amber-300 border border-amber-300 dark:border-amber-500/30">
                  {cyclone.rapidIntensificationRisk}
                </span>
                <span className="text-[11px] font-mono text-slate-600 dark:text-slate-400 font-semibold">+15 kt / 24h Trend</span>
              </div>
            </div>

            {/* AI Architecture */}
            <div className="space-y-1 pt-1 border-t border-slate-100 dark:border-slate-800/80">
              <div className="text-[10px] font-mono font-bold text-slate-600 dark:text-slate-400 uppercase">AI Architecture</div>
              <div className="text-sm font-black font-mono text-slate-950 dark:text-white leading-tight">ConvLSTM-B0</div>
              <div className="text-[10px] font-mono text-cyan-800 dark:text-cyan-400 font-bold">EfficientNet-B0</div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer Info */}
      <div className="pt-2.5 border-t border-slate-300 dark:border-slate-800 flex flex-wrap items-center justify-between gap-2 text-[10px] font-mono text-slate-600 dark:text-slate-400 font-medium">
        <span className="leading-snug">
          Inference Engine: <strong className="text-slate-900 dark:text-slate-300 font-bold">{cyclone.aiModelName}</strong>
        </span>
        <span className="leading-snug font-semibold">Telemetry Synchronized: {cyclone.lastUpdated}</span>
      </div>
    </div>
  );
};

