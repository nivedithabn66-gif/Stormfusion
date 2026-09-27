import React, { useState } from 'react';
import { 
  History, 
  Search, 
  ExternalLink, 
  Database, 
  Activity, 
  AlertTriangle, 
  ArrowRight, 
  ShieldCheck 
} from 'lucide-react';
import { HistoricalAnalogue as HistoricalAnalogueType } from '../types/cyclone';

interface HistoricalAnalogueProps {
  analogues: HistoricalAnalogueType[];
}

export const HistoricalAnalogue: React.FC<HistoricalAnalogueProps> = ({ analogues }) => {
  const [selectedAnalogue, setSelectedAnalogue] = useState<HistoricalAnalogueType>(analogues[0]);

  return (
    <div className="command-card rounded-2xl p-5 border border-slate-300 dark:border-slate-800 flex flex-col justify-between transition-colors">
      {/* Top Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-300 dark:border-slate-800">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-lg bg-sky-500/15 border border-sky-500/30 text-sky-700 dark:text-sky-400">
            <History className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-base font-bold font-mono tracking-wide text-slate-950 dark:text-white uppercase">
                Historical Cyclone Analogue (FAISS Search)
              </h3>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-sky-500/15 text-sky-800 dark:text-sky-300 border border-sky-500/30">
                IBTrACS ARCHIVE
              </span>
            </div>
            <p className="text-xs text-slate-600 dark:text-slate-400 font-mono mt-0.5 font-medium">
              Nearest Neighbor Vector Matching on Track Geometries & Central Pressure Deficits
            </p>
          </div>
        </div>

        {/* Prominent Disclaimer Badge */}
        <div className="px-2.5 py-1 rounded bg-amber-500/15 border border-amber-500/40 text-amber-800 dark:text-amber-300 text-[11px] font-mono font-bold flex items-center gap-1.5">
          <AlertTriangle className="w-3.5 h-3.5 text-amber-600 dark:text-amber-400" />
          HISTORICAL ANALOGUE — NOT A FORECAST
        </div>
      </div>

      {/* Selected Analogue Showcase Card */}
      <div className="mt-4 p-4 rounded-xl bg-[#F8FAFC] dark:bg-slate-900/80 border border-slate-300 dark:border-slate-800 space-y-3 transition-colors">
        <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-slate-300 dark:border-slate-800">
          <div>
            <span className="text-[10px] font-mono text-slate-600 dark:text-slate-400 uppercase font-bold">Closest Historical Match</span>
            <div className="text-lg font-bold font-mono text-slate-950 dark:text-white flex items-center gap-2">
              <span>{selectedAnalogue.name} ({selectedAnalogue.year})</span>
              <span className="text-xs text-emerald-900 dark:text-emerald-400 bg-emerald-500/15 px-2 py-0.5 rounded font-mono font-bold border border-emerald-500/30">
                {selectedAnalogue.similarityScore}% Vector Match
              </span>
            </div>
          </div>

          <div className="text-right font-mono text-xs">
            <span className="text-slate-600 dark:text-slate-400 text-[10px] block font-bold">Landfall Region</span>
            <span className="text-sky-800 dark:text-sky-300 font-bold">{selectedAnalogue.landfallDistrict}</span>
          </div>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5 text-xs font-mono">
          <div className="p-2.5 rounded-lg bg-white dark:bg-slate-950 border border-slate-300 dark:border-slate-850 shadow-xs">
            <span className="text-slate-600 text-[10px] block font-bold">OVERALL SIMILARITY</span>
            <span className="text-base font-black text-cyan-800 dark:text-cyan-400 mt-0.5 block">
              {selectedAnalogue.similarityScore}%
            </span>
          </div>
          <div className="p-2.5 rounded-lg bg-white dark:bg-slate-950 border border-slate-300 dark:border-slate-850 shadow-xs">
            <span className="text-slate-600 text-[10px] block font-bold">TRACK GEOMETRY</span>
            <span className="text-base font-black text-sky-800 dark:text-sky-400 mt-0.5 block">
              {selectedAnalogue.trackSimilarity}%
            </span>
          </div>
          <div className="p-2.5 rounded-lg bg-white dark:bg-slate-950 border border-slate-300 dark:border-slate-850 col-span-2 sm:col-span-1 shadow-xs">
            <span className="text-slate-600 text-[10px] block font-bold">HISTORICAL PEAK WIND</span>
            <span className="text-base font-black text-rose-700 dark:text-rose-400 mt-0.5 block">
              {selectedAnalogue.peakIntensityKts} kt
            </span>
          </div>
        </div>

        {/* Summary Description */}
        <p className="text-xs text-slate-800 dark:text-slate-300 font-mono leading-relaxed bg-white dark:bg-slate-950/60 p-2.5 rounded-lg border border-slate-300 dark:border-slate-850 shadow-xs font-medium">
          {selectedAnalogue.summary}
        </p>
      </div>

      {/* Alternative Matches Picker */}
      <div className="mt-3 pt-3 border-t border-slate-300 dark:border-slate-800">
        <div className="text-[11px] font-mono text-slate-600 dark:text-slate-400 uppercase tracking-wider mb-2 font-bold">
          Other Closest Storm Analogues in IBTrACS Cluster:
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 font-mono text-xs">
          {analogues.map((item) => (
            <button
              key={item.id}
              onClick={() => setSelectedAnalogue(item)}
              className={`p-2 rounded-lg border text-left transition-all ${
                selectedAnalogue.id === item.id
                  ? 'bg-cyan-500/15 border-cyan-500/60 text-cyan-900 dark:text-cyan-200 font-bold shadow-xs'
                  : 'bg-white dark:bg-slate-900/60 border-slate-300 dark:border-slate-800 text-slate-800 dark:text-slate-400 hover:text-slate-950 dark:hover:text-slate-200 hover:border-slate-400 dark:hover:border-slate-700'
              }`}
            >
              <div className="font-bold truncate text-slate-950 dark:text-white">{item.name} ({item.year})</div>
              <div className="text-[10px] text-slate-600 dark:text-slate-400 mt-0.5 flex justify-between font-medium">
                <span>{item.similarityScore}% Match</span>
                <span className="text-rose-700 dark:text-rose-400 font-bold">{item.peakIntensityKts}kt</span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Footer Info */}
      <div className="mt-3 pt-2 text-[10px] font-mono text-slate-600 dark:text-slate-400 flex items-center justify-between font-medium">
        <span className="flex items-center gap-1">
          <Database className="w-3 h-3 text-cyan-700 dark:text-cyan-400" />
          Powered by FAISS Vector Index (L2 Track Distance + Intensity Cosine Similarity)
        </span>
        <span className="font-semibold">1880–2025 Archive</span>
      </div>
    </div>
  );
};
