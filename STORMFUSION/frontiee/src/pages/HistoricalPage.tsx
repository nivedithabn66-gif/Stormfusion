import React, { useState, useEffect } from 'react';
import { 
  History, 
  Search, 
  Sliders, 
  Database, 
  TrendingUp, 
  AlertTriangle, 
  CheckCircle,
  ArrowUpRight,
  Filter
} from 'lucide-react';
import { HistoricalAnalogue as HistoricalAnalogueType } from '../types/cyclone';
import { cycloneApi } from '../services/api';
import { DemoModeBadge } from '../components/DemoModeBadge';

export const HistoricalPage: React.FC = () => {
  const [analogues, setAnalogues] = useState<HistoricalAnalogueType[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedItem, setSelectedItem] = useState<HistoricalAnalogueType | null>(null);

  useEffect(() => {
    cycloneApi.getHistoricalAnalogues().then(res => {
      setAnalogues(res);
      setSelectedItem(res[0]);
    });
  }, []);

  const filtered = analogues.filter(a =>
    a.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    a.landfallDistrict.toLowerCase().includes(searchQuery.toLowerCase()) ||
    a.year.toString().includes(searchQuery)
  );

  return (
    <div className="p-4 lg:p-6 space-y-6 max-w-[1700px] mx-auto font-mono">
      {/* Header */}
      <div className="command-card rounded-2xl p-5 border border-slate-200 dark:border-slate-800 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-sky-500/10 dark:bg-sky-500/15 border border-sky-500/30 text-sky-600 dark:text-sky-400">
              <History className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-wide text-slate-900 dark:text-white uppercase">
                Historical Cyclone Analogue Engine
              </h1>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                FAISS High-Dimensional Vector Similarity Search across NOAA IBTrACS & IMD Best Track Archives
              </p>
            </div>
          </div>
        </div>

        {/* Disclaimer */}
        <div className="px-3 py-1.5 rounded-lg bg-amber-500/10 dark:bg-amber-500/15 border border-amber-500/30 text-amber-800 dark:text-amber-300 text-xs font-bold flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-amber-600 dark:text-amber-400 shrink-0" />
          HISTORICAL ANALOGUE — NOT A FORECAST
        </div>
      </div>

      {/* Main Content: List + Detail Showcase */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Side: Search & Filter List */}
        <div className="lg:col-span-5 space-y-4">
          <div className="command-card rounded-2xl p-4 border border-slate-200 dark:border-slate-800 space-y-3">
            <div className="relative">
              <Search className="w-4 h-4 text-slate-400 absolute left-3 top-3" />
              <input
                type="text"
                placeholder="Search historical storms (e.g. Hudhud, Fani, 2014)..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-800 rounded-xl pl-9 pr-3 py-2 text-xs text-slate-900 dark:text-white placeholder:text-slate-400 dark:placeholder:text-slate-500 focus:outline-none focus:border-cyan-500"
              />
            </div>

            <div className="text-[11px] text-slate-500 dark:text-slate-400 flex justify-between px-1">
              <span>FAISS Index: <strong>1,420 Track Vectors</strong></span>
              <span>Metric: <strong>Cosine + L2 Distance</strong></span>
            </div>
          </div>

          <div className="space-y-2.5">
            {filtered.map((storm) => (
              <div
                key={storm.id}
                onClick={() => setSelectedItem(storm)}
                className={`command-card rounded-xl p-4 border transition-all cursor-pointer ${
                  selectedItem?.id === storm.id
                    ? 'border-cyan-500 bg-cyan-500/10 shadow-[0_0_15px_rgba(6,182,212,0.15)] ring-1 ring-cyan-500/30'
                    : 'border-slate-200 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-bold text-slate-900 dark:text-white text-sm">
                    {storm.name} ({storm.year})
                  </span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-cyan-500/10 dark:bg-cyan-500/20 text-cyan-700 dark:text-cyan-300 border border-cyan-500/30">
                    {storm.similarityScore}% SIMILARITY
                  </span>
                </div>

                <div className="mt-2 grid grid-cols-2 gap-2 text-[11px] text-slate-500 dark:text-slate-400">
                  <div>Track Fit: <strong className="text-sky-600 dark:text-sky-400">{storm.trackSimilarity}%</strong></div>
                  <div>Peak Wind: <strong className="text-rose-600 dark:text-rose-400">{storm.peakIntensityKts} kt</strong></div>
                </div>

                <div className="mt-1 text-[11px] text-slate-500 dark:text-slate-400 truncate">
                  Landfall: <strong className="text-slate-800 dark:text-slate-300">{storm.landfallDistrict}</strong>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right Side: Detailed Analogue Inspection */}
        {selectedItem && (
          <div className="lg:col-span-7 command-card rounded-2xl p-6 border border-slate-200 dark:border-slate-800 space-y-5">
            <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-200 dark:border-slate-800">
              <div>
                <span className="text-[10px] text-slate-400 dark:text-slate-500 uppercase">Analogue Profile</span>
                <h2 className="text-xl font-bold text-slate-900 dark:text-white">
                  {selectedItem.name} — {selectedItem.year}
                </h2>
                <span className="text-xs text-slate-500 dark:text-slate-400">{selectedItem.basin}</span>
              </div>

              <div className="text-right">
                <span className="text-[10px] text-slate-400 dark:text-slate-500 uppercase block">FAISS Composite Score</span>
                <span className="text-2xl font-black text-cyan-600 dark:text-cyan-400">
                  {selectedItem.similarityScore}%
                </span>
              </div>
            </div>

            {/* Comparison Metrics */}
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs">
              <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-850">
                <span className="text-slate-500 text-[10px] block">TRACK RECURVATURE</span>
                <span className="text-base font-bold text-sky-600 dark:text-sky-400 mt-1 block">
                  {selectedItem.trackSimilarity}% Geometric Fit
                </span>
              </div>

              <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-850">
                <span className="text-slate-500 text-[10px] block">HISTORICAL PEAK WIND</span>
                <span className="text-base font-bold text-rose-600 dark:text-rose-400 mt-1 block">
                  {selectedItem.peakIntensityKts} kt
                </span>
              </div>

              <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-850 col-span-2 sm:col-span-1">
                <span className="text-slate-500 text-[10px] block">LANDFALL CORRIDOR</span>
                <span className="text-xs font-bold text-slate-900 dark:text-white mt-1 block truncate">
                  {selectedItem.landfallDistrict}
                </span>
              </div>
            </div>

            {/* Narrative Case Study */}
            <div className="space-y-2">
              <h4 className="text-xs font-bold text-slate-700 dark:text-slate-300 uppercase">
                Meteorological Analogue Synopsis & Impact Summary:
              </h4>
              <p className="text-xs text-slate-700 dark:text-slate-300 leading-relaxed bg-slate-50 dark:bg-slate-900/60 p-3.5 rounded-xl border border-slate-200 dark:border-slate-850">
                {selectedItem.summary}
              </p>
            </div>

            {/* Technical Vector Pipeline Notice */}
            <div className="p-3 rounded-xl bg-slate-100 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 space-y-1 text-[11px] text-slate-600 dark:text-slate-400">
              <div className="flex items-center gap-1.5 text-cyan-700 dark:text-cyan-400 font-bold">
                <Database className="w-3.5 h-3.5" />
                <span>FAISS Vector Search Pipeline Architecture</span>
              </div>
              <p>
                Cyclone trajectories are encoded into 64-dimensional latent representations combining 6-hourly coordinates, central pressure gradients, and SST trajectories. The FAISS (Facebook AI Similarity Search) index performs sub-millisecond nearest-neighbor lookups over 140+ years of North Indian Ocean cyclone records.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
