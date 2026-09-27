import React, { useState } from 'react';
import { 
  Sparkles, 
  BrainCircuit, 
  Layers, 
  Sliders, 
  Info, 
  Cpu, 
  Eye, 
  CheckCircle2,
  AlertCircle
} from 'lucide-react';
import { GradCamAnalysis } from '../types/cyclone';
import { DemoModeBadge } from './DemoModeBadge';

interface GradCAMViewerProps {
  analysis: GradCamAnalysis;
}

export const GradCAMViewer: React.FC<GradCAMViewerProps> = ({ analysis }) => {
  const [heatmapOpacity, setHeatmapOpacity] = useState(70);
  const [viewMode, setViewMode] = useState<'overlay' | 'split'>('overlay');

  return (
    <div className="command-card rounded-2xl p-5 border border-purple-300 dark:border-purple-500/30 bg-purple-50/20 dark:bg-[#0C101E]/90 shadow-xl transition-colors">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-300 dark:border-slate-800">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-lg bg-purple-500/15 border border-purple-500/30 text-purple-700 dark:text-purple-400">
            <BrainCircuit className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-base font-bold font-mono tracking-wide text-slate-950 dark:text-white uppercase">
                Explainable AI (XAI) & Grad-CAM Attention
              </h3>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-purple-500/20 text-purple-900 dark:text-purple-300 border border-purple-500/40">
                GRAD-CAM v2
              </span>
            </div>
            <p className="text-xs text-purple-900 dark:text-purple-300/80 font-mono font-medium">
              Why did the model predict 82 kt intensity & NW recurvature?
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-amber-500/15 text-amber-800 dark:text-amber-300 border border-amber-500/30">
            DEMO VISUALIZATION
          </span>
        </div>
      </div>

      {/* Mode and Opacity Controls */}
      <div className="mt-3 flex flex-wrap items-center justify-between gap-3 text-xs font-mono">
        <div className="flex items-center bg-[#F1F5F9] dark:bg-slate-900 border border-slate-300 dark:border-slate-800 rounded-lg p-0.5 shadow-xs">
          <button
            onClick={() => setViewMode('overlay')}
            className={`px-3 py-1 rounded text-xs font-bold transition-colors ${
              viewMode === 'overlay' ? 'bg-purple-600 text-white shadow-xs' : 'text-slate-700 dark:text-slate-400 hover:text-slate-950 dark:hover:text-white'
            }`}
          >
            Overlay Mode
          </button>
          <button
            onClick={() => setViewMode('split')}
            className={`px-3 py-1 rounded text-xs font-bold transition-colors ${
              viewMode === 'split' ? 'bg-purple-600 text-white shadow-xs' : 'text-slate-700 dark:text-slate-400 hover:text-slate-950 dark:hover:text-white'
            }`}
          >
            Side-by-Side Split
          </button>
        </div>

        {viewMode === 'overlay' && (
          <div className="flex items-center gap-2 text-slate-800 dark:text-slate-300 font-semibold">
            <Sliders className="w-3.5 h-3.5 text-purple-700 dark:text-purple-400" />
            <span>Heatmap Opacity:</span>
            <input
              type="range"
              min="0"
              max="100"
              value={heatmapOpacity}
              onChange={(e) => setHeatmapOpacity(Number(e.target.value))}
              className="w-24 accent-purple-600 cursor-pointer"
            />
            <span className="w-8 text-right text-purple-900 dark:text-purple-300 font-bold">{heatmapOpacity}%</span>
          </div>
        )}
      </div>

      {/* Visualizer Area */}
      <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Visual Canvas */}
        <div className="relative rounded-xl overflow-hidden bg-slate-950 border border-slate-800 h-64 flex items-center justify-center shadow-inner">
          {/* Base Infrared Satellite Simulation */}
          <div className="absolute inset-0 bg-[#070D18] flex items-center justify-center">
            <svg className="w-48 h-48 opacity-70" viewBox="0 0 200 200">
              <path d="M100,100 Q140,50 170,90 T185,150" fill="none" stroke="#38BDF8" strokeWidth="10" strokeOpacity="0.4" />
              <path d="M100,100 Q60,150 30,110 T15,50" fill="none" stroke="#38BDF8" strokeWidth="12" strokeOpacity="0.4" />
              <circle cx="100" cy="100" r="14" fill="#04070D" stroke="#38BDF8" strokeWidth="2" />
            </svg>
          </div>

          {/* Grad-CAM Gradient Heatmap Overlay */}
          <div
            className="absolute inset-0 pointer-events-none flex items-center justify-center transition-opacity duration-200"
            style={{ opacity: heatmapOpacity / 100 }}
          >
            {/* Heatmap Gradient blobs simulating convolutional activation gradients */}
            <div className="relative w-48 h-48 flex items-center justify-center">
              <div className="absolute w-40 h-40 rounded-full bg-red-600/60 blur-xl"></div>
              <div className="absolute w-28 h-28 rounded-full bg-yellow-400/70 blur-lg"></div>
              <div className="absolute w-14 h-14 rounded-full bg-white/80 blur-md"></div>
              {/* Curved attention streak */}
              <svg className="w-full h-full" viewBox="0 0 200 200">
                <path d="M100,100 Q135,55 160,95" fill="none" stroke="#FF0055" strokeWidth="14" strokeLinecap="round" opacity="0.8" />
              </svg>
            </div>
          </div>

          {/* Canvas Tag */}
          <div className="absolute top-2 left-2 pointer-events-none">
            <span className="px-2 py-0.5 rounded bg-black/80 text-purple-300 font-mono text-[10px] border border-purple-500/30">
              Grad-CAM (ConvLSTM Layer 4)
            </span>
          </div>

          {/* Attention Core Pinpoint */}
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="w-10 h-10 rounded-full border border-yellow-300/80 animate-ping"></div>
          </div>
        </div>

        {/* Neural Feature Weight Distribution & Explanation */}
        <div className="flex flex-col justify-between space-y-3 font-mono text-xs">
          <div>
            <div className="text-slate-600 dark:text-slate-400 text-[11px] uppercase tracking-wide font-bold">
              Primary Attention Zone
            </div>
            <div className="text-sm font-bold text-slate-950 dark:text-white mt-0.5">
              {analysis.primaryAttentionZone}
            </div>
            <p className="mt-2 text-slate-800 dark:text-slate-300 text-[11px] leading-relaxed bg-white dark:bg-slate-900/60 p-2.5 rounded-lg border border-slate-300 dark:border-slate-800 shadow-xs font-medium">
              "{analysis.description}"
            </p>
          </div>

          {/* Salient Features Identified */}
          <div className="space-y-1.5 pt-1">
            <div className="text-slate-600 dark:text-slate-400 text-[11px] uppercase font-bold">
              Top Latent Salient Features
            </div>
            {analysis.featuresIdentified.map((feature, idx) => (
              <div key={idx} className="flex items-center gap-2 text-slate-800 dark:text-slate-300 text-[11px] font-medium">
                <CheckCircle2 className="w-3.5 h-3.5 text-purple-700 dark:text-purple-400 shrink-0" />
                <span>{feature}</span>
              </div>
            ))}
          </div>

          {/* Model Architecture Specs */}
          <div className="pt-2 border-t border-slate-300 dark:border-slate-800 flex items-center justify-between text-[10px] text-slate-600 dark:text-slate-400 font-medium">
            <span>Architecture: <strong className="text-slate-950 dark:text-slate-300">EfficientNet-B0 + ConvLSTM</strong></span>
            <span className="text-purple-900 dark:text-purple-300 font-bold">XAI Confidence: {analysis.confidenceScore}%</span>
          </div>
        </div>
      </div>
    </div>
  );
};
