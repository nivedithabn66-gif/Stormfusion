import React from 'react';
import { AlertTriangle, Info } from 'lucide-react';

interface DemoModeBadgeProps {
  className?: string;
  subtle?: boolean;
}

export const DemoModeBadge: React.FC<DemoModeBadgeProps> = ({ className = '', subtle = false }) => {
  if (subtle) {
    return (
      <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[11px] font-mono font-medium tracking-wider bg-amber-500/10 text-amber-700 dark:text-amber-300 border border-amber-500/30 ${className}`}>
        <span className="w-1.5 h-1.5 rounded-full bg-amber-500 dark:bg-amber-400 animate-pulse"></span>
        PROTOTYPE / SIMULATED
      </span>
    );
  }

  return (
    <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full bg-gradient-to-r from-amber-500/15 to-orange-500/15 dark:from-amber-500/20 dark:to-orange-500/20 border border-amber-500/40 text-amber-800 dark:text-amber-200 text-xs font-mono font-semibold shadow-xs backdrop-blur-md ${className}`}>
      <span className="relative flex h-2 w-2">
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
        <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
      </span>
      <span>DEMO MODE</span>
      <span className="text-amber-500/60 dark:text-amber-400/60">|</span>
      <span className="text-[11px] font-normal text-amber-700 dark:text-amber-300">SIMULATED DATA PROTOTYPE</span>
    </div>
  );
};

export const ScientificDisclaimerModal: React.FC<{ isOpen: boolean; onClose: () => void }> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 dark:bg-black/80 backdrop-blur-sm p-4">
      <div className="command-card max-w-lg w-full p-6 rounded-xl border border-slate-200 dark:border-cyan-500/30 shadow-2xl relative bg-white dark:bg-[#0A0E1A]">
        <div className="flex items-center gap-3 text-cyan-600 dark:text-cyan-400 mb-4 pb-3 border-b border-slate-200 dark:border-slate-800">
          <Info className="w-6 h-6 text-cyan-600 dark:text-cyan-400" />
          <h3 className="text-lg font-bold font-mono tracking-wide text-slate-900 dark:text-white">SCIENTIFIC & OPERATIONAL DISCLOSURE</h3>
        </div>
        <div className="space-y-3 text-sm text-slate-700 dark:text-slate-300 leading-relaxed">
          <p>
            <strong className="text-slate-900 dark:text-white">STORMFUSION</strong> is an advanced research and competition prototype engineered for 
            <span className="text-cyan-700 dark:text-cyan-300 font-semibold"> Multi-Source Early Warning Operations</span>.
          </p>
          <div className="p-3 bg-amber-50 dark:bg-amber-500/10 border border-amber-300 dark:border-amber-500/30 rounded-lg text-amber-900 dark:text-amber-200 text-xs flex items-start gap-2">
            <AlertTriangle className="w-5 h-5 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
            <p>
              Current tracks, intensities, Grad-CAM attention weights, and district damage indices are 
              <strong> realistic synthetic simulations</strong> designed to showcase end-to-end multi-modal AI architecture. 
              They must not be utilized for live public evacuation or life-critical maritime operations without official IMD / RSMC authorization.
            </p>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            For real-time operational cyclone advisories, refer exclusively to the India Meteorological Department (IMD) New Delhi and respective State Disaster Management Authorities (SDMA).
          </p>
        </div>
        <div className="mt-6 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg text-xs font-semibold uppercase tracking-wider transition-colors"
          >
            Acknowledge & Proceed
          </button>
        </div>
      </div>
    </div>
  );
};
