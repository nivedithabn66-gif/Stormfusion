import React, { useState } from 'react';
import { 
  Cpu, 
  CheckCircle2, 
  Sliders, 
  Layers, 
  ShieldCheck, 
  Activity, 
  TrendingUp, 
  BarChart3, 
  Target,
  Sparkles,
  AlertTriangle
} from 'lucide-react';
import { CycloneTelemetry, ForecastHourData } from '../types/cyclone';

interface AIModelConfidenceProps {
  cyclone: CycloneTelemetry;
  forecastHours: ForecastHourData[];
}

export const AIModelConfidence: React.FC<AIModelConfidenceProps> = ({ cyclone, forecastHours }) => {
  const [activeTab, setActiveTab] = useState<'metrics' | 'horizon' | 'architecture'>('metrics');

  return (
    <div className="command-card rounded-2xl p-5 sm:p-6 border border-slate-300 dark:border-slate-800 transition-colors space-y-5">
      {/* Section Title and Tab Controls */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-300 dark:border-slate-800">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-cyan-500/15 border border-cyan-500/30 text-cyan-700 dark:text-cyan-400">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-lg font-black font-mono tracking-wide text-slate-950 dark:text-white uppercase">
                AI Model Confidence & Neural Validation
              </h2>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-cyan-100 dark:bg-cyan-500/15 text-cyan-800 dark:text-cyan-300 border border-cyan-400/80 dark:border-cyan-500/30">
                CALIBRATED
              </span>
            </div>
            <p className="text-xs text-slate-600 dark:text-slate-400 font-mono mt-0.5 font-medium">
              Multimodal Uncertainty Estimation, Horizon Calibration & Feature Attribution Health
            </p>
          </div>
        </div>

        {/* Tab Navigation Switcher */}
        <div className="flex items-center flex-wrap bg-slate-200/80 dark:bg-[#070B16] border border-slate-300 dark:border-cyan-500/20 rounded-xl p-1 text-xs font-mono shadow-xs gap-1">
          <button
            onClick={() => setActiveTab('metrics')}
            className={`px-3 py-1.5 rounded-lg transition-all duration-200 flex items-center gap-1.5 cursor-pointer ${
              activeTab === 'metrics'
                ? 'bg-gradient-to-r from-cyan-700 to-blue-700 text-white dark:from-cyan-950 dark:to-slate-900 dark:text-cyan-300 border border-cyan-500 dark:border-cyan-400/60 shadow-md dark:shadow-[0_0_12px_rgba(6,182,212,0.3)] ring-1 ring-cyan-400/30 font-bold'
                : 'text-slate-700 dark:text-slate-300 hover:text-slate-950 dark:hover:text-white hover:bg-white/80 dark:hover:bg-slate-800/80 font-semibold'
            }`}
          >
            <Activity className={`w-3.5 h-3.5 ${activeTab === 'metrics' ? 'text-cyan-200' : 'text-slate-500 dark:text-slate-400'}`} />
            <span>Confidence Matrix</span>
          </button>
          <button
            onClick={() => setActiveTab('horizon')}
            className={`px-3 py-1.5 rounded-lg transition-all duration-200 flex items-center gap-1.5 cursor-pointer ${
              activeTab === 'horizon'
                ? 'bg-gradient-to-r from-cyan-700 to-blue-700 text-white dark:from-cyan-950 dark:to-slate-900 dark:text-cyan-300 border border-cyan-500 dark:border-cyan-400/60 shadow-md dark:shadow-[0_0_12px_rgba(6,182,212,0.3)] ring-1 ring-cyan-400/30 font-bold'
                : 'text-slate-700 dark:text-slate-300 hover:text-slate-950 dark:hover:text-white hover:bg-white/80 dark:hover:bg-slate-800/80 font-semibold'
            }`}
          >
            <TrendingUp className={`w-3.5 h-3.5 ${activeTab === 'horizon' ? 'text-cyan-200' : 'text-slate-500 dark:text-slate-400'}`} />
            <span>Horizon Calibration</span>
          </button>
          <button
            onClick={() => setActiveTab('architecture')}
            className={`px-3 py-1.5 rounded-lg transition-all duration-200 flex items-center gap-1.5 cursor-pointer ${
              activeTab === 'architecture'
                ? 'bg-gradient-to-r from-cyan-700 to-blue-700 text-white dark:from-cyan-950 dark:to-slate-900 dark:text-cyan-300 border border-cyan-500 dark:border-cyan-400/60 shadow-md dark:shadow-[0_0_12px_rgba(6,182,212,0.3)] ring-1 ring-cyan-400/30 font-bold'
                : 'text-slate-700 dark:text-slate-300 hover:text-slate-950 dark:hover:text-white hover:bg-white/80 dark:hover:bg-slate-800/80 font-semibold'
            }`}
          >
            <Cpu className={`w-3.5 h-3.5 ${activeTab === 'architecture' ? 'text-cyan-200' : 'text-slate-500 dark:text-slate-400'}`} />
            <span>Architecture Specs</span>
          </button>
        </div>
      </div>

      {/* Tab 1: Confidence Matrix */}
      {activeTab === 'metrics' && (
        <div className="space-y-4">
          {/* Key Metric Gauges */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
            {/* Detection Confidence */}
            <div className="p-4 rounded-xl bg-[#F8FAFC] dark:bg-slate-900/60 border border-slate-300 dark:border-slate-800 shadow-xs space-y-2">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-slate-600 dark:text-slate-400 font-bold uppercase">Pattern Detection</span>
                <span className="text-cyan-800 dark:text-cyan-400 font-bold">INSAT-3D IR</span>
              </div>
              <div className="text-2xl font-black font-mono text-cyan-800 dark:text-cyan-400">
                {cyclone.detectionConfidencePercent}%
              </div>
              <div className="w-full bg-slate-200 dark:bg-slate-800 h-2 rounded-full overflow-hidden">
                <div 
                  className="bg-cyan-600 dark:bg-cyan-400 h-full rounded-full transition-all duration-500" 
                  style={{ width: `${cyclone.detectionConfidencePercent}%` }}
                ></div>
              </div>
              <p className="text-[11px] font-mono text-slate-600 dark:text-slate-400 font-medium">
                Vortex boundary & curved-band spiral alignment certainty
              </p>
            </div>

            {/* Model Track Confidence */}
            <div className="p-4 rounded-xl bg-[#F8FAFC] dark:bg-slate-900/60 border border-slate-300 dark:border-slate-800 shadow-xs space-y-2">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-slate-600 dark:text-slate-400 font-bold uppercase">Track Trajectory</span>
                <span className="text-purple-800 dark:text-purple-400 font-bold">ConvLSTM</span>
              </div>
              <div className="text-2xl font-black font-mono text-purple-900 dark:text-purple-400">
                {cyclone.modelConfidencePercent}%
              </div>
              <div className="w-full bg-slate-200 dark:bg-slate-800 h-2 rounded-full overflow-hidden">
                <div 
                  className="bg-purple-600 dark:bg-purple-400 h-full rounded-full transition-all duration-500" 
                  style={{ width: `${cyclone.modelConfidencePercent}%` }}
                ></div>
              </div>
              <p className="text-[11px] font-mono text-slate-600 dark:text-slate-400 font-medium">
                72h ensemble steering field consensus score
              </p>
            </div>

            {/* RI Probability Risk */}
            <div className="p-4 rounded-xl bg-[#F8FAFC] dark:bg-slate-900/60 border border-slate-300 dark:border-slate-800 shadow-xs space-y-2">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-slate-600 dark:text-slate-400 font-bold uppercase">Rapid Intensification</span>
                <span className="text-amber-800 dark:text-amber-400 font-bold">24H Window</span>
              </div>
              <div className="text-2xl font-black font-mono text-amber-800 dark:text-amber-400">
                64% <span className="text-xs font-bold font-mono">({cyclone.rapidIntensificationRisk})</span>
              </div>
              <div className="w-full bg-slate-200 dark:bg-slate-800 h-2 rounded-full overflow-hidden">
                <div 
                  className="bg-amber-600 dark:bg-amber-400 h-full rounded-full transition-all duration-500" 
                  style={{ width: '64%' }}
                ></div>
              </div>
              <p className="text-[11px] font-mono text-slate-600 dark:text-slate-400 font-medium">
                SST: 30.8°C with low vertical wind shear (12 kt)
              </p>
            </div>

            {/* Neural Calibration Index */}
            <div className="p-4 rounded-xl bg-[#F8FAFC] dark:bg-slate-900/60 border border-slate-300 dark:border-slate-800 shadow-xs space-y-2">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-slate-600 dark:text-slate-400 font-bold uppercase">Expected Cal. Error</span>
                <span className="text-emerald-800 dark:text-emerald-400 font-bold">ECE Index</span>
              </div>
              <div className="text-2xl font-black font-mono text-emerald-800 dark:text-emerald-400">
                3.4% <span className="text-xs font-bold font-mono text-emerald-700">(Optimal)</span>
              </div>
              <div className="w-full bg-slate-200 dark:bg-slate-800 h-2 rounded-full overflow-hidden">
                <div 
                  className="bg-emerald-600 dark:bg-emerald-400 h-full rounded-full transition-all duration-500" 
                  style={{ width: '92%' }}
                ></div>
              </div>
              <p className="text-[11px] font-mono text-slate-600 dark:text-slate-400 font-medium">
                Brier score 0.082 vs. historical IMD ground truth
              </p>
            </div>
          </div>

          {/* Sub-Panel: Ensemble Spread & Diagnostics */}
          <div className="p-4 rounded-xl bg-[#F1F5F9] dark:bg-slate-900/40 border border-slate-300 dark:border-slate-800 font-mono text-xs flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-emerald-700 dark:text-emerald-400 shrink-0" />
              <span className="text-slate-800 dark:text-slate-300 font-medium">
                <strong>Ensemble Consistency:</strong> 50 perturbation members run with variance σ = 4.2 kt. No bifurcations detected in steering flow.
              </span>
            </div>
            <span className="px-2.5 py-1 rounded bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-200 font-bold border border-slate-300 dark:border-slate-700">
              Confidence Grade: A+ (High)
            </span>
          </div>
        </div>
      )}

      {/* Tab 2: Horizon Calibration */}
      {activeTab === 'horizon' && (
        <div className="space-y-4 font-mono text-xs">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3.5">
            {forecastHours.map((fc) => (
              <div 
                key={fc.hour}
                className="p-4 rounded-xl bg-[#F8FAFC] dark:bg-slate-900/60 border border-slate-300 dark:border-slate-800 space-y-3"
              >
                <div className="flex items-center justify-between pb-2 border-b border-slate-300 dark:border-slate-800">
                  <span className="font-bold text-slate-950 dark:text-white text-sm">T+{fc.hour}h Horizon</span>
                  <span className="px-2 py-0.5 rounded font-black text-cyan-800 dark:text-cyan-300 bg-cyan-100 dark:bg-cyan-500/20 border border-cyan-400/80 dark:border-cyan-500/30">
                    {fc.confidencePercent}% Conf.
                  </span>
                </div>

                <div className="space-y-1.5 text-slate-800 dark:text-slate-300">
                  <div className="flex justify-between">
                    <span className="text-slate-600 dark:text-slate-400 font-medium">Cross-Track Error:</span>
                    <span className="font-bold">±{fc.crossTrackErrorKm} km</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600 dark:text-slate-400 font-medium">Along-Track Error:</span>
                    <span className="font-bold">±{fc.alongTrackErrorKm} km</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600 dark:text-slate-400 font-medium">Projected Intensity:</span>
                    <span className="font-bold text-rose-700 dark:text-rose-400">{fc.windSpeedKts} kt</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600 dark:text-slate-400 font-medium">Expected Target:</span>
                    <span className="font-bold">{fc.targetTime}</span>
                  </div>
                </div>

                <div className="pt-2 border-t border-slate-300 dark:border-slate-800">
                  <div className="w-full bg-slate-200 dark:bg-slate-800 h-1.5 rounded-full overflow-hidden">
                    <div 
                      className="bg-cyan-600 dark:bg-cyan-400 h-full rounded-full" 
                      style={{ width: `${fc.confidencePercent}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="p-3.5 rounded-xl bg-[#F1F5F9] dark:bg-slate-900/40 border border-slate-300 dark:border-slate-800 text-slate-700 dark:text-slate-300">
            <strong>Horizon Uncertainty Law:</strong> Uncertainty envelope broadens monotonically with horizon distance (+24h ±24km → +72h ±95km) due to non-linear interaction with mid-tropospheric subtropical ridges over central India.
          </div>
        </div>
      )}

      {/* Tab 3: Architecture Specs */}
      {activeTab === 'architecture' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 font-mono text-xs">
          <div className="p-4 rounded-xl bg-[#F8FAFC] dark:bg-slate-900/60 border border-slate-300 dark:border-slate-800 space-y-2.5">
            <span className="font-bold text-slate-950 dark:text-white text-sm uppercase block pb-2 border-b border-slate-300 dark:border-slate-800">
              Neural Network Backbone
            </span>
            <div className="space-y-1.5 text-slate-800 dark:text-slate-300">
              <div className="flex justify-between py-1 border-b border-slate-200 dark:border-slate-850">
                <span className="text-slate-600 dark:text-slate-400 font-medium">Model Architecture:</span>
                <span className="font-bold">EfficientNet-B0 + ConvLSTM</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-200 dark:border-slate-850">
                <span className="text-slate-600 dark:text-slate-400 font-medium">Total Trainable Parameters:</span>
                <span className="font-bold">14.8M Params</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-200 dark:border-slate-850">
                <span className="text-slate-600 dark:text-slate-400 font-medium">Attention Mechanism:</span>
                <span className="font-bold text-purple-800 dark:text-purple-300">Spatiotemporal Multi-Head (h=8)</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-200 dark:border-slate-850">
                <span className="text-slate-600 dark:text-slate-400 font-medium">Spatial Resolution:</span>
                <span className="font-bold">4km (IR) / 1km (VIS)</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-600 dark:text-slate-400 font-medium">Inference Latency:</span>
                <span className="font-bold text-emerald-800 dark:text-emerald-400">142 ms (Edge GPU)</span>
              </div>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-[#F8FAFC] dark:bg-slate-900/60 border border-slate-300 dark:border-slate-800 space-y-2.5">
            <span className="font-bold text-slate-950 dark:text-white text-sm uppercase block pb-2 border-b border-slate-300 dark:border-slate-800">
              Training Data & Validation Baseline
            </span>
            <div className="space-y-1.5 text-slate-800 dark:text-slate-300">
              <div className="flex justify-between py-1 border-b border-slate-200 dark:border-slate-850">
                <span className="text-slate-600 dark:text-slate-400 font-medium">Satellite Ingestion:</span>
                <span className="font-bold">INSAT-3D/3DR (4 Channels)</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-200 dark:border-slate-850">
                <span className="text-slate-600 dark:text-slate-400 font-medium">Reanalysis Atmospheric Fields:</span>
                <span className="font-bold">ECMWF ERA5 (U/V 200/850 hPa)</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-200 dark:border-slate-850">
                <span className="text-slate-600 dark:text-slate-400 font-medium">Historical Ground Truth:</span>
                <span className="font-bold">IBTrACS (1880–2024 Records)</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-200 dark:border-slate-850">
                <span className="text-slate-600 dark:text-slate-400 font-medium">Loss Function:</span>
                <span className="font-bold text-cyan-800 dark:text-cyan-300">Pinball Quantile + Huber Loss</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-600 dark:text-slate-400 font-medium">Explainability:</span>
                <span className="font-bold text-purple-800 dark:text-purple-300">Grad-CAM Layer 4 Attributions</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
