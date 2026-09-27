import React from 'react';
import { 
  Info, 
  Cpu, 
  Layers, 
  Database, 
  ShieldAlert, 
  BrainCircuit, 
  GitBranch, 
  TrendingUp, 
  Radio, 
  Code2, 
  CheckCircle2,
  Sparkles
} from 'lucide-react';
import { DemoModeBadge } from '../components/DemoModeBadge';
import { CycloneLogo } from '../components/CycloneLogo';

export const AboutPage: React.FC = () => {
  return (
    <div className="p-4 lg:p-8 space-y-8 max-w-[1400px] mx-auto font-mono">
      {/* Header */}
      <div className="command-card rounded-2xl p-6 border border-slate-200 dark:border-slate-800 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-gradient-to-br from-cyan-500/15 via-slate-900/30 to-blue-600/15 border border-cyan-500/30 flex items-center justify-center shadow-[0_0_20px_rgba(6,182,212,0.2)]">
              <CycloneLogo size={36} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-2xl font-black tracking-wide text-slate-900 dark:text-white uppercase">
                  STORMFUSION <span className="text-cyan-600 dark:text-cyan-400 font-normal">| SYSTEM ARCHITECTURE</span>
                </h1>
              </div>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                Multi-Source Cyclone Intelligence & Early Warning Operational Architecture
              </p>
            </div>
          </div>
        </div>

        <DemoModeBadge subtle />
      </div>

      {/* Problem Statement & Solution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 text-xs">
        {/* Problem Statement */}
        <div className="command-card rounded-2xl p-6 border border-slate-200 dark:border-slate-800 space-y-3">
          <div className="flex items-center gap-2 text-rose-600 dark:text-rose-400 font-bold uppercase text-sm pb-2 border-b border-slate-200 dark:border-slate-800">
            <ShieldAlert className="w-5 h-5" />
            Operational Problem Statement
          </div>
          <p className="text-slate-700 dark:text-slate-300 leading-relaxed">
            "An Artificial Intelligence (AI) / Machine Learning (ML) based system for identification, classification, and prediction of different tropical cyclone patterns using multi-source satellite data."
          </p>
          <div className="space-y-1.5 text-slate-500 dark:text-slate-400 pt-2 border-t border-slate-100 dark:border-slate-850 text-[11px]">
            <div>• Severe tropical cyclones in the North Indian Ocean (Bay of Bengal & Arabian Sea) cause catastrophic storm surge inundation, gale wind destruction, and loss of life.</div>
            <div>• Conventional numerical weather models struggle with non-linear rapid intensification (RI) and localized eye feature dynamics in early genesis phases.</div>
          </div>
        </div>

        {/* Our Solution */}
        <div className="command-card rounded-2xl p-6 border border-slate-200 dark:border-slate-800 space-y-3">
          <div className="flex items-center gap-2 text-cyan-600 dark:text-cyan-400 font-bold uppercase text-sm pb-2 border-b border-slate-200 dark:border-slate-800">
            <Sparkles className="w-5 h-5" />
            The StormFusion Solution
          </div>
          <p className="text-slate-700 dark:text-slate-300 leading-relaxed">
            STORMFUSION pioneers a multi-modal spatio-temporal deep learning framework fusing geostationary satellite radiances (INSAT-3D/3DR), atmospheric soundings, ERA5 reanalysis fields, and historical tracks (IBTrACS) to output simultaneous track forecasts, intensity estimation, and explainable feature attributions.
          </p>
          <div className="space-y-1.5 text-slate-500 dark:text-slate-400 pt-2 border-t border-slate-100 dark:border-slate-850 text-[11px]">
            <div>• Dual-Stream Feature Extraction (Spatial CNN + Spatiotemporal ConvLSTM)</div>
            <div>• Multi-Horizon Track & Intensity Prediction (24h, 48h, 72h) with Uncertainty Cones</div>
            <div>• PostGIS Spatial Risk Mapping & Explainable AI (Grad-CAM) for human operators</div>
          </div>
        </div>
      </div>

      {/* End-to-End AI Pipeline Architecture Diagram */}
      <div className="command-card rounded-2xl p-6 border border-slate-200 dark:border-slate-800 space-y-6">
        <div className="flex items-center justify-between pb-3 border-b border-slate-200 dark:border-slate-800">
          <div className="flex items-center gap-2 font-bold text-slate-900 dark:text-white uppercase text-sm">
            <Cpu className="w-5 h-5 text-purple-600 dark:text-purple-400" />
            End-to-End Neural Architecture Flow
          </div>
          <span className="text-xs text-purple-700 dark:text-purple-300 font-mono bg-purple-500/10 px-2 py-0.5 rounded border border-purple-500/30">
            Multimodal Fusion Pipeline
          </span>
        </div>

        {/* Visual Pipeline Grid Flow */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-xs">
          {/* Step 1 */}
          <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 space-y-2 relative">
            <span className="w-6 h-6 rounded-full bg-cyan-500/10 dark:bg-cyan-500/20 text-cyan-700 dark:text-cyan-300 border border-cyan-500/40 font-bold flex items-center justify-center text-xs">
              1
            </span>
            <div className="font-bold text-slate-900 dark:text-white">Multi-Source Data Ingestion</div>
            <p className="text-slate-600 dark:text-slate-400 text-[11px]">
              • INSAT-3D / 3DR (IR, VIS, WV)<br />
              • Sea Surface Temperature (SST)<br />
              • ERA5 Environmental Wind Shear<br />
              • IBTrACS Historical Archives
            </p>
          </div>

          {/* Step 2 */}
          <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 space-y-2 relative">
            <span className="w-6 h-6 rounded-full bg-purple-500/10 dark:bg-purple-500/20 text-purple-700 dark:text-purple-300 border border-purple-500/40 font-bold flex items-center justify-center text-xs">
              2
            </span>
            <div className="font-bold text-slate-900 dark:text-white">Feature Extraction & Fusion</div>
            <p className="text-slate-600 dark:text-slate-400 text-[11px]">
              • Spatial: EfficientNet-B0 backbone<br />
              • Environmental: MLP Embedder<br />
              • Cross-Modal Concatenation & Channel Attention Layer
            </p>
          </div>

          {/* Step 3 */}
          <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 space-y-2 relative">
            <span className="w-6 h-6 rounded-full bg-sky-500/10 dark:bg-sky-500/20 text-sky-700 dark:text-sky-300 border border-sky-500/40 font-bold flex items-center justify-center text-xs">
              3
            </span>
            <div className="font-bold text-slate-900 dark:text-white">ConvLSTM Temporal Attention</div>
            <p className="text-slate-600 dark:text-slate-400 text-[11px]">
              • Spatiotemporal memory cells<br />
              • Temporal Attention mechanism weighting critical intensification pulses<br />
              • Latent Representation vector
            </p>
          </div>

          {/* Step 4 */}
          <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 space-y-2 relative">
            <span className="w-6 h-6 rounded-full bg-rose-500/10 dark:bg-rose-500/20 text-rose-700 dark:text-rose-300 border border-rose-500/40 font-bold flex items-center justify-center text-xs">
              4
            </span>
            <div className="font-bold text-slate-900 dark:text-white">Multi-Task Prediction Heads</div>
            <p className="text-slate-600 dark:text-slate-400 text-[11px]">
              • Classification Head (IMD scale)<br />
              • Intensity Head (Wind kts & Pressure)<br />
              • Track Prediction (24/48/72h Lat/Lng)<br />
              • Uncertainty Variance Estimation
            </p>
          </div>
        </div>

        {/* Downstream Operations Layer */}
        <div className="p-4 rounded-xl bg-slate-100 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 flex flex-wrap items-center justify-between gap-4 text-xs">
          <div className="flex items-center gap-3">
            <Layers className="w-5 h-5 text-cyan-600 dark:text-cyan-400" />
            <div>
              <div className="font-bold text-slate-900 dark:text-white">Downstream Operations & Decision-Support Stack</div>
              <div className="text-[11px] text-slate-500 dark:text-slate-400">
                Grad-CAM Heatmaps • FAISS Historical Analogue • PostGIS Risk Zones • Early Warning Broadcasts
              </div>
            </div>
          </div>
          <span className="text-xs text-emerald-700 dark:text-emerald-400 bg-emerald-500/10 px-3 py-1 rounded-lg border border-emerald-500/30">
            HUMAN-IN-THE-LOOP DESIGN
          </span>
        </div>
      </div>

      {/* Technology Stack & Decoupled Architecture */}
      <div className="command-card rounded-2xl p-6 border border-slate-200 dark:border-slate-800 space-y-4">
        <div className="flex items-center gap-2 font-bold text-slate-900 dark:text-white uppercase text-sm pb-2 border-b border-slate-200 dark:border-slate-800">
          <Code2 className="w-5 h-5 text-cyan-600 dark:text-cyan-400" />
          Production-Ready Technology Stack
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 text-xs">
          <div className="p-3 bg-slate-50 dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-850 space-y-1">
            <strong className="text-cyan-600 dark:text-cyan-400 block">Frontend Web</strong>
            <p className="text-slate-700 dark:text-slate-300">React 18, TypeScript, Tailwind CSS, React Router</p>
          </div>

          <div className="p-3 bg-slate-50 dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-850 space-y-1">
            <strong className="text-purple-600 dark:text-purple-400 block">GIS & Mapping</strong>
            <p className="text-slate-700 dark:text-slate-300">Leaflet, React-Leaflet, CartoDB Light/Dark Matter, Esri Satellite</p>
          </div>

          <div className="p-3 bg-slate-50 dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-850 space-y-1">
            <strong className="text-sky-600 dark:text-sky-400 block">Data Visualization</strong>
            <p className="text-slate-700 dark:text-slate-300">Recharts (Area, Line, Cartesian Grids), Lucide React</p>
          </div>

          <div className="p-3 bg-slate-50 dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-850 space-y-1">
            <strong className="text-emerald-600 dark:text-emerald-400 block">Future Backend Target</strong>
            <p className="text-slate-700 dark:text-slate-300">Python FastAPI, PyTorch (TorchVision), PostgreSQL/PostGIS, FAISS</p>
          </div>
        </div>
      </div>
    </div>
  );
};
