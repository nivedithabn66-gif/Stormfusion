import React, { useState } from 'react';
import { 
  ShieldAlert, 
  MapPin, 
  Wind, 
  CloudRain, 
  Waves, 
  Users, 
  AlertTriangle, 
  ChevronRight, 
  Filter 
} from 'lucide-react';
import { DistrictRisk, RiskSeverity } from '../types/cyclone';
import { getRiskSeverityColor } from '../utils/formatters';

interface RiskPanelProps {
  districts: DistrictRisk[];
  onSelectDistrict?: (d: DistrictRisk) => void;
}

export const RiskPanel: React.FC<RiskPanelProps> = ({ districts, onSelectDistrict }) => {
  const [filterSeverity, setFilterSeverity] = useState<string>('ALL');

  const filteredDistricts = filterSeverity === 'ALL'
    ? districts
    : districts.filter(d => d.riskLevel === filterSeverity);

  return (
    <div className="command-card rounded-2xl p-5 border border-slate-300 dark:border-slate-800 flex flex-col justify-between transition-colors">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-300 dark:border-slate-800">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-lg bg-rose-500/15 border border-rose-500/30 text-rose-700 dark:text-rose-400">
            <ShieldAlert className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-base font-bold font-mono tracking-wide text-slate-950 dark:text-white uppercase">
                GIS District Risk & Vulnerability Index
              </h3>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-rose-500/15 text-rose-800 dark:text-rose-300 border border-rose-500/30">
                HIGH COMPOSITE THREAT
              </span>
            </div>
            <p className="text-xs text-slate-600 dark:text-slate-400 font-mono mt-0.5 font-medium">
              Coupled Wind Hazard, Storm Surge Inundation & Precipitation Modeling
            </p>
          </div>
        </div>

        {/* Severity Filter buttons */}
        <div className="flex items-center gap-1 bg-[#F1F5F9] dark:bg-slate-900 border border-slate-300 dark:border-slate-800 rounded-lg p-0.5 text-xs font-mono shadow-xs">
          {['ALL', 'CRITICAL', 'HIGH', 'MODERATE'].map((sev) => (
            <button
              key={sev}
              onClick={() => setFilterSeverity(sev)}
              className={`px-2.5 py-1 rounded text-[10px] font-bold transition-colors ${
                filterSeverity === sev
                  ? 'bg-rose-600 text-white dark:bg-rose-500/20 dark:text-rose-300 border border-rose-600 dark:border-rose-500/40 shadow-xs'
                  : 'text-slate-700 dark:text-slate-400 hover:text-slate-950 dark:hover:text-white'
              }`}
            >
              {sev}
            </button>
          ))}
        </div>
      </div>

      {/* District Exposure Table / Cards */}
      <div className="mt-4 space-y-2.5 max-h-[360px] overflow-y-auto pr-1">
        {filteredDistricts.map((d, idx) => {
          const colors = getRiskSeverityColor(d.riskLevel);
          return (
            <div
              key={idx}
              onClick={() => onSelectDistrict && onSelectDistrict(d)}
              className={`p-3.5 rounded-xl border transition-all cursor-pointer hover:border-cyan-500/60 ${colors.bg} ${colors.border}`}
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <MapPin className="w-4 h-4 text-cyan-700 dark:text-cyan-400 shrink-0" />
                  <div>
                    <span className="text-sm font-black font-mono text-slate-950 dark:text-white">{d.district}</span>
                    <span className="text-xs text-slate-600 dark:text-slate-400 font-mono ml-2 font-bold">({d.state})</span>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-extrabold tracking-wider ${colors.badgeBg}`}>
                    {d.riskLevel}
                  </span>
                  <span className="text-[11px] font-mono text-slate-600 dark:text-slate-400 font-semibold">
                    {d.landfallProximityKm} km to landfall
                  </span>
                </div>
              </div>

              {/* Factors Sub-Grid */}
              <div className="mt-2.5 grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono">
                <div className="p-1.5 rounded bg-white dark:bg-slate-900/60 border border-slate-300 dark:border-slate-800/80 flex items-center gap-2 shadow-xs">
                  <Wind className="w-3.5 h-3.5 text-rose-700 dark:text-rose-400 shrink-0" />
                  <div>
                    <div className="text-[9px] text-slate-600 dark:text-slate-500 font-bold">PEAK WIND</div>
                    <div className="font-black text-slate-950 dark:text-slate-200">{d.windRiskKts} kt</div>
                  </div>
                </div>

                <div className="p-1.5 rounded bg-white dark:bg-slate-900/60 border border-slate-300 dark:border-slate-800/80 flex items-center gap-2 shadow-xs">
                  <Waves className="w-3.5 h-3.5 text-sky-700 dark:text-sky-400 shrink-0" />
                  <div>
                    <div className="text-[9px] text-slate-600 dark:text-slate-500 font-bold">STORM SURGE</div>
                    <div className="font-black text-slate-950 dark:text-slate-200">{d.stormSurgeMeters} m</div>
                  </div>
                </div>

                <div className="p-1.5 rounded bg-white dark:bg-slate-900/60 border border-slate-300 dark:border-slate-800/80 flex items-center gap-2 shadow-xs">
                  <CloudRain className="w-3.5 h-3.5 text-cyan-700 dark:text-cyan-400 shrink-0" />
                  <div>
                    <div className="text-[9px] text-slate-600 dark:text-slate-500 font-bold">24H RAINFALL</div>
                    <div className="font-black text-slate-950 dark:text-slate-200">{d.estimatedRainfallMm} mm</div>
                  </div>
                </div>

                <div className="p-1.5 rounded bg-white dark:bg-slate-900/60 border border-slate-300 dark:border-slate-800/80 flex items-center gap-2 shadow-xs">
                  <Users className="w-3.5 h-3.5 text-amber-700 dark:text-amber-400 shrink-0" />
                  <div>
                    <div className="text-[9px] text-slate-600 dark:text-slate-500 font-bold">EXPOSURE</div>
                    <div className="font-black text-slate-950 dark:text-slate-200">{d.populationExposed}</div>
                  </div>
                </div>
              </div>

              {/* Critical Infrastructure Tags */}
              <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[10px] font-mono text-slate-600 dark:text-slate-400 font-bold">
                <span>Critical Assets:</span>
                {d.criticalInfrastructure.map((infra, i) => (
                  <span key={i} className="px-1.5 py-0.5 rounded bg-white dark:bg-slate-800/70 text-slate-800 dark:text-slate-300 border border-slate-300 dark:border-slate-700/60 shadow-xs font-semibold">
                    {infra}
                  </span>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {/* Footer Notice */}
      <div className="mt-3 pt-3 border-t border-slate-300 dark:border-slate-800 text-[10px] font-mono text-slate-600 dark:text-slate-400 flex items-center justify-between font-medium">
        <span>Risk Indices generated from WRF-Chem + PostGIS Spatial Overlays</span>
        <span className="text-rose-700 dark:text-rose-400 font-black">Phased Evacuation Active in 4 Districts</span>
      </div>
    </div>
  );
};
