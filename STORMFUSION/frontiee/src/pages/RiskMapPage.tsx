import React, { useState, useEffect } from 'react';
import { 
  Map, 
  Layers, 
  ShieldAlert, 
  Wind, 
  Waves, 
  CloudRain, 
  Users, 
  Filter,
  CheckCircle,
  AlertTriangle,
  Download
} from 'lucide-react';
import { CycloneTelemetry, DistrictRisk, TrackPoint } from '../types/cyclone';
import { cycloneApi } from '../services/api';
import { CycloneMap } from '../components/CycloneMap';
import { RiskPanel } from '../components/RiskPanel';
import { DemoModeBadge } from '../components/DemoModeBadge';
import { getRiskSeverityColor } from '../utils/formatters';

export const RiskMapPage: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [cyclone, setCyclone] = useState<CycloneTelemetry | null>(null);
  const [districts, setDistricts] = useState<DistrictRisk[]>([]);
  const [track, setTrack] = useState<TrackPoint[]>([]);
  const [cone, setCone] = useState<[number, number][]>([]);
  const [filterState, setFilterState] = useState<string>('ALL');

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const [c, d, t, f] = await Promise.all([
          cycloneApi.getCurrentCyclone(),
          cycloneApi.getRiskZones(),
          cycloneApi.getTrack(),
          cycloneApi.getForecast()
        ]);
        setCyclone(c);
        setDistricts(d);
        setTrack(t);
        setCone(f.uncertaintyCone);
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
        <Map className="w-8 h-8 animate-spin" />
      </div>
    );
  }

  const filteredDistricts = filterState === 'ALL'
    ? districts
    : districts.filter(d => d.state.includes(filterState));

  return (
    <div className="p-4 lg:p-6 space-y-6 max-w-[1700px] mx-auto font-mono">
      {/* Header */}
      <div className="command-card rounded-2xl p-5 border border-slate-200 dark:border-slate-800 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-rose-500/10 dark:bg-rose-500/15 border border-rose-500/30 text-rose-600 dark:text-rose-400">
              <Map className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-wide text-slate-900 dark:text-white uppercase">
                GIS Spatial Risk & Vulnerability Decision Support
              </h1>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                District-level exposure, coastal storm surge inundation zones, and evacuation status
              </p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <DemoModeBadge subtle />
          {/* State Filter tabs */}
          <div className="flex items-center gap-1 bg-slate-200/80 dark:bg-[#070B16] border border-slate-300 dark:border-cyan-500/20 rounded-xl p-1 shadow-xs">
            {['ALL', 'Andhra Pradesh', 'Odisha', 'West Bengal'].map((st) => (
              <button
                key={st}
                onClick={() => setFilterState(st)}
                className={`px-3 py-1.5 rounded-lg text-[11px] font-bold transition-all duration-200 flex items-center gap-1.5 ${
                  filterState === st
                    ? 'bg-gradient-to-r from-cyan-700 to-blue-700 text-white dark:from-cyan-950 dark:to-slate-900 dark:text-cyan-300 border border-cyan-500/80 dark:border-cyan-400/60 shadow-md dark:shadow-[0_0_12px_rgba(6,182,212,0.3)] ring-1 ring-cyan-400/30'
                    : 'text-slate-700 dark:text-slate-300 hover:text-slate-950 dark:hover:text-white hover:bg-white/80 dark:hover:bg-slate-800/80 border border-transparent'
                }`}
              >
                {st === 'ALL' ? 'All States' : st}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Full GIS Map */}
      <CycloneMap
        cyclone={cyclone}
        track={track}
        uncertaintyConeCoords={cone}
        districtRisks={filteredDistricts}
        heightClass="h-[580px]"
      />

      {/* GIS District Risk & Vulnerability Index Cards */}
      <RiskPanel districts={filteredDistricts} />

      {/* District Vulnerability Exposure Comprehensive Table */}
      <div className="command-card rounded-2xl p-5 border border-slate-200 dark:border-slate-800 overflow-hidden">
        <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-200 dark:border-slate-800">
          <div>
            <h3 className="text-base font-bold text-slate-900 dark:text-white uppercase tracking-wide">
              District Impact & Vulnerability Matrix
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Quantified vulnerability scores based on WRF-Chem atmospheric overlays & PostGIS Census exposure
            </p>
          </div>

          <div className="text-xs text-slate-500 dark:text-slate-400">
            Showing <strong>{filteredDistricts.length}</strong> monitored coastal sectors
          </div>
        </div>

        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-500 dark:text-slate-400 text-[11px] uppercase bg-slate-50/50 dark:bg-transparent">
                <th className="py-2.5 px-3">District & State</th>
                <th className="py-2.5 px-3">Risk Level</th>
                <th className="py-2.5 px-3">Max Wind Hazard</th>
                <th className="py-2.5 px-3">Storm Surge</th>
                <th className="py-2.5 px-3">24h Rainfall</th>
                <th className="py-2.5 px-3">Exposed Population</th>
                <th className="py-2.5 px-3">Landfall Dist.</th>
                <th className="py-2.5 px-3">Evacuation Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-850">
              {filteredDistricts.map((d, i) => {
                const colors = getRiskSeverityColor(d.riskLevel);
                return (
                  <tr key={i} className="hover:bg-slate-50 dark:hover:bg-slate-900/50 transition-colors">
                    <td className="py-3 px-3 font-bold text-slate-900 dark:text-white">
                      <div>{d.district}</div>
                      <div className="text-[10px] text-slate-500 dark:text-slate-400 font-normal">{d.state}</div>
                    </td>
                    <td className="py-3 px-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-extrabold ${colors.badgeBg}`}>
                        {d.riskLevel}
                      </span>
                    </td>
                    <td className="py-3 px-3 font-bold text-rose-600 dark:text-rose-400">
                      {d.windRiskKts} kt
                    </td>
                    <td className="py-3 px-3 font-bold text-sky-600 dark:text-sky-400">
                      {d.stormSurgeMeters} m
                    </td>
                    <td className="py-3 px-3 font-bold text-cyan-600 dark:text-cyan-400">
                      {d.estimatedRainfallMm} mm
                    </td>
                    <td className="py-3 px-3 text-slate-700 dark:text-slate-300">
                      {d.populationExposed}
                    </td>
                    <td className="py-3 px-3 text-slate-500 dark:text-slate-400">
                      {d.landfallProximityKm} km
                    </td>
                    <td className="py-3 px-3">
                      {d.evacuationRecommended ? (
                        <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-100 dark:bg-rose-500/20 text-rose-800 dark:text-rose-300 border border-rose-300 dark:border-rose-500/40">
                          MANDATORY EVACUATION
                        </span>
                      ) : (
                        <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400">
                          STANDBY WATCH
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
