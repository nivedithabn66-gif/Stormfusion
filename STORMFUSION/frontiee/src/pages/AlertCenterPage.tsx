import React, { useState, useEffect } from 'react';
import { 
  BellRing, 
  AlertTriangle, 
  ShieldAlert, 
  CheckCircle, 
  Filter, 
  Radio, 
  Clock, 
  MapPin, 
  Check, 
  Eye,
  Send,
  MessageSquare
} from 'lucide-react';
import { EmergencyAlert, GroundReport } from '../types/cyclone';
import { cycloneApi } from '../services/api';
import { SocialReports } from '../components/SocialReports';
import { DemoModeBadge } from '../components/DemoModeBadge';

export const AlertCenterPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'alerts' | 'groundReports'>('alerts');
  const [alerts, setAlerts] = useState<EmergencyAlert[]>([]);
  const [socialReports, setSocialReports] = useState<GroundReport[]>([]);
  const [filterSeverity, setFilterSeverity] = useState<string>('ALL');
  const [filterStatus, setFilterStatus] = useState<string>('ALL');

  useEffect(() => {
    cycloneApi.getAlerts().then(setAlerts);
    cycloneApi.getSocialReports().then(setSocialReports);
  }, []);

  const handleAcknowledge = async (id: string) => {
    setAlerts(prev =>
      prev.map(a => (a.id === id ? { ...a, status: 'ACKNOWLEDGED' } : a))
    );
    await cycloneApi.acknowledgeAlert(id);
  };

  const filtered = alerts.filter(a => {
    const matchesSev = filterSeverity === 'ALL' || a.severity === filterSeverity;
    const matchesStatus = filterStatus === 'ALL' || a.status === filterStatus;
    return matchesSev && matchesStatus;
  });

  const getSeverityBadge = (severity: EmergencyAlert['severity']) => {
    switch (severity) {
      case 'CRITICAL':
        return 'bg-rose-500 text-slate-950 border-rose-400 animate-pulse';
      case 'WARNING':
        return 'bg-amber-500 text-slate-950 border-amber-400';
      case 'WATCH':
        return 'bg-sky-500 text-slate-950 border-sky-400';
      case 'ADVISORY':
      default:
        return 'bg-slate-700 text-slate-200 border-slate-600';
    }
  };

  return (
    <div className="p-4 lg:p-6 space-y-6 max-w-[1700px] mx-auto font-mono">
      {/* Header */}
      <div className="command-card rounded-2xl p-5 border border-slate-200 dark:border-slate-800 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-rose-500/10 dark:bg-rose-500/15 border border-rose-500/30 text-rose-600 dark:text-rose-400">
              <BellRing className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-wide text-slate-900 dark:text-white uppercase">
                Emergency Alert Management & Dispatch Center
              </h1>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Automated multi-channel hazard dissemination to State Disaster Authorities & Maritime Fleets
              </p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <DemoModeBadge subtle />
          <span className="text-xs text-rose-700 dark:text-rose-400 bg-rose-500/10 px-3 py-1.5 rounded-lg border border-rose-500/30 font-bold">
            {alerts.filter(a => a.status === 'ACTIVE').length} ACTIVE BULLETINS
          </span>
        </div>
      </div>

      {/* Section Navigation Tabs */}
      <div className="flex items-center gap-2 bg-slate-200/80 dark:bg-[#070B16] border border-slate-300 dark:border-cyan-500/20 rounded-xl p-1.5 shadow-xs w-full sm:w-fit">
        <button
          onClick={() => setActiveTab('alerts')}
          className={`px-4 py-2 rounded-lg text-xs font-bold transition-all duration-200 flex items-center gap-2 flex-1 sm:flex-initial ${
            activeTab === 'alerts'
              ? 'bg-gradient-to-r from-rose-600 to-rose-700 text-white dark:from-rose-950 dark:to-slate-900 dark:text-rose-300 border border-rose-500/80 dark:border-rose-500/60 shadow-md dark:shadow-[0_0_12px_rgba(244,63,94,0.3)] ring-1 ring-rose-400/30'
              : 'text-slate-700 dark:text-slate-300 hover:text-slate-950 dark:hover:text-white hover:bg-white/80 dark:hover:bg-slate-800/80 border border-transparent'
          }`}
        >
          <BellRing className="w-4 h-4" />
          <span>Alerts & Emergency Bulletins</span>
          <span className="px-1.5 py-0.5 rounded text-[10px] bg-rose-500/20 text-rose-200 border border-rose-500/30 font-extrabold">
            {alerts.filter(a => a.status === 'ACTIVE').length}
          </span>
        </button>

        <button
          onClick={() => setActiveTab('groundReports')}
          className={`px-4 py-2 rounded-lg text-xs font-bold transition-all duration-200 flex items-center gap-2 flex-1 sm:flex-initial ${
            activeTab === 'groundReports'
              ? 'bg-gradient-to-r from-cyan-700 to-blue-700 text-white dark:from-cyan-950 dark:to-slate-900 dark:text-cyan-300 border border-cyan-500/80 dark:border-cyan-400/60 shadow-md dark:shadow-[0_0_12px_rgba(6,182,212,0.3)] ring-1 ring-cyan-400/30'
              : 'text-slate-700 dark:text-slate-300 hover:text-slate-950 dark:hover:text-white hover:bg-white/80 dark:hover:bg-slate-800/80 border border-transparent'
          }`}
        >
          <Radio className="w-4 h-4" />
          <span>Ground Reports & Auxiliary Intelligence</span>
          <span className="px-1.5 py-0.5 rounded text-[10px] bg-cyan-500/20 text-cyan-200 border border-cyan-500/30 font-extrabold">
            {socialReports.length}
          </span>
        </button>
      </div>

      {activeTab === 'groundReports' ? (
        <SocialReports reports={socialReports} />
      ) : (
        <>
          {/* Filter Bar */}
          <div className="command-card rounded-2xl p-4 border border-slate-200 dark:border-slate-800/80 flex flex-wrap items-center justify-between gap-3 text-xs shadow-xs">
            {/* Severity Filter */}
            <div className="flex items-center gap-1.5 bg-slate-200/80 dark:bg-[#070B16] border border-slate-300 dark:border-rose-500/20 rounded-xl p-1.5 shadow-xs">
              <div className="flex items-center gap-1 text-slate-700 dark:text-slate-300 px-2 font-bold text-[11px]">
                <Filter className="w-3.5 h-3.5 text-rose-600 dark:text-rose-400" />
                <span>SEVERITY:</span>
              </div>
              {['ALL', 'CRITICAL', 'WARNING', 'WATCH', 'ADVISORY'].map(sev => (
                <button
                  key={sev}
                  onClick={() => setFilterSeverity(sev)}
                  className={`px-3 py-1.5 rounded-lg text-[11px] font-bold transition-all duration-200 flex items-center gap-1.5 ${
                    filterSeverity === sev
                      ? 'bg-gradient-to-r from-rose-600 to-rose-700 text-white dark:from-rose-950 dark:to-slate-900 dark:text-rose-300 border border-rose-500/80 dark:border-rose-500/60 shadow-md dark:shadow-[0_0_12px_rgba(244,63,94,0.3)] ring-1 ring-rose-400/30'
                      : 'text-slate-700 dark:text-slate-300 hover:text-slate-950 dark:hover:text-white hover:bg-white/80 dark:hover:bg-slate-800/80 border border-transparent'
                  }`}
                >
                  {sev}
                </button>
              ))}
            </div>

            {/* Status Filter */}
            <div className="flex items-center gap-1.5 bg-slate-200/80 dark:bg-[#070B16] border border-slate-300 dark:border-cyan-500/20 rounded-xl p-1.5 shadow-xs">
              <div className="flex items-center gap-1 text-slate-700 dark:text-slate-300 px-2 font-bold text-[11px]">
                <Radio className="w-3.5 h-3.5 text-cyan-600 dark:text-cyan-400" />
                <span>STATUS:</span>
              </div>
              {['ALL', 'ACTIVE', 'ACKNOWLEDGED', 'RESOLVED'].map(st => (
                <button
                  key={st}
                  onClick={() => setFilterStatus(st)}
                  className={`px-3 py-1.5 rounded-lg text-[11px] font-bold transition-all duration-200 flex items-center gap-1.5 ${
                    filterStatus === st
                      ? 'bg-gradient-to-r from-cyan-700 to-blue-700 text-white dark:from-cyan-950 dark:to-slate-900 dark:text-cyan-300 border border-cyan-500/80 dark:border-cyan-400/60 shadow-md dark:shadow-[0_0_12px_rgba(6,182,212,0.3)] ring-1 ring-cyan-400/30'
                      : 'text-slate-700 dark:text-slate-300 hover:text-slate-950 dark:hover:text-white hover:bg-white/80 dark:hover:bg-slate-800/80 border border-transparent'
                  }`}
                >
                  {st}
                </button>
              ))}
            </div>
          </div>

          {/* Alerts Feed */}
          <div className="grid grid-cols-1 gap-4">
        {filtered.map(alert => (
          <div
            key={alert.id}
            className={`command-card rounded-2xl p-5 border transition-all ${
              alert.status === 'ACKNOWLEDGED'
                ? 'bg-slate-50/70 dark:bg-slate-900/40 border-slate-200 dark:border-slate-800 opacity-80'
                : alert.severity === 'CRITICAL'
                ? 'bg-rose-50/50 dark:bg-rose-950/20 border-rose-300 dark:border-rose-500/50 shadow-md'
                : 'bg-white dark:bg-slate-900/80 border-slate-200 dark:border-slate-800'
            }`}
          >
            <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-200 dark:border-slate-800">
              <div className="flex items-center gap-2.5">
                <span className={`px-2.5 py-1 rounded text-xs font-extrabold border ${getSeverityBadge(alert.severity)}`}>
                  {alert.severity}
                </span>
                <h3 className="text-base font-bold text-slate-900 dark:text-white">
                  {alert.headline}
                </h3>
              </div>

              <div className="flex items-center gap-3 text-xs text-slate-500 dark:text-slate-400">
                <span className="flex items-center gap-1">
                  <Clock className="w-3.5 h-3.5" />
                  {alert.timestamp}
                </span>
                <span className="text-cyan-700 dark:text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/20">
                  {alert.confidencePercent}% Confidence
                </span>
              </div>
            </div>

            <p className="mt-3 text-xs text-slate-700 dark:text-slate-300 leading-relaxed">
              {alert.description}
            </p>

            {/* Affected Sectors */}
            <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
              <span className="text-slate-500 flex items-center gap-1 text-[11px]">
                <MapPin className="w-3.5 h-3.5 text-cyan-600 dark:text-cyan-400" />
                Target Zones:
              </span>
              {alert.affectedRegions.map((reg, i) => (
                <span key={i} className="px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-200 border border-slate-200 dark:border-slate-700 text-[11px]">
                  {reg}
                </span>
              ))}
            </div>

            {/* Recommended Protocol */}
            <div className="mt-3 p-3 rounded-xl bg-amber-50/60 dark:bg-slate-950 border border-amber-200 dark:border-slate-800 text-xs">
              <span className="block text-[10px] uppercase font-bold text-amber-800 dark:text-amber-400">
                Action Directive:
              </span>
              <p className="text-slate-800 dark:text-slate-300 mt-0.5">{alert.recommendedAction}</p>
            </div>

            {/* Action Bar */}
            <div className="mt-4 pt-3 border-t border-slate-200 dark:border-slate-800/80 flex flex-wrap items-center justify-between gap-3 text-xs">
              <span className="text-slate-500 text-[11px]">
                Source Authority: <strong className="text-slate-700 dark:text-slate-300">{alert.source}</strong>
              </span>

              <div className="flex items-center gap-2">
                {alert.status === 'ACTIVE' && (
                  <button
                    onClick={() => handleAcknowledge(alert.id)}
                    className="px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-bold transition-colors flex items-center gap-1.5 shadow-xs"
                  >
                    <Check className="w-4 h-4" />
                    Acknowledge Alert
                  </button>
                )}
                {alert.status === 'ACKNOWLEDGED' && (
                  <span className="text-emerald-600 dark:text-emerald-400 font-bold flex items-center gap-1 text-xs">
                    <CheckCircle className="w-4 h-4" /> Acknowledged by Watch Officer
                  </span>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
        </>
      )}
    </div>
  );
};
