import React, { useState } from 'react';
import { 
  BellRing, 
  AlertTriangle, 
  CheckCircle, 
  ShieldAlert, 
  Radio, 
  ExternalLink,
  Clock,
  MapPin,
  Check
} from 'lucide-react';
import { EmergencyAlert } from '../types/cyclone';

interface AlertPanelProps {
  alerts: EmergencyAlert[];
  onViewOnMap?: (alert: EmergencyAlert) => void;
}

export const AlertPanel: React.FC<AlertPanelProps> = ({ alerts, onViewOnMap }) => {
  const [alertList, setAlertList] = useState<EmergencyAlert[]>(alerts);
  const [selectedAlert, setSelectedAlert] = useState<EmergencyAlert | null>(null);

  const handleAcknowledge = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setAlertList(prev =>
      prev.map(a => (a.id === id ? { ...a, status: 'ACKNOWLEDGED' } : a))
    );
  };

  const getSeverityBadge = (severity: EmergencyAlert['severity']) => {
    switch (severity) {
      case 'CRITICAL':
        return 'bg-rose-600 text-white border-rose-500 animate-pulse';
      case 'WARNING':
        return 'bg-amber-500 text-slate-950 border-amber-400 font-bold';
      case 'WATCH':
        return 'bg-sky-500 text-white border-sky-400 font-bold';
      case 'ADVISORY':
      default:
        return 'bg-slate-600 text-white border-slate-500';
    }
  };

  return (
    <div className="command-card rounded-2xl p-5 border border-slate-200 dark:border-slate-800 flex flex-col justify-between transition-colors">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-200 dark:border-slate-800">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-lg bg-amber-500/15 border border-amber-500/30 text-amber-600 dark:text-amber-400">
            <BellRing className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-base font-bold font-mono tracking-wide text-slate-900 dark:text-white uppercase">
                Disaster Management Emergency Alerts
              </h3>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-rose-500/20 text-rose-700 dark:text-rose-300 border border-rose-500/40">
                {alertList.filter(a => a.status === 'ACTIVE').length} ACTIVE
              </span>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 font-mono">
              National & State Disaster Response Authority Emergency Bulletins
            </p>
          </div>
        </div>

        <span className="text-[10px] font-mono text-amber-700 dark:text-amber-400 bg-amber-500/10 px-2 py-1 rounded border border-amber-500/30 font-semibold">
          SIMULATED DISPATCH CONSOLE
        </span>
      </div>

      {/* Alert Feed */}
      <div className="mt-4 space-y-2.5 max-h-[360px] overflow-y-auto pr-1">
        {alertList.map((alert) => (
          <div
            key={alert.id}
            onClick={() => setSelectedAlert(alert)}
            className={`p-3.5 rounded-xl border transition-all cursor-pointer ${
              alert.status === 'ACKNOWLEDGED'
                ? 'bg-slate-100/60 dark:bg-slate-900/40 border-slate-200 dark:border-slate-800 opacity-75'
                : alert.severity === 'CRITICAL'
                ? 'bg-rose-50/70 dark:bg-rose-950/30 border-rose-300 dark:border-rose-500/40 hover:border-rose-500'
                : 'bg-white dark:bg-slate-900/80 border-slate-200 dark:border-slate-800 hover:border-amber-400 dark:hover:border-amber-500/40 shadow-xs'
            }`}
          >
            {/* Top row */}
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-extrabold tracking-wider border ${getSeverityBadge(alert.severity)}`}>
                  {alert.severity}
                </span>
                <span className="text-xs font-bold font-mono text-slate-900 dark:text-white">
                  {alert.headline}
                </span>
              </div>

              <div className="flex items-center gap-2 text-[10px] font-mono text-slate-500 dark:text-slate-400">
                <Clock className="w-3 h-3" />
                <span>{alert.timestamp}</span>
              </div>
            </div>

            {/* Description */}
            <p className="mt-2 text-xs text-slate-700 dark:text-slate-300 font-mono leading-relaxed line-clamp-2">
              {alert.description}
            </p>

            {/* Affected Areas & Action buttons */}
            <div className="mt-3 pt-2 border-t border-slate-200 dark:border-slate-800/80 flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
              <div className="flex items-center gap-1.5 text-[11px] text-slate-500 dark:text-slate-400">
                <MapPin className="w-3.5 h-3.5 text-cyan-600 dark:text-cyan-400" />
                <span className="truncate max-w-xs">{alert.affectedRegions.join(', ')}</span>
              </div>

              <div className="flex items-center gap-2">
                {alert.status === 'ACTIVE' ? (
                  <button
                    onClick={(e) => handleAcknowledge(alert.id, e)}
                    className="px-2.5 py-1 rounded bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white border border-slate-300 dark:border-slate-700 text-[10px] font-bold transition-colors flex items-center gap-1"
                  >
                    <Check className="w-3 h-3 text-emerald-600 dark:text-emerald-400" />
                    Acknowledge
                  </button>
                ) : (
                  <span className="text-[10px] text-emerald-600 dark:text-emerald-400 font-bold flex items-center gap-1">
                    <CheckCircle className="w-3 h-3" /> Acknowledged
                  </span>
                )}

                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedAlert(alert);
                  }}
                  className="px-2.5 py-1 rounded bg-cyan-500/15 hover:bg-cyan-500/25 border border-cyan-500/30 text-cyan-800 dark:text-cyan-300 text-[10px] font-bold transition-colors"
                >
                  Details
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Alert Detail Modal */}
      {selectedAlert && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-xs p-4">
          <div className="command-card max-w-lg w-full p-6 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-2xl relative bg-white dark:bg-[#0F172A]">
            <div className="flex items-center justify-between pb-3 border-b border-slate-200 dark:border-slate-800">
              <span className={`px-2 py-0.5 rounded text-xs font-mono font-extrabold ${getSeverityBadge(selectedAlert.severity)}`}>
                {selectedAlert.severity} ALERT
              </span>
              <span className="text-xs font-mono text-slate-500 dark:text-slate-400">{selectedAlert.timestamp}</span>
            </div>

            <div className="mt-4 space-y-3 font-mono text-xs">
              <h4 className="text-sm font-bold text-slate-900 dark:text-white">{selectedAlert.headline}</h4>
              <p className="text-slate-700 dark:text-slate-300 leading-relaxed bg-slate-50 dark:bg-slate-900/80 p-3 rounded-xl border border-slate-200 dark:border-slate-800">
                {selectedAlert.description}
              </p>

              <div>
                <span className="text-slate-500 block mb-1 text-[10px] uppercase font-semibold">Affected Administrative Sectors</span>
                <div className="flex flex-wrap gap-1.5">
                  {selectedAlert.affectedRegions.map((reg, i) => (
                    <span key={i} className="px-2 py-1 bg-slate-100 dark:bg-slate-800 text-slate-800 dark:text-slate-200 rounded border border-slate-200 dark:border-slate-700 text-[11px] shadow-xs">
                      {reg}
                    </span>
                  ))}
                </div>
              </div>

              <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-900 dark:text-amber-200">
                <span className="font-bold block text-[10px] uppercase text-amber-700 dark:text-amber-300 mb-1">Recommended Response Directive</span>
                <p className="text-[11px] leading-relaxed">{selectedAlert.recommendedAction}</p>
              </div>

              <div className="flex justify-between text-[10px] text-slate-500 pt-2 border-t border-slate-200 dark:border-slate-800">
                <span>Origin: {selectedAlert.source}</span>
                <span>Confidence: {selectedAlert.confidencePercent}%</span>
              </div>
            </div>

            <div className="mt-5 flex justify-end gap-2">
              <button
                onClick={() => setSelectedAlert(null)}
                className="px-4 py-2 rounded-lg bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-800 dark:text-white text-xs font-mono font-semibold border border-slate-300 dark:border-slate-700 transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
