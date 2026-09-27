import React, { useState, useEffect } from 'react';
import { 
  Radio, 
  MapPin, 
  MessageSquare, 
  AlertCircle, 
  ThumbsUp, 
  Send, 
  ShieldCheck, 
  HelpCircle, 
  Clock 
} from 'lucide-react';
import { GroundReport } from '../types/cyclone';

interface SocialReportsProps {
  reports: GroundReport[];
  onSubmitReport?: (text: string, location: string) => void;
}

import { cycloneApi } from '../services/api';

export const SocialReports: React.FC<SocialReportsProps> = ({ reports, onSubmitReport }) => {
  const [reportList, setReportList] = useState<GroundReport[]>(reports);
  const [newText, setNewText] = useState('');
  const [newLoc, setNewLoc] = useState('');
  const [showSubmitModal, setShowSubmitModal] = useState(false);

  useEffect(() => {
    setReportList(reports);
  }, [reports]);

  const handleUpvote = async (id: string) => {
    // Optimistic UI update
    setReportList(prev =>
      prev.map(r => (r.id === id ? { ...r, upvotes: r.upvotes + 1 } : r))
    );
    const updated = await cycloneApi.upvoteSocialReport(id);
    if (updated) {
      setReportList(prev => prev.map(r => (r.id === id ? updated : r)));
    }
  };

  const handlePostReport = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newText.trim() || !newLoc.trim()) return;

    const added = await cycloneApi.submitSocialReport({
      location: newLoc,
      state: "Coastal Zone",
      coordinates: [17.5, 83.2],
      content: newText,
      source: 'Crowdsourced App',
      status: 'INVESTIGATING',
    });

    setReportList(prev => [added, ...prev]);
    if (onSubmitReport) onSubmitReport(newText, newLoc);
    setNewText('');
    setNewLoc('');
    setShowSubmitModal(false);
  };

  return (
    <div className="command-card rounded-2xl p-5 border border-slate-200 dark:border-slate-800 flex flex-col justify-between transition-colors">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-200 dark:border-slate-800">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-600 dark:text-cyan-400">
            <Radio className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-base font-bold font-mono tracking-wide text-slate-900 dark:text-white uppercase">
                Ground Reports & Auxiliary Intelligence
              </h3>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 font-mono">
              Crowdsourced Social Sensor Network & Citizen Weather Reports
            </p>
          </div>
        </div>

        {/* Prominent Verification Notice */}
        <div className="px-2.5 py-1 rounded bg-amber-500/15 border border-amber-500/30 text-amber-700 dark:text-amber-300 text-[10px] font-mono font-bold flex items-center gap-1.5">
          <AlertCircle className="w-3.5 h-3.5 text-amber-500 dark:text-amber-400" />
          AUXILIARY / UNVERIFIED INFORMATION
        </div>
      </div>

      {/* Safety & Non-Official Disclaimer Alert */}
      <div className="mt-3 p-2.5 rounded-xl bg-slate-50 dark:bg-slate-900/90 border border-slate-200 dark:border-slate-800 flex items-start gap-2 text-[11px] font-mono text-slate-600 dark:text-slate-400">
        <HelpCircle className="w-4 h-4 text-cyan-600 dark:text-cyan-400 shrink-0 mt-0.5" />
        <p>
          <strong className="text-slate-800 dark:text-slate-300">Operational Notice:</strong> Ground reports are crowdsourced from social channels and radio monitors. They are strictly informational and do <strong>not</strong> alter automated ConvLSTM neural track forecasts or official evacuation protocols.
        </p>
      </div>

      {/* Feed List */}
      <div className="mt-3 space-y-2.5 max-h-[300px] overflow-y-auto pr-1">
        {reportList.map((item) => (
          <div
            key={item.id}
            className="p-3 rounded-xl bg-white dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800/80 hover:border-slate-300 dark:hover:border-slate-700 transition-colors space-y-2 text-xs font-mono shadow-xs"
          >
            {/* Top row */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5 font-bold text-slate-900 dark:text-white">
                <MapPin className="w-3.5 h-3.5 text-cyan-600 dark:text-cyan-400 shrink-0" />
                <span>{item.location}</span>
                <span className="text-slate-400 dark:text-slate-500 text-[10px] font-normal">({item.state})</span>
              </div>

              <div className="flex items-center gap-2">
                <span className="px-2 py-0.5 rounded text-[9px] font-bold bg-amber-500/15 text-amber-700 dark:text-amber-300 border border-amber-500/30">
                  {item.status}
                </span>
                <span className="text-[10px] text-slate-500 flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  {item.timestamp}
                </span>
              </div>
            </div>

            {/* Content */}
            <p className="text-slate-700 dark:text-slate-300 leading-relaxed text-[11px]">
              "{item.content}"
            </p>

            {/* Meta row */}
            <div className="pt-1.5 border-t border-slate-100 dark:border-slate-800/70 flex items-center justify-between text-[10px] text-slate-500 dark:text-slate-400">
              <span>Source: <strong className="text-slate-800 dark:text-slate-300 font-semibold">{item.source}</strong></span>

              <button
                onClick={() => handleUpvote(item.id)}
                className="flex items-center gap-1 px-2 py-0.5 rounded bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white border border-slate-200 dark:border-slate-700 transition-colors"
                title="Corroborate Report"
              >
                <ThumbsUp className="w-3 h-3 text-cyan-600 dark:text-cyan-400" />
                <span>{item.upvotes} Corroborations</span>
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Post Ground Report Action */}
      <div className="mt-3 pt-3 border-t border-slate-200 dark:border-slate-800 flex justify-between items-center">
        <span className="text-[10px] font-mono text-slate-500 dark:text-slate-400">
          Citizen Field Sensor Node (Crowdsource Gateway)
        </span>
        <button
          onClick={() => setShowSubmitModal(true)}
          className="px-3 py-1.5 rounded-lg bg-cyan-500/15 hover:bg-cyan-500/25 border border-cyan-500/40 text-cyan-800 dark:text-cyan-300 text-xs font-mono font-semibold transition-colors flex items-center gap-1.5"
        >
          <Send className="w-3 h-3" />
          Submit Field Observation
        </button>
      </div>

      {/* Submission Modal */}
      {showSubmitModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-xs p-4">
          <div className="command-card max-w-md w-full p-5 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-2xl bg-white dark:bg-[#0F172A]">
            <h4 className="text-sm font-bold font-mono text-slate-900 dark:text-white pb-2 border-b border-slate-200 dark:border-slate-800">
              Submit Citizen Weather Report
            </h4>
            <form onSubmit={handlePostReport} className="mt-3 space-y-3 font-mono text-xs">
              <div>
                <label className="block text-slate-600 dark:text-slate-400 text-[10px] uppercase mb-1 font-semibold">
                  Location / Coastal Town
                </label>
                <input
                  type="text"
                  placeholder="e.g. Visakhapatnam Harbor, Bheemunipatnam..."
                  value={newLoc}
                  onChange={(e) => setNewLoc(e.target.value)}
                  className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-300 dark:border-slate-800 rounded-lg px-3 py-2 text-slate-900 dark:text-white focus:outline-none focus:border-cyan-500 text-xs"
                  required
                />
              </div>

              <div>
                <label className="block text-slate-600 dark:text-slate-400 text-[10px] uppercase mb-1 font-semibold">
                  Field Observation / Damage Report
                </label>
                <textarea
                  rows={3}
                  placeholder="Describe sea conditions, wind gusts, rainfall intensity, or localized water ingress..."
                  value={newText}
                  onChange={(e) => setNewText(e.target.value)}
                  className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-300 dark:border-slate-800 rounded-lg px-3 py-2 text-slate-900 dark:text-white focus:outline-none focus:border-cyan-500 text-xs"
                  required
                />
              </div>

              <div className="p-2.5 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-900 dark:text-amber-200 text-[10px]">
                Notice: All submitted ground reports enter the auxiliary stream marked as <strong>UNVERIFIED</strong> pending community review.
              </div>

              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setShowSubmitModal(false)}
                  className="px-3 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-800 dark:text-white text-xs border border-slate-300 dark:border-slate-700 transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-3 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-bold transition-colors"
                >
                  Post Report
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
