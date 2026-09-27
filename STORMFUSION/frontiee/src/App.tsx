import React, { useState } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Navbar } from './components/Navbar';
import { Sidebar } from './components/Sidebar';
import { CycloneAIChatbot } from './components/CycloneAIChatbot';
import { Dashboard } from './pages/Dashboard';
import { LiveMonitoring } from './pages/LiveMonitoring';
import { ForecastPage } from './pages/ForecastPage';
import { RiskMapPage } from './pages/RiskMapPage';
import { HistoricalPage } from './pages/HistoricalPage';
import { AlertCenterPage } from './pages/AlertCenterPage';
import { AssistantPage } from './pages/AssistantPage';
import { AboutPage } from './pages/AboutPage';

export const App: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="min-h-screen bg-[#EBF1F6] dark:bg-[#080C14] text-slate-900 dark:text-slate-100 flex flex-col transition-colors duration-200">
      {/* Top Navbar */}
      <Navbar onToggleSidebar={() => setSidebarOpen(!sidebarOpen)} />

      {/* Main App Layout */}
      <div className="flex-1 flex">
        {/* Collapsible Command Center Sidebar */}
        <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />

        {/* Backdrop for mobile drawer */}
        {sidebarOpen && (
          <div
            onClick={() => setSidebarOpen(false)}
            className="fixed inset-0 bg-black/60 backdrop-blur-xs z-20 lg:hidden"
          />
        )}

        {/* Main Content Area (offset on large screens for sidebar) */}
        <main className="flex-1 lg:pl-64 transition-all duration-300 w-full overflow-x-hidden">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/live" element={<LiveMonitoring />} />
            <Route path="/forecast" element={<ForecastPage />} />
            <Route path="/risk" element={<RiskMapPage />} />
            <Route path="/history" element={<HistoricalPage />} />
            <Route path="/alerts" element={<AlertCenterPage />} />
            <Route path="/assistant" element={<AssistantPage />} />
            <Route path="/about" element={<AboutPage />} />
          </Routes>
        </main>
      </div>

      {/* Floating AI Assistant Chatbot (Shared layout level across all operational pages) */}
      <CycloneAIChatbot isFloating={true} />

      {/* Persistent Military / Command Center Footer Bar */}
      <footer className="lg:pl-64 bg-[#E2E8F0] dark:bg-[#050811] border-t border-slate-300 dark:border-slate-850 px-4 py-3 text-[11px] font-mono text-slate-700 dark:text-slate-400 flex flex-wrap items-center justify-between gap-3 transition-colors duration-200">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1.5 text-cyan-700 dark:text-cyan-400 font-bold">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-600 animate-pulse"></span>
            STORMFUSION v1.0-PROTOTYPE
          </span>
          <span className="text-slate-400 dark:text-slate-700">|</span>
          <span className="font-medium text-slate-700 dark:text-slate-400">Multi-Source Satellite Intelligence & Operations Command Center</span>
        </div>

        <div className="flex items-center gap-4 text-[10px] text-slate-700 dark:text-slate-400 font-medium">
          <span>Telemetry: INSAT-3D / ERA5 / IBTrACS</span>
          <span>Inference: ConvLSTM Attention</span>
          <span className="text-amber-800 dark:text-amber-400 font-bold bg-amber-100 dark:bg-transparent px-1.5 py-0.5 rounded border border-amber-300 dark:border-transparent">DEMO MODE ACTIVE</span>
        </div>
      </footer>
    </div>
  );
};
