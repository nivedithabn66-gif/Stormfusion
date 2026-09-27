import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Radio, 
  TrendingUp, 
  Map, 
  History, 
  BellRing, 
  Layers, 
  Cpu, 
  ChevronRight, 
  ShieldAlert
} from 'lucide-react';

interface SidebarProps {
  isOpen?: boolean;
  onClose?: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ isOpen = false, onClose }) => {
  const links = [
    { name: 'Dashboard', icon: LayoutDashboard, path: '/', hint: 'Command Center' },
    { name: 'Live Monitoring', icon: Radio, path: '/live', hint: 'INSAT-3D & XAI Grad-CAM' },
    { name: 'Track & Forecast', icon: TrendingUp, path: '/forecast', hint: '24/48/72h Trajectory' },
    { name: 'Risk GIS Map', icon: Map, path: '/risk', hint: 'District Vulnerability' },
    { name: 'Historical Analogue', icon: History, path: '/history', hint: 'FAISS Search' },
    { name: 'Alert Center', icon: BellRing, path: '/alerts', hint: 'Alerts & Ground Reports' },
  ];

  return (
    <aside
      className={`fixed inset-y-0 left-0 z-30 w-64 bg-[#F1F5F9]/95 dark:bg-[#0A0E19] border-r border-slate-300 dark:border-slate-800 transform transition-transform duration-300 ease-in-out lg:translate-x-0 ${
        isOpen ? 'translate-x-0' : '-translate-x-full'
      } pt-20 pb-6 flex flex-col justify-between backdrop-blur-md transition-colors duration-200`}
    >
      {/* Navigation Group */}
      <div className="px-3 space-y-1">
        <div className="px-3 py-2 text-[10px] font-mono tracking-widest text-slate-600 dark:text-slate-500 uppercase font-bold">
          Operations Core
        </div>
        {links.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              onClick={onClose}
              className={({ isActive }) =>
                `flex items-center justify-between px-3 py-2.5 rounded-lg text-xs font-medium transition-all group ${
                  isActive
                    ? 'bg-white dark:bg-cyan-500/15 text-cyan-900 dark:text-cyan-300 border border-cyan-400/80 dark:border-cyan-500/40 shadow-xs font-bold ring-1 ring-cyan-500/20 dark:ring-transparent'
                    : 'text-slate-800 dark:text-slate-400 hover:text-slate-950 dark:hover:text-white hover:bg-slate-200/70 dark:hover:bg-slate-800/60 font-semibold'
                }`
              }
            >
              <div className="flex items-center gap-3">
                <Icon className="w-4 h-4 transition-colors group-hover:text-cyan-700 dark:group-hover:text-cyan-400 text-slate-700 dark:text-slate-400" />
                <div>
                  <div className="leading-none">{item.name}</div>
                  <div className="text-[10px] text-slate-600 dark:text-slate-500 font-mono mt-0.5 font-medium">{item.hint}</div>
                </div>
              </div>
              <ChevronRight className="w-3.5 h-3.5 text-cyan-600 dark:text-slate-600 opacity-0 group-hover:opacity-100 transition-opacity" />
            </NavLink>
          );
        })}
      </div>

      {/* Model & System Health Widget */}
      <div className="px-4">
        <div className="p-3 rounded-xl bg-white dark:bg-slate-900/90 border border-slate-300 dark:border-slate-800 space-y-2.5 text-xs font-mono shadow-sm transition-colors">
          <div className="flex items-center justify-between text-slate-800 dark:text-slate-300">
            <span className="flex items-center gap-1.5 text-[11px] text-cyan-800 dark:text-cyan-400 font-bold">
              <Cpu className="w-3.5 h-3.5" />
              AI Inference Engine
            </span>
            <span className="text-[10px] text-emerald-800 dark:text-emerald-400 bg-emerald-100 dark:bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-300 dark:border-emerald-500/20 font-bold">
              READY
            </span>
          </div>

          <div className="space-y-1 text-[11px] text-slate-700 dark:text-slate-400">
            <div className="flex justify-between">
              <span className="text-slate-600 dark:text-slate-400 font-medium">Backbone:</span>
              <span className="text-slate-950 dark:text-white font-bold">EfficientNet-B0</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-600 dark:text-slate-400 font-medium">Temporal:</span>
              <span className="text-slate-950 dark:text-white font-bold">ConvLSTM Attention</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-600 dark:text-slate-400 font-medium">Resolution:</span>
              <span className="text-slate-950 dark:text-white font-bold">0.05° Grid (ERA5)</span>
            </div>
          </div>

          <div className="pt-2 border-t border-slate-200 dark:border-slate-800 flex items-center justify-between text-[10px]">
            <span className="text-slate-600 dark:text-slate-500 font-medium">FastAPI Bridge:</span>
            <span className="text-cyan-800 dark:text-cyan-400 font-bold">Simulated / Active</span>
          </div>
        </div>
      </div>
    </aside>
  );
};
