import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { 
  Radio, 
  Clock, 
  MapPin, 
  Bell, 
  Menu, 
  Activity,
  Database,
  Compass,
  ShieldCheck
} from 'lucide-react';
import { DemoModeBadge, ScientificDisclaimerModal } from './DemoModeBadge';
import { ThemeToggle } from './ThemeToggle';
import { CycloneLogo } from './CycloneLogo';

interface NavbarProps {
  onToggleSidebar?: () => void;
  activeAlertCount?: number;
}

export const Navbar: React.FC<NavbarProps> = ({ onToggleSidebar, activeAlertCount = 4 }) => {
  const [showDisclaimer, setShowDisclaimer] = useState(false);
  const [currentTime, setCurrentTime] = useState('');

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      const istTime = now.toLocaleTimeString('en-IN', {
        timeZone: 'Asia/Kolkata',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false,
      });
      const istDate = now.toLocaleDateString('en-GB', {
        timeZone: 'Asia/Kolkata',
        day: '2-digit',
        month: 'short',
        year: 'numeric',
      });
      setCurrentTime(`${istDate} • ${istTime} IST`);
    };
    updateTime();
    const timer = setInterval(updateTime, 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <>
      <header className="sticky top-0 z-40 bg-white/95 dark:bg-[#0A0E1A]/95 border-b border-slate-300 dark:border-slate-800 backdrop-blur-md transition-colors duration-200">
        {/* Main Header Bar */}
        <div className="px-4 lg:px-6 h-16 flex items-center justify-between gap-4">
          {/* Left: Branding & Mobile Sidebar Toggle */}
          <div className="flex items-center gap-3 shrink-0">
            <button 
              onClick={onToggleSidebar}
              className="p-1.5 rounded-lg text-slate-700 dark:text-slate-400 hover:text-slate-950 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800 lg:hidden transition-colors cursor-pointer"
              aria-label="Toggle Sidebar Navigation"
              title="Open Navigation Menu"
            >
              <Menu className="w-5 h-5" />
            </button>

            <Link to="/" className="flex items-center gap-3 group">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500/15 via-blue-500/10 to-slate-200 dark:via-slate-900/40 border border-cyan-500/50 dark:border-cyan-500/40 flex items-center justify-center group-hover:border-cyan-600 dark:group-hover:border-cyan-400 transition-all shadow-[0_2px_8px_rgba(2,132,199,0.15)] dark:shadow-[0_0_20px_rgba(6,182,212,0.3)]">
                <CycloneLogo size={28} />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-extrabold text-base tracking-wider text-slate-950 dark:text-white font-mono">
                    STORM<span className="text-cyan-700 dark:text-cyan-400">FUSION</span>
                  </span>
                </div>
                <p className="text-[10px] text-slate-600 dark:text-slate-400 tracking-tight hidden sm:block font-semibold">
                  Multi-Source Cyclone Intelligence & Early Warning
                </p>
              </div>
            </Link>
          </div>

          {/* Center: Region / Location & Active Monitoring Indicators */}
          <div className="hidden md:flex items-center gap-3 flex-1 justify-center max-w-xl">
            {/* Region / Location Indicator */}
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white dark:bg-slate-900/80 border border-slate-300 dark:border-slate-800 text-xs font-mono text-slate-800 dark:text-slate-300 shadow-xs transition-colors">
              <Compass className="w-3.5 h-3.5 text-cyan-700 dark:text-cyan-400 shrink-0" />
              <div className="flex items-center gap-1.5">
                <span className="text-slate-500 dark:text-slate-500 text-[10px] uppercase font-bold">AOI:</span>
                <span className="font-bold text-slate-950 dark:text-slate-200">North Indian Ocean</span>
                <span className="text-slate-600 dark:text-slate-600 text-[10px] font-medium">(Bay of Bengal & Arabian Sea)</span>
              </div>
            </div>

            {/* Live Surveillance Status Indicator */}
            <div className="hidden xl:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-50 dark:bg-emerald-950/20 border border-emerald-300 dark:border-emerald-500/20 text-xs font-mono text-emerald-800 dark:text-emerald-400 transition-colors font-bold">
              <Activity className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-500 shrink-0 animate-pulse" />
              <span className="text-[11px] font-bold">REAL-TIME SURVEILLANCE</span>
            </div>
          </div>

          {/* Right: Date/Time, Alert Notifications & Theme Toggle */}
          <div className="flex items-center gap-2 sm:gap-2.5 shrink-0">
            <DemoModeBadge subtle />
            <button 
              onClick={() => setShowDisclaimer(true)}
              className="hidden lg:inline text-cyan-800 dark:text-cyan-400 hover:text-cyan-950 dark:hover:text-cyan-300 underline text-[10px] uppercase tracking-wider font-bold cursor-pointer"
            >
              Scientific Note
            </button>
            {/* Current Date / Time Indicator */}
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white dark:bg-slate-900/80 border border-slate-300 dark:border-slate-800 text-xs font-mono text-cyan-900 dark:text-cyan-300 shadow-xs transition-colors">
              <Clock className="w-3.5 h-3.5 text-cyan-700 dark:text-cyan-400 shrink-0" />
              <span className="font-bold tabular-nums">{currentTime || 'Loading clock...'}</span>
            </div>

            {/* Notification / Alert Center Icon */}
            <Link
              to="/alerts"
              className="relative p-2 rounded-lg bg-white dark:bg-slate-900/80 border border-slate-300 dark:border-slate-800 text-slate-800 dark:text-slate-300 hover:text-slate-950 dark:hover:text-white hover:border-slate-400 dark:hover:border-slate-700 transition-all shadow-xs group"
              title="Alert Center • Active Emergency Broadcasts"
              aria-label="View Emergency Alerts"
            >
              <Bell className="w-4 h-4 text-amber-600 dark:text-amber-400 group-hover:scale-110 transition-transform" />
              {activeAlertCount > 0 && (
                <span className="absolute -top-1 -right-1 w-4 h-4 bg-rose-600 text-white text-[9px] font-bold rounded-full flex items-center justify-center animate-pulse shadow-xs">
                  {activeAlertCount}
                </span>
              )}
            </Link>

            {/* Light / Dark Theme Toggle (top-right header) */}
            <ThemeToggle />
          </div>
        </div>
      </header>

      <ScientificDisclaimerModal
        isOpen={showDisclaimer}
        onClose={() => setShowDisclaimer(false)}
      />
    </>
  );
};

