import React, { useState, useRef, useEffect } from 'react';
import { Sun, Moon, Laptop, ChevronDown, Check } from 'lucide-react';
import { useTheme, ThemeMode } from '../context/ThemeContext';

interface ThemeToggleProps {
  className?: string;
  showLabel?: boolean;
}

export const ThemeToggle: React.FC<ThemeToggleProps> = ({ className = '', showLabel = false }) => {
  const { theme, resolvedTheme, setTheme, toggleTheme } = useTheme();
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const options: { mode: ThemeMode; label: string; icon: typeof Moon }[] = [
    { mode: 'dark', label: 'Dark Mode', icon: Moon },
    { mode: 'light', label: 'Light Mode', icon: Sun },
    { mode: 'system', label: 'System Theme', icon: Laptop },
  ];

  return (
    <div className={`relative inline-block text-left ${className}`} ref={dropdownRef}>
      {/* Main Trigger Button */}
      <div className="flex items-center rounded-lg bg-white hover:bg-slate-100 dark:bg-slate-900/90 dark:hover:bg-slate-800 border border-slate-300 dark:border-slate-800 transition-all shadow-xs">
        <button
          type="button"
          onClick={toggleTheme}
          className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-mono font-bold text-slate-800 dark:text-slate-300 hover:text-slate-950 dark:hover:text-white transition-colors"
          title={`Current theme: ${theme} (${resolvedTheme}). Click to toggle Dark/Light`}
          aria-label="Toggle theme"
        >
          {resolvedTheme === 'dark' ? (
            <Moon className="w-4 h-4 text-cyan-400 animate-in fade-in" />
          ) : (
            <Sun className="w-4 h-4 text-amber-500 animate-in fade-in" />
          )}
          {showLabel && (
            <span className="hidden sm:inline capitalize">
              {theme === 'system' ? 'System' : resolvedTheme === 'dark' ? 'Dark' : 'Light'}
            </span>
          )}
        </button>

        {/* Dropdown chevron to select specific mode (Dark, Light, System) */}
        <button
          type="button"
          onClick={() => setDropdownOpen(!dropdownOpen)}
          className="px-1.5 py-1.5 text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200 border-l border-slate-300 dark:border-slate-800 transition-colors"
          title="Theme selection options (Dark / Light / System)"
          aria-expanded={dropdownOpen}
          aria-label="Open theme options"
        >
          <ChevronDown className={`w-3.5 h-3.5 transition-transform duration-200 ${dropdownOpen ? 'rotate-180' : ''}`} />
        </button>
      </div>

      {/* Dropdown Menu */}
      {dropdownOpen && (
        <div className="absolute right-0 mt-1.5 w-44 rounded-xl bg-white dark:bg-[#0E1526] border border-slate-300 dark:border-slate-700/80 shadow-2xl py-1.5 z-50 text-xs font-mono backdrop-blur-md animate-in fade-in slide-in-from-top-2 duration-150">
          <div className="px-3 py-1 text-[10px] text-slate-600 dark:text-slate-500 uppercase tracking-wider font-bold border-b border-slate-200 dark:border-slate-800">
            Interface Theme
          </div>
          {options.map((opt) => {
            const Icon = opt.icon;
            const isSelected = theme === opt.mode;
            return (
              <button
                key={opt.mode}
                type="button"
                onClick={() => {
                  setTheme(opt.mode);
                  setDropdownOpen(false);
                }}
                className={`w-full flex items-center justify-between px-3 py-2 text-left transition-colors ${
                  isSelected
                    ? 'bg-cyan-100 dark:bg-cyan-950/30 text-cyan-950 dark:text-cyan-300 font-bold'
                    : 'text-slate-800 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800/70 font-semibold'
                }`}
              >
                <div className="flex items-center gap-2">
                  <Icon className={`w-3.5 h-3.5 ${
                    opt.mode === 'dark'
                      ? 'text-cyan-500'
                      : opt.mode === 'light'
                      ? 'text-amber-500'
                      : 'text-purple-500'
                  }`} />
                  <span>{opt.label}</span>
                </div>
                {isSelected && <Check className="w-3.5 h-3.5 text-cyan-700 dark:text-cyan-400" />}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
};
