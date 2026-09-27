import React, { useState, useEffect } from 'react';
import { 
  Radio, 
  Eye, 
  Wind, 
  Activity, 
  Compass, 
  Thermometer, 
  Layers,
  Sparkles,
  RefreshCw,
  Gauge
} from 'lucide-react';
import { CycloneTelemetry, SatelliteFrame } from '../types/cyclone';
import { DemoModeBadge } from './DemoModeBadge';
import { mockCurrentCyclone } from '../data/mockCycloneData';

type SatelliteChannel = 'IR' | 'VIS' | 'THERMAL' | 'WV';

interface SatelliteViewerProps {
  cyclone?: CycloneTelemetry | null;
  frames?: SatelliteFrame[];
}

export const SatelliteViewer: React.FC<SatelliteViewerProps> = ({ 
  cyclone: propCyclone,
  frames 
}) => {
  const cyclone = propCyclone || mockCurrentCyclone;
  const [activeChannel, setActiveChannel] = useState<SatelliteChannel>('IR');
  const [showGrid, setShowGrid] = useState<boolean>(true);

  const channels: { id: SatelliteChannel; label: string; wavelength: string; desc: string }[] = [
    { 
      id: 'IR', 
      label: 'INFRARED (IR)', 
      wavelength: '10.8 µm', 
      desc: 'Cloud top brightness temperature and deep convective core structure.' 
    },
    { 
      id: 'VIS', 
      label: 'VISIBLE', 
      wavelength: '0.65 µm', 
      desc: 'High-resolution albedo optical reflectance and low-level feeder bands.' 
    },
    { 
      id: 'THERMAL', 
      label: 'THERMAL', 
      wavelength: '12.0 µm', 
      desc: 'Enhanced BD-curve thermal gradient mapping cold convective cloud tops.' 
    },
    { 
      id: 'WV', 
      label: 'WATER VAPOR', 
      wavelength: '6.8 µm', 
      desc: 'Mid-to-upper tropospheric moisture flux and synoptic atmospheric steering.' 
    }
  ];

  const currentChannelInfo = channels.find(c => c.id === activeChannel) || channels[0];

  return (
    <div className="command-card rounded-2xl p-4 sm:p-5 border border-slate-300 dark:border-slate-800 transition-colors space-y-4">
      {/* 1. Header Section */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-300 dark:border-slate-800">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-cyan-500/15 border border-cyan-500/30 text-cyan-700 dark:text-cyan-400">
            <Radio className="w-4 h-4 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <h2 className="text-sm sm:text-base font-black font-mono tracking-wide text-slate-950 dark:text-white uppercase">
                MULTI-CHANNEL SATELLITE VIEWER
              </h2>
              <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-800 dark:text-emerald-300 border border-emerald-500/30 text-[10px] font-mono font-bold">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-600 animate-pulse"></span>
                CURRENT
              </div>
            </div>
            <p className="text-xs text-slate-600 dark:text-slate-400 font-mono mt-0.5 font-medium">
              INSAT-3D Multispectral Radiometer • Geostationary 74.0°E (Spatial Res: 4km IR / 1km VIS)
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <DemoModeBadge subtle />
        </div>
      </div>

      {/* 2. Horizontal Channel Tabs & Grid Control */}
      <div className="flex flex-wrap items-center justify-between gap-2.5">
        <div className="flex items-center flex-wrap bg-slate-200/80 dark:bg-[#070B16] border border-slate-300 dark:border-cyan-500/20 rounded-xl p-1 text-xs font-mono shadow-xs gap-1">
          {channels.map((ch) => {
            const isActive = activeChannel === ch.id;
            return (
              <button
                key={ch.id}
                onClick={() => setActiveChannel(ch.id)}
                className={`px-3 py-1.5 rounded-lg transition-all duration-200 font-bold text-[11px] sm:text-xs flex items-center gap-1.5 cursor-pointer ${
                  isActive
                    ? 'bg-gradient-to-r from-cyan-700 to-blue-700 text-white dark:from-cyan-950 dark:to-slate-900 dark:text-cyan-300 border border-cyan-500 dark:border-cyan-400/60 shadow-md dark:shadow-[0_0_12px_rgba(6,182,212,0.3)] ring-1 ring-cyan-400/30'
                    : 'text-slate-700 dark:text-slate-300 hover:text-slate-950 dark:hover:text-white hover:bg-white/80 dark:hover:bg-slate-800/80'
                }`}
              >
                {ch.id === 'IR' && <Eye className={`w-3.5 h-3.5 ${isActive ? 'text-cyan-200' : 'text-slate-500 dark:text-slate-400'}`} />}
                {ch.id === 'VIS' && <Sparkles className={`w-3.5 h-3.5 ${isActive ? 'text-cyan-200' : 'text-slate-500 dark:text-slate-400'}`} />}
                {ch.id === 'THERMAL' && <Thermometer className={`w-3.5 h-3.5 ${isActive ? 'text-cyan-200' : 'text-slate-500 dark:text-slate-400'}`} />}
                {ch.id === 'WV' && <Activity className={`w-3.5 h-3.5 ${isActive ? 'text-cyan-200' : 'text-slate-500 dark:text-slate-400'}`} />}
                <span>{ch.label}</span>
                <span className={`text-[9px] font-normal ${isActive ? 'opacity-90 text-cyan-100 dark:text-cyan-200' : 'opacity-70 text-slate-500 dark:text-slate-400'}`}>
                  ({ch.wavelength})
                </span>
              </button>
            );
          })}
        </div>

        {/* Lat/Lng Grid Toggle */}
        <div className="flex items-center gap-2 text-xs font-mono">
          <button
            onClick={() => setShowGrid(!showGrid)}
            className={`px-3 py-1.5 rounded-xl border transition-all duration-200 text-[11px] font-bold cursor-pointer flex items-center gap-1.5 ${
              showGrid
                ? 'bg-gradient-to-r from-cyan-700 to-blue-700 text-white dark:from-cyan-950 dark:to-slate-900 dark:text-cyan-300 border-cyan-500 dark:border-cyan-400/60 shadow-md dark:shadow-[0_0_10px_rgba(6,182,212,0.25)] ring-1 ring-cyan-400/30'
                : 'bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-800 text-slate-700 dark:text-slate-300 hover:text-slate-950 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-850'
            }`}
          >
            <Compass className={`w-3 h-3 ${showGrid ? 'text-cyan-200' : 'text-slate-500'}`} />
            <span>Lat/Lng Grid: <strong className={showGrid ? 'text-cyan-100 dark:text-cyan-300' : 'text-slate-500'}>{showGrid ? 'ON' : 'OFF'}</strong></span>
          </button>
        </div>
      </div>

      {/* 3. Main Viewer Area with Cyclone Spiral Visualization & Vertical Scale */}
      <div className="relative rounded-2xl overflow-hidden bg-[#030712] border border-slate-800 h-80 sm:h-96 flex items-center justify-center shadow-inner group">
        {/* Dynamic Channel Backdrops & Cyclone Vortex Visual */}
        <div className="w-full h-full relative flex items-center justify-center overflow-hidden">
          
          {/* Channel 1: INFRARED (IR 10.8µm) */}
          {activeChannel === 'IR' && (
            <div className="absolute inset-0 bg-gradient-to-br from-[#020617] via-[#0B1528] to-[#040814] flex items-center justify-center">
              {/* Diffuse atmospheric halo */}
              <div className="absolute w-72 h-72 rounded-full bg-cyan-500/15 blur-3xl"></div>
              <div className="absolute w-56 h-56 rounded-full bg-sky-400/20 blur-2xl"></div>

              {/* Rotating Spiral IR Cloud Bands SVG */}
              <div className="relative w-72 h-72 flex items-center justify-center">
                <svg className="w-full h-full cyclone-vortex-spin" viewBox="0 0 240 240">
                  <defs>
                    <radialGradient id="irCoreGrad" cx="50%" cy="50%" r="50%">
                      <stop offset="0%" stop-color="#FFFFFF" stop-opacity="0.95" />
                      <stop offset="35%" stop-color="#E0F2FE" stop-opacity="0.85" />
                      <stop offset="65%" stop-color="#7DD3FC" stop-opacity="0.6" />
                      <stop offset="100%" stop-color="#0284C7" stop-opacity="0.1" />
                    </radialGradient>
                    <linearGradient id="irArmGrad1" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stop-color="#FFFFFF" stop-opacity="0.9" />
                      <stop offset="50%" stop-color="#BAE6FD" stop-opacity="0.75" />
                      <stop offset="100%" stop-color="#0284C7" stop-opacity="0.15" />
                    </linearGradient>
                    <linearGradient id="irArmGrad2" x1="100%" y1="0%" x2="0%" y2="100%">
                      <stop offset="0%" stop-color="#F0F9FF" stop-opacity="0.85" />
                      <stop offset="60%" stop-color="#7DD3FC" stop-opacity="0.65" />
                      <stop offset="100%" stop-color="#0369A1" stop-opacity="0.1" />
                    </linearGradient>
                    <filter id="irBlur" x="-20%" y="-20%" width="140%" height="140%">
                      <feGaussianBlur stdDeviation="2.5" />
                    </filter>
                  </defs>

                  {/* Outer Outflow Shield */}
                  <path
                    d="M 120,20 C 175,15 220,50 225,105 C 230,160 190,215 135,225 C 80,235 25,200 15,145 C 5,90 55,25 120,20 Z"
                    fill="url(#irCoreGrad)"
                    opacity="0.25"
                    filter="url(#irBlur)"
                  />

                  {/* Primary Spiral Inflow Arm */}
                  <path
                    d="M 215,160 C 225,115 200,60 155,30 C 110,5 50,25 25,70 C 0,115 20,170 70,195 C 115,220 170,200 185,150 C 195,110 170,70 135,60 C 100,50 75,70 70,100 C 65,125 80,145 105,145 C 125,145 138,130 135,112 C 132,100 120,92 110,95"
                    stroke="url(#irArmGrad1)"
                    strokeWidth="14"
                    strokeLinecap="round"
                    fill="none"
                    filter="url(#irBlur)"
                  />

                  {/* Secondary Opposing Inflow Arm */}
                  <path
                    d="M 25,80 C 15,135 45,190 95,210 C 145,230 200,200 220,145 C 235,90 200,35 145,20 C 95,5 50,40 45,90 C 40,130 75,165 118,162 C 150,160 175,130 170,98 C 165,70 140,58 115,62 C 95,68 85,85 90,102"
                    stroke="url(#irArmGrad2)"
                    strokeWidth="11"
                    strokeLinecap="round"
                    fill="none"
                    filter="url(#irBlur)"
                  />

                  {/* Dense Eyewall Cloud Ring */}
                  <circle cx="120" cy="120" r="18" stroke="#FFFFFF" strokeWidth="6" fill="none" opacity="0.95" filter="url(#irBlur)" />
                </svg>

                {/* Fixed Calm Eye Core */}
                <div className="absolute w-8 h-8 rounded-full bg-[#020617] border-2 border-white/80 shadow-2xl flex items-center justify-center">
                  <div className="w-2 h-2 rounded-full bg-cyan-400"></div>
                </div>
              </div>
            </div>
          )}

          {/* Channel 2: VISIBLE (0.65µm Optical Reflectance) */}
          {activeChannel === 'VIS' && (
            <div className="absolute inset-0 bg-gradient-to-br from-[#060D1A] via-[#0A182F] to-[#040A14] flex items-center justify-center">
              {/* Ocean reflectance background */}
              <div className="absolute w-80 h-80 rounded-full bg-white/10 blur-3xl"></div>

              {/* Rotating Visible Cloud Formation */}
              <div className="relative w-72 h-72 flex items-center justify-center">
                <svg className="w-full h-full cyclone-vortex-spin" viewBox="0 0 240 240">
                  <defs>
                    <linearGradient id="visBand1" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stop-color="#FFFFFF" stop-opacity="0.95" />
                      <stop offset="60%" stop-color="#E2E8F0" stop-opacity="0.85" />
                      <stop offset="100%" stop-color="#94A3B8" stop-opacity="0.2" />
                    </linearGradient>
                    <linearGradient id="visBand2" x1="100%" y1="0%" x2="0%" y2="100%">
                      <stop offset="0%" stop-color="#F8FAFC" stop-opacity="0.9" />
                      <stop offset="70%" stop-color="#CBD5E1" stop-opacity="0.75" />
                      <stop offset="100%" stop-color="#64748B" stop-opacity="0.15" />
                    </linearGradient>
                    <filter id="visBlur" x="-20%" y="-20%" width="140%" height="140%">
                      <feGaussianBlur stdDeviation="2.0" />
                    </filter>
                  </defs>

                  {/* Outer Cloud Veil */}
                  <circle cx="120" cy="120" r="95" fill="#FFFFFF" opacity="0.12" filter="url(#visBlur)" />

                  {/* Primary Bright Optical Cloud Band */}
                  <path
                    d="M 215,160 C 225,115 200,60 155,30 C 110,5 50,25 25,70 C 0,115 20,170 70,195 C 115,220 170,200 185,150 C 195,110 170,70 135,60 C 100,50 75,70 70,100 C 65,125 80,145 105,145 C 125,145 138,130 135,112 C 132,100 120,92 110,95"
                    stroke="url(#visBand1)"
                    strokeWidth="15"
                    strokeLinecap="round"
                    fill="none"
                    filter="url(#visBlur)"
                  />

                  {/* Secondary Optical Feeder Band */}
                  <path
                    d="M 25,80 C 15,135 45,190 95,210 C 145,230 200,200 220,145 C 235,90 200,35 145,20 C 95,5 50,40 45,90 C 40,130 75,165 118,162 C 150,160 175,130 170,98 C 165,70 140,58 115,62 C 95,68 85,85 90,102"
                    stroke="url(#visBand2)"
                    strokeWidth="12"
                    strokeLinecap="round"
                    fill="none"
                    filter="url(#visBlur)"
                  />

                  {/* Dense Visible Eyewall */}
                  <circle cx="120" cy="120" r="17" stroke="#FFFFFF" strokeWidth="6" fill="none" opacity="0.98" filter="url(#visBlur)" />
                </svg>

                {/* Clear Eye with Deep Ocean Color */}
                <div className="absolute w-7 h-7 rounded-full bg-[#06101E] border-2 border-white/90 shadow-xl flex items-center justify-center">
                  <div className="w-1.5 h-1.5 rounded-full bg-white/80"></div>
                </div>
              </div>
            </div>
          )}

          {/* Channel 3: THERMAL (12.0µm Enhanced BD-Curve) */}
          {activeChannel === 'THERMAL' && (
            <div className="absolute inset-0 bg-gradient-to-br from-[#0F051D] via-[#1A0933] to-[#080210] flex items-center justify-center">
              {/* Cold convective core glow */}
              <div className="absolute w-72 h-72 rounded-full bg-purple-600/20 blur-3xl"></div>
              <div className="absolute w-48 h-48 rounded-full bg-rose-600/25 blur-2xl"></div>
              <div className="absolute w-32 h-32 rounded-full bg-amber-500/20 blur-xl"></div>

              {/* Rotating Thermal Cloud Bands SVG */}
              <div className="relative w-72 h-72 flex items-center justify-center">
                <svg className="w-full h-full cyclone-vortex-spin" viewBox="0 0 240 240">
                  <defs>
                    <linearGradient id="thermGrad1" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stop-color="#FFFFFF" stop-opacity="0.95" />
                      <stop offset="30%" stop-color="#E879F9" stop-opacity="0.85" />
                      <stop offset="70%" stop-color="#F43F5E" stop-opacity="0.75" />
                      <stop offset="100%" stop-color="#0284C7" stop-opacity="0.2" />
                    </linearGradient>
                    <linearGradient id="thermGrad2" x1="100%" y1="0%" x2="0%" y2="100%">
                      <stop offset="0%" stop-color="#F43F5E" stop-opacity="0.9" />
                      <stop offset="40%" stop-color="#FB923C" stop-opacity="0.8" />
                      <stop offset="80%" stop-color="#A855F7" stop-opacity="0.6" />
                      <stop offset="100%" stop-color="#1E1B4B" stop-opacity="0.1" />
                    </linearGradient>
                    <filter id="thermBlur" x="-20%" y="-20%" width="140%" height="140%">
                      <feGaussianBlur stdDeviation="2.5" />
                    </filter>
                  </defs>

                  {/* Outer Cold Cloud Tops */}
                  <path
                    d="M 215,160 C 225,115 200,60 155,30 C 110,5 50,25 25,70 C 0,115 20,170 70,195 C 115,220 170,200 185,150 C 195,110 170,70 135,60 C 100,50 75,70 70,100 C 65,125 80,145 105,145 C 125,145 138,130 135,112 C 132,100 120,92 110,95"
                    stroke="url(#thermGrad1)"
                    strokeWidth="14"
                    strokeLinecap="round"
                    fill="none"
                    filter="url(#thermBlur)"
                  />

                  {/* Inner Intense Convective Arm */}
                  <path
                    d="M 25,80 C 15,135 45,190 95,210 C 145,230 200,200 220,145 C 235,90 200,35 145,20 C 95,5 50,40 45,90 C 40,130 75,165 118,162 C 150,160 175,130 170,98 C 165,70 140,58 115,62 C 95,68 85,85 90,102"
                    stroke="url(#thermGrad2)"
                    strokeWidth="12"
                    strokeLinecap="round"
                    fill="none"
                    filter="url(#thermBlur)"
                  />

                  {/* Cold Eyewall Core (-80°C Pink/White) */}
                  <circle cx="120" cy="120" r="18" stroke="#FFFFFF" strokeWidth="5.5" fill="none" opacity="0.95" filter="url(#thermBlur)" />
                  <circle cx="120" cy="120" r="14" stroke="#F43F5E" strokeWidth="3" fill="none" opacity="0.85" />
                </svg>

                {/* Warm Eye Core (+18°C) */}
                <div className="absolute w-7 h-7 rounded-full bg-[#1E1B4B] border-2 border-rose-400 shadow-xl flex items-center justify-center">
                  <div className="w-2 h-2 rounded-full bg-amber-400"></div>
                </div>
              </div>
            </div>
          )}

          {/* Channel 4: WATER VAPOR (6.8µm Upper Troposphere) */}
          {activeChannel === 'WV' && (
            <div className="absolute inset-0 bg-gradient-to-br from-[#02182B] via-[#052942] to-[#010E1A] flex items-center justify-center">
              {/* Moisture plume background */}
              <div className="absolute w-80 h-80 rounded-full bg-blue-500/20 blur-3xl"></div>
              <div className="absolute w-56 h-56 rounded-full bg-cyan-400/25 blur-2xl"></div>

              {/* Rotating Water Vapor Streams SVG */}
              <div className="relative w-72 h-72 flex items-center justify-center">
                <svg className="w-full h-full cyclone-vortex-spin" viewBox="0 0 240 240">
                  <defs>
                    <linearGradient id="wvGrad1" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stop-color="#67E8F9" stop-opacity="0.95" />
                      <stop offset="50%" stop-color="#38BDF8" stop-opacity="0.8" />
                      <stop offset="100%" stop-color="#1E40AF" stop-opacity="0.15" />
                    </linearGradient>
                    <linearGradient id="wvGrad2" x1="100%" y1="0%" x2="0%" y2="100%">
                      <stop offset="0%" stop-color="#93C5FD" stop-opacity="0.9" />
                      <stop offset="60%" stop-color="#2563EB" stop-opacity="0.65" />
                      <stop offset="100%" stop-color="#0F172A" stop-opacity="0.1" />
                    </linearGradient>
                    <filter id="wvBlur" x="-20%" y="-20%" width="140%" height="140%">
                      <feGaussianBlur stdDeviation="3.0" />
                    </filter>
                  </defs>

                  {/* Broad Moisture Swirl */}
                  <path
                    d="M 215,160 C 225,115 200,60 155,30 C 110,5 50,25 25,70 C 0,115 20,170 70,195 C 115,220 170,200 185,150 C 195,110 170,70 135,60 C 100,50 75,70 70,100 C 65,125 80,145 105,145 C 125,145 138,130 135,112 C 132,100 120,92 110,95"
                    stroke="url(#wvGrad1)"
                    strokeWidth="16"
                    strokeLinecap="round"
                    fill="none"
                    filter="url(#wvBlur)"
                  />

                  {/* Inflow Moisture Stream */}
                  <path
                    d="M 25,80 C 15,135 45,190 95,210 C 145,230 200,200 220,145 C 235,90 200,35 145,20 C 95,5 50,40 45,90 C 40,130 75,165 118,162 C 150,160 175,130 170,98 C 165,70 140,58 115,62 C 95,68 85,85 90,102"
                    stroke="url(#wvGrad2)"
                    strokeWidth="13"
                    strokeLinecap="round"
                    fill="none"
                    filter="url(#wvBlur)"
                  />

                  {/* Upper Circulation Core */}
                  <circle cx="120" cy="120" r="19" stroke="#38BDF8" strokeWidth="6" fill="none" opacity="0.9" filter="url(#wvBlur)" />
                </svg>

                {/* Dry Subsidence Slot at Center */}
                <div className="absolute w-7 h-7 rounded-full bg-[#021324] border-2 border-cyan-400 shadow-xl flex items-center justify-center">
                  <div className="w-1.5 h-1.5 rounded-full bg-cyan-300"></div>
                </div>
              </div>
            </div>
          )}

          {/* Lat/Lng Grid & Reticle Overlay */}
          {showGrid && (
            <div className="absolute inset-0 pointer-events-none">
              <div className="w-full h-full grid grid-cols-4 grid-rows-4 border border-cyan-500/10">
                {Array.from({ length: 16 }).map((_, i) => (
                  <div key={i} className="border border-cyan-500/10 relative">
                    {i === 0 && <span className="absolute top-1 left-1.5 text-[9px] font-mono text-cyan-400/70 font-semibold">18.0°N, 84.0°E</span>}
                    {i === 3 && <span className="absolute top-1 right-1.5 text-[9px] font-mono text-cyan-400/70 font-semibold">18.0°N, 90.0°E</span>}
                    {i === 12 && <span className="absolute bottom-1 left-1.5 text-[9px] font-mono text-cyan-400/70 font-semibold">12.0°N, 84.0°E</span>}
                    {i === 15 && <span className="absolute bottom-1 right-1.5 text-[9px] font-mono text-cyan-400/70 font-semibold">12.0°N, 90.0°E</span>}
                  </div>
                ))}
              </div>
              {/* Tactical Center Target Reticle */}
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="w-16 h-16 rounded-full border border-cyan-400/25"></div>
                <div className="absolute w-32 h-32 rounded-full border border-cyan-500/10"></div>
                <div className="absolute w-2 h-2 rounded-full border border-rose-500 animate-ping"></div>
              </div>
            </div>
          )}

          {/* Top-Left Channel Overlay Badge */}
          <div className="absolute top-3 left-3 pointer-events-none text-left font-mono space-y-1">
            <div className="flex items-center gap-1.5 bg-black/85 border border-cyan-500/30 text-cyan-300 px-2 py-0.5 rounded text-[10px] font-bold backdrop-blur-xs">
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
              INSAT-3D • {currentChannelInfo.label}
            </div>
            <div className="text-[9px] text-slate-300 bg-black/75 px-1.5 py-0.5 rounded backdrop-blur-xs">
              {cyclone.lastUpdated}
            </div>
          </div>

          {/* Top-Right Simulated Marker */}
          <div className="absolute top-3 right-12 sm:right-14 pointer-events-none">
            <span className="px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 font-mono text-[9px] font-bold border border-amber-500/40 backdrop-blur-xs">
              SIMULATED APERTURE
            </span>
          </div>

          {/* Right-Side Vertical Temperature / Intensity Scale */}
          <div className="absolute top-3 bottom-3 right-2.5 w-7 sm:w-8 bg-black/85 border border-slate-700/80 rounded-lg p-1 flex flex-col justify-between items-center text-[8px] font-mono text-slate-300 z-10 backdrop-blur-xs shadow-lg">
            <span className="text-[7px] text-cyan-300 font-bold uppercase rotate-90 my-1 whitespace-nowrap">
              {activeChannel === 'VIS' ? 'ALBEDO' : 'TEMP °C'}
            </span>

            {/* Vertical Gradient Bar */}
            <div className="w-2 flex-1 rounded-sm my-1 relative overflow-hidden">
              {activeChannel === 'THERMAL' ? (
                <div className="w-full h-full bg-gradient-to-b from-[#FFFFFF] via-[#E879F9] via-[#F43F5E] via-[#FB923C] to-[#0284C7]"></div>
              ) : activeChannel === 'VIS' ? (
                <div className="w-full h-full bg-gradient-to-b from-[#FFFFFF] via-[#CBD5E1] via-[#64748B] to-[#0F172A]"></div>
              ) : activeChannel === 'WV' ? (
                <div className="w-full h-full bg-gradient-to-b from-[#67E8F9] via-[#38BDF8] via-[#2563EB] to-[#02182B]"></div>
              ) : (
                <div className="w-full h-full bg-gradient-to-b from-[#FFFFFF] via-[#BAE6FD] via-[#38BDF8] to-[#020617]"></div>
              )}
            </div>

            {/* Scale Value Readings */}
            <div className="flex flex-col justify-between h-28 text-[7px] text-slate-300 font-bold text-center">
              <span>{activeChannel === 'VIS' ? '100%' : '-85°'}</span>
              <span>{activeChannel === 'VIS' ? '70%' : '-60°'}</span>
              <span>{activeChannel === 'VIS' ? '40%' : '-30°'}</span>
              <span>{activeChannel === 'VIS' ? '10%' : '0°'}</span>
              <span>{activeChannel === 'VIS' ? '0%' : '+28°'}</span>
            </div>
          </div>

          {/* Bottom Feature Synopsis Bar */}
          <div className="absolute bottom-2.5 left-3 right-12 sm:right-14 bg-black/80 border border-slate-800 px-3 py-1 rounded-lg text-[10px] font-mono text-slate-300 flex items-center justify-between backdrop-blur-xs truncate">
            <span className="truncate">
              <strong className="text-cyan-300 font-bold">{currentChannelInfo.label}:</strong> {currentChannelInfo.desc}
            </span>
          </div>
        </div>
      </div>

      {/* 4. Four Compact Information Cards Below Viewer */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-xs">
        {/* Card 1: CURRENT STATUS */}
        <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-900/70 border border-slate-300 dark:border-slate-800 space-y-1 shadow-xs">
          <div className="flex items-center gap-1.5 text-slate-500 dark:text-slate-400 text-[10px] font-bold uppercase">
            <Activity className="w-3 h-3 text-cyan-600 dark:text-cyan-400" />
            <span>CURRENT STATUS</span>
          </div>
          <div className="font-extrabold text-slate-950 dark:text-white text-xs truncate">
            {cyclone.category}
          </div>
          <div className="text-[10px] font-bold text-emerald-700 dark:text-emerald-400 flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-600 animate-pulse"></span>
            {cyclone.currentStatus}
          </div>
        </div>

        {/* Card 2: MAX WIND SPEED */}
        <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-900/70 border border-slate-300 dark:border-slate-800 space-y-1 shadow-xs">
          <div className="flex items-center gap-1.5 text-slate-500 dark:text-slate-400 text-[10px] font-bold uppercase">
            <Wind className="w-3 h-3 text-rose-600 dark:text-rose-400" />
            <span>MAX WIND SPEED</span>
          </div>
          <div className="font-extrabold text-rose-600 dark:text-rose-400 text-xs sm:text-sm">
            {cyclone.maxSustainedWindKts} kt
          </div>
          <div className="text-[10px] text-slate-600 dark:text-slate-400 font-medium">
            {cyclone.maxSustainedWindKmh} km/h (1-min avg)
          </div>
        </div>

        {/* Card 3: PRESSURE */}
        <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-900/70 border border-slate-300 dark:border-slate-800 space-y-1 shadow-xs">
          <div className="flex items-center gap-1.5 text-slate-500 dark:text-slate-400 text-[10px] font-bold uppercase">
            <Gauge className="w-3 h-3 text-sky-600 dark:text-cyan-400" />
            <span>PRESSURE</span>
          </div>
          <div className="font-extrabold text-slate-950 dark:text-white text-xs sm:text-sm">
            {cyclone.centralPressureHpa} hPa
          </div>
          <div className="text-[10px] text-cyan-700 dark:text-cyan-400 font-semibold">
            Central Pressure Drop
          </div>
        </div>

        {/* Card 4: MOVEMENT */}
        <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-900/70 border border-slate-300 dark:border-slate-800 space-y-1 shadow-xs">
          <div className="flex items-center gap-1.5 text-slate-500 dark:text-slate-400 text-[10px] font-bold uppercase">
            <Compass className="w-3 h-3 text-purple-600 dark:text-purple-400" />
            <span>MOVEMENT</span>
          </div>
          <div className="font-extrabold text-purple-900 dark:text-purple-300 text-xs sm:text-sm truncate">
            {cyclone.movementDirection} • {cyclone.movementSpeedKmh} km/h
          </div>
          <div className="text-[10px] text-slate-600 dark:text-slate-400 font-medium">
            Heading: 315° NW (Translational)
          </div>
        </div>
      </div>
    </div>
  );
};
