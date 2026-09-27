import React from 'react';
import { 
  ResponsiveContainer, 
  AreaChart, 
  Area, 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  Tooltip, 
  CartesianGrid, 
  Legend,
  ReferenceLine 
} from 'recharts';
import { TrendingUp, Activity, Sparkles, AlertCircle } from 'lucide-react';
import { DemoModeBadge } from './DemoModeBadge';
import { useTheme } from '../context/ThemeContext';

interface ForecastTimelineProps {
  data: Array<{
    time: string;
    observedWind: number | null;
    predictedWind: number | null;
    pressure: number;
    confidence: number;
  }>;
}

export const ForecastTimeline: React.FC<ForecastTimelineProps> = ({ data }) => {
  const { resolvedTheme } = useTheme();
  const isLight = resolvedTheme === 'light';

  return (
    <div className="command-card rounded-2xl p-5 border border-slate-300 dark:border-slate-800 transition-colors">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-slate-300 dark:border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-cyan-700 dark:text-cyan-400" />
            <h3 className="text-base font-bold font-mono tracking-wide text-slate-950 dark:text-white uppercase">
              Cyclone Intensity & Pressure Trajectory
            </h3>
          </div>
          <p className="text-xs text-slate-600 dark:text-slate-400 font-mono mt-0.5 font-medium">
            Historical Observed Winds vs. ConvLSTM Multi-Horizon Projections (Past 48h to +72h)
          </p>
        </div>

        <div className="flex items-center gap-2">
          <DemoModeBadge subtle />
        </div>
      </div>

      {/* Chart Section */}
      <div className="mt-4 h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
            <defs>
              <linearGradient id="colorObserved" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={isLight ? "#0284C7" : "#38BDF8"} stopOpacity={0.4} />
                <stop offset="95%" stopColor={isLight ? "#0284C7" : "#38BDF8"} stopOpacity={0.0} />
              </linearGradient>
              <linearGradient id="colorPredicted" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#F43F5E" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#F43F5E" stopOpacity={0.0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke={isLight ? "#CBD5E1" : "#1E293B"} vertical={false} />
            <XAxis 
              dataKey="time" 
              stroke={isLight ? "#334155" : "#64748B"} 
              fontSize={11} 
              tickLine={false}
              fontFamily="JetBrains Mono"
            />
            <YAxis 
              stroke={isLight ? "#334155" : "#64748B"} 
              fontSize={11} 
              domain={[20, 105]} 
              tickLine={false}
              fontFamily="JetBrains Mono"
              unit=" kt"
            />
            <Tooltip
              content={({ active, payload, label }) => {
                if (active && payload && payload.length) {
                  const pt = payload[0].payload;
                  return (
                    <div className="bg-white dark:bg-slate-900/95 border border-slate-300 dark:border-slate-700 p-2.5 rounded-lg shadow-xl text-xs font-mono space-y-1">
                      <div className="font-bold text-slate-950 dark:text-white border-b border-slate-300 dark:border-slate-800 pb-1">
                        Timeline: {label}
                      </div>
                      {pt.observedWind !== null && (
                        <div className="text-sky-700 dark:text-sky-400 font-semibold">
                          Observed Wind: <strong>{pt.observedWind} kt</strong>
                        </div>
                      )}
                      {pt.predictedWind !== null && (
                        <div className="text-rose-700 dark:text-rose-400 font-semibold">
                          AI Predicted: <strong>{pt.predictedWind} kt</strong>
                        </div>
                      )}
                      <div className="text-slate-800 dark:text-slate-300">
                        Est. Pressure: <strong>{pt.pressure} hPa</strong>
                      </div>
                      <div className="text-purple-800 dark:text-purple-300 text-[10px] font-semibold">
                        Model Confidence: {pt.confidence}%
                      </div>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Legend 
              verticalAlign="top" 
              align="right"
              wrapperStyle={{ paddingBottom: '10px', fontSize: '11px', fontFamily: 'JetBrains Mono' }}
            />
            <ReferenceLine 
              x="0h (Now)" 
              stroke={isLight ? "#0284C7" : "#06B6D4"} 
              strokeDasharray="4 4" 
              label={{ value: 'CURRENT (NOW)', fill: isLight ? '#0284C7' : '#06B6D4', fontSize: 10, position: 'top' }} 
            />
            
            {/* Historical Observed Wind */}
            <Area
              type="monotone"
              dataKey="observedWind"
              name="Observed Wind (INSAT-3D)"
              stroke={isLight ? "#0284C7" : "#38BDF8"}
              strokeWidth={2.5}
              fillOpacity={1}
              fill="url(#colorObserved)"
            />
            
            {/* AI Predicted Wind */}
            <Area
              type="monotone"
              dataKey="predictedWind"
              name="AI Predicted Wind (kt)"
              stroke="#F43F5E"
              strokeWidth={2.5}
              strokeDasharray="5 5"
              fillOpacity={1}
              fill="url(#colorPredicted)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Footer Notes */}
      <div className="mt-3 pt-3 border-t border-slate-300 dark:border-slate-800/80 flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono text-slate-600 dark:text-slate-400 font-medium">
        <div className="flex items-center gap-2">
          <span className="inline-block w-2.5 h-2.5 rounded-full bg-purple-600 dark:bg-purple-400"></span>
          <span>Forecast Confidence decay: 89% (24h) → 82% (48h) → 74% (72h)</span>
        </div>
        <div className="text-slate-600 dark:text-slate-500 font-semibold">
          Source: Multimodal ConvLSTM Attention Network
        </div>
      </div>
    </div>
  );
};
