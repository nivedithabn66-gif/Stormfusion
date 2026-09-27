import React from 'react';
import { 
  Clock, 
  Wind, 
  Compass, 
  ShieldAlert, 
  Activity, 
  Target, 
  ArrowUpRight,
  TrendingUp,
  AlertCircle
} from 'lucide-react';
import { ForecastHourData } from '../types/cyclone';
import { formatCoords, getCategoryColor, ktsToKmh } from '../utils/formatters';

interface ForecastCardProps {
  forecast: ForecastHourData;
  isPeak?: boolean;
  isLandfall?: boolean;
}

export const ForecastCard: React.FC<ForecastCardProps> = ({ forecast, isPeak, isLandfall }) => {
  const categoryStyles = getCategoryColor(forecast.category);

  return (
    <div className={`command-card rounded-xl p-4 border relative overflow-hidden transition-all hover:border-cyan-500/50 ${
      isPeak 
        ? 'border-purple-300 dark:border-purple-500/40 bg-purple-50/60 dark:bg-purple-950/20' 
        : isLandfall 
        ? 'border-rose-300 dark:border-rose-500/40 bg-rose-50/60 dark:bg-rose-950/20' 
        : 'border-slate-300 dark:border-slate-800'
    }`}>
      {/* Top Header Badge */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-300 dark:border-slate-800/80">
        <div className="flex items-center gap-2">
          <span className="w-8 h-8 rounded-lg bg-cyan-500/15 border border-cyan-500/30 flex items-center justify-center font-mono font-bold text-cyan-800 dark:text-cyan-300 text-xs">
            +{forecast.hour}h
          </span>
          <div>
            <div className="text-xs font-bold font-mono text-slate-950 dark:text-white flex items-center gap-1.5">
              <span>T+{forecast.hour} FORECAST</span>
              {isPeak && (
                <span className="px-1.5 py-0.2 rounded text-[9px] font-bold bg-purple-500/20 text-purple-800 dark:text-purple-300 border border-purple-500/40">
                  PEAK
                </span>
              )}
              {isLandfall && (
                <span className="px-1.5 py-0.2 rounded text-[9px] font-bold bg-rose-500/20 text-rose-800 dark:text-rose-300 border border-rose-500/40 animate-pulse">
                  LANDFALL
                </span>
              )}
            </div>
            <div className="text-[10px] text-slate-600 dark:text-slate-400 font-mono flex items-center gap-1 mt-0.5 font-medium">
              <Clock className="w-3 h-3 text-slate-500 dark:text-slate-400" />
              {forecast.targetTime}
            </div>
          </div>
        </div>

        {/* Confidence pill */}
        <div className="text-right">
          <div className="text-[10px] text-slate-600 dark:text-slate-400 font-mono font-semibold">Confidence</div>
          <div className="text-xs font-black font-mono text-cyan-800 dark:text-cyan-400">
            {forecast.confidencePercent}%
          </div>
        </div>
      </div>

      {/* Category classification */}
      <div className={`mt-3 px-2.5 py-1.5 rounded-lg border text-xs font-mono font-bold flex items-center justify-between transition-colors ${categoryStyles.bg} ${categoryStyles.border} ${categoryStyles.text}`}>
        <div className="truncate">{forecast.category}</div>
        <ShieldAlert className="w-3.5 h-3.5 shrink-0" />
      </div>

      {/* Coordinates & Wind Speed */}
      <div className="mt-3 grid grid-cols-2 gap-2 text-xs font-mono">
        {/* Coordinates */}
        <div className="p-2 rounded-lg bg-[#F8FAFC] dark:bg-slate-900/60 border border-slate-300 dark:border-slate-850 transition-colors">
          <span className="text-[10px] text-slate-600 dark:text-slate-500 block font-bold">COORDINATES</span>
          <span className="font-bold text-slate-950 dark:text-slate-200 block mt-0.5">
            {formatCoords(forecast.coordinates.lat, forecast.coordinates.lng)}
          </span>
          <span className="text-[9px] text-cyan-800 dark:text-cyan-400 mt-0.5 block font-semibold">
            {isLandfall ? 'Landfall Corridor' : `${forecast.estimatedLandfallDistanceKm} km offshore`}
          </span>
        </div>

        {/* Projected Wind */}
        <div className="p-2 rounded-lg bg-[#F8FAFC] dark:bg-slate-900/60 border border-slate-300 dark:border-slate-850 transition-colors">
          <span className="text-[10px] text-slate-600 dark:text-slate-500 block font-bold">SUSTAINED WIND</span>
          <div className="flex items-baseline gap-1 mt-0.5">
            <span className="text-base font-black text-rose-700 dark:text-rose-400">{forecast.windSpeedKts}</span>
            <span className="text-[10px] text-slate-600 dark:text-slate-400 font-semibold">kt</span>
            <span className="text-[10px] text-slate-600 dark:text-slate-400 ml-auto font-semibold">({ktsToKmh(forecast.windSpeedKts)} km/h)</span>
          </div>
        </div>
      </div>

      {/* Error Margin bar */}
      <div className="mt-3 pt-2.5 border-t border-slate-300 dark:border-slate-800/80 flex items-center justify-between text-[10px] font-mono text-slate-600 dark:text-slate-400 font-medium">
        <span title="Along-track and cross-track error envelope">
          Error: ±{forecast.crossTrackErrorKm}km (Cross) / ±{forecast.alongTrackErrorKm}km (Along)
        </span>
        <span className="text-slate-600 dark:text-slate-500 font-bold">ConvLSTM</span>
      </div>
    </div>
  );
};
