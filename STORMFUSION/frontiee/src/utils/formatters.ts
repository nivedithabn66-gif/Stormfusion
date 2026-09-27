import { CycloneCategory, RiskSeverity } from '../types/cyclone';

export const formatCoords = (lat: number, lng: number): string => {
  const latStr = `${Math.abs(lat).toFixed(2)}° ${lat >= 0 ? 'N' : 'S'}`;
  const lngStr = `${Math.abs(lng).toFixed(2)}° ${lng >= 0 ? 'E' : 'W'}`;
  return `${latStr}, ${lngStr}`;
};

export const ktsToKmh = (kts: number): number => {
  return Math.round(kts * 1.852);
};

export const getCategoryColor = (category: CycloneCategory): {
  bg: string;
  text: string;
  border: string;
  badge: string;
} => {
  switch (category) {
    case 'Super Cyclonic Storm':
      return { 
        bg: 'bg-purple-100/80 dark:bg-purple-950/70', 
        text: 'text-purple-900 dark:text-purple-300', 
        border: 'border-purple-300 dark:border-purple-500', 
        badge: 'bg-purple-700 text-white' 
      };
    case 'Extremely Severe Cyclonic Storm':
    case 'Very Severe Cyclonic Storm':
      return { 
        bg: 'bg-rose-50/90 dark:bg-rose-950/70', 
        text: 'text-rose-900 dark:text-rose-300', 
        border: 'border-rose-300 dark:border-rose-500', 
        badge: 'bg-rose-700 text-white' 
      };
    case 'Severe Cyclonic Storm':
      return { 
        bg: 'bg-orange-50/90 dark:bg-orange-950/70', 
        text: 'text-orange-900 dark:text-orange-300', 
        border: 'border-orange-300 dark:border-orange-500', 
        badge: 'bg-orange-700 text-white' 
      };
    case 'Cyclonic Storm':
      return { 
        bg: 'bg-amber-50/90 dark:bg-amber-950/70', 
        text: 'text-amber-900 dark:text-amber-300', 
        border: 'border-amber-300 dark:border-amber-500', 
        badge: 'bg-amber-600 text-slate-950 font-bold' 
      };
    case 'Deep Depression':
    case 'Depression':
      return { 
        bg: 'bg-cyan-50/90 dark:bg-cyan-950/70', 
        text: 'text-cyan-950 dark:text-cyan-300', 
        border: 'border-cyan-300 dark:border-cyan-500', 
        badge: 'bg-cyan-700 text-white' 
      };
    default:
      return { 
        bg: 'bg-slate-100 dark:bg-slate-900', 
        text: 'text-slate-900 dark:text-slate-300', 
        border: 'border-slate-300 dark:border-slate-700', 
        badge: 'bg-slate-700 text-white' 
      };
  }
};

export const getRiskSeverityColor = (risk: RiskSeverity): {
  bg: string;
  text: string;
  border: string;
  badgeBg: string;
} => {
  switch (risk) {
    case 'CRITICAL':
      return { 
        bg: 'bg-rose-50/90 dark:bg-rose-950/40', 
        text: 'text-rose-900 dark:text-rose-400', 
        border: 'border-rose-300 dark:border-rose-500/50', 
        badgeBg: 'bg-rose-700 text-white' 
      };
    case 'HIGH':
      return { 
        bg: 'bg-orange-50/90 dark:bg-orange-950/40', 
        text: 'text-orange-900 dark:text-orange-400', 
        border: 'border-orange-300 dark:border-orange-500/50', 
        badgeBg: 'bg-orange-600 text-white' 
      };
    case 'MODERATE':
      return { 
        bg: 'bg-amber-50/90 dark:bg-amber-950/40', 
        text: 'text-amber-900 dark:text-amber-400', 
        border: 'border-amber-300 dark:border-amber-500/50', 
        badgeBg: 'bg-amber-600 text-slate-950 font-bold' 
      };
    case 'LOW':
    default:
      return { 
        bg: 'bg-emerald-50/90 dark:bg-emerald-950/40', 
        text: 'text-emerald-900 dark:text-emerald-400', 
        border: 'border-emerald-300 dark:border-emerald-500/50', 
        badgeBg: 'bg-emerald-700 text-white' 
      };
  }
};

export const calculateDistanceKm = (
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number
): number => {
  const R = 6371; // Radius of Earth in km
  const dLat = (lat2 - lat1) * (Math.PI / 180);
  const dLon = (lon2 - lon1) * (Math.PI / 180);
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(lat1 * (Math.PI / 180)) *
      Math.cos(lat2 * (Math.PI / 180)) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return Math.round(R * c);
};
