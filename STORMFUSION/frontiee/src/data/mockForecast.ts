import { ForecastHourData, TrackPoint, RainfallDistrictForecast } from '../types/cyclone';

export const mockForecastHours: ForecastHourData[] = [
  {
    hour: 24,
    targetTime: "08 Sep 2026 — 14:30 IST",
    coordinates: { lat: 15.4, lng: 85.9 },
    windSpeedKts: 86,
    category: "Very Severe Cyclonic Storm",
    confidencePercent: 89,
    alongTrackErrorKm: 32,
    crossTrackErrorKm: 24,
    estimatedLandfallDistanceKm: 340
  },
  {
    hour: 48,
    targetTime: "09 Sep 2026 — 14:30 IST",
    coordinates: { lat: 16.8, lng: 84.2 },
    windSpeedKts: 91,
    category: "Very Severe Cyclonic Storm",
    confidencePercent: 82,
    alongTrackErrorKm: 68,
    crossTrackErrorKm: 52,
    estimatedLandfallDistanceKm: 180
  },
  {
    hour: 72,
    targetTime: "10 Sep 2026 — 14:30 IST",
    coordinates: { lat: 18.1, lng: 82.5 },
    windSpeedKts: 78,
    category: "Severe Cyclonic Storm",
    confidencePercent: 74,
    alongTrackErrorKm: 120,
    crossTrackErrorKm: 95,
    estimatedLandfallDistanceKm: 0 // Predicted Landfall near north AP/South Odisha border
  }
];

export const mockFullTrack: TrackPoint[] = [
  // Historical Track (Observed by INSAT-3D & IMD best track simulation)
  {
    lat: 11.2,
    lng: 91.8,
    timestamp: "05 Sep 2026 — 02:30 IST (-60h)",
    label: "T-60h (Genesis)",
    windSpeedKts: 28,
    centralPressureHpa: 1002,
    category: "Depression",
    isForecast: false
  },
  {
    lat: 12.1,
    lng: 90.6,
    timestamp: "05 Sep 2026 — 14:30 IST (-48h)",
    label: "T-48h",
    windSpeedKts: 35,
    centralPressureHpa: 996,
    category: "Deep Depression",
    isForecast: false
  },
  {
    lat: 12.9,
    lng: 89.4,
    timestamp: "06 Sep 2026 — 02:30 IST (-36h)",
    label: "T-36h",
    windSpeedKts: 48,
    centralPressureHpa: 988,
    category: "Cyclonic Storm",
    isForecast: false
  },
  {
    lat: 13.8,
    lng: 88.2,
    timestamp: "06 Sep 2026 — 14:30 IST (-24h)",
    label: "T-24h",
    windSpeedKts: 65,
    centralPressureHpa: 978,
    category: "Severe Cyclonic Storm",
    isForecast: false
  },
  {
    lat: 14.1,
    lng: 87.7,
    timestamp: "07 Sep 2026 — 02:30 IST (-12h)",
    label: "T-12h",
    windSpeedKts: 75,
    centralPressureHpa: 970,
    category: "Very Severe Cyclonic Storm",
    isForecast: false
  },
  // CURRENT POINT (0h)
  {
    lat: 14.52,
    lng: 87.21,
    timestamp: "07 Sep 2026 — 14:32 IST (Current)",
    label: "Current Eye",
    windSpeedKts: 82,
    centralPressureHpa: 965,
    category: "Very Severe Cyclonic Storm",
    isForecast: false,
    confidencePercent: 96
  },
  // FORECAST TRACK
  {
    lat: 15.4,
    lng: 85.9,
    timestamp: "08 Sep 2026 — 14:30 IST (+24h)",
    label: "+24h AI Forecast",
    windSpeedKts: 86,
    centralPressureHpa: 960,
    category: "Very Severe Cyclonic Storm",
    isForecast: true,
    forecastHour: 24,
    confidencePercent: 89,
    uncertaintyRadiusKm: 55
  },
  {
    lat: 16.8,
    lng: 84.2,
    timestamp: "09 Sep 2026 — 14:30 IST (+48h)",
    label: "+48h AI Forecast (Peak)",
    windSpeedKts: 91,
    centralPressureHpa: 954,
    category: "Very Severe Cyclonic Storm",
    isForecast: true,
    forecastHour: 48,
    confidencePercent: 82,
    uncertaintyRadiusKm: 110
  },
  {
    lat: 18.1,
    lng: 82.5,
    timestamp: "10 Sep 2026 — 14:30 IST (+72h)",
    label: "+72h Landfall Zone",
    windSpeedKts: 78,
    centralPressureHpa: 968,
    category: "Severe Cyclonic Storm",
    isForecast: true,
    forecastHour: 72,
    confidencePercent: 74,
    uncertaintyRadiusKm: 180
  }
];

// Polygonal coordinates enclosing the cone of uncertainty (from current position expanding to 72h)
export const mockUncertaintyConeCoords: [number, number][] = [
  [14.52, 87.21], // Apex at eye
  [15.8, 86.4],   // 24h right bound
  [17.4, 85.1],   // 48h right bound
  [19.0, 83.5],   // 72h right/north bound
  [18.6, 81.6],   // 72h top/west bound
  [17.6, 81.8],   // 72h left/south bound
  [16.2, 83.3],   // 48h left bound
  [15.0, 85.3],   // 24h left bound
  [14.52, 87.21]  // Close polygon
];

export const mockIntensityTimeSeries = [
  { time: "-48h", observedWind: 35, predictedWind: null, pressure: 996, confidence: 98 },
  { time: "-36h", observedWind: 48, predictedWind: null, pressure: 988, confidence: 97 },
  { time: "-24h", observedWind: 65, predictedWind: null, pressure: 978, confidence: 97 },
  { time: "-12h", observedWind: 75, predictedWind: null, pressure: 970, confidence: 96 },
  { time: "0h (Now)", observedWind: 82, predictedWind: 82, pressure: 965, confidence: 96 },
  { time: "+12h", observedWind: null, predictedWind: 84, pressure: 962, confidence: 92 },
  { time: "+24h", observedWind: null, predictedWind: 86, pressure: 960, confidence: 89 },
  { time: "+36h", observedWind: null, predictedWind: 89, pressure: 957, confidence: 85 },
  { time: "+48h", observedWind: null, predictedWind: 91, pressure: 954, confidence: 82 },
  { time: "+60h", observedWind: null, predictedWind: 85, pressure: 961, confidence: 78 },
  { time: "+72h", observedWind: null, predictedWind: 78, pressure: 968, confidence: 74 },
];

export const mockRainfallForecastZones: RainfallDistrictForecast[] = [
  {
    id: "rf-vskp",
    district: "Visakhapatnam",
    state: "Andhra Pradesh",
    coordinates: [17.6868, 83.2185],
    intensityCategory: "Extremely Heavy",
    rain24h: "Heavy",
    rain48h: "Very Heavy",
    rain72h: "Extremely Heavy",
    rain24hMm: 85,
    rain48hMm: 195,
    rain72hMm: 265,
    totalEstimatedMm: 545,
    riskLevel: "CRITICAL",
    peakWindow: "+48h to +72h (Landfall Ingress)",
    advisory: "Severe flash flood threat; storm water drainage choking in urban coastal plains."
  },
  {
    id: "rf-srkl",
    district: "Srikakulam",
    state: "Andhra Pradesh",
    coordinates: [18.2949, 83.8938],
    intensityCategory: "Extremely Heavy",
    rain24h: "Moderate",
    rain48h: "Heavy",
    rain72h: "Extremely Heavy",
    rain24hMm: 45,
    rain48hMm: 115,
    rain72hMm: 250,
    totalEstimatedMm: 410,
    riskLevel: "CRITICAL",
    peakWindow: "+60h to +72h (Eye Crossing)",
    advisory: "Nagavali & Vamsadhara river catchments at risk of sudden surge and embankment overflow."
  },
  {
    id: "rf-vznm",
    district: "Vizianagaram",
    state: "Andhra Pradesh",
    coordinates: [18.1124, 83.4158],
    intensityCategory: "Very Heavy",
    rain24h: "Moderate",
    rain48h: "Very Heavy",
    rain72h: "Heavy",
    rain24hMm: 35,
    rain48hMm: 140,
    rain72hMm: 175,
    totalEstimatedMm: 350,
    riskLevel: "HIGH",
    peakWindow: "+48h to +72h",
    advisory: "Flash floods in hilly Eastern Ghats catchments; soil saturation warning."
  },
  {
    id: "rf-gnjm",
    district: "Ganjam (Berhampur)",
    state: "Odisha",
    coordinates: [19.3150, 84.7941],
    intensityCategory: "Very Heavy",
    rain24h: "Moderate",
    rain48h: "Heavy",
    rain72h: "Very Heavy",
    rain24hMm: 25,
    rain48hMm: 95,
    rain72hMm: 185,
    totalEstimatedMm: 305,
    riskLevel: "HIGH",
    peakWindow: "+60h to +72h",
    advisory: "Rushikulya river basin swelling; coastal lowlands require precautionary dewatering."
  },
  {
    id: "rf-kknd",
    district: "Kakinada",
    state: "Andhra Pradesh",
    coordinates: [16.9891, 82.2475],
    intensityCategory: "Very Heavy",
    rain24h: "Heavy",
    rain48h: "Very Heavy",
    rain72h: "Moderate",
    rain24hMm: 75,
    rain48hMm: 130,
    rain72hMm: 55,
    totalEstimatedMm: 260,
    riskLevel: "HIGH",
    peakWindow: "+24h to +48h (Outer Spiral Rainband)",
    advisory: "Squally convective bursts; estuarine tidal locked canals risk backflow."
  },
  {
    id: "rf-puri",
    district: "Puri",
    state: "Odisha",
    coordinates: [19.8135, 85.8312],
    intensityCategory: "Heavy",
    rain24h: "Light",
    rain48h: "Moderate",
    rain72h: "Heavy",
    rain24hMm: 15,
    rain48hMm: 60,
    rain72hMm: 120,
    totalEstimatedMm: 195,
    riskLevel: "MODERATE",
    peakWindow: "+66h to +72h",
    advisory: "Intermittent heavy squalls; coastal beach erosion and squall line moisture convergence."
  }
];

// Swaths for map visualization
export const mockRainfallSwaths = {
  // Torrential Core Swath (>200mm)
  core: [
    [14.6, 87.0],
    [15.8, 85.2],
    [17.3, 83.5],
    [18.6, 83.2],
    [18.9, 84.2],
    [17.8, 84.6],
    [16.2, 85.8],
    [14.6, 87.0]
  ] as [number, number][],
  // Heavy to Very Heavy Swath (65-200mm)
  outer: [
    [13.8, 88.2],
    [15.0, 85.8],
    [16.4, 82.5],
    [18.0, 82.0],
    [19.6, 83.2],
    [20.3, 85.8],
    [18.8, 86.8],
    [16.8, 86.5],
    [13.8, 88.2]
  ] as [number, number][]
};

