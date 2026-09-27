export type CycloneCategory = 
  | 'Low Pressure Area'
  | 'Depression'
  | 'Deep Depression'
  | 'Cyclonic Storm'
  | 'Severe Cyclonic Storm'
  | 'Very Severe Cyclonic Storm'
  | 'Extremely Severe Cyclonic Storm'
  | 'Super Cyclonic Storm';

export type RapidIntensificationRisk = 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL';

export interface CycloneCoordinates {
  lat: number;
  lng: number;
}

export interface TrackPoint extends CycloneCoordinates {
  timestamp: string; // ISO string or formatted
  label: string;
  windSpeedKts: number;
  centralPressureHpa: number;
  category: CycloneCategory;
  isForecast?: boolean;
  forecastHour?: 24 | 48 | 72;
  confidencePercent?: number;
  uncertaintyRadiusKm?: number;
}

export interface CycloneTelemetry {
  id: string;
  name: string;
  basin: string;
  currentStatus: 'ACTIVE' | 'WEAKENING' | 'DISSIPATED';
  category: CycloneCategory;
  coordinates: CycloneCoordinates;
  maxSustainedWindKts: number;
  maxSustainedWindKmh: number;
  centralPressureHpa: number;
  movementDirection: string;
  movementSpeedKmh: number;
  modelConfidencePercent: number;
  detectionConfidencePercent: number;
  rapidIntensificationRisk: RapidIntensificationRisk;
  cyclonePattern: string; // e.g. "Curved Band / Developing Eye"
  aiModelName: string; // e.g. "EfficientNet-B0 + ConvLSTM"
  lastUpdated: string;
  isSimulated: boolean;
}

export interface ForecastHourData {
  hour: 24 | 48 | 72;
  targetTime: string;
  coordinates: CycloneCoordinates;
  windSpeedKts: number;
  category: CycloneCategory;
  confidencePercent: number;
  alongTrackErrorKm: number;
  crossTrackErrorKm: number;
  estimatedLandfallDistanceKm: number;
}

export interface UncertaintyConePoint {
  center: CycloneCoordinates;
  radiusKm: number;
  leftBound: CycloneCoordinates;
  rightBound: CycloneCoordinates;
  hour: number;
}

export interface SatelliteFrame {
  id: string;
  timestamp: string;
  channel: 'IR' | 'VIS' | 'WV' | 'ENHANCED_IR';
  satelliteName: string;
  imageUrl: string;
  description: string;
  colorScale: string;
}

export interface GradCamAnalysis {
  modelArchitecture: string;
  explainabilityMethod: 'Grad-CAM' | 'Integrated Gradients' | 'Attention Map';
  confidenceScore: number;
  primaryAttentionZone: string;
  description: string;
  featuresIdentified: string[];
  heatmapVisualUrl: string;
  rawSatelliteUrl: string;
}

export interface HistoricalAnalogue {
  id: string;
  name: string;
  year: number;
  basin: string;
  similarityScore: number; // percentage
  trackSimilarity: number;
  peakIntensityKts: number;
  landfallDistrict: string;
  summary: string;
  trackCoordinates: CycloneCoordinates[];
}

export type RiskSeverity = 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL';

export interface DistrictRisk {
  district: string;
  state: string;
  riskLevel: RiskSeverity;
  windRiskKts: number;
  estimatedRainfallMm: number;
  stormSurgeMeters: number;
  populationExposed: string;
  criticalInfrastructure: string[];
  landfallProximityKm: number;
  evacuationRecommended: boolean;
  coordinates: [number, number]; // Lat, Lng
}

export interface EmergencyAlert {
  id: string;
  severity: 'CRITICAL' | 'WARNING' | 'WATCH' | 'ADVISORY';
  headline: string;
  description: string;
  affectedRegions: string[];
  timestamp: string;
  confidencePercent: number;
  status: 'ACTIVE' | 'ACKNOWLEDGED' | 'RESOLVED';
  source: string;
  recommendedAction: string;
}

export interface GroundReport {
  id: string;
  location: string;
  state: string;
  coordinates: [number, number];
  content: string;
  source: 'Social Media' | 'Volunteer Weather Watcher' | 'Local Ham Radio' | 'Crowdsourced App';
  status: 'UNVERIFIED' | 'INVESTIGATING' | 'CORROBORATED';
  timestamp: string;
  upvotes: number;
  verifiedOfficial: false; // Must ALWAYS be false
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  dataPoints?: {
    label: string;
    value: string;
  }[];
}

export interface RainfallDistrictForecast {
  id: string;
  district: string;
  state: string;
  coordinates: [number, number];
  intensityCategory: 'Extremely Heavy' | 'Very Heavy' | 'Heavy' | 'Moderate';
  rain24h: string;
  rain48h: string;
  rain72h: string;
  rain24hMm: number;
  rain48hMm: number;
  rain72hMm: number;
  totalEstimatedMm: number;
  riskLevel: 'CRITICAL' | 'HIGH' | 'MODERATE';
  peakWindow: string;
  advisory: string;
}

