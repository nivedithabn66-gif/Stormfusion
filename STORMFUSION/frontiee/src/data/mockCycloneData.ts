import { CycloneTelemetry } from '../types/cyclone';

export const mockCurrentCyclone: CycloneTelemetry = {
  id: "TC-2026-NIO-02B",
  name: "CYCLONE",
  basin: "North Indian Ocean (Bay of Bengal)",
  currentStatus: "ACTIVE",
  category: "Very Severe Cyclonic Storm",
  coordinates: {
    lat: 14.52,
    lng: 87.21
  },
  maxSustainedWindKts: 82,
  maxSustainedWindKmh: 152,
  centralPressureHpa: 965,
  movementDirection: "NW",
  movementSpeedKmh: 14,
  modelConfidencePercent: 87,
  detectionConfidencePercent: 96,
  rapidIntensificationRisk: "MODERATE",
  cyclonePattern: "Curved Band / Developing Eye",
  aiModelName: "EfficientNet-B0 + ConvLSTM (Multimodal)",
  lastUpdated: "07 Sep 2026 — 14:32 IST",
  isSimulated: true
};

export const mockEnvironmentalSoundings = {
  seaSurfaceTemperatureC: 30.8, // Warm waters supporting intensification
  verticalWindShearKts: 12,      // Favorable low shear
  oceanHeatContentKjCm2: 98,    // High potential energy
  midLevelMoisturePercent: 78,  // High relative humidity
  outflowEfficiency: "High (Dual radial channel)"
};
