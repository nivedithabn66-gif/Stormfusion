import { DistrictRisk } from '../types/cyclone';

export const mockDistrictRisks: DistrictRisk[] = [
  {
    district: "Visakhapatnam",
    state: "Andhra Pradesh",
    riskLevel: "CRITICAL",
    windRiskKts: 85,
    estimatedRainfallMm: 240,
    stormSurgeMeters: 2.8,
    populationExposed: "2.35 Million",
    criticalInfrastructure: ["Visakhapatnam Port", "Naval Base", "HPCL Refinery", "Coastal Highway NH16"],
    landfallProximityKm: 42,
    evacuationRecommended: true,
    coordinates: [17.6868, 83.2185]
  },
  {
    district: "Srikakulam",
    state: "Andhra Pradesh",
    riskLevel: "CRITICAL",
    windRiskKts: 88,
    estimatedRainfallMm: 280,
    stormSurgeMeters: 3.2,
    populationExposed: "1.78 Million",
    criticalInfrastructure: ["Bhavanapadu Port", "Substation Network", "Coastal Aquafarms"],
    landfallProximityKm: 25,
    evacuationRecommended: true,
    coordinates: [18.2949, 83.8938]
  },
  {
    district: "Vizianagaram",
    state: "Andhra Pradesh",
    riskLevel: "HIGH",
    windRiskKts: 75,
    estimatedRainfallMm: 210,
    stormSurgeMeters: 1.8,
    populationExposed: "1.42 Million",
    criticalInfrastructure: ["Howrah-Chennai Trunk Rail", "Power Grid Hub"],
    landfallProximityKm: 68,
    evacuationRecommended: true,
    coordinates: [18.1124, 83.3956]
  },
  {
    district: "Ganjam",
    state: "Odisha",
    riskLevel: "HIGH",
    windRiskKts: 72,
    estimatedRainfallMm: 220,
    stormSurgeMeters: 2.1,
    populationExposed: "2.10 Million",
    criticalInfrastructure: ["Gopalpur Port", "Chilika Biosphere Ingress", "National Highway NH16"],
    landfallProximityKm: 95,
    evacuationRecommended: true,
    coordinates: [19.3800, 85.0600]
  },
  {
    district: "Puri",
    state: "Odisha",
    riskLevel: "MODERATE",
    windRiskKts: 55,
    estimatedRainfallMm: 150,
    stormSurgeMeters: 1.2,
    populationExposed: "1.69 Million",
    criticalInfrastructure: ["Puri Coastal Pilgrim Corridor", "Tourism Assets", "Telecom Towers"],
    landfallProximityKm: 185,
    evacuationRecommended: false,
    coordinates: [19.8135, 85.8312]
  },
  {
    district: "East Godavari",
    state: "Andhra Pradesh",
    riskLevel: "HIGH",
    windRiskKts: 68,
    estimatedRainfallMm: 190,
    stormSurgeMeters: 2.0,
    populationExposed: "1.95 Million",
    criticalInfrastructure: ["Kakinada Deep Water Port", "Offshore Oil & Gas Terminals"],
    landfallProximityKm: 130,
    evacuationRecommended: true,
    coordinates: [16.9891, 82.2475]
  },
  {
    district: "Kendrapara & Jagatsinghpur",
    state: "Odisha",
    riskLevel: "MODERATE",
    windRiskKts: 52,
    estimatedRainfallMm: 140,
    stormSurgeMeters: 1.1,
    populationExposed: "2.20 Million",
    criticalInfrastructure: ["Paradip Major Port", "IOCL Refinery Pipelines"],
    landfallProximityKm: 260,
    evacuationRecommended: false,
    coordinates: [20.3165, 86.6115]
  },
  {
    district: "South 24 Parganas",
    state: "West Bengal",
    riskLevel: "LOW",
    windRiskKts: 40,
    estimatedRainfallMm: 95,
    stormSurgeMeters: 0.7,
    populationExposed: "3.10 Million",
    criticalInfrastructure: ["Sundarbans Embankments", "Ferry Terminals"],
    landfallProximityKm: 420,
    evacuationRecommended: false,
    coordinates: [22.1352, 88.5428]
  }
];
