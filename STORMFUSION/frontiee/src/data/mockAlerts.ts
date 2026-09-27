import { EmergencyAlert } from '../types/cyclone';

export const mockAlerts: EmergencyAlert[] = [
  {
    id: "ALT-2026-0907-01",
    severity: "CRITICAL",
    headline: "Severe Cyclonic Landfall Imminent — North AP & South Odisha Coastal Corridor",
    description: "Multimodal AI track models project landfall between Visakhapatnam and Srikakulam within 68–74 hours. Maximum sustained winds exceeding 85 kts with catastrophic storm surge potential of 2.8m to 3.2m above astronomical tide.",
    affectedRegions: ["Visakhapatnam (AP)", "Srikakulam (AP)", "Vizianagaram (AP)", "Ganjam (Odisha)"],
    timestamp: "07 Sep 2026 — 14:15 IST",
    confidencePercent: 88,
    status: "ACTIVE",
    source: "STORMFUSION Early Warning Engine & IMD Prototype Feed",
    recommendedAction: "Activate Level-3 Red Emergency Response. Mobilize NDRF/SDRF battalions. Begin phased evacuation of low-lying coastal hamlets within 5km zone."
  },
  {
    id: "ALT-2026-0907-02",
    severity: "WARNING",
    headline: "Gale Wind Hazard & Severe Sea Condition Warning",
    description: "Winds of 70–85 kts gusting to 95 kts expected over west-central Bay of Bengal. Rough to phenomenal sea conditions. Squally winds extending up to 250 km from cyclone center.",
    affectedRegions: ["West-Central Bay of Bengal", "North Coastal Andhra Pradesh", "South Odisha Coast"],
    timestamp: "07 Sep 2026 — 13:00 IST",
    confidencePercent: 91,
    status: "ACTIVE",
    source: "Numerical Wind Field Model (ERA5-Augmented)",
    recommendedAction: "Total suspension of all fishing operations, offshore exploration, and coastal small craft. Hoist Great Danger Signal No. 8 at Visakhapatnam and Gangavaram ports."
  },
  {
    id: "ALT-2026-0907-03",
    severity: "WARNING",
    headline: "Heavy to Extremely Heavy Inundation Forecast",
    description: "Convective rainbands predicted to dump 200–280 mm cumulative precipitation within 24 hours of landfall. Flash flooding risk in Nagavali and Vamsadhara river basins.",
    affectedRegions: ["Srikakulam", "Vizianagaram", "Ganjam", "Gajapati"],
    timestamp: "07 Sep 2026 — 11:45 IST",
    confidencePercent: 84,
    status: "ACTIVE",
    source: "INSAT-3D Quantitative Precipitation Estimation (QPE)",
    recommendedAction: "Monitor minor irrigation reservoirs. Pre-position de-watering pumps and power gensets at vulnerable hospital installations."
  },
  {
    id: "ALT-2026-0907-04",
    severity: "WATCH",
    headline: "Peripheral Squall Activity & Ingress Surveillance",
    description: "Deep convective cloud clusters observed expanding north-northwestward. High cloud tops reaching -82°C indicating strong updraft sustainability.",
    affectedRegions: ["Puri", "Kendrapara", "Jagatsinghpur", "East Godavari"],
    timestamp: "07 Sep 2026 — 09:30 IST",
    confidencePercent: 78,
    status: "ACKNOWLEDGED",
    source: "INSAT-3D Water Vapour Channel & Rapid Intensification Classifier",
    recommendedAction: "Maintain district disaster control room operational readiness. Verify satellite communication backups."
  },
  {
    id: "ALT-2026-0907-05",
    severity: "ADVISORY",
    headline: "Marine Shipping Lane Diversion Advisory",
    description: "International cargo routes through southern and central Bay of Bengal diverted 150 nautical miles eastward.",
    affectedRegions: ["Bay of Bengal Shipping Corridor"],
    timestamp: "06 Sep 2026 — 18:00 IST",
    confidencePercent: 95,
    status: "RESOLVED",
    source: "DG Shipping & Maritime Early Alert Engine",
    recommendedAction: "Vessels rerouted via Andaman Sea eastern transit corridor."
  }
];
