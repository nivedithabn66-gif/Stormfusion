import { HistoricalAnalogue } from '../types/cyclone';

export const mockHistoricalAnalogues: HistoricalAnalogue[] = [
  {
    id: "ANALOGUE-HUDHUD-2014",
    name: "Cyclone Hudhud",
    year: 2014,
    basin: "North Indian Ocean (Bay of Bengal)",
    similarityScore: 87.4,
    trackSimilarity: 84.1,
    peakIntensityKts: 100,
    landfallDistrict: "Visakhapatnam, Andhra Pradesh",
    summary: "Hudhud followed an almost identical northwestward recurvature vector across the central Bay of Bengal, making catastrophic landfall over Visakhapatnam city with extreme gale force winds and severe structural damage.",
    trackCoordinates: [
      { lat: 10.8, lng: 92.5 },
      { lat: 12.3, lng: 90.1 },
      { lat: 13.9, lng: 88.0 },
      { lat: 15.2, lng: 86.1 },
      { lat: 16.5, lng: 84.4 },
      { lat: 17.7, lng: 83.3 } // Landfall Vizag
    ]
  },
  {
    id: "ANALOGUE-PHAILIN-2013",
    name: "Cyclone Phailin",
    year: 2013,
    basin: "North Indian Ocean (Bay of Bengal)",
    similarityScore: 81.2,
    trackSimilarity: 78.6,
    peakIntensityKts: 140,
    landfallDistrict: "Gopalpur, Ganjam (Odisha)",
    summary: "Extremely severe cyclonic storm with rapid intensification over warm sea surface temperatures. Made landfall near Gopalpur, causing massive storm surges and extensive communication loss.",
    trackCoordinates: [
      { lat: 11.5, lng: 93.0 },
      { lat: 13.2, lng: 90.5 },
      { lat: 15.1, lng: 88.3 },
      { lat: 16.9, lng: 86.2 },
      { lat: 18.5, lng: 85.1 },
      { lat: 19.3, lng: 84.9 } // Landfall Gopalpur
    ]
  },
  {
    id: "ANALOGUE-FANI-2019",
    name: "Cyclone Fani",
    year: 2019,
    basin: "North Indian Ocean (Bay of Bengal)",
    similarityScore: 76.5,
    trackSimilarity: 72.3,
    peakIntensityKts: 115,
    landfallDistrict: "Puri, Odisha",
    summary: "Extremely severe storm that tracked parallel to the Andhra coast before recurving into Puri, Odisha. Exhibited prominent pin-hole eye and broad outer rainbands.",
    trackCoordinates: [
      { lat: 5.2, lng: 88.5 },
      { lat: 9.1, lng: 86.8 },
      { lat: 13.8, lng: 84.5 },
      { lat: 16.2, lng: 84.9 },
      { lat: 18.4, lng: 85.5 },
      { lat: 19.8, lng: 85.8 }
    ]
  },
  {
    id: "ANALOGUE-TITLI-2018",
    name: "Cyclone Titli",
    year: 2018,
    basin: "North Indian Ocean (Bay of Bengal)",
    similarityScore: 73.8,
    trackSimilarity: 75.1,
    peakIntensityKts: 80,
    landfallDistrict: "Palasa, Srikakulam (AP)",
    summary: "Very severe storm making landfall near the Andhra-Odisha border, leading to torrential inland river floods and severe agrarian devastation.",
    trackCoordinates: [
      { lat: 12.0, lng: 90.0 },
      { lat: 14.2, lng: 87.5 },
      { lat: 16.1, lng: 85.8 },
      { lat: 17.5, lng: 84.8 },
      { lat: 18.8, lng: 84.4 }
    ]
  }
];
