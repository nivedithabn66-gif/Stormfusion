import { GroundReport } from '../types/cyclone';

export const mockSocialReports: GroundReport[] = [
  {
    id: "SOC-REP-01",
    location: "Visakhapatnam Coast (RK Beach)",
    state: "Andhra Pradesh",
    coordinates: [17.7126, 83.3195],
    content: "Massive waves breaching barricades near submarine museum road. Strong gusty winds causing tree branches to snap. Fishermen shifting fiber catamarans inland.",
    source: "Social Media",
    status: "UNVERIFIED",
    timestamp: "10 mins ago (14:22 IST)",
    upvotes: 42,
    verifiedOfficial: false
  },
  {
    id: "SOC-REP-02",
    location: "Kalingapatnam Lighthouse",
    state: "Andhra Pradesh",
    coordinates: [18.3391, 84.1287],
    content: "Sea water ingress observed up to 100 meters beyond usual high tide line. Heavy overcast skies with intermittent torrential squalls.",
    source: "Volunteer Weather Watcher",
    status: "CORROBORATED",
    timestamp: "24 mins ago (14:08 IST)",
    upvotes: 28,
    verifiedOfficial: false
  },
  {
    id: "SOC-REP-03",
    location: "Gopalpur-on-Sea",
    state: "Odisha",
    coordinates: [19.2608, 84.9083],
    content: "Wind speeds picking up steadily since 1:00 PM. Local sirens sounded by district administration for coastal hamlet evacuation.",
    source: "Local Ham Radio",
    status: "INVESTIGATING",
    timestamp: "45 mins ago (13:47 IST)",
    upvotes: 61,
    verifiedOfficial: false
  },
  {
    id: "SOC-REP-04",
    location: "Kakinada Anchorage",
    state: "Andhra Pradesh",
    coordinates: [16.9891, 82.2475],
    content: "Lighterage operations completely halted at Kakinada deep-water port. Sea swell estimated above 3.5 meters.",
    source: "Crowdsourced App",
    status: "CORROBORATED",
    timestamp: "1 hour ago (13:30 IST)",
    upvotes: 19,
    verifiedOfficial: false
  }
];
