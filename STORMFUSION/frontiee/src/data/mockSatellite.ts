import { GradCamAnalysis, SatelliteFrame } from '../types/cyclone';

export const mockSatelliteFrames: SatelliteFrame[] = [
  {
    id: "SAT-20260907-1430-IR",
    timestamp: "07 Sep 2026 — 14:32 IST (T-0h)",
    channel: "IR",
    satelliteName: "INSAT-3D",
    imageUrl: "/satellite/cyclone_ir_current.png",
    description: "Thermal Infrared (10.8 µm) showing cold cloud tops (-78°C) concentrated symmetrically around nascent eye feature.",
    colorScale: "Enhanced BD-Curve (Black/Cyan/Red/Yellow)"
  },
  {
    id: "SAT-20260907-1400-IR",
    timestamp: "07 Sep 2026 — 14:00 IST (T-0.5h)",
    channel: "IR",
    satelliteName: "INSAT-3D",
    imageUrl: "/satellite/cyclone_ir_frame2.png",
    description: "Deep convective curved banding wrapping 0.85 cycles into the low-level circulation center.",
    colorScale: "Enhanced BD-Curve"
  },
  {
    id: "SAT-20260907-1330-IR",
    timestamp: "07 Sep 2026 — 13:30 IST (T-1.0h)",
    channel: "IR",
    satelliteName: "INSAT-3D",
    imageUrl: "/satellite/cyclone_ir_frame3.png",
    description: "Expansion of outer feeder bands along the northeastern sector toward Myanmar coastline.",
    colorScale: "Enhanced BD-Curve"
  },
  {
    id: "SAT-20260907-1300-IR",
    timestamp: "07 Sep 2026 — 13:00 IST (T-1.5h)",
    channel: "IR",
    satelliteName: "INSAT-3D",
    imageUrl: "/satellite/cyclone_ir_frame4.png",
    description: "Initial consolidation of convective core following rapid intensification pulse.",
    colorScale: "Enhanced BD-Curve"
  },
  {
    id: "SAT-20260907-1430-VIS",
    timestamp: "07 Sep 2026 — 14:32 IST (T-0h)",
    channel: "VIS",
    satelliteName: "INSAT-3D",
    imageUrl: "/satellite/cyclone_vis_current.png",
    description: "Visible Channel (0.65 µm) capturing high-resolution cloud top texture, transverse cirrus banding, and shadow definition.",
    colorScale: "Monochrome Greyscale Albedo"
  },
  {
    id: "SAT-20260907-1430-WV",
    timestamp: "07 Sep 2026 — 14:32 IST (T-0h)",
    channel: "WV",
    satelliteName: "INSAT-3D",
    imageUrl: "/satellite/cyclone_wv_current.png",
    description: "Upper-tropospheric Water Vapour (6.8 µm) demonstrating dry air intrusion suppression and vigorous upper-level divergence.",
    colorScale: "Water Vapour Dynamic Rainbow"
  },
  {
    id: "SAT-20260907-1430-ENHANCED",
    timestamp: "07 Sep 2026 — 14:32 IST (T-0h)",
    channel: "ENHANCED_IR",
    satelliteName: "INSAT-3DR",
    imageUrl: "/satellite/cyclone_enhanced_current.png",
    description: "Dvorak Enhanced Infrared imagery with calibrated temperature thresholds for Dvorak T-Number estimation (T=4.5).",
    colorScale: "Calibrated Dvorak Step Table"
  }
];

export const mockGradCamAnalysis: GradCamAnalysis = {
  modelArchitecture: "EfficientNet-B0 + ConvLSTM (Temporal Attention Fusion)",
  explainabilityMethod: "Grad-CAM",
  confidenceScore: 91,
  primaryAttentionZone: "Central Dense Overcast (CDO) & Eyewall Convective Core",
  description: "Model gradient activation is overwhelmingly concentrated on the symmetric eyewall ring and high-shear inner spiral band. Feature weights in the ConvLSTM layer indicate that persistent deep convective cold tops (-80°C to -72°C) over the preceding 6 hours dictated the 82kt intensity classification.",
  featuresIdentified: [
    "Developing Eye Symmetrical Wall Ring (Weight: 42%)",
    "Primary Curved Convective Banding (Weight: 29%)",
    "Upper Tropospheric Outflow Boundary (Weight: 17%)",
    "Sea Surface Temperature (SST > 30°C) Gradient (Weight: 12%)"
  ],
  heatmapVisualUrl: "/ai/gradcam_heatmap.png",
  rawSatelliteUrl: "/satellite/cyclone_ir_current.png"
};
