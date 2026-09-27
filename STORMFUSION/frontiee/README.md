# CYCLONE AI: Multi-Source Cyclone Intelligence & Early Warning System

**Smart India Hackathon (SIH 2026)**  
**Problem Statement ID:** SIH26070  
**Theme:** Disaster Management  
**Problem Statement:** *"An Artificial Intelligence (AI) / Machine Learning (ML) based system for identification, classification, and prediction of different tropical cyclone patterns using multi-source satellite data."*

---

## 🌪 Executive Summary

**CYCLONE AI** is a state-of-the-art Meteorological Command & Control GIS platform engineered to identify, classify, monitor, and forecast tropical cyclones in the North Indian Ocean basin (Bay of Bengal and Arabian Sea). 

The system couples geostationary multispectral satellite radiances (**INSAT-3D / INSAT-3DR**), oceanic heat metrics (**SST & Ocean Heat Content**), atmospheric reanalysis (**ECMWF ERA5**), and historical tracks (**NOAA IBTrACS**) within a spatiotemporal deep learning pipeline (**EfficientNet-B0 + ConvLSTM with Temporal Attention**).

---

## 🚀 Key Features

1. **Military-Grade Meteorological Operations Center**:
   - High-contrast, dark-mode GIS command-center interface optimized for high-stress emergency response rooms.
   - Distinct, persistent **DEMO / SIMULATED DATA** indicators ensuring full scientific integrity and transparency.

2. **Full GIS Interactive Mapping (Leaflet)**:
   - Real-time animated cyclone eye marker with live pulse rings and pressure deficits.
   - Historical observed track (solid blue) and 24h/48h/72h forecasted trajectory (dashed red).
   - **67% Forecast Uncertainty Cone Polygon** dynamically scaling with prediction horizon.
   - Multi-layer basemap toggles: Dark GIS, Esri High-Resolution Satellite Imagery, and OpenTopo Terrain.
   - Interactive district vulnerability circles with popup metrics.

3. **Multi-Horizon AI Trajectory & Intensity Forecast**:
   - Discrete 24h, 48h, and 72h prediction cards with along-track vs. cross-track error bars.
   - Recharts-powered dual-area graph comparing observed wind speeds vs. AI predicted winds (kt) and central pressure (hPa).

4. **INSAT-3D Multispectral Radiometer Panel**:
   - Multi-channel selector: Thermal Infrared (10.8 µm), Visible Albedo (0.65 µm), Upper-Troposphere Water Vapour (6.8 µm), and Dvorak BD-Enhanced IR.
   - Frame scrubbing, loop animation, playback speed toggles, lat/lng graticules, and calibrated thermal color scales (-85°C to +28°C).

5. **Explainable AI (Grad-CAM)**:
   - Transparent, auditable model reasoning for disaster management officials.
   - ConvLSTM attention heatmaps with opacity slider, side-by-side mode, and neural feature weight attributions (eyewall symmetry, curved banding, outflow boundaries).

6. **Historical Analogue Discovery (FAISS)**:
   - High-dimensional nearest-neighbor vector search over 140+ years of IBTrACS cyclone data (e.g. Cyclone Hudhud 2014, Cyclone Phailin 2013, Cyclone Fani 2019).
   - Prominently designated: `HISTORICAL ANALOGUE — NOT A FORECAST`.

7. **District-Level Risk & Exposure Matrix**:
   - Quantitative risk indexing across coastal Andhra Pradesh, Odisha, and West Bengal.
   - Coupled analysis of peak sustained winds, 24h precipitation inundation, astronomical storm surge, and population exposure.

8. **Disaster Management Emergency Alerts**:
   - Multi-tier alert feed (`CRITICAL`, `WARNING`, `WATCH`, `ADVISORY`) with one-click officer acknowledgment and detailed protocol modals.

9. **Crowdsourced Auxiliary Ground Reports**:
   - Citizen field sensor stream explicitly separated with `AUXILIARY / UNVERIFIED INFORMATION` badges.

10. **Contextual AI Chatbot**:
    - Floating & dedicated full-page assistant strictly grounded in the structured system state (coordinates, intensity, landfall windows, safety advisories).

---

## 🛠 Technology Stack

- **Frontend**: React 18, Vite, TypeScript, Tailwind CSS, React Router v6
- **GIS & Mapping**: Leaflet, React-Leaflet, CartoDB Dark Matter, Esri World Imagery
- **Data Visualization**: Recharts, Lucide React Icons
- **Decoupled Architecture**: `src/services/api.ts` abstraction layer ready for immediate connection with:
  - Python FastAPI backend (`VITE_API_BASE_URL`)
  - PyTorch / TorchVision deep learning pipeline
  - PostgreSQL / PostGIS geospatial database
  - FAISS vector search engine

---

## 📂 Project Structure

```
├── .env.example               # Environment variables example
├── index.html                 # HTML entrypoint with Leaflet CSS & fonts
├── package.json               # Dependencies & scripts
├── postcss.config.js          # PostCSS configuration
├── tailwind.config.js         # Custom command-center theme & radar animations
├── tsconfig.json              # TypeScript configuration
├── vite.config.ts             # Vite bundler configuration
└── src/
    ├── App.tsx                # Layout wrapper with responsive navigation & routes
    ├── index.css              # Glassmorphism, radar pulse, and map styles
    ├── main.tsx               # Root React entrypoint
    ├── vite-env.d.ts          # Vite environment types
    ├── components/
    │   ├── AlertPanel.tsx          # Emergency alert dispatch feed
    │   ├── CycloneAIChatbot.tsx    # Contextual conversational AI assistant
    │   ├── CycloneMap.tsx          # Leaflet GIS interactive map
    │   ├── CycloneStatusCard.tsx   # Primary storm telemetry HUD
    │   ├── DemoModeBadge.tsx       # Scientific disclaimer & simulation badges
    │   ├── ForecastCard.tsx        # 24h/48h/72h prediction cards
    │   ├── ForecastTimeline.tsx    # Recharts wind & pressure curves
    │   ├── GradCAMViewer.tsx       # Explainable AI (XAI) attention visualizer
    │   ├── HistoricalAnalogue.tsx  # FAISS nearest-neighbor comparison
    │   ├── Navbar.tsx              # Top operational status bar & navigation
    │   ├── RiskPanel.tsx           # District vulnerability exposure matrix
    │   ├── SatelliteViewer.tsx     # INSAT-3D multispectral viewer
    │   ├── Sidebar.tsx             # Collapsible command-center rail
    │   └── SocialReports.tsx       # Auxiliary crowdsourced observations
    ├── data/
    │   ├── mockAlerts.ts           # Emergency bulletins
    │   ├── mockCycloneData.ts      # Active cyclone telemetry & soundings
    │   ├── mockForecast.ts         # Multi-step track, intensity, & cone coords
    │   ├── mockHistoricalAnalogues.ts # IBTrACS analogues
    │   ├── mockRiskZones.ts        # Coastal district exposure parameters
    │   ├── mockSatellite.ts        # Multispectral frames & Grad-CAM weights
    │   └── mockSocialReports.ts    # Auxiliary field sensor notes
    ├── pages/
    │   ├── AboutPage.tsx           # Architecture, AI pipeline, & SIH docs
    │   ├── AlertCenterPage.tsx     # Full emergency dispatch console
    │   ├── AssistantPage.tsx       # Dedicated full-page AI chatbot
    │   ├── Dashboard.tsx           # Main Command Center operations portal
    │   ├── ForecastPage.tsx        # Detailed track & landfall forecasting
    │   ├── HistoricalPage.tsx      # FAISS analogue exploration & comparison
    │   ├── LiveMonitoring.tsx      # Satellite feeds & atmospheric soundings
    │   └── RiskMapPage.tsx         # Full-screen GIS district exposure map
    ├── services/
    │   └── api.ts                  # Decoupled FastAPI-ready service layer
    ├── types/
    │   ├── cyclone.ts              # Core domain TypeScript models
    │   └── index.ts                # Re-exports
    └── utils/
        └── formatters.ts           # Meteorological and GIS formatting helpers
```

---

## ⚡ Installation & Local Run

### Prerequisites
- Node.js (v18 or higher; tested on Node v24)
- npm (v9 or higher)

### Setup Steps
1. **Clone or Navigate to the directory**:
   ```bash
   cd sih
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Configure Environment Variables (Optional)**:
   ```bash
   cp .env.example .env
   ```

4. **Start Development Server**:
   ```bash
   npm run dev
   ```
   Open your browser at `http://localhost:5173`.

5. **Production Build & Preview**:
   ```bash
   npm run build
   npm run preview
   ```

---

## 🔬 Scientific Credibility & Ethics

- **Simulation Mode Disclosure**: All current cyclone data, tracks, Grad-CAM heatmaps, and district damage indices are realistic synthetic demonstrations for SIH 2026.
- **Unverified Data Isolation**: Social media and crowdsourced ground observations are strictly sandboxed as auxiliary intelligence and cannot alter numerical AI model predictions.
- **Analogue Separation**: Historical analogue matches are explicitly demarcated as reference case studies rather than active forecasts.

---

## 👥 Smart India Hackathon 2026 Team
- **Problem Statement ID:** SIH26070
- **Theme:** Disaster Management
- **Organization:** Ministry of Earth Sciences / IMD / NDMA
