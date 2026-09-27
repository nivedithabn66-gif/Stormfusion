# STORMFUSION: Complete Technology Stack & Architecture Reference

**Smart India Hackathon 2026 — Problem Statement: SIH26070**  
*AI/ML System for Tropical Cyclone Identification, Classification, and Prediction in the North Indian Ocean (Bay of Bengal & Arabian Sea)*

---

## 🏗️ Architectural Overview

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                       REACT 18 + TYPESCRIPT FRONTEND                   │
│      (Leaflet GIS • Recharts • TailwindCSS • Command Center UI)         │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │  HTTP / REST (Proxy on :5173 -> :8000)
┌────────────────────────────────────▼────────────────────────────────────┐
│                        FASTAPI ASYNC BACKEND                            │
│      (Pydantic v2 • Uvicorn • Singleton Multi-Modal Inference Service)  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │  PyTorch Tensor Pipeline
┌────────────────────────────────────▼────────────────────────────────────┐
│                     STORMFUSION DEEP LEARNING CORE                      │
│   ResNet-18 (Satellite) + 2D-CNN (ERA5) + Bi-GRU (Track)               │
│   ↳ Gated Cross-Modal Fusion ↳ ConvLSTM ↳ Multi-Task Heads              │
│   ↳ Monte Carlo Dropout UQ (N=20) ↳ Grad-CAM Spatial Attribution        │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │  Remote Sensing Ingestors
┌────────────────────────────────────▼────────────────────────────────────┐
│                  METEOROLOGICAL & SATELLITE DATA LAYER                  │
│   EUMETSAT Meteosat-9 IODC (SEVIRI) • ECMWF ERA5 • NOAA IBTrACS         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. 🧠 Deep Learning & Machine Learning Core

| Component | Technology / Architecture | Description |
| :--- | :--- | :--- |
| **Deep Learning Framework** | `PyTorch 2.x` | Core neural network construction, automatic differentiation, and GPU/CPU inference execution |
| **Satellite Spatial Encoder** | `ResNet-18` (Custom 3-Channel) | Extracts spatial cloud-top convective features from SEVIRI IR 10.8µm, WV 6.2µm, and IR 12.0µm |
| **Atmospheric Reanalysis Encoder** | `2D-CNN (4-Channel)` | Ingests ECMWF ERA5 atmospheric fields ($Z_{500}, U_{850}, V_{850}, T_{200}$) |
| **Temporal Track History Encoder** | `Bidirectional GRU / LSTM` | Encodes 8-step past 3-hourly storm motion vectors, wind intensity, and pressure trends |
| **Fusion Mechanism** | `Gated Cross-Modal Fusion` | Learned attention gating dynamically weighting available modalities based on atmospheric conditions |
| **Spatiotemporal Aggregator** | `ConvLSTM` | Captures combined spatial-temporal storm evolution over sequential frames |
| **Multi-Task Output Heads** | `PyTorch Custom Heads` | 5 simultaneous multi-horizon forecast heads: <br>• **Binary Detection Head**: Sigmoid probability of active cyclogenesis<br>• **Intensity Head**: +3h, +6h, +12h, +24h max sustained surface winds (kt)<br>• **Central Pressure Head**: +3h, +6h, +12h, +24h central pressure ($h\text{Pa}$)<br>• **Track Delta Head**: Trajectory displacement ($\Delta\text{Lat}, \Delta\text{Lon}$)<br>• **Pattern Head**: Dvorak cyclone banding & eye pattern classification |
| **Uncertainty Quantification (UQ)** | `Monte Carlo Dropout (N=20)` | Epistemic uncertainty estimation, predictive variance, entropy, and 80%/90% confidence intervals |
| **Explainable AI (XAI)** | `Grad-CAM` | Gradient-weighted spatial attribution computed on ResNet-18 Layer 4 feature maps |

---

## 2. 🛰️ Remote Sensing & Scientific Data Processing

| Component | Library / Data Source | Purpose |
| :--- | :--- | :--- |
| **Satellite Ingestion** | `Satpy`, `pyresample`, `GDAL` | Native Level 1.5 decoding of EUMETSAT Meteosat-9 SEVIRI Indian Ocean Data Coverage (IODC) at 45.5°E |
| **Spatial & Raster Utilities** | `Rasterio`, `xarray`, `NetCDF4` | Atmospheric grid manipulation, geo-referencing, and bounding box cropping |
| **Historical Cyclone Dataset** | `NOAA IBTrACS v04` & `IMD RSMC` | Ground truth tracks, central pressures, and wind speeds for the North Indian Ocean basin |
| **Numerical Processing** | `NumPy`, `SciPy`, `Pandas` | Matrix operations, data normalization, physical Kelvin [190K–320K] QC, and pressure calibrations |

---

## 3. 🗄️ Spatial Database: PostgreSQL + PostGIS

The system utilizes **PostgreSQL** with the **PostGIS spatial extension** (`backend/database/schema.sql` and `backend/database/models.py`) for geospatial indexing, trajectory cone intersection, and high-performance querying:

| Component | Technology | Role & Spatial Functionality |
| :--- | :--- | :--- |
| **Relational Database** | `PostgreSQL 15+` | Primary relational database for cyclone catalogs, sensor telemetry, and forecast historical logs |
| **Spatial Engine** | `PostGIS 3+` | Geospatial extension for WGS-84 coordinate points, trajectory lines, and risk polygon geometry |
| **Spatial Coordinates** | `GEOMETRY(Point, 4326)` | Stores cyclone eye positions and automated observation locations |
| **Trajectory & Cones** | `GEOMETRY(Polygon, 4326)` | Stores spatial polygons of the 80%/90% track uncertainty cones |
| **Vulnerability Boundaries** | `GEOMETRY(MultiPolygon, 4326)` | District administrative boundaries for spatial containment (`ST_Contains`, `ST_Intersects`) |
| **Spatial Indexing** | `GiST Indexing` | Sub-millisecond GIS spatial queries for coastal proximity calculations (`ST_DWithin`, `ST_Distance`) |
| **Python ORM** | `SQLAlchemy 2.x` & `GeoAlchemy2` | Object-Relational Mapping with connection pooling and automated session management |

---

## 4. ⚡ Backend & API Infrastructure

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Language & Runtime** | `Python 3.12+` (64-bit Virtualenv) | Backend execution environment |
| **Web Framework** | `FastAPI` | High-performance asynchronous REST API framework |
| **ASGI Server** | `Uvicorn` | Production-grade ASGI server handling asynchronous concurrency |
| **Data Validation & Schemas** | `Pydantic v2` | Strict typing and validation for API request payloads and multi-horizon forecast schemas |
| **CORS & Middleware** | `fastapi.middleware.cors` | Cross-Origin Resource Sharing for seamless frontend-backend communication |
| **Static File Serving** | `fastapi.staticfiles` | Embedded hosting of standalone scientific dashboard |
| **Inference Architecture** | `Singleton Service Pattern` | Model weights loaded once into memory (`backend/app/service.py`) for low-latency (<50ms) inference |

---

## 5. 💻 Frontend Command Center & User Interface

| Component | Technology | Role |
| :--- | :--- | :--- |
| **UI Framework** | `React 18.3` | Reactive, component-based user interface |
| **Language** | `TypeScript 5.5` | Strict static typing for telemetry interfaces, API schemas, and forecast data structures |
| **Build Tool & Dev Server** | `Vite 5.4` | Fast HMR (Hot Module Replacement) and optimized production chunking with dev server proxy |
| **CSS & Design System** | `TailwindCSS 3.4` + `PostCSS` | Responsive command-center design, dark mode, glassmorphism, glowing telemetry badges |
| **Interactive GIS Mapping** | `Leaflet 1.9` & `React-Leaflet 4.2` | Interactive Bay of Bengal GIS map with animated eye markers, forecast cones, and danger zones |
| **Data Visualization & Charts** | `Recharts 2.12` | Time-series wind trends, multi-horizon error bounds, and historical analogue comparisons |
| **Iconography** | `Lucide React` | Command center and meteorological icons (radar, wind, pressure, shield, satellite) |
| **CSS Class Utilities** | `clsx`, `tailwind-merge` | Dynamic conditional styling and class resolution |
| **Routing** | `React Router DOM 6.26` | Multi-page SPA navigation (`/`, `/live`, `/forecast`, `/risk`, `/history`, `/alerts`, `/assistant`, `/about`) |

---

## 6. 🧪 Testing, Quality Assurance & Automation

| Component | Tool / Methodology | Verification Coverage |
| :--- | :--- | :--- |
| **Unit & Regression Testing** | Python `unittest` | 44 automated tests covering model architecture, loss functions, UQ runners, and data loaders |
| **Integration Testing** | Custom HTTP Client (`scripts/test_frontend_backend_integration.py`) | Automated 15-endpoint verification across all live REST routes |
| **Reproducibility** | Custom Seeding Engine (`training/reproducibility.py`) | Deterministic seeds across PyTorch, NumPy, and Python standard libraries |
| **Scientific Validation** | Empirical Prior Head Initialization & CLIPER Baseline | Strict benchmark testing against Persistence baselines to evaluate physical forecasting skill |

---

## 🚀 How to Run & Preview the Project

### 1. Start the Backend Server (Port 8000)
```powershell
# From the repository root:
& "c:\Users\Niveditha B N\OneDrive\Documents\SIH - 2026\.venv\Scripts\python.exe" -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Start the Frontend Command Center (Port 5173)
```powershell
# In a second terminal:
cd Frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

### 3. Verify System Health & Integration
```powershell
& "c:\Users\Niveditha B N\OneDrive\Documents\SIH - 2026\.venv\Scripts\python.exe" scripts/test_frontend_backend_integration.py
```

### 🌐 Live Endpoints
- **React Frontend**: `http://localhost:5173/`
- **FastAPI Interactive Swagger Docs**: `http://localhost:8000/docs`
- **Static Multi-Modal AI Dashboard**: `http://localhost:8000/dashboard`
