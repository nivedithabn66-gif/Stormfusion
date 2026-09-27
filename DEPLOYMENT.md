# STORMFUSION Free Cloud Deployment Guide

This guide provides step-by-step instructions to deploy the **STORMFUSION** tropical cyclone AI system entirely for free using **Render** (FastAPI Backend) and **Vercel** (React + Vite Frontend).

---

## Architecture Overview

```
                          ┌────────────────────────────┐
                          │   Vercel (Hobby Free)      │
                          │   React + Vite Frontend    │
                          │   https://<app>.vercel.app │
                          └─────────────┬──────────────┘
                                        │
                               HTTPS REST / WSS
                                        │
                          ┌─────────────▼──────────────┐
                          │   Render (Free Tier)       │
                          │   FastAPI Python Backend   │
                          │   https://<api>.render.com │
                          └─────────────┬──────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
       [STORMFUSION ResNet-ConvLSTM]          [SQLite / Optional PostgreSQL]
```

---

## 1. Deploying the FastAPI Backend on Render

Render provides a free Web Service tier suitable for running Python FastAPI applications.

### Step 1: Create a Web Service
1. Log in to [Render](https://dashboard.render.com/).
2. Click **New +** → **Web Service**.
3. Connect your GitHub repository: `https://github.com/nivedithabn66-gif/Stormfusion.git`.

### Step 2: Configure Service Settings
Fill in the deployment configuration:

| Field | Value | Notes |
| :--- | :--- | :--- |
| **Name** | `stormfusion-api` | Or any unique name of your choice |
| **Region** | Singapore / Frankfurt / Oregon | Choose the closest region to your users |
| **Branch** | `main` | Production branch |
| **Root Directory** | `STORMFUSION` | **Crucial:** Sets working directory to backend |
| **Runtime** | `Python 3` | |
| **Build Command** | `pip install -r requirements.txt` | Installs dependencies |
| **Start Command** | `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT` | Production ASGI start command |
| **Instance Type** | `Free` (0.1 CPU, 512 MB RAM) | Free tier |

> [!TIP]
> **Faster Builds on Render (CPU-only PyTorch)**:
> To optimize build times and disk usage on Render's 512MB free tier, you can use:
> ```bash
> pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision && pip install -r requirements.txt
> ```

### Step 3: Configure Environment Variables in Render
In the **Environment** tab of your Render service, add the following variables:

| Variable Name | Example / Default Value | Purpose |
| :--- | :--- | :--- |
| `CORS_ORIGINS` | `*` or `https://<your-app>.vercel.app,http://localhost:5173` | Allowed frontend domains for CORS |
| `PYTHONUNBUFFERED` | `1` | Ensures immediate stdout log flushing |
| `DATABASE_URL` | *(Leave blank for local SQLite fallback)* | Optional PostgreSQL connection string |
| `EUMETSAT_CONSUMER_KEY` | *(Optional)* | EUMETSAT API key for satellite data |
| `EUMETSAT_CONSUMER_SECRET`| *(Optional)* | EUMETSAT API secret |
| `CDS_API_KEY` | *(Optional)* | Copernicus ERA5 access token |

### Step 4: Verify Backend Deployment
Once Render displays `Live`, verify the endpoints:
- **Root Status**: `https://<your-render-app>.onrender.com/`
- **Health Check**: `https://<your-render-app>.onrender.com/health` (Returns `{"status": "ok", "service": "STORMFUSION API", ...}`)
- **Interactive Swagger Docs**: `https://<your-render-app>.onrender.com/docs`
- **Active Cyclone API**: `https://<your-render-app>.onrender.com/api/cyclones/current`

---

## 2. Deploying the React/Vite Frontend on Vercel

Vercel provides free global edge hosting for static and Single Page Applications (SPA).

### Step 1: Import Project into Vercel
1. Log in to [Vercel](https://vercel.com/).
2. Click **Add New...** → **Project**.
3. Select your GitHub repository (`Stormfusion`).

### Step 2: Configure Project Settings
In the **Configure Project** screen:

| Setting | Value | Notes |
| :--- | :--- | :--- |
| **Framework Preset** | `Vite` | Auto-detected |
| **Root Directory** | `STORMFUSION/frontiee` | Click **Edit** and select `STORMFUSION/frontiee` |
| **Build Command** | `npm run build` | Runs `tsc && vite build` |
| **Output Directory** | `dist` | Generated build folder |
| **Install Command** | `npm install` | |

### Step 3: Configure Environment Variables in Vercel
Expand **Environment Variables** and add:

| Variable Name | Value | Description |
| :--- | :--- | :--- |
| `VITE_API_BASE_URL` | `https://<your-render-app>.onrender.com/api` | Backend API URL (include `/api` suffix) |
| `VITE_APP_MODE` | `live` | Enables live API fetching from backend |
| `VITE_MAPBOX_TOKEN` | *(Optional)* | Optional GIS map tile token |

### Step 4: Deploy & Test
1. Click **Deploy**.
2. Once the build succeeds, click your assigned domain (e.g., `https://stormfusion.vercel.app`).
3. Verify that all tabs (Dashboard, Live Monitoring, Multi-Horizon Forecast, Risk Assessment, Explainable AI) load smoothly.

> [!NOTE]
> Single Page Application routing (preventing 404s on browser reload of `/forecast`, `/live`, etc.) is handled automatically by the included `vercel.json` rewrite configuration.

---

## 3. Important Free-Tier Operational Considerations

1. **Render Free Tier Cold Starts**:
   - Free Web Services on Render automatically spin down ("sleep") after 15 minutes of inactivity.
   - When a new request arrives, it takes ~30–50 seconds to spin back up.
   - The STORMFUSION frontend is designed with client-side latency simulation and resilient fallback caching, ensuring the UI remains usable even while the backend wakes up.

2. **Memory Limit (512 MB RAM)**:
   - PyTorch CPU inference runs efficiently within ~180MB RAM.
   - Large raw datasets (`*.nat`, `*.nc`) and heavy training checkpoints (`*.pt`) are gitignored and not required for frontend/API inference.

3. **Updating CORS After Frontend Deployment**:
   - Once your Vercel URL is known (e.g., `https://stormfusion-xyz.vercel.app`), update `CORS_ORIGINS` in your Render Environment dashboard to restrict access if desired, or leave it as `*`.

---

## 4. Local Development Verification

### Running the Backend Locally
```bash
# From workspace root or STORMFUSION directory
cd STORMFUSION
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Test health check:
```bash
curl http://127.0.0.1:8000/health
```

### Running the Frontend Locally
```bash
cd STORMFUSION/frontiee
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 5. Summary of Required Environment Variable Names

### Backend (Render)
- `PORT` (Provided automatically by Render)
- `HOST` (`0.0.0.0`)
- `CORS_ORIGINS`
- `DATABASE_URL` (Optional)
- `EUMETSAT_CONSUMER_KEY` (Optional)
- `EUMETSAT_CONSUMER_SECRET` (Optional)
- `CDS_API_KEY` (Optional)
- `CDS_API_URL` (Optional)

### Frontend (Vercel)
- `VITE_API_BASE_URL` (Points to Render backend URL + `/api`)
- `VITE_APP_MODE` (`live` or `demo`)
- `VITE_MAPBOX_TOKEN` (Optional)
