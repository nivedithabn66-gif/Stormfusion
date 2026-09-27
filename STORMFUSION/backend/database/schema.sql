-- =========================================================================
-- STORMFUSION: PostgreSQL + PostGIS Spatial Database Schema
-- SIH Problem Statement: SIH26070
-- =========================================================================

-- Enable PostGIS spatial extension
CREATE EXTENSION IF NOT EXISTS postgis;

-- 1. Cyclones Master Table
CREATE TABLE IF NOT EXISTS cyclones (
    storm_id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    basin VARCHAR(100) DEFAULT 'North Indian Ocean',
    sub_basin VARCHAR(50) DEFAULT 'Bay of Bengal', -- 'Bay of Bengal' or 'Arabian Sea'
    status VARCHAR(50) DEFAULT 'ACTIVE',           -- 'ACTIVE', 'WEAKENING', 'DISSIPATED'
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ,
    peak_intensity_kts NUMERIC(5, 1),
    min_pressure_hpa NUMERIC(6, 1),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Track Points & Observations Table (Spatial Points)
CREATE TABLE IF NOT EXISTS track_points (
    id BIGSERIAL PRIMARY KEY,
    storm_id VARCHAR(64) REFERENCES cyclones(storm_id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL,
    latitude NUMERIC(8, 4) NOT NULL,
    longitude NUMERIC(8, 4) NOT NULL,
    geom GEOMETRY(Point, 4326), -- WGS84 Geographic Point
    wind_speed_kts NUMERIC(5, 1) NOT NULL,
    central_pressure_hpa NUMERIC(6, 1),
    category VARCHAR(100) NOT NULL,
    movement_speed_kmh NUMERIC(5, 1),
    movement_direction VARCHAR(10),
    is_forecast BOOLEAN DEFAULT FALSE,
    source VARCHAR(50) DEFAULT 'IBTrACS/IMD',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Spatial index on track point geometries
CREATE INDEX IF NOT EXISTS idx_track_points_geom ON track_points USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_track_points_storm_time ON track_points(storm_id, timestamp);

-- 3. AI Multi-Horizon Forecasts & Uncertainty Cones Table
CREATE TABLE IF NOT EXISTS ai_forecasts (
    id BIGSERIAL PRIMARY KEY,
    storm_id VARCHAR(64) REFERENCES cyclones(storm_id) ON DELETE CASCADE,
    reference_timestamp TIMESTAMPTZ NOT NULL,
    forecast_horizon_hours INT NOT NULL, -- 3, 6, 12, 24, 48, 72
    target_timestamp TIMESTAMPTZ NOT NULL,
    pred_latitude NUMERIC(8, 4) NOT NULL,
    pred_longitude NUMERIC(8, 4) NOT NULL,
    pred_geom GEOMETRY(Point, 4326),
    uncertainty_cone GEOMETRY(Polygon, 4326), -- PostGIS Spatial Polygon of Uncertainty Cone
    pred_intensity_kts NUMERIC(5, 1) NOT NULL,
    intensity_ci_lower_kts NUMERIC(5, 1),
    intensity_ci_upper_kts NUMERIC(5, 1),
    pred_pressure_hpa NUMERIC(6, 1),
    confidence_percent NUMERIC(4, 1),
    model_version VARCHAR(50) DEFAULT 'STORMFUSION-v1.0',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Spatial index on forecast geometries and uncertainty cones
CREATE INDEX IF NOT EXISTS idx_ai_forecasts_pred_geom ON ai_forecasts USING GIST(pred_geom);
CREATE INDEX IF NOT EXISTS idx_ai_forecasts_cone ON ai_forecasts USING GIST(uncertainty_cone);

-- 4. Coastal District GIS Vulnerability & Risk Table
CREATE TABLE IF NOT EXISTS district_risks (
    id SERIAL PRIMARY KEY,
    district_name VARCHAR(100) NOT NULL,
    state_name VARCHAR(100) NOT NULL,
    boundary GEOMETRY(MultiPolygon, 4326), -- District administrative boundary polygon
    centroid GEOMETRY(Point, 4326),
    risk_level VARCHAR(20) NOT NULL,       -- 'LOW', 'MODERATE', 'HIGH', 'CRITICAL'
    wind_risk_kts NUMERIC(5, 1),
    estimated_rainfall_mm NUMERIC(6, 1),
    storm_surge_meters NUMERIC(4, 2),
    population_exposed INT,
    evacuation_recommended BOOLEAN DEFAULT FALSE,
    last_updated TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_district_risks_geom ON district_risks USING GIST(boundary);
CREATE INDEX IF NOT EXISTS idx_district_risks_centroid ON district_risks USING GIST(centroid);

-- 5. Crowdsourced Ground Reports Table
CREATE TABLE IF NOT EXISTS ground_reports (
    id VARCHAR(64) PRIMARY KEY,
    location_name VARCHAR(200) NOT NULL,
    state_name VARCHAR(100) NOT NULL,
    latitude NUMERIC(8, 4) NOT NULL,
    longitude NUMERIC(8, 4) NOT NULL,
    geom GEOMETRY(Point, 4326),
    content TEXT NOT NULL,
    source VARCHAR(100) DEFAULT 'Crowdsourced App',
    status VARCHAR(50) DEFAULT 'INVESTIGATING', -- 'UNVERIFIED', 'INVESTIGATING', 'CORROBORATED'
    upvotes INT DEFAULT 1,
    verified_official BOOLEAN DEFAULT FALSE,
    submitted_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ground_reports_geom ON ground_reports USING GIST(geom);

-- Trigger to automatically populate PostGIS geom from latitude/longitude
CREATE OR REPLACE FUNCTION update_geom_from_latlon()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.latitude IS NOT NULL AND NEW.longitude IS NOT NULL THEN
        NEW.geom = ST_SetSRID(ST_MakePoint(NEW.longitude, NEW.latitude), 4326);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER trg_track_points_geom
BEFORE INSERT OR UPDATE ON track_points
FOR EACH ROW EXECUTE FUNCTION update_geom_from_latlon();

CREATE OR REPLACE TRIGGER trg_ground_reports_geom
BEFORE INSERT OR UPDATE ON ground_reports
FOR EACH ROW EXECUTE FUNCTION update_geom_from_latlon();
