"""SQLAlchemy models for PostgreSQL + PostGIS spatial data."""

from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, Text, ForeignKey, Numeric
from sqlalchemy.orm import relationship
from backend.database.database import Base


class CycloneDB(Base):
    __tablename__ = "cyclones"

    storm_id = Column(String(64), primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    basin = Column(String(100), default="North Indian Ocean")
    sub_basin = Column(String(50), default="Bay of Bengal")
    status = Column(String(50), default="ACTIVE")
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    peak_intensity_kts = Column(Float, nullable=True)
    min_pressure_hpa = Column(Float, nullable=True)

    tracks = relationship("TrackPointDB", back_populates="cyclone", cascade="all, delete-orphan")
    forecasts = relationship("AIForecastDB", back_populates="cyclone", cascade="all, delete-orphan")


class TrackPointDB(Base):
    __tablename__ = "track_points"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    storm_id = Column(String(64), ForeignKey("cyclones.storm_id"), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    wind_speed_kts = Column(Float, nullable=False)
    central_pressure_hpa = Column(Float, nullable=True)
    category = Column(String(100), nullable=False)
    movement_speed_kmh = Column(Float, default=14.0)
    movement_direction = Column(String(10), default="NW")
    is_forecast = Column(Boolean, default=False)
    source = Column(String(50), default="IBTrACS/IMD")

    cyclone = relationship("CycloneDB", back_populates="tracks")


class AIForecastDB(Base):
    __tablename__ = "ai_forecasts"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    storm_id = Column(String(64), ForeignKey("cyclones.storm_id"), nullable=False, index=True)
    reference_timestamp = Column(DateTime, nullable=False)
    forecast_horizon_hours = Column(Integer, nullable=False)  # 3, 6, 12, 24, 48, 72
    target_timestamp = Column(DateTime, nullable=False)
    pred_latitude = Column(Float, nullable=False)
    pred_longitude = Column(Float, nullable=False)
    pred_intensity_kts = Column(Float, nullable=False)
    intensity_ci_lower_kts = Column(Float, nullable=True)
    intensity_ci_upper_kts = Column(Float, nullable=True)
    pred_pressure_hpa = Column(Float, nullable=True)
    confidence_percent = Column(Float, default=85.0)
    model_version = Column(String(50), default="STORMFUSION-v1.0")

    cyclone = relationship("CycloneDB", back_populates="forecasts")


class DistrictRiskDB(Base):
    __tablename__ = "district_risks"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    district_name = Column(String(100), nullable=False, index=True)
    state_name = Column(String(100), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)  # 'LOW', 'MODERATE', 'HIGH', 'CRITICAL'
    wind_risk_kts = Column(Float, default=0.0)
    estimated_rainfall_mm = Column(Float, default=0.0)
    storm_surge_meters = Column(Float, default=0.0)
    population_exposed = Column(String(50), default="0")
    evacuation_recommended = Column(Boolean, default=False)


class GroundReportDB(Base):
    __tablename__ = "ground_reports"

    id = Column(String(64), primary_key=True, index=True)
    location_name = Column(String(200), nullable=False)
    state_name = Column(String(100), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    content = Column(Text, nullable=False)
    source = Column(String(100), default="Crowdsourced App")
    status = Column(String(50), default="INVESTIGATING")
    upvotes = Column(Integer, default=1)
    verified_official = Column(Boolean, default=False)
    submitted_at = Column(DateTime, default=datetime.utcnow)
