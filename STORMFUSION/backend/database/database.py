"""PostgreSQL + PostGIS Database Connector & Session Manager.

Supports:
1. Production PostgreSQL + PostGIS connection via DATABASE_URL
2. Local development SQLite fallback if PostgreSQL server is not active
"""

import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/stormfusion"
)

# Normalize postgres:// to postgresql:// for SQLAlchemy 1.4+ and 2.0+ compatibility
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Detect if PostgreSQL or SQLite fallback
try:
    if DATABASE_URL.startswith("postgresql"):
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    else:
        engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
except Exception:
    # SQLite fallback
    sqlite_path = Path(__file__).resolve().parent.parent.parent / "data" / "stormfusion.db"
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{sqlite_path}", connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency to yield database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
