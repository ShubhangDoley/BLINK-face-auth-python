"""
BLINK — Cloud Database Connection Manager
Handles PostgreSQL connection via SQLAlchemy, with an automatic
local SQLite fallback ('cloud_blink.db') for offline sandbox testing.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base

# Retrieve DATABASE_URL from environment variables (e.g. RDS connection string).
# Fallback to local SQLite file for offline sandbox development.
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "sqlite:///cloud_blink.db"
)

print(f"[DB] Initializing Cloud DB Engine on: {DATABASE_URL}")

# SQLite requires different connection parameters for multi-threaded uvicorn
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Bootstrap the relational database tables."""
    Base.metadata.create_all(bind=engine)
    print("[DB] PostgreSQL/SQLite tables verified and instantiated.")

def get_db():
    """FastAPI database session context provider."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
