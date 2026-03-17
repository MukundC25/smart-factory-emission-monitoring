"""Database connection and session management."""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Database URL is configured via the environment to support different environments
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")

# Make database optional - only initialize if DATABASE_URL is set
if SQLALCHEMY_DATABASE_URL:
    # For file-based SQLite, use default pool (not StaticPool) to avoid thread-safety issues
    # StaticPool should only be used for in-memory SQLite in tests
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        connect_args={"check_same_thread": False} if "sqlite" in SQLALCHEMY_DATABASE_URL else {},
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
else:
    engine = None
    SessionLocal = None

Base = declarative_base()


def get_db():
    """Dependency to get DB session."""
    if SessionLocal is None:
        raise RuntimeError("Database not configured. Set DATABASE_URL environment variable.")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()