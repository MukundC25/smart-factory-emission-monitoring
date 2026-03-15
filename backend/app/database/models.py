"""Database models for factories, pollution, and recommendations."""

from sqlalchemy import Column, Float, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.sql import func

from .db import Base


class Factory(Base):
    """Factory model."""
    __tablename__ = "factories"

    id = Column(Integer, primary_key=True, index=True)
    factory_id = Column(String, unique=True, index=True, nullable=False)
    factory_name = Column(String, nullable=False)
    industry_type = Column(String, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    city = Column(String, nullable=False)
    state = Column(String)
    country = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class PollutionReading(Base):
    """Pollution reading model."""
    __tablename__ = "pollution_readings"

    id = Column(Integer, primary_key=True, index=True)
    station_name = Column(String, nullable=False)
    station_lat = Column(Float, nullable=False)
    station_lon = Column(Float, nullable=False)
    city = Column(String, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    pm25 = Column(Float)
    pm10 = Column(Float)
    co = Column(Float)
    no2 = Column(Float)
    so2 = Column(Float)
    o3 = Column(Float)
    aqi_index = Column(Float)
    source = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Recommendation(Base):
    """Recommendation model."""
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, index=True)
    factory_id = Column(String, ForeignKey("factories.factory_id"), nullable=False)
    risk_level = Column(String, nullable=False)
    recommendation_text = Column(Text, nullable=False)
    predicted_score = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())