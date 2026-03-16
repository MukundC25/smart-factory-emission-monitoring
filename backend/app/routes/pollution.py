from fastapi import APIRouter
from typing import Optional

from ..services.recommendation_service import load_pollution_readings

router = APIRouter(prefix="/pollution", tags=["Pollution"])

@router.get("/")
def get_pollution_data(city: Optional[str] = None, limit: int = 500):
    items = load_pollution_readings(city=city, limit=limit)
    return {"count": len(items), "items": items}
