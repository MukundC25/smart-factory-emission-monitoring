from fastapi import APIRouter, HTTPException
from typing import Optional

from ..services.recommendation_service import (
    get_recommendation,
    load_recommendations,
)

router = APIRouter(prefix="/recommendation", tags=["Recommendations"])


@router.get("/")
def get_recommendations(city: Optional[str] = None, limit: int = 300):
    items = load_recommendations(city=city, limit=limit)
    return {"count": len(items), "items": items}

@router.get("/{factory_id}")
def get_recommendation_by_factory(factory_id: str):
    item = get_recommendation(factory_id)
    if not item:
        raise HTTPException(status_code=404, detail="Factory recommendation not found")
    return item
