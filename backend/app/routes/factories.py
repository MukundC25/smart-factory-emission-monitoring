from fastapi import APIRouter
from typing import Optional

from ..services.recommendation_service import load_factory_catalog

router = APIRouter(prefix="/factories", tags=["Factories"])

@router.get("/")
def get_factories(city: Optional[str] = None, limit: int = 300):
    items = load_factory_catalog(city=city, limit=limit)
    return {"count": len(items), "items": items}
