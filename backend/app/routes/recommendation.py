from fastapi import APIRouter

router = APIRouter(prefix="/recommendation", tags=["Recommendations"])

@router.get("/{factory_id}")
def get_recommendation(factory_id: int):
    return {"factory_id": factory_id, "recommendation": "Example recommendation"}
