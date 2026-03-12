from fastapi import APIRouter

router = APIRouter(prefix="/pollution", tags=["Pollution"])

@router.get("/")
def get_pollution_data():
    return {"data": "Pollution metrics"}
