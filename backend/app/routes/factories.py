from fastapi import APIRouter

router = APIRouter(prefix="/factories", tags=["Factories"])

@router.get("/")
def get_factories():
    return {"data": "List of factories"}
