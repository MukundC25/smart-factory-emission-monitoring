from fastapi import FastAPI
from .routes import factories, pollution, recommendation

app = FastAPI(
    title="Smart Factory Emission Monitoring API",
    version="1.0"
)

app.include_router(factories.router)
app.include_router(pollution.router)
app.include_router(recommendation.router)

@app.get("/")
def root():
    return {"message": "Smart Factory Emission Monitoring System"}