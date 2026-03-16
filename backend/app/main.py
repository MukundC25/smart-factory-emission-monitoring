from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import factories, pollution, recommendation

app = FastAPI(
    title="Smart Factory Emission Monitoring API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(factories.router)
app.include_router(pollution.router)
app.include_router(recommendation.router)

@app.get("/")
def root():
    return {"message": "Smart Factory Emission Monitoring System"}