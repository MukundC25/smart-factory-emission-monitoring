import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from .routes import factories, pollution, recommendation
from .schemas import HealthCheckResponse, ModelMetrics
from .services.ml_service import get_ml_service

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smart Factory Emission Monitoring API",
    version="1.0",
    description="ML-powered API for factory pollution impact prediction and risk assessment",
)

# Enable CORS
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

# Include routers
app.include_router(factories.router)
app.include_router(pollution.router)
app.include_router(recommendation.router)


@app.get("/")
def root():
    """API root endpoint."""
    return {
        "service": "Smart Factory Emission Monitoring System",
        "version": "1.0",
        "status": "operational",
        "endpoints": {
            "factories": "/factories",
            "pollution": "/pollution",
            "recommendations": "/recommendation",
            "health": "/health",
            "model_info": "/model-info",
        },
    }


@app.get("/health", response_model=HealthCheckResponse)
def health_check():
    """Health check endpoint with model status."""
    service = get_ml_service()
    return HealthCheckResponse(
        status="healthy",
        model_loaded=service.is_ready(),
        version="1.0",
    )


@app.get("/model-info", response_model=dict)
def get_model_info():
    """Get information about the trained ML model."""
    service = get_ml_service()
    return service.get_model_info()


@app.get("/metrics", response_model=dict)
def get_model_metrics():
    """Get model performance metrics."""
    service = get_ml_service()
    model_info = service.get_model_info()

    metrics_response = {
        "model_type": model_info.get("model_type", "unknown"),
        "is_loaded": model_info.get("is_loaded", False),
        "performance_metrics": model_info.get("metrics", {}),
    }

    if model_info.get("metrics"):
        metrics_list = []
        for model_name, metric_values in model_info["metrics"].items():
            metrics_list.append(
                {
                    "model_name": model_name,
                    "rmse": metric_values.get("rmse"),
                    "mae": metric_values.get("mae"),
                    "r2": metric_values.get("r2"),
                }
            )
        metrics_response["all_models"] = metrics_list

    return metrics_response