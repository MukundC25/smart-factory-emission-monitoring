"""FastAPI application entry point for the Smart Factory Emission Monitoring API.

Run with:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.services.openaq_service import OpenAQService, get_openaq_service, set_openaq_service
from backend.config import get_settings
from backend.dependencies import get_data_loader
from backend.utils.data_loader import DataLoader
from backend.routers.factories import router as factories_router
from backend.routers.pollution import router as pollution_router
from backend.routers.recommendations import router as recommendations_router
from backend.routers.tree_calculator import router as tree_calculator_router

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load datasets on startup and emit a shutdown log on app exit.

    Note: Database schema should be prepared via Alembic migrations or an
    external migration step before starting the application. This function
    intentionally does not perform any synchronous schema-creation calls.
    """
    override = getattr(app, "dependency_overrides", {}).get(get_data_loader)
    loader = override() if override is not None else get_data_loader()
    app.state.data_loader = loader
    info = loader.dataset_info()
    for name, count in info.items():
        if count == 0:
            logger.warning(
                "Dataset '%s' is empty — endpoints using it will return empty results.",
                name,
            )
        else:
            logger.info("Dataset '%s' loaded: %d rows.", name, count)
    
    # Initialize OpenAQ service
    api_key = os.getenv("OPENAQ_API_KEY")
    openaq_service = OpenAQService(api_key=api_key)
    set_openaq_service(openaq_service)
    app.state.openaq = openaq_service
    logger.info("OpenAQ service initialized (API key: %s)", "configured" if api_key else "none")
    
    try:
        yield
    finally:
        logger.info("Smart Factory Emission Monitoring API shutting down.")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Smart Factory Emission Monitoring API",
    description=(
        "Production REST API for querying factory locations, pollution readings, "
        "ML-predicted emission impact scores, and city-level air quality statistics.\n\n"
        "Tree Calculator: /factories/{id}/tree-recommendation\n\n"
        "Pollution endpoints return empty results gracefully when the dataset has not "
        "yet been populated — zero code changes required once real data lands."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
_cors_uses_wildcard = any(o == "*" for o in settings.CORS_ORIGINS)
_cors_allow_credentials = not _cors_uses_wildcard
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=_cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(factories_router)
app.include_router(pollution_router)
app.include_router(recommendations_router)
app.include_router(tree_calculator_router)

# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def log_requests(request: Request, call_next):  # type: ignore[return]
    """Log HTTP method, path, status code, and response time for every request."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s → %d (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Return structured JSON for all HTTPException errors.

    404s include a hint about the /docs endpoint.
    """
    if exc.status_code == 404:
        return JSONResponse(
            status_code=404,
            content={
                "error": "not_found",
                "message": str(exc.detail),
                "hint": "Use GET /docs to explore all available endpoints.",
            },
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "http_error", "message": str(exc.detail)},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return structured JSON for pydantic/query-param validation errors (422)."""
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "detail": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unhandled exceptions.

    Returns a UUID error_id for log tracing — stack trace is never exposed to clients.
    """
    if isinstance(exc, HTTPException):
        # Let the HTTPException handler deal with this
        return await http_exception_handler(request, exc)
    error_id = str(uuid.uuid4())
    logger.exception("Unhandled exception [error_id=%s]", error_id, exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "error_id": error_id,
            "message": "An unexpected error occurred. Quote the error_id when reporting.",
        },
    )


# ---------------------------------------------------------------------------
# Core endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["System"], summary="Health check")
def health_check(loader: DataLoader = Depends(get_data_loader)) -> dict:
    """Return server health status and dataset row counts.

    Always returns HTTP 200 as long as the server process is alive.
    Use this for uptime monitoring and readiness probes.
    """
    return {
        "status": "ok",
        "datasets_loaded": loader.dataset_info(),
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }


@app.get("/", tags=["System"], summary="API info and endpoint listing")
def root() -> dict:
    """Return API metadata and the list of all available endpoints."""
    return {
        "name": "Smart Factory Emission Monitoring API",
        "version": "2.0.0",
        "description": "Recommendations endpoints: /recommendations, /recommendations/stats, /recommendations/{factory_id}, /recommendations/generate. Tree calculator endpoints: /factories/{factory_id}/tree-recommendation, /factories/tree-recommendation/bulk, /tree-calculator/constants",
        "endpoints": [
            "GET /factories            — paginated factory list with filters",
            "GET /factory/{id}         — factory detail with risk score & recommendations",
            "GET /pollution            — paginated pollution readings with filters",
            "GET /pollution/stats      — per-city PM2.5 / PM10 / AQI aggregates",
            "GET /pollution/heatmap/data — frontend-ready pollution heatmap points",
            "GET /recommendations      — paginated recommendation summaries",
            "GET /recommendations/{factory_id} — full recommendation report",
            "GET /recommendations/stats — recommendation aggregate stats",
            "POST /recommendations/generate — regenerate recommendations synchronously",
            "GET /factories/{factory_id}/tree-recommendation — single-factory tree recommendation",
            "POST /factories/tree-recommendation/bulk — bulk tree recommendations (max 50 IDs)",
            "GET /tree-calculator/constants — constants and methodology reference",
            "GET /health               — health check (always 200)",
            "GET /docs                 — interactive Swagger UI",
            "GET /redoc                — ReDoc API reference",
        ],
    }
