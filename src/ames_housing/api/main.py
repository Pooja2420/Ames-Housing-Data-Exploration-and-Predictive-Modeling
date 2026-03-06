"""FastAPI application — entry point for the Ames Housing prediction API.

Start the server
----------------
    make serve
    # or
    uvicorn ames_housing.api.main:app --reload --host 0.0.0.0 --port 8000

Endpoints
---------
    GET  /health          liveness + model-ready check
    GET  /metrics         stored model performance metrics
    POST /predict         single property price prediction
    POST /predict/batch   batch predictions (up to 100)
    GET  /docs            Swagger UI (auto-generated)
    GET  /redoc           ReDoc UI (auto-generated)
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from ames_housing.api.routes import router
from ames_housing.config import settings
from ames_housing.utils.logging import setup_logging


# ── Lifespan: startup & shutdown ───────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup; release resources on shutdown."""
    # ── Startup ────────────────────────────────────────────────────────────────
    setup_logging()
    settings.paths.ensure_dirs()

    logger.info("=" * 55)
    logger.info("Ames Housing API  v{}", settings.project.version)
    logger.info("=" * 55)

    # Attempt to load saved pipeline
    try:
        from ames_housing.models.registry import load_pipeline
        app.state.pipeline = load_pipeline()
        logger.success("Model loaded and ready.")
    except FileNotFoundError:
        logger.warning(
            "No saved model found. Run 'make train' then restart the API."
        )
        app.state.pipeline = None

    yield  # ← server is running

    # ── Shutdown ───────────────────────────────────────────────────────────────
    logger.info("Shutting down Ames Housing API.")
    app.state.pipeline = None


# ── App factory ────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title       = "Ames Housing Price Predictor",
        description = (
            "Production ML API that predicts residential home sale prices "
            "using the Ames Housing dataset.\n\n"
            "**Best model:** LightGBM + Optuna HPO | **R² = 0.940** | **MAE ≈ $13,530**"
        ),
        version     = settings.project.version,
        lifespan    = lifespan,
        docs_url    = "/docs",
        redoc_url   = "/redoc",
        openapi_url = "/openapi.json",
    )

    # ── CORS middleware ────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins     = ["*"],   # tighten in production
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    # ── Request timing middleware ──────────────────────────────────────────────
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        logger.debug(
            "{} {} → {} | {:.1f}ms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    # ── Global exception handler ───────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception on {} {}", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "detail":     str(exc),
                "error_type": type(exc).__name__,
            },
        )

    # ── Register routes ────────────────────────────────────────────────────────
    app.include_router(router)

    # ── Root redirect ──────────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "name":    "Ames Housing Price Predictor",
            "version": settings.project.version,
            "docs":    "/docs",
            "health":  "/health",
        }

    return app


# ── Singleton app instance (used by uvicorn) ───────────────────────────────────
app = create_app()


# ── Programmatic start (used by `ames-serve` CLI) ─────────────────────────────
def start() -> None:
    import uvicorn
    cfg = settings.api
    uvicorn.run(
        "ames_housing.api.main:app",
        host    = cfg.host,
        port    = cfg.port,
        workers = cfg.workers,
        reload  = False,
        log_level = cfg.log_level,
    )


if __name__ == "__main__":
    start()
