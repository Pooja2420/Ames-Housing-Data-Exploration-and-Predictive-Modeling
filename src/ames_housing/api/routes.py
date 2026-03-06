"""FastAPI route handlers.

Routes
------
GET  /health       liveness + model-ready check
GET  /metrics      stored model performance metrics
POST /predict      single property price prediction
POST /predict/batch  batch predictions (list of properties)
"""

from __future__ import annotations

import time
import uuid
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, status
from loguru import logger

from ames_housing.api.schemas import (
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    PredictionRequest,
    PredictionResponse,
)
from ames_housing.config import settings
from ames_housing.models.registry import load_model_meta

router = APIRouter()

# ── Confidence interval width (±9% approximation from CV std) ─────────────────
_CI_FACTOR = 0.09


# ── Dependency: validated pipeline from app state ──────────────────────────────

def get_pipeline(request: Request):
    """Retrieve the loaded pipeline from the app state."""
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run 'make train' first.",
        )
    return pipeline


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Ops"],
)
async def health(request: Request) -> HealthResponse:
    """Return API liveness status and model readiness."""
    return HealthResponse(
        status="ok",
        model_loaded=request.app.state.pipeline is not None,
        version=settings.project.version,
    )


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Model performance metrics",
    tags=["Ops"],
)
async def model_metrics() -> MetricsResponse:
    """Return the evaluation metrics from the last training run."""
    meta = load_model_meta()
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No model metadata found. Train the model first.",
        )
    return MetricsResponse(
        model_type    = meta.get("model_type", "unknown"),
        best_params   = meta.get("best_params", {}),
        metrics       = meta.get("metrics", {}),
        mlflow_run_id = meta.get("mlflow_run_id"),
        version       = meta.get("version", settings.project.version),
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict sale price for a single property",
    tags=["Prediction"],
    responses={
        503: {"model": ErrorResponse, "description": "Model not loaded"},
        422: {"description": "Validation error in request body"},
    },
)
async def predict(
    body: PredictionRequest,
    pipeline=Depends(get_pipeline),
) -> PredictionResponse:
    """Predict the sale price for a single residential property.

    Returns the predicted price plus a 90% prediction interval.
    """
    start = time.perf_counter()

    try:
        df = body.to_dataframe()
        raw_pred = pipeline.predict(df)[0]

        # If the target was log-transformed, inverse-transform
        if settings.features.log_transform_target:
            predicted_price = float(np.expm1(raw_pred))
        else:
            predicted_price = float(raw_pred)

        lower = predicted_price * (1 - _CI_FACTOR)
        upper = predicted_price * (1 + _CI_FACTOR)

        elapsed_ms = (time.perf_counter() - start) * 1000
        pred_id = str(uuid.uuid4())

        logger.info(
            "predict | id={} | price=${:,.0f} | [{:,.0f}–{:,.0f}] | {:.1f}ms",
            pred_id[:8], predicted_price, lower, upper, elapsed_ms,
        )

        return PredictionResponse(
            predicted_price = round(predicted_price, 2),
            lower_bound     = round(lower, 2),
            upper_bound     = round(upper, 2),
            model_version   = settings.project.version,
            prediction_id   = pred_id,
        )

    except Exception as exc:
        logger.exception("Prediction failed: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {exc}",
        ) from exc


@router.post(
    "/predict/batch",
    response_model=list[PredictionResponse],
    summary="Predict sale prices for multiple properties",
    tags=["Prediction"],
)
async def predict_batch(
    bodies: list[PredictionRequest],
    pipeline=Depends(get_pipeline),
) -> list[PredictionResponse]:
    """Predict sale prices for a list of properties in one request.

    Maximum 100 properties per request.
    """
    if len(bodies) > 100:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Batch size cannot exceed 100 properties.",
        )

    import pandas as pd
    start = time.perf_counter()

    try:
        df = pd.concat([b.to_dataframe() for b in bodies], ignore_index=True)
        raw_preds = pipeline.predict(df)

        if settings.features.log_transform_target:
            prices = np.expm1(raw_preds)
        else:
            prices = raw_preds

        responses = [
            PredictionResponse(
                predicted_price = round(float(p), 2),
                lower_bound     = round(float(p) * (1 - _CI_FACTOR), 2),
                upper_bound     = round(float(p) * (1 + _CI_FACTOR), 2),
                model_version   = settings.project.version,
                prediction_id   = str(uuid.uuid4()),
            )
            for p in prices
        ]

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "batch_predict | n={} | {:.1f}ms ({:.2f}ms/item)",
            len(bodies), elapsed_ms, elapsed_ms / len(bodies),
        )
        return responses

    except Exception as exc:
        logger.exception("Batch prediction failed: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {exc}",
        ) from exc
