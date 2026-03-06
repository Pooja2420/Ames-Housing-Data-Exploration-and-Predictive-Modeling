"""Shared pytest fixtures for the Ames Housing test suite.

All tests use synthetic data generated in memory — no real CSV required.
Fixtures are scoped carefully so expensive objects (pipelines, models) are
built once per session, while mutable state is fresh per test.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from ames_housing.features.engineering import build_pipeline


# ── Synthetic dataset ──────────────────────────────────────────────────────────

N_ROWS = 200
RNG    = np.random.default_rng(42)

# Column sets mirroring the real Ames dataset structure
_NUM_COLS = {
    "Overall Qual":    lambda n: RNG.integers(1, 11, n),
    "Overall Cond":    lambda n: RNG.integers(1, 11, n),
    "Gr Liv Area":     lambda n: RNG.integers(800, 4000, n).astype(float),
    "Total Bsmt SF":   lambda n: RNG.integers(0, 2000, n).astype(float),
    "1st Flr SF":      lambda n: RNG.integers(400, 2000, n).astype(float),
    "2nd Flr SF":      lambda n: RNG.integers(0, 1200, n).astype(float),
    "Garage Cars":     lambda n: RNG.integers(0, 4, n).astype(float),
    "Garage Area":     lambda n: RNG.integers(0, 900, n).astype(float),
    "Full Bath":       lambda n: RNG.integers(1, 4, n),
    "Half Bath":       lambda n: RNG.integers(0, 2, n),
    "Bsmt Full Bath":  lambda n: RNG.integers(0, 2, n).astype(float),
    "Bsmt Half Bath":  lambda n: RNG.integers(0, 1, n).astype(float),
    "TotRms AbvGrd":   lambda n: RNG.integers(4, 12, n),
    "Fireplaces":      lambda n: RNG.integers(0, 3, n),
    "Lot Area":        lambda n: RNG.integers(2000, 20000, n).astype(float),
    "Lot Frontage":    lambda n: RNG.uniform(20, 150, n),
    "Year Built":      lambda n: RNG.integers(1950, 2010, n),
    "Year Remod/Add":  lambda n: RNG.integers(1970, 2010, n),
    "Yr Sold":         lambda n: RNG.integers(2006, 2011, n),
    "Mo Sold":         lambda n: RNG.integers(1, 13, n),
    "Pool Area":       lambda n: np.where(RNG.random(n) > 0.97, RNG.integers(100, 500, n), 0).astype(float),
    "Wood Deck SF":    lambda n: RNG.integers(0, 500, n).astype(float),
    "Open Porch SF":   lambda n: RNG.integers(0, 200, n).astype(float),
    "Enclosed Porch":  lambda n: RNG.integers(0, 200, n).astype(float),
    "3Ssn Porch":      lambda n: RNG.integers(0, 100, n).astype(float),
    "Screen Porch":    lambda n: RNG.integers(0, 200, n).astype(float),
    "Mas Vnr Area":    lambda n: RNG.integers(0, 400, n).astype(float),
    "Misc Val":        lambda n: RNG.integers(0, 5000, n).astype(float),
    "SalePrice":       lambda n: (
        80000
        + RNG.integers(1, 11, n) * 18000           # quality
        + RNG.integers(800, 4000, n) * 55           # living area
        + RNG.normal(0, 15000, n)                   # noise
    ).clip(50000, 700000).astype(float),
}

_CAT_COLS = {
    "MS Zoning":      ["RL", "RM", "FV", "RH", "C (all)"],
    "Neighborhood":   ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst",
                       "NridgHt", "Gilbert", "Sawyer", "NWAmes", "SawyerW"],
    "Bldg Type":      ["1Fam", "2FmCon", "Duplx", "TwnhsE", "TwnhsI"],
    "House Style":    ["1Story", "2Story", "1.5Fin", "SFoyer", "SLvl"],
    "Exter Qual":     ["Ex", "Gd", "TA", "Fa"],
    "Exter Cond":     ["Ex", "Gd", "TA", "Fa", "Po"],
    "Foundation":     ["PConc", "CBlock", "BrkTil", "Slab", "Stone"],
    "Bsmt Qual":      ["Ex", "Gd", "TA", "Fa"],
    "Bsmt Exposure":  ["Gd", "Av", "Mn", "No"],
    "Heating QC":     ["Ex", "Gd", "TA", "Fa"],
    "Central Air":    ["Y", "N"],
    "Kitchen Qual":   ["Ex", "Gd", "TA", "Fa"],
    "Garage Type":    ["Attchd", "Detchd", "BuiltIn", "CarPort", "NA"],
    "Garage Finish":  ["Fin", "RFn", "Unf", "NA"],
    "Paved Drive":    ["Y", "P", "N"],
    "Sale Type":      ["WD", "New", "COD", "ConLD", "CWD"],
    "Sale Condition": ["Normal", "Abnorml", "Partial", "Family"],
}


def make_synthetic_df(n: int = N_ROWS, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic Ames-like DataFrame for testing."""
    rng = np.random.default_rng(seed)
    data: dict = {}

    for col, fn in _NUM_COLS.items():
        data[col] = fn(n)

    for col, choices in _CAT_COLS.items():
        data[col] = rng.choice(choices, n)

    return pd.DataFrame(data)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """Session-scoped synthetic raw DataFrame (200 rows)."""
    return make_synthetic_df(N_ROWS)


@pytest.fixture(scope="session")
def small_df() -> pd.DataFrame:
    """Tiny 20-row DataFrame for fast unit tests."""
    return make_synthetic_df(20, seed=99)


@pytest.fixture(scope="session")
def X_y(raw_df: pd.DataFrame):
    """Split raw_df into features and log-transformed target."""
    X = raw_df.drop(columns=["SalePrice"])
    y = np.log1p(raw_df["SalePrice"].values)
    return X, y


@pytest.fixture(scope="session")
def fitted_feature_pipeline(X_y) -> Pipeline:
    """Session-scoped fitted feature engineering pipeline (no scaling)."""
    X, _ = X_y
    pipeline = build_pipeline(scale=False)
    pipeline.fit(X)
    return pipeline


@pytest.fixture(scope="session")
def fitted_full_pipeline(X_y) -> Pipeline:
    """Session-scoped full pipeline (features + GBM model) fitted on synthetic data."""
    X, y = X_y
    feature_pipe = build_pipeline(scale=False)

    model = GradientBoostingRegressor(
        n_estimators=50, max_depth=3, random_state=42
    )
    pipeline = Pipeline(steps=[
        ("features", feature_pipe),
        ("model",    model),
    ])
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture(scope="session")
def sample_feature_names(fitted_feature_pipeline) -> list[str]:
    """Output feature names from the fitted feature pipeline."""
    from ames_housing.features.engineering import get_feature_names
    return get_feature_names(fitted_feature_pipeline)


@pytest.fixture()
def tmp_models_dir(tmp_path: Path) -> Path:
    """Temp directory for model artefact tests."""
    d = tmp_path / "models"
    d.mkdir()
    return d


@pytest.fixture()
def api_client(fitted_full_pipeline) -> TestClient:
    """FastAPI TestClient with the fitted pipeline injected into app state."""
    from ames_housing.api.main import create_app

    app = create_app()
    app.state.pipeline = fitted_full_pipeline

    # Bypass lifespan so we don't try to load from disk
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client


@pytest.fixture()
def valid_prediction_payload() -> dict:
    """Minimal valid request body for the /predict endpoint."""
    return {
        "OverallQual": 7,
        "GrLivArea":   1710.0,
        "YearBuilt":   2003,
        "Neighborhood":"CollgCr",
        "GarageCars":  2.0,
        "GarageArea":  548.0,
        "TotalBsmtSF": 856.0,
        "FullBath":    2,
        "YrSold":      2008,
    }
