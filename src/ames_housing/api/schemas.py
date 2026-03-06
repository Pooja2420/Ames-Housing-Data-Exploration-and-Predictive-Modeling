"""Pydantic v2 request and response schemas for the prediction API.

All field names mirror the raw Ames Housing column names so the API
stays consistent with the training data — no silent renaming.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Request ────────────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    """Input features for a single property prediction.

    Only the most influential features are required; the rest default to
    median/typical values so callers don't need every one of the 80 columns.
    """

    # ── Required — top predictors ──────────────────────────────────────────────
    OverallQual: int = Field(
        ..., ge=1, le=10,
        description="Overall material and finish quality (1=Very Poor … 10=Very Excellent)",
        examples=[7],
    )
    GrLivArea: float = Field(
        ..., gt=0,
        description="Above-grade (ground) living area in square feet",
        examples=[1710],
    )
    YearBuilt: int = Field(
        ..., ge=1800, le=2025,
        description="Original construction year",
        examples=[2003],
    )

    # ── Optional — sensible defaults ──────────────────────────────────────────
    OverallCond: int          = Field(5,    ge=1, le=10,  description="Overall condition rating")
    TotalBsmtSF: float        = Field(0.0,  ge=0,         description="Total basement area (sqft)")
    FirstFlrSF: float         = Field(None, ge=0,         description="First floor square feet", alias="1stFlrSF")
    SecondFlrSF: float        = Field(0.0,  ge=0,         description="Second floor square feet", alias="2ndFlrSF")
    GarageCars: float         = Field(1.0,  ge=0, le=5,   description="Garage capacity (cars)")
    GarageArea: float         = Field(0.0,  ge=0,         description="Garage size (sqft)")
    FullBath: int             = Field(1,    ge=0, le=6,   description="Full bathrooms above grade")
    HalfBath: int             = Field(0,    ge=0, le=3,   description="Half bathrooms above grade")
    BsmtFullBath: float       = Field(0.0,  ge=0,         description="Basement full bathrooms")
    BsmtHalfBath: float       = Field(0.0,  ge=0,         description="Basement half bathrooms")
    TotRmsAbvGrd: int         = Field(6,    ge=1, le=20,  description="Total rooms above grade (excl. bathrooms)")
    Fireplaces: int           = Field(0,    ge=0, le=4,   description="Number of fireplaces")
    LotArea: float            = Field(9600, gt=0,         description="Lot size (sqft)")
    LotFrontage: float        = Field(69.0, ge=0,         description="Linear feet of street connected to property")
    YearRemodAdd: int         = Field(None, ge=1800, le=2025, description="Remodel year (same as YearBuilt if none)")
    YrSold: int               = Field(2010, ge=2006, le=2025, description="Year sold")
    MoSold: int               = Field(6,    ge=1, le=12,  description="Month sold")
    PoolArea: float           = Field(0.0,  ge=0,         description="Pool area (sqft)")
    WoodDeckSF: float         = Field(0.0,  ge=0,         description="Wood deck area (sqft)")
    OpenPorchSF: float        = Field(0.0,  ge=0,         description="Open porch area (sqft)")
    EnclosedPorch: float      = Field(0.0,  ge=0,         description="Enclosed porch area (sqft)")
    ScreenPorch: float        = Field(0.0,  ge=0,         description="Screen porch area (sqft)")
    MasVnrArea: float         = Field(0.0,  ge=0,         description="Masonry veneer area (sqft)")

    # ── Categorical fields ─────────────────────────────────────────────────────
    MSZoning: str             = Field("RL",     description="Zoning classification")
    Neighborhood: str         = Field("NAmes",  description="Physical location within Ames city limits")
    BldgType: str             = Field("1Fam",   description="Type of dwelling")
    HouseStyle: str           = Field("1Story", description="Style of dwelling")
    ExterQual: str            = Field("TA",     description="Exterior material quality (Ex/Gd/TA/Fa/Po)")
    ExterCond: str            = Field("TA",     description="Exterior material condition")
    Foundation: str           = Field("PConc",  description="Type of foundation")
    BsmtQual: str             = Field("TA",     description="Basement height quality")
    BsmtExposure: str         = Field("No",     description="Walkout / garden-level basement walls")
    HeatingQC: str            = Field("Ex",     description="Heating quality and condition")
    CentralAir: str           = Field("Y",      description="Central air conditioning (Y/N)")
    KitchenQual: str          = Field("TA",     description="Kitchen quality")
    GarageType: str           = Field("Attchd", description="Garage location")
    GarageFinish: str         = Field("Unf",    description="Interior finish of the garage")
    PavedDrive: str           = Field("Y",      description="Paved driveway (Y/P/N)")
    SaleType: str             = Field("WD",     description="Type of sale")
    SaleCondition: str        = Field("Normal", description="Condition of sale")

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _set_defaults(self) -> "PredictionRequest":
        """Fill derived defaults after all fields are set."""
        if self.YearRemodAdd is None:
            self.YearRemodAdd = self.YearBuilt
        if self.FirstFlrSF is None:
            self.FirstFlrSF = self.GrLivArea
        return self

    @field_validator("ExterQual", "ExterCond", "BsmtQual", "HeatingQC", "KitchenQual")
    @classmethod
    def _validate_quality(cls, v: str) -> str:
        valid = {"Ex", "Gd", "TA", "Fa", "Po", "NA"}
        if v not in valid:
            raise ValueError(f"Must be one of {valid}, got {v!r}")
        return v

    def to_dataframe(self) -> "pd.DataFrame":
        """Convert to a single-row DataFrame matching training column names."""
        import pandas as pd

        row = self.model_dump(by_alias=True)
        # Rename fields that use Python-safe aliases back to original names
        renames = {"1stFlrSF": "1st Flr SF", "2ndFlrSF": "2nd Flr SF"}
        ames_row = {}
        col_map = {
            "OverallQual": "Overall Qual", "OverallCond": "Overall Cond",
            "GrLivArea": "Gr Liv Area",   "YearBuilt": "Year Built",
            "TotalBsmtSF": "Total Bsmt SF", "1stFlrSF": "1st Flr SF",
            "2ndFlrSF": "2nd Flr SF",     "GarageCars": "Garage Cars",
            "GarageArea": "Garage Area",  "FullBath": "Full Bath",
            "HalfBath": "Half Bath",      "BsmtFullBath": "Bsmt Full Bath",
            "BsmtHalfBath": "Bsmt Half Bath", "TotRmsAbvGrd": "TotRms AbvGrd",
            "Fireplaces": "Fireplaces",   "LotArea": "Lot Area",
            "LotFrontage": "Lot Frontage", "YearRemodAdd": "Year Remod/Add",
            "YrSold": "Yr Sold",          "MoSold": "Mo Sold",
            "PoolArea": "Pool Area",      "WoodDeckSF": "Wood Deck SF",
            "OpenPorchSF": "Open Porch SF", "EnclosedPorch": "Enclosed Porch",
            "ScreenPorch": "Screen Porch", "MasVnrArea": "Mas Vnr Area",
            "MSZoning": "MS Zoning",      "Neighborhood": "Neighborhood",
            "BldgType": "Bldg Type",      "HouseStyle": "House Style",
            "ExterQual": "Exter Qual",    "ExterCond": "Exter Cond",
            "Foundation": "Foundation",   "BsmtQual": "Bsmt Qual",
            "BsmtExposure": "Bsmt Exposure", "HeatingQC": "Heating QC",
            "CentralAir": "Central Air",  "KitchenQual": "Kitchen Qual",
            "GarageType": "Garage Type",  "GarageFinish": "Garage Finish",
            "PavedDrive": "Paved Drive",  "SaleType": "Sale Type",
            "SaleCondition": "Sale Condition",
        }
        for pydantic_key, ames_key in col_map.items():
            val = row.get(pydantic_key) or row.get(pydantic_key.replace("_", " "))
            if val is not None:
                ames_row[ames_key] = val

        return pd.DataFrame([ames_row])


# ── Response ───────────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Prediction result returned by the API."""

    predicted_price: float = Field(
        ..., description="Predicted sale price in USD", examples=[208500.0]
    )
    lower_bound: float = Field(
        ..., description="Lower bound of 90% prediction interval", examples=[190200.0]
    )
    upper_bound: float = Field(
        ..., description="Upper bound of 90% prediction interval", examples=[226800.0]
    )
    model_version: str = Field(..., description="Model version string", examples=["0.1.0"])
    prediction_id: str = Field(..., description="Unique identifier for this prediction")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """API health check response."""
    status: str        = Field(..., examples=["ok"])
    model_loaded: bool = Field(..., description="Whether the model is loaded and ready")
    version: str       = Field(..., examples=["0.1.0"])
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MetricsResponse(BaseModel):
    """Stored model performance metrics."""
    model_type: str
    best_params: dict[str, Any]
    metrics: dict[str, dict[str, float]]
    mlflow_run_id: str | None
    version: str


class ErrorResponse(BaseModel):
    """Standard error envelope."""
    detail: str
    error_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
