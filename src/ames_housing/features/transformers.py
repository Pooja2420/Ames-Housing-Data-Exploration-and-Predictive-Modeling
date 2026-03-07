"""Custom sklearn-compatible transformers for the Ames Housing dataset.

Every transformer follows the sklearn ``BaseEstimator`` + ``TransformerMixin``
protocol so they can be dropped into any ``Pipeline`` or ``ColumnTransformer``
and work correctly with ``cross_val_score``, ``GridSearchCV``, and Optuna.

Transformers
------------
AmesFeatureEngineer   – creates all domain-driven interaction features
HighMissingDropper    – drops columns whose missing-value rate exceeds a threshold
SkewnessCorrector     – log1p-transforms columns whose |skew| exceeds a threshold
RareLabelEncoder      – collapses infrequent categories into an 'Other' bucket
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin


# ── 1. Domain feature engineering ─────────────────────────────────────────────

class AmesFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add domain-driven composite and interaction features.

    All new columns are computed from raw Ames column names (pre-encoding).
    Must be applied BEFORE one-hot encoding.

    New features
    ------------
    TotalSF         total finished square footage across all floors
    TotalBath       weighted bathroom count (half-bath counts as 0.5)
    HouseAge        age of the house at time of sale
    RemodelAge      years since last remodel at time of sale
    IsRemodeled     binary flag — was the house ever remodelled?
    GarageScore     garage cars × garage area (size × capacity)
    QualCond        overall quality × overall condition interaction
    PorchSF         total porch square footage across all porch types
    HasPool         binary — does the property have a pool?
    HasFireplace    binary — does the property have a fireplace?
    HasGarage       binary — does the property have a garage?
    HasBasement     binary — does the property have a basement?
    PricePerSF      lot area price efficiency proxy (LotArea / GrLivArea)
    """

    # Map of new feature → (required source columns)
    _FEATURE_DEPS: dict[str, list[str]] = {
        "TotalSF":      ["Total Bsmt SF", "1st Flr SF", "2nd Flr SF"],
        "TotalBath":    ["Full Bath", "Half Bath", "Bsmt Full Bath", "Bsmt Half Bath"],
        "HouseAge":     ["Yr Sold", "Year Built"],
        "RemodelAge":   ["Yr Sold", "Year Remod/Add"],
        "IsRemodeled":  ["Year Built", "Year Remod/Add"],
        "GarageScore":  ["Garage Cars", "Garage Area"],
        "QualCond":     ["Overall Qual", "Overall Cond"],
        "PorchSF":      ["Open Porch SF", "Enclosed Porch", "3Ssn Porch", "Screen Porch"],
        "HasPool":      ["Pool Area"],
        "HasFireplace": ["Fireplaces"],
        "HasGarage":    ["Garage Area"],
        "HasBasement":  ["Total Bsmt SF"],
        "PricePerSF":   ["Lot Area", "Gr Liv Area"],
    }

    def fit(self, X: pd.DataFrame, y=None) -> "AmesFeatureEngineer":
        """No fitting required — all features are deterministic transforms."""
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        df = X.copy()

        def _col(name: str) -> pd.Series:
            """Return column if present, else zeros (graceful degradation)."""
            return df[name] if name in df.columns else pd.Series(0, index=df.index)

        # ── Size features ──────────────────────────────────────────────────────
        df["TotalSF"] = (
            _col("Total Bsmt SF") + _col("1st Flr SF") + _col("2nd Flr SF")
        )

        df["TotalBath"] = (
            _col("Full Bath")
            + 0.5 * _col("Half Bath")
            + _col("Bsmt Full Bath")
            + 0.5 * _col("Bsmt Half Bath")
        )

        df["PorchSF"] = (
            _col("Open Porch SF")
            + _col("Enclosed Porch")
            + _col("3Ssn Porch")
            + _col("Screen Porch")
        )

        # ── Age features ───────────────────────────────────────────────────────
        df["HouseAge"]   = (_col("Yr Sold") - _col("Year Built")).clip(lower=0)
        df["RemodelAge"] = (_col("Yr Sold") - _col("Year Remod/Add")).clip(lower=0)
        df["IsRemodeled"] = (
            (_col("Year Built") != _col("Year Remod/Add")).astype(np.float32)
        )

        # ── Quality / condition interaction ────────────────────────────────────
        df["QualCond"]   = _col("Overall Qual") * _col("Overall Cond")
        df["GarageScore"] = _col("Garage Cars") * _col("Garage Area")

        # ── Binary indicators ──────────────────────────────────────────────────
        df["HasPool"]      = (_col("Pool Area")      > 0).astype(np.float32)
        df["HasFireplace"] = (_col("Fireplaces")     > 0).astype(np.float32)
        df["HasGarage"]    = (_col("Garage Area")    > 0).astype(np.float32)
        df["HasBasement"]  = (_col("Total Bsmt SF")  > 0).astype(np.float32)

        # ── Efficiency ratio ───────────────────────────────────────────────────
        liv = _col("Gr Liv Area").replace(0, np.nan)
        df["LotToLivRatio"] = (_col("Lot Area") / liv).fillna(0)

        engineered = [
            "TotalSF", "TotalBath", "PorchSF", "HouseAge", "RemodelAge",
            "IsRemodeled", "QualCond", "GarageScore", "HasPool", "HasFireplace",
            "HasGarage", "HasBasement", "LotToLivRatio",
        ]
        logger.debug("AmesFeatureEngineer: added {} feature(s)", len(engineered))
        return df


# ── 2. Drop high-missing columns ───────────────────────────────────────────────

class HighMissingDropper(BaseEstimator, TransformerMixin):
    """Drop columns whose missing-value fraction exceeds *threshold* on fit.

    Parameters
    ----------
    threshold : float
        Columns with ``null_fraction > threshold`` are dropped. Default 0.30.
    """

    def __init__(self, threshold: float = 0.30) -> None:
        self.threshold = threshold
        self.cols_to_drop_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "HighMissingDropper":
        missing_frac = X.isnull().mean()
        self.cols_to_drop_ = missing_frac[missing_frac > self.threshold].index.tolist()
        logger.info(
            "HighMissingDropper: will drop {} column(s) (>{:.0%} missing): {}",
            len(self.cols_to_drop_),
            self.threshold,
            self.cols_to_drop_,
        )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        cols = [c for c in self.cols_to_drop_ if c in X.columns]
        return X.drop(columns=cols)


# ── 3. Skewness corrector ──────────────────────────────────────────────────────

class SkewnessCorrector(BaseEstimator, TransformerMixin):
    """Apply ``log1p`` to numerical columns whose |skewness| > *threshold*.

    Fitted on training data; applies the same column list to validation/test.

    Parameters
    ----------
    threshold : float
        Skewness magnitude above which correction is applied. Default 0.75.
    exclude : list[str]
        Columns to skip (e.g. the target, binary flags). Default [].
    """

    def __init__(self, threshold: float = 0.75, exclude: list[str] | None = None) -> None:
        self.threshold = threshold
        # Store exactly what was passed — do NOT resolve None here.
        # sklearn's clone() uses an `is` identity check between get_params()
        # and the re-constructed object's get_params(); transforming the default
        # (None → []) causes a clone failure inside cross_val_score.
        self.exclude = exclude
        self.skewed_cols_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "SkewnessCorrector":
        _exclude = self.exclude or []   # resolve None safely here, not in __init__
        num_cols = (
            X.select_dtypes(include=[np.number])
            .columns.difference(_exclude)
        )
        skewness = X[num_cols].skew().abs()
        self.skewed_cols_ = skewness[skewness > self.threshold].index.tolist()
        logger.info(
            "SkewnessCorrector: {} column(s) with |skew|>{:.2f}: {}",
            len(self.skewed_cols_),
            self.threshold,
            self.skewed_cols_,
        )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        df = X.copy()
        cols = [c for c in self.skewed_cols_ if c in df.columns]
        if cols:
            df[cols] = np.log1p(df[cols].clip(lower=0))
        return df


# ── 4. Rare label encoder ──────────────────────────────────────────────────────

class RareLabelEncoder(BaseEstimator, TransformerMixin):
    """Collapse infrequent category labels into an ``'Other'`` bucket.

    Prevents one-hot encoding from generating hundreds of near-zero columns
    for categories that appear only once or twice.

    Parameters
    ----------
    threshold : float
        Categories with relative frequency < threshold are replaced.
        Default 0.01 (1 %).
    fill_value : str
        Replacement label for rare categories. Default ``'Other'``.
    """

    def __init__(self, threshold: float = 0.01, fill_value: str = "Other") -> None:
        self.threshold = threshold
        self.fill_value = fill_value
        self.frequent_labels_: dict[str, list] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "RareLabelEncoder":
        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            freq = X[col].value_counts(normalize=True)
            self.frequent_labels_[col] = freq[freq >= self.threshold].index.tolist()
        logger.debug(
            "RareLabelEncoder fitted on {} categorical column(s).", len(cat_cols)
        )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        df = X.copy()
        for col, labels in self.frequent_labels_.items():
            if col in df.columns:
                df[col] = df[col].where(df[col].isin(labels), other=self.fill_value)
        return df
